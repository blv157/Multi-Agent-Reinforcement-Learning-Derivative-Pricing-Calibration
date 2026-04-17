"""
experiments/exp2_bermudan.py
=============================
Experiment 2: Bermudan Option Price Minimisation

Replicates Figures 2, 3, 4 of Vadori (2022).

Objective
----------
Train MARL agents to choose volatility paths sigma(t, S, ...) that
MINIMISE the price of a Bermudan call option while simultaneously
maintaining vanilla call calibration via Gyongy localisation.

Two sub-experiments are run:
  A. Path-dependent state:     s_it = (t, S_it, sigma_{t-1}, S_{t^t1}, sigma_{(t-1)^t1})
  B. Non-path-dependent state: s_it = (t, S_it, sigma_{t-1})

Comparing A vs B reproduces Figures 3 and 4 of the paper.

Bermudan specification
-----------------------
  Strike:         S0 (ATM)
  Exercise dates: daily from t1=21 to t2=T=51 days
  Type:           call

State dimensions
-----------------
  A (path-dependent):     5
  B (non-path-dependent): 3

Reward
-------
  r_T = -P_bermudan(sigma)   (minimise the Bermudan price)

Vanilla calibration is maintained via Gyongy localisation in the training
loop (not via the reward).  See marl_vol.py for implementation details.

How to run
-----------
    cd "E:/AMS 517/Term Project"
    python experiments/exp2_bermudan.py

    # To run only one sub-experiment:
    python experiments/exp2_bermudan.py --mode path_dep
    python experiments/exp2_bermudan.py --mode non_path_dep

Results are saved to results/exp2/:
    exp2_path_dep_ep{N}.pt        -- path-dependent checkpoints
    exp2_non_path_dep_ep{N}.pt    -- non-path-dependent checkpoints
    exp2_training_log.csv         -- Bermudan price per episode (both runs)
    exp2_bermudan_price_plot.png  -- Figure 2: Bermudan price over training
    exp2_smile_comparison.png     -- Figure 3: smile under optimal vol
    exp2_state_comparison.png     -- Figure 4: path-dep vs non-path-dep
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from market_data    import VolSurface
from diffusion      import generate_brownian, simulate_paths, DELTA
from options        import mc_call_prices, make_bermudan
from american_mc    import bermudan_price
from reward         import implied_vol_batch, calibration_loss
from marl_vol       import MARLVolTrainer, TrainConfig
from policy         import (build_state_exp2_path_dependent,
                             build_state_exp2_nonpath,
                             SIGMA_MIN, SIGMA_MAX)


# ---------------------------------------------------------------------------
# Warm-start helper
# ---------------------------------------------------------------------------

def warm_start_from_exp1(trainer: MARLVolTrainer, exp1_ckpt_path: str) -> None:
    """
    Initialise the exp2 policy from a trained exp1 checkpoint.

    Why this is needed
    ------------------
    Exp2 starts from a randomly-initialised policy, which outputs sigma ≈ exp(0) ≈ 1%
    (the network bias is near zero).  With the combined Bermudan + calibration reward,
    the calibration gradient strongly drives sigma upward toward market vol (~20%), but
    the policy overshoots to ~50% before the Bermudan penalty can compensate.  Once
    sigma > 30%, both rewards are deeply negative and clipped, the value function
    diverges, and training never recovers.

    By warm-starting from the trained exp1 policy (which already knows how to output
    calibration-appropriate sigma values), we start exp2 in the well-calibrated regime:
      - Bermudan reward ≈ -1  (b_price ≈ European ref, since sigma ≈ market vol)
      - Calib reward  ≈ +0.5..+0.9  (already calibrated)
      - Combined      ≈ -0.5  (small negative, much better than -2.75)

    PPO then only needs to fine-tune the sigma surface shape to further reduce the
    Bermudan price, which is a much easier optimisation problem.

    Implementation
    --------------
    - exp1 policy: state_dim=2  (net.0.weight: [50, 2])
    - exp2 policy: state_dim=3 or 5  (net.0.weight: [50, 3 or 5])
    - Copy first 2 input columns from exp1; zero-init the extra columns so that
      the additional state features (sigma_prev, S_at_t1, sig_at_t1) start with
      zero influence.  All other layers (hidden + output) are copied directly.
    - The state normalizer is extended: first 2 features get exp1 learned mean/var;
      extra features get neutral warm-prior (sigma_prev ≈ 0.10 of SIGMA_MAX, ATM).
    - The VALUE network is NOT warm-started: the exp2 reward scale is completely
      different from exp1, so exp1 value estimates would be misleading.  The value
      network learns from scratch but quickly adapts (it has a very high MSE loss
      initially which drives rapid correction).
    """
    ckpt = torch.load(exp1_ckpt_path, map_location="cpu", weights_only=False)

    # ── Policy network weights ────────────────────────────────────────────────
    sd1 = ckpt["policy"]                          # exp1 weights (state_dim=2)
    sd2 = trainer.policy.state_dict()            # exp2 weights (state_dim=3 or 5)

    for key in sd2:
        if key not in sd1:
            continue
        v1, v2 = sd1[key], sd2[key]
        if v1.shape == v2.shape:
            sd2[key] = v1.clone()                 # identical layers: copy directly
        elif key == "net.0.weight":
            # First layer: [50, 2] → [50, 3 or 5]
            # Copy first 2 columns; zero-init the extra exp2-only features
            new_w = torch.zeros_like(v2)
            new_w[:, : v1.shape[1]] = v1
            sd2[key] = new_w

    trainer.policy.load_state_dict(sd2)

    # ── Policy state normalizer ───────────────────────────────────────────────
    norm1 = ckpt["policy_norm"]
    n1    = norm1["mean"].shape[0]         # 2
    n2    = trainer.cfg.state_dim          # 3 or 5

    new_mean  = torch.zeros(n2)
    new_var   = torch.ones(n2)
    new_mean[: n1] = norm1["mean"]
    new_var[: n1]  = norm1["var"]

    # Extra features warm-prior (index meaning depends on exp2 state builder)
    if n2 > n1:                            # sigma_prev / SIGMA_MAX
        new_mean[n1]   = 0.10              # ~20% vol / 200% max
        new_var[n1]    = 0.002
    if n2 > n1 + 1:                        # S_at_t1 / S0
        new_mean[n1 + 1] = 1.00
        new_var[n1 + 1]  = 0.0022
    if n2 > n1 + 2:                        # sig_at_t1 / SIGMA_MAX
        new_mean[n1 + 2] = 0.10
        new_var[n1 + 2]  = 0.002

    norm2_sd           = trainer.policy.norm.state_dict()
    norm2_sd["mean"]   = new_mean
    norm2_sd["var"]    = new_var
    norm2_sd["count"]  = norm1["count"]   # inherit count → normaliser stays stable
    trainer.policy.norm.load_state_dict(norm2_sd)

    # ── Value state normalizer (same extension as policy norm) ────────────────
    norm2v_sd           = trainer.value.norm.state_dict()
    norm2v_sd["mean"]   = new_mean.clone()
    norm2v_sd["var"]    = new_var.clone()
    norm2v_sd["count"]  = norm1["count"]
    trainer.value.norm.load_state_dict(norm2v_sd)

    print(f"  Warm-started exp2 policy from: {exp1_ckpt_path}")
    print(f"  exp1 state_dim=2 -> exp2 state_dim={n2}; extra input cols zero-init")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset registry — mirrors exp1
DATASETS = {
    "aug2018": {
        "path":  "data/spx_smiles_aug2018.csv",
        "label": "Aug 2018 (CBOE EOD, spot=2813)",
    },
    "apr2026": {
        "path":  "data/spx_smiles_clean.csv",
        "label": "Apr 2026 (CBOE EOD, spot=6575)",
    },
}

# Base PPO hyperparameters — aligned with fixes_v3 (exp1 best run)
# Key exp2 differences from exp1:
#   n_paths=40_000: fewer paths than exp1's 120k.  With 120k + antithetic variates the
#               Bermudan MC estimates become so precise that all B=10 rollouts yield
#               nearly identical rewards → PPO gradient ≈ 0 → policy random-walks and
#               diverges.  40k paths retain enough MC noise to create reward variance
#               across rollouts (PPO gets a clear gradient signal) while still
#               producing accurate enough LS Bermudan estimates.
#   lr=5e-5     (half of exp1): prevents sigma from overshooting market vol during
#               the first 20–50 episodes when the combined reward gradient is noisy.
#   calib_weight=2.0: stronger Gyongy tether ensures calibration is maintained even
#               when the Bermudan gradient tries to push sigma below market vol.
BASE_CFG = dict(
    n_paths        = 40_000,
    T_steps        = 51,        # needs t=51 for Bermudan exercise window t1=21..t2=51
    delta          = DELTA,
    n_basis        = 100,
    bp_method      = "knn",
    bp_k           = 1,
    noise_std      = 1.0,       # unit noise; policy std scales exploration (fixes_v3)
    lr             = 5e-5,      # half of exp1: avoids overshoot in early Bermudan training
    lr_min         = 1e-6,      # cosine annealing floor
    clip           = 0.3,
    kl_target      = 0.01,
    K_epochs       = 30,
    B_envs         = 10,
    mb_frac        = 0.1,
    c_value        = 0.5,
    c_entropy      = 0.01,
    gamma          = 1.0,
    use_antithetic = True,      # antithetic variates for MC variance reduction
    experiment     = "exp2",
    calib_weight   = 2.0,       # stronger Gyongy tether (2x calibration penalty)
    n_episodes     = 2000,
    n_strikes      = 10,
    log_every      = 10,
    save_every     = 100,
    conv_window    = 100,
    conv_tol       = 1e-3,
    conv_patience  = 5,
    device         = DEVICE,
)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_bermudan_price_curve(
    log_path: dict,
    save_path: str,
    surface: VolSurface,
):
    """
    Figure 2: Bermudan price over training for path-dep and non-path-dep.

    The Bermudan price is the negative of the terminal reward.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    colours = {"path_dep": "steelblue", "non_path_dep": "tomato"}
    labels  = {"path_dep": "Path-dependent state (dim=5)",
               "non_path_dep": "Non-path-dependent state (dim=3)"}

    for key, log_df in log_path.items():
        # Bermudan price = -(terminal reward)
        prices = -log_df["reward"].values
        ep     = log_df["episode"].values
        ax.plot(ep, prices, color=colours[key], label=labels[key],
                linewidth=1, alpha=0.8)

    # Reference: European call price (Black-Scholes ATM)
    from market_data import bs_call_vectorised
    S0 = surface.spot
    T_yr = 51 / 252.0
    sigma_atm = surface.get_iv(S0, T_yr)
    euro_price = float(bs_call_vectorised(S0, S0, T_yr, sigma_atm))
    ax.axhline(euro_price, color="green", linestyle="--", linewidth=1.5,
               label=f"European call (BS ATM, price={euro_price:.1f})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Bermudan call price")
    ax.set_title("Exp 2: Bermudan Price over Training")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Bermudan price plot saved: {save_path}")


def plot_vol_surface_comparison(
    trainer_pd:  MARLVolTrainer,
    trainer_npd: MARLVolTrainer,
    surface:     VolSurface,
    save_path:   str,
    n_eval:      int = 100_000,
):
    """
    Figure 3/4: Implied vol smile under path-dependent vs non-path-dependent
    trained policy, compared to the market smile.
    """
    fig, axes = plt.subplots(
        2, len(surface.maturities_days),
        figsize=(4 * len(surface.maturities_days), 7),
        sharey="row",
    )

    rows = [
        (trainer_pd,  "Path-dependent (dim=5)"),
        (trainer_npd, "Non-path-dependent (dim=3)"),
    ]

    for row_idx, (trainer, row_label) in enumerate(rows):
        cfg    = trainer.cfg
        device = cfg.device
        S0     = surface.spot
        policy = trainer.policy

        # Evaluation rollout (no noise, use policy mean)
        with torch.no_grad():
            Z = generate_brownian(n_eval, cfg.T_steps, seed=42, device=device)
            sigmas = torch.zeros(n_eval, cfg.T_steps, device=device)
            sigma_prev = torch.full((n_eval,), 0.20, device=device)
            t1 = 21
            S_at_t1   = torch.full((n_eval,), S0, device=device)
            sig_at_t1 = torch.full((n_eval,), 0.20, device=device)

            S_paths = torch.empty(n_eval, cfg.T_steps + 1, device=device)
            S_paths[:, 0] = S0
            S_cur = torch.full((n_eval,), S0, device=device)

            for t in range(cfg.T_steps):
                if cfg.path_dep:
                    state = build_state_exp2_path_dependent(
                        t, S_cur, sigma_prev, S_at_t1, sig_at_t1,
                        cfg.T_steps, S0)
                else:
                    state = build_state_exp2_nonpath(
                        t, S_cur, sigma_prev, cfg.T_steps, S0)

                mu_logsig, _ = policy(state)
                # Policy outputs log(sigma) mean — convert to sigma
                sigma_t = torch.exp(mu_logsig).clamp(SIGMA_MIN, SIGMA_MAX)
                sigmas[:, t] = sigma_t
                S_cur = S_cur * torch.exp(
                    -0.5 * sigma_t ** 2 * DELTA + sigma_t * DELTA ** 0.5 * Z[:, t]
                )
                S_paths[:, t + 1] = S_cur
                sigma_prev = sigma_t.clone()
                if t + 1 == t1:
                    S_at_t1   = S_cur.clone()
                    sig_at_t1 = sigma_t.clone()

        # Plot smile for each maturity — restrict to training range [0.88, 1.09]
        moneyness_grid = np.linspace(0.88, 1.09, 50)
        K_grid = torch.tensor(moneyness_grid * S0, dtype=torch.float32, device=device)

        for col_idx, (dte_days, dte_years) in enumerate(
            zip(surface.maturities_days.astype(int), surface.maturities_years)
        ):
            ax   = axes[row_idx, col_idx]
            t_idx = int(dte_days)
            T_yr  = float(dte_years)

            if t_idx >= S_paths.shape[1]:
                continue

            S_t = S_paths[:, t_idx]
            T_arr = torch.full_like(K_grid, T_yr)

            from options import call_payoff_grid
            payoffs     = call_payoff_grid(S_t, K_grid)
            mc_px_dense = payoffs.mean(dim=0)
            iv_mc = implied_vol_batch(mc_px_dense, S0, K_grid, T_arr)

            iv_mkt = torch.tensor(
                [surface.get_iv(float(k), T_yr) for k in K_grid.cpu().tolist()],
                dtype=torch.float32,
            )

            valid_mc  = ~torch.isnan(iv_mc)
            valid_mkt = ~torch.isnan(iv_mkt)

            ax.plot(moneyness_grid[valid_mc.cpu()],
                    iv_mc[valid_mc].cpu().numpy() * 100,
                    color="steelblue", linewidth=1.8, label="Model")
            ax.plot(moneyness_grid[valid_mkt.cpu()],
                    iv_mkt[valid_mkt].numpy() * 100,
                    color="tomato", linestyle="--", linewidth=1.8, label="Market")

            if row_idx == 0:
                ax.set_title(f"{dte_days}d", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nImplied Vol (%)", fontsize=8)
            ax.set_xlabel("K/S", fontsize=8)
            ax.grid(True, alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7)

    fig.suptitle("Exp 2: Vol Smiles under Bermudan-Minimising Policy", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Smile comparison plot saved: {save_path}")


# ---------------------------------------------------------------------------
# Run one sub-experiment
# ---------------------------------------------------------------------------

def run_subexp(
    surface:        VolSurface,
    path_dep:       bool,
    tag:            str,
    results_dir:    str,
    exp1_ckpt_path: str = "",
) -> tuple:
    """
    Train one Exp2 sub-experiment (path-dep or non-path-dep).

    Parameters
    ----------
    surface        : VolSurface for the dataset
    path_dep       : True for path-dependent state (dim=5), False for dim=3
    tag            : checkpoint filename prefix  ('path_dep' or 'non_path_dep')
    results_dir    : directory to save checkpoints and plots
    exp1_ckpt_path : optional path to exp1 best checkpoint for warm-starting the
                     policy.  If given and the file exists, the exp2 policy is
                     initialised from the exp1 weights (first 2 input features
                     copied; extra features zero-initialised).  This prevents
                     sigma from diverging away from the calibrated regime during
                     the early training episodes.

    Returns
    -------
    (trainer, log_df)
    """
    state_dim = 5 if path_dep else 3
    cfg = TrainConfig(
        **BASE_CFG,
        state_dim  = state_dim,
        path_dep   = path_dep,
        S0         = surface.spot,
        save_dir   = results_dir,
    )

    bermudan = make_bermudan(
        strike    = surface.spot,
        t1_step   = 21,
        t2_step   = cfg.T_steps,
    )

    print(f"\n{'='*60}")
    print(f"Sub-experiment: {tag}  (state_dim={state_dim})")
    print(f"  lr={cfg.lr}  calib_weight={cfg.calib_weight}")
    print(f"{'='*60}")

    trainer = MARLVolTrainer(surface=surface, cfg=cfg, bermudan=bermudan)
    # Tag checkpoints with sub-experiment name so path-dep / non-path-dep
    # checkpoints don't overwrite each other
    trainer.cfg.experiment = f"exp2_{tag}"

    # Optional warm-start from exp1 calibration checkpoint
    if exp1_ckpt_path and os.path.exists(exp1_ckpt_path):
        warm_start_from_exp1(trainer, exp1_ckpt_path)
    else:
        if exp1_ckpt_path:
            print(f"  WARNING: exp1 ckpt not found at {exp1_ckpt_path}; "
                  f"training from random init")

    log = trainer.train()

    log_df = pd.DataFrame(log)
    return trainer, log_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2: Bermudan Option Price Minimisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["both", "path_dep", "non_path_dep"],
                        default="both",
                        help="Which sub-experiment to run (default: both)")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()),
                        default="aug2018",
                        help="Market data to train on (default: aug2018)")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional suffix for results directory")
    parser.add_argument(
        "--exp1_ckpt", type=str, default="",
        help=(
            "Path to exp1 best checkpoint for warm-starting the exp2 policy. "
            "Example: results/exp1_aug2018_fixes_v3/exp1_best.pt  "
            "Strongly recommended: prevents sigma from diverging during early training."
        ),
    )
    args = parser.parse_args()

    ds_cfg     = DATASETS[args.dataset]
    DATA_PATH  = ds_cfg["path"]
    DATA_LABEL = ds_cfg["label"]
    tag_suffix = f"_{args.tag}" if args.tag else ""
    RESULTS_DIR = f"results/exp2_{args.dataset}{tag_suffix}"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 60)
    print("Experiment 2: Bermudan Option Price Minimisation")
    print("=" * 60)
    print(f"Dataset: {args.dataset}  ({DATA_LABEL})")
    print(f"Device : {DEVICE}")
    print(f"Mode   : {args.mode}")
    print(f"Results: {RESULTS_DIR}\n")

    surface = VolSurface(DATA_PATH)
    surface.summary()

    log_dfs   = {}
    trainers  = {}

    exp1_ckpt = args.exp1_ckpt
    # Auto-detect exp1 checkpoint when --exp1_ckpt is not specified
    if not exp1_ckpt:
        auto = f"results/exp1_{args.dataset}_fixes_v3/exp1_best.pt"
        if os.path.exists(auto):
            exp1_ckpt = auto
            print(f"Auto-detected exp1 checkpoint: {auto}")

    if args.mode in ("both", "path_dep"):
        trainer_pd, log_pd = run_subexp(
            surface, path_dep=True, tag="path_dep",
            results_dir=RESULTS_DIR, exp1_ckpt_path=exp1_ckpt,
        )
        log_dfs["path_dep"]  = log_pd
        trainers["path_dep"] = trainer_pd

    if args.mode in ("both", "non_path_dep"):
        trainer_npd, log_npd = run_subexp(
            surface, path_dep=False, tag="non_path_dep",
            results_dir=RESULTS_DIR, exp1_ckpt_path=exp1_ckpt,
        )
        log_dfs["non_path_dep"]  = log_npd
        trainers["non_path_dep"] = trainer_npd

    # ── Save combined log ─────────────────────────────────────────────────────
    if log_dfs:
        combined = pd.concat(
            [df.assign(mode=key) for key, df in log_dfs.items()],
            ignore_index=True,
        )
        log_path = os.path.join(RESULTS_DIR, "exp2_training_log.csv")
        combined.to_csv(log_path, index=False, float_format="%.6f")
        print(f"\nTraining log saved: {log_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if len(log_dfs) >= 1:
        plot_bermudan_price_curve(
            log_path  = log_dfs,
            save_path = os.path.join(RESULTS_DIR, "exp2_bermudan_price_plot.png"),
            surface   = surface,
        )

    if "path_dep" in trainers and "non_path_dep" in trainers:
        plot_vol_surface_comparison(
            trainer_pd  = trainers["path_dep"],
            trainer_npd = trainers["non_path_dep"],
            surface     = surface,
            save_path   = os.path.join(RESULTS_DIR, "exp2_smile_comparison.png"),
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Experiment 2 complete.")
    for key, log_df in log_dfs.items():
        final_price = -log_df["reward"].iloc[-1]
        print(f"  [{key}] Final Bermudan price: {final_price:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
