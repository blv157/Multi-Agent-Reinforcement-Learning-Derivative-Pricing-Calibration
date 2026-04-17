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
BASE_CFG = dict(
    n_paths        = 120_000,
    T_steps        = 51,        # needs t=51 for Bermudan exercise window t1=21..t2=51
    delta          = DELTA,
    n_basis        = 100,
    bp_method      = "knn",
    bp_k           = 1,
    noise_std      = 1.0,       # unit noise; policy std scales exploration (fixes_v3)
    lr             = 1e-4,
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
    surface:     VolSurface,
    path_dep:    bool,
    tag:         str,
    results_dir: str,
) -> tuple:
    """
    Train one Exp2 sub-experiment (path-dep or non-path-dep).
    Returns (trainer, log_df).
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
    print(f"{'='*60}")

    trainer = MARLVolTrainer(surface=surface, cfg=cfg, bermudan=bermudan)
    # Tag checkpoints with sub-experiment name so path-dep / non-path-dep
    # checkpoints don't overwrite each other
    trainer.cfg.experiment = f"exp2_{tag}"
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

    if args.mode in ("both", "path_dep"):
        trainer_pd, log_pd = run_subexp(surface, path_dep=True,
                                         tag="path_dep", results_dir=RESULTS_DIR)
        log_dfs["path_dep"]  = log_pd
        trainers["path_dep"] = trainer_pd

    if args.mode in ("both", "non_path_dep"):
        trainer_npd, log_npd = run_subexp(surface, path_dep=False,
                                           tag="non_path_dep", results_dir=RESULTS_DIR)
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
