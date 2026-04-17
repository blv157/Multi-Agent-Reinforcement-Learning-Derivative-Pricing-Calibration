"""
experiments/exp1_local_vol.py
==============================
Experiment 1: Local Volatility Calibration

Replicates Figure 1 of Vadori (2022).

Objective
----------
Train MARL agents to find a local volatility function sigma(t, S) such
that the MC prices of vanilla calls match the SPX market smile across
5 maturities and 10 strikes per maturity.

State:     s_it = (t/T, S_it/S0)  -- 2-dimensional
Reward:    r_T  = L_ref - L(sigma) (improvement over flat BS baseline)
Algorithm: PPO with basis player exploration (Algorithm 1)

How to run
----------
    cd "E:/AMS 517/Term Project"

    # Train on August 2018 data (paper replication):
    python experiments/exp1_local_vol.py --dataset aug2018

    # Train on April 2026 data (current market):
    python experiments/exp1_local_vol.py --dataset apr2026

    # Fast diagnostic run (10k paths, 500 episodes):
    python experiments/exp1_local_vol.py --dataset aug2018 --small

Results are saved to results/exp1_{dataset}/:
    exp1_best.pt            -- best checkpoint (by reward) during training
    exp1_ep{N}.pt           -- periodic checkpoints every 100 episodes
    exp1_training_log.csv   -- loss and reward per episode
    exp1_smile_plot.png     -- fitted vs market implied vol smiles (Figure 1)
    exp1_learning_curve.png -- calibration loss over training

The best checkpoint is used by Experiment 2 (Bermudan pricing) as the
calibrated local vol model for the exotic option application.
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from market_data    import VolSurface
from diffusion      import generate_brownian, simulate_paths, DELTA
from options        import mc_call_prices
from reward         import implied_vol_batch, calibration_loss
from marl_vol       import MARLVolTrainer, TrainConfig
from policy         import build_state_exp1, SIGMA_MIN, SIGMA_MAX


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset registry — add new dates here
DATASETS = {
    "aug2018": {
        "path":       "data/spx_smiles_aug2018.csv",
        "label":      "Aug 2018 (CBOE EOD, spot=2813)",
        "oos_key":    "apr2026",          # which dataset to use for OOS plot
    },
    "apr2026": {
        "path":       "data/spx_smiles_clean.csv",
        "label":      "Apr 2026 (CBOE EOD, spot=6575)",
        "oos_key":    "aug2018",          # OOS = go back to 2018 regime
    },
}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_smile_comparison(
    trainer:    MARLVolTrainer,
    surface:    VolSurface,
    save_path:  str,
    n_eval:     int = 200_000,
    data_label: str = "",
):
    """
    Generate Figure 1: fitted vs market implied vol smiles.

    Replicates the paper's full smile grid: 15 maturities from T=23 to
    T=51 in steps of 2 trading days, arranged in a 3-row x 5-column figure.

    Of these 15 panels, the calibration maturities are marked with (*) in
    their titles.  All other panels show interpolated smiles that emerge from
    forward propagation of the trained local vol surface at maturities the
    model was never explicitly calibrated to.
    """
    cfg    = trainer.cfg
    device = cfg.device
    S0     = surface.spot
    policy = trainer.policy

    # Maturity grid: T=23 to T=51, step 2 -> 15 maturities
    all_dte   = list(range(23, 52, 2))
    calib_set = set(surface.maturities_days.astype(int).tolist())
    N_COLS, N_ROWS = 5, 3

    print(f"\nGenerating smile comparison plot "
          f"({len(all_dte)} maturities, {N_ROWS}x{N_COLS} grid)...")

    # Evaluation rollout (no noise) — deterministic mean policy.
    # policy(state) returns (mu_logsig, log_std): the MLP outputs the mean of
    # log(sigma).  Convert to sigma via exp() before advancing paths.
    with torch.no_grad():
        Z      = generate_brownian(n_eval, cfg.T_steps, seed=999, device=device)
        sigmas = torch.zeros(n_eval, cfg.T_steps, device=device)

        S_cur = torch.full((n_eval,), S0, device=device)
        for t in range(cfg.T_steps):
            state          = build_state_exp1(t, S_cur, cfg.T_steps, S0)
            mu_logsig, _   = policy(state)
            sigma_t        = torch.exp(mu_logsig).clamp(SIGMA_MIN, SIGMA_MAX)
            sigmas[:, t]   = sigma_t
            S_cur = S_cur * torch.exp(
                -0.5 * sigma_t ** 2 * DELTA + sigma_t * DELTA ** 0.5 * Z[:, t]
            )

        # Full price matrix
        S_full = torch.empty(n_eval, cfg.T_steps + 1, device=device)
        S_full[:, 0] = S0
        S_temp = torch.full((n_eval,), S0, device=device)
        for t in range(cfg.T_steps):
            S_temp = S_temp * torch.exp(
                -0.5 * sigmas[:, t] ** 2 * DELTA
                + sigmas[:, t] * DELTA ** 0.5 * Z[:, t]
            )
            S_full[:, t + 1] = S_temp

    # Dense moneyness grid for smooth curves
    moneyness_grid = np.linspace(0.88, 1.12, 50)
    K_grid = torch.tensor(moneyness_grid * S0, dtype=torch.float32, device=device)

    from options import call_payoff_grid

    def _smile_curves(dte_days):
        t_idx = int(dte_days)
        T_yr  = dte_days / 252.0
        if t_idx >= S_full.shape[1]:
            return None, None, None, None
        S_t   = S_full[:, t_idx]
        T_arr = torch.full_like(K_grid, T_yr)
        payoffs = call_payoff_grid(S_t, K_grid)
        mc_px   = payoffs.mean(dim=0)
        iv_mc   = implied_vol_batch(mc_px, S0, K_grid, T_arr)
        iv_mkt  = torch.tensor(
            [surface.get_iv(float(k), T_yr) for k in K_grid.cpu().tolist()],
            dtype=torch.float32,
        )
        valid_mc  = ~torch.isnan(iv_mc)
        valid_mkt = ~torch.isnan(iv_mkt)
        return (
            iv_mc[valid_mc].cpu().numpy()  * 100,
            iv_mkt[valid_mkt].numpy()      * 100,
            moneyness_grid[valid_mc.cpu()],
            moneyness_grid[valid_mkt.cpu()],
        )

    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(4 * N_COLS, 3.5 * N_ROWS),
        sharey=True,
        constrained_layout=True,
    )

    for idx, dte_days in enumerate(all_dte):
        row, col = divmod(idx, N_COLS)
        ax = axes[row, col]
        iv_model, iv_mkt, m_model, m_mkt = _smile_curves(dte_days)
        if iv_model is None:
            ax.set_visible(False)
            continue
        is_calib    = dte_days in calib_set
        model_color = "steelblue" if is_calib else "seagreen"
        tag         = " (*)" if is_calib else ""
        ax.plot(m_model, iv_model,
                color=model_color, linewidth=2, label="Model (MARL)")
        ax.plot(m_mkt, iv_mkt,
                color="tomato", linestyle="--", linewidth=2, label="Market")
        ax.set_title(f"{dte_days}d{tag}", fontsize=10,
                     fontweight="bold" if is_calib else "normal")
        ax.set_xlabel("Moneyness (K/S)", fontsize=8)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("Implied Vol (%)", fontsize=9)
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        f"Exp 1: MARL Local Vol Calibration — SPX (spot={S0:.0f}, "
        f"data: {data_label})\n"
        f"T = 23 to 51 DTE, step 2  |  "
        f"(*) = calibration maturity  |  others = interpolated",
        fontsize=10,
    )
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Smile plot saved: {save_path}")


def plot_learning_curve(log: list, save_path: str, l_ref: float):
    """Plot calibration loss and reward over training episodes."""
    episodes = [r["episode"]  for r in log]
    losses   = [r["loss"]     for r in log]
    rewards  = [r["reward"]   for r in log]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax1.plot(episodes, losses, color="steelblue", linewidth=1, alpha=0.7)
    ax1.axhline(l_ref, color="tomato", linestyle="--",
                linewidth=1.5, label=f"BS baseline ({l_ref:.6f})")
    ax1.set_ylabel("Calibration loss (IV space)")
    ax1.set_title("Experiment 1: Learning Curve")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    ax2.plot(episodes, rewards, color="seagreen", linewidth=1, alpha=0.7)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_ylabel("Episode reward")
    ax2.set_xlabel("Episode")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Learning curve saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1: MARL Local Volatility Calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", choices=list(DATASETS.keys()), default="aug2018",
        help="Market data to train on  (default: aug2018)",
    )
    parser.add_argument(
        "--small", action="store_true",
        help="Fast diagnostic run (10k paths, 500 episodes)",
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Optional suffix appended to results directory (e.g. 'fixes_v1')",
    )
    args = parser.parse_args()

    ds_cfg      = DATASETS[args.dataset]
    DATA_PATH   = ds_cfg["path"]
    DATA_LABEL  = ds_cfg["label"]
    OOS_KEY     = ds_cfg["oos_key"]
    tag_suffix  = f"_{args.tag}" if args.tag else ""
    RESULTS_DIR = f"results/exp1_{args.dataset}{tag_suffix}"

    if args.small:
        RESULTS_DIR = RESULTS_DIR + "_small"
        CFG = TrainConfig(
            n_paths        = 10_000,
            T_steps        = 47,   # matches longest calibration maturity (47 DTE)
            delta          = DELTA,
            n_basis        = 50,
            bp_method      = "knn",
            bp_k           = 1,
            noise_std      = 1.0,  # unit noise; policy's own std scales exploration (paper eq. 15)
            lr             = 1e-4,
            lr_min         = 1e-6, # cosine annealing floor
            clip           = 0.3,
            kl_target      = 0.01,
            K_epochs       = 30,
            B_envs         = 5,
            mb_frac        = 0.1,
            c_value        = 0.5,
            c_entropy      = 0.01,
            gamma          = 1.0,
            use_antithetic = True, # antithetic variates for MC variance reduction
            experiment     = "exp1",
            state_dim      = 2,
            n_episodes     = 500,
            n_strikes      = 10,
            log_every      = 5,
            save_every     = 50,
            conv_window    = 30,
            conv_tol       = 1e-3,
            save_dir       = RESULTS_DIR,
            device         = DEVICE,
        )
    else:
        CFG = TrainConfig(
            n_paths        = 120_000,
            T_steps        = 47,   # matches longest calibration maturity (47 DTE)
            delta          = DELTA,
            n_basis        = 100,
            bp_method      = "knn",
            bp_k           = 1,
            noise_std      = 1.0,  # unit noise; policy's own std scales exploration (paper eq. 15)
            lr             = 1e-4,
            lr_min         = 1e-6, # cosine annealing floor
            clip           = 0.3,
            kl_target      = 0.01,
            K_epochs       = 30,
            B_envs         = 10,
            mb_frac        = 0.1,
            c_value        = 0.5,
            c_entropy      = 0.01,
            gamma          = 1.0,
            use_antithetic = True, # antithetic variates for MC variance reduction
            experiment     = "exp1",
            state_dim      = 2,
            n_episodes     = 2000,
            n_strikes      = 10,
            log_every      = 10,
            save_every     = 100,
            conv_window    = 100,
            conv_tol       = 1e-3,
            conv_patience  = 5,
            save_dir       = RESULTS_DIR,
            device         = DEVICE,
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 60)
    print("Experiment 1: Local Volatility Calibration")
    print("=" * 60)
    print(f"Dataset : {args.dataset}  ({DATA_LABEL})")
    print(f"Device  : {DEVICE}")
    print(f"Results : {RESULTS_DIR}\n")

    # Load market data
    surface = VolSurface(DATA_PATH)
    surface.summary()
    CFG.S0 = surface.spot

    # Train
    trainer = MARLVolTrainer(surface=surface, cfg=CFG)
    log     = trainer.train()

    # Load best checkpoint for plotting
    best_ckpt = os.path.join(RESULTS_DIR, "exp1_best.pt")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=CFG.device, weights_only=False)
        trainer.policy.load_state_dict(ckpt["policy"])
        trainer.policy.norm.load_state_dict(ckpt["policy_norm"])
        print(f"  Loaded best checkpoint (ep {ckpt['episode']}) for plotting.")

    # Save training log
    log_df   = pd.DataFrame(log)
    log_path = os.path.join(RESULTS_DIR, "exp1_training_log.csv")
    log_df.to_csv(log_path, index=False, float_format="%.6f")
    print(f"\nTraining log saved: {log_path}")

    # Learning curve
    plot_learning_curve(
        log,
        save_path = os.path.join(RESULTS_DIR, "exp1_learning_curve.png"),
        l_ref     = trainer.l_ref.item(),
    )

    # In-sample smile plot (training dataset)
    plot_smile_comparison(
        trainer,
        surface,
        save_path  = os.path.join(RESULTS_DIR, "exp1_smile_plot.png"),
        data_label = DATA_LABEL,
    )

    # Out-of-sample smile plot (other dataset — same trained model)
    oos_cfg  = DATASETS.get(OOS_KEY, {})
    oos_path = oos_cfg.get("path", "")
    if oos_path and os.path.exists(oos_path):
        surface_oos = VolSurface(oos_path)
        plot_smile_comparison(
            trainer,
            surface_oos,
            save_path  = os.path.join(RESULTS_DIR, f"exp1_smile_plot_oos_{OOS_KEY}.png"),
            data_label = oos_cfg.get("label", OOS_KEY),
        )
    else:
        print(f"  OOS data not found ({oos_path}) — skipping OOS plot.")

    # Final summary
    best_loss   = min(r["loss"]   for r in log)
    best_ep     = min(log, key=lambda r: r["loss"])["episode"]
    best_reward = max(r["reward"] for r in log)
    final_loss  = log[-1]["loss"]
    l_ref_val   = trainer.l_ref.item()
    improvement = (l_ref_val - best_loss) / l_ref_val * 100

    print(f"\n{'='*60}")
    print(f"Experiment 1 complete  [{args.dataset}]")
    print(f"  BS baseline loss (IV): {l_ref_val:.6f}")
    print(f"  Best train loss  (IV): {best_loss:.6f}  (ep {best_ep})")
    print(f"  Best reward          : {best_reward:+.6f}")
    print(f"  Final train loss     : {final_loss:.6f}")
    print(f"  Improvement          : {improvement:+.1f}%")
    print(f"  Best checkpoint      : {best_ckpt}")
    print(f"{'='*60}")
    print(f"\nTo use this model for Experiment 2 (Bermudan pricing):")
    print(f"  python experiments/exp2_bermudan.py --dataset {args.dataset}")


if __name__ == "__main__":
    main()
