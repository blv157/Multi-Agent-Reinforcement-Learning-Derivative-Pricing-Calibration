"""
experiments/exp1_local_vol.py
==============================
Experiment 1: Local Volatility Calibration

Replicates Figure 1 of Vadori (2022).

Objective
----------
Train MARL agents to find a local volatility function sigma(t, S) such
that the MC prices of vanilla calls match the SPX market smile across
5 maturities (23, 27, 34, 41, 48 DTE) and 10 strikes per maturity.

State:     s_it = (t/T, S_it/S0)  -- 2-dimensional
Reward:    r_T  = L_ref - L(sigma) (improvement over flat BS baseline)
Algorithm: PPO with basis player exploration (Algorithm 1)

How to run
----------
    cd "E:/AMS 517/Term Project"
    python experiments/exp1_local_vol.py

Results are saved to results/exp1/:
    exp1_ep{N}.pt           -- checkpoints every 100 episodes
    exp1_training_log.csv   -- loss and reward per episode
    exp1_smile_plot.png     -- fitted vs market implied vol smiles (Figure 1)
    exp1_learning_curve.png -- calibration loss over training
"""

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
from policy         import build_state_exp1


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH   = "data/spx_smiles_clean.csv"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Toggle between small diagnostic run and full-scale paper replication
# ---------------------------------------------------------------------------
SMALL_RUN = False   # set False to run full 120k-path / 500-episode experiment

if SMALL_RUN:
    # Fast diagnostic: ~10 min on GPU. Use this to confirm convergence
    # before committing to the full run.
    RESULTS_DIR = "results/exp1_small"
    CFG = TrainConfig(
        n_paths    = 10_000,
        T_steps    = 51,
        delta      = DELTA,
        n_basis    = 50,
        bp_method  = "knn",
        bp_k       = 1,
        noise_std  = 0.02,
        lr         = 1e-4,
        clip       = 0.3,
        kl_target  = 0.01,
        K_epochs   = 30,
        B_envs     = 5,
        mb_frac    = 0.1,
        c_value    = 0.5,
        c_entropy  = 0.01,
        gamma      = 1.0,
        experiment = "exp1",
        state_dim  = 2,
        n_episodes  = 500,
        n_strikes   = 10,
        log_every   = 5,
        save_every  = 50,
        conv_window = 30,
        conv_tol    = 1e-3,
        save_dir    = RESULTS_DIR,
        device      = DEVICE,
    )
else:
    # Full-scale config matching the paper
    RESULTS_DIR = "results/exp1"
    CFG = TrainConfig(
        n_paths     = 120_000,
        T_steps     = 51,
        delta       = DELTA,
        n_basis     = 100,
        bp_method   = "knn",
        bp_k        = 1,
        noise_std   = 0.02,
        lr          = 1e-4,
        clip        = 0.3,
        kl_target   = 0.01,
        K_epochs    = 30,
        B_envs      = 10,
        mb_frac     = 0.1,
        c_value     = 0.5,
        c_entropy   = 0.01,
        gamma       = 1.0,
        experiment  = "exp1",
        state_dim   = 2,
        n_episodes  = 2000,   # max cap; convergence check will stop earlier
        n_strikes   = 10,
        log_every   = 10,
        save_every  = 100,
        conv_window = 50,     # check over 50-episode windows
        conv_tol    = 1e-3,
        save_dir    = RESULTS_DIR,
        device      = DEVICE,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_smile_comparison(
    trainer:    MARLVolTrainer,
    surface:    VolSurface,
    save_path:  str,
    n_eval:     int = 200_000,
    data_label: str = "Apr 2026",
):
    """
    Generate Figure 1: fitted vs market implied vol smiles.

    Replicates the paper's full smile grid: 15 maturities from T=23 to
    T=51 in steps of 2 trading days, arranged in a 3-row × 5-column figure.

    Of these 15 panels, the calibration maturities (23, 27, 41 DTE) are
    marked with (*) in their titles — these were directly in the training
    loss.  All other panels show interpolated smiles that emerge from
    forward propagation of the trained local vol surface sigma(t, S) at
    maturities the model was never explicitly calibrated to.

    Note: calibration maturities 34 and 48 DTE fall between the step-2
    grid points and are therefore not shown separately; their influence is
    captured by the surrounding panels.

    Market IVs at every maturity (calibration or interpolated) are obtained
    from VolSurface.get_iv(), which performs cubic-spline interpolation of
    total variance between the available market maturities.

    Parameters
    ----------
    trainer   : trained MARLVolTrainer
    surface   : VolSurface with market data
    save_path : where to write the PNG
    n_eval    : number of MC paths for the evaluation rollout
    """
    cfg    = trainer.cfg
    device = cfg.device
    S0     = surface.spot
    policy = trainer.policy

    # ── Maturity grid: T=23 to T=51, step 2 → 15 maturities ────────────────
    all_dte   = list(range(23, 52, 2))   # [23, 25, 27, ..., 51]
    calib_set = set(surface.maturities_days.astype(int).tolist())
    N_COLS, N_ROWS = 5, 3                # 3 × 5 = 15 panels

    print(f"\nGenerating smile comparison plot "
          f"({len(all_dte)} maturities, {N_ROWS}×{N_COLS} grid)...")

    # ── Evaluation rollout (no noise) ────────────────────────────────────────
    # Run the trained policy deterministically (mean mu, no exploration noise)
    # and store every price level so smiles at any DTE ≤ T_steps can be read
    # from a single simulation pass.
    with torch.no_grad():
        Z      = generate_brownian(n_eval, cfg.T_steps, seed=999, device=device)
        sigmas = torch.zeros(n_eval, cfg.T_steps, device=device)

        S_cur = torch.full((n_eval,), S0, device=device)
        for t in range(cfg.T_steps):
            state        = build_state_exp1(t, S_cur, cfg.T_steps, S0)
            mu, _        = policy(state)
            sigmas[:, t] = mu
            S_cur = S_cur * torch.exp(
                -0.5 * mu ** 2 * DELTA + mu * DELTA ** 0.5 * Z[:, t]
            )

        # Full price matrix: S_full[:, t] = prices at t trading days
        S_full = torch.empty(n_eval, cfg.T_steps + 1, device=device)
        S_full[:, 0] = S0
        S_temp = torch.full((n_eval,), S0, device=device)
        for t in range(cfg.T_steps):
            S_temp = S_temp * torch.exp(
                -0.5 * sigmas[:, t] ** 2 * DELTA
                + sigmas[:, t] * DELTA ** 0.5 * Z[:, t]
            )
            S_full[:, t + 1] = S_temp

    # ── Dense moneyness grid for smooth curves ───────────────────────────────
    moneyness_grid = np.linspace(0.88, 1.12, 50)
    K_grid = torch.tensor(moneyness_grid * S0, dtype=torch.float32, device=device)

    from options import call_payoff_grid

    def _smile_curves(dte_days):
        """MC model IVs and market IVs for a single maturity."""
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

    # ── 3 × 5 figure ─────────────────────────────────────────────────────────
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

        # Calibration maturities use steelblue; interpolated use seagreen
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
                linewidth=1.5, label=f"BS baseline ({l_ref:.4f})")
    ax1.set_ylabel("Calibration loss")
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
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 60)
    print("Experiment 1: Local Volatility Calibration")
    print("=" * 60)
    print(f"Device : {DEVICE}")
    print(f"Results: {RESULTS_DIR}\n")

    # ── Load market data ──────────────────────────────────────────────────────
    surface = VolSurface(DATA_PATH)
    surface.summary()
    CFG.S0 = surface.spot

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = MARLVolTrainer(surface=surface, cfg=CFG)
    log     = trainer.train()

    # ── Save training log ─────────────────────────────────────────────────────
    log_df = pd.DataFrame(log)
    log_path = os.path.join(RESULTS_DIR, "exp1_training_log.csv")
    log_df.to_csv(log_path, index=False, float_format="%.6f")
    print(f"\nTraining log saved: {log_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_learning_curve(
        log,
        save_path = os.path.join(RESULTS_DIR, "exp1_learning_curve.png"),
        l_ref     = trainer.l_ref.item(),
    )

    plot_smile_comparison(
        trainer,
        surface,
        save_path  = os.path.join(RESULTS_DIR, "exp1_smile_plot.png"),
        data_label = "Apr 2026",
    )

    # ── August 2018 out-of-sample comparison ──────────────────────────────────
    # Run the same trained model against a synthetic August 1, 2018 SPX surface
    # (spot=2816, VIX~11.6, mild skew) to show how the local vol function
    # transfers across market regimes.  No retraining — same policy weights.
    aug2018_path = "data/spx_smiles_aug2018.csv"
    if os.path.exists(aug2018_path):
        surface_aug = VolSurface(aug2018_path)
        plot_smile_comparison(
            trainer,
            surface_aug,
            save_path  = os.path.join(RESULTS_DIR, "exp1_smile_plot_aug2018.png"),
            data_label = "Aug 2018 (synthetic, spot=2816)",
        )
    else:
        print(f"  Aug 2018 data not found at {aug2018_path} — skipping.")

    # ── Final summary ─────────────────────────────────────────────────────────
    final_loss = log[-1]["loss"]
    l_ref_val  = trainer.l_ref.item()
    improvement = (l_ref_val - final_loss) / l_ref_val * 100

    print(f"\n{'='*60}")
    print(f"Experiment 1 complete.")
    print(f"  BS baseline loss : {l_ref_val:.6f}")
    print(f"  Final train loss : {final_loss:.6f}")
    print(f"  Improvement      : {improvement:+.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
