"""
scripts/generate_plots_v2.py
============================
Regenerate smile comparison plots from fixes_v2 best checkpoint (ep 226)
with three corrections vs the original plots:

  Fix 1 — Correct maturity grid:
    Original plot used range(23,52,2), which missed 3/5 calibration
    maturities (22, 32, 42 -> shown as 23, 31, 41 instead).  This plot
    uses the 5 actual calibration maturities in a focused 1x5 panel,
    plus a 3x5 interpolation panel built around those maturities.

  Fix 2 — Correct moneyness range:
    Original used [0.88, 1.12]; training only covers [0.88, 1.09].
    The 1.09-1.12 region is pure extrapolation with no training signal.
    This plot uses [0.88, 1.09] throughout.

  Fix 3 — Stochastic evaluation:
    Original used deterministic sigma = exp(mu_logsig) for every path.
    The model was trained on STOCHASTIC sigmas drawn from the policy.
    We average B_eval=5 independent stochastic rollouts so the plotted
    smile represents what the calibration loss actually measured.

Outputs (saved to results/exp1_aug2018_fixes_v2/):
  exp1_smile_calib_v2.png     -- 1x5 focused calibration maturities
  exp1_smile_interp_v2.png    -- 3x5 broader interpolation grid
  exp1_learning_curve_v2.png  -- learning curve (unchanged, clean replot)

Run:
    cd "E:/AMS 517/Term Project"
    python scripts/generate_plots_v2.py
"""

import os, sys, argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from market_data import VolSurface
from diffusion   import generate_brownian, DELTA
from reward      import implied_vol_batch
from policy      import build_state_exp1, SIGMA_MIN, SIGMA_MAX
from marl_vol    import MARLVolTrainer
from options     import call_payoff_grid

# ---------------------------------------------------------------------------
# CLI args — override defaults to regenerate plots for any run/dataset
# ---------------------------------------------------------------------------
_DATASETS = {
    "aug2018": {
        "data":    "data/spx_smiles_aug2018.csv",
        "oos":     "data/spx_smiles_clean.csv",
        "oos_label": "Apr 2026 (out-of-sample)",
    },
    "apr2026": {
        "data":    "data/spx_smiles_clean.csv",
        "oos":     "data/spx_smiles_aug2018.csv",
        "oos_label": "Aug 2018 (out-of-sample)",
    },
}

_parser = argparse.ArgumentParser(description="Generate corrected smile plots")
_parser.add_argument("--dataset",  default="aug2018", choices=list(_DATASETS.keys()),
                     help="Which dataset the model was trained on (default: aug2018)")
_parser.add_argument("--tag",      default="fixes_v2",
                     help="Run tag suffix (default: fixes_v2)")
_parser.add_argument("--ckpt",     default="",
                     help="Override checkpoint path (default: results/exp1_<dataset>_<tag>/exp1_best.pt)")
_parser.add_argument("--n-eval",   type=int, default=400_000,
                     help="Paths per stochastic rollout (default 400k)")
_parser.add_argument("--b-eval",   type=int, default=5,
                     help="Number of stochastic rollouts to average (default 5)")
_args = _parser.parse_args()

_ds      = _DATASETS[_args.dataset]
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = f"results/exp1_{_args.dataset}_{_args.tag}"
CKPT_PATH   = _args.ckpt or f"{RESULTS_DIR}/exp1_best.pt"
DATA_PATH   = _ds["data"]
OOS_PATH    = _ds["oos"]
OOS_LABEL   = _ds["oos_label"]
N_EVAL      = _args.n_eval   # paths per stochastic rollout (more = smoother)
B_EVAL      = _args.b_eval   # stochastic rollouts to average


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------

def load_best(ckpt_path, surface, results_dir=None):
    ckpt    = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg     = ckpt["cfg"]
    cfg.device = DEVICE
    trainer = MARLVolTrainer(surface=surface, cfg=cfg)
    trainer.policy.load_state_dict(ckpt["policy"])
    trainer.policy.norm.load_state_dict(ckpt["policy_norm"])
    ep      = ckpt["episode"]
    log     = ckpt["log"]
    best_r  = max(r["reward"] for r in log)
    print(f"  Loaded ep {ep}  best_reward={best_r:+.4f}  ({best_r*100:.1f}%)")

    # Try to load the full training log from the last periodic checkpoint
    # (exp1_best.pt only has entries up to the best episode; the full run
    # may have continued for many more episodes — use the last ep*.pt for
    # the complete learning curve.)
    if results_dir is not None:
        import glob as _glob
        ep_ckpts = sorted(
            _glob.glob(os.path.join(results_dir, "exp1_ep*.pt")),
            key=lambda p: int(p.split("_ep")[-1].split(".")[0]),
        )
        if ep_ckpts:
            last_ckpt = torch.load(ep_ckpts[-1], map_location="cpu", weights_only=False)
            full_log  = last_ckpt.get("log", log)
            if len(full_log) > len(log):
                print(f"  Using full log from {os.path.basename(ep_ckpts[-1])} "
                      f"({len(full_log)} episodes)")
                log = full_log

    return trainer, log, ep, best_r


# ---------------------------------------------------------------------------
# Build S_full via stochastic rollouts averaged
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_stochastic_paths(trainer, n_eval=N_EVAL, b_eval=B_EVAL, seed_base=1000):
    """
    Average B_eval stochastic rollouts to get smooth MC prices.
    Each rollout uses the policy with noise (sigma = exp(mu + noise*policy_std)),
    matching exactly what the training loss measured.
    """
    cfg    = trainer.cfg
    policy = trainer.policy
    S0     = cfg.S0
    T      = cfg.T_steps
    device = cfg.device

    # Accumulate payoff sums across rollouts
    # We'll store full path arrays for IV computation at arbitrary maturities
    S_sum = torch.zeros(n_eval, T + 1, device=device)

    policy_std = policy.get_std()   # scalar

    for b in range(b_eval):
        Z     = generate_brownian(n_eval, T, seed=seed_base + b, device=device)
        S_cur = torch.full((n_eval,), S0, device=device)
        S_b   = torch.empty(n_eval, T + 1, device=device)
        S_b[:, 0] = S0

        for t in range(T):
            state      = build_state_exp1(t, S_cur, T, S0)
            mu_logsig, _ = policy(state)
            # Stochastic: add noise ~ N(0, policy_std^2) in log-sigma space
            noise      = torch.randn_like(mu_logsig) * policy_std
            log_sigma  = mu_logsig + noise
            sigma_t    = torch.exp(log_sigma).clamp(SIGMA_MIN, SIGMA_MAX)
            S_cur = S_cur * torch.exp(
                -0.5 * sigma_t ** 2 * DELTA + sigma_t * DELTA ** 0.5 * Z[:, t]
            )
            S_b[:, t + 1] = S_cur

        S_sum += S_b
        print(f"    Stochastic rollout {b+1}/{b_eval} done")

    # Average paths — note: E[payoff(average(S))] ≠ E[payoff(S)] in general.
    # We store the SEPARATE rollout paths and average option PRICES, not paths.
    # So re-do: accumulate payoffs, not paths.
    # (Averaging paths would be wrong; we need to average prices across rollouts.)
    return None   # signal to caller to use payoff-averaging approach


@torch.no_grad()
def compute_mc_ivs_stochastic(trainer, surface, n_eval=N_EVAL, b_eval=B_EVAL,
                               seed_base=1000, moneyness_grid=None):
    """
    Compute IV curves by averaging MC prices over B_eval stochastic rollouts.
    Returns dict: dte_days -> (iv_mc_array, moneyness_valid_mc).
    """
    cfg    = trainer.cfg
    policy = trainer.policy
    S0     = cfg.S0
    T      = cfg.T_steps
    device = cfg.device

    if moneyness_grid is None:
        moneyness_grid = np.linspace(0.88, 1.09, 60)

    K_grid = torch.tensor(moneyness_grid * S0, dtype=torch.float32, device=device)

    # Determine which maturities to evaluate (union of calib + interpolation)
    calib_dte = sorted(surface.maturities_days.astype(int).tolist())
    interp_dte = [20, 24, 29, 35, 37, 39, 44, 49, 51]
    all_eval_dte = sorted(set(calib_dte + interp_dte))
    max_dte = max(all_eval_dte)
    T_needed = max_dte + 1

    policy_std = policy.get_std()

    # Accumulate MC price sums: dte -> price_sum tensor
    price_sums = {dte: torch.zeros(len(K_grid), device=device) for dte in all_eval_dte}

    print(f"  Running {b_eval} stochastic evaluation rollouts ({n_eval:,} paths each)...")
    for b in range(b_eval):
        # We need paths up to max_dte steps
        Z     = generate_brownian(n_eval, T, seed=seed_base + b, device=device)
        S_cur = torch.full((n_eval,), S0, device=device)

        for t in range(T):
            state      = build_state_exp1(t, S_cur, T, S0)
            mu_logsig, _ = policy(state)
            noise      = torch.randn_like(mu_logsig) * policy_std
            sigma_t    = torch.exp(mu_logsig + noise).clamp(SIGMA_MIN, SIGMA_MAX)
            S_cur = S_cur * torch.exp(
                -0.5 * sigma_t ** 2 * DELTA + sigma_t * DELTA ** 0.5 * Z[:, t]
            )

            t_day = t + 1
            if t_day in price_sums:
                payoffs = call_payoff_grid(S_cur, K_grid)   # (n, M)
                price_sums[t_day] += payoffs.mean(dim=0)    # (M,)

        print(f"    rollout {b+1}/{b_eval} done")

    # Average over rollouts and convert to IVs
    results = {}
    for dte in all_eval_dte:
        mc_px  = price_sums[dte] / b_eval                  # averaged MC price
        T_yr   = dte / 252.0
        T_arr  = torch.full_like(K_grid, T_yr)
        iv_mc  = implied_vol_batch(mc_px, S0, K_grid, T_arr)
        valid  = ~torch.isnan(iv_mc)
        results[dte] = {
            "iv_mc":       iv_mc[valid].cpu().numpy() * 100,
            "moneyness":   moneyness_grid[valid.cpu().numpy()],
        }

    return results, moneyness_grid


# ---------------------------------------------------------------------------
# Plot 1: Focused 1x5 — calibration maturities only
# ---------------------------------------------------------------------------

def plot_calib_focused(trainer, surface, save_path, data_label="",
                       n_eval=N_EVAL, b_eval=B_EVAL):
    cfg     = trainer.cfg
    S0      = surface.spot
    device  = cfg.device
    calib_dte = sorted(surface.maturities_days.astype(int).tolist())

    moneyness_grid = np.linspace(0.88, 1.09, 60)

    print(f"\nGenerating focused calibration plot ({len(calib_dte)} panels)...")
    mc_results, _ = compute_mc_ivs_stochastic(
        trainer, surface, n_eval=n_eval, b_eval=b_eval,
        moneyness_grid=moneyness_grid,
    )

    K_grid = torch.tensor(moneyness_grid * S0, dtype=torch.float32)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True,
                             constrained_layout=True)

    for i, dte in enumerate(calib_dte):
        ax    = axes[i]
        T_yr  = dte / 252.0
        res   = mc_results.get(dte, {})

        # Market smile
        iv_mkt = np.array([surface.get_iv(float(k), T_yr)
                            for k in K_grid.tolist()]) * 100
        valid_mkt = ~np.isnan(iv_mkt)

        ax.plot(res["moneyness"], res["iv_mc"],
                color="steelblue", linewidth=2.5, label="Model (MARL)")
        ax.plot(moneyness_grid[valid_mkt], iv_mkt[valid_mkt],
                color="tomato", linestyle="--", linewidth=2.5, label="Market")

        ax.set_title(f"{dte} DTE (calib)", fontsize=11, fontweight="bold")
        ax.set_xlabel("K/S", fontsize=9)
        ax.set_xlim(0.875, 1.095)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("Implied Vol (%)", fontsize=10)
            ax.legend(fontsize=9)

    fig.suptitle(
        f"Exp 1: MARL Local Vol Calibration — Calibration Maturities\n"
        f"SPX Aug 2018 (spot={S0:.0f})  |  {b_eval}x{n_eval//1000}k stochastic rollouts averaged  |  {data_label}",
        fontsize=10,
    )
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Focused calibration plot saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 2: 3x5 interpolation grid — fixed maturities + fixed moneyness
# ---------------------------------------------------------------------------

def plot_interp_grid(trainer, surface, save_path, data_label="",
                     n_eval=N_EVAL, b_eval=B_EVAL):
    cfg   = trainer.cfg
    S0    = surface.spot

    # Build a 15-maturity grid that includes ALL 5 calibration maturities
    calib_dte = sorted(surface.maturities_days.astype(int).tolist())  # [22,27,32,42,47]
    # Fill in with interpolation points, keeping calibration maturities
    # Row 1: 20, 22*, 24, 27*, 29
    # Row 2: 32*, 35, 37, 39, 42*
    # Row 3: 44, 47*, 49, 51, (blank)
    all_dte_grid = [20, 22, 24, 27, 29,
                    32, 35, 37, 39, 42,
                    44, 47, 49, 51]
    N_PANELS = len(all_dte_grid)
    N_COLS, N_ROWS = 5, 3

    moneyness_grid = np.linspace(0.88, 1.09, 60)

    print(f"\nGenerating 3x5 interpolation grid ({N_PANELS} panels)...")
    mc_results, _ = compute_mc_ivs_stochastic(
        trainer, surface, n_eval=n_eval, b_eval=b_eval,
        moneyness_grid=moneyness_grid,
    )

    K_grid     = torch.tensor(moneyness_grid * S0, dtype=torch.float32)
    calib_set  = set(calib_dte)

    fig, axes = plt.subplots(N_ROWS, N_COLS,
                             figsize=(4 * N_COLS, 3.5 * N_ROWS),
                             sharey=True, constrained_layout=True)

    for idx in range(N_ROWS * N_COLS):
        row, col = divmod(idx, N_COLS)
        ax = axes[row, col]

        if idx >= N_PANELS:
            ax.set_visible(False)
            continue

        dte     = all_dte_grid[idx]
        T_yr    = dte / 252.0
        is_cal  = dte in calib_set
        res     = mc_results.get(dte, {})

        iv_mkt = np.array([surface.get_iv(float(k), T_yr)
                           for k in K_grid.tolist()]) * 100
        valid_mkt = ~np.isnan(iv_mkt)

        color = "steelblue" if is_cal else "seagreen"
        tag   = " (*)" if is_cal else ""
        ax.plot(res.get("moneyness", []), res.get("iv_mc", []),
                color=color, linewidth=2, label="Model (MARL)")
        ax.plot(moneyness_grid[valid_mkt], iv_mkt[valid_mkt],
                color="tomato", linestyle="--", linewidth=2, label="Market")

        ax.set_title(f"{dte}d{tag}", fontsize=10,
                     fontweight="bold" if is_cal else "normal")
        ax.set_xlabel("K/S", fontsize=8)
        ax.set_xlim(0.875, 1.095)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("Implied Vol (%)", fontsize=9)
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        f"Exp 1: MARL Local Vol Calibration — SPX Aug 2018 (spot={S0:.0f})  |  {data_label}\n"
        f"(*) = calibration maturity  |  others = interpolated  |  "
        f"moneyness [0.88, 1.09]  |  {b_eval}x{n_eval//1000}k stochastic rollouts",
        fontsize=9,
    )
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Interpolation grid saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 3: Learning curve (clean replot)
# ---------------------------------------------------------------------------

def plot_learning_curve_v2(log, save_path, l_ref, tag="", best_ep=None, best_r=None):
    episodes = [r["episode"] for r in log]
    losses   = [r["loss"]    for r in log]
    rewards  = [r["reward"]  for r in log]

    if best_ep is None:
        best_ep = max(log, key=lambda r: r["reward"])["episode"]
    if best_r is None:
        best_r  = max(r["reward"] for r in log)

    # Smoothed reward (50-ep rolling mean, falls back to shorter window)
    win = min(50, len(rewards) // 2)
    if win > 1:
        smooth    = np.convolve(rewards, np.ones(win)/win, mode="valid")
        smooth_ep = episodes[win - 1:]
    else:
        smooth, smooth_ep = rewards, episodes

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(episodes, losses, color="steelblue", linewidth=0.8, alpha=0.6,
             label="IV loss per episode")
    ax1.axhline(l_ref, color="tomato", linestyle="--", linewidth=1.5,
                label=f"BS baseline L_ref={l_ref:.4f}")
    ax1.set_ylabel("Calibration loss (IV space)")
    run_label = f"  [{tag}]" if tag else ""
    ax1.set_title(
        f"Experiment 1{run_label}: Learning Curve  "
        f"(best: ep {best_ep}, {best_r*100:.1f}%)"
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    ax2.plot(episodes, rewards, color="seagreen", linewidth=0.8, alpha=0.4,
             label="Episode reward")
    ax2.plot(smooth_ep, smooth, color="darkgreen", linewidth=2,
             label=f"{win}-ep rolling mean")
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.axvline(best_ep, color="tomato", linestyle=":", linewidth=1.5,
                label=f"Best ep ({best_ep}, {best_r*100:.1f}%)")
    ax2.set_ylabel("Normalised reward  (L_ref - L) / L_ref")
    ax2.set_xlabel("Episode")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Learning curve saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Generating fixes_v2 plots (corrected evaluation)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"n_eval={N_EVAL:,}  b_eval={B_EVAL}  (total paths: {N_EVAL*B_EVAL:,})")

    surface = VolSurface(DATA_PATH)
    surface.summary()

    print("\nLoading checkpoint...")
    trainer, log, best_ep, best_r = load_best(CKPT_PATH, surface,
                                               results_dir=RESULTS_DIR)

    # Learning curve
    l_ref_val = trainer.l_ref.item()
    plot_learning_curve_v2(
        log,
        save_path=os.path.join(RESULTS_DIR, "exp1_learning_curve_v2.png"),
        l_ref=l_ref_val,
        tag=_args.tag,
        best_ep=best_ep,
        best_r=best_r,
    )

    # Focused 1x5 calibration plot (in-sample)
    plot_calib_focused(
        trainer, surface,
        save_path=os.path.join(RESULTS_DIR, "exp1_smile_calib_v2.png"),
        data_label="Aug 2018 (in-sample)",
        n_eval=N_EVAL, b_eval=B_EVAL,
    )

    # 3x5 interpolation grid (in-sample)
    plot_interp_grid(
        trainer, surface,
        save_path=os.path.join(RESULTS_DIR, "exp1_smile_interp_v2.png"),
        data_label="Aug 2018 (in-sample)",
        n_eval=N_EVAL, b_eval=B_EVAL,
    )

    # OOS focused plot (Apr 2026)
    if os.path.exists(OOS_PATH):
        surface_oos = VolSurface(OOS_PATH)
        oos_tag = "aug2018" if _args.dataset == "apr2026" else "apr2026"
        plot_calib_focused(
            trainer, surface_oos,
            save_path=os.path.join(RESULTS_DIR, f"exp1_smile_calib_v2_oos_{oos_tag}.png"),
            data_label=OOS_LABEL,
            n_eval=N_EVAL, b_eval=B_EVAL,
        )

    print("\n" + "=" * 60)
    print("All v2 plots saved to", RESULTS_DIR)
    print("=" * 60)
