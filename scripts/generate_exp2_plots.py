"""
scripts/generate_exp2_plots.py
================================
Reconstruct exp2 training logs and generate plots from saved checkpoints.

Used when the non_path_dep run was killed before writing its CSV — this
script parses the raw log file and generates all exp2 figures.

Usage:
    cd "E:\AMS 517\Term Project"
    python scripts/generate_exp2_plots.py --dataset aug2018 --tag fixes_v3
"""

import os
import sys
import re
import argparse

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from market_data import VolSurface
from marl_vol    import MARLVolTrainer, TrainConfig
from options     import make_bermudan
from diffusion   import DELTA
from exp2_bermudan import (
    BASE_CFG,
    plot_bermudan_price_curve,
    plot_vol_surface_comparison,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_log(log_path: str, mode: str) -> pd.DataFrame:
    """Parse a raw training log file into a DataFrame."""
    records = []
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(
                r"Ep\s+(\d+)/\d+\s+loss=([\d.]+)\s+reward=([+-]?[\d.]+)",
                line.strip(),
            )
            if m:
                records.append({
                    "episode": int(m.group(1)),
                    "loss":    float(m.group(2)),
                    "reward":  float(m.group(3)),
                    "mode":    mode,
                })
    return pd.DataFrame(records)


def load_trainer(path_dep: bool, ckpt_path: str, surface: VolSurface) -> MARLVolTrainer:
    """Reconstruct a MARLVolTrainer from a saved checkpoint."""
    state_dim   = 5 if path_dep else 3
    results_dir = os.path.dirname(ckpt_path)

    cfg = TrainConfig(
        **{k: v for k, v in BASE_CFG.items() if k != "device"},
        state_dim = state_dim,
        path_dep  = path_dep,
        S0        = surface.spot,
        save_dir  = results_dir,
        device    = DEVICE,
    )

    bermudan = make_bermudan(
        strike   = surface.spot,
        t1_step  = 21,
        t2_step  = cfg.T_steps,
    )

    trainer = MARLVolTrainer(surface=surface, cfg=cfg, bermudan=bermudan)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Load policy weights
    trainer.policy.load_state_dict(ckpt["policy"])
    trainer.policy.to(DEVICE)
    trainer.policy.eval()

    # Load policy state normalizer
    pn = ckpt["policy_norm"]
    norm_sd          = trainer.policy.norm.state_dict()
    norm_sd["mean"]  = pn["mean"]
    norm_sd["var"]   = pn["var"]
    norm_sd["count"] = pn["count"]
    trainer.policy.norm.load_state_dict(norm_sd)

    # Load value weights (for completeness)
    if "value" in ckpt:
        trainer.value.load_state_dict(ckpt["value"])
        trainer.value.to(DEVICE)

    tag = "path_dep" if path_dep else "non_path_dep"
    ep  = ckpt.get("episode", "?")
    print(f"  Loaded {tag} checkpoint (ep={ep}) from {ckpt_path}")
    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="aug2018",
                        choices=["aug2018", "apr2026"])
    parser.add_argument("--tag",     default="fixes_v3")
    parser.add_argument("--log_suffix", default="v5",
                        help="Suffix for log filenames, e.g. 'v5' -> "
                             "exp2_{dataset}_path_dep_v5.log")
    args = parser.parse_args()

    DATASETS = {
        "aug2018": "data/spx_smiles_aug2018.csv",
        "apr2026": "data/spx_smiles_clean.csv",
    }
    tag_suffix  = f"_{args.tag}" if args.tag else ""
    RESULTS_DIR = f"results/exp2_{args.dataset}{tag_suffix}"
    DATA_PATH   = DATASETS[args.dataset]
    LOG_SUFFIX  = f"_{args.log_suffix}" if args.log_suffix else ""

    print(f"Dataset   : {args.dataset}")
    print(f"Results   : {RESULTS_DIR}")
    print(f"Device    : {DEVICE}")

    # ── Load surface ─────────────────────────────────────────────────────────
    surface = VolSurface(DATA_PATH)
    surface.summary()

    # ── Build / load training logs ───────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "exp2_training_log.csv")

    # Always re-parse from raw log files so the CSV reflects full history
    log_dfs = {}

    # For path_dep: prefer the full per-episode CSV (written by trainer.train())
    # For non_path_dep: parse from the raw log (run was killed before CSV was written)
    existing_csv = pd.read_csv(csv_path) if os.path.exists(csv_path) else None

    for mode in ("path_dep", "non_path_dep"):
        # Try full CSV first (has every episode, not just every log_every)
        if existing_csv is not None:
            subset = existing_csv[existing_csv["mode"] == mode]
            if len(subset) > 0:
                log_dfs[mode] = subset.reset_index(drop=True)
                print(f"  Loaded {len(subset)} episodes from CSV for {mode}")
                continue

        # Fall back to raw log file
        log_file = f"logs/exp2_{args.dataset}_{mode}{LOG_SUFFIX}.log"
        if os.path.exists(log_file):
            df = parse_log(log_file, mode)
            if len(df):
                log_dfs[mode] = df
                print(f"  Parsed {len(df)} episodes from {log_file}")
            else:
                print(f"  WARNING: no episode lines found in {log_file}")
        else:
            print(f"  WARNING: no data found for {mode}")

    if not log_dfs:
        print("ERROR: no training data found. Exiting.")
        return

    # Save combined CSV
    combined = pd.concat(list(log_dfs.values()), ignore_index=True)
    combined.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nCombined training log saved: {csv_path}  ({len(combined)} rows)")

    # ── Bermudan price curve (just needs log DataFrames) ─────────────────────
    price_plot_path = os.path.join(RESULTS_DIR, "exp2_bermudan_price_plot.png")
    plot_bermudan_price_curve(
        log_path  = log_dfs,
        save_path = price_plot_path,
        surface   = surface,
    )

    # ── Vol surface comparison (needs trainer objects) ────────────────────────
    trainers = {}
    for mode, path_dep in [("path_dep", True), ("non_path_dep", False)]:
        ckpt = os.path.join(RESULTS_DIR, f"exp2_{mode}_best.pt")
        if os.path.exists(ckpt):
            trainers[mode] = load_trainer(path_dep, ckpt, surface)
        else:
            print(f"  WARNING: checkpoint not found: {ckpt}")

    if "path_dep" in trainers and "non_path_dep" in trainers:
        plot_vol_surface_comparison(
            trainer_pd  = trainers["path_dep"],
            trainer_npd = trainers["non_path_dep"],
            surface     = surface,
            save_path   = os.path.join(RESULTS_DIR, "exp2_smile_comparison.png"),
        )
    else:
        print("  Skipping smile comparison (need both checkpoints)")

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Exp2 plot generation complete.")
    for mode, df in log_dfs.items():
        best_idx = df["reward"].idxmax()
        best_ep  = df.loc[best_idx, "episode"]
        best_r   = df.loc[best_idx, "reward"]
        final_r  = df["reward"].iloc[-1]
        mean_last100 = df.tail(100)["reward"].mean()
        print(f"  [{mode}] best r={best_r:+.4f} (ep{best_ep})  "
              f"final r={final_r:+.4f}  last-100 mean={mean_last100:+.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
