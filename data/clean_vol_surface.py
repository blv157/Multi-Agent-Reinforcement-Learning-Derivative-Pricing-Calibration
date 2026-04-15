"""
SPX Vol Surface Cleaning
========================
Takes the raw spx_smiles.csv collected by collect_vol_surface.py and produces
a clean spx_smiles_clean.csv suitable for use in the MARL calibration project.

Two problems are fixed here:

  Problem 1 — Too many maturities
  --------------------------------
  SPX has both monthly AND weekly options, so the raw data contains 14
  expirations clustered in the 18–48 DTE window. The paper works with a small
  number of distinct maturities (they use t1=21 to t2=51 days). Training
  against 14 nearly-identical maturities doesn't improve calibration quality —
  it just multiplies the computational cost.

  Fix: pick a small set of representative maturities spread across the window,
  keeping one expiration per "slot". We target 5 maturities roughly evenly
  spaced between 21 and 51 DTE.

  Problem 2 — IV outliers at deep OTM strikes
  --------------------------------------------
  Deep out-of-the-money options are illiquid. Their bid/ask spreads are wide,
  last-trade prices are often stale, and the implied vols computed from mid
  prices can be unreliable. In the raw data some slices show IV > 0.5 or even
  0.74, which are real quotes but unreliable.

  If we train the model to match these outliers it wastes representational
  capacity on noisy data points. We remove strikes whose IV is more than
  N standard deviations above the median within each maturity slice, and also
  enforce a hard cap.

  Additionally we tighten the moneyness band: the paper's Figure 1 shows smiles
  across roughly [0.88, 1.12] moneyness. We don't need anything beyond that.

Run:
    python data/clean_vol_surface.py

Inputs:  data/spx_smiles.csv
Outputs: data/spx_smiles_clean.csv
         data/spx_smiles_clean_plot.png
"""

import numpy as np
import pandas as pd
import datetime

# ── Cleaning parameters (adjust if needed) ────────────────────────────────────

# Target number of maturities to keep.
# We'll pick this many expirations spread as evenly as possible across the
# DTE window defined below.
N_MATURITIES = 5

# DTE window that the paper uses (21–51 days).
# We only consider expirations inside this range when selecting maturities.
DTE_MIN = 21
DTE_MAX = 51

# Moneyness band: only keep strikes where DTE_MIN <= K/S <= MAX_MONEYNESS.
# The paper's smiles are shown roughly in [0.88, 1.12]; we use a tiny buffer.
MONO_MIN = 0.875
MONO_MAX = 1.125

# Outlier threshold: within each maturity, drop any strike whose IV is more
# than this many median-absolute-deviations (MAD) above the median.
# MAD is used instead of std because it is itself robust to outliers.
MAD_THRESHOLD = 3.0

# Hard IV cap: regardless of MAD filter, drop any strike with IV above this.
# 0.60 = 60% annualised vol, which is extreme even for short-dated SPX options
# in a stressed market. Anything above this in our data is almost certainly a
# stale or erroneous quote.
IV_HARD_CAP = 0.60

# Minimum number of strikes required to keep a maturity slice.
# If after cleaning a slice has fewer than this many strikes, drop it entirely
# (a sparse smile can't constrain the calibration well).
MIN_STRIKES = 8

# ─────────────────────────────────────────────────────────────────────────────


def select_maturities(available_dtes, n, dte_min, dte_max):
    """
    Choose n DTE values from available_dtes that are:
      (a) inside [dte_min, dte_max], and
      (b) as evenly spaced as possible across that range.

    Strategy: create n evenly-spaced target DTE values between dte_min and
    dte_max, then for each target pick the closest available DTE. Deduplicate
    in case two targets map to the same expiration.
    """
    # Candidate DTE values that fall in the window
    candidates = sorted([d for d in available_dtes if dte_min <= d <= dte_max])

    if len(candidates) == 0:
        raise ValueError(
            f"No maturities found in [{dte_min}, {dte_max}] DTE window. "
            f"Available: {sorted(available_dtes)}"
        )

    if len(candidates) <= n:
        # Fewer candidates than requested — just take all of them
        print(f"  Only {len(candidates)} maturities available in window; using all.")
        return candidates

    # Ideal evenly-spaced targets
    targets = np.linspace(dte_min, dte_max, n)

    selected = []
    for t in targets:
        best = min(candidates, key=lambda d: abs(d - t))
        if best not in selected:
            selected.append(best)

    # If deduplication reduced us below n, fill gaps from remaining candidates
    remaining = [d for d in candidates if d not in selected]
    while len(selected) < n and remaining:
        # Add whichever remaining DTE is furthest from any already-selected DTE
        best_gap = max(remaining, key=lambda d: min(abs(d - s) for s in selected))
        selected.append(best_gap)
        remaining.remove(best_gap)

    return sorted(selected)


def remove_iv_outliers(df, mad_threshold, iv_hard_cap):
    """
    Within each maturity slice, flag and remove IV outliers.

    Two-pass filter:
      Pass 1: hard cap — any IV above iv_hard_cap is unconditionally removed.
      Pass 2: MAD filter — compute the median and MAD of the remaining IVs,
              remove any point more than mad_threshold * MAD above the median.
              We only filter the upper tail (high IV = deep OTM) because the
              lower tail (low IV = deep ITM calls) is naturally bounded and
              doesn't produce the same stale-quote artefacts.
    """
    cleaned_slices = []

    for dte, grp in df.groupby("maturity_days"):
        n_start = len(grp)

        # Pass 1: hard cap
        grp = grp[grp["implied_vol"] <= iv_hard_cap].copy()
        n_after_cap = len(grp)

        # Pass 2: MAD filter on upper tail
        median_iv = grp["implied_vol"].median()
        mad = (grp["implied_vol"] - median_iv).abs().median()

        if mad > 0:
            upper_bound = median_iv + mad_threshold * mad
            grp = grp[grp["implied_vol"] <= upper_bound].copy()

        n_end = len(grp)
        removed = n_start - n_end
        if removed > 0:
            print(f"  {dte:3d} DTE: removed {removed} outlier(s)  "
                  f"({n_start} -> {n_end} strikes)")

        cleaned_slices.append(grp)

    return pd.concat(cleaned_slices, ignore_index=True)


def clean(input_path="data/spx_smiles.csv",
          output_path="data/spx_smiles_clean.csv"):

    # ── Load raw data ─────────────────────────────────────────────────────────
    raw = pd.read_csv(input_path)
    print(f"Loaded {len(raw)} rows from {input_path}")
    print(f"Raw maturities (DTE): {sorted(raw['maturity_days'].unique())}\n")

    # ── Step 1: Tighten moneyness band ────────────────────────────────────────
    # Keep only strikes in [MONO_MIN, MONO_MAX]. This removes deep OTM and deep
    # ITM options that have poor liquidity and aren't part of the "smile" we're
    # trying to calibrate to.
    print(f"Step 1: Restricting to moneyness [{MONO_MIN}, {MONO_MAX}]")
    df = raw[(raw["moneyness"] >= MONO_MIN) & (raw["moneyness"] <= MONO_MAX)].copy()
    print(f"  {len(raw)} -> {len(df)} rows\n")

    # ── Step 2: Remove IV outliers ────────────────────────────────────────────
    print(f"Step 2: Removing IV outliers  "
          f"(hard cap={IV_HARD_CAP}, MAD threshold={MAD_THRESHOLD})")
    df = remove_iv_outliers(df, MAD_THRESHOLD, IV_HARD_CAP)
    print(f"  {len(df)} rows remain\n")

    # ── Step 3: Select representative maturities ──────────────────────────────
    # After the above filtering, pick N_MATURITIES expirations spread across
    # the DTE window. All other expirations are dropped.
    print(f"Step 3: Selecting {N_MATURITIES} representative maturities "
          f"in [{DTE_MIN}, {DTE_MAX}] DTE")
    available_dtes = df["maturity_days"].unique()
    chosen_dtes = select_maturities(available_dtes, N_MATURITIES, DTE_MIN, DTE_MAX)
    print(f"  Chosen DTE values: {chosen_dtes}")
    df = df[df["maturity_days"].isin(chosen_dtes)].copy()
    print(f"  {len(df)} rows remain\n")

    # ── Step 4: Drop slices with too few strikes ───────────────────────────────
    # A smile with fewer than MIN_STRIKES points can't meaningfully constrain
    # the calibration — it will produce an overfit to just a few quotes.
    print(f"Step 4: Dropping maturities with fewer than {MIN_STRIKES} strikes")
    before = df["maturity_days"].nunique()
    counts = df.groupby("maturity_days").size()
    keep_dtes = counts[counts >= MIN_STRIKES].index
    df = df[df["maturity_days"].isin(keep_dtes)].copy()
    after = df["maturity_days"].nunique()
    if before != after:
        dropped = set(chosen_dtes) - set(keep_dtes)
        print(f"  Dropped {before - after} maturity/maturities: {dropped}")
    else:
        print(f"  All maturities passed.")
    print()

    # ── Step 5: Sort and reset index ──────────────────────────────────────────
    df = df.sort_values(["maturity_days", "strike"]).reset_index(drop=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("-" * 60)
    print("Clean surface summary:")
    print(f"  Total rows : {len(df)}")
    print(f"  Spot price : {df['spot'].iloc[0]:.2f}")
    print(f"  Collection : {df['expiry'].iloc[0][:10]} data")
    print()
    for dte, grp in df.groupby("maturity_days"):
        expiry = grp["expiry"].iloc[0]
        print(f"  {dte:3d} DTE  ({expiry})  "
              f"{len(grp):3d} strikes  "
              f"IV [{grp['implied_vol'].min():.3f}, {grp['implied_vol'].max():.3f}]  "
              f"moneyness [{grp['moneyness'].min():.3f}, {grp['moneyness'].max():.3f}]")
    print()

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"Saved -> {output_path}")

    # ── Plot: raw vs clean side by side ───────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        today = datetime.date.today()
        spot  = df["spot"].iloc[0]

        # Left: raw surface (all maturities, all strikes)
        ax = axes[0]
        for dte, grp in raw.groupby("maturity_days"):
            ax.scatter(grp["moneyness"], grp["implied_vol"] * 100,
                       s=6, alpha=0.4, label=f"{dte}d")
        ax.set_title(f"Raw  ({len(raw)} points, {raw['maturity_days'].nunique()} maturities)")
        ax.set_xlabel("Moneyness (K/S)")
        ax.set_ylabel("Implied Vol (%)")
        ax.set_xlim(0.80, 1.20)
        ax.grid(True, alpha=0.3)

        # Right: clean surface (selected maturities, outliers removed)
        ax = axes[1]
        colours = plt.cm.viridis(np.linspace(0.15, 0.85, df["maturity_days"].nunique()))
        for (dte, grp), col in zip(df.groupby("maturity_days"), colours):
            expiry = grp["expiry"].iloc[0]
            ax.plot(grp["moneyness"], grp["implied_vol"] * 100,
                    marker="o", markersize=4, color=col, label=f"{dte}d ({expiry})")
        ax.set_title(f"Clean  ({len(df)} points, {df['maturity_days'].nunique()} maturities)")
        ax.set_xlabel("Moneyness (K/S)")
        ax.set_ylabel("Implied Vol (%)")
        ax.legend(title="DTE", fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"SPX Vol Surface — {today}  (spot={spot:.0f})", fontsize=13)
        plt.tight_layout()
        plot_path = output_path.replace(".csv", "_plot.png")
        plt.savefig(plot_path, dpi=130)
        print(f"Plot saved  -> {plot_path}")
        plt.close()
    except ImportError:
        pass


if __name__ == "__main__":
    clean()
