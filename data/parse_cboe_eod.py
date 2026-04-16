#!/usr/bin/env python3
"""
data/parse_cboe_eod.py
======================
Convert a CBOE EOD Options Calculations CSV into the project's smile CSV format.

The CBOE file (UnderlyingOptionsEODCalcs_YYYY-MM-DD.csv) contains:
  - implied_volatility_1545  : CBOE-computed IV at 3:45 PM
  - bid_eod / ask_eod        : end-of-day best bid / ask
  - underlying_bid_eod / underlying_ask_eod : EOD underlying quotes
  - root: 'SPX' (monthly) or 'SPXW' (weekly) — same economics, may duplicate strikes

Output CSV columns (matches spx_smiles_clean.csv):
  maturity_days, expiry, strike, mid, implied_vol, spot, moneyness

Usage
-----
  # 2018-08-01 data -> replace our synthetic Aug 2018 file:
  python data/parse_cboe_eod.py \\
      --cboe data/UnderlyingOptionsEODCalcs_2018-08-01.csv \\
      --out  data/spx_smiles_aug2018.csv

  # 2026-04-01 data -> replace our existing training surface:
  python data/parse_cboe_eod.py \\
      --cboe data/UnderlyingOptionsEODCalcs_2026-04-01.csv \\
      --out  data/spx_smiles_clean.csv
"""

import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd

# ── Project constants ─────────────────────────────────────────────────────────
TARGET_DTE   = [23, 27, 34, 41, 48]   # paper calibration maturities (trading days)
MONEYNESS_LO = 0.85
MONEYNESS_HI = 1.15
MAX_STRIKES  = 80                      # cap per maturity (sub-sampled evenly if exceeded)
MIN_IV       = 0.01                    # drop options with IV < 1%
MAX_IV       = 3.00                    # drop options with IV > 300% (data error)
MIN_MID      = 0.05                    # drop options with mid < $0.05


# ── Calendar helper ───────────────────────────────────────────────────────────

def trading_days_between(d1: datetime.date, d2: datetime.date) -> int:
    """Count Mon-Fri trading days strictly between d1 and d2 (approximate — no holiday calendar)."""
    days, cur = 0, d1
    while cur < d2:
        cur += datetime.timedelta(days=1)
        if cur.weekday() < 5:
            days += 1
    return days


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_cboe(cboe_path: str, out_path: str, max_dte: int = 51) -> None:
    print(f"\nReading: {cboe_path}")
    df = pd.read_csv(cboe_path, low_memory=False)
    print(f"  {len(df):,} rows loaded")

    # ── Quote date and spot price ─────────────────────────────────────────────
    quote_date_str = str(df["quote_date"].iloc[0])          # e.g. '2018-08-01'
    quote_date     = datetime.datetime.strptime(quote_date_str, "%Y-%m-%d").date()

    spot_bid = df["underlying_bid_eod"].median()
    spot_ask = df["underlying_ask_eod"].median()
    spot     = (spot_bid + spot_ask) / 2.0
    print(f"  Quote date : {quote_date_str}")
    print(f"  Spot (eod mid): {spot:.2f}")

    # ── Filter to calls with valid IV ─────────────────────────────────────────
    df = df[df["option_type"] == "C"].copy()
    df["iv"]  = pd.to_numeric(df["implied_volatility_1545"], errors="coerce")
    df["bid"] = pd.to_numeric(df["bid_eod"],                 errors="coerce")
    df["ask"] = pd.to_numeric(df["ask_eod"],                 errors="coerce")
    df["mid"] = (df["bid"] + df["ask"]) / 2.0

    df = df[
        df["iv"].notna()  & (df["iv"]  > MIN_IV)  & (df["iv"]  < MAX_IV) &
        df["mid"].notna() & (df["mid"] > MIN_MID)
    ].copy()
    print(f"  Calls with valid IV and mid: {len(df):,}")

    # ── Deduplicate SPX vs SPXW (same strike / expiry, different root) ────────
    # Keep the row with higher open interest; if tied, take 'SPX' (monthly).
    df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0)
    df = (
        df.sort_values(["expiration", "strike", "open_interest", "root"],
                       ascending=[True, True, False, True])
          .drop_duplicates(subset=["expiration", "strike"])
          .reset_index(drop=True)
    )
    print(f"  After SPX/SPXW dedup: {len(df):,} rows")

    # ── Compute moneyness ─────────────────────────────────────────────────────
    df["moneyness"] = df["strike"] / spot

    # ── Find closest expiration to each target DTE (unique per target) ────────
    expirations = sorted(df["expiration"].unique())
    print(f"\n  {len(expirations)} expirations in file")
    print(f"  Mapping target DTEs {TARGET_DTE} -> nearest available expirations:")

    # Pre-compute trading days for every expiry once
    exp_td: dict[str, int] = {}
    for exp_str in expirations:
        exp_date = datetime.datetime.strptime(exp_str, "%Y-%m-%d").date()
        td = trading_days_between(quote_date, exp_date)
        if 5 <= td <= max_dte:
            exp_td[exp_str] = td

    # Greedy assignment: sort targets by closeness to any expiry, assign best
    # available expiry one at a time so no two targets share the same expiry.
    used_exps: set[str] = set()
    selected: dict[int, tuple[str, int]] = {}
    for target in sorted(TARGET_DTE):
        best_exp, best_diff, best_dte = None, 9999, None
        for exp_str, td in exp_td.items():
            if exp_str in used_exps:
                continue
            diff = abs(td - target)
            if diff < best_diff:
                best_diff, best_exp, best_dte = diff, exp_str, td
        if best_exp is None:
            print(f"    [WARN] No unique expiry available for target {target} DTE - skipping.")
            continue
        selected[target] = (best_exp, best_dte)
        used_exps.add(best_exp)
        print(f"    Target {target:2d} DTE -> {best_exp}  ({best_dte} trading days)")

    # ── Build output rows ─────────────────────────────────────────────────────
    records = []
    for target_dte, (exp_str, actual_dte) in selected.items():
        sub = df[
            (df["expiration"]  == exp_str) &
            (df["moneyness"]   >= MONEYNESS_LO) &
            (df["moneyness"]   <= MONEYNESS_HI)
        ].copy().sort_values("strike")

        if sub.empty:
            print(f"\n  [WARN] No options in moneyness window for {exp_str} - skipping.")
            continue

        # Sub-sample evenly if more strikes than MAX_STRIKES
        if len(sub) > MAX_STRIKES:
            idx = np.round(np.linspace(0, len(sub) - 1, MAX_STRIKES)).astype(int)
            sub = sub.iloc[idx]

        print(f"\n  {exp_str} ({actual_dte} trading days):")
        print(f"    {len(sub)} strikes  [{sub['strike'].min():.0f} - {sub['strike'].max():.0f}]  "
              f"IV [{sub['iv'].min():.4f}, {sub['iv'].max():.4f}]")

        for _, row in sub.iterrows():
            records.append({
                "maturity_days": actual_dte,
                "expiry":        exp_str,
                "strike":        round(float(row["strike"]), 2),
                "mid":           round(float(row["mid"]),    4),
                "implied_vol":   round(float(row["iv"]),     6),
                "spot":          round(spot,                  2),
                "moneyness":     round(float(row["moneyness"]), 6),
            })

    if not records:
        sys.exit("[ERROR] No records produced. Check the input file and filters.")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_df = (
        pd.DataFrame(records)
          .sort_values(["maturity_days", "strike"])
          .reset_index(drop=True)
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"\nSaved {len(out_df)} rows -> {out_path}")
    print("\n-- Summary ----------------------------------------------------------")
    for dte, g in out_df.groupby("maturity_days"):
        print(
            f"  {int(dte):2d} DTE: {len(g):4d} strikes  "
            f"moneyness [{g['moneyness'].min():.3f}, {g['moneyness'].max():.3f}]  "
            f"IV [{g['implied_vol'].min():.4f}, {g['implied_vol'].max():.4f}]"
        )
    print("---------------------------------------------------------------------\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cboe", required=True,
                        help="Path to CBOE EOD CSV (UnderlyingOptionsEODCalcs_YYYY-MM-DD.csv)")
    parser.add_argument("--out",     required=True,
                        help="Output CSV path")
    parser.add_argument("--max-dte", type=int, default=51,
                        help="Exclude expirations beyond this many trading days "
                             "(default 51, matching the simulation horizon T)")
    args = parser.parse_args()

    if not os.path.exists(args.cboe):
        sys.exit(f"[ERROR] File not found: {args.cboe}")

    parse_cboe(args.cboe, args.out, max_dte=args.max_dte)


if __name__ == "__main__":
    main()
