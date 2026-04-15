#!/usr/bin/env python3
"""
data/fetch_spx_smile_ib.py
===========================
Two modes:

  live  — Pull a fresh SPX smile from Interactive Brokers using reqMktData
           snapshots.  Requires OPRA market data subscription in IBKR Client
           Portal (Settings -> User Settings -> Market Data Subscriptions ->
           "OPRA (US Options Exchanges)").

  wrds  — Parse an OptionMetrics / WRDS "Option Prices" CSV download into the
           project format.  This is the recommended path for historical dates
           (e.g. August 1, 2018).  Download instructions at the bottom of
           this file.

Output CSV columns (identical to spx_smiles_clean.csv):
  maturity_days, expiry, strike, mid, implied_vol, spot, moneyness

Usage examples
--------------
  # Pull live smile from IB -> overwrites the main training file:
  python data/fetch_spx_smile_ib.py --mode live --out data/spx_smiles_clean.csv

  # Parse a WRDS download for August 1, 2018:
  python data/fetch_spx_smile_ib.py --mode wrds \\
      --wrds-file ~/Downloads/om_spx_20180801.csv \\
      --date 20180801 --spot 2816.29 \\
      --out data/spx_smiles_aug2018.csv

WRDS Download Instructions
--------------------------
1. Go to https://wrds-www.wharton.upenn.edu  (use your university login)
2. Navigate: Data -> OptionMetrics -> Ivy DB US -> Option Prices
3. Set filters:
     Security Type:   Index options
     Ticker:          SPX
     Date range:      2018-08-01 to 2018-08-01   (single day)
     Option Type:     Call (or Both — script filters to calls)
4. Select output variables (at minimum):
     date, exdate, cp_flag, strike_price,
     best_bid, best_offer, impl_volatility, forward_price
5. Download as CSV and pass the path to --wrds-file
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import datetime
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

# ── IB import (only needed for --mode live) ──────────────────────────────────
try:
    from ib_insync import IB, Index, Option, util
    HAS_IB = True
except ImportError:
    HAS_IB = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Project constants
# ─────────────────────────────────────────────────────────────────────────────
TARGET_DTE     = [23, 27, 34, 41, 48]   # paper calibration maturities (trading days)
MONEYNESS_LO   = 0.85                    # minimum K/S0 to include
MONEYNESS_HI   = 1.15                    # maximum K/S0 to include
MAX_STRIKES    = 80                      # cap per maturity (sub-sampled evenly if exceeded)
BATCH_SIZE     = 45                      # simultaneous market data lines (IB limit ≈ 100)
SNAPSHOT_WAIT  = 6.0                     # seconds to wait for snapshot data
RISK_FREE_RATE = 0.0                     # use 0 for simplicity; set to T-bill rate if desired
MIN_PREMIUM    = 0.05                    # drop options with mid < $0.05 (noise floor)
MAX_SPREAD_PCT = 0.60                    # drop options where (ask-bid)/mid > 60%


# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes undiscounted call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def implied_vol_scalar(
    price: float, S: float, K: float, T: float,
    r: float = RISK_FREE_RATE,
) -> float:
    """Back out implied vol from a call mid-price using Brent's method."""
    intrinsic = max(S - K * math.exp(-r * T), 0.0)
    if price <= intrinsic + 1e-4 or T <= 0:
        return float("nan")
    lo, hi = 1e-4, 5.0
    try:
        f = lambda sig: _bs_call(S, K, T, r, sig) - price
        if f(lo) * f(hi) > 0:   # root not bracketed
            return float("nan")
        return brentq(f, lo, hi, xtol=1e-6, maxiter=200)
    except Exception:
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Calendar helpers
# ─────────────────────────────────────────────────────────────────────────────

def trading_days_between(d1: datetime.date, d2: datetime.date) -> int:
    """Count Mon–Fri trading days strictly between d1 and d2."""
    days, cur = 0, d1
    while cur < d2:
        cur += datetime.timedelta(days=1)
        if cur.weekday() < 5:
            days += 1
    return days


def parse_expiry(s: str) -> datetime.date:
    return datetime.datetime.strptime(str(s).replace("-", ""), "%Y%m%d").date()


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame) -> None:
    print("\n── Output summary ───────────────────────────────────────────────")
    for dte, g in df.groupby("maturity_days"):
        print(
            f"  {int(dte):2d} DTE: {len(g):4d} strikes  "
            f"moneyness [{g['moneyness'].min():.3f}, {g['moneyness'].max():.3f}]  "
            f"IV [{g['implied_vol'].min():.3f}, {g['implied_vol'].max():.3f}]"
        )
    print("─────────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1: Live data from Interactive Brokers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_live(out_path: str, host: str, port: int, client_id: int) -> None:
    """Pull current SPX option chain from IB and write project CSV."""
    if not HAS_IB:
        sys.exit("ib_insync not installed. Run:  pip install ib_insync")

    util.logToConsole("ERROR")   # suppress ib_insync debug noise

    ib = IB()
    print(f"\nConnecting to TWS at {host}:{port} (clientId={client_id}) ...")
    ib.connect(host, port, clientId=client_id)
    print(f"  Connected.  Account(s): {ib.managedAccounts()}")

    # ── 1. SPX spot price ─────────────────────────────────────────────────────
    spx_idx = Index("SPX", "CBOE")
    ib.qualifyContracts(spx_idx)
    spx_tick = ib.reqMktData(spx_idx, "", snapshot=True, regulatorySnapshot=False)
    ib.sleep(4)
    spot = spx_tick.marketPrice()
    if not spot or math.isnan(spot) or spot <= 0:
        spot = spx_tick.close
    if not spot or math.isnan(spot) or spot <= 0:
        ib.disconnect()
        sys.exit(
            "[ERROR] Could not retrieve SPX spot price.\n"
            "  If market is closed, try running during market hours or\n"
            "  use the last known close price via --spot flag."
        )
    print(f"  SPX spot: {spot:.2f}")

    # ── 2. Option chain metadata (expirations only) ───────────────────────────
    # We only need the expiration list here.  Strikes will be discovered per-
    # expiry via reqContractDetails so we never try a strike that doesn't exist.
    chains = ib.reqSecDefOptParams("SPX", "", "IND", spx_idx.conId)
    cboe_chain = next((c for c in chains if c.exchange == "CBOE"), None)
    if cboe_chain is None:
        ib.disconnect()
        sys.exit("[ERROR] No CBOE option chain found for SPX.")

    today = datetime.date.today()
    all_expirations = sorted(cboe_chain.expirations)
    print(f"  Chain: {len(all_expirations)} expirations available")

    # ── 3. Select closest expiration to each target DTE ───────────────────────
    print(f"\n  Mapping target DTEs {TARGET_DTE} -> nearest available expirations:")
    selected: dict[int, tuple[str, int]] = {}
    for target in TARGET_DTE:
        best_exp, best_diff, best_dte = None, 9999, None
        for exp_str in all_expirations:
            exp_date = parse_expiry(exp_str)
            td = trading_days_between(today, exp_date)
            diff = abs(td - target)
            if diff < best_diff and 10 <= td <= 80:
                best_diff, best_exp, best_dte = diff, exp_str, td
        if best_exp is None:
            print(f"  [WARN] No valid expiration found for target {target} DTE - skipping.")
            continue
        selected[target] = (best_exp, best_dte)
        print(f"    Target {target:2d} DTE -> {best_exp}  (actual {best_dte} trading days)")

    # ── 4. Request market data snapshots ─────────────────────────────────────
    records = []
    n_no_data = 0

    for target_dte, (exp_str, actual_dte) in selected.items():
        T_yr = actual_dte / 252.0

        print(f"\n  Expiry {exp_str} ({actual_dte} trading days):")

        # Discover the strikes that actually exist for this specific expiration.
        # reqContractDetails with an incomplete contract returns all matches —
        # this avoids the Error 200 flood from guessing out of the global list.
        print(f"    Querying available strikes via reqContractDetails ...")
        template = Option("SPX", exp_str, 0, "C", "CBOE")
        details  = ib.reqContractDetails(template)
        if not details:
            # Some expirations trade as SPXW — try that trading class too
            template = Option("SPX", exp_str, 0, "C", "CBOE")
            template.tradingClass = "SPXW"
            details = ib.reqContractDetails(template)
        if not details:
            print(f"    [WARN] No contracts found via reqContractDetails — skipping.")
            continue

        # Filter by moneyness window
        all_exp_contracts = [d.contract for d in details]
        valid_contracts = [
            c for c in all_exp_contracts
            if MONEYNESS_LO <= c.strike / spot <= MONEYNESS_HI
        ]
        # Sub-sample evenly if too many strikes
        if len(valid_contracts) > MAX_STRIKES:
            idx = np.round(np.linspace(0, len(valid_contracts) - 1, MAX_STRIKES)).astype(int)
            valid_contracts = [valid_contracts[i] for i in idx]

        if not valid_contracts:
            print(f"    [WARN] No contracts in moneyness window [{MONEYNESS_LO}, {MONEYNESS_HI}] — skipping.")
            continue

        strikes_found = sorted(c.strike for c in valid_contracts)
        print(f"    {len(valid_contracts)} strikes in window  "
              f"[{strikes_found[0]:.0f} - {strikes_found[-1]:.0f}]"
              f"  (of {len(all_exp_contracts)} total for this expiry)")

        # Contracts from reqContractDetails are already fully qualified
        qualified = valid_contracts

        # Batch snapshot requests
        ticker_map: list[tuple] = []
        for i in range(0, len(qualified), BATCH_SIZE):
            batch = qualified[i: i + BATCH_SIZE]
            tickers = [
                ib.reqMktData(c, "", snapshot=True, regulatorySnapshot=False)
                for c in batch
            ]
            ib.sleep(SNAPSHOT_WAIT)
            ticker_map.extend(zip(batch, tickers))
            for c in batch:
                try:
                    ib.cancelMktData(c)
                except Exception:
                    pass

        # Parse results
        n_ok = 0
        for contract, ticker in ticker_map:
            bid = ticker.bid
            ask = ticker.ask

            # Fallback: if snapshot returned no bid/ask, try 'last'
            if (not bid or math.isnan(bid) or bid <= 0 or
                    not ask or math.isnan(ask) or ask <= 0):
                # Try last traded price as a mid proxy (less reliable)
                last = ticker.last
                if last and not math.isnan(last) and last > 0:
                    bid = ask = last   # treat last as mid
                else:
                    n_no_data += 1
                    continue

            mid = (bid + ask) / 2.0
            if mid < MIN_PREMIUM:
                continue
            spread_pct = (ask - bid) / mid
            if spread_pct > MAX_SPREAD_PCT:
                continue

            iv = implied_vol_scalar(mid, spot, contract.strike, T_yr)
            if math.isnan(iv) or iv < 0.01 or iv > 3.0:
                continue

            records.append({
                "maturity_days": actual_dte,
                "expiry":        exp_str,
                "strike":        float(contract.strike),
                "mid":           round(mid, 4),
                "implied_vol":   round(iv, 6),
                "spot":          round(spot, 2),
                "moneyness":     round(contract.strike / spot, 6),
            })
            n_ok += 1

        print(f"    Clean options with IV: {n_ok}")

    ib.disconnect()
    print("\nDisconnected from TWS.")

    if n_no_data > 0:
        print(
            f"\n  [NOTE] {n_no_data} option contracts returned no bid/ask data.\n"
            f"  If this number is large, you likely need the OPRA market data\n"
            f"  subscription.  In IBKR Client Portal:\n"
            f"    Settings -> User Settings -> Market Data Subscriptions\n"
            f"    -> 'OPRA (US Options Exchanges)'"
        )

    if not records:
        sys.exit("[ERROR] No valid option data collected. Check subscription and connection.")

    # Save
    df = pd.DataFrame(records)
    df = df.sort_values(["maturity_days", "strike"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows -> {out_path}")
    _print_summary(df)


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2: Parse OptionMetrics / WRDS CSV
# ─────────────────────────────────────────────────────────────────────────────

# OptionMetrics standard column names (lower-cased) and common aliases
_OM_COL_ALIASES: dict[str, list[str]] = {
    "date":            ["date", "trading_date", "quotedate", "quote_date"],
    "exdate":          ["exdate", "expiry", "expiration", "exp_date", "expiration_date"],
    "cp_flag":         ["cp_flag", "type", "put_call", "callput", "optiontype", "option_type"],
    "strike_price":    ["strike_price", "strike", "strikeprice", "strike_px"],
    "best_bid":        ["best_bid", "bid", "bid_price"],
    "best_offer":      ["best_offer", "ask", "offer", "ask_price", "offer_price"],
    "impl_volatility": ["impl_volatility", "implied_vol", "impliedvol", "iv", "impvol"],
    "forward_price":   ["forward_price", "forward", "fwd_price", "fwd", "underprice"],
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common OptionMetrics column variants to canonical names."""
    lower = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=lower)
    rename = {}
    for canonical, aliases in _OM_COL_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and canonical not in df.columns:
                rename[alias] = canonical
                break
    return df.rename(columns=rename)


def parse_wrds(
    wrds_file: str,
    out_path: str,
    date_str: str,
    spot_override: Optional[float],
) -> None:
    """
    Parse an OptionMetrics 'Option Prices' CSV from WRDS.

    Key OptionMetrics conventions handled automatically:
      - strike_price stored as dollars x 1000  (e.g. 2800000 -> $2800)
      - impl_volatility in decimal form        (e.g. 0.1500 = 15%)
      - date / exdate as integer YYYYMMDD or 'YYYY-MM-DD' string
    """
    print(f"\nReading WRDS file: {wrds_file}")
    raw = pd.read_csv(wrds_file, low_memory=False)
    raw = _normalise_columns(raw)
    print(f"  {len(raw):,} rows  |  columns: {list(raw.columns)}")

    # ── Validate required columns ─────────────────────────────────────────────
    required = ["date", "exdate", "cp_flag", "strike_price", "best_bid", "best_offer"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        sys.exit(
            f"[ERROR] Required columns not found: {missing}\n"
            f"  Available columns: {list(raw.columns)}\n"
            f"  Check your WRDS download includes: "
            f"date, exdate, cp_flag, strike_price, best_bid, best_offer"
        )

    # ── Normalise date columns to YYYYMMDD strings ────────────────────────────
    for col in ["date", "exdate"]:
        raw[col] = (
            raw[col].astype(str)
                    .str.strip()
                    .str.replace("-", "", regex=False)
                    .str[:8]    # truncate any time component
        )

    # ── Filter to target date ─────────────────────────────────────────────────
    df = raw[raw["date"] == date_str].copy()
    if df.empty:
        avail = sorted(raw["date"].unique())[:15]
        sys.exit(
            f"[ERROR] No rows found for date={date_str}.\n"
            f"  Dates in file (first 15): {avail}"
        )
    print(f"  Rows for {date_str}: {len(df):,}")

    # ── Calls only ────────────────────────────────────────────────────────────
    df = df[df["cp_flag"].astype(str).str.strip().str.upper().str.startswith("C")].copy()
    print(f"  After call filter: {len(df):,} rows")
    if df.empty:
        sys.exit("[ERROR] No call options found after filtering. Check cp_flag values in file.")

    # ── Strike normalisation ──────────────────────────────────────────────────
    # OptionMetrics stores strike_price as dollars x 1000 (integer cents convention).
    # Detect by checking whether max strike is plausibly > 100,000.
    df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce")
    if df["strike_price"].max() > 100_000:
        df["strike_price"] /= 1000.0
        print("  Detected OptionMetrics x1000 strike convention — dividing by 1000.")

    # ── Spot price ────────────────────────────────────────────────────────────
    if spot_override is not None:
        spot = spot_override
        print(f"  Spot (from --spot flag): {spot:.2f}")
    elif "forward_price" in df.columns:
        fp = pd.to_numeric(df["forward_price"], errors="coerce").dropna()
        if len(fp) > 0:
            spot = float(fp.median())
            print(f"  Spot (from forward_price median): {spot:.2f}")
        else:
            sys.exit("[ERROR] forward_price column is empty. Pass --spot explicitly.")
    else:
        sys.exit(
            "[ERROR] Spot price not available in file.\n"
            "  Pass --spot <value>  (e.g. --spot 2816.29)"
        )

    # ── Compute DTE for each expiry ───────────────────────────────────────────
    target_date = datetime.datetime.strptime(date_str, "%Y%m%d").date()

    def _dte(exdate_str: str) -> int:
        try:
            exp_date = parse_expiry(exdate_str)
            return trading_days_between(target_date, exp_date)
        except Exception:
            return -1

    exp_dte_map: dict[str, int] = {
        e: _dte(e) for e in df["exdate"].unique()
    }

    # ── Select closest expiry to each target DTE ──────────────────────────────
    print(f"\n  Mapping target DTEs {TARGET_DTE} -> nearest available expirations:")
    selected: dict[int, tuple[str, int]] = {}
    for target in TARGET_DTE:
        best_exp, best_diff, best_dte = None, 9999, None
        for exp_str, td in exp_dte_map.items():
            diff = abs(td - target)
            if diff < best_diff and 5 <= td <= 90:
                best_diff, best_exp, best_dte = diff, exp_str, td
        if best_exp is None:
            print(f"    [WARN] No expiry found for target {target} DTE — skipping.")
            continue
        selected[target] = (best_exp, best_dte)
        print(f"    Target {target:2d} DTE -> {best_exp}  (actual {best_dte} trading days)")

    # ── Build output records ──────────────────────────────────────────────────
    records = []
    for target_dte, (exp_str, actual_dte) in selected.items():
        T_yr = actual_dte / 252.0
        sub = df[df["exdate"] == exp_str].copy()

        sub["best_bid"]   = pd.to_numeric(sub["best_bid"],   errors="coerce")
        sub["best_offer"] = pd.to_numeric(sub["best_offer"], errors="coerce")
        sub["mid"]        = (sub["best_bid"] + sub["best_offer"]) / 2.0
        sub["moneyness"]  = sub["strike_price"] / spot

        # Moneyness filter
        sub = sub[
            (sub["moneyness"] >= MONEYNESS_LO) &
            (sub["moneyness"] <= MONEYNESS_HI) &
            (sub["mid"]       >= MIN_PREMIUM)
        ].copy()

        # Implied vol: use OptionMetrics column if available, else recompute
        if "impl_volatility" in sub.columns:
            sub["impl_volatility"] = pd.to_numeric(sub["impl_volatility"], errors="coerce")
            # OptionMetrics: decimal (0.15 = 15%).  Sanity check: if median > 1, it's in percent.
            med_iv = sub["impl_volatility"].dropna().median()
            if med_iv > 1.0:
                print(f"  [NOTE] impl_volatility looks like percentage (median={med_iv:.1f})"
                      f" — dividing by 100.")
                sub["impl_volatility"] /= 100.0
            sub = sub[
                sub["impl_volatility"].notna() &
                (sub["impl_volatility"] > 0.01) &
                (sub["impl_volatility"] < 3.00)
            ]
            iv_col = "impl_volatility"
            print(f"  Expiry {exp_str} ({actual_dte}d):  {len(sub)} options  "
                  f"[using OptionMetrics IV]")
        else:
            # Recompute IV from mid prices
            sub["iv_recomputed"] = sub.apply(
                lambda r: implied_vol_scalar(r["mid"], spot, r["strike_price"], T_yr),
                axis=1,
            )
            sub = sub[
                sub["iv_recomputed"].notna() &
                (sub["iv_recomputed"] > 0.01) &
                (sub["iv_recomputed"] < 3.00)
            ]
            iv_col = "iv_recomputed"
            print(f"  Expiry {exp_str} ({actual_dte}d):  {len(sub)} options  "
                  f"[IV recomputed from mid]")

        # Sub-sample evenly if more than MAX_STRIKES
        if len(sub) > MAX_STRIKES:
            idx = np.round(np.linspace(0, len(sub) - 1, MAX_STRIKES)).astype(int)
            sub = sub.iloc[idx]

        for _, row in sub.iterrows():
            records.append({
                "maturity_days": actual_dte,
                "expiry":        exp_str,
                "strike":        round(float(row["strike_price"]), 2),
                "mid":           round(float(row["mid"]),          4),
                "implied_vol":   round(float(row[iv_col]),         6),
                "spot":          round(spot,                        2),
                "moneyness":     round(float(row["moneyness"]),    6),
            })

    if not records:
        sys.exit("[ERROR] No valid records produced. Check filters and file contents.")

    out_df = pd.DataFrame(records)
    out_df = out_df.sort_values(["maturity_days", "strike"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(out_df)} rows -> {out_path}")
    _print_summary(out_df)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode")

    # -- live sub-command
    live_p = sub.add_parser("live", help="Pull live smile from IB TWS")
    live_p.add_argument("--out",       default="data/spx_smiles_clean.csv",
                        help="Output CSV path")
    live_p.add_argument("--host",      default="127.0.0.1")
    live_p.add_argument("--port",      type=int, default=7497)
    live_p.add_argument("--client-id", type=int, default=12,
                        help="IB client ID (avoid conflicts with open TWS windows)")

    # -- wrds sub-command
    wrds_p = sub.add_parser("wrds", help="Parse OptionMetrics/WRDS CSV")
    wrds_p.add_argument("--wrds-file", required=True,
                        help="Path to downloaded OptionMetrics CSV")
    wrds_p.add_argument("--date",      required=True,
                        help="Trading date to extract, e.g. 20180801")
    wrds_p.add_argument("--spot",      type=float, default=None,
                        help="SPX spot price on that date (overrides forward_price in file)")
    wrds_p.add_argument("--out",       default="data/spx_smiles_parsed.csv",
                        help="Output CSV path")

    # Backwards-compatible flat --mode flag (original interface)
    parser.add_argument("--mode",      choices=["live", "wrds"], dest="_flat_mode")
    parser.add_argument("--out",       default=None)
    parser.add_argument("--host",      default="127.0.0.1")
    parser.add_argument("--port",      type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=12)
    parser.add_argument("--wrds-file", default=None)
    parser.add_argument("--date",      default=None)
    parser.add_argument("--spot",      type=float, default=None)

    args = parser.parse_args()

    # Determine mode (sub-command takes priority over flat --mode)
    mode = args.mode if args.mode else getattr(args, "_flat_mode", None)
    if mode is None:
        parser.print_help()
        sys.exit(1)

    out = args.out or ("data/spx_smiles_clean.csv" if mode == "live"
                       else "data/spx_smiles_parsed.csv")

    if mode == "live":
        fetch_live(out, args.host, args.port, args.client_id)
    else:
        if not args.wrds_file or not args.date:
            sys.exit("[ERROR] --wrds-file and --date are required for wrds mode.")
        parse_wrds(args.wrds_file, out, args.date, args.spot)


if __name__ == "__main__":
    main()
