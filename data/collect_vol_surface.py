"""
SPX Vol Surface Collection — yfinance
======================================
Pulls the SPX implied vol surface from Yahoo Finance and saves it as
spx_smiles.csv. Works offline/on weekends (Yahoo caches option data).

Usage:
    python data/collect_vol_surface.py

Output: data/spx_smiles.csv with columns:
    maturity_days, expiry, strike, mid, implied_vol, spot, moneyness

Requirements:
    pip install yfinance pandas scipy numpy
"""

import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm

# ── Config ────────────────────────────────────────────────────────────────────
MIN_DTE = 18       # include expirations at least this far out
MAX_DTE = 60       # and at most this far out
MIN_MONEYNESS = 0.85   # K/S lower bound (strike / spot)
MAX_MONEYNESS = 1.15   # K/S upper bound
MIN_STRIKES_PER_EXPIRY = 6   # drop expiries with too few liquid strikes
# ─────────────────────────────────────────────────────────────────────────────


def bs_call_price(S, K, T, sigma, r=0.0):
    """Black-Scholes call price (zero rate / dividend for SPX fwd)."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol_newton(price, S, K, T, r=0.0, tol=1e-6, max_iter=100):
    """
    Newton-Raphson implied vol solver.
    Returns NaN if the price is outside no-arbitrage bounds or fails to converge.
    """
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-8 or price >= S:
        return np.nan
    try:
        iv = brentq(
            lambda s: bs_call_price(S, K, T, s, r) - price,
            1e-6, 10.0, xtol=tol, maxiter=max_iter
        )
        return iv
    except (ValueError, RuntimeError):
        return np.nan


def collect(output_path="data/spx_smiles.csv"):
    today = datetime.date.today()
    ticker = yf.Ticker("^SPX")

    # ── Spot price ────────────────────────────────────────────────────────────
    hist = ticker.history(period="2d")
    spot = float(hist["Close"].iloc[-1])
    print(f"SPX spot: {spot:.2f}  (as of {hist.index[-1].date()})")
    print(f"Collection date: {today}\n")

    # ── Filter expirations to DTE window ──────────────────────────────────────
    all_expiries = ticker.options   # tuple of 'YYYY-MM-DD' strings
    target_expiries = []
    for e in all_expiries:
        exp_date = datetime.datetime.strptime(e, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if MIN_DTE <= dte <= MAX_DTE:
            target_expiries.append((dte, e))

    if not target_expiries:
        print(f"No expirations found in [{MIN_DTE}, {MAX_DTE}] DTE window.")
        print(f"Available expirations: {all_expiries[:10]}")
        return

    print(f"Found {len(target_expiries)} expiration(s) in window:")
    for dte, e in target_expiries:
        print(f"  {e}  ({dte} DTE)")
    print()

    # ── Pull options chains and build surface ─────────────────────────────────
    records = []

    for dte, expiry_str in target_expiries:
        T = dte / 252.0   # time in years (daily convention matching the paper)

        chain = ticker.option_chain(expiry_str)
        calls = chain.calls.copy()

        # Filter to liquid strikes in moneyness band
        calls = calls[
            (calls["strike"] >= spot * MIN_MONEYNESS) &
            (calls["strike"] <= spot * MAX_MONEYNESS) &
            (calls["bid"] > 0.05) &              # discard zero-bid (illiquid)
            (calls["ask"] > calls["bid"])         # valid spread
        ].copy()

        if len(calls) < MIN_STRIKES_PER_EXPIRY:
            print(f"  {expiry_str}: only {len(calls)} liquid strikes — skipping")
            continue

        # Mid price
        calls["mid"] = (calls["bid"] + calls["ask"]) / 2.0

        # Prefer Yahoo's impliedVolatility; fall back to Newton-Raphson from mid
        ivs = []
        for _, row in calls.iterrows():
            yahoo_iv = row["impliedVolatility"]
            if yahoo_iv > 0.01:
                ivs.append(float(yahoo_iv))
            else:
                ivs.append(implied_vol_newton(row["mid"], spot, row["strike"], T))
        calls["implied_vol"] = ivs
        calls = calls[calls["implied_vol"].notna() & (calls["implied_vol"] > 0.01)]

        calls["maturity_days"] = dte
        calls["expiry"]        = expiry_str
        calls["spot"]          = spot
        calls["moneyness"]     = calls["strike"] / spot

        print(f"  {expiry_str} ({dte} DTE): {len(calls)} strikes  "
              f"IV range [{calls['implied_vol'].min():.3f}, {calls['implied_vol'].max():.3f}]")

        records.append(calls[["maturity_days", "expiry", "strike", "mid",
                                "implied_vol", "spot", "moneyness"]])

    if not records:
        print("\nNo usable data collected.")
        return

    surface = pd.concat(records, ignore_index=True).sort_values(
        ["maturity_days", "strike"]
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    surface.to_csv(output_path, index=False, float_format="%.6f")
    print(f"\nSaved {len(surface)} rows -> {output_path}")
    print(f"Maturities: {sorted(surface['maturity_days'].unique())} days")
    print(f"Strikes per maturity:")
    for dte, grp in surface.groupby("maturity_days"):
        print(f"  {dte:3d} DTE: {len(grp):3d} strikes  "
              f"({grp['moneyness'].min():.2f} – {grp['moneyness'].max():.2f} moneyness)")

    # ── Quick sanity plot (optional — skipped if matplotlib unavailable) ───────
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        for dte, grp in surface.groupby("maturity_days"):
            ax.plot(grp["moneyness"], grp["implied_vol"] * 100,
                    marker="o", markersize=4, label=f"{dte}d")
        ax.set_xlabel("Moneyness (K/S)")
        ax.set_ylabel("Implied Vol (%)")
        ax.set_title(f"SPX Vol Surface — {today}  (spot={spot:.0f})")
        ax.legend(title="DTE")
        ax.grid(True, alpha=0.3)
        plot_path = output_path.replace(".csv", "_plot.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=120)
        print(f"Plot saved -> {plot_path}")
        plt.close()
    except ImportError:
        pass


if __name__ == "__main__":
    collect()
