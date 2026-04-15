"""
IBKR Live Snapshot Test — SPX Vol Surface
==========================================
Tests whether reqMktData snapshot works for current SPX options.
If this succeeds we can pull a full vol surface for any live trading day
and save it as the CSV the project needs.

Run with TWS/IB Gateway open (paper account OK, data may be 15-min delayed).
"""

from ib_insync import IB, Index, Option, util
import datetime, time

HOST = "127.0.0.1"
PORT = 7497
CLIENT_ID = 98   # different from the previous script in case it's still cached

# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def ok(msg):   print(f"  [OK]   {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ib = IB()
    ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=10)
    ok(f"Connected — account: {ib.managedAccounts()}")

    # ── Resolve SPX and get the chain ─────────────────────────────────────────
    section("1. Get SPX options chain")
    spx = Index("SPX", "CBOE", "USD")
    ib.qualifyContracts(spx)

    chains = ib.reqSecDefOptParams("SPX", "", "IND", spx.conId)
    chain = next((c for c in chains if c.exchange == "CBOE"), None)
    if not chain:
        fail("No CBOE chain found"); ib.disconnect(); return

    today = datetime.date.today()

    # Pick expirations in the 21–55 day window
    target_expiries = sorted([
        e for e in chain.expirations
        if 21 <= (datetime.datetime.strptime(e, "%Y%m%d").date() - today).days <= 55
    ])
    ok(f"Expirations in 21–55 day window: {target_expiries}")

    if not target_expiries:
        warn("No expirations in window — using the nearest two available")
        target_expiries = sorted(chain.expirations)[:2]

    # ── Get current SPX spot price ────────────────────────────────────────────
    section("2. Get SPX spot price")
    [spx_ticker] = ib.reqTickers(spx)
    ib.sleep(2)
    spx_ticker = ib.ticker(spx)
    spot = spx_ticker.marketPrice()
    if spot != spot or spot <= 0:   # NaN check
        # Fallback: use last close from the chain's underlying
        spot = spx_ticker.close or 5500.0
        warn(f"Live price unavailable, using fallback spot={spot:.0f}")
    else:
        ok(f"SPX spot price: {spot:.2f}")

    # ── Snapshot test: one expiry, a few strikes ──────────────────────────────
    section("3. Snapshot test — bid/ask/IV for sample options")

    test_expiry = target_expiries[0]
    dte = (datetime.datetime.strptime(test_expiry, "%Y%m%d").date() - today).days
    ok(f"Testing expiry {test_expiry} ({dte} DTE)")

    # Pick 5 strikes bracketing spot: -10%, -5%, ATM, +5%, +10%
    all_strikes = sorted(chain.strikes)
    # Build a proper grid: nearest strike to each moneyness level
    moneyness_targets = [0.90, 0.95, 1.00, 1.05, 1.10]
    test_strikes = []
    for m in moneyness_targets:
        target_k = spot * m
        nearest = min(all_strikes, key=lambda k: abs(k - target_k))
        if nearest not in test_strikes:
            test_strikes.append(nearest)
    test_strikes.sort()
    ok(f"Test strikes: {test_strikes}")

    # Build option contracts
    opts = [Option("SPX", test_expiry, k, "C", "CBOE") for k in test_strikes]
    qualified = ib.qualifyContracts(*opts)
    ok(f"Qualified {len(qualified)}/{len(opts)} contracts")

    if not qualified:
        fail("No contracts qualified — check market data subscription (OPRA)")
        ib.disconnect()
        return

    # Request snapshot tickers
    # Generic tick 106 = impliedVolatility, 13 = lastTimestamp
    tickers = [ib.reqMktData(c, genericTickList="106,107", snapshot=False) for c in qualified]
    ok("Market data requested — waiting 4 seconds for data to populate...")
    ib.sleep(4)

    # Read results
    print(f"\n  {'Strike':>8}  {'Bid':>8}  {'Ask':>8}  {'Mid':>8}  {'ModelIV':>9}  {'Status'}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*20}")

    successes = 0
    for ticker in tickers:
        strike = ticker.contract.strike
        bid    = ticker.bid
        ask    = ticker.ask
        mid    = (bid + ask) / 2 if (bid == bid and ask == ask and bid > 0) else float('nan')
        iv     = ticker.modelGreeks.impliedVol if ticker.modelGreeks else float('nan')

        if bid == bid and bid > 0:
            status = "OK"
            successes += 1
        elif bid != bid:
            status = "no data (NaN)"
        else:
            status = f"bid={bid}"

        bid_s  = f"{bid:.2f}"   if bid == bid and bid > 0  else "—"
        ask_s  = f"{ask:.2f}"   if ask == ask and ask > 0  else "—"
        mid_s  = f"{mid:.2f}"   if mid == mid and mid > 0  else "—"
        iv_s   = f"{iv:.4f}"    if iv  == iv  and iv  > 0  else "—"
        print(f"  {strike:>8.0f}  {bid_s:>8}  {ask_s:>8}  {mid_s:>8}  {iv_s:>9}  {status}")

    # Cancel subscriptions
    for ticker in tickers:
        ib.cancelMktData(ticker.contract)

    # ── Result ────────────────────────────────────────────────────────────────
    section("Result")
    if successes == len(qualified):
        ok(f"All {successes} options returned bid/ask data.")
        ok("Snapshot market data works. Ready to build the full vol surface.")
    elif successes > 0:
        warn(f"{successes}/{len(qualified)} options returned data. Partial — may be a delayed-data or subscription issue.")
        warn("Still likely usable; deep OTM options sometimes have no quotes.")
    else:
        fail("No options returned any data.")
        print("""
  Possible causes:
  1. Market is closed and delayed snapshots aren't available on paper accounts
     → Try running during NYSE/CBOE trading hours (9:30am–4:15pm ET)
  2. Missing OPRA subscription
     → In IBKR Client Portal: Settings > Market Data Subscriptions
       Add: "US Options Exchanges (OPRA)" or
            "US Equity and Options Add-On Streaming Bundle"
  3. Paper account data limitations
     → Some paper accounts only receive delayed data during market hours
        """)

    ib.disconnect()
    print("Disconnected.\n")


if __name__ == "__main__":
    util.logToConsole(level=30)
    main()
