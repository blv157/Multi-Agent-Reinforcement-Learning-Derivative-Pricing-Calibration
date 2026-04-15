"""
IBKR Data Sourcing Diagnostic
==============================
Tests whether your Interactive Brokers account has the necessary
subscriptions to pull SPX options data for the AMS 517 project.

Requirements:
  - TWS or IB Gateway running and logged in
  - ib_insync installed: pip install ib_insync

TWS ports (check your TWS settings under Edit > Global Configuration > API > Settings):
  - Live account:  7496
  - Paper account: 7497

Run this script, then check the output to see what works.
"""

from ib_insync import IB, Index, Option, util
import datetime

# ── Configuration ─────────────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 7497          # Change to 7496 if using a live (non-paper) account
CLIENT_ID = 99       # Arbitrary client ID; change if 99 is in use

TARGET_DATE = "20180801"   # August 1, 2018 — the date used in the paper
# If 2018 data is unavailable we'll also try a recent date as a fallback
FALLBACK_DATE = (datetime.date.today() - datetime.timedelta(days=5)).strftime("%Y%m%d")
# ──────────────────────────────────────────────────────────────────────────────


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def ok(msg):  print(f"  [OK]   {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")


def main():
    ib = IB()

    # ── 1. Connection ──────────────────────────────────────────────────────────
    section("1. Connecting to TWS / IB Gateway")
    try:
        ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=10)
        ok(f"Connected  (host={HOST}, port={PORT}, clientId={CLIENT_ID})")
        acc = ib.managedAccounts()
        ok(f"Account(s): {acc}")
    except Exception as e:
        fail(f"Could not connect: {e}")
        print("\n  Make sure TWS or IB Gateway is running and API connections")
        print("  are enabled (Edit > Global Configuration > API > Settings).")
        return

    # ── 2. SPX Index contract ──────────────────────────────────────────────────
    section("2. Resolving SPX Index Contract")
    spx = Index("SPX", "CBOE", "USD")
    try:
        details = ib.reqContractDetails(spx)
        if details:
            ok(f"SPX resolved: {details[0].contract}")
        else:
            warn("SPX contract details returned empty — may still work")
    except Exception as e:
        fail(f"SPX contract details error: {e}")

    # ── 3. SPX Options Chain ───────────────────────────────────────────────────
    section("3. Requesting SPX Options Chain")
    try:
        chains = ib.reqSecDefOptParams("SPX", "", "IND", spx.conId if spx.conId else 416904)
        cboe_chain = next((c for c in chains if c.exchange == "CBOE"), None)
        if cboe_chain:
            expirations = sorted(cboe_chain.expirations)
            strikes = sorted(cboe_chain.strikes)
            ok(f"Chain found on CBOE")
            ok(f"  {len(expirations)} expirations available  (earliest: {expirations[0]}, latest: {expirations[-1]})")
            ok(f"  {len(strikes)} strikes available  ({min(strikes):.0f} – {max(strikes):.0f})")

            # Check whether any expiration falls in the 21–51 day window
            today = datetime.date.today()
            near_expiries = [
                e for e in expirations
                if 21 <= (datetime.datetime.strptime(e, "%Y%m%d").date() - today).days <= 60
            ]
            if near_expiries:
                ok(f"  Expirations in 21–60 day window: {near_expiries}")
            else:
                warn("  No expirations found in 21–60 day window")
        else:
            warn("No CBOE chain found in response")
            print(f"  Exchanges returned: {[c.exchange for c in chains]}")
    except Exception as e:
        fail(f"Options chain request failed: {e}")
        chains = []
        cboe_chain = None

    # ── 4. Historical Data — Target Date (2018) ────────────────────────────────
    section(f"4. Historical Data — Target Date {TARGET_DATE} (paper date)")
    spx_full = Index("SPX", "CBOE", "USD")
    ib.qualifyContracts(spx_full)

    try:
        bars = ib.reqHistoricalData(
            spx_full,
            endDateTime=f"{TARGET_DATE} 16:00:00",
            durationStr="5 D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        if bars:
            ok(f"SPX daily OHLC available around {TARGET_DATE}")
            for b in bars:
                print(f"      {b.date}  close={b.close:.2f}")
        else:
            warn(f"No bars returned for {TARGET_DATE} — historical data may not go back that far")
    except Exception as e:
        fail(f"Historical data request failed: {e}")
        print(f"  Error: {e}")

    # ── 5. Historical Option IV — Target Date ─────────────────────────────────
    section(f"5. Historical SPX Option Implied Vol — Target Date {TARGET_DATE}")
    # Pick one option contract near ATM for ~30 DTE as a probe
    # SPX was ~2816 on Aug 1, 2018; use a round ATM-ish strike
    test_expiry = "20180831"   # ~30 DTE from Aug 1 2018
    test_strike = 2800.0
    test_right = "C"

    opt = Option("SPX", test_expiry, test_strike, test_right, "CBOE")
    try:
        ib.qualifyContracts(opt)
        ok(f"Option contract qualified: {opt}")
    except Exception as e:
        warn(f"Could not qualify option {opt}: {e}")

    for what in ("BID_ASK", "OPTION_IMPLIED_VOLATILITY"):
        try:
            bars = ib.reqHistoricalData(
                opt,
                endDateTime=f"{TARGET_DATE} 16:00:00",
                durationStr="3 D",
                barSizeSetting="1 day",
                whatToShow=what,
                useRTH=True,
                formatDate=1,
            )
            if bars:
                ok(f"  whatToShow={what} returned {len(bars)} bar(s)")
                for b in bars:
                    print(f"      {b.date}  open={b.open:.4f}  close={b.close:.4f}")
            else:
                warn(f"  whatToShow={what} returned no data")
        except Exception as e:
            fail(f"  whatToShow={what} failed: {e}")

    # ── 6. Fallback — Recent Date ──────────────────────────────────────────────
    section(f"6. Fallback: Recent Option Data ({FALLBACK_DATE})")
    print("  (If 2018 data is unavailable, any clean recent date works for the project)")
    try:
        if cboe_chain:
            # Pick a nearby expiry and ATM-ish strike for the probe
            recent_expiry = next(
                (e for e in sorted(cboe_chain.expirations)
                 if (datetime.datetime.strptime(e, "%Y%m%d").date() - datetime.date.today()).days >= 25),
                None
            )
            if recent_expiry:
                mid_strike = sorted(cboe_chain.strikes)[len(cboe_chain.strikes) // 2]
                opt2 = Option("SPX", recent_expiry, mid_strike, "C", "CBOE")
                ib.qualifyContracts(opt2)
                bars2 = ib.reqHistoricalData(
                    opt2,
                    endDateTime=f"{FALLBACK_DATE} 16:00:00",
                    durationStr="3 D",
                    barSizeSetting="1 day",
                    whatToShow="OPTION_IMPLIED_VOLATILITY",
                    useRTH=True,
                    formatDate=1,
                )
                if bars2:
                    ok(f"Recent option IV data works: expiry={recent_expiry}, strike={mid_strike}")
                    for b in bars2:
                        print(f"      {b.date}  IV={b.close:.4f}")
                else:
                    warn("Recent option IV returned no data either")
            else:
                warn("Could not find a suitable nearby expiry for fallback test")
    except Exception as e:
        fail(f"Fallback test failed: {e}")

    # ── Summary ────────────────────────────────────────────────────────────────
    section("Summary")
    print("""
  Check the results above:

  - If step 5 returned OPTION_IMPLIED_VOLATILITY bars for 2018:
      You're fully set. Market data subscriptions cover historical SPX options.

  - If step 5 failed but step 6 succeeded:
      Historical data doesn't go back to 2018. You can still use the project
      with a recent date — the paper's methodology is date-agnostic. Just pick
      any clean date with a full SPX smile visible.

  - If steps 5 and 6 both failed:
      You likely need to add a market data subscription. In IBKR Client Portal:
      go to Settings > User Settings > Market Data Subscriptions and look for
      "US Equity and Options Add-On Streaming Bundle" or
      "OPRA (US Options Exchanges)".

  - If connection failed (step 1):
      Enable API in TWS: Edit > Global Configuration > API > Settings >
      check "Enable ActiveX and Socket Clients", set socket port to 7497.
    """)

    ib.disconnect()
    print("Disconnected.\n")


if __name__ == "__main__":
    util.logToConsole(level=30)  # suppress ib_insync debug noise
    main()
