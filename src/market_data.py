"""
src/market_data.py
==================
Loads the cleaned SPX implied vol surface and exposes it for use throughout
the project.

The central object is VolSurface. Everything else in the project that needs
market option prices or implied vols goes through this class — it is the
single source of truth for what we are trying to calibrate to.

Public interface
----------------
  surface = VolSurface("data/spx_smiles_clean.csv")

  surface.spot                  float   -- SPX spot at collection time
  surface.maturities_years      array   -- T values in years (e.g. [0.091, ...])
  surface.maturities_days       array   -- same in calendar days (e.g. [23, ...])

  surface.get_iv(K, T)          float   -- interpolated implied vol at (strike K, maturity T in years)
  surface.get_call_price(K, T)  float   -- Black-Scholes call price at market IV

  surface.build_instrument_grid(n_strikes_per_mat)
      Returns a DataFrame with one row per calibration instrument:
      columns: T (years), K (strike), iv_mkt, price_mkt
      This is the grid the calibration loss is computed over during training.

Design notes
------------
Interpolation strategy
  Within each maturity we fit a cubic spline in moneyness space (m = K/S).
  Cubic splines give smooth, differentiable smiles that don't wiggle between
  data points the way high-degree polynomials do. We work in moneyness rather
  than absolute strike so the surface is naturally normalised — useful if we
  ever want to reuse it after a large spot move.

  Across maturities we use linear interpolation in T (years). The total vol
  variance sigma^2 * T is smoother in T than sigma alone, so we actually
  interpolate total variance and convert back.

  If asked for a (K, T) outside the data range we extrapolate flat (i.e. use
  the boundary value). This keeps the calibration loss well-behaved for any
  trajectory the RL agent might generate.

Black-Scholes pricing
  We assume zero interest rate and zero dividends throughout, matching the
  paper's assumption (eq. 5: dS = S*sigma*dW, no drift). This means the
  forward price equals the spot, and the BS call formula simplifies.

Units
  Time is always in years using the daily convention: T_years = DTE / 252.
  This matches delta = 1/252 used in the diffusion discretisation (eq. 5).
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Black-Scholes helpers (zero rate, zero dividend)
# ---------------------------------------------------------------------------

def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """
    Black-Scholes call price with zero interest rate and dividends.

    With r = q = 0 the formula reduces to:
      C = S * N(d1) - K * N(d2)
      d1 = [log(S/K) + 0.5 * sigma^2 * T] / (sigma * sqrt(T))
      d2 = d1 - sigma * sqrt(T)

    Parameters
    ----------
    S     : spot (or forward, since r=0 means they're equal)
    K     : strike
    T     : time to expiry in years
    sigma : implied vol (annualised)

    Returns intrinsic value max(S-K, 0) when T <= 0 or sigma <= 0.
    """
    if T <= 0.0 or sigma <= 0.0:
        return max(S - K, 0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return float(S * norm.cdf(d1) - K * norm.cdf(d2))


def bs_call_vectorised(S, K, T, sigma):
    """
    Vectorised version of bs_call for numpy arrays.
    All inputs broadcast against each other in the usual numpy sense.
    Returns an array of call prices.
    """
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    price = np.where(
        (T <= 0) | (sigma <= 0),
        np.maximum(S - K, 0.0),
        _bs_core(S, K, T, sigma)
    )
    return price


def _bs_core(S, K, T, sigma):
    sqrtT = np.sqrt(np.maximum(T, 1e-10))
    d1 = (np.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * norm.cdf(d1) - K * norm.cdf(d2)


# ---------------------------------------------------------------------------
# VolSurface
# ---------------------------------------------------------------------------

class VolSurface:
    """
    Holds the market implied vol surface and answers interpolation queries.

    Parameters
    ----------
    csv_path : path to spx_smiles_clean.csv produced by data/clean_vol_surface.py
    """

    def __init__(self, csv_path: str = "data/spx_smiles_clean.csv"):
        df = pd.read_csv(csv_path)

        # ── Basic attributes ────────────────────────────────────────────────
        self.spot = float(df["spot"].iloc[0])

        # Convert DTE to years (1/252 per day, matching the paper's delta)
        self.maturities_days  = np.array(sorted(df["maturity_days"].unique()), dtype=float)
        self.maturities_years = self.maturities_days / 252.0

        # ── Build one cubic spline per maturity ─────────────────────────────
        # We fit total variance V(m) = sigma(m)^2 * T as a function of
        # moneyness m = K/S. Total variance is smoother and more amenable to
        # spline fitting than raw IV (it avoids the "vol smile term structure
        # crossing" problem where nearby maturities produce non-monotone var).
        #
        # Splines are fitted in moneyness space and evaluated by converting
        # the query strike to moneyness first.
        self._splines = {}   # DTE -> CubicSpline object (in total-var space)
        self._T_years  = {}  # DTE -> T in years (for total-var conversion)

        for dte, grp in df.groupby("maturity_days"):
            grp = grp.sort_values("moneyness")
            T   = float(dte) / 252.0
            m   = grp["moneyness"].values          # moneyness = K / S
            iv  = grp["implied_vol"].values
            tv  = iv ** 2 * T                      # total variance

            # CubicSpline with "not-a-knot" boundary (scipy default) — this
            # minimises oscillation at the edges without forcing zero curvature.
            # extrapolate=False means we clip to boundary values outside range.
            self._splines[int(dte)] = CubicSpline(m, tv, extrapolate=False)
            self._T_years[int(dte)] = T

        # Store the raw dataframe for later use (e.g. building instrument grids)
        self._df = df

    # ── Core query methods ──────────────────────────────────────────────────

    def get_iv(self, K: float, T: float) -> float:
        """
        Return the market implied vol for a call option with strike K and
        expiry T (in years).

        Interpolation steps:
          1. Convert K to moneyness m = K / spot.
          2. For each bracketing maturity, evaluate the total-variance spline
             at m to get TV_lo and TV_hi.
          3. Linearly interpolate TV in T between the two maturities.
          4. Convert total variance back to IV: sigma = sqrt(TV / T).

        If T is outside the available maturity range, we extrapolate flat
        (use the nearest boundary maturity). Same for m outside the strike range.
        """
        m = K / self.spot
        T = float(T)

        Ts = self.maturities_years    # sorted array of available T values
        Ds = self.maturities_days.astype(int)

        if T <= Ts[0]:
            # Below the shortest maturity — use shortest maturity's smile
            tv = self._eval_spline(Ds[0], m)
            return float(np.sqrt(max(tv / Ts[0], 1e-8)))

        if T >= Ts[-1]:
            # Beyond the longest maturity — use longest maturity's smile
            tv = self._eval_spline(Ds[-1], m)
            return float(np.sqrt(max(tv / Ts[-1], 1e-8)))

        # Find the two bracketing maturities
        idx = np.searchsorted(Ts, T, side="right") - 1
        T_lo, T_hi = Ts[idx], Ts[idx + 1]
        D_lo, D_hi = Ds[idx], Ds[idx + 1]

        tv_lo = self._eval_spline(D_lo, m)
        tv_hi = self._eval_spline(D_hi, m)

        # Linear interpolation of total variance in T
        w = (T - T_lo) / (T_hi - T_lo)   # weight for the upper maturity
        tv = (1 - w) * tv_lo + w * tv_hi

        return float(np.sqrt(max(tv / T, 1e-8)))

    def _eval_spline(self, dte: int, m: float) -> float:
        """
        Evaluate the total-variance spline for a given DTE at moneyness m.
        Clips to the spline's domain (i.e. flat extrapolation outside range).
        """
        spline = self._splines[dte]
        m_lo, m_hi = spline.x[0], spline.x[-1]
        m_clipped = np.clip(m, m_lo, m_hi)
        tv = float(spline(m_clipped))
        return max(tv, 1e-8)   # never negative

    def get_call_price(self, K: float, T: float) -> float:
        """
        Return the market call price for strike K, expiry T (years).
        Uses BS formula with the interpolated market IV.
        """
        iv = self.get_iv(K, T)
        return bs_call(self.spot, K, T, iv)

    # ── Instrument grid ─────────────────────────────────────────────────────

    def build_instrument_grid(self, n_strikes: int = 10) -> pd.DataFrame:
        """
        Build the discrete set of (K, T) pairs used in the calibration loss.

        For each maturity we select n_strikes strikes spread across the
        moneyness range [0.88, 1.12] (covering the main body of the smile),
        snapping each target to the nearest available strike in the CSV.

        Returns a DataFrame with columns:
          T_years   -- expiry in years
          T_days    -- expiry in calendar days
          K         -- strike
          iv_mkt    -- market implied vol (from spline)
          price_mkt -- market call price (from BS at iv_mkt)

        This DataFrame is fixed for the lifetime of the surface object and is
        the target that the calibration loss measures error against.
        """
        target_moneyness = np.linspace(0.88, 1.12, n_strikes)
        rows = []

        for dte, grp in self._df.groupby("maturity_days"):
            T = float(dte) / 252.0
            strikes_available = grp["strike"].values

            for m_target in target_moneyness:
                K_target = m_target * self.spot
                # Snap to nearest available strike in the CSV
                K = float(strikes_available[np.argmin(np.abs(strikes_available - K_target))])
                iv   = self.get_iv(K, T)
                price = bs_call(self.spot, K, T, iv)
                rows.append({
                    "T_years"  : T,
                    "T_days"   : int(dte),
                    "K"        : K,
                    "iv_mkt"   : iv,
                    "price_mkt": price,
                })

        return pd.DataFrame(rows)

    def otm_denom_tensor(
        self,
        grid:   "pd.DataFrame",
        device: str = "cpu",
    ) -> "torch.Tensor":
        """
        Compute the OTM-adjusted denominator tensor for the calibration loss.

        For each instrument in ``grid``:
          - K >= S0  (OTM call):  denom = call_price_mkt           (unchanged)
          - K <  S0  (ITM call):  denom = call_price_mkt - (S0-K)  (OTM put price)

        The put price equals the call price minus intrinsic value by put-call
        parity at zero interest rate.  Using the OTM put as the denominator on
        the left wing gives the same gradient weight as an OTM call of the same
        time-value — fixing the ~2000x gradient imbalance between the left and
        right sides of the smile.

        Parameters
        ----------
        grid   : DataFrame from build_instrument_grid()
        device : torch device string

        Returns
        -------
        denom : (M,) tensor of OTM prices to use as the loss denominator
        """
        import torch
        S0        = self.spot
        strikes   = torch.tensor(grid["K"].values,         dtype=torch.float32)
        mkt_call  = torch.tensor(grid["price_mkt"].values, dtype=torch.float32)
        intrinsic = torch.clamp(torch.tensor(S0, dtype=torch.float32) - strikes, min=0.0)
        denom     = (mkt_call - intrinsic).clamp(min=1e-4)
        return denom.to(device)

    # ── Convenience ─────────────────────────────────────────────────────────

    def summary(self):
        """Print a short human-readable description of the surface."""
        print(f"VolSurface  spot={self.spot:.2f}")
        print(f"  Maturities (days):  {list(self.maturities_days.astype(int))}")
        print(f"  Maturities (years): {[round(t, 4) for t in self.maturities_years]}")
        print(f"  Strikes per maturity (raw data):")
        for dte, grp in self._df.groupby("maturity_days"):
            print(f"    {int(dte):3d} DTE: {len(grp):3d} strikes  "
                  f"moneyness [{grp['moneyness'].min():.3f}, {grp['moneyness'].max():.3f}]  "
                  f"IV [{grp['implied_vol'].min():.3f}, {grp['implied_vol'].max():.3f}]")


# ---------------------------------------------------------------------------
# Quick self-test — run this file directly to verify everything works
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    surface = VolSurface("data/spx_smiles_clean.csv")
    surface.summary()

    # ── Test 1: interpolated smile matches raw data at known points ──────────
    print("\nTest 1: Spot-check get_iv() against raw data")
    sample = surface._df.sample(5, random_state=0)
    for _, row in sample.iterrows():
        T    = row["maturity_days"] / 252.0
        K    = row["strike"]
        iv_raw  = row["implied_vol"]
        iv_interp = surface.get_iv(K, T)
        print(f"  DTE={int(row['maturity_days']):3d}  K={K:.0f}  "
              f"iv_raw={iv_raw:.4f}  iv_interp={iv_interp:.4f}  "
              f"diff={abs(iv_interp - iv_raw):.4f}")

    # ── Test 2: interpolation between two maturities ─────────────────────────
    print("\nTest 2: Interpolated maturity (30 DTE, between 27 and 34)")
    T_mid = 30 / 252.0
    for m in [0.90, 0.95, 1.00, 1.05]:
        K = m * surface.spot
        iv = surface.get_iv(K, T_mid)
        print(f"  K/S={m:.2f}  K={K:.0f}  IV={iv:.4f}  price={surface.get_call_price(K, T_mid):.2f}")

    # ── Test 3: instrument grid ───────────────────────────────────────────────
    print("\nTest 3: Instrument grid (10 strikes x 5 maturities)")
    grid = surface.build_instrument_grid(n_strikes=10)
    print(grid.to_string(index=False))
    print(f"\n  Total instruments: {len(grid)}")

    # ── Plot: fitted smiles vs raw data ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(surface.maturities_days)))

    for (dte, grp), col in zip(surface._df.groupby("maturity_days"), colours):
        T = float(dte) / 252.0
        # Raw data points
        ax.scatter(grp["moneyness"], grp["implied_vol"] * 100,
                   s=8, color=col, alpha=0.4)
        # Fitted spline
        m_grid = np.linspace(grp["moneyness"].min(), grp["moneyness"].max(), 200)
        iv_grid = np.array([surface.get_iv(m * surface.spot, T) for m in m_grid])
        ax.plot(m_grid, iv_grid * 100, color=col, linewidth=1.5,
                label=f"{int(dte)}d")

    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Implied Vol (%)")
    ax.set_title(f"VolSurface: fitted splines vs raw data  (spot={surface.spot:.0f})")
    ax.legend(title="DTE", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/market_data_splines_plot.png", dpi=130)
    print("\nSpline fit plot saved -> data/market_data_splines_plot.png")
    plt.close()
