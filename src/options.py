"""
src/options.py
==============
Payoff functions and Monte Carlo price estimators for the two option types
used in this project.

Vanilla calls (Experiment 1)
-----------------------------
A vanilla European call with strike K and expiry t pays max(S_t - K, 0)
at time t and nothing otherwise. The risk-neutral price is:

    C(K, t) = E[ max(S_t - K, 0) ]

where the expectation is over the simulated paths. With zero interest rate
the discount factor is 1.

We need to evaluate C(K, t) for every instrument in our calibration grid
(5 maturities x 10 strikes = 50 instruments) at every training step. This
must be fast — the entire grid is computed in one vectorised operation.

Bermudan calls (Experiment 2)
------------------------------
A Bermudan option can be exercised on any date in a discrete set
{t_ex_1, t_ex_2, ..., t_ex_E}. At each exercise date the holder decides:
  - exercise now and receive max(S_t - K, 0), or
  - hold on and wait for a future exercise opportunity.

The optimal exercise policy maximises the expected payoff. Computing this
requires working backwards through the exercise dates, estimating the
"continuation value" (expected payoff from holding on) at each step via
regression — the Longstaff-Schwartz algorithm, implemented in american_mc.py.

This module provides:
  - the payoff functions themselves (exercise value at each date), and
  - the MC price estimator that wraps Longstaff-Schwartz.

In Experiment 2 the RL reward is the *negative* Bermudan price — agents
minimise the Bermudan price while maintaining vanilla calibration separately
via Gyongy localisation (handled in marl_vol.py).

Shapes and conventions
-----------------------
Throughout this module:
  n       = number of Monte Carlo trajectories    (axis 0 of S)
  T+1     = number of price levels in S           (axis 1 of S)
  M       = number of calibration instruments
  t_idx   = integer timestep index (0 = S0, 1 = S_delta, ..., T = S_T)

Time in years  = t_idx * delta,   delta = 1/252.
All tensors are PyTorch float32 on whatever device S lives on.
"""

import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

from diffusion import DELTA


# ---------------------------------------------------------------------------
# Vanilla call payoffs
# ---------------------------------------------------------------------------

def call_payoff(S_t: torch.Tensor, K: float) -> torch.Tensor:
    """
    European call payoff for a single strike K.

    Parameters
    ----------
    S_t : (n,)  prices at the option's expiry for all trajectories
    K   : float strike price

    Returns
    -------
    (n,) tensor of non-negative payoffs: max(S_t - K, 0)
    """
    return torch.clamp(S_t - K, min=0.0)


def call_payoff_grid(
    S_t: torch.Tensor,
    strikes: torch.Tensor,
) -> torch.Tensor:
    """
    European call payoffs for multiple strikes at a single maturity.

    Uses broadcasting to compute all payoffs in one operation — no loop
    over strikes.

    Parameters
    ----------
    S_t     : (n,)  prices at expiry for all trajectories
    strikes : (M_t,) strike prices for this maturity

    Returns
    -------
    (n, M_t) tensor — entry [i, j] = max(S_t[i] - strikes[j], 0)
    """
    # S_t[:, None] broadcasts (n,1) against (M_t,) → (n, M_t)
    return torch.clamp(S_t[:, None] - strikes[None, :], min=0.0)


# ---------------------------------------------------------------------------
# Monte Carlo vanilla call prices for the full instrument grid
# ---------------------------------------------------------------------------

def mc_call_prices(
    S:           torch.Tensor,
    instruments: pd.DataFrame,
    delta:       float = DELTA,
) -> torch.Tensor:
    """
    Compute MC call prices for every instrument in the calibration grid.

    This is called inside the training loop at every step to evaluate how
    well the current volatility policy prices the target instruments.

    Strategy
    --------
    For each unique maturity in the instrument grid:
      1. Convert T_days to a timestep index t_idx = T_days.
         (The simulation has one step per trading day, so DTE == step index.)
      2. Extract S[:, t_idx] — prices at that maturity across all paths.
      3. Compute payoffs against all strikes for that maturity in one shot.
      4. Average over paths to get MC prices.

    Parameters
    ----------
    S           : (n, T+1) full price paths from simulate_paths()
    instruments : DataFrame with columns T_days (int) and K (float),
                  one row per calibration instrument. Produced by
                  VolSurface.build_instrument_grid().
    delta       : timestep size in years (used only for bounds checking)

    Returns
    -------
    prices : (M,) tensor of MC call prices, one per row of instruments,
             in the same row order as the input DataFrame.
    """
    device = S.device
    M      = len(instruments)
    prices = torch.empty(M, device=device)

    # Process one maturity slice at a time
    idx = 0   # position in the output tensor
    for t_days, group in instruments.groupby("T_days"):

        # The path tensor S has shape (n, T+1); S[:, t] is prices at step t.
        # t_days == number of trading days == timestep index (step 0 is S0).
        t_idx = int(t_days)
        if t_idx >= S.shape[1]:
            raise ValueError(
                f"Instrument maturity t_idx={t_idx} exceeds path length "
                f"S.shape[1]={S.shape[1]}. Increase T_steps in MCEngine."
            )

        S_t     = S[:, t_idx]                                   # (n,)
        strikes = torch.tensor(group["K"].values,
                               dtype=torch.float32, device=device)  # (M_t,)

        # (n, M_t) payoffs, then mean over paths -> (M_t,) prices
        payoffs = call_payoff_grid(S_t, strikes)                 # (n, M_t)
        mc_px   = payoffs.mean(dim=0)                            # (M_t,)

        n_instruments = len(group)
        prices[idx : idx + n_instruments] = mc_px
        idx += n_instruments

    return prices


# ---------------------------------------------------------------------------
# Bermudan option specification
# ---------------------------------------------------------------------------

@dataclass
class BermudanSpec:
    """
    Specifies a Bermudan call option.

    Attributes
    ----------
    strike          : strike price K
    exercise_steps  : list of timestep indices at which early exercise is
                      allowed, e.g. [21, 22, ..., 51] for daily exercise
                      from t1=21 to t2=T=51.
    option_type     : 'call' (only calls are used in the paper)

    The final entry in exercise_steps is the European expiry — at that
    step the option *must* be exercised (or expire worthless).
    """
    strike:         float
    exercise_steps: List[int]
    option_type:    str = "call"

    def __post_init__(self):
        if len(self.exercise_steps) < 1:
            raise ValueError("exercise_steps must have at least one entry.")
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")
        self.exercise_steps = sorted(self.exercise_steps)

    @property
    def expiry_step(self) -> int:
        """Last (European) exercise date as a timestep index."""
        return self.exercise_steps[-1]

    def intrinsic(
        self,
        S_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Immediate exercise value at prices S_t.

        Parameters
        ----------
        S_t : (n,) prices at the current exercise date

        Returns
        -------
        (n,) non-negative exercise values
        """
        if self.option_type == "call":
            return torch.clamp(S_t - self.strike, min=0.0)
        else:
            return torch.clamp(self.strike - S_t, min=0.0)


def make_bermudan(
    strike:        float,
    t1_step:       int = 21,
    t2_step:       int = 51,
    exercise_freq: int = 1,
    option_type:   str = "call",
) -> BermudanSpec:
    """
    Convenience constructor for a Bermudan option with equally-spaced
    exercise dates between t1_step and t2_step (inclusive).

    Parameters
    ----------
    strike        : strike price
    t1_step       : first exercise date (timestep index)
    t2_step       : last exercise date  (timestep index = European expiry)
    exercise_freq : steps between exercise dates (1 = daily, 5 = weekly)
    option_type   : 'call' or 'put'

    Example
    -------
    make_bermudan(strike=100, t1_step=21, t2_step=51)
    produces daily exercise from day 21 to day 51 inclusive.
    """
    steps = list(range(t1_step, t2_step + 1, exercise_freq))
    # Always include t2_step as the final (compulsory) exercise date
    if steps[-1] != t2_step:
        steps.append(t2_step)
    return BermudanSpec(strike=strike, exercise_steps=steps,
                        option_type=option_type)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from diffusion import MCEngine, simulate_paths, generate_brownian
    from market_data import VolSurface, bs_call_vectorised

    print("=" * 60)
    print("options.py self-test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    torch.manual_seed(0)
    S0    = 100.0
    sigma = 0.20
    T     = 51
    n     = 200_000

    # Simulate paths under constant sigma for ground-truth checks
    Z      = generate_brownian(n, T, seed=0, device=device)
    sigmas = torch.full((n, T), sigma, device=device)
    S      = simulate_paths(S0, sigmas, Z)   # (n, T+1)

    # ── Test 1: call_payoff matches torch.clamp directly ─────────────────────
    print("Test 1: call_payoff correctness")
    K = 100.0
    payoff_fn  = call_payoff(S[:, -1], K)
    payoff_ref = torch.clamp(S[:, -1] - K, min=0.0)
    assert torch.allclose(payoff_fn, payoff_ref), "call_payoff mismatch"
    print("  PASSED\n")

    # ── Test 2: call_payoff_grid shape and values ─────────────────────────────
    print("Test 2: call_payoff_grid shape and values")
    strikes = torch.tensor([90.0, 100.0, 110.0], device=device)
    grid    = call_payoff_grid(S[:, -1], strikes)
    assert grid.shape == (n, 3), f"Expected ({n}, 3), got {grid.shape}"
    # Higher strike -> lower average payoff (call is worth less)
    means = grid.mean(dim=0)
    assert means[0] > means[1] > means[2], "Payoff not decreasing in strike"
    print(f"  Shape: {list(grid.shape)}  mean payoffs: {means.tolist()}")
    print("  PASSED\n")

    # ── Test 3: mc_call_prices vs Black-Scholes ───────────────────────────────
    # Build a small synthetic instrument grid and check MC prices vs BS
    print("Test 3: mc_call_prices vs Black-Scholes")
    T_yr   = T * DELTA
    Ks     = [90.0, 95.0, 100.0, 105.0, 110.0]
    instr  = pd.DataFrame({
        "T_days": [T] * len(Ks),
        "K"     : Ks,
    })

    mc_px  = mc_call_prices(S, instr, delta=DELTA)
    bs_px  = torch.tensor(
        [float(bs_call_vectorised(S0, K, T_yr, sigma)) for K in Ks],
        dtype=torch.float32, device=device,
    )
    rel_err = ((mc_px - bs_px).abs() / bs_px * 100)

    print(f"  {'K':>6}  {'MC':>8}  {'BS':>8}  {'Err%':>6}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*6}")
    for K, mc, bs, err in zip(Ks, mc_px.tolist(), bs_px.tolist(), rel_err.tolist()):
        print(f"  {K:>6.0f}  {mc:>8.4f}  {bs:>8.4f}  {err:>5.2f}%")

    assert rel_err.max().item() < 1.5, "MC/BS price error > 1.5%"
    print("  PASSED\n")

    # ── Test 4: BermudanSpec construction ─────────────────────────────────────
    print("Test 4: BermudanSpec and make_bermudan")
    berm = make_bermudan(strike=100.0, t1_step=21, t2_step=51)
    assert berm.strike == 100.0
    assert berm.exercise_steps[0]  == 21
    assert berm.exercise_steps[-1] == 51
    assert len(berm.exercise_steps) == 31  # 21,22,...,51 inclusive
    print(f"  Strike: {berm.strike}")
    print(f"  Exercise steps: {berm.exercise_steps[0]} to "
          f"{berm.exercise_steps[-1]}  ({len(berm.exercise_steps)} dates)")
    print("  PASSED\n")

    # ── Test 5: BermudanSpec.intrinsic ────────────────────────────────────────
    print("Test 5: BermudanSpec.intrinsic")
    S_test = torch.tensor([85.0, 100.0, 115.0], device=device)
    iv     = berm.intrinsic(S_test)
    expected = torch.tensor([0.0, 0.0, 15.0], device=device)
    assert torch.allclose(iv, expected), f"Intrinsic mismatch: {iv}"
    print(f"  S={S_test.tolist()}  intrinsic={iv.tolist()}")
    print("  PASSED\n")

    # ── Test 6: mc_call_prices with real vol surface instruments ─────────────
    print("Test 6: mc_call_prices against real vol surface grid")
    try:
        surface = VolSurface("data/spx_smiles_clean.csv")
        S0_real = surface.spot
        sigma_atm = surface.get_iv(S0_real, 34 / 252.0)
        print(f"  Using spot={S0_real:.0f}, ATM IV (34 DTE)={sigma_atm:.4f}")

        n_real  = 100_000
        T_real  = 51
        Z_real  = generate_brownian(n_real, T_real, seed=1, device=device)
        sig_real = torch.full((n_real, T_real), sigma_atm, device=device)
        S_real   = simulate_paths(S0_real, sig_real, Z_real)

        grid_real = surface.build_instrument_grid(n_strikes=5)
        # Only keep maturities within T_real steps
        grid_real = grid_real[grid_real["T_days"] <= T_real]
        mc_real   = mc_call_prices(S_real, grid_real)
        bs_real   = torch.tensor(grid_real["price_mkt"].values,
                                 dtype=torch.float32, device=device)
        # Under constant ATM vol, prices won't match the full smile —
        # just verify shapes and that prices are positive and finite
        assert mc_real.shape[0] == len(grid_real)
        assert (mc_real >= 0).all()
        assert torch.isfinite(mc_real).all()
        print(f"  Returned {mc_real.shape[0]} MC prices — all positive and finite")
        print("  PASSED\n")
    except FileNotFoundError:
        print("  (Skipped — data/spx_smiles_clean.csv not found)\n")

    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)
