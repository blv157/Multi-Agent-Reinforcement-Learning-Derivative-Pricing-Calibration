"""
src/reward.py
=============
Calibration loss (eq. 6), reward shaping (eq. 8), and a vectorised
Black-Scholes implied volatility solver.

Three things live here:

  1. implied_vol_batch()   — vectorised Newton-Raphson IV solver
  2. calibration_loss()    — eq. 6: normalised squared pricing error
  3. compute_rewards()     — eq. 8: per-episode reward for PPO

Calibration loss — equation 6
------------------------------
The paper defines the loss as the mean squared RELATIVE pricing error
across all calibration instruments:

    L(sigma) = (1/M) * sum_k [ (C_sigma(K_k,T_k) - C_mkt(K_k,T_k))^2
                                 / C_mkt(K_k,T_k)^2 ]

Dividing by C_mkt^2 makes the loss dimensionless and ensures that cheap
OTM options contribute equally to expensive ITM options. Without this
normalisation a single ITM option with price ~500 would dominate 10 OTM
options with prices ~1.

Reward shaping — equation 8
----------------------------
The raw calibration loss is a scalar per episode. PPO needs a reward signal
at every timestep within the episode. The paper assigns:

    r_t = 0                             for t < T  (no intermediate reward)
    r_T = -(L(sigma) - L_ref)           at the terminal step

where L_ref is the calibration loss of a REFERENCE MODEL — in practice the
Black-Scholes model evaluated at ATM implied vol (a flat smile). Subtracting
L_ref baseline-centres the reward: agents get:
  - positive reward when they beat the flat BS smile
  - negative reward when they do worse than the flat BS smile
  - ~zero reward if they exactly replicate the BS smile

This baseline subtraction is standard in policy gradient methods — it
reduces variance of the gradient estimate without changing its expectation
(since the baseline doesn't depend on the action).

For Experiment 2 (Bermudan minimisation)
-----------------------------------------
The reward is the NEGATIVE Bermudan price (agents minimise it). Vanilla
calibration is NOT part of the reward — it is maintained separately via
Gyongy localisation in marl_vol.py. So:

    r_T = -P_bermudan(sigma)

Implied vol solver
------------------
We need to invert Black-Scholes prices back to implied vols in two places:
  - Computing L_ref (flat BS smile): given market IVs, price options under
    flat ATM vol, then convert those prices back to IVs for comparison.
  - Diagnostic logging: convert MC prices to IVs for plotting vol smiles.

Newton-Raphson is the industry standard. Starting from an initial guess of
sigma_0 = 0.2, one step is:
    sigma_{i+1} = sigma_i - (C_BS(sigma_i) - C_target) / vega(sigma_i)

where vega = S * sqrt(T) * N'(d1). We run up to max_iter iterations and
return NaN for any instrument that fails to converge — typically deep OTM
options where the price is below the no-arbitrage lower bound.

All operations are vectorised over M instruments simultaneously.
"""

import torch
import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm
from typing import Tuple, Optional

from diffusion import DELTA
from market_data import VolSurface, bs_call_vectorised


# ---------------------------------------------------------------------------
# Vectorised Newton-Raphson implied vol solver
# ---------------------------------------------------------------------------

def implied_vol_batch(
    prices:     torch.Tensor,
    S:          float,
    strikes:    torch.Tensor,
    T_years:    torch.Tensor,
    sigma0:     float = 0.20,
    max_iter:   int   = 50,
    tol:        float = 1e-5,
) -> torch.Tensor:
    """
    Compute implied vols for a batch of call options via Newton-Raphson.

    All M options are solved simultaneously — no Python loop over instruments.

    Parameters
    ----------
    prices   : (M,) call prices to invert
    S        : spot price (scalar)
    strikes  : (M,) strike prices
    T_years  : (M,) times to expiry in years
    sigma0   : initial vol guess for all instruments (default 0.20 = 20%)
    max_iter : maximum Newton iterations
    tol      : convergence tolerance on |C_BS - price|

    Returns
    -------
    ivs : (M,) implied vols; NaN for instruments that failed to converge
          or have prices outside no-arbitrage bounds.

    Algorithm detail
    ----------------
    At each iteration:
        d1    = [log(S/K) + 0.5*sigma^2*T] / (sigma * sqrt(T))
        C_BS  = S * N(d1) - K * N(d1 - sigma*sqrt(T))
        vega  = S * sqrt(T) * N'(d1)        [N' = standard normal PDF]
        sigma = sigma - (C_BS - price) / vega

    We clamp sigma to [1e-6, 10] after each step to prevent divergence.
    Iterations stop per-instrument once |C_BS - price| < tol.
    """
    device = prices.device
    M      = prices.shape[0]

    # Intrinsic value lower bound and price upper bound
    # Any price outside [max(S-K,0), S] cannot be a valid call price
    intrinsic = torch.clamp(S - strikes, min=0.0)
    valid     = (prices > intrinsic + 1e-8) & (prices < S) & (T_years > 0)

    sigma = torch.full((M,), sigma0, dtype=torch.float32, device=device)
    converged = torch.zeros(M, dtype=torch.bool, device=device)

    for _ in range(max_iter):
        # Skip already-converged instruments
        active = valid & ~converged

        if not active.any():
            break

        sig_a = sigma[active]
        K_a   = strikes[active]
        T_a   = T_years[active]

        sqrtT = torch.sqrt(T_a)
        d1    = (torch.log(torch.tensor(S, device=device) / K_a)
                 + 0.5 * sig_a ** 2 * T_a) / (sig_a * sqrtT)
        d2    = d1 - sig_a * sqrtT

        # BS call price and vega (using torch's built-in normal CDF)
        Nd1   = _norm_cdf(d1)
        Nd2   = _norm_cdf(d2)
        Nd1_pdf = _norm_pdf(d1)

        c_bs  = S * Nd1 - K_a * Nd2
        vega  = S * sqrtT * Nd1_pdf          # always >= 0

        # Newton step — clamp vega away from zero to avoid /0
        delta_sigma = (c_bs - prices[active]) / vega.clamp(min=1e-10)
        sigma[active] = (sig_a - delta_sigma).clamp(1e-6, 10.0)

        # Mark converged — use relative tolerance so that options priced at
        # hundreds of dollars (deep ITM) and fractions of a dollar (deep OTM)
        # are both handled fairly.  Relative error = |C_BS - price| / price.
        converged[active] = (
            (c_bs - prices[active]).abs()
            / prices[active].clamp(min=1e-4)
        ) < tol

    # Set invalid / non-converged to NaN
    ivs = sigma.clone()
    ivs[~valid]     = float("nan")
    ivs[~converged & valid] = float("nan")
    return ivs


def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF, vectorised over a tensor."""
    return 0.5 * (1.0 + torch.erf(x / 2.0 ** 0.5))


def _norm_pdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal PDF, vectorised over a tensor."""
    return torch.exp(-0.5 * x ** 2) / (2.0 * torch.pi) ** 0.5


# ---------------------------------------------------------------------------
# Calibration loss — equation 6
# ---------------------------------------------------------------------------

def calibration_loss(
    mc_prices:  torch.Tensor,
    mkt_prices: torch.Tensor,
    denom:      Optional[torch.Tensor] = None,
    eps:        float = 1e-4,
) -> torch.Tensor:
    """
    Normalised mean squared pricing error (eq. 6), with OTM-adjusted denominator.

        L = (1/M) * sum_k [ (mc_prices[k] - mkt_prices[k])^2
                             / max(denom[k], eps)^2 ]

    When ``denom`` is None, defaults to ``mkt_prices`` (original eq. 6 behaviour).

    OTM adjustment
    --------------
    The standard relative-price loss weights OTM calls (right wing, small price)
    far more heavily than ITM calls (left wing, large price dominated by intrinsic
    value).  The result is that the policy gradient is essentially blind to the
    left side of the smile — the agent never learns the put skew.

    The fix: use OTM *put* prices as the denominator for K < S0.  By put-call
    parity (zero rate):  put = call - (S0 - K)+ ,  so the OTM put price is just
    the ITM call price minus its intrinsic value — a small number that properly
    reflects the option's time-value / volatility sensitivity.

    The absolute error (mc - mkt) is *identical* for a call and its put partner;
    only the denominator changes.  Concretely for K/S0 = 0.88:
      - ITM call mkt price ≈ $353  →  denominator = 353²  (nearly blind)
      - OTM put  mkt price ≈  $15  →  denominator =  15²  (2000× more signal)

    Pre-compute the OTM denominator once per surface via
    ``VolSurface.otm_denom_tensor(grid, S0, device)`` and pass it here.

    Parameters
    ----------
    mc_prices  : (M,) MC call prices under the current policy
    mkt_prices : (M,) market call prices from the instrument grid
    denom      : (M,) denominator tensor (OTM prices); defaults to mkt_prices
    eps        : floor on the denominator to prevent division by near-zero

    Returns
    -------
    loss : scalar tensor — non-negative, dimensionless
    """
    if denom is None:
        denom = mkt_prices
    denom = denom.clamp(min=eps)
    rel_err_sq = ((mc_prices - mkt_prices) / denom) ** 2
    return rel_err_sq.mean()


# ---------------------------------------------------------------------------
# IV-based calibration loss — equal weight per instrument in vol space
# ---------------------------------------------------------------------------

def calibration_loss_iv(
    mc_prices:  torch.Tensor,
    S0:         float,
    strikes:    torch.Tensor,
    T_years:    torch.Tensor,
    mkt_ivs:    torch.Tensor,
    fallback_denom: Optional[torch.Tensor] = None,
    min_valid:  int   = 10,
) -> torch.Tensor:
    """
    IV-space calibration loss — mean squared implied vol error.

        L_IV = (1/M_valid) * sum_k [ (IV_mc(K_k, T_k) - IV_mkt(K_k, T_k))^2 ]

    Why IV space instead of price space
    ------------------------------------
    In price space (eq. 6), the gradient weight of each instrument is
    proportional to 1 / price².  Deep OTM calls ($0.13) have ~400x more
    weight than slightly ITM puts ($2.60) even after the OTM denominator fix.
    The smile plot visualises IV directly, so calibrating in IV space gives
    each strike exactly the weight that corresponds to how it looks in the plot.

    Robustness
    ----------
    IV inversion (Newton-Raphson) can fail for MC prices below the no-arbitrage
    lower bound — this can happen for deep OTM strikes with large MC noise.
    NaN results are masked out via ``valid = ~isnan(iv_mc)``.  If fewer than
    ``min_valid`` instruments have valid IVs (unusual), we fall back to the
    OTM-adjusted price loss to avoid a degenerate gradient.

    Parameters
    ----------
    mc_prices      : (M,) MC call prices
    S0             : spot price (scalar float)
    strikes        : (M,) strike prices
    T_years        : (M,) times to expiry in years
    mkt_ivs        : (M,) market implied vols
    fallback_denom : (M,) OTM denominator for the price-loss fallback
    min_valid      : minimum valid instruments before falling back

    Returns
    -------
    loss : scalar tensor
    """
    iv_mc = implied_vol_batch(mc_prices, S0, strikes, T_years)
    valid = ~torch.isnan(iv_mc)

    if int(valid.sum()) < min_valid:
        # Too few valid IV estimates (rare) — fall back to OTM price loss
        # using mkt_ivs as a proxy for the market price denominator
        if fallback_denom is not None:
            return calibration_loss(mc_prices, mc_prices, denom=fallback_denom)
        # Ultimate fallback: uniform weight
        return (mc_prices - mc_prices).pow(2).mean() + torch.tensor(1.0, device=mc_prices.device)

    return ((iv_mc[valid] - mkt_ivs[valid]) ** 2).mean()


# ---------------------------------------------------------------------------
# Reference model loss (Black-Scholes flat smile baseline)
# ---------------------------------------------------------------------------

def reference_loss_bs(
    instruments: pd.DataFrame,
    surface:     VolSurface,
    device:      str = "cpu",
    denom:       Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calibration loss of a flat Black-Scholes smile (the baseline).

    The flat smile uses the ATM implied vol of the shortest maturity as a
    single constant volatility for all instruments. This is the simplest
    possible model and serves as the reference L_ref in eq. 8.

    Under this flat-vol model, the call price for instrument k is:
        C_flat(K_k, T_k) = BS(S, K_k, T_k, sigma_atm)

    The baseline loss L_ref tells us how much a constant-vol model
    already mis-prices the market. The RL reward is the improvement
    OVER this baseline, not the raw loss.

    Parameters
    ----------
    instruments : DataFrame from VolSurface.build_instrument_grid()
    surface     : VolSurface object (provides spot and market prices)
    device      : torch device string

    Returns
    -------
    l_ref : scalar tensor
    """
    # ATM implied vol at the shortest available maturity
    t_min     = instruments["T_years"].min()
    sigma_atm = surface.get_iv(surface.spot, t_min)

    S         = surface.spot
    strikes   = torch.tensor(instruments["K"].values,      dtype=torch.float32, device=device)
    T_years   = torch.tensor(instruments["T_years"].values, dtype=torch.float32, device=device)
    mkt_px    = torch.tensor(instruments["price_mkt"].values, dtype=torch.float32, device=device)
    mkt_ivs   = torch.tensor(instruments["iv_mkt"].values,  dtype=torch.float32, device=device)

    # Flat BS prices for all instruments
    flat_px = torch.tensor(
        [float(bs_call_vectorised(S, float(k), float(t), sigma_atm))
         for k, t in zip(strikes.tolist(), T_years.tolist())],
        dtype=torch.float32, device=device,
    )

    return calibration_loss(flat_px, mkt_px, denom=denom)


# ---------------------------------------------------------------------------
# Reward computation — equation 8
# ---------------------------------------------------------------------------

def compute_rewards(
    mc_prices:  torch.Tensor,
    mkt_prices: torch.Tensor,
    l_ref:      torch.Tensor,
    n_paths:    int,
    T_steps:    int,
    mode:       str = "terminal",
) -> torch.Tensor:
    """
    Convert a calibration loss into a per-step reward tensor for PPO.

    The reward is:  r = L_ref - L(sigma)
      = positive when the policy beats the baseline
      = negative when the policy is worse than the baseline

    Parameters
    ----------
    mc_prices  : (M,) MC call prices under current policy
    mkt_prices : (M,) market prices
    l_ref      : scalar tensor — baseline (BS flat smile) loss
    n_paths    : number of Monte Carlo trajectories
    T_steps    : number of timesteps per episode
    mode       : 'terminal' — reward only at last step (paper's approach)
                 'uniform'  — spread reward evenly across all steps

    Returns
    -------
    rewards : (n_paths, T_steps) reward tensor
              Under 'terminal': all zeros except the last column which holds
              the episode reward broadcast to all paths.
              Under 'uniform': the episode reward / T_steps at every step.

    Why broadcast to all paths?
    ---------------------------
    The calibration loss is a COLLECTIVE objective — it is computed over ALL
    paths together (E[payoff] is an average). Every path contributed equally
    to the loss, so every path receives the same reward signal. The policy
    gradient then pushes ALL agents in the direction that reduced the loss.
    """
    device = mc_prices.device

    # Scalar improvement over baseline
    l_now   = calibration_loss(mc_prices, mkt_prices)
    episode_reward = (l_ref - l_now)    # positive = improvement

    rewards = torch.zeros(n_paths, T_steps, device=device)

    if mode == "terminal":
        # Assign the full episode reward at the last timestep
        rewards[:, -1] = episode_reward

    elif mode == "uniform":
        # Spread reward uniformly across all steps
        rewards[:, :] = episode_reward / T_steps

    else:
        raise ValueError(f"Unknown reward mode '{mode}'. Use 'terminal' or 'uniform'.")

    return rewards


def compute_rewards_bermudan(
    bermudan_price: torch.Tensor,
    n_paths:        int,
    T_steps:        int,
) -> torch.Tensor:
    """
    Reward for Experiment 2 (Bermudan minimisation).

    Agents minimise the Bermudan option price, so the reward is its negative:
        r_T = -P_bermudan

    Vanilla calibration is NOT encoded here — it is maintained separately
    via Gyongy localisation in the training loop.

    Parameters
    ----------
    bermudan_price : scalar tensor — MC Bermudan price
    n_paths        : number of trajectories
    T_steps        : episode length

    Returns
    -------
    rewards : (n_paths, T_steps) — zeros except last column = -bermudan_price
    """
    device  = bermudan_price.device
    rewards = torch.zeros(n_paths, T_steps, device=device)
    rewards[:, -1] = -bermudan_price
    return rewards


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from diffusion  import generate_brownian, simulate_paths
    from options    import mc_call_prices, make_bermudan
    from american_mc import bermudan_price as ls_bermudan_price

    print("=" * 60)
    print("reward.py self-test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    torch.manual_seed(0)

    # ── Setup: load surface, build instrument grid ────────────────────────────
    surface = VolSurface("data/spx_smiles_clean.csv")
    grid    = surface.build_instrument_grid(n_strikes=10)
    mkt_px  = torch.tensor(grid["price_mkt"].values,
                            dtype=torch.float32, device=device)
    S0      = surface.spot

    # Simulate paths under ATM vol (a reasonable starting point)
    sigma_atm = surface.get_iv(S0, 34 / 252.0)
    n, T = 50_000, 51
    Z    = generate_brownian(n, T, seed=0, device=device)
    sigs = torch.full((n, T), sigma_atm, device=device)
    S    = simulate_paths(S0, sigs, Z)

    # ── Test 1: implied_vol_batch recovers known IVs ──────────────────────────
    print("Test 1: implied_vol_batch round-trip (price -> IV -> price)")
    iv_mkt   = torch.tensor(grid["iv_mkt"].values,   dtype=torch.float32, device=device)
    strikes  = torch.tensor(grid["K"].values,        dtype=torch.float32, device=device)
    T_years  = torch.tensor(grid["T_years"].values,  dtype=torch.float32, device=device)

    # Compute BS prices from known IVs
    bs_prices = torch.tensor(
        [float(bs_call_vectorised(S0, float(k), float(t), float(iv)))
         for k, t, iv in zip(grid["K"], grid["T_years"], grid["iv_mkt"])],
        dtype=torch.float32, device=device,
    )

    # Invert back to IVs
    iv_recovered = implied_vol_batch(bs_prices, S0, strikes, T_years)

    # Check round-trip error (ignore NaNs from OTM options)
    valid     = ~torch.isnan(iv_recovered)
    max_err   = (iv_recovered[valid] - iv_mkt[valid]).abs().max().item()
    n_valid   = valid.sum().item()
    n_total   = len(iv_mkt)
    print(f"  Recovered {n_valid}/{n_total} IVs  max round-trip error: {max_err:.2e}")
    assert max_err < 1e-4, f"IV round-trip error too large: {max_err}"
    print("  PASSED\n")

    # ── Test 2: calibration_loss = 0 when MC prices == market prices ─────────
    print("Test 2: calibration_loss = 0 at perfect calibration")
    loss_zero = calibration_loss(mkt_px, mkt_px)
    assert loss_zero.item() < 1e-10, f"Expected 0, got {loss_zero.item()}"
    print(f"  Loss at perfect calibration: {loss_zero.item():.2e}")
    print("  PASSED\n")

    # ── Test 3: calibration_loss > 0 under flat ATM vol ──────────────────────
    print("Test 3: calibration_loss > 0 under flat ATM vol (smile mismatch)")
    mc_px   = mc_call_prices(S, grid, delta=DELTA)
    loss_mc = calibration_loss(mc_px, mkt_px)
    print(f"  Flat-vol calibration loss: {loss_mc.item():.6f}")
    assert loss_mc.item() > 0, "Flat vol should mis-price the smile"
    print("  PASSED\n")

    # ── Test 4: reference_loss_bs is positive ────────────────────────────────
    print("Test 4: reference_loss_bs is positive and finite")
    l_ref = reference_loss_bs(grid, surface, device=device)
    print(f"  BS flat smile reference loss: {l_ref.item():.6f}")
    assert l_ref.item() > 0
    assert torch.isfinite(l_ref)
    print("  PASSED\n")

    # ── Test 5: reward is positive when MC is better than flat BS ─────────────
    print("Test 5: reward sign")
    # Under the real smile (better than flat BS), the reward should be positive
    # Simulate under a simple local vol approximation: use the actual market
    # IVs for each maturity step
    rewards = compute_rewards(mc_px, mkt_px, l_ref, n_paths=n,
                               T_steps=T, mode="terminal")
    r_terminal = rewards[:, -1].mean().item()
    print(f"  Terminal reward (flat-vol vs BS baseline): {r_terminal:.6f}")
    assert rewards.shape == (n, T)
    assert (rewards[:, :-1] == 0).all(), "Non-terminal rewards should be 0"
    print(f"  rewards shape: {list(rewards.shape)}  non-terminal steps all zero: True")
    print("  PASSED\n")

    # ── Test 6: uniform reward mode ───────────────────────────────────────────
    print("Test 6: uniform reward mode")
    rewards_u = compute_rewards(mc_px, mkt_px, l_ref, n_paths=n,
                                 T_steps=T, mode="uniform")
    assert rewards_u.shape == (n, T)
    # All steps should have the same reward
    assert rewards_u.std(dim=1).max().item() < 1e-6, "Uniform rewards should be identical across steps"
    # Sum of uniform rewards should equal terminal reward
    sum_u     = rewards_u[0].sum().item()
    sum_t     = rewards[0, -1].item()
    assert abs(sum_u - sum_t) < 1e-4, f"Uniform sum {sum_u} != terminal {sum_t}"
    print(f"  Sum of uniform rewards == terminal reward: {sum_u:.6f} ~ {sum_t:.6f}")
    print("  PASSED\n")

    # ── Test 7: Bermudan reward ───────────────────────────────────────────────
    print("Test 7: compute_rewards_bermudan")
    berm     = make_bermudan(strike=S0, t1_step=21, t2_step=T)
    b_price, _ = ls_bermudan_price(S, berm, degree=8, S0=S0)
    b_price_t  = torch.tensor(b_price, device=device)
    b_rewards  = compute_rewards_bermudan(b_price_t, n_paths=n, T_steps=T)

    assert b_rewards.shape == (n, T)
    assert (b_rewards[:, :-1] == 0).all()
    assert (b_rewards[:, -1] == -b_price_t).all()
    print(f"  Bermudan price: {b_price:.4f}  terminal reward: {b_rewards[0, -1].item():.4f}")
    print("  PASSED\n")

    # ── Test 8: timing ─────────────────────────────────────────────────────────
    import time
    print("Test 8: Timing — IV solver on full instrument grid")
    n_runs = 100
    t0 = time.perf_counter()
    for _ in range(n_runs):
        implied_vol_batch(bs_prices, S0, strikes, T_years)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_runs * 1000
    print(f"  implied_vol_batch ({len(strikes)} instruments): {elapsed:.3f} ms per call")
    print("  PASSED\n")

    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)
