"""
src/american_mc.py
==================
Longstaff-Schwartz (LS) algorithm for pricing Bermudan options via Monte Carlo.

Reference: Longstaff & Schwartz (2001), "Valuing American Options by Simulation:
A Simple Least-Squares Approach", Review of Financial Studies.

What is a Bermudan option?
--------------------------
A Bermudan option can be exercised on any date in a discrete set of exercise
dates {t_1, t_2, ..., t_E}. At each date the holder chooses:
  - Exercise now and receive the intrinsic value h(S_t), or
  - Continue and keep the right to exercise at a future date.

The holder maximises expected payoff by exercising as soon as the intrinsic
value exceeds the "continuation value" — the expected payoff from holding on.

The Longstaff-Schwartz algorithm
---------------------------------
LS estimates the continuation value via REGRESSION on the simulated paths.

Step 1 — Initialise at expiry (date t_E, the last exercise date):
  Every path must exercise or expire. Cash flow = h(S_{t_E}).

Step 2 — Backward induction: for e = E-1, E-2, ..., 1:
  (a) Identify in-the-money (ITM) paths: those where h(S_{t_e}) > 0.
      Only ITM paths are used in the regression — paths that are deep
      out-of-the-money won't exercise anyway, and including them adds
      noise to the regression without adding information.

  (b) For ITM paths, regress the cash flows (payoffs from continuing,
      which we already have from the forward induction) onto basis
      functions of the current stock price S_{t_e}. The fitted values
      are estimates of the continuation value.

      The paper specifies degree-8 polynomial regression:
        basis(x) = [1, x, x^2, ..., x^8]  where x = S_t / S0

  (c) Exercise if intrinsic value >= estimated continuation value.
      Update cash flows: paths that exercise early now contribute h(S_{t_e})
      rather than whatever future payoff they would have received.

Step 3 — Price = average of all cash flows (no discounting since r=0).

Why degree-8 polynomials?
--------------------------
Higher-degree polynomials capture more of the nonlinearity in the continuation
value surface (which is a function of S_t). Degree 8 is the paper's choice and
is also used in some standard LS implementations for equity options.

Numerical note: raw monomials x^8 can reach enormous values, causing poor
conditioning in the regression. We normalise by S0 so that x = S_t / S0
stays near 1.0, keeping x^8 O(1) for realistic price paths.

With zero interest rate
------------------------
Theoretically, early exercise of a plain call is never optimal when r=0 and
there are no dividends (American call = European call). This means the LS
price will be close to the European price for calls in Experiment 2.

In the paper, the Bermudan price is used as the *reward* for the RL agents
who choose the volatility paths — so what matters is not the absolute price
but how it changes as a function of the chosen sigma paths. The early exercise
option still creates a path-dependent objective even when r=0 because the
agents are choosing sigma, not the risk-neutral measure.

Shapes
------
  n  = number of paths
  E  = number of exercise dates  (len(bermudan.exercise_steps))
  T  = total simulation timesteps (S has shape (n, T+1))
"""

import torch
import numpy as np
from typing import Tuple, Optional

from options import BermudanSpec
from diffusion import DELTA


# ---------------------------------------------------------------------------
# Polynomial basis functions
# ---------------------------------------------------------------------------

def polynomial_features(x: torch.Tensor, degree: int = 8) -> torch.Tensor:
    """
    Build a Vandermonde-style feature matrix for least-squares regression.

    For input x of shape (n,), returns X of shape (n, degree+1) where:
        X[:, k] = x ** k,   k = 0, 1, ..., degree

    The first column (k=0) is all ones — the intercept term.

    Parameters
    ----------
    x      : (n,) normalised stock prices, typically S_t / S0
    degree : polynomial degree (paper uses 8)

    Returns
    -------
    X : (n, degree+1) feature matrix
    """
    n = x.shape[0]
    # Pre-allocate and fill column by column — avoids a Python list and
    # a single torch.stack call, which is slightly more memory-efficient
    X = torch.empty(n, degree + 1, dtype=x.dtype, device=x.device)
    X[:, 0] = 1.0          # intercept
    for k in range(1, degree + 1):
        X[:, k] = x ** k
    return X


def _lstsq(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Solve the ordinary least-squares problem  min || X @ beta - y ||^2
    and return the fitted values  X @ beta.

    We use the normal equations via torch.linalg.solve, which is numerically
    stable for well-conditioned problems and runs on GPU.

    For n >> p (many paths, few features), forming X^T X costs O(n*p^2) and
    solving it costs O(p^3). With n=120,000 and p=9 this is extremely fast.

    Parameters
    ----------
    X : (n, p) feature matrix
    y : (n,)   target values (cash flows)

    Returns
    -------
    y_hat : (n,) fitted values  X @ beta
    """
    # Normal equations: (X^T X) beta = X^T y
    XtX  = X.T @ X           # (p, p)
    Xty  = X.T @ y           # (p,)

    # Add a small ridge to handle near-singular cases (e.g. all paths at same
    # price, which makes some polynomial columns identical)
    ridge = 1e-6 * torch.eye(XtX.shape[0], dtype=X.dtype, device=X.device)
    beta  = torch.linalg.solve(XtX + ridge, Xty)   # (p,)
    return X @ beta                                  # (n,) fitted values


# ---------------------------------------------------------------------------
# Main Longstaff-Schwartz function
# ---------------------------------------------------------------------------

def longstaff_schwartz(
    S:        torch.Tensor,
    bermudan: BermudanSpec,
    degree:   int   = 8,
    S0:       Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Price a Bermudan option via the Longstaff-Schwartz algorithm.

    Parameters
    ----------
    S        : (n, T+1) simulated price paths from simulate_paths()
    bermudan : BermudanSpec — holds strike, exercise dates, option type
    degree   : polynomial degree for continuation value regression (default 8)
    S0       : normalisation constant for polynomial features.
               Defaults to S[:, 0].mean() (the average initial price).

    Returns
    -------
    price      : scalar tensor — Monte Carlo estimate of the Bermudan price
    cash_flows : (n,) tensor — individual path payoffs at optimal stopping time.
                 Averaging these gives the price; they are also useful for
                 computing the standard error of the MC estimate.
    """
    device    = S.device
    n         = S.shape[0]
    ex_steps  = bermudan.exercise_steps   # sorted list of timestep indices

    if S0 is None:
        S0 = S[:, 0].mean().item()

    # ── Step 1: Initialise cash flows at the final exercise date ─────────────
    # At expiry every path receives the intrinsic value (or 0 if OTM).
    t_last     = ex_steps[-1]
    cash_flows = bermudan.intrinsic(S[:, t_last]).clone()   # (n,)

    # ── Step 2: Backward induction over exercise dates ───────────────────────
    # We iterate from the second-to-last exercise date backwards to the first.
    for e in range(len(ex_steps) - 2, -1, -1):
        t = ex_steps[e]
        S_t = S[:, t]                          # (n,) current prices

        # (a) Immediate exercise value
        h_t = bermudan.intrinsic(S_t)          # (n,) intrinsic value

        # (b) Identify in-the-money paths
        # Only paths with positive intrinsic value are candidates for exercise.
        # Out-of-the-money paths will not exercise, so we skip them.
        itm_mask = h_t > 0.0                   # (n,) boolean mask
        n_itm    = itm_mask.sum().item()

        if n_itm == 0:
            # No ITM paths at this step — no early exercise possible
            continue

        # (c) Build polynomial features for ITM paths
        x_itm = S_t[itm_mask] / S0            # normalised prices, (n_itm,)
        X_itm = polynomial_features(x_itm, degree)   # (n_itm, degree+1)

        # (d) Continuation values for ITM paths via regression
        # We regress cash_flows (the payoff each ITM path will receive if it
        # continues) onto the polynomial features of S_t.
        y_itm  = cash_flows[itm_mask]          # (n_itm,) future cash flows
        cv_itm = _lstsq(X_itm, y_itm)         # (n_itm,) estimated continuation

        # (e) Exercise decision: exercise if intrinsic >= continuation value
        exercise_itm = h_t[itm_mask] >= cv_itm   # (n_itm,) boolean

        # (f) Update cash flows for paths that exercise early
        # Find the global indices of ITM paths that choose to exercise
        itm_indices   = itm_mask.nonzero(as_tuple=True)[0]
        ex_indices    = itm_indices[exercise_itm]
        cash_flows[ex_indices] = h_t[ex_indices]

    # ── Step 3: Price = average cash flow (zero discount rate) ───────────────
    price = cash_flows.mean()

    return price, cash_flows


# ---------------------------------------------------------------------------
# Convenience wrapper: price + standard error
# ---------------------------------------------------------------------------

def bermudan_price(
    S:        torch.Tensor,
    bermudan: BermudanSpec,
    degree:   int   = 8,
    S0:       Optional[float] = None,
) -> Tuple[float, float]:
    """
    Compute the Bermudan price and its Monte Carlo standard error.

    Returns
    -------
    (price, std_err) — both as plain Python floats
    """
    price, cf = longstaff_schwartz(S, bermudan, degree=degree, S0=S0)
    n         = cf.shape[0]
    std_err   = cf.std().item() / np.sqrt(n)
    return price.item(), std_err


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from diffusion import generate_brownian, simulate_paths, DELTA
    from options   import make_bermudan
    from market_data import bs_call_vectorised

    print("=" * 60)
    print("american_mc.py self-test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    torch.manual_seed(0)
    S0    = 100.0
    sigma = 0.20
    T     = 51
    n     = 200_000

    Z      = generate_brownian(n, T, seed=0, device=device)
    sigmas = torch.full((n, T), sigma, device=device)
    S      = simulate_paths(S0, sigmas, Z)

    # ── Test 1: polynomial_features shape and values ──────────────────────────
    print("Test 1: polynomial_features")
    x    = torch.tensor([1.0, 2.0, 3.0], device=device)
    X    = polynomial_features(x, degree=3)
    expected = torch.tensor([
        [1., 1., 1.,  1.],
        [1., 2., 4.,  8.],
        [1., 3., 9., 27.],
    ], device=device)
    assert torch.allclose(X, expected), f"Feature matrix wrong:\n{X}"
    assert X.shape == (3, 4)
    print(f"  Shape (3,4) correct, values correct")
    print("  PASSED\n")

    # ── Test 2: _lstsq recovers a known linear function ───────────────────────
    print("Test 2: _lstsq on a known quadratic")
    x_fit = torch.linspace(0.8, 1.2, 500, device=device)
    y_fit = 3.0 + 2.0 * x_fit + 1.5 * x_fit ** 2   # true quadratic
    X_fit = polynomial_features(x_fit, degree=2)
    y_hat = _lstsq(X_fit, y_fit)
    max_err = (y_hat - y_fit).abs().max().item()
    print(f"  Max fit error on quadratic: {max_err:.2e}  (should be ~0)")
    assert max_err < 1e-3, "lstsq failed to recover quadratic"
    print("  PASSED\n")

    # ── Test 3: Bermudan call price >= European call price ────────────────────
    # With r=0 and no dividends the early-exercise premium for a call is zero,
    # so the Bermudan price should equal the European price to within MC noise.
    print("Test 3: Bermudan call price vs European call (r=0, no early-exercise premium)")
    K    = 100.0
    berm = make_bermudan(strike=K, t1_step=21, t2_step=T)
    T_yr = T * DELTA

    berm_price, std_err = bermudan_price(S, berm, degree=8, S0=S0)
    euro_mc    = float(torch.clamp(S[:, -1] - K, min=0).mean().item())
    bs         = float(bs_call_vectorised(S0, K, T_yr, sigma))

    print(f"  Bermudan price : {berm_price:.4f}  (std err: {std_err:.4f})")
    print(f"  European MC    : {euro_mc:.4f}")
    print(f"  BS price       : {bs:.4f}")
    # Bermudan >= European (it has more rights), but with r=0 the gap is ~0
    assert berm_price >= euro_mc - 3 * std_err, "Bermudan < European by more than 3 sigma"
    # Gap should be small (less than 2% of BS price)
    assert abs(berm_price - bs) / bs < 0.02, "Gap between Bermudan and BS > 2%"
    print("  PASSED\n")

    # ── Test 4: ITM option (deep in the money) ────────────────────────────────
    # For a deep ITM option, early exercise IS attractive even with r=0
    # if sigma is low. Test that LS handles this without errors.
    print("Test 4: Deep ITM Bermudan (K=80, S0=100)")
    berm_itm = make_bermudan(strike=80.0, t1_step=21, t2_step=T)
    price_itm, se_itm = bermudan_price(S, berm_itm, degree=8, S0=S0)
    bs_itm    = float(bs_call_vectorised(S0, 80.0, T_yr, sigma))
    print(f"  Bermudan (K=80): {price_itm:.4f}  (std err: {se_itm:.4f})")
    print(f"  BS European    : {bs_itm:.4f}")
    assert price_itm > 0, "Deep ITM Bermudan should be positive"
    print("  PASSED\n")

    # ── Test 5: OTM option ────────────────────────────────────────────────────
    print("Test 5: Deep OTM Bermudan (K=130, S0=100)")
    berm_otm = make_bermudan(strike=130.0, t1_step=21, t2_step=T)
    price_otm, se_otm = bermudan_price(S, berm_otm, degree=8, S0=S0)
    bs_otm    = float(bs_call_vectorised(S0, 130.0, T_yr, sigma))
    print(f"  Bermudan (K=130): {price_otm:.4f}  (std err: {se_otm:.4f})")
    print(f"  BS European     : {bs_otm:.4f}")
    assert price_otm >= 0, "OTM Bermudan price should be non-negative"
    print("  PASSED\n")

    # ── Test 6: Timing at full scale ──────────────────────────────────────────
    import time
    print("Test 6: Timing at full scale (n=120,000, T=51, degree=8)")
    n_full = 120_000
    Z_f    = generate_brownian(n_full, T, seed=1, device=device)
    sig_f  = torch.full((n_full, T), sigma, device=device)
    S_f    = simulate_paths(S0, sig_f, Z_f)
    berm_f = make_bermudan(strike=S0, t1_step=21, t2_step=T)

    t0 = time.perf_counter()
    p_f, se_f = bermudan_price(S_f, berm_f, degree=8, S0=S0)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print(f"  Price: {p_f:.4f}  (std err: {se_f:.4f})")
    print(f"  Time : {elapsed*1000:.1f} ms  (n={n_full:,}, {len(berm_f.exercise_steps)} exercise dates, degree={8})")
    print("  PASSED\n")

    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)
