"""
src/diffusion.py
================
Vectorised Monte Carlo engine for GBM paths with stochastic local volatility.

The model (eq. 5 in the paper)
-------------------------------
The underlying asset follows a zero-drift diffusion:

    dS_t = S_t * sigma_t * dW_t

where sigma_t is the volatility *chosen by the RL agent* at each step.
Zero drift is consistent with the assumption of zero interest rate and
dividends used throughout (the spot equals the forward price).

Log-Euler discretisation
------------------------
Applying Ito's lemma to log(S) and discretising gives the exact scheme for
GBM (no discretisation error when sigma is constant within a step):

    S_{t+1} = S_t * exp( -0.5 * sigma_t^2 * delta + sigma_t * sqrt(delta) * Z_t )

where:
  delta = 1/252   (one trading day, matching the paper)
  Z_t ~ N(0, 1)   i.i.d. standard normals

This is sometimes called the "log-Euler" or "exact GBM" scheme.
The -0.5 * sigma^2 * delta term is the Ito correction that keeps S a
martingale (i.e. E[S_{t+1}] = S_t) under the risk-neutral measure.

Shapes and conventions
----------------------
Everything is a 2-D tensor:

    n   = number of Monte Carlo trajectories   (axis 0)
    T   = number of timesteps                  (axis 1)

    sigmas  : (n, T)    -- volatility each agent applies at each step
    Z       : (n, T)    -- pre-drawn standard normals (one per step)
    S       : (n, T+1)  -- price path including S_0 at index 0

The batch dimension B (paper uses B=10 parallel environments) is handled
by the training loop in marl_vol.py, which calls this module B times or
stacks tensors along a leading batch axis.

Why pre-draw random numbers?
----------------------------
PPO reuses the same episode data for multiple gradient update steps. If we
re-drew Z inside each update we would be evaluating the policy on different
trajectories each time, which breaks the importance-sampling correction.
Pre-drawing Z once per episode and storing it ensures all PPO mini-batch
updates see the same realisations of the Brownian motion.
"""

import torch
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DELTA = 1.0 / 252.0        # one trading day in years
SIGMA_MIN = 1e-4            # floor on volatility (prevents division by zero)
SIGMA_MAX = 5.0             # ceiling (500% vol is unphysically large)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def generate_brownian(
    n_paths: int,
    T_steps: int,
    seed:    Optional[int] = None,
    device:  str = "cpu",
) -> torch.Tensor:
    """
    Draw the matrix of standard normal increments Z ~ N(0, 1).

    Shape: (n_paths, T_steps)

    These represent the *scaled* Brownian increments — the actual Brownian
    increment over one step is sigma * sqrt(delta) * Z, but we keep Z
    separate so the same random numbers can be reused across PPO updates.

    Parameters
    ----------
    n_paths  : number of Monte Carlo trajectories
    T_steps  : number of timesteps (the path has T_steps+1 price levels)
    seed     : optional RNG seed for reproducibility
    device   : 'cpu' or 'cuda'
    """
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
    else:
        gen = None

    Z = torch.randn(n_paths, T_steps, device=device, generator=gen)
    return Z


def euler_step(
    S_t:     torch.Tensor,
    sigma_t: torch.Tensor,
    Z_t:     torch.Tensor,
    delta:   float = DELTA,
) -> torch.Tensor:
    """
    Advance all trajectories by one timestep using the log-Euler scheme.

        S_{t+1} = S_t * exp( -0.5 * sigma_t^2 * delta
                             + sigma_t * sqrt(delta) * Z_t )

    Parameters
    ----------
    S_t     : (n,) current price for each trajectory
    sigma_t : (n,) volatility chosen by each agent at this step
    Z_t     : (n,) pre-drawn standard normal for this step
    delta   : timestep size in years

    Returns
    -------
    S_{t+1} : (n,) next price — always positive (log-Euler preserves positivity)
    """
    sigma_t = sigma_t.clamp(SIGMA_MIN, SIGMA_MAX)
    log_return = -0.5 * sigma_t ** 2 * delta + sigma_t * delta ** 0.5 * Z_t
    return S_t * torch.exp(log_return)


def simulate_paths(
    S0:     float,
    sigmas: torch.Tensor,
    Z:      torch.Tensor,
    delta:  float = DELTA,
) -> torch.Tensor:
    """
    Simulate full price paths given a pre-computed volatility array.

    This function applies euler_step repeatedly but does so with a Python
    loop over T_steps. The loop is over TIME (T=51 iterations), NOT over
    trajectories — all n paths are advanced in parallel at each step.
    A 51-iteration Python loop has negligible overhead.

    Parameters
    ----------
    S0      : initial spot price (scalar)
    sigmas  : (n, T) volatility for each trajectory at each step
    Z       : (n, T) pre-drawn standard normals (from generate_brownian)
    delta   : timestep size in years

    Returns
    -------
    S : (n, T+1) full price path with S[:, 0] = S0
    """
    n, T = sigmas.shape
    device = sigmas.device

    # Pre-allocate the output tensor — more memory efficient than torch.cat
    S = torch.empty(n, T + 1, device=device, dtype=sigmas.dtype)
    S[:, 0] = S0

    for t in range(T):
        S[:, t + 1] = euler_step(S[:, t], sigmas[:, t], Z[:, t], delta)

    return S


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

class MCEngine:
    """
    Stateful wrapper around the diffusion functions.

    Holds fixed simulation parameters and pre-draws Brownian increments
    at the start of each episode (call reset_episode() once per rollout).

    Parameters
    ----------
    n_paths  : number of trajectories (paper uses 120,000)
    T_steps  : number of timesteps    (paper uses 51)
    S0       : initial spot price
    delta    : timestep in years      (1/252 by default)
    device   : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        n_paths: int   = 120_000,
        T_steps: int   = 51,
        S0:      float = 100.0,
        delta:   float = DELTA,
        device:  str   = "cpu",
    ):
        self.n_paths = n_paths
        self.T_steps = T_steps
        self.S0      = S0
        self.delta   = delta
        self.device  = device

        # Will be populated by reset_episode()
        self.Z: Optional[torch.Tensor] = None   # (n, T) Brownian increments
        self.S: Optional[torch.Tensor] = None   # (n, T+1) price paths
        self._t: int = 0                         # current timestep

    def reset_episode(self, seed: Optional[int] = None):
        """
        Draw fresh Brownian increments and initialise the path buffer.
        Call this once at the start of each training episode.
        """
        self.Z  = generate_brownian(self.n_paths, self.T_steps,
                                     seed=seed, device=self.device)
        self.S  = torch.full(
            (self.n_paths, self.T_steps + 1),
            self.S0,
            device=self.device,
            dtype=torch.float32,
        )
        self._t = 0

    def step(self, sigma_t: torch.Tensor) -> torch.Tensor:
        """
        Advance all trajectories one step using the provided volatilities.

        Parameters
        ----------
        sigma_t : (n,) volatility for each trajectory at the current step

        Returns
        -------
        S_{t+1} : (n,) updated prices; also stored in self.S[:, t+1]
        """
        if self.Z is None:
            raise RuntimeError("Call reset_episode() before step().")
        if self._t >= self.T_steps:
            raise RuntimeError(
                f"Episode already at T={self.T_steps}. Call reset_episode()."
            )

        S_next = euler_step(
            self.S[:, self._t],
            sigma_t,
            self.Z[:, self._t],
            self.delta,
        )
        self.S[:, self._t + 1] = S_next
        self._t += 1
        return S_next

    def replay(self, sigmas: torch.Tensor) -> torch.Tensor:
        """
        Re-simulate the full episode using stored Brownian increments Z but
        new volatilities. Used inside the PPO update loop.

        Parameters
        ----------
        sigmas : (n, T) updated volatilities from the new policy

        Returns
        -------
        S : (n, T+1) re-simulated price paths
        """
        if self.Z is None:
            raise RuntimeError("Call reset_episode() first.")
        return simulate_paths(self.S0, sigmas, self.Z, self.delta)

    @property
    def current_prices(self) -> torch.Tensor:
        """Current prices S_t for all trajectories, shape (n,)."""
        return self.S[:, self._t]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from market_data import bs_call_vectorised

    print("=" * 60)
    print("diffusion.py self-test")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    S0    = 100.0
    T     = 51
    delta = DELTA
    sigma_true = 0.20   # 20% flat vol

    # ── Test 1: Law of S_T under constant sigma ───────────────────────────────
    # Under log-Euler with constant sigma, S_T ~ S0 * exp(sigma*sqrt(T*delta)*Z
    #   - 0.5*sigma^2*T*delta) where Z ~ N(0,1).
    # Check: mean of S_T should be S0 (martingale), std of log(S_T/S0) should
    # be sigma * sqrt(T * delta).
    print("Test 1: Terminal distribution under constant sigma=0.20")
    n = 500_000
    Z = generate_brownian(n, T, seed=0, device=device)
    sigmas = torch.full((n, T), sigma_true, device=device)
    S = simulate_paths(S0, sigmas, Z, delta)
    S_T = S[:, -1]

    mean_ST  = S_T.mean().item()
    std_logR = S_T.log().sub(np.log(S0)).std().item()
    theory_std = sigma_true * np.sqrt(T * delta)

    print(f"  E[S_T]            = {mean_ST:.4f}  (should be {S0:.4f})")
    print(f"  Std(log S_T/S0)   = {std_logR:.4f}  (should be {theory_std:.4f})")
    assert abs(mean_ST - S0) / S0 < 0.005, "Martingale property violated"
    assert abs(std_logR - theory_std) / theory_std < 0.005, "Vol mis-match"
    print("  PASSED\n")

    # ── Test 2: ATM call price via MC vs Black-Scholes ────────────────────────
    # C_BS(S0, K=S0, T, sigma) should match the MC price
    # E[max(S_T - K, 0)] to within a few basis points (Monte Carlo error).
    print("Test 2: ATM call price — MC vs Black-Scholes")
    K    = S0
    T_yr = T * delta
    payoffs = torch.clamp(S_T - K, min=0.0)
    mc_price = payoffs.mean().item()
    bs_price = float(bs_call_vectorised(S0, K, T_yr, sigma_true))
    print(f"  MC price  = {mc_price:.4f}")
    print(f"  BS price  = {bs_price:.4f}")
    print(f"  Diff      = {abs(mc_price - bs_price):.4f}  (MC std err ~ {payoffs.std().item() / n**0.5:.4f})")
    assert abs(mc_price - bs_price) / bs_price < 0.01, "MC/BS price discrepancy > 1%"
    print("  PASSED\n")

    # ── Test 3: MCEngine step-by-step matches simulate_paths ─────────────────
    print("Test 3: MCEngine.step() matches simulate_paths()")
    n_small = 1_000
    engine  = MCEngine(n_paths=n_small, T_steps=T, S0=S0, device=device)
    engine.reset_episode(seed=7)
    Z_stored = engine.Z.clone()

    sigmas_small = torch.full((n_small, T), sigma_true, device=device)

    # Step-by-step
    for t in range(T):
        engine.step(sigmas_small[:, t])
    S_stepwise = engine.S.clone()

    # Full simulation with the same Z
    S_batch = simulate_paths(S0, sigmas_small, Z_stored, delta)

    max_diff = (S_stepwise - S_batch).abs().max().item()
    print(f"  Max absolute difference: {max_diff:.2e}  (should be ~0)")
    assert max_diff < 1e-4, "Step-by-step and batch simulation disagree"
    print("  PASSED\n")

    # ── Test 4: MCEngine.replay() ─────────────────────────────────────────────
    print("Test 4: MCEngine.replay() with same sigmas reproduces stored paths")
    engine.reset_episode(seed=7)
    for t in range(T):
        engine.step(sigmas_small[:, t])

    S_replayed = engine.replay(sigmas_small)
    max_diff = (engine.S - S_replayed).abs().max().item()
    print(f"  Max absolute difference: {max_diff:.2e}  (should be ~0)")
    assert max_diff < 1e-4, "Replay does not reproduce stored paths"
    print("  PASSED\n")

    # ── Test 5: Performance at full scale ────────────────────────────────────
    print("Test 5: Timing at full scale (n=120,000, T=51)")
    import time
    n_full = 120_000
    engine_full = MCEngine(n_paths=n_full, T_steps=T, S0=S0, device=device)
    engine_full.reset_episode(seed=0)
    sigmas_full = torch.full((n_full, T), sigma_true, device=device)

    t0 = time.perf_counter()
    S_full = simulate_paths(S0, sigmas_full, engine_full.Z, delta)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print(f"  n={n_full:,}, T={T}: {elapsed*1000:.1f} ms  "
          f"(shape: {list(S_full.shape)})")
    print(f"  Memory: ~{S_full.element_size() * S_full.nelement() / 1e6:.1f} MB for paths tensor")
    print("  PASSED\n")

    # ── Plot: sample paths ────────────────────────────────────────────────────
    t_axis = np.arange(T + 1) * delta * 252   # in trading days
    fig, ax = plt.subplots(figsize=(9, 4))
    # Plot 200 sample paths from the full simulation
    sample_idx = torch.randperm(n_full)[:200]
    for i in sample_idx:
        ax.plot(t_axis, S_full[i].cpu().numpy(), color="steelblue",
                alpha=0.08, linewidth=0.5)
    # Plot the mean path (should stay near S0)
    ax.plot(t_axis, S_full.mean(dim=0).cpu().numpy(),
            color="red", linewidth=2, label="Mean path")
    ax.axhline(S0, color="black", linestyle="--", linewidth=1, label=f"S0={S0}")
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Price")
    ax.set_title(f"GBM sample paths  (sigma={sigma_true}, n={n_full:,}, T={T})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/diffusion_paths_plot.png", dpi=130)
    print("Sample path plot saved -> data/diffusion_paths_plot.png")
    plt.close()

    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)
