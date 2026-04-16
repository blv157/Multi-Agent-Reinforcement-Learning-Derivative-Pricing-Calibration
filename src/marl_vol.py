"""
src/marl_vol.py
===============
Algorithm 1 from Vadori (2022): the PPO training loop for MARL volatility
calibration.

This module assembles all prior components into a working training algorithm.

High-level structure
--------------------
The outer loop runs for N_episodes. Each episode consists of:

  Phase 1 — Rollout (collect experience):
    - Draw fresh Brownian increments Z
    - For each timestep t = 0, ..., T-1:
        a. Build state s_it = f(t, S_it, ...) for all n trajectories
        b. Query policy: mu_i = pi_theta(s_it) for all n
        c. Sample basis player noise: epsilon_j ~ N(0, noise_std^2) for np basis players
        d. Interpolate noise to all trajectories: noise_i = sum_j W_ij * epsilon_j
        e. Apply: sigma_it = clamp(mu_i + noise_i, sigma_min, sigma_max)
        f. Step MC engine: S_{i,t+1} = S_it * exp(-0.5*sigma^2*dt + sigma*sqrt(dt)*Z_it)
        g. Store (s_it, sigma_it, log_prob_it) for PPO

  Phase 2 — Reward:
    - Compute MC option prices over all n trajectories
    - Compute calibration loss L and episode reward r = L_ref - L
    - Build reward tensor: zeros except r at the terminal step

  Phase 3 — Advantage estimation:
    - Query value network V(s_it) for all stored states
    - Compute returns G_t = r_T for all t (terminal reward, no discounting)
    - Advantages: A_t = G_t - V(s_it)
    - Normalise advantages to zero mean, unit variance

  Phase 4 — PPO update (K=30 SGD iterations):
    - Subsample minibatch from BASIS PLAYER trajectories only
    - Compute importance sampling ratio:
        rho_it = pi_theta_new(sigma_it | s_it) / pi_theta_old(sigma_it | s_it)
    - Clipped PPO objective:
        L_clip = -E[ min(rho * A, clip(rho, 1-eps, 1+eps) * A) ]
    - Value loss: L_V = E[ (V(s) - G)^2 ]
    - Total loss: L_clip + c_V * L_V - c_ent * entropy
    - Adam step

PPO hyperparameters (Table 1 of the paper)
-------------------------------------------
  clip          = 0.3          (epsilon in the clipped objective)
  KL target     = 0.01         (early stop if KL exceeds this)
  lr            = 1e-4         (Adam learning rate)
  K             = 30           (SGD iterations per update)
  minibatch     = 0.1 * np * T * B   (fraction of basis player data)
  B             = 10           (parallel environments / rollouts per update)

Why only update on basis player trajectories?
----------------------------------------------
The calibration loss and reward are computed over ALL n=120,000 trajectories
(the aggregate behaviour matters), but the PPO policy gradient is computed
only over the BASIS PLAYER trajectories. This is because:

1. Only basis players explore — the other trajectories just inherit noise via
   interpolation. The log-probability ratio rho is well-defined only for
   trajectories whose actions were actually sampled from the policy.

2. The n/np amplification factor: the signal from np basis players is
   amplified by the n/np ratio when it enters the calibration loss. The
   gradient signal is already strong without needing to back-propagate
   through all n trajectories.

Gyongy localisation (Experiment 2 only)
-----------------------------------------
In Experiment 2, vanilla calibration is maintained via Gyongy's theorem,
which states that any Ito process has the same marginal distributions as
a Markovian diffusion with local volatility:

    sigma_loc(t, S) = sqrt( E[sigma_it^2 | S_it = S] )

We estimate this empirically at each timestep by binning trajectories by
their current price and computing the conditional variance of sigma within
each bin. The result is used to define a reference model whose smile the
basis players must match.

In this implementation Gyongy is approximated via a simple binning scheme:
at each step we create M_bins price bins, compute mean sigma^2 within each
bin, and interpolate to produce a smooth local vol function. This is then
used to re-weight the calibration loss.
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field

from diffusion      import MCEngine, simulate_paths, DELTA, generate_brownian
from market_data    import VolSurface
from options        import mc_call_prices, make_bermudan, BermudanSpec
from american_mc    import bermudan_price
from reward         import (calibration_loss, calibration_loss_iv,
                             reference_loss_bs,
                             compute_rewards, compute_rewards_bermudan)
from policy         import (PolicyNet, ValueNet,
                             build_state_exp1,
                             build_state_exp2_path_dependent,
                             build_state_exp2_nonpath,
                             SIGMA_MIN, SIGMA_MAX)
from basis_players  import BasisPlayerManager, interpolate_noise


# ---------------------------------------------------------------------------
# Running return normalizer
# ---------------------------------------------------------------------------

class ReturnNormalizer:
    """
    Online running mean / std of episode returns (Welford algorithm).

    Motivation
    ----------
    The raw calibration reward r = log(L_ref / L) is bounded but can span
    several orders of magnitude early in training (e.g. large negative values
    when the policy is miscalibrated).  If we feed raw returns directly into
    the value-function loss the MSE can be O(10^4), causing huge gradients
    and training instability.

    By maintaining a running mean/std across episodes and normalising the
    returns before they enter the value-loss, we keep the value-function MSE
    approximately O(1) throughout training — which lets the Adam optimiser
    use a consistent effective step size.

    Usage
    -----
    Call  update(scalar_return)  once per rollout (BEFORE compute_advantages).
    Then pass this object to compute_advantages; it will normalise the episode
    return to zero-mean / unit-std using the current running statistics.
    """

    def __init__(self):
        self.mean:  float = 0.0
        self.M2:    float = 0.0   # sum of squared deviations (Welford)
        self.count: int   = 0

    def update(self, x: float) -> None:
        """Update running statistics with a new return value."""
        self.count += 1
        delta       = x - self.mean
        self.mean  += delta / self.count
        delta2      = x - self.mean
        self.M2    += delta * delta2

    @property
    def std(self) -> float:
        """Current running standard deviation (clamped to avoid /0)."""
        if self.count < 2:
            return 1.0
        return max((self.M2 / (self.count - 1)) ** 0.5, 1e-8)

    def normalise(self, x: float) -> float:
        """Return (x - mean) / std using current running statistics."""
        return (x - self.mean) / self.std

    def normalise_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise a tensor of returns using current running statistics."""
        return (x - self.mean) / self.std


# ---------------------------------------------------------------------------
# Training configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """All hyperparameters for a training run."""

    # --- Simulation ---
    n_paths:    int   = 120_000   # total Monte Carlo trajectories
    T_steps:    int   = 51        # timesteps per episode
    S0:         float = None      # initial spot (set from VolSurface)
    delta:      float = DELTA     # timestep in years

    # --- Basis players ---
    n_basis:    int   = 100       # number of basis players np
    bp_method:  str   = "knn"     # 'knn' or 'linear'
    bp_k:       int   = 1         # neighbourhood size
    noise_std:  float = 0.02      # basis player exploration noise std

    # --- PPO ---
    lr:         float = 1e-4      # Adam learning rate
    clip:       float = 0.3       # PPO clip epsilon
    kl_target:  float = 0.01      # early-stop KL threshold
    K_epochs:   int   = 30        # SGD iterations per update
    B_envs:     int   = 10        # parallel rollouts per update
    mb_frac:    float = 0.1       # minibatch = mb_frac * np * T * B
    c_value:    float = 0.5       # value loss coefficient
    c_entropy:  float = 0.01      # entropy bonus coefficient
    gamma:      float = 1.0       # discount factor (1.0 = no discounting)

    # --- Experiment ---
    experiment: str   = "exp1"    # 'exp1' or 'exp2'
    state_dim:  int   = 2         # 2 for exp1, 3 or 5 for exp2
    path_dep:   bool  = False     # path-dependent state for exp2

    # --- Training ---
    n_episodes:   int   = 2000    # maximum episodes (training stops earlier if converged)
    n_strikes:    int   = 10      # strikes per maturity in instrument grid
    log_every:    int   = 10      # print progress every N episodes
    save_every:   int   = 100     # checkpoint every N episodes
    save_dir:     str   = "results"

    # --- Convergence ---
    # Patience-based early stopping: track the best rolling-window mean reward
    # seen so far.  Each episode we check whether the latest window mean beats
    # the best by more than conv_tol.  If not, increment a patience counter.
    # Stop when the counter reaches conv_patience windows.
    # Set conv_window=0 to disable entirely and run all n_episodes.
    conv_window:   int   = 100     # rolling window size for convergence check
    conv_tol:      float = 1e-3    # minimum improvement over best window mean
    conv_patience: int   = 5       # stop after this many non-improving windows

    # --- Device ---
    device:     str   = "cpu"     # 'cpu' or 'cuda'

    def minibatch_size(self) -> int:
        """Minibatch size = 0.1 * np * T * B (paper Table 1)."""
        return max(1, int(self.mb_frac * self.n_basis * self.T_steps * self.B_envs))


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class RolloutBuffer:
    """
    Stores experience collected during B_envs rollouts for PPO update.

    Only stores data for the n_basis basis player trajectories — the
    other trajectories are used only for the calibration loss computation.

    Fields (all tensors):
      states    : (B, np, T, d)  states at each step
      sigmas    : (B, np, T)     volatilities applied
      log_probs : (B, np, T)     log pi_old(sigma | state)
      returns   : (B, np, T)     discounted returns G_t
      advantages: (B, np, T)     normalised advantages A_t
    """
    states:     Optional[torch.Tensor] = None
    sigmas:     Optional[torch.Tensor] = None
    log_probs:  Optional[torch.Tensor] = None
    returns:    Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None

    def flatten(self) -> Dict[str, torch.Tensor]:
        """
        Flatten the (B, np, T, ...) buffers to (B*np*T, ...) for minibatch sampling.
        """
        return {
            "states":     self.states.flatten(0, 2),     # (B*np*T, d)
            "sigmas":     self.sigmas.flatten(0, 2),     # (B*np*T,)
            "log_probs":  self.log_probs.flatten(0, 2),  # (B*np*T,)
            "returns":    self.returns.flatten(0, 2),    # (B*np*T,)
            "advantages": self.advantages.flatten(0, 2), # (B*np*T,)
        }


# ---------------------------------------------------------------------------
# Single rollout collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_rollout(
    engine:        MCEngine,
    policy:        PolicyNet,
    value:         ValueNet,
    manager:       BasisPlayerManager,
    surface:       VolSurface,
    grid:          pd.DataFrame,
    cfg:           TrainConfig,
    bermudan:      Optional[BermudanSpec] = None,
    l_ref:         Optional[torch.Tensor] = None,
    mkt_otm:       Optional[torch.Tensor] = None,
    inst_strikes:  Optional[torch.Tensor] = None,
    inst_T_years:  Optional[torch.Tensor] = None,
    inst_mkt_ivs:  Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, float, float
]:
    """
    Run one episode and collect trajectory data.

    Returns
    -------
    states_bp   : (np, T, d)  states at each step for basis players only
    sigmas_bp   : (np, T)     volatilities for basis players
    log_probs_bp: (np, T)     log-probs for basis players
    rewards_full: (n, T)      reward tensor for all trajectories
    loss        : float       calibration loss this episode
    episode_r   : float       episode reward (scalar)
    """
    device  = cfg.device
    n       = cfg.n_paths
    T       = cfg.T_steps
    np_     = cfg.n_basis
    d       = cfg.state_dim
    S0      = cfg.S0

    # Allocate storage (basis players only)
    states_bp    = torch.zeros(np_, T, d,  device=device)
    sigmas_bp    = torch.zeros(np_, T,     device=device)
    log_probs_bp = torch.zeros(np_, T,     device=device)

    # Reset episode — draw new Brownian increments
    engine.reset_episode()

    # Path-dependent state tracking (Exp 2 only)
    t1_step    = 21   # first Bermudan exercise date
    sigma_prev = torch.full((n,), 0.20, device=device)  # initial sigma guess
    S_at_t1    = torch.full((n,), S0,   device=device)  # price frozen at t1
    sig_at_t1  = torch.full((n,), 0.20, device=device)  # sigma frozen at t1

    # Sample basis player noise ONCE per episode (not per timestep).
    # The paper's eq. 15 writes epsilon_j with no time subscript — each basis
    # player maintains a fixed noise offset throughout the episode.  This means
    # each player consistently explores a slightly shifted sigma function,
    # creating a clean causal link between its noise and the calibration loss
    # it produces.  Per-timestep noise would average out over T steps, making
    # it nearly impossible to attribute outcome differences to specific players.
    epsilon = manager.sample_noise()                     # (np,)  fixed for episode

    for t in range(T):
        S_t = engine.current_prices                      # (n,)

        # ── Build state ─────────────────────────────────────────────────────
        if cfg.experiment == "exp1":
            state = build_state_exp1(t, S_t, T, S0)     # (n, 2)
        elif cfg.path_dep:
            state = build_state_exp2_path_dependent(
                t, S_t, sigma_prev, S_at_t1, sig_at_t1, T, S0)
        else:
            state = build_state_exp2_nonpath(
                t, S_t, sigma_prev, T, S0)

        # Update normaliser (online, no grad)
        policy.norm.update(state)
        value.norm.update(state)

        # ── Policy mean for all trajectories ────────────────────────────────
        mu, _ = policy(state)                            # (n,)

        # ── Basis player exploration noise ──────────────────────────────────
        # epsilon is fixed for this episode (sampled above the loop)
        W       = manager.compute_weights(state)         # (n, np)
        noise   = interpolate_noise(W, epsilon)          # (n,)

        # ── Volatility actions ──────────────────────────────────────────────
        sigma_t = (mu + noise).clamp(SIGMA_MIN, SIGMA_MAX)  # (n,)

        # ── Log-prob for basis players only ─────────────────────────────────
        bp_idx   = manager.basis_idx                     # (np,)
        state_bp = state[bp_idx]                         # (np, d)
        sigma_bp = sigma_t[bp_idx]                       # (np,)
        lp_bp    = policy.log_prob(state_bp, sigma_bp)   # (np,)

        # ── Store basis player data ──────────────────────────────────────────
        states_bp[:, t, :]  = state_bp
        sigmas_bp[:, t]     = sigma_bp
        log_probs_bp[:, t]  = lp_bp

        # ── Advance simulation ───────────────────────────────────────────────
        engine.step(sigma_t)

        # ── Update path-dependent features ──────────────────────────────────
        sigma_prev = sigma_t.clone()
        if cfg.experiment == "exp2":
            if t + 1 == t1_step:
                S_at_t1  = engine.current_prices.clone()
                sig_at_t1 = sigma_t.clone()

    # ── Compute rewards ──────────────────────────────────────────────────────
    S_full   = engine.S                                  # (n, T+1)
    mkt_px   = torch.tensor(grid["price_mkt"].values,
                             dtype=torch.float32, device=device)

    if cfg.experiment == "exp1":
        mc_px = mc_call_prices(S_full, grid, delta=cfg.delta)

        # OTM-adjusted relative price loss.
        # For K < S0 the denominator is the OTM put price (= call - intrinsic)
        # rather than the ITM call price, giving equal gradient weight across
        # both wings of the smile.  The IV tensors are retained for diagnostic
        # use (smile plots) but are not used in the training loss.
        loss_val = calibration_loss(mc_px, mkt_px, denom=mkt_otm).item()

        # ── Reward: r = L_ref - L  (paper eq. 8) ────────────────────────────────
        # Positive when the policy beats the flat BS baseline, negative when worse.
        if l_ref is None:
            raw_r = torch.tensor(0.0, device=device)
        else:
            l_now = torch.tensor(loss_val, device=device)
            raw_r = l_ref - l_now                            # scalar

        rewards   = torch.zeros(n, T, device=device)
        rewards[:, -1] = raw_r                               # terminal reward
        episode_r = raw_r.item()

    else:  # exp2
        assert bermudan is not None, "BermudanSpec required for exp2"
        b_price, _ = bermudan_price(S_full, bermudan, degree=8, S0=S0)
        b_price_t  = torch.tensor(b_price, device=device)
        loss_val   = b_price
        rewards    = compute_rewards_bermudan(b_price_t, n_paths=n, T_steps=T)
        episode_r  = rewards[0, -1].item()

    return states_bp, sigmas_bp, log_probs_bp, rewards, loss_val, episode_r


# ---------------------------------------------------------------------------
# Advantage estimation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_advantages(
    states_bp:      torch.Tensor,
    rewards:        torch.Tensor,
    value:          ValueNet,
    cfg:            TrainConfig,
    bp_idx:         torch.Tensor,
    ret_normalizer: Optional["ReturnNormalizer"] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute returns and advantages for basis player trajectories.

    With a terminal reward only (r_t = 0 for t < T, r_T = episode_reward),
    the undiscounted return is the same for every timestep:
        G_t = r_T   for all t

    The advantage is:
        A_t = G_t - V(s_t)

    We normalise advantages across the batch for stable PPO updates.

    Parameters
    ----------
    states_bp      : (np, T, d)   states for basis players
    rewards        : (n,  T)      reward tensor (full n paths)
    value          : ValueNet
    cfg            : TrainConfig
    bp_idx         : (np,) basis player indices
    ret_normalizer : ReturnNormalizer (optional)
        If provided, the episode return is normalised to approximately zero
        mean / unit std before being stored as the return target.  This keeps
        the value-function MSE ~ O(1) regardless of the reward scale, which
        prevents gradient explosion in the value network.

    Returns
    -------
    returns    : (np, T)  G_t for each basis player step (possibly normalised)
    advantages : (np, T)  normalised A_t
    """
    np_, T, d = states_bp.shape
    device    = states_bp.device

    # Terminal reward (same for all paths, take from first)
    episode_reward = rewards[0, -1]                   # scalar

    # ── Optionally normalise the return target ───────────────────────────────
    # The value function learns to predict NORMALISED returns.  We apply the
    # same normalisation at inference (advantage = normalised_G - V), so the
    # advantage scale stays consistent even when the raw reward drifts.
    if ret_normalizer is not None and ret_normalizer.count >= 2:
        ep_r_val = ret_normalizer.normalise(episode_reward.item())
        ep_r_t   = torch.tensor(ep_r_val, device=device, dtype=torch.float32)
    else:
        ep_r_t   = episode_reward                     # use raw until 2 samples

    # Return is ep_r_t at every timestep (undiscounted terminal reward)
    returns = ep_r_t.expand(np_, T).clone()           # (np, T)

    # Value estimates for all basis player states
    states_flat = states_bp.reshape(np_ * T, d)       # (np*T, d)
    V_flat      = value(states_flat)                   # (np*T,)
    V           = V_flat.reshape(np_, T)               # (np, T)

    advantages  = returns - V                         # (np, T)

    # Do NOT normalise here.
    #
    # Why: all np*T samples in one rollout share the SAME G (terminal reward).
    # If we normalise within a single rollout, G cancels out:
    #
    #   A_norm = (G - V(s) - mean(G-V)) / std(G-V)
    #          = (V_mean - V(s)) / std(V)      <- G has vanished!
    #
    # The policy gradient would then carry zero information about whether the
    # episode was good or bad — only the value function's within-rollout spread
    # remains, creating spurious signal that drives sigma in random directions.
    #
    # Instead, normalisation is done ONCE in train() across all B rollouts
    # stacked together.  That preserves the between-rollout G variation (which
    # is the real learning signal) while still stabilising gradient magnitude.

    return returns, advantages


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    buffer:  RolloutBuffer,
    policy:  PolicyNet,
    value:   ValueNet,
    opt_p:   torch.optim.Optimizer,
    opt_v:   torch.optim.Optimizer,
    cfg:     TrainConfig,
) -> Dict[str, float]:
    """
    Run K_epochs of PPO updates on the collected rollout buffer.

    Returns a dict of training statistics (losses, KL, etc.).
    """
    flat    = buffer.flatten()
    N_total = flat["states"].shape[0]     # B * np * T
    mb_size = cfg.minibatch_size()
    mb_size = min(mb_size, N_total)

    stats = {
        "policy_loss": 0.0, "value_loss": 0.0,
        "entropy":     0.0, "kl":         0.0,
        "n_updates":   0,
    }

    for k in range(cfg.K_epochs):
        # Random minibatch
        idx = torch.randperm(N_total, device=cfg.device)[:mb_size]

        states_mb    = flat["states"][idx]        # (mb, d)
        sigmas_mb    = flat["sigmas"][idx]        # (mb,)
        log_probs_old = flat["log_probs"][idx]    # (mb,)
        returns_mb   = flat["returns"][idx]       # (mb,)
        adv_mb       = flat["advantages"][idx]    # (mb,)

        # ── New log-probs under current policy ──────────────────────────────
        log_probs_new = policy.log_prob(states_mb, sigmas_mb)   # (mb,)
        entropy       = policy.entropy(states_mb).mean()        # scalar

        # ── Importance sampling ratio ────────────────────────────────────────
        # rho = pi_new(a|s) / pi_old(a|s) = exp(log_pi_new - log_pi_old)
        #
        # Clamp the log-ratio before exponentiating.  If the policy drifts far
        # from the rollout distribution the raw difference can be ±100, causing
        # exp() to overflow float32 (~3.4e38 max) and produce Inf/NaN that
        # cascades into an astronomically large policy loss (seen at ep 610:
        # pl = 3.2e11).  Clamping to ±10 limits ratio to [~5e-5, ~22000]
        # which is already well beyond any meaningful importance-weight range
        # (the PPO clip at 0.3 confines useful updates to ratio ∈ [0.7, 1.3]).
        log_ratio = (log_probs_new - log_probs_old.detach()).clamp(-10.0, 10.0)
        ratio     = log_ratio.exp()                              # (mb,)

        # ── Approximate KL divergence (for early stopping) ──────────────────
        # Using the approximation: KL ≈ (ratio - 1) - log(ratio)
        with torch.no_grad():
            kl_approx = ((ratio - 1) - log_ratio).mean().item()

        # ── PPO clipped objective ────────────────────────────────────────────
        # L_clip = -E[ min(rho*A, clip(rho, 1-eps, 1+eps)*A) ]
        clip_eps  = cfg.clip
        surr1     = ratio * adv_mb
        surr2     = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_mb
        policy_loss = -torch.min(surr1, surr2).mean()

        # ── Value function loss ──────────────────────────────────────────────
        V_pred     = value(states_mb)                            # (mb,)
        value_loss = (V_pred - returns_mb.detach()).pow(2).mean()

        # ── Total loss ──────────────────────────────────────────────────────
        total_loss = (policy_loss
                      + cfg.c_value   * value_loss
                      - cfg.c_entropy * entropy)

        # ── Gradient step ───────────────────────────────────────────────────
        opt_p.zero_grad()
        opt_v.zero_grad()
        total_loss.backward()

        # Gradient clipping — prevents very large updates early in training
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        nn.utils.clip_grad_norm_(value.parameters(),  max_norm=0.5)

        opt_p.step()
        opt_v.step()

        # Accumulate stats
        stats["policy_loss"] += policy_loss.item()
        stats["value_loss"]  += value_loss.item()
        stats["entropy"]     += entropy.item()
        stats["kl"]          += kl_approx
        stats["n_updates"]   += 1

        # ── Early stopping on KL ─────────────────────────────────────────────
        if kl_approx > cfg.kl_target:
            break

    # Average over completed updates
    n = stats["n_updates"]
    for key in ("policy_loss", "value_loss", "entropy", "kl"):
        stats[key] /= max(n, 1)

    return stats


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

class MARLVolTrainer:
    """
    Top-level trainer.  Assembles all components and runs Algorithm 1.

    Parameters
    ----------
    surface : VolSurface — market data
    cfg     : TrainConfig — all hyperparameters
    bermudan: BermudanSpec — required for exp2, ignored for exp1
    """

    def __init__(
        self,
        surface:  VolSurface,
        cfg:      TrainConfig,
        bermudan: Optional[BermudanSpec] = None,
    ):
        self.surface  = surface
        self.cfg      = cfg
        self.bermudan = bermudan

        # Infer S0 from surface if not set
        if cfg.S0 is None:
            cfg.S0 = surface.spot

        device = cfg.device

        # ── Instrument grid ─────────────────────────────────────────────────
        self.grid   = surface.build_instrument_grid(n_strikes=cfg.n_strikes)
        self.mkt_px = torch.tensor(
            self.grid["price_mkt"].values, dtype=torch.float32, device=device
        )

        # ── Instrument tensors for IV-based calibration loss ────────────────
        # IV-space loss weights every strike identically — no gradient
        # imbalance between cheap OTM calls and expensive ITM calls.
        # We precompute strikes, maturities, and market IVs once here.
        self.inst_strikes = torch.tensor(
            self.grid["K"].values, dtype=torch.float32, device=device)
        self.inst_T_years = torch.tensor(
            self.grid["T_years"].values, dtype=torch.float32, device=device)
        self.inst_mkt_ivs = torch.tensor(
            self.grid["iv_mkt"].values, dtype=torch.float32, device=device)

        # OTM-adjusted price denominator — kept as fallback for IV loss when
        # Newton-Raphson fails to converge (rare but possible for deep OTM MC prices).
        self.mkt_otm = surface.otm_denom_tensor(self.grid, device=device)

        # ── Reference loss (flat BS smile) — in IV space ─────────────────────
        self.l_ref = reference_loss_bs(self.grid, surface, device=device,
                                       denom=self.mkt_otm)
        print(f"Reference loss (BS flat smile): {self.l_ref.item():.6f}")

        # ── Policy and value networks ────────────────────────────────────────
        self.policy = PolicyNet(state_dim=cfg.state_dim).to(device)
        self.value  = ValueNet(state_dim=cfg.state_dim).to(device)

        # ── Optimisers ──────────────────────────────────────────────────────
        # Both networks use the same learning rate (paper Table 1: lr=1e-4).
        # The value function needs to track returns at the same pace as the
        # policy changes them — a slower value LR makes advantages noisier,
        # which hurts the policy gradient, especially with small reward scales.
        self.opt_p = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.opt_v = torch.optim.Adam(self.value.parameters(),  lr=cfg.lr)

        # ── Running return normalizer ────────────────────────────────────────
        # Keeps value-function MSE ~ O(1) across all training episodes.
        self.ret_normalizer = ReturnNormalizer()

        # ── MC engine ───────────────────────────────────────────────────────
        self.engine = MCEngine(
            n_paths=cfg.n_paths,
            T_steps=cfg.T_steps,
            S0=cfg.S0,
            delta=cfg.delta,
            device=device,
        )

        # ── Basis player manager ─────────────────────────────────────────────
        self.manager = BasisPlayerManager(
            n_paths=cfg.n_paths,
            n_basis=cfg.n_basis,
            method=cfg.bp_method,
            k=cfg.bp_k,
            noise_std=cfg.noise_std,
            seed=0,
            device=device,
        )

        # ── Log storage ──────────────────────────────────────────────────────
        self.log: List[Dict] = []

        os.makedirs(cfg.save_dir, exist_ok=True)

    def train(self) -> List[Dict]:
        """
        Run the full training loop.

        Returns
        -------
        log : list of dicts, one per episode, with keys:
              episode, loss, reward, policy_loss, value_loss, kl, entropy,
              n_ppo_updates, wall_time
        """
        cfg    = self.cfg
        device = cfg.device

        print(f"\nStarting training: {cfg.n_episodes} episodes, "
              f"experiment={cfg.experiment}, device={device}")
        print(f"n_paths={cfg.n_paths:,}, n_basis={cfg.n_basis}, "
              f"T={cfg.T_steps}, B={cfg.B_envs}")
        print(f"PPO: lr={cfg.lr}, clip={cfg.clip}, "
              f"K={cfg.K_epochs}, mb={cfg.minibatch_size()}\n")

        t_start = time.perf_counter()

        # ── Best-checkpoint tracking ─────────────────────────────────────────
        best_reward      = -float("inf")   # best single-episode reward seen
        best_episode     = 0
        # Patience counter for convergence check
        best_window_mean = -float("inf")   # best rolling-window mean seen
        no_improve_count = 0               # consecutive non-improving windows

        for episode in range(1, cfg.n_episodes + 1):
            t_ep = time.perf_counter()

            # ── Collect B_envs rollouts ──────────────────────────────────────
            buf_states    = []
            buf_sigmas    = []
            buf_log_probs = []
            buf_returns   = []
            buf_advantages= []
            ep_losses     = []
            ep_rewards    = []

            for b in range(cfg.B_envs):
                # Single rollout
                (states_bp, sigmas_bp, log_probs_bp,
                 rewards, loss_val, ep_r) = collect_rollout(
                    engine        = self.engine,
                    policy        = self.policy,
                    value         = self.value,
                    manager       = self.manager,
                    surface       = self.surface,
                    grid          = self.grid,
                    cfg           = cfg,
                    bermudan      = self.bermudan,
                    l_ref         = self.l_ref,
                    mkt_otm       = self.mkt_otm,
                    inst_strikes  = self.inst_strikes,
                    inst_T_years  = self.inst_T_years,
                    inst_mkt_ivs  = self.inst_mkt_ivs,
                )

                # Track return statistics (for logging/diagnostics only)
                self.ret_normalizer.update(ep_r)

                # Compute advantages.
                # We do NOT pass ret_normalizer here — the log-transform on
                # the reward already bounds returns to roughly [-14, +5], so
                # the value-function MSE stays O(1–100) without normalization.
                # Normalizing the returns would collapse all targets to ~0 and
                # make the value function predict a constant, zeroing out all
                # advantages and killing the policy gradient.
                returns, advantages = compute_advantages(
                    states_bp, rewards, self.value, cfg,
                    bp_idx=self.manager.basis_idx,
                )

                buf_states.append(states_bp)        # (np, T, d)
                buf_sigmas.append(sigmas_bp)        # (np, T)
                buf_log_probs.append(log_probs_bp)  # (np, T)
                buf_returns.append(returns)         # (np, T)
                buf_advantages.append(advantages)   # (np, T)
                ep_losses.append(loss_val)
                ep_rewards.append(ep_r)

            # Stack B rollouts: (B, np, T, ...)
            adv_all = torch.stack(buf_advantages, dim=0)   # (B, np, T)

            # Normalise advantages across ALL B*np*T samples together.
            # This is the standard PPO practice and, crucially, it preserves
            # the between-rollout variation in G:
            #   - Rollouts with low calibration loss → high G → raw A = G - V is large
            #   - Rollouts with high calibration loss → low G  → raw A = G - V is small
            # After cross-rollout normalisation these differences survive and
            # the policy gradient correctly points toward the better rollouts.
            adv_flat = adv_all.flatten()
            adv_flat = (adv_flat - adv_flat.mean()) / adv_flat.std().clamp(min=1)
            adv_all  = adv_flat.reshape_as(adv_all)

            buffer = RolloutBuffer(
                states     = torch.stack(buf_states,     dim=0),
                sigmas     = torch.stack(buf_sigmas,     dim=0),
                log_probs  = torch.stack(buf_log_probs,  dim=0),
                returns    = torch.stack(buf_returns,    dim=0),
                advantages = adv_all,
            )

            # ── PPO update ───────────────────────────────────────────────────
            ppo_stats = ppo_update(
                buffer = buffer,
                policy = self.policy,
                value  = self.value,
                opt_p  = self.opt_p,
                opt_v  = self.opt_v,
                cfg    = cfg,
            )

            # ── Logging ──────────────────────────────────────────────────────
            avg_loss   = float(np.mean(ep_losses))
            avg_reward = float(np.mean(ep_rewards))
            wall_time  = time.perf_counter() - t_start

            record = {
                "episode":      episode,
                "loss":         avg_loss,
                "reward":       avg_reward,
                "policy_loss":  ppo_stats["policy_loss"],
                "value_loss":   ppo_stats["value_loss"],
                "entropy":      ppo_stats["entropy"],
                "kl":           ppo_stats["kl"],
                "n_ppo_updates":ppo_stats["n_updates"],
                "wall_time":    wall_time,
            }
            self.log.append(record)

            if episode % cfg.log_every == 0:
                ep_time = time.perf_counter() - t_ep
                print(
                    f"Ep {episode:4d}/{cfg.n_episodes}  "
                    f"loss={avg_loss:.5f}  reward={avg_reward:+.5f}  "
                    f"pl={ppo_stats['policy_loss']:+.4f}  "
                    f"vl={ppo_stats['value_loss']:.4f}  "
                    f"kl={ppo_stats['kl']:.5f}  "
                    f"ppo_iters={ppo_stats['n_updates']:2d}  "
                    f"ep_time={ep_time:.1f}s"
                )

            if episode % cfg.save_every == 0:
                self._save_checkpoint(episode)

            # ── Best-checkpoint tracking ──────────────────────────────────────
            # Save a dedicated 'best' checkpoint whenever a new single-episode
            # reward record is set.  This guarantees plots always use the model
            # with the lowest calibration loss seen during the entire run,
            # regardless of where training is stopped.
            if avg_reward > best_reward:
                best_reward  = avg_reward
                best_episode = episode
                self._save_checkpoint(episode, tag="best")

            # ── Convergence check (patience-based, non-overlapping windows) ──
            # Evaluate one NON-OVERLAPPING window every conv_window episodes.
            # This ensures the patience counter represents truly independent
            # windows, not episode-by-episode sliding-mean noise.
            #
            # At episode w, 2w, 3w, ... we compute the mean reward of the
            # just-completed window of w episodes.  If this mean does not beat
            # the best window mean by more than conv_tol we increment the
            # patience counter.  Training stops after conv_patience consecutive
            # non-improving windows (i.e., conv_patience * conv_window episodes
            # of genuine no-progress).
            w = cfg.conv_window
            if w > 0 and episode >= w and episode % w == 0:
                # Mean of the most-recently completed w-episode window
                window_mean = np.mean([r["reward"] for r in self.log[-w:]])
                if window_mean > best_window_mean + cfg.conv_tol:
                    best_window_mean = window_mean
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= cfg.conv_patience:
                        print(f"\nConverged at episode {episode} "
                              f"(no improvement > {cfg.conv_tol} for "
                              f"{cfg.conv_patience} consecutive {w}-ep windows; "
                              f"best was ep {best_episode}, reward={best_reward:.5f})")
                        break

        # Final checkpoint
        self._save_checkpoint(episode)
        print(f"  Best checkpoint: ep {best_episode}  reward={best_reward:.5f}")
        print(f"\nTraining complete. Total time: "
              f"{time.perf_counter() - t_start:.1f}s")
        return self.log

    def _save_checkpoint(self, episode: int, tag: str = ""):
        """Save policy, value, normalizer state, and log to disk.

        Parameters
        ----------
        episode : episode number (used in the default filename)
        tag     : optional suffix override; if given, the file is saved as
                  ``{experiment}_{tag}.pt`` instead of ``{experiment}_ep{N}.pt``.
                  Use tag='best' to maintain a single always-current best file.
        """
        if tag:
            fname = f"{self.cfg.experiment}_{tag}.pt"
        else:
            fname = f"{self.cfg.experiment}_ep{episode}.pt"
        path = os.path.join(self.cfg.save_dir, fname)
        torch.save({
            "episode":        episode,
            "policy":         self.policy.state_dict(),
            "value":          self.value.state_dict(),
            "policy_norm":    self.policy.norm.state_dict(),
            "value_norm":     self.value.norm.state_dict(),
            # Save return normalizer state for correct resume behaviour
            "ret_norm_mean":  self.ret_normalizer.mean,
            "ret_norm_M2":    self.ret_normalizer.M2,
            "ret_norm_count": self.ret_normalizer.count,
            "log":            self.log,
            "cfg":            self.cfg,
        }, path)
        print(f"  Checkpoint saved: {path}")

    @staticmethod
    def load_checkpoint(path: str, surface: VolSurface,
                        bermudan: Optional[BermudanSpec] = None
                        ) -> "MARLVolTrainer":
        """Load a trainer from a saved checkpoint."""
        ckpt    = torch.load(path, weights_only=False)
        cfg     = ckpt["cfg"]
        trainer = MARLVolTrainer(surface, cfg, bermudan)
        trainer.policy.load_state_dict(ckpt["policy"])
        trainer.value.load_state_dict(ckpt["value"])
        trainer.policy.norm.load_state_dict(ckpt["policy_norm"])
        trainer.value.norm.load_state_dict(ckpt["value_norm"])
        # Restore return normalizer state (backwards compatible)
        if "ret_norm_count" in ckpt:
            trainer.ret_normalizer.mean  = ckpt["ret_norm_mean"]
            trainer.ret_normalizer.M2    = ckpt["ret_norm_M2"]
            trainer.ret_normalizer.count = ckpt["ret_norm_count"]
        trainer.log = ckpt["log"]
        return trainer


# ---------------------------------------------------------------------------
# Self-test  (smoke test only — full training runs in experiments/)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")

    print("=" * 60)
    print("marl_vol.py smoke test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    torch.manual_seed(0)

    surface = VolSurface("data/spx_smiles_clean.csv")

    # ── Smoke test: 3 episodes of Exp 1 with reduced scale ───────────────────
    cfg = TrainConfig(
        n_paths    = 10_000,    # reduced from 120k for speed
        T_steps    = 51,
        n_basis    = 20,        # reduced from 100
        B_envs     = 2,         # reduced from 10
        K_epochs   = 3,         # reduced from 30
        n_episodes = 3,
        experiment = "exp1",
        state_dim  = 2,
        log_every  = 1,
        save_every = 999,       # no saves during smoke test
        save_dir   = "results/smoke_test",
        device     = device,
    )

    trainer = MARLVolTrainer(surface=surface, cfg=cfg)
    log     = trainer.train()

    assert len(log) == 3, f"Expected 3 log entries, got {len(log)}"
    assert all("loss" in r for r in log), "Missing 'loss' in log"
    assert all(np.isfinite(r["loss"]) for r in log), "Non-finite loss"
    assert all(np.isfinite(r["reward"]) for r in log), "Non-finite reward"

    print(f"\nSmoke test passed.")
    print(f"Initial loss: {log[0]['loss']:.5f}")
    print(f"Final   loss: {log[-1]['loss']:.5f}")
    print(f"\n{'='*60}")
    print("All tests passed.")
    print("=" * 60)
