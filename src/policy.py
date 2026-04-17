"""
src/policy.py
=============
Actor-critic neural networks for the MARL volatility calibration agents.

Network architecture (from the paper, Section 3.2)
----------------------------------------------------
Both the policy (actor) and value function (critic) share the same
architecture:
    - 3 hidden layers
    - 50 nodes per layer
    - tanh activations throughout
    - Linear output layer

The actor outputs the MEAN of a Gaussian distribution over volatility.
The standard deviation is a separate learnable parameter (log_std), NOT
a function of the state — this is the standard PPO setup for continuous
action spaces.

The critic outputs a single scalar: the estimated value V(s).

State representations
----------------------
Two state types are used across the two experiments:

Experiment 1 — Local vol (non-path-dependent):
    state = (t, S_it)
    where t is the current timestep (normalised) and S_it is the current
    price of trajectory i. Dimension: 2.

Experiment 2 — Bermudan (path-dependent):
    state = (t, S_it, sigma_{i,t-1}, S_{i, t^t1}, sigma_{i, (t-1)^t1})
    where t1 is the first exercise date, and x^t1 = min(x, t1).
    The extra features capture the path up to t1, encoding the
    information needed to compute Gyongy localisation.
    Dimension: 5.

    The non-path-dependent version for comparison uses only:
    state = (t, S_it, sigma_{i,t-1})
    Dimension: 3.

State normalisation
--------------------
Raw state values have very different scales:
  - t ranges over [0, 51] (timestep index, or fraction of T)
  - S_it ranges around spot (~6817 for SPX, ~100 for normalised)
  - sigma is typically in [0.05, 0.80]

We normalise each dimension to have zero mean and unit variance using
running statistics (updated during training). This is essential for stable
training — unnormalised inputs cause slow convergence and gradient issues.

Volatility output
-----------------
The actor outputs a MEAN volatility mu. The actual volatility used in
the simulation is sampled as:
    sigma ~ N(mu, exp(log_std)^2)    (clamped to [sigma_min, sigma_max])

During PPO updates we need log_prob(sigma | state) for the importance
sampling ratio. The Policy class provides this directly.

Why separate actor and critic?
--------------------------------
We use the "separate networks" design (not shared parameters) because:
1. The value function and policy have different learning targets and can
   benefit from different learning rates in principle.
2. It's simpler to reason about — no gradient interference between them.
3. The paper does not specify shared vs separate; separate is the safer
   default for PPO.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Constants matching the paper
# ---------------------------------------------------------------------------

HIDDEN_DIM  = 50          # nodes per hidden layer
N_LAYERS    = 3           # number of hidden layers
SIGMA_MIN   = 0.01        # minimum allowed volatility (1%)
SIGMA_MAX   = 2.00        # maximum allowed volatility (200%)
LOG_STD_MIN = -4.0        # clamp on log_std (prevents collapse to deterministic)
LOG_STD_MAX = 0.5         # clamp on log_std (prevents explosion)


# ---------------------------------------------------------------------------
# Shared MLP backbone
# ---------------------------------------------------------------------------

def _make_mlp(input_dim: int, output_dim: int,
              hidden_dim: int = HIDDEN_DIM,
              n_layers:   int = N_LAYERS) -> nn.Sequential:
    """
    Build a fully-connected MLP:
        input -> [Linear(hidden) -> tanh] * n_layers -> Linear(output)

    Parameters
    ----------
    input_dim  : dimension of the input state vector
    output_dim : dimension of the output
    hidden_dim : nodes per hidden layer (paper: 50)
    n_layers   : number of hidden layers (paper: 3)
    """
    layers = []
    in_dim = input_dim
    for _ in range(n_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Tanh())
        in_dim = hidden_dim
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Running normaliser (online mean/variance)
# ---------------------------------------------------------------------------

class RunningNormaliser(nn.Module):
    """
    Maintains running mean and variance of inputs and normalises them.

    Uses Welford's online algorithm for numerically stable updates.
    Parameters are registered as non-gradient buffers — they are updated
    manually during rollout collection, not by the optimizer.

    Parameters
    ----------
    dim   : dimension of the input vector
    eps   : small constant to prevent division by zero in std
    clip  : clip normalised values to [-clip, +clip] for stability
    """

    def __init__(self, dim: int, eps: float = 1e-6, clip: float = 10.0):
        super().__init__()
        self.dim  = dim
        self.eps  = eps
        self.clip = clip

        # Register as buffers so they move to the correct device with .to()
        # and are included in state_dict for checkpointing
        self.register_buffer("mean",  torch.zeros(dim))
        self.register_buffer("var",   torch.ones(dim))
        self.register_buffer("count", torch.tensor(0.0))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """
        Update running statistics with a batch of observations.

        Parameters
        ----------
        x : (..., dim) — any leading batch dimensions are flattened
        """
        x_flat = x.reshape(-1, self.dim).float()
        batch_mean  = x_flat.mean(dim=0)
        batch_var   = x_flat.var(dim=0, unbiased=False)
        batch_count = torch.tensor(x_flat.shape[0], dtype=torch.float32,
                                   device=self.mean.device)

        # Welford parallel update
        total = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var   * self.count
        m_b = batch_var  * batch_count
        new_var = (m_a + m_b + delta ** 2 * self.count * batch_count / total) / total

        self.mean  = new_mean
        self.var   = new_var
        self.count = total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise x using running statistics."""
        std = (self.var + self.eps).sqrt()
        x_norm = (x - self.mean) / std
        return x_norm.clamp(-self.clip, self.clip)


# ---------------------------------------------------------------------------
# Policy (Actor) network
# ---------------------------------------------------------------------------

class PolicyNet(nn.Module):
    """
    Stochastic policy: maps state -> Gaussian distribution over volatility.

    The mean mu(s) is output by the MLP. The standard deviation is a
    global learnable parameter (state-independent), following the standard
    PPO convention for continuous action spaces.

    Parameters
    ----------
    state_dim  : dimension of the state vector (2 for Exp1, 3 or 5 for Exp2)
    hidden_dim : MLP hidden layer width (default 50, matches paper)
    n_layers   : number of hidden layers  (default 3, matches paper)
    init_std   : initial standard deviation for exploration
    """

    def __init__(
        self,
        state_dim:  int,
        hidden_dim: int   = HIDDEN_DIM,
        n_layers:   int   = N_LAYERS,
        init_std:   float = 0.20,
    ):
        super().__init__()
        # MLP outputs the MEAN of log(sigma) — paper: ln sigma ~ N(mu_theta, sigma_pi^2)
        self.net     = _make_mlp(state_dim, 1, hidden_dim, n_layers)
        # log_std is the log of the policy exploration std (in log-sigma space).
        # Initialise at log(init_std) so early std = init_std = 0.20.
        import math
        self.log_std = nn.Parameter(torch.tensor(math.log(init_std)))
        # Running normaliser for the input state
        self.norm    = RunningNormaliser(state_dim)

        # Initialise MLP final layer so the policy starts at sigma ≈ init_std.
        # With log(sigma) parameterisation, the MLP should output ≈ log(init_std).
        # log(0.20) ≈ -1.609.  We set the bias to this and keep weights near zero.
        init_bias = math.log(init_std)    # e.g. log(0.20) ≈ -1.609
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        state : (n, state_dim) normalised state vectors

        Returns
        -------
        mu_logsig : (n,) mean of log(sigma) — the MLP output directly
                    (paper: ln sigma ~ N(mu_theta(x), sigma_pi^2))
        log_std   : (n,) log of the policy exploration std, broadcast to n
        """
        state_norm = self.norm(state)
        # Raw MLP output IS the log-sigma mean (no softplus).
        # sigma is recovered as exp(mu_logsig), which is always positive.
        mu_logsig = self.net(state_norm).squeeze(-1)       # (n,)
        return mu_logsig, self.log_std.expand_as(mu_logsig)

    def get_std(self) -> torch.Tensor:
        """Return the current policy exploration std (scalar, in log-sigma space)."""
        return self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()

    def get_distribution(self, state: torch.Tensor) -> Normal:
        """Return a Normal distribution object for the given states."""
        mu, log_std = self.forward(state)
        std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        return Normal(mu, std)

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample volatilities and compute their log-probabilities.

        With log(sigma) parameterisation:
          1. Sample log_sigma ~ N(mu_logsig, std^2)
          2. sigma = exp(log_sigma), clamped to [SIGMA_MIN, SIGMA_MAX]
          3. log_prob = log p(log_sigma) = Normal(mu, std).log_prob(log_sigma)

        Returns
        -------
        sigma    : (n,) sampled volatilities, clamped to [SIGMA_MIN, SIGMA_MAX]
        log_prob : (n,) log-probability of each sampled log-sigma
        """
        dist      = self.get_distribution(state)       # Normal over log-sigma
        log_sigma = dist.rsample()                     # sample log-sigma
        log_prob  = dist.log_prob(log_sigma)           # p(log_sigma)
        sigma     = torch.exp(log_sigma).clamp(SIGMA_MIN, SIGMA_MAX)
        return sigma, log_prob

    def log_prob(self, state: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute log-probability of given volatilities under the current policy.
        Used in the PPO importance-sampling ratio.

        With log(sigma) parameterisation we evaluate the Normal(mu, std) PDF at
        log(sigma) — not at sigma.  Both old and new log_probs use the same
        formula, so the Jacobian term (-log sigma) cancels in the ratio
        log(pi_new / pi_old), and importance sampling is exact.

        Parameters
        ----------
        state : (n, state_dim)
        sigma : (n,) volatility actions taken during the rollout

        Returns
        -------
        log_prob : (n,)
        """
        dist = self.get_distribution(state)
        return dist.log_prob(torch.log(sigma.clamp(min=1e-8)))

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Entropy of the policy distribution.
        Used as an optional regularisation term in PPO to encourage exploration.

        Returns
        -------
        entropy : (n,) per-state entropy values
        """
        dist = self.get_distribution(state)
        return dist.entropy()


# ---------------------------------------------------------------------------
# Value (Critic) network
# ---------------------------------------------------------------------------

class ValueNet(nn.Module):
    """
    State-value function V(s): maps state -> scalar expected return.

    Used in PPO to compute advantages:
        A(s, a) = Q(s, a) - V(s) ≈ r + gamma*V(s') - V(s)

    Parameters match PolicyNet for consistency.
    """

    def __init__(
        self,
        state_dim:  int,
        hidden_dim: int = HIDDEN_DIM,
        n_layers:   int = N_LAYERS,
    ):
        super().__init__()
        self.net  = _make_mlp(state_dim, 1, hidden_dim, n_layers)
        self.norm = RunningNormaliser(state_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : (n, state_dim)

        Returns
        -------
        values : (n,) estimated state values
        """
        state_norm = self.norm(state)
        return self.net(state_norm).squeeze(-1)   # (n,)


# ---------------------------------------------------------------------------
# State builder helpers
# ---------------------------------------------------------------------------

def build_state_exp1(
    t:    int,
    S_t:  torch.Tensor,
    T:    int,
    S0:   float,
) -> torch.Tensor:
    """
    Build state tensor for Experiment 1 (local vol).

    State = (t/T, S_it/S0)
    Normalising by T and S0 puts both features in a comparable range near 1.

    Parameters
    ----------
    t   : current timestep index
    S_t : (n,) current prices
    T   : total timesteps (for normalisation)
    S0  : initial spot (for normalisation)

    Returns
    -------
    state : (n, 2)
    """
    t_feat = torch.full_like(S_t, t / T)           # (n,) scalar broadcast
    s_feat = S_t / S0                               # (n,)
    return torch.stack([t_feat, s_feat], dim=1)     # (n, 2)


def build_state_exp2_path_dependent(
    t:          int,
    S_t:        torch.Tensor,
    sigma_prev: torch.Tensor,
    S_at_t1:    torch.Tensor,
    sigma_at_t1: torch.Tensor,
    T:          int,
    S0:         float,
) -> torch.Tensor:
    """
    Build path-dependent state for Experiment 2 (Bermudan).

    State = (t/T, S_it/S0, sigma_{i,t-1}, S_{i,t^t1}/S0, sigma_{i,(t-1)^t1})

    The features S_{i,t^t1} and sigma_{i,(t-1)^t1} record the price and vol
    at the first exercise date t1 (or the current value if t < t1). This
    gives the policy the information it needs to reason about the Bermudan
    exercise boundary relative to the current path.

    Parameters
    ----------
    t             : current timestep
    S_t           : (n,) current prices
    sigma_prev    : (n,) vol applied at t-1 (or sigma_0 at t=0)
    S_at_t1       : (n,) price at min(t, t1) — frozen at t1 once t >= t1
    sigma_at_t1   : (n,) vol at min(t-1, t1) — similarly frozen
    T             : total timesteps
    S0            : initial spot

    Returns
    -------
    state : (n, 5)
    """
    t_feat    = torch.full_like(S_t, t / T)
    s_feat    = S_t / S0
    s_t1_feat = S_at_t1 / S0
    return torch.stack(
        [t_feat, s_feat, sigma_prev, s_t1_feat, sigma_at_t1], dim=1
    )   # (n, 5)


def build_state_exp2_nonpath(
    t:          int,
    S_t:        torch.Tensor,
    sigma_prev: torch.Tensor,
    T:          int,
    S0:         float,
) -> torch.Tensor:
    """
    Build non-path-dependent state for Experiment 2 comparison.

    State = (t/T, S_it/S0, sigma_{i,t-1})

    Used to demonstrate that path-dependent state improves Bermudan minimisation.

    Returns
    -------
    state : (n, 3)
    """
    t_feat = torch.full_like(S_t, t / T)
    s_feat = S_t / S0
    return torch.stack([t_feat, s_feat, sigma_prev], dim=1)   # (n, 3)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    print("=" * 60)
    print("policy.py self-test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    torch.manual_seed(42)

    n    = 120_000
    S0   = 6817.0
    T    = 51

    # ── Test 1: PolicyNet forward pass (Exp 1 state) ──────────────────────────
    print("Test 1: PolicyNet forward — Exp 1 state (dim=2)")
    policy1 = PolicyNet(state_dim=2).to(device)
    S_t    = torch.randn(n, device=device) * 500 + S0
    state1 = build_state_exp1(t=25, S_t=S_t, T=T, S0=S0)
    assert state1.shape == (n, 2)

    # Update normaliser with some data
    policy1.norm.update(state1)
    mu, log_std = policy1(state1)
    assert mu.shape == (n,), f"Expected ({n},), got {mu.shape}"
    assert (mu > 0).all(), "Mean vol must be positive"
    print(f"  mu  range: [{mu.min().item():.4f}, {mu.max().item():.4f}]")
    print(f"  std: {log_std[0].exp().item():.4f}")
    print("  PASSED\n")

    # ── Test 2: PolicyNet sample ──────────────────────────────────────────────
    print("Test 2: PolicyNet.sample() — sigma clamped, log_prob finite")
    sigma, lp = policy1.sample(state1)
    assert sigma.shape == (n,)
    assert lp.shape    == (n,)
    assert (sigma >= SIGMA_MIN).all() and (sigma <= SIGMA_MAX).all(), \
        "Sampled sigma outside [SIGMA_MIN, SIGMA_MAX]"
    assert torch.isfinite(lp).all(), "log_prob contains inf/nan"
    print(f"  sigma range: [{sigma.min().item():.4f}, {sigma.max().item():.4f}]")
    print(f"  log_prob range: [{lp.min().item():.3f}, {lp.max().item():.3f}]")
    print("  PASSED\n")

    # ── Test 3: PolicyNet log_prob matches sample's log_prob ─────────────────
    print("Test 3: log_prob(state, sigma) consistency")
    lp2 = policy1.log_prob(state1, sigma.detach())
    # The clamped sigma used in simulation may differ slightly from the
    # unclamped sample used for log_prob; if no clamping occurred they match
    no_clamp_mask = (sigma > SIGMA_MIN + 1e-4) & (sigma < SIGMA_MAX - 1e-4)
    max_diff = (lp[no_clamp_mask] - lp2[no_clamp_mask]).abs().max().item()
    print(f"  Max log_prob diff (unclamped actions): {max_diff:.2e}")
    assert max_diff < 1e-4
    print("  PASSED\n")

    # ── Test 4: ValueNet forward ──────────────────────────────────────────────
    print("Test 4: ValueNet forward — Exp 1 state")
    value1 = ValueNet(state_dim=2).to(device)
    value1.norm.update(state1)
    V = value1(state1)
    assert V.shape == (n,)
    assert torch.isfinite(V).all()
    print(f"  V range: [{V.min().item():.4f}, {V.max().item():.4f}]")
    print("  PASSED\n")

    # ── Test 5: Path-dependent state (Exp 2, dim=5) ───────────────────────────
    print("Test 5: PolicyNet — Exp 2 path-dependent state (dim=5)")
    policy5 = PolicyNet(state_dim=5).to(device)
    sigma_prev  = torch.full((n,), 0.20, device=device)
    S_at_t1     = torch.randn(n, device=device) * 300 + S0
    sig_at_t1   = torch.full((n,), 0.18, device=device)
    state5 = build_state_exp2_path_dependent(
        t=35, S_t=S_t, sigma_prev=sigma_prev,
        S_at_t1=S_at_t1, sigma_at_t1=sig_at_t1,
        T=T, S0=S0,
    )
    assert state5.shape == (n, 5), f"Expected ({n},5), got {state5.shape}"
    policy5.norm.update(state5)
    mu5, _ = policy5(state5)
    assert mu5.shape == (n,)
    assert (mu5 > 0).all()
    print(f"  state shape: {list(state5.shape)}")
    print(f"  mu range: [{mu5.min().item():.4f}, {mu5.max().item():.4f}]")
    print("  PASSED\n")

    # ── Test 6: Non-path-dependent state (Exp 2, dim=3) ───────────────────────
    print("Test 6: PolicyNet — Exp 2 non-path-dependent state (dim=3)")
    policy3 = PolicyNet(state_dim=3).to(device)
    state3  = build_state_exp2_nonpath(
        t=35, S_t=S_t, sigma_prev=sigma_prev, T=T, S0=S0,
    )
    assert state3.shape == (n, 3), f"Expected ({n},3), got {state3.shape}"
    policy3.norm.update(state3)
    mu3, _ = policy3(state3)
    assert mu3.shape == (n,)
    print(f"  state shape: {list(state3.shape)}")
    print("  PASSED\n")

    # ── Test 7: RunningNormaliser Welford stability ───────────────────────────
    print("Test 7: RunningNormaliser online statistics")
    norm = RunningNormaliser(dim=2).to(device)
    # Feed 10 batches of data from N(3, 2^2)
    true_mean = torch.tensor([3.0, -1.0], device=device)
    true_std  = torch.tensor([2.0,  0.5], device=device)
    for _ in range(10):
        x = torch.randn(10_000, 2, device=device) * true_std + true_mean
        norm.update(x)
    mean_err = (norm.mean - true_mean).abs().max().item()
    std_err  = ((norm.var.sqrt()) - true_std).abs().max().item()
    print(f"  Mean error: {mean_err:.4f}  Std error: {std_err:.4f}")
    assert mean_err < 0.05 and std_err < 0.05, "Normaliser stats inaccurate"
    # Check that normalised output has ~zero mean and ~unit std
    x_test = torch.randn(100_000, 2, device=device) * true_std + true_mean
    x_norm = norm(x_test)
    print(f"  Normalised mean: {x_norm.mean(dim=0).tolist()}  (should be ~[0,0])")
    print(f"  Normalised std : {x_norm.std(dim=0).tolist()}   (should be ~[1,1])")
    print("  PASSED\n")

    # ── Test 8: Parameter count ───────────────────────────────────────────────
    print("Test 8: Parameter counts")
    for name, net in [("Policy(dim=2)", policy1), ("Value(dim=2)", value1),
                       ("Policy(dim=5)", policy5), ("Policy(dim=3)", policy3)]:
        n_params = sum(p.numel() for p in net.parameters())
        print(f"  {name}: {n_params:,} parameters")
    # A 3-layer 50-node network with dim=2 input:
    # Layer 1: 2*50 + 50 = 150, Layer 2: 50*50 + 50 = 2550, Layer 3: 50*50+50=2550
    # Output: 50*1 + 1 = 51.  Total MLP = 5301 + 1 (log_std) = 5302
    expected_policy2 = 5302
    actual = sum(p.numel() for p in policy1.parameters())
    assert actual == expected_policy2, \
        f"Expected {expected_policy2} params, got {actual}"
    print(f"  Policy(dim=2) param count verified: {expected_policy2}")
    print("  PASSED\n")

    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)
