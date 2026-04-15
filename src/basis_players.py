"""
src/basis_players.py
====================
Basis player exploration and interpolation — the core algorithmic
contribution of Vadori (2022), equation 15.

The problem being solved
-------------------------
The RL policy maps state -> volatility for each of n=120,000 trajectories.
A naive implementation would compute a separate policy gradient for each
trajectory, which is both statistically noisy and computationally expensive.

The basis player idea
----------------------
Instead of having all 120,000 trajectories explore independently, only a
small number np << n of "basis players" actually explore (apply random
perturbations to the policy mean). The exploration noise of these np basis
players is then INTERPOLATED to all n trajectories.

Formally (eq. 15 in the paper):
    sigma_i(t) = pi_theta(s_it) + sum_j w_ij(t) * epsilon_j(t)

where:
    pi_theta(s_it)   = policy mean for trajectory i at time t
    epsilon_j(t)     = exploration noise of basis player j at time t
    w_ij(t)          = interpolation weight from basis player j to trajectory i

The weights w_ij are determined by the distances between trajectory states.
Two interpolation methods are implemented:

  Method 1 — k-Nearest Neighbours (kNN):
    Each trajectory borrows noise from its k nearest basis players.
    w_ij = 1/k if basis player j is among the k nearest to trajectory i,
           0   otherwise.
    Simple and robust.

  Method 2 — Linear (Barycentric):
    Each trajectory's state is expressed as a convex combination of nearby
    basis player states. The weights are the barycentric coordinates.
    Smoother interpolation but requires the basis players to form a
    Delaunay triangulation of the state space.

Why does this help?
--------------------
1. Statistical efficiency: the policy gradient is computed by averaging over
   n trajectories, but only np << n trajectories need to maintain individual
   noise. The effective signal-to-noise ratio improves as n/np.

2. Computational efficiency: the "basis" exploration space has dimension np
   not n. The number of policy parameters doesn't grow with n.

3. Cooperative behaviour emerges: all trajectories in the neighbourhood of a
   basis player share the same noise, so they collectively learn whether that
   exploration direction was good or bad.

State space for interpolation
-------------------------------
Interpolation is done in the (normalised) state space. For Experiment 1 this
is (t, S_it/S0) — a 2D space. The basis players are spread across this space
at initialisation (uniform grid or random) and their states evolve during
the episode as their prices move.

At each timestep the state of every trajectory is known, so we can compute
the distances to all basis players and determine the weights.

Implementation note
--------------------
With n=120,000 trajectories and np=100 basis players, computing all n*np
distances naively is a (120000, 100) matrix multiply — very fast on GPU.
We implement both kNN and linear interpolation using pure PyTorch operations
so everything stays on GPU.
"""

import torch
import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Basis player initialisation
# ---------------------------------------------------------------------------

def init_basis_players(
    n_paths:    int,
    n_basis:    int,
    seed:       Optional[int] = None,
    device:     str = "cpu",
) -> torch.Tensor:
    """
    Select indices of the basis player trajectories from the full set of n paths.

    Basis players are chosen uniformly at random from {0, 1, ..., n-1}.
    Their indices are fixed for the lifetime of the training run.

    Parameters
    ----------
    n_paths : total number of trajectories n
    n_basis : number of basis players np (paper uses ~100)
    seed    : optional RNG seed
    device  : torch device

    Returns
    -------
    basis_idx : (np,) integer indices identifying the basis player trajectories
    """
    if n_basis > n_paths:
        raise ValueError(f"n_basis ({n_basis}) cannot exceed n_paths ({n_paths})")

    rng = torch.Generator(device="cpu")
    if seed is not None:
        rng.manual_seed(seed)

    # torch.randperm on CPU then move — avoids device issues with Generator
    perm = torch.randperm(n_paths, generator=rng)[:n_basis]
    return perm.to(device)


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------

def pairwise_distances(
    states_all:   torch.Tensor,
    states_basis: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Euclidean distances between every trajectory state and every
    basis player state.

    Parameters
    ----------
    states_all   : (n,  d) states of all trajectories
    states_basis : (np, d) states of basis players

    Returns
    -------
    D : (n, np) distance matrix, D[i, j] = ||states_all[i] - states_basis[j]||_2
    """
    # Efficient computation via the identity:
    #   ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a^T b
    # This avoids materialising the (n, np, d) difference tensor.
    sq_all   = (states_all   ** 2).sum(dim=1, keepdim=True)   # (n,  1)
    sq_basis = (states_basis ** 2).sum(dim=1, keepdim=True).T  # (1, np)
    cross    = states_all @ states_basis.T                     # (n, np)
    D_sq     = (sq_all + sq_basis - 2 * cross).clamp(min=0.0)  # numerical safety
    return D_sq.sqrt()


# ---------------------------------------------------------------------------
# kNN interpolation weights
# ---------------------------------------------------------------------------

def knn_weights(
    states_all:   torch.Tensor,
    states_basis: torch.Tensor,
    k:            int = 1,
) -> torch.Tensor:
    """
    Compute kNN interpolation weights.

    Each trajectory borrows noise from its k nearest basis players with
    equal weight 1/k. All other weights are zero.

    Parameters
    ----------
    states_all   : (n,  d)
    states_basis : (np, d)
    k            : number of nearest basis players to use

    Returns
    -------
    W : (n, np) weight matrix, each row sums to 1.0
    """
    D = pairwise_distances(states_all, states_basis)    # (n, np)
    n, n_basis = D.shape
    k = min(k, n_basis)

    # Get indices of k nearest basis players for each trajectory
    # torch.topk with largest=False gives k smallest distances
    _, knn_idx = torch.topk(D, k, dim=1, largest=False)   # (n, k)

    # Build sparse weight matrix
    W = torch.zeros(n, n_basis, dtype=states_all.dtype, device=states_all.device)
    W.scatter_(1, knn_idx, 1.0 / k)   # set k entries to 1/k per row
    return W


# ---------------------------------------------------------------------------
# Linear (barycentric-style) interpolation weights
# ---------------------------------------------------------------------------

def linear_weights(
    states_all:   torch.Tensor,
    states_basis: torch.Tensor,
    k:            int = 4,
    eps:          float = 1e-8,
) -> torch.Tensor:
    """
    Compute inverse-distance-weighted interpolation weights.

    This is a soft version of kNN: weights are proportional to 1/distance
    for the k nearest basis players, then normalised to sum to 1.
    It gives smoother interpolation than hard kNN.

    (True barycentric interpolation requires a Delaunay triangulation which
    is expensive to maintain as states evolve. Inverse-distance weighting
    is a practical and commonly used approximation.)

    Parameters
    ----------
    states_all   : (n,  d)
    states_basis : (np, d)
    k            : neighbourhood size
    eps          : small constant to prevent 1/0 for exact matches

    Returns
    -------
    W : (n, np) weight matrix, each row sums to 1.0
    """
    D = pairwise_distances(states_all, states_basis)    # (n, np)
    n, n_basis = D.shape
    k = min(k, n_basis)

    # Get k nearest distances and indices
    knn_dist, knn_idx = torch.topk(D, k, dim=1, largest=False)   # (n, k)

    # Inverse distance weights (add eps for numerical stability)
    inv_dist = 1.0 / (knn_dist + eps)                            # (n, k)
    inv_dist_norm = inv_dist / inv_dist.sum(dim=1, keepdim=True)  # (n, k) normalised

    # Scatter into full (n, np) weight matrix
    W = torch.zeros(n, n_basis, dtype=states_all.dtype, device=states_all.device)
    W.scatter_(1, knn_idx, inv_dist_norm)
    return W


# ---------------------------------------------------------------------------
# Noise interpolation — the core of eq. 15
# ---------------------------------------------------------------------------

def interpolate_noise(
    W:       torch.Tensor,
    epsilon: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolate basis player exploration noise to all trajectories.

    sigma_i = pi_theta(s_i) + sum_j W_ij * epsilon_j

    This function computes the sum_j W_ij * epsilon_j part.

    Parameters
    ----------
    W       : (n, np) weight matrix from knn_weights or linear_weights
    epsilon : (np,)   exploration noise for each basis player at this timestep

    Returns
    -------
    noise_all : (n,) interpolated noise for all trajectories
    """
    # W @ epsilon: (n, np) @ (np,) -> (n,)
    return W @ epsilon


# ---------------------------------------------------------------------------
# BasisPlayerManager — stateful wrapper
# ---------------------------------------------------------------------------

class BasisPlayerManager:
    """
    Manages basis player state and noise interpolation throughout an episode.

    Usage pattern (inside the training loop):
        manager = BasisPlayerManager(n_paths=120_000, n_basis=100, ...)
        manager.reset(states_t0)            # at start of episode

        for t in range(T):
            states = build_state(...)       # (n, d) current states
            epsilon = manager.sample_noise(...)   # basis player noise

            # Compute interpolation weights from current states
            sigma_mean = policy(states)     # (n,) policy mean

            # Interpolated noise
            W     = manager.compute_weights(states)    # (n, np)
            noise = interpolate_noise(W, epsilon)      # (n,)
            sigma = (sigma_mean + noise).clamp(...)    # (n,) actual vols

    Parameters
    ----------
    n_paths   : total trajectories n
    n_basis   : number of basis players np
    method    : 'knn' or 'linear'
    k         : neighbourhood size for interpolation
    noise_std : standard deviation of basis player exploration noise
    seed      : RNG seed for basis player selection
    device    : torch device
    """

    def __init__(
        self,
        n_paths:   int   = 120_000,
        n_basis:   int   = 100,
        method:    str   = "knn",
        k:         int   = 1,
        noise_std: float = 0.02,
        seed:      Optional[int] = None,
        device:    str   = "cpu",
    ):
        self.n_paths   = n_paths
        self.n_basis   = n_basis
        self.method    = method
        self.k         = k
        self.noise_std = noise_std
        self.device    = device

        # Fixed basis player trajectory indices
        self.basis_idx = init_basis_players(n_paths, n_basis, seed=seed, device=device)

    def compute_weights(
        self,
        states_all: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute interpolation weights from current states.

        Parameters
        ----------
        states_all : (n, d) current states for ALL trajectories

        Returns
        -------
        W : (n, np) weight matrix
        """
        states_basis = states_all[self.basis_idx]   # (np, d)

        if self.method == "knn":
            return knn_weights(states_all, states_basis, k=self.k)
        elif self.method == "linear":
            return linear_weights(states_all, states_basis, k=self.k)
        else:
            raise ValueError(f"Unknown method '{self.method}'. Use 'knn' or 'linear'.")

    def sample_noise(
        self,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Sample independent exploration noise for each basis player.

        Returns
        -------
        epsilon : (np,) i.i.d. N(0, noise_std^2)
        """
        return torch.randn(
            self.n_basis,
            device=self.device,
            generator=generator,
        ) * self.noise_std

    def get_basis_states(self, states_all: torch.Tensor) -> torch.Tensor:
        """Extract states of basis player trajectories from the full state matrix."""
        return states_all[self.basis_idx]   # (np, d)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os, time
    sys.path.insert(0, os.path.dirname(__file__))

    from diffusion import generate_brownian, simulate_paths, DELTA
    from policy    import build_state_exp1, PolicyNet

    print("=" * 60)
    print("basis_players.py self-test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    torch.manual_seed(0)

    n       = 120_000
    n_basis = 100
    S0      = 100.0
    T       = 51

    # ── Test 1: init_basis_players ────────────────────────────────────────────
    print("Test 1: init_basis_players")
    idx = init_basis_players(n, n_basis, seed=42, device=device)
    assert idx.shape == (n_basis,), f"Expected ({n_basis},), got {idx.shape}"
    assert idx.max().item() < n,    "Index out of bounds"
    assert len(idx.unique()) == n_basis, "Duplicate basis player indices"
    print(f"  Selected {n_basis} unique indices in [0, {n-1}]")
    print("  PASSED\n")

    # ── Test 2: pairwise_distances ────────────────────────────────────────────
    print("Test 2: pairwise_distances")
    # Construct points where distances are known
    a = torch.tensor([[0., 0.], [3., 4.]], device=device)   # 2 trajectories
    b = torch.tensor([[0., 0.]], device=device)              # 1 basis player
    D = pairwise_distances(a, b)
    assert D.shape == (2, 1)
    assert abs(D[0, 0].item() - 0.0) < 1e-5   # distance (0,0) to (0,0) = 0
    assert abs(D[1, 0].item() - 5.0) < 1e-5   # distance (3,4) to (0,0) = 5
    print(f"  Distances: {D.squeeze().tolist()}  (expected [0.0, 5.0])")
    print("  PASSED\n")

    # ── Test 3: knn_weights row-sum = 1 ──────────────────────────────────────
    print("Test 3: knn_weights — row sums, shape, non-negative")
    states = torch.randn(n, 2, device=device)
    basis  = states[idx]
    W_knn  = knn_weights(states, basis, k=3)
    assert W_knn.shape == (n, n_basis)
    row_sums = W_knn.sum(dim=1)
    assert (row_sums - 1.0).abs().max().item() < 1e-5, "kNN weights don't sum to 1"
    assert (W_knn >= 0).all(), "Negative kNN weight"
    # Each row should have exactly k=3 non-zero entries
    nnz_per_row = (W_knn > 0).sum(dim=1)
    assert (nnz_per_row == 3).all(), "Wrong number of non-zero kNN weights"
    print(f"  Shape {list(W_knn.shape)}, row sums = 1.0, each row has 3 non-zeros")
    print("  PASSED\n")

    # ── Test 4: linear_weights row-sum = 1 ───────────────────────────────────
    print("Test 4: linear_weights — row sums, shape, non-negative")
    W_lin = linear_weights(states, basis, k=4)
    assert W_lin.shape == (n, n_basis)
    row_sums_lin = W_lin.sum(dim=1)
    assert (row_sums_lin - 1.0).abs().max().item() < 1e-5
    assert (W_lin >= 0).all()
    print(f"  Shape {list(W_lin.shape)}, row sums = 1.0, all weights >= 0")
    print("  PASSED\n")

    # ── Test 5: interpolate_noise ─────────────────────────────────────────────
    print("Test 5: interpolate_noise — mean and variance")
    epsilon = torch.randn(n_basis, device=device) * 0.02
    noise   = interpolate_noise(W_knn, epsilon)
    assert noise.shape == (n,)
    # With kNN(k=3), each trajectory gets the mean of 3 basis player noises.
    # The mean of n noises should be ~0; std should be ~0.02/sqrt(3) ~ 0.012
    print(f"  noise mean: {noise.mean().item():.5f}  (expected ~0)")
    print(f"  noise std : {noise.std().item():.5f}   (expected ~0.02/sqrt(3) = {0.02/3**0.5:.5f})")
    assert noise.abs().mean().item() < 0.05, "Noise magnitude seems wrong"
    print("  PASSED\n")

    # ── Test 6: Full episode with BasisPlayerManager ──────────────────────────
    print("Test 6: BasisPlayerManager full episode")
    manager = BasisPlayerManager(
        n_paths=n, n_basis=n_basis, method="knn", k=1,
        noise_std=0.02, seed=0, device=device,
    )
    policy = PolicyNet(state_dim=2).to(device)

    Z      = generate_brownian(n, T, seed=0, device=device)
    sigmas = torch.zeros(n, T, device=device)

    S_cur  = torch.full((n,), S0, device=device)

    for t in range(T):
        state   = build_state_exp1(t, S_cur, T, S0)
        policy.norm.update(state)
        mu, _   = policy(state)            # (n,) policy mean
        epsilon = manager.sample_noise()   # (np,) basis noise
        W       = manager.compute_weights(state)  # (n, np)
        noise   = interpolate_noise(W, epsilon)   # (n,) interpolated noise
        sigma_t = (mu + noise).clamp(0.01, 2.0)
        sigmas[:, t] = sigma_t
        # Advance prices
        S_cur = S_cur * torch.exp(
            -0.5 * sigma_t ** 2 * DELTA + sigma_t * DELTA ** 0.5 * Z[:, t]
        )

    assert sigmas.shape == (n, T)
    assert (sigmas > 0).all()
    print(f"  sigmas shape: {list(sigmas.shape)}")
    print(f"  sigma range: [{sigmas.min().item():.4f}, {sigmas.max().item():.4f}]")
    print("  PASSED\n")

    # ── Test 7: Timing ────────────────────────────────────────────────────────
    print("Test 7: Timing — compute_weights at full scale")
    state_t = build_state_exp1(25, S_cur, T, S0)
    # Warm-up
    _ = manager.compute_weights(state_t)
    if device == "cuda":
        torch.cuda.synchronize()

    n_runs = 20
    t0 = time.perf_counter()
    for _ in range(n_runs):
        W = manager.compute_weights(state_t)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_runs * 1000

    print(f"  compute_weights (n={n:,}, np={n_basis}): {elapsed:.2f} ms per call")
    print("  PASSED\n")

    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)
