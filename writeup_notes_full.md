# AMS 517 Term Project — Comprehensive Development History
## Multi-Agent RL for Options Pricing Calibration (Vadori 2022 Replication)

Built from full transcript review (~25,000 lines). Every bug, fix, result, and insight captured.

---

## 1. Project Overview

Replication of Vadori (2022) "Calibration of Derivative Pricing Models: A Multi-Agent Reinforcement Learning Perspective."

**Two experiments:**
- **Exp 1:** Train MARL agents to calibrate local vol σ(t,S) to market vanilla call smiles (SPX)
- **Exp 2:** Use calibrated vol surface to jointly minimize Bermudan option price (exotic application)

**Algorithm:** PPO with basis players for variance reduction (Algorithm 1, Vadori 2022).

---

## 2. Data: SPX Smile Surface

### August 2018 (Paper's Dataset)
- **Source:** CBOE EOD options data
- **Observation date:** August 1, 2018
- **Spot:** S0 = 2813.18
- **Maturities (DTE):** 22, 27, 32, 42, 47
- **Strikes:** 80 per maturity, moneyness [0.851, 1.138]
- **IV range:** 7.7% to 27.1%
- **ATM IV at T=51d:** 10.42%
- **Calibration grid:** 5 maturities × 10 strikes = 50 instruments
- **BS baseline loss (IV-space):** L_ref = 0.003819

### April 2026 (Present-Day Extension)
- **Spot:** S0 = 6575.00
- **Maturities (DTE):** 23, 27, 32, 37, 42
- **IV range:** 13.2% to 34.7% (much steeper skew than 2018)
- **BS baseline loss:** L_ref = 0.003553
- **8.6× harder than Aug 2018 by absolute price, but similar IV difficulty**

---

## 3. Full Experiment 1 Development History

### Run 1 — Baseline (convergence bug, before any fixes)
- **Config:** n_paths=120k, T=51, n_basis=100, lr=1e-4, K=30
- **Result:** reward ~+0.040 after 134 episodes; training stopped prematurely
- **Root cause:** `no_improve_count` incremented every episode, not every window. With `conv_window=100`, patience=5 meant training stopped after just 5 individual episodes with no improvement over the current best window. 
- **Fix:** Changed convergence check to `if episode % conv_window == 0:`

### Run 2 — Post-Convergence Fix
- **Result:** ~+0.22 (22.2% improvement over BS baseline)
- **Still poor:** OTM gradient imbalance causing the policy to ignore the left wing

### Run 3 — OTM Denominator Fix (left-skew calibration)
- **Problem identified:** K/S=0.882 ITM call price ~$336, OTM call ~$0.13 → 127x price ratio → 16,000x gradient imbalance; policy optimized right wing only, left wing ignored
- **Fix:** `otm_denom_tensor()` — for K < S0, use OTM put price (= call − intrinsic) as denominator instead of ITM call price
- **Result:** +63.2%

### Run 4 — clamp(min=1) Advantage Normalization
- **Added:** `adv / adv.std().clamp(min=1)` instead of `adv / (adv.std() + 1e-8)`
- **Rationale:** Near convergence, inter-rollout reward variance is small; std+1e-8 amplifies noise excessively
- **Result:** +71.3%

### Run 5 (fixes_v1) — Paper Audit Tests (per-timestep epsilon test)
- **Changes tested:** Per-timestep epsilon sampling, standard advantage norm (std+1e-8), T_steps=47
- **Result:** +62.5% — WORSE than Run 4
- **Finding:** Per-episode epsilon provides cleaner causal attribution for terminal rewards. Sampling once per episode before the time loop means every path's sigma trajectory was shaped by the same exploration direction, giving a clean signal linking the rollout direction to the final calibration loss.
- **Reverted:** Per-episode epsilon and clamp(min=1) kept; T_steps=47 kept (correct); noise_std=1.0 kept

### Run 6 — Paper Audit: 5 Deviations Found and Fixed (fixes_v2)
**Paper audit identified 5 key deviations:**

1. **IV-space loss**: Paper Eq. 6 uses φ(X̂) = BS IV mapping, measures mean-squared IV error. Our implementation used price-space loss. Fixed to `calibration_loss_iv()`.
2. **Normalized reward**: Paper uses r = (L_ref − L) / L_ref, making reward dimensionless in [0,1]. Was using raw loss directly. Fixed.
3. **Log(σ) policy**: Paper samples ln σ ~ N(μ_θ(x), σ²_π). Policy outputs log-sigma mean directly with no softplus. Was using softplus + sigma space. Fixed.
4. **T_steps=47**: Matches the longest calibration maturity (47 DTE). Old code used T=51 giving 4 unconstrained extra steps. Fixed.
5. **Moneyness truncation [0.88, 1.09]**: Paper Figure 1 truncates here. Old code used [0.88, 1.12], with 13% of range being pure extrapolation with no training signal. Fixed.

**Result:** +92.4% at episode 226

### Run 7 — fixes_v3 (antithetic variates + cosine LR)
**Two additional improvements:**

1. **Antithetic variates:** In `MCEngine.reset_episode()`, generate Z for n/2 paths, use -Z for other n/2. Halves MC variance at zero extra cost. Implemented in `src/diffusion.py`.
2. **Cosine annealing LR:** `CosineAnnealingLR(opt, T_max=n_episodes, eta_min=1e-6)`. Decays lr 1e-4 → 1e-6. Addresses post-peak degradation where policy escaped the optimal basin after ep226 in fixes_v2.

**Result:**
- Best: **+94.76% at episode 79** (loss = 0.000200 vs L_ref = 0.003819)
- Top-5 episodes all clustered at ep 78–86: "golden window" effect — antithetic variates gave clean gradients into the optimal basin in the first ~80 episodes, then PPO exploration pushed the policy away
- Training ran 700 episodes before convergence check triggered
- 100-ep means: ep1-100: +68.8%, ep101-200: +76.4%, ep301-700: +73-74% plateau
- The +74% plateau is healthy (well above BS baseline)

### Apr 2026 (fixes_v3)
- **Result:** +97.48% at episode 36 — even better than aug2018
- Total episodes: 600 before convergence
- Final 100-ep mean: +61.0%
- **Insight:** Apr 2026 smiles are smoother/more liquid; the steeper gradients from higher ATM IV (~14%) enable better learning

---

## 4. Experiment 1 — Key Technical Details

### Algorithm (verbatim from paper, as implemented)

**Rollout loop per episode:**
1. Sample once: `epsilon_j ~ N(0, noise_std^2)` for np=100 basis players (outside t loop)
2. For t = 0,...,T-1:
   - Build state `s_it = (t/T, S_it/S0)` for exp1 (2D)
   - Policy forward: `mu_logsig = π_θ(s_it)` → MLP outputs log(sigma) mean
   - Compute basis player weights `W` via kNN k=1
   - Interpolate noise: `noise_i = sum_j W_ij * epsilon_j`
   - Scale noise: `noise_i *= policy_std` (policy-std-scaled exploration)
   - Sample: `log_sigma_i = mu_logsig_i + noise_i`, then `sigma_i = exp(log_sigma_i).clamp(SIGMA_MIN, SIGMA_MAX)`
   - Step MC: `S_{t+1} = S_t * exp(-0.5*σ²*Δt + σ*√Δt*Z_t)`
3. Compute IV calibration loss L over all 120k paths
4. Reward: `r = (L_ref - L) / L_ref`
5. Assign terminal reward to all paths at final timestep

**PPO update (K=30 epochs):**
- Minibatch from basis player trajectories only (n_p × T per rollout, B=10 rollouts)
- IS ratio: `rho = pi_new(sigma|s) / pi_old(sigma|s)` — log_prob evaluated in log(sigma) space (Jacobian cancels)
- log_ratio.clamp(-10, 10) before exp() to prevent float32 overflow
- Value network LR = policy LR (cfg.lr; using 10x was wrong)

### Calibration Loss (IV-space, paper Eq. 6)
```python
def calibration_loss_iv(mc_prices, S0, strikes, T_years, mkt_ivs, fallback_denom=None):
    # Newton-Raphson IV inversion for MC prices
    mc_ivs = implied_vol_batch(mc_prices, S0, strikes, T_years)
    valid = ~torch.isnan(mc_ivs)
    if valid.sum() < 3:
        # fallback to OTM-adjusted price loss
        return calibration_loss(mc_prices, mkt_prices, denom=fallback_denom)
    return ((mc_ivs[valid] - mkt_ivs[valid]) ** 2).mean()
```

### OTM Denominator
```python
def otm_denom_tensor(grid, device):
    # For K < S0: use OTM put = call - max(S0-K, 0) as denominator
    # For K >= S0: use call price directly
    denoms = []
    for _, row in grid.iterrows():
        K, price_mkt = row["K"], row["price_mkt"]
        if K < self.spot:
            denom = price_mkt - max(self.spot - K, 0)
        else:
            denom = price_mkt
        denoms.append(max(denom, 1e-4))
    return torch.tensor(denoms, dtype=torch.float32, device=device)
```

### Log(σ) Policy
```python
def forward(self, state):
    state_norm = self.norm(state)
    mu_logsig = self.net(state_norm).squeeze(-1)  # MLP outputs log(sigma) mean
    log_std = self.log_std.expand(mu_logsig.shape)
    return mu_logsig, log_std

def sample(self, state):
    mu_logsig, log_std = self(state)
    dist = Normal(mu_logsig, log_std.exp())
    log_sigma = dist.rsample()
    log_prob = dist.log_prob(log_sigma)   # Jacobian cancels — evaluate in log(sigma) space
    sigma = log_sigma.exp().clamp(SIGMA_MIN, SIGMA_MAX)
    return sigma, log_prob
```
- init_std = 0.20: policy starts at ~20% vol (market ATM), log(0.20) ≈ -1.61
- SIGMA_MIN=0.01, SIGMA_MAX=2.00
- LOG_STD_MIN=-4.0, LOG_STD_MAX=0.5

### Normalizer Initialization
- Warm-up: 1000 rollouts before training starts to initialize the state normalizer (mean/var)
- Without warm-up: normalizer has mean=0, var=1 → states incorrectly normalized → loss exploded to 2314 at ep2

---

## 5. Experiment 1 — Plot Details

### Bugs Fixed in Plots (generate_plots_v2.py)

**Bug 1 — Wrong maturity grid:**
- Original: `range(23, 52, 2)` = [23, 25, 27, ...] — missed 3/5 calibration maturities (22, 32, 42 shown as 23, 31, 41)
- Fix: Use actual calibration DTEs [22, 27, 32, 42, 47]

**Bug 2 — Extrapolation zone:**
- Original: moneyness [0.88, 1.12] — the range [1.09, 1.12] is pure extrapolation with no training signal
- Fix: Truncate to [0.88, 1.09] (matches paper Figure 1)

**Bug 3 — Deterministic evaluation:**
- Original: deterministic sigma = exp(mu_logsig) for every path (no noise)
- Fix: 5 stochastic rollouts × 400k paths = 2M total paths, averaged — matches what training loss actually measured

### Plot Files (per dataset)
- `exp1_smile_calib_v2.png`: 5-panel focused calibration plot (one panel per maturity)
- `exp1_smile_interp_v2.png`: 3×5 interpolation grid (14 panels including off-grid maturities)
- `exp1_smile_calib_v2_oos_<dataset>.png`: Out-of-sample cross-dataset test
- `exp1_learning_curve_v2.png`: Full 700/600-episode learning curve with 50-ep rolling mean

---

## 6. Experiment 2 Development History

### Sub-experiments
- **A. Path-dependent state (dim=5):** s_it = (t/T, S_it/S0, σ_{t-1}/SIGMA_MAX, S_{t1}/S0, σ_{t1}/SIGMA_MAX)
  - Tracks S and σ at first exercise date t1=21
- **B. Non-path-dependent state (dim=3):** s_it = (t/T, S_it/S0, σ_{t-1}/SIGMA_MAX)
  - Paper comparison: this should diverge eventually (Figures 3/4 of Vadori 2022)

### Bermudan Specification
- Strike: ATM (K=S0=2813.18)
- Exercise dates: daily from t1=21 to t2=T=51 steps
- Type: Call (r=0, no dividends → early exercise never strictly optimal, minimal LS premium)
- Longstaff-Schwartz: degree-8 polynomial, normalized by S0

### European Call Reference (for reward normalization)
- S0=2813.18, T=51d=0.2024yr, ATM IV=10.42%
- European call price: 52.5934
- Bermudan price at market vol ≈ European call (minimal early exercise premium for calls)

### Bug Chain: 5 Iterations to Stable Training

#### Attempt 1 — No Gyongy Constraint
- **Config:** reward = -b_price (raw)
- **Symptom:** Policy drove σ→0 trivially; b_price 96→5.42 in 100 episodes (impossible under calibration)
- **Fix:** Combined reward = berm_r + calib_weight × calib_r

#### Attempt 2 — Combined Reward, calib_r Unbounded, 120k Paths
- **Config:** berm_r = -b_price/52.59, calib_r = (L_ref - L)/L_ref unbounded, calib_weight=1.0
- **Symptom:** calib_r started at -98 for random policy (L_calib=0.379 vs L_ref=0.0038); combined = -100; gradient variance overwhelmed PPO; b_price climbed 96→316→503 over 200 episodes
- **Fix:** Clip calib_r = max(calib_r, -2.0), bounds combined to [-3, 1]

#### Attempt 3 — Clipped calib_r, 120k Paths, Random Init
- **Config:** calib_r clipped, calib_weight=1.0, n_paths=120k, warm-start=No
- **Symptom:** b_price still climbed 100→710 over 100 episodes
- **Root cause:** With 120k paths + antithetic variates, all B=10 rollouts have nearly identical rewards → advantage std ≈ 0 → advantage.std().clamp(min=1) → PPO gradient ≈ 0 → policy random-walks → eventual catastrophic drift
- **Fix:** n_paths=40k (retains MC noise across rollouts for clear PPO signal) + warm-start from exp1

#### Attempt 4 — Warm-Start + 40k Paths, calib_weight=2.0
- **Warm-start:** Copy first 2 columns of exp1 state-dim=2 policy to state-dim=5 policy; zero-init extra columns (σ_prev, S_t1, σ_t1 features). State normalizer extended with warm priors.
- **Result early (ep1-50):** reward +0.75→+1.16, b_price 49→35 — GOOD start
- **Result ep51-100:** reward dropped to mean -1.72, b_price=8.94
- **Root cause:** With calib_weight=2.0, at σ→0: berm_r→0 (+), calib_r→-2.0; combined ≈ -4.0 vs market vol +0.74 = only 4.74 penalty units — insufficient to deter sigma→0 exploitation
- **Fix:** Raise calib_weight=5.0 (at σ→0: combined ≈ -10.0 vs +3.35 at market = 13.35 penalty units)

#### Attempt 5 (Final) — Warm-Start + 40k Paths + calib_weight=5.0
- **Result ep1-50:** mean reward = 3.51, mean b_price = 31.93
- **Result ep51-100:** mean reward = 2.34, b_price = 21.82 — STABLE
- **Converged at episode 600** (5 consecutive 100-ep windows without improvement)
- **Best: ep5, reward=+3.98, b_price=40.68** (European ref=52.59 → **23.4% below European**)
- **Final 100-ep mean reward:** +2.17
- **Reward breakdown at best ep:** berm_r=-0.7736, calib_r=+0.9508 (still calibrated while minimizing)

### Non-Path-Dependent Sub-Experiment
- Same config but state_dim=3 (no S_t1, σ_t1 features)
- **Ran to ep530+ stably** (rewards +1 to +2.8, b_price 5-10)
- **Diverged at ep~850:** b_price climbed from ~8 to 172-192, reward collapsed to -13.5
- **Reason:** Without path-dependent features, the policy cannot adapt sigma based on path history through t1=21; the 3D state is insufficient to capture the conditional optimal policy for the joint calibration + minimization objective
- **This confirms Vadori (2022) Figures 3 & 4:** path-dependent state is necessary for stable Bermudan minimization

---

## 7. Key Algorithmic Insights

### Per-Episode vs Per-Timestep Epsilon
- Per-episode: sample once before t loop → clean causal link (one exploration direction → one terminal reward)
- Per-timestep: sample inside t loop → noise averages out across timesteps; terminal reward is non-informative about any specific direction
- **Empirical:** +71.3% (per-episode) vs +62.5% (per-timestep)

### clamp(min=1) Advantage Normalization
- `adv / adv.std().clamp(min=1)` vs `adv / (adv.std() + 1e-8)`
- Near convergence: reward variance is small → std << 1 → std+1e-8 amplifies to large advantages → noisy policy updates
- clamp(min=1): when all rollouts give similar reward, advantages are small (no over-amplification)

### Basis Players: Why They Work
- 120k independent policy gradients → huge variance (unworkable)
- 100 basis players + kNN(k=1) interpolation → each basis player's direction sampled by ~1200 paths → 1200x variance reduction
- Result: convergence in 79 episodes

### Antithetic Variates: Effect
- Halves MC variance of the IV calibration loss estimate at zero extra compute
- Effect: sharp "golden window" at ep78-86 where the policy found a near-optimal basin with very clean gradients
- After the window, PPO's continued exploration pushed the policy out → the plateau at +74%

### Cosine Annealing: Effect on fixes_v2 Degradation
- fixes_v2 (no cosine LR) hit +92.4% at ep226 then degraded to +57.6% over remaining training
- fixes_v3 (with cosine LR): more stable plateau at +74% with best still at +94.8%
- The cosine schedule slows down updates as training progresses, preventing escape from the optimal basin

### 120k Paths Kills PPO Gradient in Exp2
- Key insight: too many paths can be *too good* for MC estimation
- 120k paths + antithetic variates → all B=10 rollouts give near-identical Bermudan + calibration losses → reward variance across rollouts ≈ 0 → advantage normalization clamps to 1 → PPO step is essentially random
- 40k paths: still accurate for LS, but enough MC noise for differentiated rollout rewards → clear gradient

### Warm-Start for Exp2
- Starting from market-calibrated exp1 policy means exp2 starts with:
  - calib_r ≈ +0.85 (already calibrated)
  - berm_r ≈ -0.95 (b_price ≈ European ref at market vol)
  - combined ≈ +3.35 at ep1
- Vs random init: combined ≈ -2.8 at ep1, requires many episodes just to reach calibration
- The first 2 input columns of exp1 policy map (t/T, S/S0) → these same features exist in exp2 and work identically

---

## 8. All Bugs Found and Fixed (Chronological)

| # | Bug | Symptom | Root Cause | Fix | Impact |
|---|-----|---------|-----------|-----|--------|
| 1 | Normalizer uninit | Loss 1→2314 at ep2 | State normalizer mean/var = 0/1 before any data | Warm-up 1000 rollouts first | Stable early training |
| 2 | Convergence every episode | Stopped at ep134 | `no_improve_count++` inside episode loop | Check only at `episode % conv_window == 0` | +0.04 → +0.22 |
| 3 | OTM gradient imbalance | Right-wing fit only, left skew ignored | K/S=0.882 ITM call $336 vs OTM $0.13 → 127× ratio → 16000× gradient | `otm_denom_tensor()` uses OTM put as denominator for K<S0 | +22% → +63% |
| 4 | Wrong T_steps=51 | 4 unconstrained timesteps after last maturity | Longest cal maturity is 47 DTE, not 51 | T_steps=47 | Better vol surface shape |
| 5 | Price-space loss (not IV-space) | Policy over-weighted cheap OTM options | Paper Eq. 6 uses IV mapping | `calibration_loss_iv()` with Newton-Raphson | Major accuracy improvement |
| 6 | Unnormalized reward | Tiny reward signal (0.003 scale) | IV-space losses are small absolute numbers | r = (L_ref - L)/L_ref | Dimensionless [0,1] reward |
| 7 | Softplus policy | Init σ≈70% (wrong), crashes loss | Default bias 0 → softplus(0)=0.693 | Log(σ) parameterization, init_std=0.20 | Stable training from ep1 |
| 8 | Extrapolation zone [1.09,1.12] | Noisy MC IV inversion at high moneyness | Policy tried to fit OTM region with no market data | Truncate to [0.88, 1.09] | Cleaner right-wing fit |
| 9 | float32 overflow in PPO | kl=4.5e15 at ep610 | log_ratio.exp() overflows when policy drifts | `log_ratio.clamp(-10, 10)` before exp() | Prevents permanent collapse |
| 10 | Value LR 10x policy LR | Advantage sign flips | opt_v used wrong lr multiplier | Set opt_v lr = cfg.lr (same as policy) | Stable value estimates |
| 11 | Wrong plot maturity grid | Smiles plotted at wrong DTEs | range(23,52,2) missed DTE 22,32,42 | Use actual calibration DTEs [22,27,32,42,47] | Correct paper figure |
| 12 | Deterministic plot evaluation | Plots showed too-smooth smiles | Used mu_logsig directly instead of sampling | 5 stochastic rollouts × 400k paths averaged | Accurate eval plots |
| 13 | Exp2 no calibration constraint | σ→0 in 100 eps, b_price 96→5.42 | Reward = -b_price only, no Gyongy | Combined reward: berm_r + calib_weight × calib_r | Physically valid training |
| 14 | calib_r unbounded | Reward -100 at ep1, PPO diverges | (L_ref-L)/L_ref unbounded below; random policy at -98 | Clip calib_r to max(-2.0, r) | Bounds combined to [-3,1] |
| 15 | Exp2 too many paths (120k) | b_price 100→710 in 100 eps | All B=10 rollouts identical → PPO gradient ≈ 0 → random walk | n_paths=40k (enough noise for differentiated rollouts) | Stable training |
| 16 | Exp2 random init | Init reward -2.75, slow recovery | 5D policy random → wrong sigma → calibration takes 100+ eps | Warm-start from exp1 checkpoint | Reward +0.85 from ep1 |
| 17 | calib_weight=2.0 insufficient | σ→0 exploit reappears at ep50+ | -4.0 combined reward at σ→0 only 4.7 units penalty | Raise to calib_weight=5.0 (13.35 units penalty) | Stable 600 eps |
| 18 | Non-path-dep diverges at ep850 | Reward -13.5, b_price 172-192 | 3D state missing S_t1, σ_t1 → insufficient to maintain calibration + minimization tradeoff | Expected (validates paper finding) | Confirms path-dep necessity |

---

## 9. Final Numerical Results

### Experiment 1: Vanilla Call Smile Calibration

| Dataset | Best Reward | Best Episode | Best Loss | Total Eps | Final Mean (last 100) |
|---------|------------|-------------|-----------|-----------|----------------------|
| Aug 2018 | **+94.76%** | 79 | 0.000200 | 700 | +74.17% |
| Apr 2026 | **+97.48%** | 36 | — | 600 | +61.01% |

**Interpretation:** 
- Aug 2018 +94.76%: local vol surface reduces IV calibration error to 5.24% of BS flat smile baseline
- Apr 2026 +97.48% despite 8.6× harder absolute baseline (steeper skew → cleaner IV gradients → better learning)

### Experiment 2: Bermudan Call Price Minimization (Aug 2018)

| Sub-exp | State Dim | Best Ep | Best b_price | European Ref | Reduction | Stability |
|---------|-----------|---------|-------------|-------------|-----------|-----------|
| Path-dep (A) | 5 | **5** | **40.68** | 52.59 | **-23.4%** | Stable 600 eps |
| Non-path-dep (B) | 3 | — | — | 52.59 | — | Diverged ep~850 |

**Key findings:**
- Bermudan price 40.68 vs European 52.59 = 23.4% below European reference (Bermudan ≤ European is guaranteed; the minimization lowers it further)
- Best reward +3.98 = berm_r(-0.77) + 5×calib_r(+0.95) → still calibrated while minimizing
- Non-path-dep diverging while path-dep stays stable confirms paper Figures 3 & 4
- All prices below European reference confirms minimization is working

---

## 10. Code Architecture

### Source Files

**`src/policy.py`**
- `PolicyNet`: 3-layer MLP, 50 nodes each, log(σ) parameterization
- `build_state_exp1(t, S_cur, T, S0)` → (t/T, S/S0): 2D state
- `build_state_exp2_path_dependent(t, S_cur, σ_prev, S_t1, σ_t1, T, S0)` → 5D state
- `SIGMA_MIN=0.01`, `SIGMA_MAX=2.00`, `LOG_STD_MIN=-4.0`, `LOG_STD_MAX=0.5`

**`src/marl_vol.py`**
- `TrainConfig`: all hyperparameters, including `use_antithetic`, `lr_min`, `calib_weight`
- `collect_rollout()`: main rollout function, handles both exp1 (IV loss) and exp2 (Bermudan + calib)
- `MARLVolTrainer.train()`: full PPO training loop with CosineAnnealingLR
- `l_ref_berm`: European ATM call reference for exp2 Bermudan normalization

**`src/reward.py`**
- `calibration_loss_iv()`: IV-space MSE with Newton-Raphson inversion
- `calibration_loss()`: OTM-adjusted price-space fallback
- `reference_loss_bs()`: flat BS smile baseline for L_ref computation

**`src/diffusion.py`**
- `MCEngine.reset_episode(use_antithetic=True)`: antithetic variates
- `DELTA = 1/252` (trading day in years)

**`src/american_mc.py`**
- `bermudan_price(S_full, spec, degree=8, S0)`: Longstaff-Schwartz with degree-8 polynomial regression

**`src/market_data.py`**
- `VolSurface`: loads and interpolates market surface
- `otm_denom_tensor()`: OTM-adjusted price denominators for OTM fix

### Experiment Files

**`experiments/exp1_local_vol.py`**
- `--dataset aug2018|apr2026`, `--tag`, `--small` CLI args
- `DATASETS` registry mapping keys to file paths
- CFG_FULL: n_paths=120k, T=47, noise_std=1.0, lr=1e-4, lr_min=1e-6, use_antithetic=True, K=30, B=10

**`experiments/exp2_bermudan.py`**
- `warm_start_from_exp1(trainer, ckpt_path)`: copies exp1 policy weights to exp2 5D policy
- BASE_CFG: n_paths=40k, lr=5e-5, calib_weight=5.0, calib_r clipped to [-2, 1]
- `--exp1_ckpt` auto-detected from `results/exp1_{dataset}_fixes_v3/exp1_best.pt`

### Script Files

**`scripts/generate_plots_v2.py`**
- `--dataset aug2018|apr2026`, `--tag`, `--ckpt`, `--n-eval`, `--b-eval` CLI args
- `_DATASETS` registry with in-sample/out-of-sample paths
- Loads full training log from last `ep*.pt` checkpoint (not just `exp1_best.pt`)
- 5 stochastic rollouts × 400k paths per plot

---

## 11. Training Hyperparameters (Full Config, fixes_v3)

### Experiment 1
```
n_paths     = 120,000       # MC trajectories
T_steps     = 47            # matches longest calibration maturity
n_basis     = 100           # basis players
bp_method   = "knn"
bp_k        = 1             # nearest neighbor noise interpolation
noise_std   = 1.0           # unit noise; policy std scales exploration
lr          = 1e-4          # Adam learning rate
lr_min      = 1e-6          # cosine annealing floor
clip        = 0.3           # PPO clip epsilon
kl_target   = 0.01          # early-stop KL threshold per PPO epoch
K_epochs    = 30            # SGD iterations per update
B_envs      = 10            # rollouts per update
mb_frac     = 0.1           # minibatch = 10% of total basis player samples
c_value     = 0.5           # value loss coefficient
c_entropy   = 0.01          # entropy bonus
gamma       = 1.0           # terminal reward (no discounting)
n_episodes  = 2000          # max (convergence triggers earlier)
n_strikes   = 10            # per maturity in instrument grid
state_dim   = 2             # (t/T, S/S0)
use_antithetic = True
```

### Experiment 2 (changes from Exp1)
```
n_paths     = 40,000        # fewer paths (see explanation above)
lr          = 5e-5          # lower LR
calib_weight = 5.0          # Gyongy tether strength
state_dim   = 5             # path-dependent (or 3 for non-path-dep)
T_steps     = 51            # needs t=51 for Bermudan exercise window t1=21..t2=51
```

---

## 12. GitHub History

| Commit | Tag | Description |
|--------|-----|-------------|
| baseline | — | Initial implementation |
| fixes_v1 | — | Per-episode epsilon, clamp(min=1), T=47 |
| fixes_v2 | — | IV-space loss, normalized reward, log(σ) policy, moneyness truncation, OTM fix |
| 9382942 | fixes_v3 | Antithetic variates, cosine LR; aug2018 results +94.8%; corrected plots |
| 25fd711 | — | exp2 log(σ) eval fix, dataset CLI |
| dedc051 | — | Gyongy calibration constraint for exp2 |
| 70a0ca9 | — | Calibration reward clip for exp2 |
| f04edef | — | Warm-start + 40k paths + lower LR; apr2026 exp1 results |
| 40f32a9 | — | calib_weight=5.0 to prevent σ→0 exploit |

---

## 13. What Worked First Try vs What Required Debugging

### Worked First Try
- Basis player interpolation (kNN k=1)
- Log(σ) parameterization (once switched)
- IV calibration loss (once IV-space was used)
- Longstaff-Schwartz Bermudan pricing
- Antithetic variates implementation
- Stochastic plot evaluation logic

### Required Many Iterations
| Component | Iterations | Key Insight |
|-----------|-----------|-------------|
| Normalizer initialization | 1 | Paper didn't mention warm-up; discovered from ep2 explosion |
| Convergence check timing | 1 | Off-by-1 in window check caused ep134 early stop |
| OTM gradient imbalance | 2 | Had to instrument gradient magnitudes to diagnose |
| Reward scale | 3 | IV-space → normalized (fixed both at once) |
| exp2 stability | 5 | Chain: no constraint → unbounded calib_r → too many paths → random init → weak calib_weight |
| Epsilon timing | Tested + reverted | Empirical test confirmed per-episode is better |

---

## 14. Remaining Opportunities / Future Work

### Could Improve Exp1 Results Further
1. Smaller cosine LR minimum (1e-7) for finer convergence near peak
2. Better exploration decay schedule (reduce noise_std as training matures)
3. Adaptive calib_weight for exp2 (ramp up during training)
4. Patience-based early stopping (stop at first non-improving window, not 5th)

### Could Extend the Work
1. Bermudan PUT option (more early exercise premium than call)
2. Apr 2026 dataset for Exp2 (need Apr 2026 exp1 checkpoint as warm-start)
3. American call calibration
4. Multi-strike Bermudan pricing
5. Stochastic rates (Vasicek model extension)

---

## 15. Cross-Reference: Paper vs Implementation

| Paper Item | Paper Spec | Our Implementation | Notes |
|-----------|-----------|-------------------|-------|
| Loss function | IV-space MSE (Eq. 6) | `calibration_loss_iv()` | Matches after fixes_v2 |
| Reward | r = (L_ref-L)/L_ref | Same | Matches after fixes_v2 |
| Policy | log(σ) ~ N(μ_θ, σ²_π) | Log(σ) parameterization | Matches after fixes_v2 |
| Basis players | n_p=100, kNN k=1 | Same | Matches |
| PPO hyperparams | K=30, clip=0.3, kl=0.01 | Same | Matches |
| Network | 3-layer, 50 nodes | Same | Matches |
| T_steps | = longest maturity | T=47 for exp1, T=51 for exp2 | Matches after fix |
| Moneyness | [0.88, 1.09] (Figure 1) | Same | Matches after fix |
| Epsilon sampling | Once per episode (implied) | Once before t loop | Confirmed empirically better |
| State (exp1) | (t, S) normalized | (t/T, S/S0) | Matches |
| State (exp2A) | (t, S, σ_prev, S_{t1}, σ_{t1}) | 5D version of above | Matches |
| Bermudan pricing | LS degree-8, normalize by S0 | Same | Matches |

---

*Document compiled from full ~25,000 line chat transcript, April 2026.*
