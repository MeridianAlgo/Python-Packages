"""
Example 10: Monte Carlo Simulation

Demonstrates MeridianAlgo's Monte Carlo engine:
- Geometric Brownian Motion paths and option pricing
- Antithetic variates variance reduction
- Correlated multi-asset portfolio simulation
- Heston stochastic volatility model
- Merton jump-diffusion model
- CIR interest rate model and bond pricing
- Unified MonteCarloEngine interface
- Portfolio VaR from simulation
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from meridianalgo.monte_carlo import (
    CIRModel,
    GeometricBrownianMotion,
    HestonModel,
    JumpDiffusionModel,
    MonteCarloEngine,
    QuasiRandomSampler,
)


def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes analytical call price for comparison."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# 1. Geometric Brownian Motion
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. GEOMETRIC BROWNIAN MOTION")
print("=" * 60)

gbm = GeometricBrownianMotion(mu=0.08, sigma=0.20, seed=42)
result = gbm.simulate(S0=100, T=1.0, n_paths=100_000, n_steps=252, antithetic=True)

expected_mean = 100 * np.exp(0.08 * 1.0)
print(f"Parameters: mu=8%, sigma=20%, S0=100, T=1yr")
print(f"Paths: {result.n_paths:,}  Steps: {result.n_steps}")
print()
print(f"Theoretical E[S_T]:    ${expected_mean:.4f}")
print(f"Simulated mean:        ${result.mean:.4f}")
print(f"Simulation error:      {abs(result.mean - expected_mean) / expected_mean:.4%}")
print()
print(f"Simulated std:         ${result.std:.4f}")
print(f"5th percentile:        ${result.percentile_5:.4f}")
print(f"25th percentile:       ${result.percentile_25:.4f}")
print(f"Median:                ${result.median:.4f}")
print(f"75th percentile:       ${result.percentile_75:.4f}")
print(f"95th percentile:       ${result.percentile_95:.4f}")
print(f"95% CI for mean:       (${result.confidence_interval_95[0]:.4f}, "
      f"${result.confidence_interval_95[1]:.4f})")

# Option pricing comparison
print("\n--- European Option Pricing (MC vs Black-Scholes) ---")
S0, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20

gbm_rn = GeometricBrownianMotion(mu=r, sigma=sigma, seed=0)
mc_call = gbm_rn.call_price(S0=S0, K=K, T=T, r=r, n_paths=200_000)
bs_call = black_scholes_call(S0, K, T, r, sigma)
bs_put = black_scholes_put(S0, K, T, r, sigma)

print(f"S={S0}, K={K}, T={T}yr, r={r:.0%}, sigma={sigma:.0%}")
print(f"  BS Call:  ${bs_call:.4f}")
print(f"  MC Call:  ${mc_call['price']:.4f}  (std err: ${mc_call['std_error']:.4f})")
print(f"  Error:    {abs(mc_call['price'] - bs_call) / bs_call:.4%}")
print(f"  95% CI:   (${mc_call['confidence_interval'][0]:.4f}, ${mc_call['confidence_interval'][1]:.4f})")
print(f"  BS Put:   ${bs_put:.4f}  (put-call parity check: "
      f"{abs((mc_call['price'] - bs_put) - (S0 - K * np.exp(-r * T))):.6f})")


# ---------------------------------------------------------------------------
# 2. Correlated Multi-Asset Portfolio
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("2. CORRELATED MULTI-ASSET PORTFOLIO")
print("=" * 60)

S0 = np.array([100.0, 80.0, 120.0, 50.0])
weights = np.array([0.35, 0.25, 0.25, 0.15])
corr = np.array([
    [1.00,  0.65,  0.55,  0.10],
    [0.65,  1.00,  0.50,  0.15],
    [0.55,  0.50,  1.00,  0.05],
    [0.10,  0.15,  0.05,  1.00],
])
asset_names = ["AAPL", "MSFT", "GOOGL", "GLD"]

gbm_port = GeometricBrownianMotion(mu=0.09, sigma=0.22, seed=42)
result_port = gbm_port.simulate_portfolio(S0, weights, corr, T=1.0, n_paths=50_000)

print(f"Portfolio: {dict(zip(asset_names, weights))}")
print(f"Correlated simulation: {result_port.n_paths:,} paths")
print()
print(f"Portfolio 1yr return distribution:")
port_returns = result_port.terminal_values - 1
print(f"  Expected return:  {result_port.mean - 1:.2%}")
print(f"  Std deviation:    {result_port.std:.4f}")
print(f"  5th pct (VaR):    {np.percentile(port_returns, 5):.2%}")
print(f"  Median return:    {result_port.median - 1:.2%}")
print(f"  95th pct:         {np.percentile(port_returns, 95):.2%}")


# ---------------------------------------------------------------------------
# 3. Heston Stochastic Volatility
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("3. HESTON STOCHASTIC VOLATILITY")
print("=" * 60)

heston = HestonModel(
    mu=0.05,
    v0=0.04,    # initial variance = 20% vol
    kappa=2.0,  # mean reversion speed
    theta=0.04, # long-run variance = 20% vol
    xi=0.30,    # vol of vol
    rho=-0.70,  # leverage correlation
    seed=42,
)

result_h = heston.simulate(S0=100, T=1.0, n_paths=50_000, n_steps=252, antithetic=True)

print(f"Heston parameters: kappa={heston.kappa}, theta={heston.theta} ({np.sqrt(heston.theta):.0%} vol)")
print(f"                   xi={heston.xi}, rho={heston.rho}")
print()
print(f"Simulated terminal distribution:")
print(f"  Mean:          ${result_h.mean:.4f}")
print(f"  Std:           ${result_h.std:.4f}")
print(f"  5th pct:       ${result_h.percentile_5:.4f}")
print(f"  95th pct:      ${result_h.percentile_95:.4f}")

# Heston vs GBM: Heston has heavier tails due to stochastic vol
gbm_comp = GeometricBrownianMotion(mu=0.05, sigma=np.sqrt(0.04), seed=42)
result_gbm_comp = gbm_comp.simulate(S0=100, T=1.0, n_paths=50_000)

print(f"\nTail comparison (same mean/vol parameters):")
print(f"  {'Model':12}  {'5th pct':>10}  {'95th pct':>10}  {'IQR':>10}  {'Tail ratio':>12}")
heston_iqr = result_h.percentile_75 - result_h.percentile_25
gbm_iqr = result_gbm_comp.percentile_75 - result_gbm_comp.percentile_25
print(f"  {'Heston':12}  ${result_h.percentile_5:>8.4f}  ${result_h.percentile_95:>8.4f}"
      f"  ${heston_iqr:>8.4f}  {'(stoch vol)':>12}")
print(f"  {'GBM':12}  ${result_gbm_comp.percentile_5:>8.4f}  ${result_gbm_comp.percentile_95:>8.4f}"
      f"  ${gbm_iqr:>8.4f}  {'(const vol)':>12}")


# ---------------------------------------------------------------------------
# 4. Merton Jump Diffusion
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("4. MERTON JUMP DIFFUSION")
print("=" * 60)

jdm = JumpDiffusionModel(
    mu=0.05,
    sigma=0.15,      # diffusion vol
    lam=0.10,        # 0.1 jumps/year on average
    mu_jump=-0.05,   # average log jump = -5%
    sigma_jump=0.08, # jump size std = 8%
    seed=42,
)

result_j = jdm.simulate(S0=100, T=1.0, n_paths=100_000, n_steps=252)

print(f"Parameters: mu={jdm.mu:.0%}, sigma={jdm.sigma:.0%}")
print(f"            lambda={jdm.lam} jumps/yr, mu_J={jdm.mu_jump:.0%}, sigma_J={jdm.sigma_jump:.0%}")
print()
print(f"Terminal distribution ({result_j.n_paths:,} paths):")
print(f"  Mean:     ${result_j.mean:.4f}  (expected: ${100 * np.exp(jdm.mu * 1.0):.4f})")
print(f"  Std:      ${result_j.std:.4f}")
print(f"  5th pct:  ${result_j.percentile_5:.4f}")
print(f"  95th pct: ${result_j.percentile_95:.4f}")

# Compare with GBM (same total vol approximately)
gbm_same_vol = GeometricBrownianMotion(mu=0.05, sigma=0.18, seed=42)
res_gbm = gbm_same_vol.simulate(S0=100, T=1.0, n_paths=100_000)
print(f"\nLeft tail comparison (GBM vs Jump-Diffusion at similar vol):")
print(f"  GBM  1st pct: ${np.percentile(res_gbm.terminal_values, 1):.4f}")
print(f"  JDM  1st pct: ${np.percentile(result_j.terminal_values, 1):.4f}")
print(f"  Jump model has heavier left tail: "
      f"{np.percentile(result_j.terminal_values, 1) < np.percentile(res_gbm.terminal_values, 1)}")


# ---------------------------------------------------------------------------
# 5. CIR Interest Rate Model
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("5. CIR INTEREST RATE MODEL")
print("=" * 60)

cir = CIRModel(
    r0=0.03,    # initial rate 3%
    kappa=0.80, # mean reversion speed
    theta=0.04, # long-run mean 4%
    sigma=0.05, # vol parameter
    seed=42,
)

result_cir = cir.simulate(T=10.0, n_paths=20_000, n_steps=2520)

print(f"CIR parameters: r0={cir.r0:.2%}, kappa={cir.kappa}, theta={cir.theta:.2%}, sigma={cir.sigma}")
print(f"10-year terminal rate distribution ({result_cir.n_paths:,} paths):")
print(f"  Mean:    {result_cir.mean:.4%}  (theoretical long-run: {cir.theta:.4%})")
print(f"  Std:     {result_cir.std:.4%}")
print(f"  5th pct: {result_cir.percentile_5:.4%}")
print(f"  Median:  {result_cir.median:.4%}")
print(f"  95th pct:{result_cir.percentile_95:.4%}")
print(f"  Min rate: {result_cir.terminal_values.min():.6%}  (CIR guarantees non-negative rates)")

# Analytical zero-coupon bond prices
print(f"\nAnalytical CIR zero-coupon bond prices (r0={cir.r0:.2%}):")
print(f"  {'Maturity':>10}  {'Bond Price':>12}  {'Implied Yield':>14}")
maturities = [1, 2, 3, 5, 7, 10, 20, 30]
for T_bond in maturities:
    price = cir.zero_coupon_bond_price(t=0, T=T_bond, r=cir.r0)
    implied_yield = -np.log(price) / T_bond
    print(f"  {T_bond:>10}y  {price:>12.6f}  {implied_yield:>14.4%}")


# ---------------------------------------------------------------------------
# 6. Unified MonteCarloEngine
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("6. UNIFIED MONTECARLO ENGINE")
print("=" * 60)

# GBM engine
engine_gbm = MonteCarloEngine(model="gbm", seed=0)
engine_gbm.configure(mu=0.05, sigma=0.20)
engine_gbm.simulate(S0=100, T=1.0, n_paths=100_000)

call_result = engine_gbm.price_option(K=100, r=0.05, T=1.0, option_type="call")
put_result = engine_gbm.price_option(K=100, r=0.05, T=1.0, option_type="put")
bs_call_check = black_scholes_call(100, 100, 1.0, 0.05, 0.20)

print(f"GBM Engine — ATM options (S=100, K=100, T=1yr, r=5%, sigma=20%):")
print(f"  MC Call:   ${call_result['price']:.4f}  (BS: ${bs_call_check:.4f})")
print(f"  MC Put:    ${put_result['price']:.4f}")
print(f"  Call SE:   ${call_result['std_error']:.4f}")

var_result = engine_gbm.portfolio_var(initial_value=100, confidence=0.95)
print(f"\n1-year 95% VaR from GBM simulation:")
print(f"  VaR:      ${var_result['var']:.4f}  ({var_result['var'] / 100:.2%} of portfolio)")
print(f"  CVaR:     ${var_result['cvar']:.4f}")
print(f"  Mean P&L: ${var_result['mean_pnl']:.4f}")
print(f"  Worst P&L:${var_result['worst_pnl']:.4f}")

# Heston engine
engine_h = MonteCarloEngine(model="heston", seed=0)
engine_h.configure(mu=0.05, v0=0.04, kappa=2.0, theta=0.04, xi=0.30, rho=-0.70)
engine_h.simulate(S0=100, T=1.0, n_paths=100_000)
heston_call = engine_h.price_option(K=100, r=0.05, T=1.0, option_type="call")

print(f"\nHeston Engine — ATM call (same parameters, stochastic vol):")
print(f"  MC Call:   ${heston_call['price']:.4f}")
print(f"  Diff vs BS: ${heston_call['price'] - bs_call_check:+.4f}  "
      f"({'premium' if heston_call['price'] > bs_call_check else 'discount'} from stoch vol)")

# Quasi-random sampler
print("\n--- Quasi-Random Sampling vs Pseudo-Random ---")
sampler = QuasiRandomSampler(dimensions=1, seed=42)
quasi_samples = sampler.normal(10_000)
pseudo_rng = np.random.default_rng(42)
pseudo_samples = pseudo_rng.standard_normal(10_000)

print(f"  Quasi-random mean: {np.mean(quasi_samples):.6f}  (target: 0.000000)")
print(f"  Pseudo-random mean:{np.mean(pseudo_samples):.6f}")
print(f"  Quasi-random std:  {np.std(quasi_samples):.6f}  (target: 1.000000)")
print(f"  Pseudo-random std: {np.std(pseudo_samples):.6f}")
