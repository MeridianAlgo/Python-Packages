"""
Example 07: Derivatives Pricing and Options Analytics

Covers:
- Black-Scholes-Merton pricing for calls and puts
- Put-call parity verification
- Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Implied volatility computation
- Monte Carlo pricing for exotics
- Volatility surface construction
"""

import os

import numpy as np
import pandas as pd

os.environ["MERIDIANALGO_QUIET"] = "1"

from meridianalgo.derivatives import (
    BlackScholes,
    ExoticOptions,
    GreeksCalculator,
    ImpliedVolatility,
    MonteCarloPricer,
    OptionChain,
    OptionsPricer,
    VolatilitySurface,
)

# ============================================================================
# 1. Black-Scholes pricing
# ============================================================================

print("=" * 60)
print("1. Black-Scholes-Merton Pricing")
print("=" * 60)

S, K, T, r, sigma = 100.0, 105.0, 0.25, 0.05, 0.20

call_result = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma, option_type="call")
put_result = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma, option_type="put")

print(f"\nUnderlying: ${S:.0f}, Strike: ${K:.0f}, T: {T:.2f}yr, r: {r:.0%}, σ: {sigma:.0%}")
print(f"\nCall:  price=${call_result['price']:.4f}  delta={call_result['delta']:.4f}")
print(f"Put:   price={put_result['price']:.4f}   delta={put_result['delta']:.4f}")

# Verify put-call parity: C - P = S - K * exp(-rT)
parity_lhs = call_result["price"] - put_result["price"]
parity_rhs = S - K * np.exp(-r * T)
print(f"\nPut-Call Parity: C-P={parity_lhs:.4f}, S-Ke^{{-rT}}={parity_rhs:.4f}")
print(f"Parity holds: {abs(parity_lhs - parity_rhs) < 0.01}")

# ============================================================================
# 2. Full Greeks profile
# ============================================================================

print("\n" + "=" * 60)
print("2. Greeks Profile")
print("=" * 60)

pricer = OptionsPricer()
greeks = pricer.calculate_greeks(S=S, K=K, T=T, r=r, sigma=sigma)
if isinstance(greeks, dict):
    for greek, value in greeks.items():
        if isinstance(value, dict):
            print(f"\n{greek}:")
            for opt_type, v in value.items():
                print(f"  {opt_type}: {v:.4f}")
        else:
            print(f"  {greek}: {value:.4f}")

# ============================================================================
# 3. Implied volatility
# ============================================================================

print("\n" + "=" * 60)
print("3. Implied Volatility")
print("=" * 60)

market_prices = [2.0, 3.0, 5.0, 8.0, 12.0]
strikes = [110, 107, 105, 103, 100]

print(f"\n{'Strike':>8} {'Market':>8} {'IV':>8}")
print("-" * 28)
for strike, price in zip(strikes, market_prices):
    try:
        iv = ImpliedVolatility(
            market_price=price, S=S, K=strike, T=T, r=r, option_type="call"
        )
        if isinstance(iv, dict):
            iv_val = iv.get("implied_volatility", iv.get("iv", 0))
        else:
            iv_val = float(iv)
        print(f"{strike:>8} {price:>8.2f} {iv_val:>8.2%}")
    except Exception:
        print(f"{strike:>8} {price:>8.2f} {'N/A':>8}")

# ============================================================================
# 4. Monte Carlo pricing
# ============================================================================

print("\n" + "=" * 60)
print("4. Monte Carlo Pricing")
print("=" * 60)

try:
    mc_result = MonteCarloPricer(
        S=S, K=K, T=T, r=r, sigma=sigma, option_type="call", n_simulations=50000
    )
    bs_price = call_result["price"]
    if isinstance(mc_result, dict):
        mc_price = mc_result.get("price", mc_result.get("call_price", 0))
    else:
        mc_price = float(mc_result)
    print(f"\nBlack-Scholes price:  {bs_price:.4f}")
    print(f"Monte Carlo price:    {mc_price:.4f}")
    print(f"Difference:           {abs(mc_price - bs_price):.4f}")
except Exception as e:
    print(f"Monte Carlo skipped: {e}")

# ============================================================================
# 5. Delta-hedging P&L simulation
# ============================================================================

print("\n" + "=" * 60)
print("5. Delta Hedging Simulation")
print("=" * 60)

np.random.seed(42)
n_steps = 20
dt = T / n_steps
paths = [S]
for _ in range(n_steps):
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * np.random.randn()
    paths.append(paths[-1] * np.exp(drift + diffusion))

print(f"\nSimulated path: S_0={paths[0]:.2f}  S_T={paths[-1]:.2f}")
print(f"Call payoff at expiry: ${max(paths[-1] - K, 0):.2f}")

initial_delta = call_result["delta"]
print(f"Initial delta (hedge ratio): {initial_delta:.4f}")
print(f"Initial hedge: short {initial_delta:.4f} shares per option")
