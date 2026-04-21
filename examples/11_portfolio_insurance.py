"""
Example 11: Portfolio Insurance (CPPI and TIPP)

Demonstrates MeridianAlgo's portfolio insurance module:
- Standard CPPI with configurable multiplier and floor
- Time-Invariant Portfolio Protection (TIPP): ratcheting floor
- Sensitivity analysis across multiplier / floor combinations
- Comparison: CPPI vs buy-and-hold vs 100% cash in a bear market
- Floor protection analysis
"""

import numpy as np
import pandas as pd

from meridianalgo.portfolio.insurance import CPPI, TimeInvariantCPPI


def generate_market(n: int, daily_drift: float, daily_vol: float, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal(n) * daily_vol + daily_drift
    dates = pd.date_range("2018-01-02", periods=n, freq="B")
    return pd.Series(returns, index=dates, name="equity_returns")


# ---------------------------------------------------------------------------
# 1. Standard CPPI — Bull Market
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. CPPI — BULL MARKET (2018-2020 simulated)")
print("=" * 60)

bull_returns = generate_market(n=500, daily_drift=0.0004, daily_vol=0.010, seed=42)

cppi_bull = CPPI(
    multiplier=3.0,
    floor_pct=0.80,
    safe_rate=0.04,
    rebalance_frequency=1,
    max_leverage=1.0,
)
result_bull = cppi_bull.run(bull_returns, initial_value=1_000_000, trading_days=252)

# Benchmark: buy-and-hold
bah_value = 1_000_000 * (1 + bull_returns).cumprod()

print(f"Initial portfolio:     $1,000,000")
print(f"Floor protection:      {cppi_bull.floor_pct:.0%}")
print(f"Multiplier:            {cppi_bull.multiplier}x")
print()
print(f"CPPI Final Value:      ${result_bull.portfolio_value.iloc[-1]:>12,.0f}")
print(f"Buy-Hold Final Value:  ${bah_value.iloc[-1]:>12,.0f}")
print()
print(f"CPPI  Total Return:    {result_bull.total_return:.2%}")
print(f"B&H   Total Return:    {(bah_value.iloc[-1] / 1_000_000) - 1:.2%}")
print()
print(f"CPPI  Ann. Return:     {result_bull.annualized_return:.2%}")
print(f"CPPI  Ann. Volatility: {result_bull.annualized_volatility:.2%}")
print(f"CPPI  Max Drawdown:    {result_bull.max_drawdown:.2%}")
print(f"CPPI  Floor Breaches:  {result_bull.floor_breaches}")
print(f"Final Floor Value:     ${result_bull.floor_value.iloc[-1]:>12,.0f}")
print(f"Final Cushion:         ${result_bull.cushion.iloc[-1]:>12,.0f}")


# ---------------------------------------------------------------------------
# 2. CPPI — Bear Market Protection
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("2. CPPI — BEAR MARKET (GFC-style simulated)")
print("=" * 60)

bear_returns = generate_market(n=252, daily_drift=-0.0008, daily_vol=0.018, seed=99)
# Add a shock event
shock = pd.Series(
    np.where(np.arange(252) == 60, -0.08, 0.0),
    index=bear_returns.index,
)
bear_returns_shocked = bear_returns + shock

initial_value = 1_000_000
floor_pct = 0.80

cppi_bear = CPPI(multiplier=3.0, floor_pct=floor_pct, safe_rate=0.04)
result_bear = cppi_bear.run(bear_returns_shocked, initial_value=initial_value)

bah_bear = initial_value * (1 + bear_returns_shocked).cumprod()

print(f"1-year bear market scenario (shocked with one-day -8% crash)")
print(f"Initial capital: $1,000,000  Floor: {floor_pct:.0%}")
print()
print(f"{'Strategy':20}  {'Final Value':>14}  {'Return':>8}  {'Max DD':>8}  {'Floor Breach':>14}")
print(f"{'CPPI (3x, 80%)':20}  ${result_bear.portfolio_value.iloc[-1]:>12,.0f}  "
      f"{result_bear.total_return:>8.2%}  {result_bear.max_drawdown:>8.2%}  "
      f"{'YES' if result_bear.floor_breaches > 0 else 'NO':>14}")

bah_final = bah_bear.iloc[-1]
bah_return = (bah_final / initial_value) - 1
bah_drawdown = float((bah_bear / bah_bear.cummax() - 1).min())
print(f"{'Buy-and-Hold':20}  ${bah_final:>12,.0f}  {bah_return:>8.2%}  {bah_drawdown:>8.2%}  {'N/A':>14}")

cash_value = initial_value * (1 + 0.04 / 252) ** 252
print(f"{'100% Cash (4%)':20}  ${cash_value:>12,.0f}  {(cash_value/initial_value)-1:>8.2%}  {'0.00%':>8}  {'N/A':>14}")

protected_floor = initial_value * floor_pct
print(f"\nCPPI floor protection: ${protected_floor:,.0f}")
print(f"CPPI terminal value:   ${result_bear.portfolio_value.iloc[-1]:,.0f} "
      f"({'ABOVE' if result_bear.portfolio_value.iloc[-1] >= protected_floor else 'BELOW'} floor)")


# ---------------------------------------------------------------------------
# 3. Sensitivity Analysis
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("3. SENSITIVITY ANALYSIS — MULTIPLIER x FLOOR")
print("=" * 60)

mixed_returns = generate_market(n=504, daily_drift=0.0002, daily_vol=0.013, seed=7)
cppi_base = CPPI(multiplier=3.0, floor_pct=0.80, safe_rate=0.04)

sensitivity = cppi_base.sensitivity_analysis(
    mixed_returns,
    multipliers=[1.0, 2.0, 3.0, 4.0, 5.0],
    floor_pcts=[0.70, 0.80, 0.90],
    initial_value=1_000_000,
)

print(f"{'Multiplier':>12}  {'Floor':>8}  {'Return':>10}  {'Ann Vol':>10}  {'Max DD':>10}  {'Breaches':>10}")
for _, row in sensitivity.iterrows():
    print(f"  {row['multiplier']:>10.1f}  {row['floor_pct']:>8.0%}  {row['total_return']:>10.2%}  "
          f"{row['ann_vol']:>10.2%}  {row['max_drawdown']:>10.2%}  {int(row['floor_breaches']):>10}")


# ---------------------------------------------------------------------------
# 4. TIPP (Time-Invariant Portfolio Protection)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("4. TIME-INVARIANT PORTFOLIO PROTECTION (TIPP)")
print("=" * 60)

bull_then_bear = pd.concat([
    generate_market(252, daily_drift=0.0005, daily_vol=0.010, seed=1),
    generate_market(252, daily_drift=-0.0006, daily_vol=0.016, seed=2),
])
bull_then_bear.index = pd.date_range("2020-01-02", periods=504, freq="B")

initial = 1_000_000

cppi_std = CPPI(multiplier=3.0, floor_pct=0.80, safe_rate=0.04)
r_cppi = cppi_std.run(bull_then_bear, initial_value=initial)

tipp = TimeInvariantCPPI(multiplier=3.0, floor_pct=0.80, safe_rate=0.04)
r_tipp = tipp.run(bull_then_bear, initial_value=initial)

bah_btb = initial * (1 + bull_then_bear).cumprod()

print(f"Bull market (yr 1) then bear market (yr 2)")
print(f"Initial: ${initial:,}  Multiplier: 3x  Floor: 80%")
print()
print(f"{'Strategy':25}  {'Final Value':>14}  {'Return':>9}  {'Max DD':>9}  {'Final Floor':>14}")
print(f"  {'CPPI (fixed 80% floor)':23}  ${r_cppi.portfolio_value.iloc[-1]:>12,.0f}  "
      f"{r_cppi.total_return:>9.2%}  {r_cppi.max_drawdown:>9.2%}  "
      f"${r_cppi.floor_value.iloc[-1]:>12,.0f}")
print(f"  {'TIPP (ratcheting floor)':23}  ${r_tipp.portfolio_value.iloc[-1]:>12,.0f}  "
      f"{r_tipp.total_return:>9.2%}  {r_tipp.max_drawdown:>9.2%}  "
      f"${r_tipp.floor_value.iloc[-1]:>12,.0f}")
print(f"  {'Buy-and-Hold':23}  ${bah_btb.iloc[-1]:>12,.0f}  "
      f"{(bah_btb.iloc[-1]/initial)-1:>9.2%}  "
      f"{float((bah_btb/bah_btb.cummax()-1).min()):>9.2%}  {'N/A':>14}")

print(f"\nKey insight: TIPP floor ratchets up from ${initial*0.80:,.0f} to "
      f"${r_tipp.floor_value.max():,.0f} during the bull phase,")
print(f"providing stronger downside protection than standard CPPI in the subsequent bear market.")

# Show risky allocation over time
print(f"\nRisky allocation dynamics (CPPI):")
rw = r_cppi.risky_weight
print(f"  Bull phase avg weight: {rw.iloc[:252].mean():.2%}")
print(f"  Bear phase avg weight: {rw.iloc[252:].mean():.2%}")
print(f"  Min weight (max fear): {rw.min():.2%}")
print(f"  Max weight (max greed):{rw.max():.2%}")
