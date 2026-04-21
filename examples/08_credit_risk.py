"""
Example 08: Credit Risk Analysis

Demonstrates MeridianAlgo's credit risk module:
- Merton structural model: equity as call option on firm assets
- Default probability term structure
- CDS pricing and hazard rate bootstrapping
- Portfolio expected loss and credit VaR
- Z-spread and DV01 computation
"""

import numpy as np
import pandas as pd

from meridianalgo.credit import (
    CreditDefaultSwap,
    CreditRiskAnalyzer,
    MertonModel,
    ZSpreadCalculator,
)


# ---------------------------------------------------------------------------
# 1. Merton Structural Model
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. MERTON STRUCTURAL MODEL")
print("=" * 60)

# A leveraged firm: $500M equity, $800M debt, 35% equity vol
model = MertonModel(
    equity_value=500e6,
    equity_volatility=0.35,
    debt_face_value=800e6,
    time_to_maturity=1.0,
    risk_free_rate=0.05,
)
result = model.calibrate()

print(f"Input equity value:     ${500e6/1e6:.0f}M")
print(f"Input debt face value:  ${800e6/1e6:.0f}M")
print(f"Input equity vol:       35.00%")
print()
print(f"Implied asset value:    ${result['asset_value']/1e6:.1f}M")
print(f"Implied asset vol:      {result['asset_volatility']:.2%}")
print(f"Leverage ratio:         {result['leverage_ratio']:.2%}")
print(f"Distance to default:    {result['distance_to_default']:.4f} sigma")
print(f"Default probability:    {result['default_probability']:.4%}")
print(f"Expected recovery rate: {result['expected_recovery_rate']:.4%}")

# Compare three firms: low, medium, and high leverage
print("\n--- Leverage Comparison ---")
firms = [
    ("Low leverage",    800e6, 0.25,  200e6),
    ("Medium leverage", 500e6, 0.35,  800e6),
    ("High leverage",   200e6, 0.50, 1500e6),
]
for name, equity, ev, debt in firms:
    m = MertonModel(equity, ev, debt, time_to_maturity=1.0, risk_free_rate=0.05)
    r = m.calibrate()
    print(f"  {name:20s}  DD={r['distance_to_default']:6.3f}  PD={r['default_probability']:.4%}")


# ---------------------------------------------------------------------------
# 2. Default Probability Term Structure
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("2. DEFAULT PROBABILITY TERM STRUCTURE")
print("=" * 60)

model_ts = MertonModel(
    equity_value=400e6,
    equity_volatility=0.40,
    debt_face_value=900e6,
    time_to_maturity=1.0,
    risk_free_rate=0.05,
)
horizons = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
term_struct = model_ts.default_probability_term_structure(horizons)

print(f"{'Horizon':>10}  {'PD':>10}  {'Annualized':>12}")
for t, pd_val in term_struct.items():
    ann_pd = 1 - (1 - pd_val) ** (1 / t) if t > 0 else 0
    print(f"{t:>10.2f}  {pd_val:>10.4%}  {ann_pd:>12.4%}")


# ---------------------------------------------------------------------------
# 3. CDS Pricing
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("3. CDS PRICING")
print("=" * 60)

# Constant hazard rate CDS
cds = CreditDefaultSwap(
    hazard_rate=0.02,
    recovery_rate=0.40,
    risk_free_rate=0.05,
    maturity=5.0,
    payment_frequency=4,
)
r = cds.price()

print(f"Hazard rate:          {cds.hazard_rate:.4f}")
print(f"Recovery rate:        {cds.recovery_rate:.2%}")
print(f"Fair spread:          {r.fair_spread * 10000:.2f} bps")
print(f"Risky annuity (DV01): {r.risky_annuity:.6f}")
print(f"5yr survival prob:    {r.survival_probability:.6f}")

# Spread sensitivity to hazard rate
print("\n--- Spread vs Hazard Rate ---")
print(f"{'Hazard Rate':>14}  {'5yr Spread (bps)':>18}  {'5yr Survival':>14}")
for h in [0.005, 0.010, 0.020, 0.040, 0.080, 0.150]:
    c = CreditDefaultSwap(h, recovery_rate=0.40, maturity=5.0)
    r = c.price()
    print(f"  {h:12.3%}  {r.fair_spread * 10000:>16.2f}  {r.survival_probability:>14.6f}")

# Recover hazard rate from market spread
print("\n--- Bootstrap Hazard Rate from Market Spread ---")
market_spread = 0.0180
cds_from_spread = CreditDefaultSwap.from_spread(
    spread=market_spread,
    recovery_rate=0.40,
    risk_free_rate=0.05,
    maturity=5.0,
)
print(f"Market spread:    {market_spread * 10000:.1f} bps")
print(f"Implied hazard:   {cds_from_spread.hazard_rate:.6f}")
check = cds_from_spread.price().fair_spread
print(f"Verified spread:  {check * 10000:.2f} bps (roundtrip error: {abs(check - market_spread)*1e6:.1f} micro-bps)")

# Bootstrap full hazard curve
print("\n--- CDS Curve Bootstrap ---")
maturities = [1.0, 3.0, 5.0, 7.0, 10.0]
market_spreads = [0.0080, 0.0120, 0.0150, 0.0175, 0.0210]
hazard_curve = CreditDefaultSwap.bootstrap_hazard_curve(
    maturities, market_spreads, recovery_rate=0.40, risk_free_rate=0.05
)
print(f"{'Maturity':>10}  {'Market Spread':>16}  {'Hazard Rate':>14}")
for mat, spread, hazard in zip(maturities, market_spreads, hazard_curve.values):
    print(f"  {mat:8.1f}y  {spread * 10000:>14.1f}bps  {hazard:>14.6f}")


# ---------------------------------------------------------------------------
# 4. Portfolio Credit Risk: EL, UL, Credit VaR
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("4. PORTFOLIO CREDIT RISK")
print("=" * 60)

analyzer = CreditRiskAnalyzer()

# Loan portfolio
portfolio = pd.DataFrame({
    "obligor": ["Corp A", "Corp B", "Corp C", "Corp D", "Corp E",
                 "Corp F", "Corp G", "Corp H"],
    "rating":  ["BBB",   "BB",    "B",     "BBB",   "A",
                 "BB",    "CCC",   "A"],
    "pd":  [0.0020, 0.0080, 0.0250, 0.0020, 0.0005,
            0.0080, 0.1000, 0.0005],
    "lgd": [0.45,   0.50,   0.60,   0.40,   0.35,
            0.50,   0.70,   0.35],
    "ead": [5e6,    3e6,    1e6,    8e6,    10e6,
            2e6,    0.5e6,  15e6],
})

portfolio["el"] = portfolio.apply(
    lambda r: analyzer.expected_loss(r["pd"], r["lgd"], r["ead"]), axis=1
)
portfolio["ul"] = portfolio.apply(
    lambda r: analyzer.unexpected_loss(r["pd"], r["lgd"], r["ead"]), axis=1
)
portfolio["credit_var_99"] = portfolio.apply(
    lambda r: analyzer.credit_var(r["pd"], r["lgd"], r["ead"], confidence=0.999), axis=1
)

print(portfolio[["obligor", "rating", "pd", "lgd", "ead", "el", "credit_var_99"]]
      .to_string(index=False, float_format=lambda x: f"{x:,.0f}" if x > 100 else f"{x:.4f}"))

result = analyzer.portfolio_expected_loss(portfolio[["pd", "lgd", "ead"]])
print(f"\nPortfolio Summary:")
print(f"  Total EAD:             ${portfolio['ead'].sum()/1e6:.1f}M")
print(f"  Total Expected Loss:   ${result['total_el']/1e3:.1f}K ({result['el_rate']:.3%} of EAD)")
print(f"  Total Unexpected Loss: ${result['total_ul']/1e3:.1f}K")
print(f"  Herfindahl Index:      {result['herfindahl_index']:.4f}")
print(f"  Top-10 Concentration:  {result['top10_concentration']:.2%}")


# ---------------------------------------------------------------------------
# 5. Z-Spread and DV01
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("5. Z-SPREAD AND DV01")
print("=" * 60)

# 5% coupon bond, 5-year maturity
cash_flows = [50, 50, 50, 50, 1050]
times = [1.0, 2.0, 3.0, 4.0, 5.0]
risk_free_rates = [0.035, 0.038, 0.040, 0.042, 0.044]

calc = ZSpreadCalculator(cash_flows, times, risk_free_rates)

par_price = calc.theoretical_price(0.0)
print(f"Risk-free par price: ${par_price:.4f}")

market_prices = [par_price - 5, par_price - 2, par_price, par_price + 2, par_price + 5]
print(f"\n{'Market Price':>14}  {'Z-Spread (bps)':>16}  {'DV01':>10}")
for price in market_prices:
    z = calc.z_spread(price)
    dv01 = calc.dv01(z)
    print(f"  ${price:10.4f}  {z * 10000:>14.2f}  {dv01:>10.4f}")

print(f"\nDV01 at z=0: {calc.dv01():.4f}  (approx ${calc.dv01() * 1000:,.2f} per $1M notional per bp)")
