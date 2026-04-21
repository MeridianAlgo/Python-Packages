"""
Example 13: Scenario Analysis and Stress Testing

Demonstrates MeridianAlgo's scenario module:
- Historical stress scenarios (GFC, COVID, dot-com, 2022 rate shock)
- Custom macro-factor shock scenarios
- Reverse stress testing: find the shock that causes a target loss
- Correlated multi-asset scenario generation
- Correlation stress (crises correlate assets toward 1)
- Portfolio-level P&L and severity classification
"""

import numpy as np
import pandas as pd

from meridianalgo.risk.scenario import (
    HISTORICAL_SCENARIOS,
    CorrelationScenario,
    ScenarioAnalyzer,
)


# ---------------------------------------------------------------------------
# Portfolio setup: 60/40-style with factor sensitivities
# ---------------------------------------------------------------------------

# Portfolio weights
weights = pd.Series({
    "US_Equity":        0.35,
    "Intl_Equity":      0.15,
    "US_Bonds":         0.25,
    "TIPS":             0.05,
    "Gold":             0.05,
    "Real_Estate":      0.05,
    "Commodities":      0.05,
    "Cash":             0.05,
})

# Factor sensitivity matrix: how each asset responds to each macro factor
# Rows = assets, Columns = macro factors
factor_sensitivities = pd.DataFrame({
    "equity":       [1.00,  1.20,  0.00,  0.10,  0.10,  0.80,  0.20,  0.00],
    "bonds":        [-0.10, -0.10,  1.00,  0.80,  0.20, -0.20, -0.05,  0.00],
    "usd":          [-0.10, -0.25,  0.05,  0.00, -0.25, -0.10, -0.30,  0.00],
    "commodities":  [0.05,  0.05, -0.05, -0.10,  1.00, -0.05,  1.00,  0.00],
    "real_estate":  [0.30,  0.20, -0.10, -0.05,  0.00,  1.00,  0.10,  0.00],
    "gold":         [0.00,  0.00,  0.10,  0.20,  1.00,  0.00,  0.20,  0.00],
    "technology":   [0.80,  0.70,  0.00,  0.00,  0.00,  0.30,  0.10,  0.00],
    "energy":       [0.20,  0.20, -0.05,  0.20,  0.80,  0.00,  0.90,  0.00],
}, index=weights.index)

analyzer = ScenarioAnalyzer(
    portfolio_weights=weights,
    factor_sensitivities=factor_sensitivities,
    portfolio_value=10_000_000,
)

portfolio_value = 10_000_000


# ---------------------------------------------------------------------------
# 1. Historical Stress Scenarios
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. HISTORICAL STRESS SCENARIOS")
print("=" * 60)

results = analyzer.run_all_historical()
summary = analyzer.summary_table(results)

print(f"Portfolio: ${portfolio_value/1e6:.0f}M  [{', '.join(f'{k}:{v:.0%}' for k, v in weights.items())}]")
print()
print(f"{'Scenario':35}  {'Return':>10}  {'P&L':>14}  {'Severity':>12}")
for _, row in summary.iterrows():
    name = row["scenario"].replace("_", " ")
    print(f"  {name:33}  {row['portfolio_return']:>10.2%}  "
          f"${row['portfolio_pnl']:>12,.0f}  {row['severity']:>12}")

print(f"\nWorst scenario:  {summary.iloc[0]['scenario']}")
print(f"Best scenario:   {summary.iloc[-1]['scenario']}")
worst_loss = summary.iloc[0]['portfolio_pnl']
print(f"Worst P&L:       ${worst_loss:,.0f}")


# ---------------------------------------------------------------------------
# 2. Key Historical Scenarios Deep Dive
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("2. KEY SCENARIO DEEP DIVE")
print("=" * 60)

key_scenarios = ["gfc_2008_2009", "covid_crash_march_2020", "rate_shock_2022"]
for scenario_name in key_scenarios:
    if scenario_name not in results:
        continue
    r = results[scenario_name]
    label = scenario_name.replace("_", " ").title()
    print(f"\n{label}:")
    print(f"  Portfolio Return: {r.portfolio_return:.2%}  P&L: ${r.portfolio_pnl:,.0f}  [{r.severity}]")
    print(f"  Worst asset: {r.worst_asset}  Best asset: {r.best_asset}")
    print(f"  Asset-level returns:")
    for asset, ret in r.asset_returns.sort_values().items():
        bar = "+" * max(0, int(ret * 200)) if ret >= 0 else "-" * max(0, int(-ret * 200))
        print(f"    {asset:20s}: {ret:>8.2%}  {bar}")


# ---------------------------------------------------------------------------
# 3. Custom Macro Shock Scenarios
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("3. CUSTOM MACRO SHOCK SCENARIOS")
print("=" * 60)

custom_scenarios = [
    ("Equity crash -30%",     -0.30,  0.05,  0.02,  0.05,  0.00),
    ("Rate shock +200bps",     0.00, -0.10,  0.00,  0.00,  0.00),
    ("Stagflation",           -0.15, -0.05,  0.00,  0.20,  0.00),
    ("USD surge +15%",        -0.10,  0.02,  0.15, -0.10,  0.00),
    ("Commodity rally +25%",   0.05, -0.02, -0.05,  0.25,  0.00),
    ("Risk-off flight",       -0.20,  0.08,  0.10, -0.15,  0.00),
]

print(f"{'Scenario':25}  {'Eq Shock':>10}  {'Bnd Shock':>10}  {'Return':>10}  {'P&L':>14}")
for name, eq, bnd, fx, comm, gold_s in custom_scenarios:
    r = analyzer.run_custom_scenario(
        name=name,
        equity_shock=eq,
        bond_shock=bnd,
        fx_shock=fx,
        commodity_shock=comm,
    )
    print(f"  {name:23}  {eq:>10.2%}  {bnd:>10.2%}  {r.portfolio_return:>10.2%}  ${r.portfolio_pnl:>12,.0f}")


# ---------------------------------------------------------------------------
# 4. Reverse Stress Testing
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("4. REVERSE STRESS TESTING")
print("=" * 60)

print(f"Find the equity shock that causes target losses:")
print()
print(f"  {'Target Loss':>14}  {'Required Equity Shock':>22}  {'Interpretation':>20}")
targets = [-0.05, -0.10, -0.15, -0.20, -0.25]
for target in targets:
    shock = analyzer.reverse_stress_test(target_loss=target, factor="equity")
    if shock <= -0.80:
        interp = "extreme tail event"
    elif shock <= -0.40:
        interp = "severe crash (GFC-level)"
    elif shock <= -0.20:
        interp = "significant correction"
    elif shock <= -0.10:
        interp = "moderate drawdown"
    else:
        interp = "mild pullback"
    print(f"  {target:>14.2%}  {shock:>22.2%}  {interp:>20}")

# Combined shock reverse test
print(f"\nWith bonds rallying +5% and gold +10%, equity shock for -10% portfolio loss:")
equity_shock_combined = analyzer.reverse_stress_test(
    target_loss=-0.10,
    factor="equity",
    other_shocks={"bonds": 0.05, "gold": 0.10},
)
print(f"  Required equity shock: {equity_shock_combined:.2%}")
print(f"  (Bond/gold hedge reduces required shock magnitude)")


# ---------------------------------------------------------------------------
# 5. Correlated Scenario Generation
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("5. CORRELATED SCENARIO GENERATION")
print("=" * 60)

assets = ["US_Equity", "Intl_Equity", "US_Bonds", "Gold"]
port_weights_simple = pd.Series([0.40, 0.20, 0.30, 0.10], index=assets)

# Expected daily returns and vols
mean_rets = pd.Series([0.0004, 0.0004, 0.0001, 0.0002], index=assets)
daily_vols = pd.Series([0.012, 0.014, 0.004, 0.008], index=assets)

# Normal correlation matrix
normal_corr = pd.DataFrame([
    [1.00,  0.85,  -0.20,  0.05],
    [0.85,  1.00,  -0.18,  0.08],
    [-0.20, -0.18,  1.00,  0.15],
    [0.05,   0.08,  0.15,  1.00],
], index=assets, columns=assets)

gen = CorrelationScenario(mean_rets, normal_corr, daily_vols, port_weights_simple)

# Normal regime
normal = gen.generate(n_scenarios=100_000, horizon_days=1, stress_correlation=False)

# Stressed regime (correlations move toward 1 during crises)
stressed = gen.generate(n_scenarios=100_000, horizon_days=1, stress_correlation=True, stress_factor=0.60)

print(f"1-day scenario generation ({100_000:,} scenarios)")
print(f"Portfolio: {dict(zip(assets, port_weights_simple))}")
print()
print(f"{'Metric':25}  {'Normal Regime':>16}  {'Stress Regime':>16}")
print(f"  {'Mean P&L':23}  {normal['mean']:>16.4%}  {stressed['mean']:>16.4%}")
print(f"  {'Std':23}  {normal['std']:>16.4%}  {stressed['std']:>16.4%}")
print(f"  {'VaR 95%':23}  {normal['var_95']:>16.4%}  {stressed['var_95']:>16.4%}")
print(f"  {'VaR 99%':23}  {normal['var_99']:>16.4%}  {stressed['var_99']:>16.4%}")
print(f"  {'CVaR 95%':23}  {normal['cvar_95']:>16.4%}  {stressed['cvar_95']:>16.4%}")

# Scale to portfolio value
normal_var_dollar = abs(normal['var_99']) * portfolio_value
stressed_var_dollar = abs(stressed['var_99']) * portfolio_value
print()
print(f"1-day 99% VaR on ${portfolio_value/1e6:.0f}M portfolio:")
print(f"  Normal:   ${normal_var_dollar:>12,.0f}")
print(f"  Stressed: ${stressed_var_dollar:>12,.0f}")
print(f"  Stress multiplier: {stressed_var_dollar / normal_var_dollar:.2f}x")

# Multi-day VaR (10-day, Basel requirement)
ten_day = gen.generate(n_scenarios=100_000, horizon_days=10, stress_correlation=True, stress_factor=0.60)
print(f"\n10-day 99% VaR (Basel-style):")
ten_day_var = abs(ten_day['var_99']) * portfolio_value
print(f"  Stressed: ${ten_day_var:>12,.0f}  ({abs(ten_day['var_99']):.3%} of portfolio)")
sqrt_10_approx = abs(stressed['var_99']) * np.sqrt(10) * portfolio_value
print(f"  sqrt(10) approx: ${sqrt_10_approx:>12,.0f}")
print(f"  Ratio (actual/sqrt10): {ten_day_var / sqrt_10_approx:.4f}")


# ---------------------------------------------------------------------------
# 6. Full Risk Report Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("6. SUMMARY RISK REPORT")
print("=" * 60)

print(f"Portfolio Value: ${portfolio_value/1e6:.0f}M")
print(f"Composition:     60% equity (US+Intl), 30% bonds+TIPS, 10% alternatives")
print()
print(f"Scenario Stress Results:")
print(f"  GFC 2008-09:            {results.get('gfc_2008_2009', type('', (), {'portfolio_return': float('nan')})()).portfolio_return:.2%}")
print(f"  COVID Mar 2020:         {results.get('covid_crash_march_2020', type('', (), {'portfolio_return': float('nan')})()).portfolio_return:.2%}")
print(f"  2022 Rate Shock:        {results.get('rate_shock_2022', type('', (), {'portfolio_return': float('nan')})()).portfolio_return:.2%}")
print()
print(f"Reverse Stress (break-even equity shock for -10% loss): "
      f"{analyzer.reverse_stress_test(-0.10, 'equity'):.2%}")
print()
print(f"Monte Carlo VaR (100k scenarios, 1-day, 99%, stress corr):")
print(f"  ${abs(stressed['var_99']) * portfolio_value:,.0f}  ({abs(stressed['var_99']):.3%} of NAV)")
