# MeridianAlgo

[![PyPI version](https://img.shields.io/pypi/v/meridianalgo.svg?style=flat-square&color=blue)](https://pypi.org/project/meridianalgo/)
[![Python versions](https://img.shields.io/pypi/pyversions/meridianalgo.svg?style=flat-square)](https://pypi.org/project/meridianalgo/)
[![License](https://img.shields.io/github/license/MeridianAlgo/Python-Packages.svg?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/MeridianAlgo/Python-Packages/ci.yml?branch=main&style=flat-square&label=tests)](https://github.com/MeridianAlgo/Python-Packages/actions)

Institutional-grade Python library for quantitative finance and algorithmic trading. Covers the complete quant stack: portfolio optimization, risk management, derivatives pricing, backtesting, credit risk, volatility modeling, Monte Carlo simulation, portfolio insurance, machine learning signal generation, execution algorithms, fixed income analytics, and market microstructure analysis.

## Installation

```bash
pip install meridianalgo
```

```bash
# Selective extras
pip install meridianalgo[ml]           # scikit-learn, torch, statsmodels, hmmlearn
pip install meridianalgo[optimization] # cvxpy, cvxopt for convex portfolio optimization
pip install meridianalgo[volatility]   # arch (GARCH family models)
pip install meridianalgo[data]         # lxml, beautifulsoup4, polygon-api-client
pip install meridianalgo[distributed]  # ray, dask for parallel computing
pip install meridianalgo[all]          # all optional dependencies
```

**Requirements:** Python >= 3.10

---

## Module Overview

| Module | Key Classes / Functions |
|--------|------------------------|
| `portfolio` | `MeanVariance`, `HierarchicalRiskParity`, `RiskParity`, `BlackLitterman`, `KellyCriterion` |
| `portfolio.insurance` | `CPPI`, `TimeInvariantCPPI` |
| `risk` | `RiskAnalyzer`, `VaRCalculator`, `CVaRCalculator`, `StressTesting`, `RiskBudgeting` |
| `risk.scenario` | `ScenarioAnalyzer`, `CorrelationScenario` |
| `credit` | `MertonModel`, `CreditDefaultSwap`, `CreditRiskAnalyzer`, `ZSpreadCalculator` |
| `volatility` | `GARCHModel`, `RealizedVolatility`, `VolatilityForecaster`, `VolatilityTermStructure` |
| `monte_carlo` | `GeometricBrownianMotion`, `HestonModel`, `JumpDiffusionModel`, `CIRModel`, `MonteCarloEngine` |
| `derivatives` | `BlackScholes`, `GreeksCalculator`, `ImpliedVolatility`, `MonteCarloPricer`, `OptionChain` |
| `fixed_income` | `BondPricer`, `YieldCurve`, `CreditSpreadAnalyzer` |
| `backtesting` | `BacktestEngine`, `Strategy`, `Backtest` |
| `ml` | `LSTMPredictor`, `WalkForwardValidator`, `FeatureEngineer`, `ModelSelector` |
| `execution` | `VWAP`, `TWAP`, `POV`, `ImplementationShortfall` |
| `analytics` | `PerformanceAnalyzer`, `BenchmarkAnalytics`, `ActiveShare`, `BrinsonAttribution` |
| `quant` | `StatisticalArbitrage`, `RegimeDetector`, `MarketMicrostructure` |
| `signals` | `RSI`, `MACD`, `BollingerBands`, 50+ indicators |
| `liquidity` | `OrderBook`, `SpreadAnalyzer`, `MarketImpact` |
| `factors` | `FamaFrenchModel`, `FactorExposure` |

---

## Examples

### Portfolio Optimization

```python
import meridianalgo as ma
from meridianalgo.portfolio import PortfolioOptimizer

prices = ma.get_market_data(["AAPL", "MSFT", "GOOGL", "JPM", "GLD"], start="2020-01-01")
returns = ma.calculate_returns(prices)

opt = PortfolioOptimizer(returns)

hrp = opt.optimize(method="hrp")
min_var = opt.optimize(method="min_variance")
max_sharpe = opt.optimize(method="max_sharpe")
risk_parity = opt.optimize(method="risk_parity")

print("HRP weights:")
print(hrp.sort_values(ascending=False))
print(f"\nMin Variance vol:  {min_var.volatility:.4f}")
print(f"Max Sharpe ratio:  {max_sharpe.sharpe_ratio:.4f}")
```

### Kelly Criterion Position Sizing

```python
from meridianalgo import KellyCriterion

kc = KellyCriterion(fraction=0.5)  # half-Kelly

# Single asset (discrete binary bet)
f = kc.single_asset(win_prob=0.55, win_loss_ratio=1.0)
print(f"Kelly fraction: {f:.2%}")

# Multi-asset continuous Kelly from return history
weights = kc.optimize(returns)
print("Kelly weights:")
print(weights.sort_values(ascending=False))

# From moments
f_moments = kc.from_moments(expected_return=0.12, volatility=0.18)
print(f"Kelly (moments): {f_moments:.2%}")

# Expected long-run growth rate
g = kc.growth_rate(expected_return=0.12, volatility=0.18)
print(f"Expected growth: {g:.2%}")
```

### Performance and Risk Metrics

```python
import meridianalgo as ma

# Top-level convenience functions
sharpe = ma.calculate_sharpe_ratio(returns["AAPL"])
sortino = ma.calculate_sortino_ratio(returns["AAPL"])
calmar = ma.calculate_calmar_ratio(returns["AAPL"])
max_dd = ma.calculate_max_drawdown(returns["AAPL"])
cvar_95 = ma.calculate_expected_shortfall(returns["AAPL"])

print(f"Sharpe:       {sharpe:.3f}")
print(f"Sortino:      {sortino:.3f}")
print(f"Calmar:       {calmar:.3f}")
print(f"Max Drawdown: {max_dd:.2%}")
print(f"95% CVaR:     {cvar_95:.2%}")

# Full performance report
from meridianalgo import PerformanceAnalyzer

analyzer = PerformanceAnalyzer(returns["AAPL"], benchmark=returns["SPY"], risk_free_rate=0.05)
metrics = analyzer.calculate_all_metrics()
print(metrics)
```

### Benchmark-Relative Analytics

```python
from meridianalgo import BenchmarkAnalytics, ActiveShare, BrinsonAttribution

# Manager analytics
analytics = BenchmarkAnalytics(
    portfolio_returns=portfolio_daily_returns,
    benchmark_returns=spy_daily_returns,
    risk_free_rate=0.05,
)
m = analytics.active_metrics()

print(f"Active Return:     {m.active_return:.2%}")
print(f"Tracking Error:    {m.tracking_error:.2%}")
print(f"Information Ratio: {m.information_ratio:.3f}")
print(f"Up Capture:        {m.up_capture:.2%}")
print(f"Down Capture:      {m.down_capture:.2%}")
print(f"Batting Average:   {m.batting_average:.2%}")
print(f"Beta:              {m.beta:.3f}")
print(f"Alpha (ann.):      {m.alpha_annualized:.2%}")

# Active share from holdings
active_share = ActiveShare.compute(portfolio_weights, benchmark_weights)
print(f"Active Share: {active_share:.2%}")
print(f"Category:     {ActiveShare.categorize(active_share)}")

# Brinson-Hood-Beebower attribution
attribution = BrinsonAttribution(
    portfolio_weights=sector_weights_portfolio,
    benchmark_weights=sector_weights_benchmark,
    portfolio_returns=sector_returns_portfolio,
    benchmark_returns=sector_returns_benchmark,
)
result = attribution.compute()
print("\nAllocation effect by sector:")
print(result.allocation_effect.sort_values())
print("\nSelection effect by sector:")
print(result.selection_effect.sort_values())
print(f"\nTotal active return: {result.total_active_return:.4f}")
```

### Value at Risk and Stress Testing

```python
from meridianalgo import RiskAnalyzer, VaRCalculator

# VaR/CVaR
risk = RiskAnalyzer(returns["portfolio"])
var_95 = risk.value_at_risk(confidence=0.95, method="historical")
var_99 = risk.value_at_risk(confidence=0.99, method="cornish_fisher")
cvar_95 = risk.conditional_var(confidence=0.95)

print(f"Historical VaR 95%:        {var_95:.2%}")
print(f"Cornish-Fisher VaR 99%:    {var_99:.2%}")
print(f"CVaR 95%:                  {cvar_95:.2%}")

# Scenario stress testing
from meridianalgo import ScenarioAnalyzer

analyzer = ScenarioAnalyzer(
    portfolio_weights=weights,
    factor_sensitivities=factor_betas,
    portfolio_value=10_000_000,
)

# Built-in historical scenarios
results = analyzer.run_all_historical()
summary = analyzer.summary_table(results)
print(summary[["scenario", "portfolio_return", "portfolio_pnl", "severity"]].head(10))

# Reverse stress test: what equity shock causes -10% loss?
shock = analyzer.reverse_stress_test(target_loss=-0.10, factor="equity")
print(f"Equity shock for -10% loss: {shock:.2%}")

# Correlated scenario generation
from meridianalgo import CorrelationScenario

gen = CorrelationScenario(mean_returns, correlation_matrix, volatilities, weights)
scenarios = gen.generate(n_scenarios=100_000, stress_correlation=True, stress_factor=0.5)
print(f"Stressed 99% VaR:  {scenarios['var_99']:.2%}")
print(f"Stressed 99% CVaR: {scenarios['cvar_95']:.2%}")
```

### Credit Risk

```python
from meridianalgo import MertonModel, CreditDefaultSwap, CreditRiskAnalyzer, ZSpreadCalculator

# Merton structural model — equity as a call option on firm assets
model = MertonModel(
    equity_value=500e6,      # $500M market cap
    equity_volatility=0.35,  # 35% equity vol
    debt_face_value=800e6,   # $800M total debt
    time_to_maturity=1.0,
    risk_free_rate=0.05,
)
result = model.calibrate()
print(f"Asset Value:          ${result['asset_value']/1e6:.1f}M")
print(f"Distance to Default:  {result['distance_to_default']:.4f}")
print(f"Default Probability:  {result['default_probability']:.2%}")

# Default probability term structure
ts = model.default_probability_term_structure([0.5, 1.0, 2.0, 3.0, 5.0])
print("\nDefault Probability Term Structure:")
for t, pd in ts.items():
    print(f"  {t:4.1f}y: {pd:.2%}")

# CDS pricing
cds = CreditDefaultSwap(hazard_rate=0.02, recovery_rate=0.40, maturity=5.0)
r = cds.price()
print(f"\nCDS Fair Spread:  {r.fair_spread * 10000:.1f} bps")
print(f"Survival Prob:    {r.survival_probability:.4f}")

# Bootstrap CDS curve from market spreads
curve = CreditDefaultSwap.bootstrap_hazard_curve(
    maturities=[1, 3, 5, 7, 10],
    spreads=[0.0080, 0.0120, 0.0150, 0.0170, 0.0200],
)
print("\nHazard Rate Curve:")
print(curve)

# Portfolio expected loss
import pandas as pd
exposures = pd.DataFrame({
    "pd":  [0.010, 0.025, 0.050, 0.005],
    "lgd": [0.45,  0.40,  0.60,  0.35],
    "ead": [2e6,   1.5e6, 0.5e6, 3e6],
})
analyzer = CreditRiskAnalyzer()
el = analyzer.portfolio_expected_loss(exposures)
print(f"\nPortfolio Expected Loss: ${el['total_el']:,.0f}")
print(f"EL Rate:                 {el['el_rate']:.2%}")
print(f"Herfindahl Index:        {el['herfindahl_index']:.4f}")

# Z-spread
calc = ZSpreadCalculator(
    cash_flows=[6, 6, 6, 6, 106],
    times=[1, 2, 3, 4, 5],
    risk_free_rates=[0.035, 0.038, 0.040, 0.042, 0.044],
)
z = calc.z_spread(market_price=97.5)
print(f"\nZ-Spread: {z * 10000:.1f} bps")
print(f"DV01:     {calc.dv01():.4f}")
```

### Volatility Modeling

```python
from meridianalgo import (
    GARCHModel,
    RealizedVolatility,
    VolatilityForecaster,
    VolatilityTermStructure,
    VolatilityRegimeDetector,
)

# Realized volatility estimators from OHLCV data
rv = RealizedVolatility(ohlcv_data)
estimators = rv.all_estimators(window=21)

print("21-day Annualized Volatility Estimates:")
print(f"  Close-to-Close:  {estimators['close_to_close_vol'].iloc[-1]:.2%}")
print(f"  Parkinson:       {estimators['parkinson_vol'].iloc[-1]:.2%}")
print(f"  Garman-Klass:    {estimators['garman_klass_vol'].iloc[-1]:.2%}")
print(f"  Rogers-Satchell: {estimators['rogers_satchell_vol'].iloc[-1]:.2%}")
print(f"  Yang-Zhang:      {estimators['yang_zhang_vol'].iloc[-1]:.2%}")

# GARCH conditional volatility (uses arch library if installed)
garch = GARCHModel(daily_returns, model_type="garch", p=1, q=1)
result = garch.fit()
print(f"\nGARCH(1,1):")
print(f"  Persistence:   {result.persistence:.4f}")
print(f"  Half-life:     {result.half_life:.1f} days")
print(f"  AIC:           {result.aic:.2f}")

# 10-day volatility forecast
forecast = garch.forecast(horizon=10)
print("\n10-day vol forecast:")
print(forecast.point_forecast)

# Volatility term structure
vts = VolatilityTermStructure(daily_returns)
term_struct = vts.build(horizons=[5, 10, 21, 63, 126, 252])
print(f"\nVol term structure slope: {vts.slope():.4f}")
print(f"VIX-style index: {vts.vix_style_index():.2f}")

# Regime classification
detector = VolatilityRegimeDetector(daily_returns)
regimes = detector.classify()
print("\nRegime distribution:")
print(regimes.value_counts())
print("\nRegime statistics:")
print(detector.regime_statistics())

# HAR-RV model
har = VolatilityForecaster(realized_variance_series)
params = har.fit()
print(f"\nHAR-RV R-squared: {params['r_squared']:.4f}")
forecast_har = har.forecast(horizon=5)
print("5-day HAR-RV forecast:", forecast_har.values)
```

### Monte Carlo Simulation

```python
from meridianalgo import (
    GeometricBrownianMotion,
    HestonModel,
    JumpDiffusionModel,
    CIRModel,
    MonteCarloEngine,
)

# Geometric Brownian Motion
gbm = GeometricBrownianMotion(mu=0.08, sigma=0.20)
result = gbm.simulate(S0=100, T=1.0, n_paths=100_000, n_steps=252, antithetic=True)

print("GBM Simulation (1yr, 100k paths):")
print(f"  Mean:         ${result.mean:.2f}")
print(f"  Std:          ${result.std:.2f}")
print(f"  5th pct:      ${result.percentile_5:.2f}")
print(f"  95th pct:     ${result.percentile_95:.2f}")

# Option pricing via MC
call = gbm.call_price(S0=100, K=105, T=0.25, r=0.05, n_paths=200_000)
print(f"\nEuropean Call (K=105, T=3m):")
print(f"  MC Price:     ${call['price']:.4f}")
print(f"  Std Error:    ${call['std_error']:.4f}")
print(f"  95% CI:       (${call['confidence_interval'][0]:.4f}, ${call['confidence_interval'][1]:.4f})")

# Heston stochastic volatility
heston = HestonModel(
    mu=0.05, v0=0.04, kappa=2.0, theta=0.04, xi=0.30, rho=-0.70
)
result_h = heston.simulate(S0=100, T=1.0, n_paths=50_000, n_steps=252)
print(f"\nHeston Mean: ${result_h.mean:.2f}, Std: ${result_h.std:.2f}")

# Merton jump diffusion
jdm = JumpDiffusionModel(
    mu=0.05, sigma=0.15, lam=0.10, mu_jump=-0.03, sigma_jump=0.06
)
result_j = jdm.simulate(S0=100, T=1.0, n_paths=50_000, n_steps=252)
print(f"Merton  Mean: ${result_j.mean:.2f}, Std: ${result_j.std:.2f}")

# CIR interest rate paths
cir = CIRModel(r0=0.03, kappa=0.80, theta=0.04, sigma=0.06)
rates = cir.simulate(T=10.0, n_paths=10_000, n_steps=2520)
print(f"\nCIR 10yr rate mean: {rates.mean:.4f}")

# Unified engine with variance reduction
engine = MonteCarloEngine(model="heston")
engine.configure(mu=0.05, v0=0.04, kappa=2.0, theta=0.04, xi=0.30, rho=-0.7)
engine.simulate(S0=100, T=1.0, n_paths=100_000)
put = engine.price_option(K=95, r=0.05, T=1.0, option_type="put")
print(f"\nHeston Put (K=95): ${put['price']:.4f}")

var_result = engine.portfolio_var(initial_value=100, confidence=0.95)
print(f"MC 95% VaR: ${var_result['var']:.2f}")
```

### Portfolio Insurance (CPPI)

```python
from meridianalgo import CPPI, TimeInvariantCPPI

# Standard CPPI
cppi = CPPI(
    multiplier=3.0,         # 3x leverage on cushion
    floor_pct=0.80,         # 80% capital protection
    safe_rate=0.04,         # 4% money market rate
    rebalance_frequency=1,  # daily rebalancing
)
result = cppi.run(equity_returns, initial_value=1_000_000)

print(f"Total Return:    {result.total_return:.2%}")
print(f"Ann. Return:     {result.annualized_return:.2%}")
print(f"Ann. Volatility: {result.annualized_volatility:.2%}")
print(f"Max Drawdown:    {result.max_drawdown:.2%}")
print(f"Floor Breaches:  {result.floor_breaches}")
print(f"Final Portfolio: ${result.portfolio_value.iloc[-1]:,.0f}")
print(f"Final Floor:     ${result.floor_value.iloc[-1]:,.0f}")

# Sensitivity analysis across multiplier/floor combinations
sensitivity = cppi.sensitivity_analysis(
    equity_returns,
    multipliers=[1.0, 2.0, 3.0, 4.0, 5.0],
    floor_pcts=[0.70, 0.80, 0.90],
)
print("\nSensitivity Analysis:")
print(sensitivity.to_string(index=False))

# TIPP (floor ratchets up with portfolio peaks)
tipp = TimeInvariantCPPI(multiplier=3.0, floor_pct=0.80)
result_tipp = tipp.run(equity_returns, initial_value=1_000_000)
print(f"\nTIPP Final Floor: ${result_tipp.floor_value.iloc[-1]:,.0f}")
```

### Derivatives Pricing

```python
from meridianalgo import BlackScholes, GreeksCalculator, ImpliedVolatility
from meridianalgo.derivatives import OptionsPricer

pricer = OptionsPricer()

# Black-Scholes pricing with full Greeks
call = BlackScholes(S=100, K=105, T=0.25, r=0.05, sigma=0.20, option_type="call")
put = BlackScholes(S=100, K=105, T=0.25, r=0.05, sigma=0.20, option_type="put")

print(f"Call price: ${call['price']:.4f}  Put price: ${put['price']:.4f}")
print(f"Delta:      {call['delta']:.4f}   Delta:     {put['delta']:.4f}")
print(f"Gamma:      {call['gamma']:.4f}   Gamma:     {put['gamma']:.4f}")
print(f"Theta:      {call['theta']:.4f}   Theta:     {put['theta']:.4f}")
print(f"Vega:       {call['vega']:.4f}    Vega:      {put['vega']:.4f}")
print(f"Rho:        {call['rho']:.4f}     Rho:       {put['rho']:.4f}")

# Verify put-call parity
parity = call['price'] - put['price'] - (100 - 105 * (0.05 * 0.25))
print(f"Put-Call Parity check: {parity:.6f}")

# Implied volatility from market price
iv = ImpliedVolatility(market_price=3.50, S=100, K=105, T=0.25, r=0.05, option_type="call")
print(f"Implied Volatility: {iv:.4f}")
```

### Fixed Income

```python
from meridianalgo import BondPricer, YieldCurve
from meridianalgo import ZSpreadCalculator

pricer = BondPricer()

# Bond pricing and risk measures
price = pricer.price(
    face_value=1000, coupon_rate=0.05, maturity=10,
    yield_to_maturity=0.06, frequency=2,
)
duration = pricer.modified_duration(
    face_value=1000, coupon_rate=0.05, maturity=10, ytm=0.06
)
convexity = pricer.convexity(
    face_value=1000, coupon_rate=0.05, maturity=10, ytm=0.06
)

print(f"Bond Price:        ${price:.4f}")
print(f"Modified Duration: {duration:.4f}")
print(f"Convexity:         {convexity:.4f}")

# Yield curve construction
curve = YieldCurve()
for maturity, rate in [(0.25, 0.04), (0.5, 0.042), (1, 0.045), (2, 0.048),
                        (5, 0.052), (10, 0.055), (30, 0.058)]:
    curve.add_point(maturity, rate)

curve.build_curve(method="nelson_siegel")
print(f"\n7yr rate (interpolated): {curve.get_yield(7):.4f}")
print(f"5y5y forward rate:       {curve.get_forward_rate(5, 10):.4f}")
```

### Backtesting

```python
from meridianalgo.backtesting import BacktestEngine, Strategy
import pandas as pd

class MACrossover(Strategy):
    def __init__(self, short_window: int = 20, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(0, index=data.index, columns=data.columns)
        for asset in data.columns:
            short_ma = data[asset].rolling(self.short_window).mean()
            long_ma = data[asset].rolling(self.long_window).mean()
            signals[asset] = (short_ma > long_ma).astype(int)
        return signals

engine = BacktestEngine(initial_capital=100_000)
results = engine.run(MACrossover(20, 50), prices, returns)

print(f"Total Return:   {results.get('total_return', 0):.2%}")
print(f"Sharpe Ratio:   {results.get('sharpe_ratio', 0):.3f}")
print(f"Max Drawdown:   {results.get('max_drawdown', 0):.2%}")
print(f"Annualized Vol: {results.get('annualized_volatility', 0):.2%}")
```

### Machine Learning

```python
from meridianalgo.ml import FeatureEngineer, WalkForwardValidator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

fe = FeatureEngineer()
features = fe.create_features(
    prices,
    features=["returns", "rsi", "macd", "volume_ratio", "volatility", "momentum"],
)
labels = (returns.shift(-1) > 0).astype(int)

validator = WalkForwardValidator()
results = validator.validate(
    features,
    labels,
    model=RandomForestClassifier(n_estimators=200, random_state=42),
    train_window=252,
    test_window=21,
)

print(f"Average Accuracy:  {results['accuracy'].mean():.2%}")
print(f"Average Precision: {results['precision'].mean():.2%}")
print(f"Average Recall:    {results['recall'].mean():.2%}")
```

### Statistical Arbitrage

```python
from meridianalgo.quant import StatisticalArbitrage

stat_arb = StatisticalArbitrage()
pairs = stat_arb.find_cointegrated_pairs(prices, p_value=0.05)

for (a, b), p_val in sorted(pairs.items(), key=lambda x: x[1]):
    spread = prices[a] - prices[b]
    half_life = stat_arb.calculate_half_life(spread)
    signals = stat_arb.generate_pairs_signals(spread, z_entry=2.0, z_exit=0.5)
    print(f"{a}/{b}: p={p_val:.4f}, half-life={half_life:.1f}d, signals={len(signals)}")
```

### Execution Algorithms

```python
from meridianalgo import VWAP, TWAP, POV, ImplementationShortfall

vwap = VWAP()
twap = TWAP()
pov = POV()

vwap_schedule = vwap.schedule(shares=10_000, volume_profile=volume_data)
twap_schedule = twap.schedule(shares=10_000, duration_minutes=60, interval_minutes=5)
pov_schedule = pov.schedule(shares=10_000, participation_rate=0.10, volume_data=volume_data)
```

---

## CLI

```bash
meridianalgo version          # show installed version
meridianalgo info             # show module availability and loaded extras
meridianalgo demo             # run portfolio optimization demo
meridianalgo metrics AAPL --period 2y   # compute metrics for a ticker
```

---

## Performance Benchmarks

*Python 3.11, NumPy 1.26, Intel i7-10700K, 32GB RAM*

| Operation | Dataset | Time | Memory |
|-----------|---------|------|--------|
| HRP Optimization | 100 assets, 5yr | 45ms | 12MB |
| Historical VaR | 500 assets, 10yr | 120ms | 8MB |
| GBM Simulation | 100k paths, 252 steps | 280ms | 45MB |
| Heston Simulation | 50k paths, 252 steps | 380ms | 60MB |
| GARCH(1,1) Fit | 2000 obs | 95ms | 8MB |
| Backtest (simple strategy) | 50 assets, 5yr | 200ms | 15MB |
| Options Greeks (1,000 contracts) | — | 35ms | 5MB |
| Merton Model (single firm) | — | 2ms | <1MB |
| BHB Attribution (10 sectors) | — | <1ms | <1MB |

---

## Architecture

```
meridianalgo/
├── portfolio/          mean-variance, HRP, Black-Litterman, Kelly, CPPI
├── risk/               VaR, CVaR, stress testing, scenario analysis
├── credit/             Merton model, CDS, Z-spread, expected loss
├── volatility/         GARCH, realized vol (5 estimators), HAR-RV, regime detection
├── monte_carlo/        GBM, Heston, jump-diffusion, CIR, variance reduction
├── derivatives/        Black-Scholes, Greeks, implied vol, exotic options
├── fixed_income/       bond pricing, yield curves, credit spreads
├── analytics/          performance metrics, benchmark attribution
├── backtesting/        event-driven engine, order management, slippage
├── ml/                 LSTM/GRU/Transformer, walk-forward CV, feature engineering
├── execution/          VWAP, TWAP, POV, implementation shortfall
├── quant/              stat arb, pairs trading, regime detection, HFT
├── signals/            RSI, MACD, Bollinger Bands, 50+ indicators
├── liquidity/          order book, bid-ask spread, market impact
├── factors/            Fama-French 3/5-factor, PCA, alpha generation
├── data/               yfinance, Polygon.io, streaming, storage
└── utils/              logging, validation, visualization
```

---

## Optional Dependencies

| Extra | Packages | Enables |
|-------|----------|---------|
| `ml` | scikit-learn, torch, statsmodels, hmmlearn | LSTM models, walk-forward CV, HMM regime |
| `optimization` | cvxpy, cvxopt | Convex portfolio optimization, CVaR minimization |
| `volatility` | arch | GARCH(p,q), EGARCH, GJR-GARCH maximum likelihood |
| `data` | lxml, beautifulsoup4, polygon-api-client | Polygon.io, web scraping |
| `distributed` | ray, dask | Parallel backtesting and optimization |
| `all` | all of the above | Full feature set |

---

## Documentation

- API Reference: [meridianalgo.readthedocs.io](https://meridianalgo.readthedocs.io)
- Examples: [examples/](examples/)
- Changelog: [CHANGELOG.md](CHANGELOG.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Pull requests welcome.

## License

MIT License. See [LICENSE](LICENSE).

## Disclaimer

For research and educational purposes. Trading financial instruments involves substantial risk of loss. Past performance does not guarantee future results. The authors accept no responsibility for financial losses arising from use of this software.
