# Portfolio Management

Portfolio optimization, risk analysis, and performance analytics.

## PortfolioOptimizer

The primary, easy-to-use optimizer. Construct it with a returns DataFrame and
call `optimize_portfolio` with a method.

```python
import meridianalgo as ma

data = ma.get_market_data(["AAPL", "MSFT", "GOOGL", "TSLA"], start_date="2023-01-01")
returns = data.pct_change().dropna()

optimizer = ma.PortfolioOptimizer(returns, risk_free_rate=0.02)
result = optimizer.optimize_portfolio(method="sharpe")

print(result["weights"])
print(result["volatility"])
```

**`optimize_portfolio(method=...)`** supports:
`"sharpe"`, `"min_vol"`, `"max_return"`, `"risk_parity"`, `"hrp"`,
`"equal_weight"`, `"black_litterman"`.

```python
frontier = optimizer.calculate_efficient_frontier(n_portfolios=1000)
```

## Specialized Optimizers

`RiskParity`, `MeanVariance`, `HierarchicalRiskParity`, and `BlackLitterman`
share an `optimize(expected_returns, covariance_matrix, ...)` interface and
return an `OptimizationResult`.

```python
from meridianalgo import RiskParity, MeanVariance, HierarchicalRiskParity

expected_returns = returns.mean()
cov = returns.cov()

rp = RiskParity().optimize(expected_returns, cov)
mv = MeanVariance().optimize(expected_returns, cov, objective="max_sharpe")
hrp = HierarchicalRiskParity().optimize(expected_returns, cov)

print(rp.weights)
```

## Kelly Criterion

```python
from meridianalgo import KellyCriterion

kc = KellyCriterion(fraction=0.5)                       # half-Kelly
f = kc.single_asset(win_prob=0.55, win_loss_ratio=1.0)
weights = kc.optimize(returns)                          # multi-asset
f_moments = kc.from_moments(expected_return=0.12, volatility=0.18)
```

## Risk Analysis

`VaRCalculator`, `RiskAnalyzer`, and `CVaRCalculator` share the same interface.

```python
import meridianalgo as ma

portfolio_returns = returns.mean(axis=1)
var = ma.VaRCalculator(portfolio_returns)

var_95 = var.value_at_risk(confidence=0.95, method="historical")
var_99 = var.value_at_risk(confidence=0.99, method="cornish_fisher")
cvar_95 = var.conditional_var(confidence=0.95)
```

`value_at_risk` methods: `"historical"`, `"parametric"`, `"cornish_fisher"`,
`"monte_carlo"`.

### Stress Testing

```python
import numpy as np
from meridianalgo import StressTesting

stress = StressTesting()
weights = np.repeat(1 / returns.shape[1], returns.shape[1])

# Smallest shock that produces a -10% loss
result = stress.reverse_stress_test(weights, returns, target_loss=-0.10)
```

For correlated, scenario-based stress testing see `ScenarioAnalyzer` and
`CorrelationScenario` in the project README.

## Performance Analytics

```python
import meridianalgo as ma

analyzer = ma.PerformanceAnalyzer(portfolio_returns, risk_free_rate=0.02)
metrics = analyzer.calculate_all_metrics()
print(analyzer.summary())

# Or the one-call helper
print(ma.tearsheet(portfolio_returns))
```

### Benchmark-Relative

```python
from meridianalgo import BenchmarkAnalytics

analytics = BenchmarkAnalytics(
    portfolio_returns=portfolio_returns,
    benchmark_returns=benchmark_returns,
    risk_free_rate=0.02,
)
active = analytics.active_metrics()
print(active.information_ratio, active.tracking_error)
```

See the [project README](https://github.com/MeridianAlgo/Python-Packages#readme)
for end-to-end examples.
