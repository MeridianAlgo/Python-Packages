# MeridianAlgo

[![PyPI version](https://img.shields.io/pypi/v/meridianalgo.svg?style=flat-square&color=blue)](https://pypi.org/project/meridianalgo/)
[![Python versions](https://img.shields.io/pypi/pyversions/meridianalgo.svg?style=flat-square)](https://pypi.org/project/meridianalgo/)
[![License](https://img.shields.io/github/license/MeridianAlgo/Python-Packages.svg?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/MeridianAlgo/Python-Packages/ci.yml?branch=main&style=flat-square&label=tests)](https://github.com/MeridianAlgo/Python-Packages/actions)

Institutional-grade Python library for quantitative finance, algorithmic trading, and financial machine learning. Covers the full quant stack: portfolio optimization, risk management, derivatives pricing, backtesting, ML-driven signal generation, execution algorithms, fixed income analytics, and market microstructure analysis.

## Installation

```bash
pip install meridianalgo
```

```bash
# All optional dependencies
pip install meridianalgo[all]

# Selective extras
pip install meridianalgo[ml]           # scikit-learn, torch, statsmodels, hmmlearn
pip install meridianalgo[optimization] # cvxpy, cvxopt
pip install meridianalgo[volatility]   # arch (GARCH models)
pip install meridianalgo[data]         # lxml, beautifulsoup4, polygon-api-client
pip install meridianalgo[distributed]  # ray, dask
```

**Requirements:** Python ≥ 3.10

## Quick Start

```python
import meridianalgo as ma

# Check loaded modules
print(ma.__version__)
print(ma.ModuleRegistry.status())
```

---

## Features

| Module | Capabilities |
|--------|-------------|
| `portfolio` | Mean-variance, HRP, Risk Parity, Black-Litterman, Kelly Criterion |
| `risk` | VaR, CVaR, Cornish-Fisher VaR, stress testing, risk budgeting |
| `backtesting` | Event-driven engine, order management, slippage, performance analytics |
| `ml` | LSTM/GRU/Transformer, walk-forward CV, purged CV, model registry |
| `derivatives` | Black-Scholes, Greeks, Monte Carlo, implied volatility, exotic options |
| `fixed_income` | Bond pricing, yield curves, duration, convexity, credit spreads |
| `execution` | VWAP, TWAP, POV, implementation shortfall |
| `quant` | Stat arb, pairs trading, regime detection, HFT, market microstructure |
| `signals` | RSI, MACD, Bollinger Bands, 50+ technical indicators |
| `liquidity` | Order book analysis, bid-ask spread, market impact models |
| `factors` | Fama-French factor models, factor exposure, alpha generation |
| `analytics` | Sharpe, Sortino, Calmar, drawdown, tear sheets |

---

## Examples

### Market Data and Technical Indicators

```python
import meridianalgo as ma

# Fetch price data
prices = ma.get_market_data(["AAPL", "MSFT", "GOOGL"], start="2022-01-01")
returns = ma.calculate_returns(prices)

# Technical indicators
rsi = ma.calculate_rsi(prices["AAPL"], window=14)
macd = ma.calculate_macd(prices["MSFT"])
upper, middle, lower = ma.calculate_bollinger_bands(prices["GOOGL"], period=20, std_dev=2.0)

# Performance metrics
sharpe = ma.calculate_sharpe_ratio(returns["AAPL"])
sortino = ma.calculate_sortino_ratio(returns["AAPL"])
calmar = ma.calculate_calmar_ratio(returns["AAPL"])
max_dd = ma.calculate_max_drawdown(returns["AAPL"])
cvar = ma.calculate_expected_shortfall(returns["AAPL"])

print(f"Sharpe:       {sharpe:.3f}")
print(f"Sortino:      {sortino:.3f}")
print(f"Calmar:       {calmar:.3f}")
print(f"Max Drawdown: {max_dd:.2%}")
print(f"95% CVaR:     {cvar:.2%}")
```

### Portfolio Optimization

```python
from meridianalgo.portfolio import PortfolioOptimizer

opt = PortfolioOptimizer(returns)

# Multiple optimization methods
hrp_weights = opt.optimize(method="hrp")           # Hierarchical Risk Parity
min_var_weights = opt.optimize(method="min_variance")
risk_parity_weights = opt.optimize(method="risk_parity")
max_sharpe_weights = opt.optimize(method="max_sharpe")

print("HRP Weights:")
print(hrp_weights.sort_values(ascending=False))
```

### Kelly Criterion Position Sizing

```python
from meridianalgo import KellyCriterion

kc = KellyCriterion(fraction=0.5)  # Half-Kelly for lower volatility

# Single asset (binary bet)
f = kc.single_asset(win_prob=0.55, win_loss_ratio=1.0)
print(f"Kelly fraction: {f:.2%}")

# Multi-asset continuous Kelly
weights = kc.optimize(returns)
print("Kelly weights:", weights)

# From expected return and volatility
f_moments = kc.from_moments(expected_return=0.12, volatility=0.18)
print(f"Kelly (moments): {f_moments:.2%}")

# Expected geometric growth rate
g = kc.growth_rate(expected_return=0.12, volatility=0.18)
print(f"Expected growth: {g:.2%}")
```

### Risk Management

```python
from meridianalgo import RiskAnalyzer, StressTesting, VaRCalculator

# VaR and CVaR
risk = RiskAnalyzer(returns)
var_95 = risk.calculate_var(confidence_level=0.95, method="historical")
cvar_95 = risk.calculate_cvar(confidence_level=0.95)

print(f"95% VaR:  {var_95:.2%}")
print(f"95% CVaR: {cvar_95:.2%}")

# Stress testing
stress = StressTesting(returns)
scenarios = stress.run_historical_scenarios()
print("Stress scenarios:", list(scenarios.keys()))
```

### Derivatives Pricing

```python
from meridianalgo import BlackScholes, GreeksCalculator, ImpliedVolatility
from meridianalgo.derivatives import OptionsPricer

pricer = OptionsPricer()

# Black-Scholes pricing (returns price + all Greeks)
call = BlackScholes(S=100, K=105, T=0.25, r=0.05, sigma=0.20, option_type="call")
put = BlackScholes(S=100, K=105, T=0.25, r=0.05, sigma=0.20, option_type="put")

print(f"Call price: ${call['price']:.4f}")
print(f"Call delta: {call['delta']:.4f}")
print(f"Call gamma: {call['gamma']:.4f}")
print(f"Call theta: {call['theta']:.4f}")
print(f"Call vega:  {call['vega']:.4f}")

# Implied volatility from market price
iv = ImpliedVolatility(market_price=3.50, S=100, K=105, T=0.25, r=0.05, option_type="call")
print(f"Implied volatility: {iv}")
```

### Backtesting

```python
from meridianalgo.backtesting import BacktestEngine, Strategy

class MACrossover(Strategy):
    def __init__(self, short_window: int = 20, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        import pandas as pd
        signals = pd.DataFrame(0, index=data.index, columns=data.columns)
        for asset in data.columns:
            short_ma = data[asset].rolling(self.short_window).mean()
            long_ma = data[asset].rolling(self.long_window).mean()
            signals[asset] = (short_ma > long_ma).astype(int)
        return signals

engine = BacktestEngine(initial_capital=100_000)
strategy = MACrossover(short_window=20, long_window=50)
results = engine.run(strategy, prices, returns)
print(f"Total Return: {results.get('total_return', 0):.2%}")
print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
```

### Machine Learning

```python
from meridianalgo.ml import FeatureEngineer, WalkForwardValidator
from sklearn.ensemble import RandomForestClassifier

fe = FeatureEngineer()
features = fe.create_features(prices, features=["returns", "rsi", "macd", "volume_ratio"])

labels = (returns.shift(-1) > 0).astype(int)

validator = WalkForwardValidator()
results = validator.validate(
    features,
    labels,
    model=RandomForestClassifier(n_estimators=100, random_state=42),
    train_window=252,
    test_window=21,
)
print(f"Average Accuracy: {results['accuracy'].mean():.2%}")
```

### Statistical Arbitrage

```python
from meridianalgo.quant import StatisticalArbitrage

pairs_trader = StatisticalArbitrage()
pairs = pairs_trader.find_cointegrated_pairs(prices, p_value=0.05)

for pair, p_val in pairs.items():
    print(f"{pair}: p={p_val:.4f}")
    spread = prices[pair[0]] - prices[pair[1]]
    half_life = pairs_trader.calculate_half_life(spread)
    print(f"  Half-life: {half_life:.1f} days")
    signals = pairs_trader.generate_pairs_signals(spread, z_entry=2.0, z_exit=0.5)
    print(f"  Signals: {len(signals)}")
```

### Execution Algorithms

```python
from meridianalgo import VWAP, TWAP, POV, ImplementationShortfall

vwap = VWAP()
twap = TWAP()
pov = POV()

# Schedule a 10,000 share order
vwap_schedule = vwap.schedule(shares=10_000, volume_profile=volume_data)
twap_schedule = twap.schedule(shares=10_000, duration_minutes=60, interval_minutes=5)
pov_schedule = pov.schedule(shares=10_000, participation_rate=0.10, volume_data=volume_data)
```

### Fixed Income

```python
from meridianalgo import BondPricer, YieldCurve

pricer = BondPricer()

# Price a bond
price = pricer.price(
    face_value=1000,
    coupon_rate=0.05,
    maturity=10,
    yield_to_maturity=0.06,
    frequency=2,
)
duration = pricer.modified_duration(face_value=1000, coupon_rate=0.05, maturity=10, ytm=0.06)
convexity = pricer.convexity(face_value=1000, coupon_rate=0.05, maturity=10, ytm=0.06)

print(f"Bond Price:        ${price:.2f}")
print(f"Modified Duration: {duration:.4f}")
print(f"Convexity:         {convexity:.4f}")
```

---

## CLI

```bash
# Show version
meridianalgo version

# Show module availability
meridianalgo info

# Run portfolio optimization demo
meridianalgo demo

# Compute metrics for a ticker
meridianalgo metrics AAPL --period 2y
```

---

## Performance Benchmarks

*Intel i7-10700K, 32GB RAM, Python 3.11*

| Operation | Dataset | Time | Memory |
|-----------|---------|------|--------|
| HRP Optimization | 100 assets, 5yr | 45ms | 12MB |
| Historical VaR | 500 assets, 10yr | 120ms | 8MB |
| Backtest (simple strategy) | 50 assets, 5yr | 200ms | 15MB |
| Options Greeks (1,000 contracts) | — | 35ms | 5MB |
| Feature Engineering | 10 assets, 3yr | 180ms | 20MB |

---

## Optional Dependencies

```bash
pip install meridianalgo[ml]           # LSTM, GRU, walk-forward CV
pip install meridianalgo[optimization] # CVXPY convex optimization
pip install meridianalgo[volatility]   # GARCH/EGARCH via arch
pip install meridianalgo[data]         # Polygon.io, lxml, BeautifulSoup
pip install meridianalgo[distributed]  # Ray and Dask for parallel computing
pip install meridianalgo[all]          # Everything above
```

---

## Documentation

- API Reference: [meridianalgo.readthedocs.io](https://meridianalgo.readthedocs.io)
- Examples: [examples/](examples/)
- Changelog: [CHANGELOG.md](CHANGELOG.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License. See [LICENSE](LICENSE).

## Disclaimer

For educational and research purposes only. Trading financial instruments involves substantial risk of loss. The authors are not responsible for financial losses incurred through use of this software.
