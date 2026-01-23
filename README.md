# MeridianAlgo

## The Complete Quantitative Finance Platform

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI Version](https://img.shields.io/badge/pypi-6.2.1-orange.svg)](https://pypi.org/project/meridianalgo/)
[![Tests](https://img.shields.io/badge/tests-300%2B%20passing-brightgreen.svg)](tests/)

MeridianAlgo is a comprehensive, institutional-grade Python platform for quantitative finance. It provides a complete suite of tools for algorithmic trading, portfolio optimization, risk management, derivatives pricing, and market microstructure analysis. Built for professional quants, researchers, and trading firms.

**Key Highlights:**
- 50+ performance metrics and analytics
- Event-driven backtesting engine with realistic execution
- Optimal execution algorithms (VWAP, TWAP, POV, Implementation Shortfall)
- Market microstructure analysis (order book, VPIN, liquidity metrics)
- Statistical arbitrage and pairs trading
- Factor models (Fama-French, APT, custom)
- Options pricing and Greeks
- Machine learning integration
- GPU acceleration support
- Distributed computing ready

---

## Installation

### Standard Installation

```bash
pip install meridianalgo
```

### With Optional Dependencies

```bash
# Machine learning support (scikit-learn, PyTorch, statsmodels)
pip install meridianalgo[ml]

# Optimization (CVXPY, CVXOPT)
pip install meridianalgo[optimization]

# Volatility modeling (ARCH)
pip install meridianalgo[volatility]

# Alternative data (web scraping, API clients)
pip install meridianalgo[data]

# Distributed computing (Ray, Dask)
pip install meridianalgo[distributed]

# Everything
pip install meridianalgo[all]
```

---

## Quick Start

### Portfolio Analytics

```python
import meridianalgo as ma
import pandas as pd

# Load returns data
returns = pd.read_csv('returns.csv', index_col=0, parse_dates=True)

# Calculate comprehensive metrics
from meridianalgo.analytics import PerformanceAnalyzer

analyzer = PerformanceAnalyzer(returns)
metrics = analyzer.summary()

print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.1%}")
print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
```

### Backtesting

```python
from meridianalgo.backtesting import Backtest, SimpleMovingAverageStrategy
import yfinance as yf

# Get data
data = yf.download('AAPL', start='2020-01-01', end='2023-12-31')

# Create and run backtest
strategy = SimpleMovingAverageStrategy(short_window=20, long_window=50)
backtest = Backtest(data, strategy, initial_capital=100000)
results = backtest.run()

print(f"Total Return: {results['total_return']:.1%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

### Execution Algorithms

```python
from meridianalgo.quant.execution_algorithms import VWAP, TWAP, POV

# VWAP execution
vwap = VWAP(total_quantity=100000, start_time='09:30', end_time='16:00')
schedule = vwap.calculate_schedule(historical_volume)

# TWAP execution
twap = TWAP(total_quantity=100000, duration_minutes=480, slice_interval_minutes=5)
for i in range(twap.n_slices):
    execution = twap.execute_slice(market_price=150.0, available_liquidity=500000)
    print(f"Execute {execution['quantity']} shares at {execution['price']}")

# POV execution
pov = POV(total_quantity=100000, target_pov=0.10)
result = pov.execute(market_volume=1000000, market_price=150.0, time_remaining_pct=0.5)
```

### Market Microstructure

```python
from meridianalgo.liquidity import OrderBook, VPIN, MarketImpact

# Order book analysis
ob = OrderBook()
ob.add_bid(price=100.0, quantity=1000)
ob.add_ask(price=100.1, quantity=1000)

print(f"Spread: {ob.spread():.4f}")
print(f"Mid Price: {ob.mid_price():.2f}")
print(f"Depth: {ob.depth(levels=5)}")

# Volume-Synchronized PIN
vpin = VPIN(trades_data)
print(f"Current VPIN: {vpin.current_vpin():.3f}")
print(f"Toxicity Regime: {vpin.toxicity_regime()}")

# Market impact estimation
impact = MarketImpact()
cost = impact.estimate_total_cost(quantity=10000, volatility=0.02, volume=1000000)
print(f"Estimated Impact Cost: {cost:.4f}")
```

### Statistical Arbitrage

```python
from meridianalgo.quant.statistical_arbitrage import PairsTrading, CointegrationAnalyzer

# Pairs trading
pairs = PairsTrading(asset1_prices, asset2_prices)
signals = pairs.generate_signals(z_score_threshold=2.0)

# Cointegration analysis
analyzer = CointegrationAnalyzer(asset1_prices, asset2_prices)
result = analyzer.test_cointegration()
print(f"Cointegration p-value: {result['p_value']:.4f}")
print(f"Is cointegrated: {result['is_cointegrated']}")
```

### Factor Models

```python
from meridianalgo.quant.factor_models import FamaFrenchModel, FactorRiskDecomposition

# Fama-French 3-factor model
ff = FamaFrenchModel(returns, market_excess_returns, smb, hml)
alpha, beta_market, beta_smb, beta_hml = ff.fit()

# Factor risk decomposition
decomp = FactorRiskDecomposition(returns, factors)
risk_contrib = decomp.factor_contribution_to_risk()
print(f"Factor Risk Contributions: {risk_contrib}")
```

### Technical Analysis

```python
from meridianalgo.signals import RSI, MACD, BollingerBands, TechnicalAnalyzer

# Individual indicators
rsi = RSI(prices, period=14)
macd = MACD(prices, fast=12, slow=26, signal=9)
bb = BollingerBands(prices, period=20, std_dev=2)

# Comprehensive technical analysis
analyzer = TechnicalAnalyzer(prices)
signals = analyzer.calculate_all()
summary = analyzer.summary()
```

---

## Core Modules

### Analytics (`meridianalgo.analytics`)
- **PerformanceAnalyzer**: 50+ metrics (Sharpe, Sortino, Calmar, Information Ratio, etc.)
- **RiskAnalyzer**: VaR, CVaR, stress testing, scenario analysis
- **DrawdownAnalyzer**: Drawdown analysis, underwater plots, recovery metrics
- **TearSheet**: Pyfolio-style comprehensive performance reports

### Backtesting (`meridianalgo.backtesting`)
- **Event-driven engine**: Realistic market simulation with bid-ask spreads
- **Order management**: Market, limit, stop, bracket orders
- **Execution simulation**: Market impact, slippage, commission modeling
- **Pre-built strategies**: SMA crossover, momentum, mean reversion

### Liquidity (`meridianalgo.liquidity`)
- **OrderBook**: Depth analysis, microprice, spread metrics
- **VPIN**: Volume-Synchronized Probability of Informed Trading
- **MarketImpact**: Linear, square-root, power-law impact models
- **Microstructure**: Tick data analysis, volume profiles

### Quant (`meridianalgo.quant`)
- **Execution**: VWAP, TWAP, POV, Implementation Shortfall (Almgren-Chriss)
- **Statistical Arbitrage**: Pairs trading, cointegration, mean reversion
- **Factor Models**: Fama-French, APT, custom factor models
- **High-Frequency**: Market making, latency arbitrage, order book dynamics
- **Regime Detection**: Hidden Markov Models, structural breaks, volatility regimes

### Signals (`meridianalgo.signals`)
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic, ADX, OBV
- **Signal Generation**: Multi-indicator signal generation and evaluation
- **Pattern Recognition**: Chart patterns, support/resistance levels

### Portfolio (`meridianalgo.portfolio`)
- **Optimization**: Mean-variance, risk parity, Black-Litterman
- **Rebalancing**: Calendar, threshold, and drift-based rebalancing
- **Performance Attribution**: Brinson-Fachler attribution analysis
- **Risk Management**: Position sizing, concentration limits, Greeks hedging

### Derivatives (`meridianalgo.derivatives`)
- **Options Pricing**: Black-Scholes, binomial, Monte Carlo
- **Greeks**: Delta, gamma, vega, theta, rho calculations
- **Volatility Surfaces**: Smile, skew, term structure modeling
- **Exotic Options**: Barrier, Asian, lookback options

### Data (`meridianalgo.data`)
- **Providers**: Yahoo Finance, Polygon, custom data sources
- **Processing**: OHLCV normalization, corporate actions adjustment
- **Storage**: Efficient time-series storage and retrieval
- **Streaming**: Real-time data feed integration

---

## Performance

MeridianAlgo is optimized for performance:
- **Vectorized operations**: NumPy/Pandas for fast computation
- **GPU acceleration**: CUDA support for matrix operations
- **Distributed computing**: Ray/Dask integration for parallel processing
- **Efficient memory**: Optimized data structures for large datasets

Benchmark results on typical workloads:
- Portfolio analytics: 10,000+ assets in <1 second
- Backtesting: 10 years of daily data in <5 seconds
- Factor model fitting: 1,000+ factors in <10 seconds

---

## Documentation

Full documentation available at: https://meridianalgo.readthedocs.io

- [API Reference](docs/API_REFERENCE.md)
- [User Guide](docs/README.md)
- [Examples](examples/)
- [Benchmarks](docs/benchmarks.md)

---

## Citation

If you use MeridianAlgo in your research, please cite:

```bibtex
@software{meridianalgo2026,
  title = {MeridianAlgo: The Complete Quantitative Finance Platform},
  author = {Meridian Algorithmic Research Team},
  year = {2026},
  version = {6.2.1},
  url = {https://github.com/MeridianAlgo/Python-Packages}
}
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MeridianAlgo is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/MeridianAlgo/Python-Packages/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MeridianAlgo/Python-Packages/discussions)
- **Email**: support@meridianalgo.com

---

## Disclaimer

MeridianAlgo is provided for educational and research purposes. Past performance does not guarantee future results. Always conduct thorough testing and validation before deploying trading strategies in production.
toxicity = vpin.current_vpin()
```

### Risk Management
Institutional risk metrics including VaR, CVaR, and stress testing.

```python
from meridianalgo.risk import RiskAnalyzer

risk = RiskAnalyzer(returns)
var_95 = risk.value_at_risk(0.95, method='cornish_fisher')
stress_results = risk.stress_test({'Market Crash': -0.20})
```

---

## Testing

MeridianAlgo maintains a high standard of code quality with extensive test coverage.

```bash
# Run the full test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=meridianalgo --cov-report=term
```

---

## Governance and Community

MeridianAlgo is committed to maintaining a professional and secure environment for contributors and users.

- **[Contributing](CONTRIBUTING.md)**: Guidelines for contributing to the project.
- **[Code of Conduct](CODE_OF_CONDUCT.md)**: Our expectations for community behavior.
- **[Security Policy](SECURITY.md)**: Procedures for reporting vulnerabilities.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use MeridianAlgo in your research or business, please cite it:

```bibtex
@software{meridianalgo2026,
  title = {MeridianAlgo: The Complete Quantitative Finance Platform},
  author = {Meridian Algorithmic Research Team},
  year = {2026},
  version = {6.1.1},
  url = {https://github.com/MeridianAlgo/Python-Packages}
}
```

**MeridianAlgo**  *Empowering Finance for Everyone*