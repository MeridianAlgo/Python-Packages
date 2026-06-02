# MeridianAlgo Documentation

The complete quantitative finance platform for Python — portfolio optimization,
risk management, derivatives pricing, backtesting, machine learning, execution
algorithms, fixed income, credit risk, volatility modeling, and Monte Carlo.

## Table of Contents

- [Installation Guide](installation.md)
- [Quick Start Guide](quickstart.md)
- [API Reference](API_REFERENCE.md)
  - [Technical Indicators](api/technical_indicators.md)
  - [Portfolio Management](api/portfolio_management.md)
- [Performance Benchmarks](benchmarks.md)
- [Changelog](https://github.com/MeridianAlgo/Python-Packages/blob/main/CHANGELOG.md)
- [Contributing](https://github.com/MeridianAlgo/Python-Packages/blob/main/CONTRIBUTING.md)

## What is MeridianAlgo?

MeridianAlgo is a comprehensive Python library for quantitative finance,
algorithmic trading, and statistical analysis. It provides a single toolkit for:

- **Technical Analysis**: 40+ indicators including RSI, MACD, Bollinger Bands, ADX, Ichimoku
- **Portfolio Management**: Mean-Variance, Black-Litterman, Risk Parity, HRP, Kelly Criterion
- **Risk Analysis**: VaR, CVaR, stress testing, scenario analysis, risk budgeting
- **Derivatives**: Black-Scholes, Greeks, implied volatility, binomial trees, exotics
- **Volatility & Monte Carlo**: GARCH, realized volatility, GBM, Heston, jump diffusion
- **Machine Learning**: LSTM models, walk-forward validation, feature engineering
- **Execution & Microstructure**: VWAP, TWAP, POV, implementation shortfall, order book analytics

## Installation

```bash
# Core install
pip install meridianalgo

# With machine-learning extras (scikit-learn, torch, statsmodels, hmmlearn)
pip install "meridianalgo[ml]"

# Everything
pip install "meridianalgo[all]"
```

The core install is dependency-light and imports cleanly on its own. Heavier
capabilities (ML, convex optimization, GARCH, alternative data providers) live
behind optional extras — see [installation.md](installation.md) for the full list.

## Quick Start

```python
import meridianalgo as ma

# Market data
data = ma.get_market_data(['AAPL', 'MSFT', 'GOOGL'], start_date='2023-01-01')
returns = data.pct_change().dropna()

# Technical indicators
rsi = ma.calculate_rsi(data['AAPL'], period=14)
macd_line, signal_line, histogram = ma.calculate_macd(data['AAPL'])

# Portfolio optimization
optimizer = ma.PortfolioOptimizer(returns)
result = optimizer.optimize_portfolio(method='sharpe')

# Risk analysis
var = ma.VaRCalculator(returns['AAPL']).value_at_risk(confidence=0.95, method='historical')

# One-call performance summary
stats = ma.summary_stats(returns['AAPL'])
print(ma.tearsheet(returns['AAPL']))
```

## Module Availability

Every module loads behind a registry so the package imports even when an optional
dependency is missing. Check what is available at runtime:

```python
import meridianalgo as ma
print(ma.ModuleRegistry.status())
```

## License

MIT License — see [LICENSE](../LICENSE).

## Acknowledgments

Built on NumPy, Pandas, SciPy, scikit-learn, and PyTorch, and inspired by
quantitative finance best practices.

---

**MeridianAlgo** — Empowering quantitative finance with professional-grade tooling.
