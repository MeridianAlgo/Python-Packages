# MeridianAlgo

[![PyPI version](https://img.shields.io/pypi/v/meridianalgo.svg?style=flat-square&color=blue)](https://pypi.org/project/meridianalgo/)
[![Python versions](https://img.shields.io/pypi/pyversions/meridianalgo.svg?style=flat-square)](https://pypi.org/project/meridianalgo/)
[![License](https://img.shields.io/pypi/l/meridianalgo.svg?style=flat-square)](https://github.com/MeridianAlgo/Python-Packages/blob/main/LICENSE)

**The complete quantitative finance platform for Python.** Portfolio
optimization, risk management, derivatives pricing, backtesting, machine
learning, execution algorithms, and more — in one library.

## Installation

```bash
pip install meridianalgo
```

Optional extras add heavier capabilities on demand:

```bash
pip install "meridianalgo[ml]"            # scikit-learn, torch, statsmodels, hmmlearn
pip install "meridianalgo[optimization]"  # cvxpy, cvxopt
pip install "meridianalgo[volatility]"    # arch (GARCH family)
pip install "meridianalgo[all]"           # everything
```

The core install imports cleanly on its own; modules that need an optional
dependency report as unavailable via `meridianalgo.ModuleRegistry` until the
matching extra is installed.

## Quick Start

```python
import meridianalgo as ma

# Market data and returns
data = ma.get_market_data(["AAPL", "MSFT", "GOOGL"], start_date="2023-01-01")
returns = data.pct_change().dropna()

# One-call performance and risk summary
print(ma.tearsheet(returns["AAPL"]))

# Portfolio optimization
opt = ma.PortfolioOptimizer(returns)
result = opt.optimize_portfolio(method="sharpe")

# Risk analysis
var = ma.VaRCalculator(returns["AAPL"]).value_at_risk(confidence=0.95)
```

## What's Inside

| Domain | Highlights |
|--------|-----------|
| Portfolio | Mean-Variance, HRP, Black-Litterman, Risk Parity, Kelly, CPPI |
| Risk | VaR, CVaR, stress testing, scenario analysis, risk budgeting |
| Derivatives | Black-Scholes, Greeks, implied vol, binomial trees, exotics |
| Volatility | GARCH/EGARCH/GJR, realized-vol estimators, HAR-RV, regimes |
| Monte Carlo | GBM, Heston, jump-diffusion, CIR, variance reduction |
| Credit | Merton model, CDS pricing, Z-spread, expected loss |
| Fixed Income | Bond pricing, duration/convexity, yield curves |
| Backtesting | Event-driven engine, order management, slippage |
| Machine Learning | LSTM models, walk-forward CV, feature engineering |
| Execution | VWAP, TWAP, POV, implementation shortfall |
| Signals | 40+ technical indicators (functional and OOP APIs) |

## Links

- **Documentation:** https://meridianalgo.readthedocs.io
- **Source & Issues:** https://github.com/MeridianAlgo/Python-Packages
- **Changelog:** https://github.com/MeridianAlgo/Python-Packages/blob/main/CHANGELOG.md

## License

MIT License. For research and educational purposes — trading involves
substantial risk of loss, and past performance does not guarantee future results.
