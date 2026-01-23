# MeridianAlgo

## The Complete Quantitative Finance Platform

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI Version](https://img.shields.io/badge/pypi-6.0.0-orange.svg)](https://pypi.org/project/meridianalgo/)
[![Tests](https://img.shields.io/badge/tests-300%2B%20passing-brightgreen.svg)](tests/)

MeridianAlgo is a comprehensive Python platform designed for institutional-grade quantitative finance. It provides a robust suite of tools for trading research, portfolio analytics, liquidity analysis, options pricing, and high-frequency execution. 

Designed for scalability and performance, MeridianAlgo integrates seamlessly into quantitative workflows, offering professional-grade modularity and performance.

---

## Key Features

| Feature | MeridianAlgo | QuantLib | Zipline | Pyfolio |
|---------|--------------|----------|---------|---------|
| Portfolio Analytics | Included | Limited | Partial | Included |
| Options Pricing | Included | Included | Not Included | Not Included |
| Market Microstructure | Included | Not Included | Not Included | Not Included |
| Backtesting | Included | Not Included | Included | Not Included |
| Execution Algorithms | Included | Not Included | Partial | Not Included |
| Risk Management | Included | Included | Not Included | Partial |
| Factor Models | Included | Not Included | Partial | Not Included |
| Machine Learning | Included | Not Included | Not Included | Not Included |
| Liquidity Analysis | Included | Not Included | Not Included | Not Included |
| Performance Metrics | Included | Not Included | Not Included | Included |

---

## Installation

### Standard Installation

Install the base package via pip:

```bash
pip install meridianalgo
```

### Optional Dependencies

MeridianAlgo supports several optional feature sets:

```bash
# Machine learning support (PyTorch/TensorFlow)
pip install meridianalgo[ml]

# Full suite (recommended for researchers)
pip install meridianalgo[full]

# Distributed computing support (Ray/Dask)
pip install meridianalgo[all]
```

---

## Basic Usage

```python
import meridianalgo as ma

# Quick analysis of market data
data = ma.get_market_data_quick(['AAPL', 'MSFT', 'GOOGL'], start='2023-01-01')
analysis = ma.quick_analysis(data['AAPL']['Close'])

print(f"Sharpe Ratio: {analysis['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {analysis['max_drawdown']:.1%}")
print(f"Win Rate: {analysis['win_rate']:.1%}")
```

---

## Core Modules

### Portfolio Analytics
Generate comprehensive performance metrics and tear sheets, similar to Pyfolio but enhanced for modern workloads.

```python
from meridianalgo.analytics import TearSheet

ts = TearSheet(returns, benchmark=spy_returns)
ts.create_full_tear_sheet(filename='report.pdf')
metrics = ts.get_metrics_summary()
```

### Market Microstructure
Analyze order book dynamics, toxicity, and liquidity using state-of-the-art models.

```python
from meridianalgo.liquidity import OrderBookAnalyzer, VPIN

# Order book insights
analyzer = OrderBookAnalyzer()
imbalance = analyzer.order_imbalance()

# Flow toxicity estimation
vpin = VPIN(trades)
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
  version = {6.0.0},
  url = {https://github.com/MeridianAlgo/Python-Packages}
}
```

**MeridianAlgo**  *Empowering Finance for Everyone*