# 🌌 MeridianAlgo
## The Institutional-Grade Quantitative Finance Platform for Professional Developers

[![PyPI version](https://img.shields.io/pypi/v/meridianalgo.svg?style=flat-square&color=blue)](https://pypi.org/project/meridianalgo/)
[![Python versions](https://img.shields.io/pypi/pyversions/meridianalgo.svg?style=flat-square)](https://pypi.org/project/meridianalgo/)
[![License](https://img.shields.io/github/license/MeridianAlgo/Python-Packages.svg?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/MeridianAlgo/Python-Packages/ci.yml?branch=main&style=flat-square&label=tests)](https://github.com/MeridianAlgo/Python-Packages/actions)

**MeridianAlgo** is a comprehensive, production-ready quantitative finance ecosystem for Python. Designed for hedge funds, proprietary trading desks, and independent researchers, it provides a unified interface for the entire quantitative lifecycle—from data acquisition and signal generation to optimal execution and performance attribution.

---

## 💎 Institutional Modules

MeridianAlgo is built on a modular "Enterprise Foundation" where every component is optimized for performance and reliability.

### 📈 Core Financial Primitives
The bedrock of the platform, providing high-performance implementations of essential financial calculations.
- **Statistical Arbitrage Engine**: Cointegration analysis, Hurst exponent calculation, and Half-life estimation.
- **Advanced Technical Indicators**: Vectorized RSI, MACD, Bollinger Bands, and 50+ other institutional indicators.
- **Robust Market Data**: Unified API for multi-vendor data acquisition with built-in cleaning and alignment.

### 🏗 Portfolio Management & Optimization
Beyond standard Mean-Variance optimization, we implement robust allocation strategies.
- **Modern Portfolio Theory+**: MVO, Black-Litterman, and Risk Parity (ERC).
- **Hierarchical Risk Parity (HRP)**: Machine-learning based diversification that handles high correlations.
- **Nested Clustered Optimization (NCO)**: Addressing the instability of quadratic programming in financial datasets.
- **Transaction Cost Optimization**: Incorporating market impact and slippage directly into the allocation process.

### 🛡 Risk Management & Analytics
Comprehensive risk assessment and performance monitoring.
- **Multi-Method VaR**: Parametric (Delta-Normal), Historical Simulation, and Monte Carlo models.
- **Conditional VaR (CVaR)**: Expected Shortfall with tail risk decomposition.
- **Cornish-Fisher Adjustments**: Accounting for non-normality in returns (skewness/kurtosis).
- **Stress Testing Engine**: Scenario analysis for historical crashes (2008, 2020) and custom macroeconomic shocks.

### 🤖 Financial Machine Learning
Productionizing ML for time-series without the common pitfalls of overfitting.
- **Deep Learning Architectures**: High-fidelity LSTM, GRU, and Transformer models for financial time-series.
- **Purged Cross-Validation**: Preventing information leakage across overlapping time intervals.
- **Feature Engineering Pipeline**: 500+ alpha factors with built-in feature selection (Mutual Information, RF importance).
- **Walk-Forward Validation**: Simulating realistic model retraining and deployment cycles.

### ⚡ Optimal Execution
Production-grade algorithms to minimize market impact.
- **Standard Algos**: VWAP, TWAP, and Percentage of Volume (POV) with adaptive participation.
- **Implementation Shortfall**: Almgren-Chriss optimal trajectory for risk-averse liquidation.
- **Market Microstructure**: VPIN (Volume-Synchronized Probability of Informed Trading) and LOB dynamics.

---

## 🚀 Quick Start

### 1. Unified API Access
MeridianAlgo provides a clean "one-stop" API for baseline quantitative tasks.

```python
import meridianalgo as ma

# Fetch data and perform quick analysis
prices = ma.get_market_data(['AAPL', 'MSFT', 'GOOGL'])
returns = prices.pct_change().dropna()

# Get institutional performance metrics
metrics = ma.calculate_metrics(returns['AAPL'])
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

# Generate signals
rsi = ma.calculate_rsi(prices['AAPL'], window=14)
```

### 2. Advanced Portfolio Optimization
```python
from meridianalgo.portfolio import PortfolioOptimizer

# Initialize and optimize using HRP
opt = PortfolioOptimizer(returns)
weights = opt.optimize(method='hrp')

print("Institutional Allocations:")
print(weights.sort_values(ascending=False).head())
```

### 3. Pricing & Greeks (Derivatives)
```python
from meridianalgo.derivatives import BlackScholes, GreeksCalculator

# Calculate BS Price and Delta
price = BlackScholes.call_price(S=100, K=105, T=0.5, r=0.05, sigma=0.2)
delta = GreeksCalculator.delta('call', S=100, K=105, T=0.5, r=0.05, sigma=0.2)
```

---

## 📊 Performance Benchmarks
*Tested on Intel i9-12900K, 64GB RAM, Ubuntu 22.04*

| Operation | Scale | Latency | Efficiency |
| :--- | :--- | :--- | :--- |
| Portfolio VaR | 5,000 assets | < 120ms | Optimized Cython/NumPy |
| GARCH(1,1) Fit | 10 years daily | < 250ms | Parallelized Scipy |
| Backtest Engine | 10M events | < 3.2s | Event-driven C-Speed |
| Option Greeks | 500k contracts | < 400ms | Vectorized Broadcasters |

---

## 🛠 Enterprise Configuration

### Logging & Auditing
Standardized logging for production environments to ensure every trade decision is auditable.
```python
from meridianalgo.utils.logging import setup_logger
logger = setup_logger("prod_trading", log_file="audit.log")
```

### Data Integrity
Automated data validation for high-stakes trading systems.
```python
from meridianalgo.utils.validation import DataValidator
DataValidator.validate_timeseries(raw_data) # Validates index, continuity, and NaNs
```

---

## 📖 Documentation
Visit [docs.meridianalgo.com](https://meridianalgo.readthedocs.io) for full API documentation, mathematical derivations, and research notebooks.

---

## ⚖️ Legal Disclaimer
*MeridianAlgo is a research and development platform. Trading financial instruments involves significant risk. The authors provide no warranties and are not responsible for financial losses incurred through the use of this software.*

---

**Built with pride by the Meridian Algorithmic Research Team.**
