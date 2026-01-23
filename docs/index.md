# MeridianAlgo Documentation

Welcome to the comprehensive documentation for MeridianAlgo, the advanced algorithmic trading and statistical analysis library.

##  Table of Contents

- [Quick Start Guide](quickstart.md)
- [Installation Guide](installation.md)
- [API Reference](api/)
  - [Core Module](api/core.md)
  - [Technical Indicators](api/technical_indicators.md)
  - [Portfolio Management](api/portfolio_management.md)
  - [Risk Analysis](api/risk_analysis.md)
  - [Data Processing](api/data_processing.md)
  - [Statistics](api/statistics.md)
  - [Machine Learning](api/ml.md)
- [Examples](examples/)
  - [Basic Usage](examples/basic_usage.md)
  - [Advanced Strategies](examples/advanced_strategies.md)
  - [Portfolio Optimization](examples/portfolio_optimization.md)
  - [Risk Management](examples/risk_management.md)
- [Performance Benchmarks](benchmarks.md)
- [Contributing Guide](contributing.md)
- [Changelog](changelog.md)

##  What is MeridianAlgo?

MeridianAlgo is a comprehensive Python library designed for quantitative finance, algorithmic trading, and statistical analysis. It provides a complete toolkit for:

- **Technical Analysis**: 50+ technical indicators including RSI, MACD, Bollinger Bands, and more
- **Portfolio Management**: Advanced optimization strategies including Modern Portfolio Theory, Black-Litterman, and Risk Parity
- **Risk Analysis**: Comprehensive risk metrics including VaR, Expected Shortfall, and stress testing
- **Machine Learning**: LSTM models and feature engineering for financial time series
- **Data Processing**: Data cleaning, validation, and feature engineering utilities

##  Key Features

###  Technical Analysis Suite
- **Momentum Indicators**: RSI, Stochastic, Williams %R, ROC, Momentum
- **Trend Indicators**: Moving averages, MACD, ADX, Aroon, Parabolic SAR, Ichimoku
- **Volatility Indicators**: Bollinger Bands, ATR, Keltner Channels, Donchian Channels
- **Volume Indicators**: OBV, AD Line, Chaikin Oscillator, Money Flow Index
- **Overlay Indicators**: Pivot Points, Fibonacci Retracement, Support/Resistance

###  Portfolio Management
- **Optimization Strategies**: MPT, Black-Litterman, Risk Parity
- **Risk Management**: VaR, Expected Shortfall, Maximum Drawdown
- **Performance Analysis**: Attribution analysis, benchmark comparison
- **Rebalancing**: Calendar and threshold-based rebalancing

###  Risk Analysis
- **Value at Risk**: Historical, Parametric, Monte Carlo methods
- **Expected Shortfall**: Tail risk analysis
- **Stress Testing**: Scenario analysis and historical stress tests
- **Risk Metrics**: Sharpe, Sortino, Calmar ratios
- **Regime Analysis**: Market regime detection

###  Machine Learning
- **Feature Engineering**: Technical and fundamental features
- **LSTM Models**: Time series prediction
- **Model Evaluation**: Comprehensive metrics and validation
- **Data Processing**: Cleaning and validation utilities

##  Installation

```bash
# Install latest version
pip install meridianalgo

# Install with development dependencies
pip install meridianalgo[dev]
```

##  Quick Start

```python
import meridianalgo as ma
import pandas as pd

# Get market data
data = ma.get_market_data(['AAPL', 'MSFT', 'GOOGL'], start_date='2023-01-01')

# Technical Analysis
rsi = ma.RSI(data['AAPL'], period=14)
macd_line, signal_line, histogram = ma.MACD(data['AAPL'])

# Portfolio Optimization
returns = data.pct_change().dropna()
optimizer = ma.PortfolioOptimizer(returns)
optimal_portfolio = optimizer.optimize_portfolio(objective='sharpe')

# Risk Analysis
var_95 = ma.calculate_value_at_risk(returns['AAPL'], confidence_level=0.95)
```

##  Performance

- **Prediction Accuracy**: 78-85% (within 3% of actual price)
- **Excellent Predictions**: 25-35% (within 1% of actual price)
- **Average Error**: 1.8-2.4%
- **Test Coverage**: 40+ comprehensive tests
- **All Demos**: 6/6 passing 

##  Contributing

We welcome contributions! Please see our [Contributing Guide](contributing.md) for details.

##  License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

##  Acknowledgments

- **Quant Analytics**: Portions integrate concepts from [quant-analytics](https://pypi.org/project/quant-analytics/) by Anthony Baxter
- **Open Source**: Built on NumPy, Pandas, SciPy, Scikit-learn, PyTorch
- **Community**: Inspired by quantitative finance best practices

---

**MeridianAlgo** - Empowering quantitative finance with advanced algorithmic trading tools.

*Built with  by the Meridian Algorithmic Research Team*
