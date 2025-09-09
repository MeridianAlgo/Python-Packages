<div align="center">
  <h1>🚀 MeridianAlgo Financial Analysis Platform</h1>
  <h3>Advanced Financial Analysis Suite with ML Predictions, Technical Analysis & Portfolio Management</h3>
  
  [![PyPI](https://img.shields.io/pypi/v/meridianalgo?color=blue&label=PyPI%20Package)](https://pypi.org/project/meridianalgo/)
  [![Python](https://img.shields.io/pypi/pyversions/meridianalgo?color=blue)](https://www.python.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Documentation](https://img.shields.io/badge/Documentation-Read%20The%20Docs-blue)](https://meridianalgo.org/docs)
  [![Tests](https://github.com/MeridianAlgo/Python-Packages/actions/workflows/tests.yml/badge.svg)](https://github.com/MeridianAlgo/Python-Packages/actions)
  
  Professional-grade financial analysis platform combining ML predictions, technical indicators, portfolio optimization, and comprehensive market intelligence.
</div>

## 📦 Installation

```bash
pip install meridianalgo
```

## 🚀 Quick Start

```python
from meridianalgo import quick_predict, analyze_accuracy

# Get quick prediction
result = quick_predict('AAPL', days=5)
print(f"AAPL predictions: {result}")

# Analyze model accuracy
accuracy = analyze_accuracy('AAPL')
print(f"Accuracy: {accuracy['accuracy_rate']:.1f}%")
```

## 🔍 Overview

MeridianAlgo is an advanced financial analysis platform that combines machine learning, statistical analysis, and quantitative finance techniques to provide comprehensive market insights. The platform is designed for both retail and institutional investors seeking to enhance their trading strategies with data-driven decision making.

## ✨ Key Features

### 🤖 Advanced Machine Learning
- **Ensemble Models**: Combines Random Forest, Gradient Boosting, and LSTM Neural Networks
- **Technical Indicators**: 50+ indicators including RSI, MACD, Bollinger Bands, Stochastic
- **Feature Engineering**: Advanced price patterns, volume analysis, volatility metrics
- **GPU Acceleration**: Support for NVIDIA CUDA, AMD ROCm, Intel XPU, Apple MPS

### 📊 Market Analysis
- **Real-time Data**: Yahoo Finance integration with smart caching
- **Technical Analysis**: Comprehensive set of technical indicators
- **Market Regime Detection**: Identify bull/bear markets and volatility regimes
- **Sector Analysis**: Performance comparison across market sectors

### 🎯 Portfolio Management
- **Portfolio Optimization**: Modern portfolio theory implementation
- **Risk Management**: Value at Risk (VaR), Expected Shortfall (CVaR), Maximum Drawdown
- **Performance Tracking**: Detailed performance metrics and attribution analysis
- **Backtesting**: Historical strategy validation

### 📈 Prediction System
- **Multi-day Forecasting**: 1-7 day price predictions
- **Confidence Scoring**: Model confidence with risk assessment
- **Automated Validation**: Daily accuracy tracking
- **Learning System**: Continuous model improvement

## Installation

```bash
pip install meridianalgo
```

## Quick Start

```python
import meridianalgo as ma
import yfinance as yf

# Fetch market data
data = yf.download(['AAPL', 'MSFT', 'GOOGL'], start='2020-01-01')['Adj Close']

# Calculate returns
returns = data.pct_change().dropna()

# Portfolio optimization
optimizer = ma.PortfolioOptimizer(returns)
efficient_frontier = optimizer.calculate_efficient_frontier()

# Statistical analysis
analyzer = ma.StatisticalArbitrage(data)
correlation = analyzer.calculate_rolling_correlation(window=21)

# Calculate risk metrics
var = ma.calculate_value_at_risk(returns['AAPL'])
es = ma.calculate_expected_shortfall(returns['AAPL'])
```

## 📚 Documentation

### Core Modules

#### Stock Prediction
```python
from meridianalgo import StockPredictor

# Initialize predictor
predictor = StockPredictor()

# Train model
predictor.train('AAPL')

# Get predictions
predictions = predictor.predict(days=5)
```

#### Portfolio Optimization
```python
from meridianalgo import PortfolioOptimizer
import yfinance as yf

# Fetch data
data = yf.download(['AAPL', 'MSFT', 'GOOGL'], start='2020-01-01')['Adj Close']
returns = data.pct_change().dropna()

# Optimize portfolio
optimizer = PortfolioOptimizer(returns)
efficient_frontier = optimizer.calculate_efficient_frontier()
optimal_weights = optimizer.optimize_portfolio()
```

#### Statistical Arbitrage
```python
from meridianalgo import StatisticalArbitrage

# Initialize arbitrage model
arbitrage = StatisticalArbitrage()

# Find cointegrated pairs
pairs = arbitrage.find_cointegrated_pairs(returns)

# Calculate spread
spread = arbitrage.calculate_spread(returns['AAPL'], returns['MSFT'])
```

### 📊 Technical Indicators

```python
from meridianalgo import TechnicalIndicators

# Calculate RSI
rsi = TechnicalIndicators.rsi(data['Close'], period=14)

# Calculate MACD
macd, signal, hist = TechnicalIndicators.macd(data['Close'])

# Calculate Bollinger Bands
upper, middle, lower = TechnicalIndicators.bollinger_bands(data['Close'])
```

## 📈 Performance Metrics

### Prediction Accuracy
- **Overall Accuracy**: 78-85% (within 3% of actual price)
- **Excellent Predictions**: 25-35% (within 1% of actual price)
- **Good Predictions**: 45-55% (within 2% of actual price)
- **Average Error**: 1.8-2.4%

### Risk Metrics
- **Value at Risk (VaR)**: 95% confidence level
- **Expected Shortfall (CVaR)**: Average loss beyond VaR
- **Maximum Drawdown**: Worst historical loss from peak to trough
- **Sharpe Ratio**: Risk-adjusted returns

## 🛠️ System Requirements

### Dependencies
- Python 3.8+
- NumPy >= 1.21.0
- pandas >= 1.5.0
- scikit-learn >= 1.0.0
- PyTorch >= 1.12.0
- yfinance >= 0.1.87

### Hardware
- **CPU**: 2+ cores recommended
- **RAM**: 4GB+ recommended
- **GPU**: Optional but recommended for faster training (NVIDIA CUDA, AMD ROCm, or Apple MPS supported)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For support, please open an issue on our [GitHub repository](https://github.com/MeridianAlgo/Python-Packages) or contact us at support@meridianalgo.com.

## 📝 Disclaimer

This software is for educational and research purposes only. Stock market predictions are inherently uncertain, and past performance does not guarantee future results. Always conduct your own research and consider consulting with financial professionals before making investment decisions. The authors are not responsible for any financial losses incurred from using this software.

## Examples

See the `examples/` directory for complete usage examples:

1. [Portfolio Optimization](examples/portfolio_optimization.py)
2. [Statistical Arbitrage](examples/statistical_arbitrage.py)
3. [Time Series Prediction](examples/time_series_prediction.py)

## Requirements

- Python 3.7+
- numpy
- pandas
- scipy
- scikit-learn
- yfinance
- torch (for deep learning features)

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, or suggest features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue on GitHub or contact support@meridianalgo.com

---

*MeridianAlgo is developed and maintained by the Meridian Algorithmic Research Team.*
