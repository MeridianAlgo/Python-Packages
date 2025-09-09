# MeridianAlgo Documentation

This directory contains the documentation for the MeridianAlgo package.

## Contents

- `README.md` - This file
- `api/` - API reference documentation
- `examples/` - Usage examples and tutorials
- `guides/` - User guides and best practices

## Quick Start

```python
import meridianalgo as ma

# Get market data
data = ma.get_market_data(['AAPL', 'MSFT'], start_date='2023-01-01')

# Analyze time series
analyzer = ma.TimeSeriesAnalyzer(data['AAPL'])
returns = analyzer.calculate_returns()
volatility = analyzer.calculate_volatility()

# Calculate risk metrics
var = ma.calculate_value_at_risk(returns)
es = ma.calculate_expected_shortfall(returns)
```

## Features

- **Portfolio Optimization**: Modern portfolio theory and efficient frontier calculation
- **Time Series Analysis**: Returns, volatility, and technical indicators
- **Risk Management**: VaR, Expected Shortfall, and other risk metrics
- **Statistical Arbitrage**: Cointegration testing and correlation analysis
- **Machine Learning**: LSTM models for time series prediction
- **Feature Engineering**: Technical indicators and feature creation

## Installation

```bash
pip install meridianalgo
```

## Requirements

- Python 3.7+
- NumPy, Pandas, SciPy
- Scikit-learn
- PyTorch (for ML features)
- yfinance (for market data)
