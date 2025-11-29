.. _getting_started:

Getting Started
==============

This guide will help you get up and running with MeridianAlgo. We'll cover the basic workflow and show you how to use the main components of the library.

Basic Workflow
--------------

The typical workflow when using MeridianAlgo is:

1. **Data Loading**: Load financial data
2. **Preprocessing**: Clean and prepare the data
3. **Analysis**: Perform statistical and technical analysis
4. **Modeling**: Build and train models
5. **Backtesting**: Test your strategies
6. **Optimization**: Optimize your portfolio

Example: Basic Usage
-------------------

Let's walk through a simple example that demonstrates loading market data, calculating returns, and optimizing a portfolio.

.. code-block:: python

    import meridianalgo as ma
    import pandas as pd
    import matplotlib.pyplot as plt

    # 1. Get market data
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    data = ma.get_market_data(symbols, start_date='2020-01-01', end_date='2021-01-01')
    
    # 2. Calculate returns
    returns = data.pct_change().dropna()
    
    # 3. Optimize portfolio
    weights = ma.optimize_portfolio(returns, method='sharpe')
    
    # 4. Calculate portfolio returns
    portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
    
    # 5. Calculate risk metrics
    metrics = ma.calculate_risk_metrics(portfolio_returns)
    
    print("Optimal Weights:")
    for ticker, weight in weights.items():
        print(f"{ticker}: {weight:.2%}")
    
    print("\nPortfolio Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

Key Features
-----------

### Data Loading

- Load data from various sources (Yahoo Finance, CSV, etc.)
- Handle missing data and outliers
- Resample and align time series data

### Portfolio Optimization

- Mean-variance optimization
- Risk parity strategies
- Minimum volatility portfolios
- Maximum Sharpe ratio portfolios

### Risk Analysis

- Value at Risk (VaR)
- Expected Shortfall (CVaR)
- Drawdown analysis
- Risk-adjusted return metrics

### Technical Analysis

- Moving averages
- Oscillators (RSI, MACD, etc.)
- Volatility indicators
- Volume indicators

### Machine Learning

- Feature engineering
- Time series forecasting
- Classification models
- Dimensionality reduction

Next Steps
----------

- Check out the :ref:`user_guide` for more detailed examples
- Explore the :ref:`api_reference` for a complete list of available functions
- Try the :ref:`examples` for practical use cases
- Learn how to :ref:`contribute` to the project
