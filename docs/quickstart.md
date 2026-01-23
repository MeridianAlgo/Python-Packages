# Quick Start Guide

Get up and running with MeridianAlgo in minutes!

##  Installation

```bash
# Install MeridianAlgo
pip install meridianalgo

# Verify installation
python -c "import meridianalgo; print(meridianalgo.__version__)"
```

##  Basic Usage

### 1. Import the Library

```python
import meridianalgo as ma
import pandas as pd
import numpy as np
```

### 2. Get Market Data

```python
# Fetch stock data
data = ma.get_market_data(['AAPL', 'MSFT', 'GOOGL'], start_date='2023-01-01')
print(f"Retrieved data for {len(data.columns)} stocks")
print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
```

### 3. Technical Analysis

```python
# Calculate technical indicators
close_prices = data['AAPL']

# Momentum indicators
rsi = ma.RSI(close_prices, period=14)
stoch_k, stoch_d = ma.Stochastic(data['AAPL'].high, data['AAPL'].low, close_prices)

# Trend indicators
sma_20 = ma.SMA(close_prices, 20)
ema_12 = ma.EMA(close_prices, 12)
macd_line, signal_line, histogram = ma.MACD(close_prices)

# Volatility indicators
bb_upper, bb_middle, bb_lower = ma.BollingerBands(close_prices)
atr = ma.ATR(data['AAPL'].high, data['AAPL'].low, close_prices)

print(f"RSI: {rsi.iloc[-1]:.2f}")
print(f"MACD: {macd_line.iloc[-1]:.4f}")
```

### 4. Portfolio Optimization

```python
# Calculate returns
returns = data.pct_change().dropna()

# Create portfolio optimizer
optimizer = ma.PortfolioOptimizer(returns)

# Calculate efficient frontier
frontier_data = optimizer.calculate_efficient_frontier(num_portfolios=1000)

# Optimize portfolio
optimal_portfolio = optimizer.optimize_portfolio(objective='sharpe')

print(f"Optimal weights: {optimal_portfolio['weights']}")
print(f"Expected return: {optimal_portfolio['return']:.2%}")
print(f"Volatility: {optimal_portfolio['volatility']:.2%}")
print(f"Sharpe ratio: {optimal_portfolio['sharpe_ratio']:.2f}")
```

### 5. Risk Analysis

```python
# Calculate risk metrics
portfolio_returns = returns.mean(axis=1)

# Value at Risk
var_95 = ma.calculate_value_at_risk(portfolio_returns, confidence_level=0.95)
var_99 = ma.calculate_value_at_risk(portfolio_returns, confidence_level=0.99)

# Expected Shortfall
es_95 = ma.calculate_expected_shortfall(portfolio_returns, confidence_level=0.95)

# Maximum Drawdown
max_dd = ma.calculate_max_drawdown(portfolio_returns)

print(f"95% VaR: {var_95:.2%}")
print(f"99% VaR: {var_99:.2%}")
print(f"95% ES: {es_95:.2%}")
print(f"Max Drawdown: {max_dd:.2%}")
```

### 6. Machine Learning

```python
# Feature engineering
engineer = ma.FeatureEngineer()
features = engineer.create_features(close_prices)

# LSTM prediction (if PyTorch is available)
try:
    predictor = ma.LSTMPredictor(sequence_length=10, epochs=50)
    
    # Prepare data
    X, y = ma.prepare_data_for_lstm(features.values, target_col=0)
    
    # Train model
    predictor.fit(X, y)
    
    # Make predictions
    predictions = predictor.predict(X[-10:])
    print(f"Next 5 predictions: {predictions[:5]}")
    
except ImportError:
    print("PyTorch not available. Install with: pip install torch")
```

##  Complete Example

Here's a complete example that demonstrates all major features:

```python
import meridianalgo as ma
import pandas as pd
import numpy as np

def comprehensive_example():
    """Comprehensive example showcasing MeridianAlgo features."""
    
    print(" MeridianAlgo Comprehensive Example")
    print("=" * 50)
    
    # 1. Get market data
    print("\n Fetching market data...")
    data = ma.get_market_data(['AAPL', 'MSFT', 'GOOGL'], start_date='2023-01-01')
    returns = data.pct_change().dropna()
    
    # 2. Technical Analysis
    print("\n Technical Analysis...")
    close = data['AAPL']
    rsi = ma.RSI(close, period=14)
    macd_line, signal_line, histogram = ma.MACD(close)
    bb_upper, bb_middle, bb_lower = ma.BollingerBands(close)
    
    print(f"  RSI: {rsi.iloc[-1]:.2f}")
    print(f"  MACD: {macd_line.iloc[-1]:.4f}")
    print(f"  Price vs BB: {close.iloc[-1]:.2f} (Upper: {bb_upper.iloc[-1]:.2f}, Lower: {bb_lower.iloc[-1]:.2f})")
    
    # 3. Portfolio Optimization
    print("\n Portfolio Optimization...")
    optimizer = ma.PortfolioOptimizer(returns)
    optimal = optimizer.optimize_portfolio(objective='sharpe')
    
    print(f"  Optimal weights: {optimal['weights']}")
    print(f"  Expected return: {optimal['return']:.2%}")
    print(f"  Volatility: {optimal['volatility']:.2%}")
    print(f"  Sharpe ratio: {optimal['sharpe_ratio']:.2f}")
    
    # 4. Risk Analysis
    print("\n Risk Analysis...")
    portfolio_returns = returns.mean(axis=1)
    var_95 = ma.calculate_value_at_risk(portfolio_returns, confidence_level=0.95)
    es_95 = ma.calculate_expected_shortfall(portfolio_returns, confidence_level=0.95)
    max_dd = ma.calculate_max_drawdown(portfolio_returns)
    
    print(f"  95% VaR: {var_95:.2%}")
    print(f"  95% ES: {es_95:.2%}")
    print(f"  Max Drawdown: {max_dd:.2%}")
    
    # 5. Statistical Analysis
    print("\n Statistical Analysis...")
    hurst = ma.hurst_exponent(close)
    autocorr = ma.calculate_autocorrelation(returns['AAPL'])
    
    print(f"  Hurst Exponent: {hurst:.3f}")
    print(f"  Autocorrelation: {autocorr:.3f}")
    
    print("\n Example completed successfully!")

# Run the example
if __name__ == "__main__":
    comprehensive_example()
```

##  Next Steps

1. **Explore the API**: Check out the [API Reference](api/) for detailed documentation
2. **Try Examples**: See [Examples](examples/) for more complex use cases
3. **Learn Advanced Features**: Dive into [Portfolio Management](api/portfolio_management.md) and [Risk Analysis](api/risk_analysis.md)
4. **Contribute**: Help improve MeridianAlgo by contributing to the project

##  Getting Help

- **Documentation**: Browse the comprehensive [API Reference](api/)
- **Examples**: Check out [example scripts](examples/)
- **Issues**: Report bugs or request features on [GitHub](https://github.com/MeridianAlgo/Python-Packages/issues)
- **Discussions**: Join the community on [GitHub Discussions](https://github.com/MeridianAlgo/Python-Packages/discussions)

##  Additional Resources

- [Installation Guide](installation.md) - Detailed installation instructions
- [Performance Benchmarks](benchmarks.md) - Performance metrics and comparisons
- [Contributing Guide](contributing.md) - How to contribute to the project
- [Changelog](changelog.md) - Version history and updates
