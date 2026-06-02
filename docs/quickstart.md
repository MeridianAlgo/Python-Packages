# Quick Start Guide

Get up and running with MeridianAlgo in minutes.

## Installation

```bash
pip install meridianalgo

# Verify installation
python -c "import meridianalgo; print(meridianalgo.__version__)"
```

For machine learning, optimization, or volatility extras, install the matching
group (see [installation.md](installation.md)):

```bash
pip install "meridianalgo[ml]"
```

## Basic Usage

### 1. Import the library

```python
import meridianalgo as ma
from meridianalgo.signals.indicators import RSI, MACD, SMA, EMA, BollingerBands
```

### 2. Get market data

```python
data = ma.get_market_data(['AAPL', 'MSFT', 'GOOGL'], start_date='2023-01-01')
returns = data.pct_change().dropna()
print(f"Retrieved {data.shape[1]} tickers, {data.shape[0]} rows")
```

### 3. Technical analysis

```python
close = data['AAPL']

rsi = RSI(close, period=14)
macd_line, signal_line, histogram = MACD(close)
sma_20 = SMA(close, 20)
bb_upper, bb_middle, bb_lower = BollingerBands(close)

print(f"RSI: {rsi.iloc[-1]:.2f}")
print(f"MACD: {macd_line.iloc[-1]:.4f}")
```

The top-level shortcuts `ma.calculate_rsi` and `ma.calculate_macd` are also
available; the full indicator set lives in `meridianalgo.signals.indicators`.

### 4. Portfolio optimization

```python
optimizer = ma.PortfolioOptimizer(returns)
result = optimizer.optimize_portfolio(method='sharpe')

print(f"Weights: {result['weights']}")
print(f"Expected return: {result['return']:.2%}")
print(f"Volatility: {result['volatility']:.2%}")
print(f"Sharpe ratio: {result['sharpe_ratio']:.2f}")
```

### 5. Risk analysis

```python
portfolio_returns = returns.mean(axis=1)

var = ma.VaRCalculator(portfolio_returns)
print(f"95% VaR: {var.value_at_risk(confidence=0.95):.2%}")
print(f"95% CVaR: {var.conditional_var(confidence=0.95):.2%}")

es_95 = ma.calculate_expected_shortfall(portfolio_returns)
max_dd = ma.calculate_max_drawdown(portfolio_returns)
print(f"95% ES: {es_95:.2%}")
print(f"Max Drawdown: {max_dd:.2%}")
```

### 6. Performance summary

```python
# One-call summary of ~28 performance and risk statistics
stats = ma.summary_stats(portfolio_returns)
print(stats['sharpe_ratio'], stats['max_drawdown'])

# Formatted, human-readable report
print(ma.tearsheet(portfolio_returns))
```

### 7. Machine learning (requires the `ml` extra)

```python
try:
    predictor = ma.LSTMPredictor(sequence_length=10, epochs=50)
    X, y = ma.prepare_data_for_lstm(close.values, target_col=0)
    predictor.fit(X, y)
    predictions = predictor.predict(X[-10:])
    print(f"Predictions: {predictions[:5]}")
except ImportError:
    print("PyTorch not available. Install with: pip install 'meridianalgo[ml]'")
```

## Checking Module Availability

```python
import meridianalgo as ma
print(ma.ModuleRegistry.status())
```

## Next Steps

- [API Reference](API_REFERENCE.md) — complete module and function reference
- [Technical Indicators](api/technical_indicators.md) — full indicator catalog
- [Portfolio Management](api/portfolio_management.md) — optimization in depth
- [Performance Benchmarks](benchmarks.md) — speed and accuracy metrics

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/MeridianAlgo/Python-Packages/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MeridianAlgo/Python-Packages/discussions)
