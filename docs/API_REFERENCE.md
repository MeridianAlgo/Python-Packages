# API Reference - MeridianAlgo v4.0.0

##  Complete API Documentation

This document provides comprehensive API reference for all MeridianAlgo modules.

##  Module Overview

### Core Modules
- [Data Infrastructure](#data-infrastructure) - Multi-source data providers and processing
- [Technical Analysis](#technical-analysis) - 200+ indicators and pattern recognition
- [Portfolio Management](#portfolio-management) - Optimization and risk management
- [Backtesting](#backtesting) - Event-driven backtesting engine
- [Machine Learning](#machine-learning) - Financial ML models and features
- [Fixed Income](#fixed-income) - Bond pricing and derivatives
- [Risk Analysis](#risk-analysis) - Risk metrics and compliance

---

## Data Infrastructure

### `meridianalgo.data`

#### Fetching Market Data

```python
import meridianalgo as ma

# Free historical data via yfinance
data = ma.get_market_data(["AAPL", "GOOGL"], start_date="2023-01-01", end_date="2023-12-31")
```

#### Data Processing Pipeline

```python
from meridianalgo.data.processing import DataPipeline, DataValidator, OutlierDetector

# Create processing pipeline
pipeline = DataPipeline([
    DataValidator(strict=False),
    OutlierDetector(method='iqr'),
    MissingDataHandler(method='forward_fill')
])

# Process data
clean_data = pipeline.fit_transform(raw_data)
```

---

## Technical Analysis

### `meridianalgo.technical_analysis`

#### Indicators

```python
from meridianalgo.technical_analysis import RSI, MACD, BollingerBands

# RSI Indicator
rsi = RSI(period=14)
rsi_values = rsi.calculate(price_data)

# MACD
macd = MACD(fast=12, slow=26, signal=9)
macd_line, signal_line, histogram = macd.calculate(price_data)

# Bollinger Bands
bb = BollingerBands(period=20, std_dev=2)
upper, middle, lower = bb.calculate(price_data)
```

#### Pattern Recognition

```python
from meridianalgo.technical_analysis.patterns import CandlestickPatterns

# Candlestick patterns (ohlc_data must have Open/High/Low/Close columns)
patterns = CandlestickPatterns()
doji = patterns.detect_doji(ohlc_data)
hammer = patterns.detect_hammer(ohlc_data)
```

---

## Portfolio Management

### `meridianalgo.portfolio`

#### Portfolio Optimization

```python
import meridianalgo as ma

# Mean-Variance / Max-Sharpe optimization
optimizer = ma.PortfolioOptimizer(returns_data)
result = optimizer.optimize_portfolio(method='sharpe')
weights = result['weights']

# Other methods: 'min_vol', 'max_return', 'risk_parity', 'hrp', 'equal_weight'
min_vol = optimizer.optimize_portfolio(method='min_vol')
```

#### Risk Management

```python
import meridianalgo as ma

var = ma.VaRCalculator(returns)

# Value at Risk
var_95 = var.value_at_risk(confidence=0.95)
var_99 = var.value_at_risk(confidence=0.99)

# Conditional VaR (Expected Shortfall)
cvar_95 = var.conditional_var(confidence=0.95)
```

---

## Backtesting

### `meridianalgo.backtesting`

#### Event-Driven Backtesting

```python
from meridianalgo.backtesting import BacktestEngine, Strategy

class MyStrategy(Strategy):
    def generate_signals(self, data):
        # Your strategy logic -> DataFrame of signals
        return signals

# Run backtest
engine = BacktestEngine(initial_capital=100_000)
results = engine.run(MyStrategy(), prices, returns)
```

#### Performance Analytics

```python
from meridianalgo.backtesting.performance_analytics import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()
metrics = analyzer.analyze_returns(strategy_returns)

print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
```

---

## Machine Learning

### `meridianalgo.ml`

#### Feature Engineering

```python
from meridianalgo.ml import FeatureEngineer

engineer = FeatureEngineer()
features = engineer.create_features(price_data)
```

#### Models

```python
from meridianalgo.ml import LSTMPredictor, ModelFactory

# LSTM Model
lstm = LSTMPredictor(sequence_length=60, epochs=100)
lstm.fit(features, targets)
predictions = lstm.predict(test_features)

# Model Factory
model = ModelFactory.create_model('random_forest')
```

---

## Fixed Income

### `meridianalgo.fixed_income`

#### Bond Pricing

```python
from meridianalgo.fixed_income.bonds import BondPricer, YieldCurve

# Yield Curve
curve = YieldCurve.from_treasury_rates(rates_data)

# Bond Pricing
pricer = BondPricer(yield_curve=curve)
bond_price = pricer.price_bond(coupon=0.05, maturity=10, face_value=1000)
```

#### Options Pricing

```python
from meridianalgo.fixed_income.options import BlackScholesModel, MonteCarloModel

# Black-Scholes
bs = BlackScholesModel()
option_price = bs.price_option(
    spot=100, strike=105, time_to_expiry=0.25, 
    risk_free_rate=0.05, volatility=0.2, option_type='call'
)

# Greeks
greeks = bs.calculate_greeks(spot=100, strike=105, time_to_expiry=0.25)
```

---

## Risk Analysis

### `meridianalgo.risk`

#### Risk Metrics

```python
import meridianalgo as ma

# VaR / CVaR
var_calc = ma.VaRCalculator(returns)
historical_var = var_calc.value_at_risk(confidence=0.95, method="historical")
parametric_var = var_calc.value_at_risk(confidence=0.95, method="parametric")
cvar = var_calc.conditional_var(confidence=0.95)

# Stress testing
from meridianalgo import StressTesting
stress = StressTesting()
```

---

## Module Availability

Every submodule loads behind a registry, so the package imports even when an
optional dependency is missing. Check what is available at runtime:

```python
import meridianalgo as ma

print(ma.ModuleRegistry.status())          # {'core': True, 'ml': True, ...}
print(ma.ModuleRegistry.is_available("ml"))
```

Install the matching extra (e.g. `pip install "meridianalgo[ml]"`) to enable a
module reported as unavailable.

---

##  Quick Start Examples

### Basic Portfolio Analysis

```python
import meridianalgo as ma

# Get data
data = ma.get_market_data(['AAPL', 'GOOGL', 'MSFT'], '2023-01-01')

# Calculate returns
returns = data.pct_change().dropna()

# Optimize portfolio
optimizer = ma.PortfolioOptimizer(returns)
result = optimizer.optimize_portfolio(method='sharpe')

# Calculate risk metrics
risk = ma.RiskAnalyzer(returns.mean(axis=1))
var = risk.value_at_risk(confidence=0.95)

print(f"Optimal weights: {result['weights']}")
print(f"Portfolio VaR: {var:.4f}")
```

### Technical Analysis

```python
import meridianalgo as ma

# Get price data
prices = ma.get_market_data(['AAPL'], '2023-01-01')['AAPL']

# Calculate indicators
rsi = ma.RSI(prices, period=14)
macd_line, signal, histogram = ma.MACD(prices)
bb_upper, bb_middle, bb_lower = ma.BollingerBands(prices)
sma_20 = ma.SMA(prices, period=20)
```

### Machine Learning Pipeline

```python
import meridianalgo as ma

# Feature engineering
from meridianalgo.ml import FeatureEngineer

engineer = FeatureEngineer()
features = engineer.create_features(price_data)

# Train model
model = ma.LSTMPredictor(sequence_length=60)
model.fit(features, targets)

# Make predictions
predictions = model.predict(test_features)
```

---

##  Performance Considerations

### Optimization Tips

1. **Use vectorized operations** for large datasets
2. **Enable caching** for frequently accessed data
3. **Use parallel processing** for independent calculations
4. **Consider GPU acceleration** for ML models

### Memory Management

```python
import pandas as pd

# Process large CSVs in chunks to bound memory use
for chunk in pd.read_csv("data.csv", chunksize=10_000):
    process_chunk(chunk)
```

---

##  Error Handling

### Exception Types

```python
from meridianalgo.data.exceptions import (
    DataError, ValidationError, NetworkError, RateLimitError
)

try:
    data = ma.get_market_data(['INVALID_SYMBOL'])
except DataError as e:
    print(f"Data error: {e}")
except ValidationError as e:
    print(f"Validation error: {e}")
```

---

##  Support

For detailed API documentation and examples:
- **Online Docs**: [docs.meridianalgo.com](https://docs.meridianalgo.com)
- **GitHub**: [github.com/MeridianAlgo/Python-Packages](https://github.com/MeridianAlgo/Python-Packages)
- **Support**: support@meridianalgo.com

---

*MeridianAlgo v4.0.0 - Complete API Reference*