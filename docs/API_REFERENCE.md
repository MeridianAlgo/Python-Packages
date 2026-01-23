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

#### DataProvider Classes

```python
from meridianalgo.data import YahooFinanceProvider, AlphaVantageProvider

# Yahoo Finance Provider
yahoo = YahooFinanceProvider()
data = yahoo.get_historical_data(['AAPL', 'GOOGL'], '2023-01-01', '2023-12-31')

# Alpha Vantage Provider
alpha = AlphaVantageProvider(api_key='your_key')
data = alpha.get_intraday_data('AAPL', interval='1min')
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
from meridianalgo.technical_analysis.patterns import CandlestickPatterns, ChartPatterns

# Candlestick patterns
patterns = CandlestickPatterns()
doji = patterns.detect_doji(ohlc_data)
hammer = patterns.detect_hammer(ohlc_data)

# Chart patterns
chart_patterns = ChartPatterns()
triangles = chart_patterns.detect_triangles(price_data)
```

---

## Portfolio Management

### `meridianalgo.portfolio`

#### Portfolio Optimization

```python
from meridianalgo.portfolio import PortfolioOptimizer, BlackLittermanOptimizer

# Mean-Variance Optimization
optimizer = PortfolioOptimizer()
weights = optimizer.optimize(returns_data, method='mean_variance')

# Black-Litterman Model
bl_optimizer = BlackLittermanOptimizer()
bl_weights = bl_optimizer.optimize(returns_data, views=market_views)
```

#### Risk Management

```python
from meridianalgo.portfolio.risk_management import RiskManager

risk_manager = RiskManager()

# Value at Risk
var_95 = risk_manager.calculate_var(returns, confidence_level=0.95)
var_99 = risk_manager.calculate_var(returns, confidence_level=0.99)

# Expected Shortfall
es_95 = risk_manager.calculate_expected_shortfall(returns, confidence_level=0.95)
```

---

## Backtesting

### `meridianalgo.backtesting`

#### Event-Driven Backtesting

```python
from meridianalgo.backtesting import EventDrivenBacktester, Strategy

class MyStrategy(Strategy):
    def generate_signals(self, market_data):
        # Your strategy logic
        return signals

# Run backtest
backtester = EventDrivenBacktester(initial_capital=100000)
backtester.set_strategy(MyStrategy())
results = backtester.run_backtest(data)
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

### `meridianalgo.machine_learning`

#### Feature Engineering

```python
from meridianalgo.machine_learning.feature_engineering import FinancialFeatureEngineer

engineer = FinancialFeatureEngineer()
features = engineer.create_features(price_data)
```

#### Models

```python
from meridianalgo.machine_learning.models import LSTMPredictor, ModelFactory

# LSTM Model
lstm = LSTMPredictor(sequence_length=60, epochs=100)
lstm.fit(features, targets)
predictions = lstm.predict(test_features)

# Model Factory
model = ModelFactory.create_model('random_forest', n_estimators=100)
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

### `meridianalgo.risk_analysis`

#### Risk Metrics

```python
from meridianalgo.risk_analysis import VaRCalculator, StressTester

# VaR Calculation
var_calc = VaRCalculator()
historical_var = var_calc.historical_var(returns, confidence_level=0.95)
parametric_var = var_calc.parametric_var(returns, confidence_level=0.95)

# Stress Testing
stress_tester = StressTester()
stress_results = stress_tester.run_historical_scenarios(portfolio, scenarios)
```

---

##  Configuration

### Global Configuration

```python
import meridianalgo as ma

# Set global configuration
ma.config.set_data_provider('yahoo')
ma.config.set_cache_enabled(True)
ma.config.set_parallel_processing(True)

# API Keys
ma.config.set_api_key('alpha_vantage', 'your_key')
ma.config.set_api_key('quandl', 'your_key')
```

### Logging Configuration

```python
import meridianalgo as ma

# Enable logging
ma.logging.set_level('INFO')
ma.logging.enable_file_logging('meridianalgo.log')
```

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
optimizer = ma.PortfolioOptimizer()
weights = optimizer.optimize(returns, method='sharpe')

# Calculate risk metrics
risk_manager = ma.RiskManager()
var = risk_manager.calculate_var(returns)

print(f"Optimal weights: {weights}")
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

# Detect patterns
patterns = ma.detect_candlestick_patterns(prices)
```

### Machine Learning Pipeline

```python
import meridianalgo as ma

# Feature engineering
engineer = ma.FeatureEngineer()
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
# Use chunking for large datasets
for chunk in ma.data.read_large_dataset('data.csv', chunksize=10000):
    process_chunk(chunk)

# Enable memory mapping
ma.config.enable_memory_mapping(True)
```

---

##  Error Handling

### Exception Types

```python
from meridianalgo.exceptions import (
    DataError, CalculationError, ValidationError, BacktestError
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