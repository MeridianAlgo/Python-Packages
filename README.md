# MeridianAlgo
## A Powerful Python Library for Quantitative Finance

[![PyPI version](https://img.shields.io/pypi/v/meridianalgo.svg?style=flat-square&color=blue)](https://pypi.org/project/meridianalgo/)
[![Python versions](https://img.shields.io/pypi/pyversions/meridianalgo.svg?style=flat-square)](https://pypi.org/project/meridianalgo/)
[![License](https://img.shields.io/github/license/MeridianAlgo/Python-Packages.svg?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/MeridianAlgo/Python-Packages/ci.yml?branch=main&style=flat-square&label=tests)](https://github.com/MeridianAlgo/Python-Packages/actions)

**MeridianAlgo** is a comprehensive Python library for quantitative finance, algorithmic trading, and financial machine learning. Whether you're a retail trader, quant researcher, or financial analyst, this library provides the tools you need to analyze markets, build strategies, and manage risk.

---

## Installation

```bash
pip install meridianalgo
```

For full functionality with all optional dependencies:
```bash
pip install meridianalgo[all]
```

---

## Quick Examples

### 🚀 Market Data & Basic Analysis

```python
import meridianalgo as ma
import pandas as pd

# Get stock data
prices = ma.get_market_data(['AAPL', 'MSFT', 'GOOGL'], start='2020-01-01')
returns = prices.pct_change().dropna()

# Calculate technical indicators
rsi = ma.calculate_rsi(prices['AAPL'], window=14)
macd = ma.calculate_macd(prices['MSFT'])
bollinger = ma.calculate_bollinger_bands(prices['GOOGL'])

# Basic statistics
print(f"AAPL Sharpe Ratio: {ma.calculate_sharpe_ratio(returns['AAPL']):.2f}")
print(f"MSFT Max Drawdown: {ma.calculate_max_drawdown(returns['MSFT']):.2%}")
```

### 📊 Portfolio Optimization

```python
from meridianalgo.portfolio import PortfolioOptimizer

# Create portfolio optimizer
opt = PortfolioOptimizer(returns)

# Different optimization methods
hrp_weights = opt.optimize(method='hrp')  # Hierarchical Risk Parity
min_var_weights = opt.optimize(method='min_variance')  # Minimum Variance
risk_parity_weights = opt.optimize(method='risk_parity')  # Risk Parity

print("Hierarchical Risk Parity Weights:")
print(hrp_weights.sort_values(ascending=False).head())

# Portfolio metrics
portfolio_return = (returns * hrp_weights).sum(axis=1)
portfolio_vol = portfolio_return.std() * (252 ** 0.5)
sharpe_ratio = portfolio_return.mean() / portfolio_return.std() * (252 ** 0.5)

print(f"Portfolio Annual Return: {portfolio_return.mean() * 252:.2%}")
print(f"Portfolio Annual Volatility: {portfolio_vol:.2%}")
print(f"Portfolio Sharpe Ratio: {sharpe_ratio:.2f}")
```

### ⚡ Risk Management

```python
from meridianalgo.risk import RiskMetrics

# Calculate various risk metrics
risk = RiskMetrics(returns)

# Value at Risk (VaR)
var_95 = risk.calculate_var(level=0.95, method='historical')
var_99 = risk.calculate_var(level=0.99, method='gaussian')

# Conditional VaR (Expected Shortfall)
cvar_95 = risk.calculate_cvar(level=0.95)

# Cornish-Fisher adjusted VaR (accounts for skewness/kurtosis)
cf_var = risk.calculate_cornish_fisher_var(level=0.95)

print(f"95% VaR (Historical): {var_95['AAPL']:.2%}")
print(f"99% VaR (Gaussian): {var_99['AAPL']:.2%}")
print(f"95% CVaR: {cvar_95['AAPL']:.2%}")
print(f"95% Cornish-Fisher VaR: {cf_var['AAPL']:.2%}")
```

### 🤖 Machine Learning for Finance

```python
from meridianalgo.ml import FeatureEngineer, ModelValidator
from sklearn.ensemble import RandomForestClassifier

# Engineer features
fe = FeatureEngineer()
features = fe.create_features(prices, 
    features=['returns', 'rsi', 'macd', 'bollinger_position', 'volume_ratio'])

# Prepare labels (next day direction)
labels = (returns.shift(-1) > 0).astype(int)

# Walk-forward validation
validator = ModelValidator()
results = validator.walk_forward_validation(
    features, labels, 
    model=RandomForestClassifier(n_estimators=100),
    train_window=252,  # 1 year training
    test_window=21     # 1 month testing
)

print(f"Average Accuracy: {results['accuracy'].mean():.2%}")
print(f"Average Precision: {results['precision'].mean():.2%}")
print(f"Average F1 Score: {results['f1_score'].mean():.2%}")
```

### 📈 Statistical Arbitrage

```python
from meridianalgo.quant import StatisticalArbitrage

# Find cointegrated pairs
pairs_trader = StatisticalArbitrage()
pairs = pairs_trader.find_cointegrated_pairs(prices, p_value=0.05)

print("Cointegrated Pairs:")
for pair, p_val in pairs.items():
    print(f"{pair}: p-value = {p_val:.4f}")

# Calculate half-life for mean reversion
if pairs:
    pair = list(pairs.keys())[0]
    spread = prices[pair[0]] - prices[pair[1]]
    half_life = pairs_trader.calculate_half_life(spread)
    print(f"{pair} Half-Life: {half_life:.1f} days")
    
    # Generate trading signals
    signals = pairs_trader.generate_pairs_signals(spread, z_entry=2.0, z_exit=0.5)
    print(f"Generated {len(signals)} trading signals")
```

### 💰 Options Pricing & Greeks

```python
from meridianalgo.derivatives import BlackScholes, GreeksCalculator, ImpliedVolatility

# Option pricing
call_price = BlackScholes.call_price(S=100, K=105, T=0.25, r=0.05, sigma=0.2)
put_price = BlackScholes.put_price(S=100, K=105, T=0.25, r=0.05, sigma=0.2)

# Greeks
delta = GreeksCalculator.delta('call', S=100, K=105, T=0.25, r=0.05, sigma=0.2)
gamma = GreeksCalculator.gamma(S=100, K=105, T=0.25, r=0.05, sigma=0.2)
theta = GreeksCalculator.theta('call', S=100, K=105, T=0.25, r=0.05, sigma=0.2)
vega = GreeksCalculator.vega(S=100, K=105, T=0.25, r=0.05, sigma=0.2)

print(f"Call Price: ${call_price:.2f}")
print(f"Put Price: ${put_price:.2f}")
print(f"Delta: {delta:.3f}")
print(f"Gamma: {gamma:.3f}")
print(f"Theta: {theta:.3f}")
print(f"Vega: {vega:.3f}")

# Implied volatility
market_price = 5.50
implied_vol = ImpliedVolatility.calculate('call', S=100, K=105, T=0.25, r=0.05, market_price=market_price)
print(f"Implied Volatility: {implied_vol:.2%}")
```

### ⚡ Backtesting Engine

```python
from meridianalgo.backtesting import Backtester, Strategy

# Simple moving average crossover strategy
class MACrossover(Strategy):
    def __init__(self, short_window=20, long_window=50):
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index, columns=data.columns)
        
        for asset in data.columns:
            short_ma = data[asset].rolling(self.short_window).mean()
            long_ma = data[asset].rolling(self.long_window).mean()
            
            # Buy when short MA crosses above long MA
            signals[asset] = (short_ma > long_ma).astype(int)
            
        return signals

# Run backtest
strategy = MACrossover(short_window=20, long_window=50)
backtest = Backtester(strategy, prices, returns)
results = backtest.run()

print(f"Total Return: {results['total_return']:.2%}")
print(f"Annual Return: {results['annual_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Win Rate: {results['win_rate']:.2%}")
```

### 📊 Advanced Technical Analysis

```python
# Advanced indicators
from meridianalgo.signals import TechnicalIndicators

ti = TechnicalIndicators()

# Multiple timeframes
data_1h = ma.get_market_data(['BTC-USD'], interval='1h', start='2024-01-01')
data_1d = ma.get_market_data(['BTC-USD'], interval='1d', start='2024-01-01')

# Calculate indicators
indicators = {
    'RSI': ti.rsi(data_1h['BTC-USD'], window=14),
    'MACD': ti.macd(data_1h['BTC-USD']),
    'ATR': ti.atr(data_1h['BTC-USD'], window=14),
    'Stochastic': ti.stochastic(data_1h['BTC-USD'], k_window=14, d_window=3),
    'Williams %R': ti.williams_r(data_1h['BTC-USD'], window=14),
    'CCI': ti.cci(data_1h['BTC-USD'], window=20),
    'MFI': ti.mfi(data_1h['BTC-USD'], window=14),
    'OBV': ti.obv(data_1h['BTC-USD'])
}

# Generate composite signal
def generate_composite_signal(indicators):
    signals = pd.Series(0, index=indicators['RSI'].index)
    
    # RSI oversold/overbought
    signals[indicators['RSI'] < 30] += 1  # Buy signal
    signals[indicators['RSI'] > 70] -= 1  # Sell signal
    
    # MACD crossover
    macd_signal = indicators['MACD']['MACD'] > indicators['MACD']['Signal']
    signals[macd_signal] += 1
    signals[~macd_signal] -= 1
    
    return signals

composite_signal = generate_composite_signal(indicators)
print(f"Generated {len(composite_signal)} composite signals")
```

---

## Performance Benchmarks

*Tested on Intel i7-10700K, 32GB RAM, Python 3.11*

| Operation | Dataset Size | Time | Memory Usage |
|-----------|--------------|------|--------------|
| Portfolio Optimization (HRP) | 100 assets, 5 years | 45ms | 12MB |
| VaR Calculation (Historical) | 500 assets, 10 years | 120ms | 8MB |
| Backtest (Simple Strategy) | 50 assets, 5 years | 200ms | 15MB |
| Options Greeks (1000 contracts) | 1000 contracts | 35ms | 5MB |
| Feature Engineering | 10 assets, 3 years | 180ms | 20MB |

---

## Data Sources

MeridianAlgo supports multiple data providers:

```python
# Free data sources
yf_data = ma.get_yahoo_data(['AAPL', 'MSFT'])  # Yahoo Finance
fred_data = ma.get_fred_data(['GDP', 'CPI'])   # FRED

# Premium data (API keys required)
polygon_data = ma.get_polygon_data(['SPY'], api_key='your_key')
alpha_vantage = ma.get_alpha_vantage_data(['TSLA'], api_key='your_key')

# Crypto data
crypto_data = ma.get_crypto_data(['BTC-USD', 'ETH-USD'])
```

---

## Configuration & Utilities

```python
# Setup logging
from meridianalgo.utils import setup_logger
logger = setup_logger('trading_bot', level='INFO')

# Data validation
from meridianalgo.utils import DataValidator
validator = DataValidator()
validator.check_missing_values(prices)
validator.check_outliers(returns, threshold=3)

# Performance monitoring
from meridianalgo.utils import PerformanceTimer
with PerformanceTimer('portfolio_optimization'):
    weights = opt.optimize(method='hrp')
```

---

## Optional Dependencies

Install specific functionality:

```bash
# Machine learning
pip install meridianalgo[ml]

# Portfolio optimization
pip install meridianalgo[optimization]

# Volatility modeling
pip install meridianalgo[volatility]

# Data providers
pip install meridianalgo[data]

# Distributed computing
pip install meridianalgo[distributed]

# Everything
pip install meridianalgo[all]
```

---

## Documentation

- **Full API Documentation**: [meridianalgo.readthedocs.io](https://meridianalgo.readthedocs.io)
- **Examples Gallery**: [github.com/MeridianAlgo/Python-Packages/tree/main/examples](https://github.com/MeridianAlgo/Python-Packages/tree/main/examples)
- **Research Papers**: [github.com/MeridianAlgo/Python-Packages/tree/main/docs/research](https://github.com/MeridianAlgo/Python-Packages/tree/main/docs/research)

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Disclaimer

This software is for educational and research purposes. Trading financial instruments involves substantial risk of loss. The authors are not responsible for any financial losses incurred through the use of this software. Always do your own research and consider consulting with a financial advisor before making investment decisions.

---

**Built with ❤️ by the Quantitative Finance Community**
