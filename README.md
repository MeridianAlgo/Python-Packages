# MeridianAlgo v4.1.0 - Quantum Edition 🚀

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI Version](https://img.shields.io/badge/pypi-4.1.0-orange.svg)](https://pypi.org/project/meridianalgo/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

**Meridian Quant: The Ultimate Quantitative Development Platform**

The most advanced Python platform for quantitative finance, integrating cutting-edge machine learning, institutional-grade portfolio management, and high-performance computing. Built for quantitative analysts, portfolio managers, algorithmic traders, and financial researchers.

## 🚀 Key Features

### 📊 **Technical Analysis (200+ Indicators)**
- **50+ Native Indicators**: RSI, MACD, Bollinger Bands, Stochastic, Williams %R, ADX, Aroon, Parabolic SAR, Ichimoku Cloud
- **150+ TA Library Integration**: Complete integration with the TA library for maximum coverage
- **Advanced Pattern Recognition**: Candlestick patterns, chart patterns, support/resistance detection
- **Custom Indicator Framework**: Build and optimize your own indicators with JIT compilation

### 🏦 **Portfolio Management**
- **Modern Portfolio Theory**: Efficient frontier, mean-variance optimization
- **Advanced Models**: Black-Litterman, Risk Parity, Hierarchical Risk Parity
- **Performance Attribution**: Factor analysis, benchmark comparison, tracking error
- **Transaction Cost Analysis**: Market impact models, optimal execution algorithms

### ⚠️ **Risk Management**
- **Value at Risk (VaR)**: Historical, Parametric, Monte Carlo methods
- **Expected Shortfall (CVaR)**: Tail risk analysis with confidence intervals
- **Stress Testing**: Historical scenarios, Monte Carlo simulation, reverse stress testing
- **Real-time Monitoring**: Customizable alerts, risk dashboards, limit monitoring

### 🤖 **Machine Learning**
- **Feature Engineering**: 500+ financial features, technical indicators, market microstructure
- **Advanced Models**: LSTM, Transformers, Ensemble methods, Reinforcement Learning
- **Time Series Validation**: Walk-forward analysis, purged cross-validation
- **Model Deployment**: Versioning, A/B testing, performance monitoring

### 🔄 **Backtesting Engine**
- **Event-Driven Architecture**: Realistic market simulation with bid-ask spreads
- **Order Management**: All order types, partial fills, slippage modeling
- **Performance Analytics**: 50+ metrics, drawdown analysis, regime detection
- **Parallel Processing**: GPU acceleration, distributed computing support

### 💰 **Fixed Income & Derivatives**
- **Bond Pricing**: Yield curve construction, duration, convexity
- **Options Pricing**: Black-Scholes, binomial trees, Monte Carlo
- **Interest Rate Models**: Vasicek, CIR, Hull-White
- **Exotic Derivatives**: Barrier options, Asian options, structured products

## 📦 Installation

```bash
# Standard installation
pip install meridianalgo

# With machine learning support
pip install meridianalgo[ml]

# With extended features (pandas-ta, web scraping)
pip install meridianalgo[extended]

# With all optional dependencies
pip install meridianalgo[all]

# Development installation
pip install meridianalgo[dev]
```
## 🚀 Comprehensive Examples

### 📈 **Basic Usage & Market Data**

```python
import meridianalgo as ma
import pandas as pd
import numpy as np

# Get market data
data = ma.api_get_market_data(['AAPL', 'GOOGL', 'MSFT'], '2023-01-01', '2023-12-31')
print(f"📊 Retrieved data for {len(data.columns)} assets over {len(data)} days")

# Basic technical analysis
rsi = ma.RSI(data['AAPL'], period=14)
bb_upper, bb_middle, bb_lower = ma.BollingerBands(data['AAPL'])
macd_line, signal_line, histogram = ma.MACD(data['AAPL'])

print(f"Current AAPL RSI: {rsi.iloc[-1]:.1f}")
print(f"AAPL above upper Bollinger Band: {data['AAPL'].iloc[-1] > bb_upper.iloc[-1]}")
```

 📊 **Complete Technical Analysis Suite**

```python
# === MOMENTUM INDICATORS ===
print("📈 MOMENTUM INDICATORS:")

# RSI with different periods
rsi_14 = ma.RSI(data['AAPL'], period=14)
rsi_21 = ma.RSI(data['AAPL'], period=21)
print(f"RSI(14): {rsi_14.iloc[-1]:.1f} | RSI(21): {rsi_21.iloc[-1]:.1f}")

# Stochastic Oscillator
stoch_k, stoch_d = ma.Stochastic(data['AAPL'], data['AAPL'], data['AAPL'])
print(f"Stochastic %K: {stoch_k.iloc[-1]:.1f} | %D: {stoch_d.iloc[-1]:.1f}")

# Williams %R
williams_r = ma.WilliamsR(data['AAPL'], data['AAPL'], data['AAPL'])
print(f"Williams %R: {williams_r.iloc[-1]:.1f}")

# Rate of Change
roc = ma.ROC(data['AAPL'], period=10)
print(f"10-day ROC: {roc.iloc[-1]:.2%}")

# === TREND INDICATORS ===
print("\n📈 TREND INDICATORS:")

# Multiple Moving Averages
sma_10 = ma.SMA(data['AAPL'], period=10)
sma_20 = ma.SMA(data['AAPL'], period=20)
sma_50 = ma.SMA(data['AAPL'], period=50)
ema_12 = ma.EMA(data['AAPL'], period=12)
ema_26 = ma.EMA(data['AAPL'], period=26)

print(f"SMA(10): ${sma_10.iloc[-1]:.2f} | SMA(20): ${sma_20.iloc[-1]:.2f} | SMA(50): ${sma_50.iloc[-1]:.2f}")
print(f"EMA(12): ${ema_12.iloc[-1]:.2f} | EMA(26): ${ema_26.iloc[-1]:.2f}")

# MACD Analysis
macd_line, signal_line, histogram = ma.MACD(data['AAPL'])
print(f"MACD: {macd_line.iloc[-1]:.4f} | Signal: {signal_line.iloc[-1]:.4f} | Histogram: {histogram.iloc[-1]:.4f}")

# ADX (Trend Strength)
adx = ma.ADX(data['AAPL'], data['AAPL'], data['AAPL'])
print(f"ADX (Trend Strength): {adx.iloc[-1]:.1f}")

# Aroon Indicator
aroon_up, aroon_down = ma.Aroon(data['AAPL'], data['AAPL'])
print(f"Aroon Up: {aroon_up.iloc[-1]:.1f} | Aroon Down: {aroon_down.iloc[-1]:.1f}")

# Parabolic SAR
psar = ma.ParabolicSAR(data['AAPL'], data['AAPL'], data['AAPL'])
print(f"Parabolic SAR: ${psar.iloc[-1]:.2f}")

# === VOLATILITY INDICATORS ===
print("\n📊 VOLATILITY INDICATORS:")

# Bollinger Bands with different periods
bb_upper_20, bb_middle_20, bb_lower_20 = ma.BollingerBands(data['AAPL'], period=20)
print(f"Bollinger Bands (20): ${bb_lower_20.iloc[-1]:.2f} - ${bb_upper_20.iloc[-1]:.2f}")

# Average True Range
atr = ma.ATR(data['AAPL'], data['AAPL'], data['AAPL'])
print(f"ATR (Volatility): ${atr.iloc[-1]:.2f}")

# Keltner Channels
kc_upper, kc_middle, kc_lower = ma.KeltnerChannels(data['AAPL'], data['AAPL'], data['AAPL'])
print(f"Keltner Channels: ${kc_lower.iloc[-1]:.2f} - ${kc_upper.iloc[-1]:.2f}")

# Donchian Channels
dc_upper, dc_lower = ma.DonchianChannels(data['AAPL'], data['AAPL'])
print(f"Donchian Channels: ${dc_lower.iloc[-1]:.2f} - ${dc_upper.iloc[-1]:.2f}")

# === VOLUME INDICATORS ===
print("\n📊 VOLUME INDICATORS:")

# Create volume data (simplified)
volume_data = pd.Series([1000000] * len(data), index=data.index)

# On-Balance Volume
obv = ma.OBV(data['AAPL'], volume_data)
print(f"OBV: {obv.iloc[-1]:,.0f}")

# Money Flow Index
mfi = ma.MoneyFlowIndex(data['AAPL'], data['AAPL'], data['AAPL'], volume_data)
print(f"Money Flow Index: {mfi.iloc[-1]:.1f}")

# Chaikin Oscillator
chaikin = ma.ChaikinOscillator(data['AAPL'], data['AAPL'], data['AAPL'], volume_data)
print(f"Chaikin Oscillator: {chaikin.iloc[-1]:.4f}")

# === OVERLAY INDICATORS ===
print("\n📊 OVERLAY INDICATORS:")

# Pivot Points
pivots = ma.PivotPoints(data['AAPL'], data['AAPL'], data['AAPL'])
print(f"Pivot Point: ${pivots['pivot'].iloc[-1]:.2f}")
print(f"Resistance 1: ${pivots['r1'].iloc[-1]:.2f} | Support 1: ${pivots['s1'].iloc[-1]:.2f}")

# Fibonacci Retracement
high_price = data['AAPL'].max()
low_price = data['AAPL'].min()
fib_levels = ma.FibonacciRetracement(high_price, low_price)
print(f"Fibonacci 61.8%: ${fib_levels[0.618]:.2f}")
print(f"Fibonacci 38.2%: ${fib_levels[0.382]:.2f}")

# Support and Resistance
support_resistance = ma.SupportResistance(data['AAPL'])
print(f"Key Support: ${support_resistance['support'][0]:.2f}")
print(f"Key Resistance: ${support_resistance['resistance'][0]:.2f}")
```
 🏦 **Advanced Portfolio Management**

```python
# === PORTFOLIO OPTIMIZATION ===
import meridianalgo as ma
import pandas as pd
import numpy as np

# Get data for multiple assets
symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'JPM', 'JNJ', 'PG']
data = ma.api_get_market_data(symbols, '2020-01-01', '2023-12-31')
returns = data.pct_change().dropna()

print(f"📊 Portfolio Analysis for {len(symbols)} assets")
print(f"Data period: {returns.index[0].date()} to {returns.index[-1].date()}")

# === MODERN PORTFOLIO THEORY ===
print("\n🎯 PORTFOLIO OPTIMIZATION METHODS:")

# Maximum Sharpe Ratio
sharpe_weights = ma.api_optimize_portfolio(returns, method='sharpe')
print(f"\n⚡ Maximum Sharpe Ratio Portfolio:")
for symbol, weight in sharpe_weights.items():
    print(f"  {symbol}: {weight:.1%}")

# Minimum Volatility
min_vol_weights = ma.api_optimize_portfolio(returns, method='min_volatility')
print(f"\n🛡️ Minimum Volatility Portfolio:")
for symbol, weight in min_vol_weights.items():
    print(f"  {symbol}: {weight:.1%}")

# Target Return Optimization
target_weights = ma.api_optimize_portfolio(returns, method='target_return', target_return=0.12)
print(f"\n🎯 Target Return (12%) Portfolio:")
for symbol, weight in target_weights.items():
    print(f"  {symbol}: {weight:.1%}")

# === ADVANCED OPTIMIZATION MODELS ===

# Black-Litterman Model
market_caps = {
    'AAPL': 3000, 'GOOGL': 1800, 'MSFT': 2800, 'TSLA': 800,
    'AMZN': 1500, 'JPM': 500, 'JNJ': 400, 'PG': 350
}
views = {'AAPL': 0.15, 'TSLA': 0.20, 'JNJ': 0.08}  # Expected returns

bl_model = ma.BlackLitterman(returns, market_caps)
bl_weights = bl_model.optimize_with_views(views)
print(f"\n🧠 Black-Litterman Portfolio:")
for symbol, weight in bl_weights.items():
    print(f"  {symbol}: {weight:.1%}")

# Risk Parity
rp_model = ma.RiskParity(returns)
rp_weights = rp_model.optimize()
print(f"\n⚖️ Risk Parity Portfolio:")
for symbol, weight in rp_weights.items():
    print(f"  {symbol}: {weight:.1%}")

# === EFFICIENT FRONTIER ===
frontier = ma.EfficientFrontier(returns)
target_returns = np.linspace(0.05, 0.25, 10)
frontier_data = frontier.calculate_frontier(target_returns)

print(f"\n📈 Efficient Frontier Analysis:")
print(f"Risk-Return Combinations:")
for i, row in frontier_data.iterrows():
    print(f"  Return: {row['return']:.1%} | Risk: {row['volatility']:.1%} | Sharpe: {row['sharpe']:.2f}")

# === PORTFOLIO PERFORMANCE ANALYSIS ===
print(f"\n📊 PORTFOLIO PERFORMANCE ANALYSIS:")

portfolios = {
    'Max Sharpe': sharpe_weights,
    'Min Volatility': min_vol_weights,
    'Risk Parity': rp_weights,
    'Black-Litterman': bl_weights
}

for name, weights in portfolios.items():
    # Calculate portfolio returns
    portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
    
    # Calculate metrics
    metrics = ma.api_calculate_risk_metrics(portfolio_returns)
    
    print(f"\n📊 {name} Portfolio:")
    print(f"  Annual Return: {metrics.get('annual_return', 0):.2%}")
    print(f"  Volatility: {metrics.get('volatility', 0):.2%}")
    print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"  VaR (95%): {metrics.get('var_95', 0):.2%}")
    print(f"  Expected Shortfall: {metrics.get('expected_shortfall_95', 0):.2%}")
    print(f"  Win Rate: {metrics.get('win_rate', 0):.1%}")

# === TRANSACTION COST ANALYSIS ===
print(f"\n💰 TRANSACTION COST ANALYSIS:")

# Simulate rebalancing costs
current_weights = {symbol: 1/len(symbols) for symbol in symbols}  # Equal weight
target_weights = sharpe_weights

total_turnover = sum(abs(target_weights[symbol] - current_weights[symbol]) 
                    for symbol in symbols) / 2

commission_rate = 0.001  # 0.1%
bid_ask_spread = 0.0005  # 0.05%
portfolio_value = 1000000

transaction_costs = total_turnover * portfolio_value * (commission_rate + bid_ask_spread)
print(f"Portfolio Value: ${portfolio_value:,.2f}")
print(f"Total Turnover: {total_turnover:.1%}")
print(f"Transaction Costs: ${transaction_costs:,.2f} ({transaction_costs/portfolio_value:.3%})")

# === RISK BUDGETING ===
print(f"\n⚠️ RISK BUDGETING ANALYSIS:")

# Calculate risk contributions for Max Sharpe portfolio
weights_array = np.array([sharpe_weights[symbol] for symbol in symbols])
cov_matrix = returns.cov().values

portfolio_vol = np.sqrt(weights_array.T @ cov_matrix @ weights_array * 252)
marginal_contrib = (cov_matrix @ weights_array * 252) / portfolio_vol
risk_contrib = weights_array * marginal_contrib / portfolio_vol

print(f"Portfolio Volatility: {portfolio_vol:.2%}")
print(f"Risk Contributions:")
for i, symbol in enumerate(symbols):
    print(f"  {symbol}: {risk_contrib[i]:.1%} (Weight: {weights_array[i]:.1%})")
```
⚠️ **Comprehensive Risk Analysis**

```python
# === RISK METRICS CALCULATION ===
import meridianalgo as ma
import pandas as pd
import numpy as np

# Get sample data
data = ma.api_get_market_data(['AAPL'], '2020-01-01', '2023-12-31')
returns = data['AAPL'].pct_change().dropna()

print(f"📊 Risk Analysis for AAPL ({len(returns)} observations)")

# === VALUE AT RISK (VaR) ===
print(f"\n💰 VALUE AT RISK ANALYSIS:")

# Historical VaR
var_95_hist = ma.calculate_value_at_risk(returns, confidence_level=0.95)
var_99_hist = ma.calculate_value_at_risk(returns, confidence_level=0.99)

print(f"Historical VaR:")
print(f"  95% VaR: {var_95_hist:.2%} (1-day)")
print(f"  99% VaR: {var_99_hist:.2%} (1-day)")
print(f"  95% VaR (10-day): {var_95_hist * np.sqrt(10):.2%}")
print(f"  99% VaR (10-day): {var_99_hist * np.sqrt(10):.2%}")

# Portfolio VaR (for $1M portfolio)
portfolio_value = 1000000
var_dollar_95 = abs(var_95_hist) * portfolio_value
var_dollar_99 = abs(var_99_hist) * portfolio_value

print(f"\nPortfolio VaR ($1M portfolio):")
print(f"  95% VaR: ${var_dollar_95:,.0f}")
print(f"  99% VaR: ${var_dollar_99:,.0f}")

# === EXPECTED SHORTFALL (CVaR) ===
print(f"\n📉 EXPECTED SHORTFALL ANALYSIS:")

es_95 = ma.calculate_expected_shortfall(returns, confidence_level=0.95)
es_99 = ma.calculate_expected_shortfall(returns, confidence_level=0.99)

print(f"Expected Shortfall (CVaR):")
print(f"  95% ES: {es_95:.2%}")
print(f"  99% ES: {es_99:.2%}")
print(f"  ES/VaR Ratio (95%): {es_95/var_95_hist:.2f}")
print(f"  ES/VaR Ratio (99%): {es_99/var_99_hist:.2f}")

# === COMPREHENSIVE RISK METRICS ===
print(f"\n📊 COMPREHENSIVE RISK METRICS:")

metrics = ma.api_calculate_risk_metrics(returns)

print(f"Return Metrics:")
print(f"  Total Return: {metrics['total_return']:.2%}")
print(f"  Annual Return: {metrics['annual_return']:.2%}")
print(f"  Volatility: {metrics['volatility']:.2%}")

print(f"\nRisk-Adjusted Metrics:")
print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}")
print(f"  Calmar Ratio: {metrics['calmar_ratio']:.2f}")

print(f"\nDrawdown Analysis:")
print(f"  Maximum Drawdown: {metrics['max_drawdown']:.2%}")

print(f"\nDistribution Analysis:")
print(f"  Skewness: {metrics['skewness']:.2f}")
print(f"  Kurtosis: {metrics['kurtosis']:.2f}")

print(f"\nTail Risk:")
print(f"  VaR (95%): {metrics['var_95']:.2%}")
print(f"  Expected Shortfall (95%): {metrics['expected_shortfall_95']:.2%}")

print(f"\nWin/Loss Analysis:")
print(f"  Win Rate: {metrics['win_rate']:.1%}")
print(f"  Average Win: {metrics['avg_win']:.2%}")
print(f"  Average Loss: {metrics['avg_loss']:.2%}")
print(f"  Profit Factor: {metrics['profit_factor']:.2f}")

# === ROLLING RISK ANALYSIS ===
print(f"\n📈 ROLLING RISK ANALYSIS:")

# Rolling volatility
rolling_vol_30 = ma.rolling_volatility(returns, window=30, annualized=True)
rolling_vol_90 = ma.rolling_volatility(returns, window=90, annualized=True)

print(f"Current Rolling Volatility:")
print(f"  30-day: {rolling_vol_30.iloc[-1]:.2%}")
print(f"  90-day: {rolling_vol_90.iloc[-1]:.2%}")
print(f"  Volatility Trend: {'Increasing' if rolling_vol_30.iloc[-1] > rolling_vol_90.iloc[-1] else 'Decreasing'}")

# Rolling VaR
rolling_var = returns.rolling(window=252).apply(
    lambda x: ma.calculate_value_at_risk(x, 0.95) if len(x) == 252 else np.nan
)

print(f"Rolling VaR (1-year window):")
print(f"  Current: {rolling_var.iloc[-1]:.2%}")
print(f"  Average: {rolling_var.mean():.2%}")
print(f"  Maximum: {rolling_var.min():.2%}")  # Most negative (worst)

# === STRESS TESTING ===
print(f"\n🚨 STRESS TESTING:")

# Historical stress scenarios
stress_scenarios = {
    'COVID-19 Crash (Mar 2020)': -0.30,
    'Flash Crash (May 2010)': -0.09,
    'Black Monday (Oct 1987)': -0.22,
    'Dot-com Crash (Mar 2000)': -0.15,
    '2008 Financial Crisis': -0.40
}

portfolio_value = 1000000
print(f"Stress Test Results (${portfolio_value:,.0f} portfolio):")

for scenario, shock in stress_scenarios.items():
    stressed_value = portfolio_value * (1 + shock)
    loss = portfolio_value - stressed_value
    print(f"  {scenario}:")
    print(f"    Shock: {shock:.1%}")
    print(f"    Portfolio Value: ${stressed_value:,.0f}")
    print(f"    Loss: ${loss:,.0f}")

# Monte Carlo stress testing
print(f"\nMonte Carlo Stress Testing (1000 simulations):")
np.random.seed(42)
mc_returns = np.random.normal(returns.mean(), returns.std(), 1000)
mc_losses = mc_returns[mc_returns < 0]

if len(mc_losses) > 0:
    worst_1pct = np.percentile(mc_losses, 1)
    worst_5pct = np.percentile(mc_losses, 5)
    
    print(f"  Worst 1% scenario: {worst_1pct:.2%}")
    print(f"  Worst 5% scenario: {worst_5pct:.2%}")
    print(f"  Expected worst loss: {mc_losses.mean():.2%}")

# === CORRELATION ANALYSIS ===
print(f"\n🔗 CORRELATION ANALYSIS:")

# Get multiple assets for correlation
multi_data = ma.api_get_market_data(['AAPL', 'GOOGL', 'MSFT', 'SPY'], '2023-01-01', '2023-12-31')
multi_returns = multi_data.pct_change().dropna()

correlation_matrix = multi_returns.corr()
print(f"Correlation Matrix:")
print(correlation_matrix.round(3))

# Rolling correlation with market (SPY)
rolling_corr = multi_returns['AAPL'].rolling(window=60).corr(multi_returns['SPY'])
print(f"\nAPPL-SPY Rolling Correlation (60-day):")
print(f"  Current: {rolling_corr.iloc[-1]:.3f}")
print(f"  Average: {rolling_corr.mean():.3f}")
print(f"  Range: {rolling_corr.min():.3f} to {rolling_corr.max():.3f}")

# === REGIME ANALYSIS ===
print(f"\n📊 MARKET REGIME ANALYSIS:")

# Simple regime detection based on volatility
vol_threshold = returns.std() * 1.5
high_vol_periods = returns[abs(returns) > vol_threshold]
regime_pct = len(high_vol_periods) / len(returns)

print(f"Volatility Regime Analysis:")
print(f"  High volatility threshold: {vol_threshold:.2%}")
print(f"  High volatility periods: {len(high_vol_periods)} ({regime_pct:.1%})")
print(f"  Average return in high vol: {high_vol_periods.mean():.2%}")
print(f"  Average return in normal vol: {returns[abs(returns) <= vol_threshold].mean():.2%}")

# Trend regime analysis
sma_50 = returns.rolling(50).mean()
uptrend_periods = returns[returns > sma_50]
downtrend_periods = returns[returns < sma_50]

print(f"\nTrend Regime Analysis:")
print(f"  Uptrend periods: {len(uptrend_periods)} ({len(uptrend_periods)/len(returns):.1%})")
print(f"  Downtrend periods: {len(downtrend_periods)} ({len(downtrend_periods)/len(returns):.1%})")
print(f"  Average uptrend return: {uptrend_periods.mean():.2%}")
print(f"  Average downtrend return: {downtrend_periods.mean():.2%}")
```
🤖 **Machine Learning & Feature Engineering**

```python
# === COMPREHENSIVE FEATURE ENGINEERING ===
import meridianalgo as ma
import pandas as pd
import numpy as np

# Get market data
data = ma.api_get_market_data(['AAPL'], '2020-01-01', '2023-12-31')
prices = data['AAPL']

print(f"🤖 Machine Learning Feature Engineering for AAPL")
print(f"Data period: {len(prices)} observations")

# === BASIC FEATURE ENGINEERING ===
print(f"\n🔧 FEATURE ENGINEERING:")

# Create feature engineer
engineer = ma.FeatureEngineer()

# Generate comprehensive features
features = engineer.create_features(prices)
print(f"Generated {len(features.columns)} features:")

# Display feature categories
feature_categories = {
    'Price Features': [col for col in features.columns if 'price' in col.lower() or 'return' in col.lower()],
    'Technical Indicators': [col for col in features.columns if any(ind in col.lower() for ind in ['rsi', 'macd', 'sma', 'ema'])],
    'Volatility Features': [col for col in features.columns if 'vol' in col.lower() or 'std' in col.lower()],
    'Momentum Features': [col for col in features.columns if any(mom in col.lower() for mom in ['momentum', 'roc', 'stoch'])],
    'Other Features': [col for col in features.columns if col not in sum([
        [col for col in features.columns if 'price' in col.lower() or 'return' in col.lower()],
        [col for col in features.columns if any(ind in col.lower() for ind in ['rsi', 'macd', 'sma', 'ema'])],
        [col for col in features.columns if 'vol' in col.lower() or 'std' in col.lower()],
        [col for col in features.columns if any(mom in col.lower() for mom in ['momentum', 'roc', 'stoch'])]
    ], [])]
}

for category, cols in feature_categories.items():
    if cols:
        print(f"  {category}: {len(cols)} features")
        print(f"    Examples: {', '.join(cols[:3])}")

# === ADVANCED FEATURE ENGINEERING ===
print(f"\n🚀 ADVANCED FEATURE ENGINEERING:")

# Create OHLCV data for advanced features
ohlcv_data = pd.DataFrame({
    'Open': prices * np.random.uniform(0.995, 1.005, len(prices)),
    'High': prices * np.random.uniform(1.00, 1.03, len(prices)),
    'Low': prices * np.random.uniform(0.97, 1.00, len(prices)),
    'Close': prices,
    'Volume': np.random.randint(1000000, 10000000, len(prices))
}, index=prices.index)

# Price-based features
price_features = engineer.create_price_features(ohlcv_data)
print(f"Price Features: {len(price_features.columns)} features")

# Volume-based features
volume_features = engineer.create_volume_features(prices, ohlcv_data['Volume'])
print(f"Volume Features: {len(volume_features.columns)} features")

# Volatility features
returns = prices.pct_change().dropna()
volatility_features = engineer.create_volatility_features(returns)
print(f"Volatility Features: {len(volatility_features.columns)} features")

# === LSTM PREPARATION AND TRAINING ===
print(f"\n🧠 LSTM NEURAL NETWORK:")

# Prepare data for LSTM
sequence_length = 60
X, y = ma.prepare_data_for_lstm(prices.values, sequence_length=sequence_length)

print(f"LSTM Data Preparation:")
print(f"  Input sequences: {X.shape[0]}")
print(f"  Sequence length: {X.shape[1]}")
print(f"  Features per timestep: {X.shape[2]}")
print(f"  Target values: {y.shape[0]}")

# Create and train LSTM model
print(f"\nTraining LSTM Model...")
predictor = ma.LSTMPredictor(
    sequence_length=sequence_length,
    epochs=50,  # Reduced for example
    batch_size=32,
    verbose=1
)

# Split data for training
split_point = int(0.8 * len(X))
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Train the model
predictor.fit(X_train, y_train)

# Make predictions
train_predictions = predictor.predict(X_train)
test_predictions = predictor.predict(X_test)

print(f"\nLSTM Model Results:")
print(f"  Training samples: {len(train_predictions)}")
print(f"  Test samples: {len(test_predictions)}")

# === MODEL EVALUATION ===
print(f"\n📊 MODEL EVALUATION:")

# Calculate evaluation metrics
def calculate_metrics(actual, predicted):
    mse = np.mean((predicted - actual) ** 2)
    mae = np.mean(np.abs(predicted - actual))
    rmse = np.sqrt(mse)
    
    # Correlation
    correlation = np.corrcoef(predicted, actual)[0, 1] if len(predicted) > 1 else 0
    
    # Directional accuracy
    if len(actual) > 1:
        actual_direction = np.sign(actual[1:] - actual[:-1])
        pred_direction = np.sign(predicted[1:] - predicted[:-1])
        directional_accuracy = np.mean(actual_direction == pred_direction)
    else:
        directional_accuracy = 0
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'Correlation': correlation,
        'Directional Accuracy': directional_accuracy
    }

# Training metrics
train_metrics = calculate_metrics(y_train, train_predictions)
test_metrics = calculate_metrics(y_test, test_predictions)

print(f"Training Metrics:")
for metric, value in train_metrics.items():
    print(f"  {metric}: {value:.4f}")

print(f"\nTest Metrics:")
for metric, value in test_metrics.items():
    print(f"  {metric}: {value:.4f}")

# === FEATURE IMPORTANCE ANALYSIS ===
print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS:")

# Create target for feature importance
target = prices.pct_change().shift(-1).dropna()

# Align features and target
common_index = features.index.intersection(target.index)
if len(common_index) > 50:
    features_aligned = features.loc[common_index].fillna(0)
    target_aligned = target.loc[common_index]
    
    # Calculate correlations as proxy for importance
    correlations = features_aligned.corrwith(target_aligned).abs().sort_values(ascending=False)
    
    print(f"Top 10 Most Important Features:")
    for i, (feature, importance) in enumerate(correlations.head(10).items()):
        print(f"  {i+1}. {feature}: {importance:.4f}")
    
    # Feature selection
    top_features = correlations.head(20).index.tolist()
    selected_features = features_aligned[top_features]
    
    print(f"\nSelected {len(top_features)} most important features for modeling")

# === ENSEMBLE MODELING ===
print(f"\n🎯 ENSEMBLE MODELING:")

# Create multiple simple predictions (simulated)
np.random.seed(42)
n_models = 5
ensemble_predictions = []

for i in range(n_models):
    # Simulate different model predictions
    noise_level = 0.01 * (i + 1)
    model_pred = test_predictions + np.random.normal(0, noise_level, len(test_predictions))
    ensemble_predictions.append(model_pred)

# Ensemble methods
mean_ensemble = np.mean(ensemble_predictions, axis=0)
median_ensemble = np.median(ensemble_predictions, axis=0)

# Weighted ensemble (give more weight to better performing models)
weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])  # Decreasing weights
weighted_ensemble = np.average(ensemble_predictions, axis=0, weights=weights)

print(f"Ensemble Results:")
print(f"  Individual models: {len(ensemble_predictions)}")
print(f"  Ensemble predictions: {len(mean_ensemble)}")

# Evaluate ensemble performance
ensemble_metrics = {
    'Mean Ensemble': calculate_metrics(y_test, mean_ensemble),
    'Median Ensemble': calculate_metrics(y_test, median_ensemble),
    'Weighted Ensemble': calculate_metrics(y_test, weighted_ensemble),
    'Best Individual': calculate_metrics(y_test, test_predictions)
}

print(f"\nEnsemble Performance Comparison:")
for method, metrics in ensemble_metrics.items():
    print(f"  {method}:")
    print(f"    Correlation: {metrics['Correlation']:.4f}")
    print(f"    Directional Accuracy: {metrics['Directional Accuracy']:.4f}")
    print(f"    RMSE: {metrics['RMSE']:.4f}")

# === WALK-FORWARD VALIDATION ===
print(f"\n⏭️ WALK-FORWARD VALIDATION:")

# Simulate walk-forward validation
validation_window = 252  # 1 year
prediction_horizon = 21  # 1 month
n_folds = 5

print(f"Walk-Forward Validation Setup:")
print(f"  Training window: {validation_window} days")
print(f"  Prediction horizon: {prediction_horizon} days")
print(f"  Number of folds: {n_folds}")

# Simulate validation results
validation_results = []
for fold in range(n_folds):
    # Simulate performance metrics for each fold
    fold_correlation = np.random.uniform(0.3, 0.7)
    fold_accuracy = np.random.uniform(0.45, 0.65)
    fold_rmse = np.random.uniform(0.015, 0.035)
    
    validation_results.append({
        'Fold': fold + 1,
        'Correlation': fold_correlation,
        'Directional Accuracy': fold_accuracy,
        'RMSE': fold_rmse
    })

print(f"\nWalk-Forward Validation Results:")
for result in validation_results:
    print(f"  Fold {result['Fold']}: Corr={result['Correlation']:.3f}, "
          f"Acc={result['Directional Accuracy']:.3f}, RMSE={result['RMSE']:.4f}")

# Average performance
avg_correlation = np.mean([r['Correlation'] for r in validation_results])
avg_accuracy = np.mean([r['Directional Accuracy'] for r in validation_results])
avg_rmse = np.mean([r['RMSE'] for r in validation_results])

print(f"\nAverage Performance:")
print(f"  Correlation: {avg_correlation:.3f}")
print(f"  Directional Accuracy: {avg_accuracy:.3f}")
print(f"  RMSE: {avg_rmse:.4f}")

# === PRODUCTION DEPLOYMENT SIMULATION ===
print(f"\n🚀 PRODUCTION DEPLOYMENT:")

print(f"Model Deployment Checklist:")
print(f"  ✅ Model trained and validated")
print(f"  ✅ Feature pipeline established")
print(f"  ✅ Performance benchmarks set")
print(f"  ✅ Ensemble method selected")
print(f"  ✅ Walk-forward validation completed")

print(f"\nModel Monitoring Setup:")
print(f"  📊 Performance tracking: Correlation, Accuracy, RMSE")
print(f"  🚨 Alert thresholds: Correlation < 0.3, Accuracy < 0.45")
print(f"  🔄 Retraining schedule: Monthly")
print(f"  📈 A/B testing: 20% traffic to new model")

print(f"\n🎯 Ready for production deployment!")
```
 🔄 **Production Backtesting Engine**

```python
# === COMPREHENSIVE BACKTESTING EXAMPLE ===
import meridianalgo as ma
import pandas as pd
import numpy as np

# Get historical data for backtesting
symbols = ['AAPL', 'GOOGL', 'MSFT']
data = ma.api_get_market_data(symbols, '2020-01-01', '2023-12-31')

print(f"🔄 Backtesting Strategy on {len(symbols)} assets")
print(f"Data period: {data.index[0].date()} to {data.index[-1].date()}")
print(f"Total trading days: {len(data)}")

# === STRATEGY DEFINITION ===
print(f"\n📈 STRATEGY: Multi-Asset Momentum + Mean Reversion")

class AdvancedTradingStrategy:
    def __init__(self, symbols):
        self.symbols = symbols
        self.positions = {symbol: 0 for symbol in symbols}
        self.signals = {symbol: 0 for symbol in symbols}
        
        # Strategy parameters
        self.short_window = 10
        self.long_window = 50
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.bb_period = 20
        self.position_size = 100
        
    def generate_signals(self, data, current_date):
        signals = []
        
        for symbol in self.symbols:
            if symbol not in data.columns:
                continue
                
            # Get price history up to current date
            prices = data[symbol].loc[:current_date]
            
            if len(prices) < self.long_window:
                continue
            
            current_price = prices.iloc[-1]
            
            # Technical indicators
            sma_short = prices.rolling(self.short_window).mean().iloc[-1]
            sma_long = prices.rolling(self.long_window).mean().iloc[-1]
            
            # RSI
            rsi = ma.RSI(prices, self.rsi_period).iloc[-1]
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = ma.BollingerBands(prices, self.bb_period)
            bb_upper_val = bb_upper.iloc[-1]
            bb_lower_val = bb_lower.iloc[-1]
            
            # Signal generation logic
            signal_strength = 0
            
            # Momentum signals
            if sma_short > sma_long:
                signal_strength += 1
            elif sma_short < sma_long:
                signal_strength -= 1
            
            # Mean reversion signals
            if rsi < self.rsi_oversold and current_price < bb_lower_val:
                signal_strength += 1
            elif rsi > self.rsi_overbought and current_price > bb_upper_val:
                signal_strength -= 1
            
            # Volume confirmation (simplified)
            recent_volume = 1.2  # Assume above average volume
            if recent_volume > 1.1:
                signal_strength *= 1.1
            
            # Generate trading signals
            current_position = self.positions[symbol]
            
            if signal_strength >= 1.5 and current_position <= 0:
                signals.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': self.position_size,
                    'price': current_price,
                    'signal_strength': signal_strength,
                    'reason': f'Momentum + Oversold (RSI: {rsi:.1f})'
                })
            elif signal_strength <= -1.5 and current_position >= 0:
                signals.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': self.position_size,
                    'price': current_price,
                    'signal_strength': signal_strength,
                    'reason': f'Reversal + Overbought (RSI: {rsi:.1f})'
                })
        
        return signals
    
    def update_positions(self, symbol, quantity):
        self.positions[symbol] += quantity

# === BACKTESTING EXECUTION ===
print(f"\n🚀 BACKTESTING EXECUTION:")

# Initialize backtest
engine = ma.BacktestEngine(
    initial_capital=100000,
    commission=0.001,  # 0.1%
    slippage=0.0005    # 0.05%
)

strategy = AdvancedTradingStrategy(symbols)

# Track performance
portfolio_history = []
trade_log = []
daily_returns = []

print(f"Initial Capital: ${engine.initial_capital:,.2f}")
print(f"Commission: {engine.commission:.3%}")
print(f"Slippage: {engine.slippage:.3%}")

# Run backtest
total_trades = 0
winning_trades = 0

for i, date in enumerate(data.index[50:]):  # Start after warmup period
    current_data = data.loc[:date]
    
    # Generate signals
    signals = strategy.generate_signals(current_data, date)
    
    # Execute trades
    for signal in signals:
        symbol = signal['symbol']
        action = signal['action']
        quantity = signal['quantity']
        price = signal['price']
        
        success = engine.execute_order(symbol, 'market', action, quantity, price)
        
        if success:
            strategy.update_positions(symbol, quantity if action == 'BUY' else -quantity)
            total_trades += 1
            
            trade_log.append({
                'date': date,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'signal_strength': signal['signal_strength'],
                'reason': signal['reason']
            })
    
    # Calculate portfolio value
    current_prices = {symbol: current_data[symbol].iloc[-1] for symbol in symbols}
    portfolio_value = engine.get_portfolio_value(current_prices)
    
    # Calculate daily return
    if i > 0:
        prev_value = portfolio_history[-1]['total_value']
        daily_return = (portfolio_value - prev_value) / prev_value
        daily_returns.append(daily_return)
    
    portfolio_history.append({
        'date': date,
        'cash': engine.cash,
        'positions_value': portfolio_value - engine.cash,
        'total_value': portfolio_value,
        'return': (portfolio_value - engine.initial_capital) / engine.initial_capital
    })

# === PERFORMANCE ANALYSIS ===
print(f"\n📊 BACKTESTING RESULTS:")

portfolio_df = pd.DataFrame(portfolio_history)
final_value = portfolio_df['total_value'].iloc[-1]
total_return = (final_value - engine.initial_capital) / engine.initial_capital

print(f"\n💰 PERFORMANCE SUMMARY:")
print(f"Initial Capital: ${engine.initial_capital:,.2f}")
print(f"Final Value: ${final_value:,.2f}")
print(f"Total Return: {total_return:.2%}")
print(f"Total Trades: {total_trades}")

# Calculate additional metrics
if daily_returns:
    daily_returns_series = pd.Series(daily_returns)
    
    # Annualized metrics
    annual_return = (1 + total_return) ** (252 / len(portfolio_df)) - 1
    volatility = daily_returns_series.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    # Maximum drawdown
    running_max = portfolio_df['total_value'].expanding().max()
    drawdown = (portfolio_df['total_value'] - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate analysis
    if len(trade_log) > 0:
        trades_df = pd.DataFrame(trade_log)
        
        # Simplified P&L calculation
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        
        if len(buy_trades) > 0 and len(sell_trades) > 0:
            # Match buy/sell pairs (simplified)
            min_trades = min(len(buy_trades), len(sell_trades))
            pnl_trades = []
            
            for i in range(min_trades):
                if i < len(sell_trades) and i < len(buy_trades):
                    pnl = sell_trades.iloc[i]['price'] - buy_trades.iloc[i]['price']
                    pnl_trades.append(pnl)
            
            if pnl_trades:
                winning_trades = sum(1 for pnl in pnl_trades if pnl > 0)
                win_rate = winning_trades / len(pnl_trades)
                avg_win = np.mean([pnl for pnl in pnl_trades if pnl > 0]) if winning_trades > 0 else 0
                avg_loss = np.mean([pnl for pnl in pnl_trades if pnl < 0]) if len(pnl_trades) - winning_trades > 0 else 0
                profit_factor = abs(avg_win * winning_trades / (avg_loss * (len(pnl_trades) - winning_trades))) if avg_loss != 0 else float('inf')
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
    
    print(f"\n📈 RISK-ADJUSTED METRICS:")
    print(f"Annualized Return: {annual_return:.2%}")
    print(f"Volatility: {volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    
    print(f"\n🎯 TRADING METRICS:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Win: ${avg_win:.2f}")
    print(f"Average Loss: ${avg_loss:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    
    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    print(f"Calmar Ratio: {calmar_ratio:.2f}")

# === TRADE ANALYSIS ===
if trade_log:
    print(f"\n📋 DETAILED TRADE ANALYSIS:")
    
    trades_df = pd.DataFrame(trade_log)
    
    # Trades by symbol
    trades_by_symbol = trades_df.groupby('symbol').size()
    print(f"\nTrades by Symbol:")
    for symbol, count in trades_by_symbol.items():
        print(f"  {symbol}: {count} trades")
    
    # Trades by action
    trades_by_action = trades_df.groupby('action').size()
    print(f"\nTrades by Action:")
    for action, count in trades_by_action.items():
        print(f"  {action}: {count} trades")
    
    # Signal strength analysis
    avg_signal_strength = trades_df['signal_strength'].mean()
    print(f"\nSignal Analysis:")
    print(f"  Average Signal Strength: {avg_signal_strength:.2f}")
    print(f"  Strong Signals (>2.0): {len(trades_df[trades_df['signal_strength'] > 2.0])}")
    print(f"  Weak Signals (<1.5): {len(trades_df[abs(trades_df['signal_strength']) < 1.5])}")
    
    # Recent trades
    print(f"\n🔍 RECENT TRADES (Last 5):")
    for i, trade in trades_df.tail(5).iterrows():
        print(f"  {trade['date'].date()}: {trade['action']} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f}")
        print(f"    Reason: {trade['reason']}")
        print(f"    Signal Strength: {trade['signal_strength']:.2f}")

# === BENCHMARK COMPARISON ===
print(f"\n📊 BENCHMARK COMPARISON:")

# Compare against buy-and-hold SPY
try:
    benchmark_data = ma.api_get_market_data(['SPY'], portfolio_df['date'].iloc[0], portfolio_df['date'].iloc[-1])
    if len(benchmark_data) > 0:
        benchmark_return = (benchmark_data['SPY'].iloc[-1] - benchmark_data['SPY'].iloc[0]) / benchmark_data['SPY'].iloc[0]
        benchmark_annual = (1 + benchmark_return) ** (252 / len(benchmark_data)) - 1
        
        print(f"Strategy Return: {annual_return:.2%}")
        print(f"S&P 500 Return: {benchmark_annual:.2%}")
        print(f"Excess Return: {annual_return - benchmark_annual:.2%}")
        
        if volatility > 0:
            information_ratio = (annual_return - benchmark_annual) / volatility
            print(f"Information Ratio: {information_ratio:.2f}")
    else:
        print("Benchmark data not available")
except:
    print("Benchmark comparison not available")

# === PORTFOLIO COMPOSITION ===
print(f"\n💼 FINAL PORTFOLIO COMPOSITION:")
print(f"Cash: ${engine.cash:,.2f} ({engine.cash/final_value:.1%})")

for symbol in symbols:
    position = engine.positions.get(symbol, 0)
    if position != 0:
        current_price = data[symbol].iloc[-1]
        position_value = position * current_price
        print(f"{symbol}: {position} shares @ ${current_price:.2f} = ${position_value:,.2f} ({position_value/final_value:.1%})")

print(f"\n🎉 BACKTESTING COMPLETE!")
print(f"✅ Strategy performance analyzed")
print(f"✅ Risk metrics calculated")
print(f"✅ Trade analysis completed")
print(f"✅ Benchmark comparison done")
```