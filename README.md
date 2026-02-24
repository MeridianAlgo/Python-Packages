# MeridianAlgo Quant Packages

## The Complete Quantitative Finance Platform

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI Version](https://img.shields.io/badge/pypi-6.2.1-orange.svg)](https://pypi.org/project/meridianalgo/)
[![Tests](https://img.shields.io/badge/tests-300%2B%20passing-brightgreen.svg)](tests/)

MeridianAlgo is a comprehensive, institutional-grade Python platform for quantitative finance. It provides a complete suite of tools for algorithmic trading, portfolio optimization, risk management, derivatives pricing, and market microstructure analysis. Built for professional quants, researchers, and trading firms.

**Key Highlights:**
- 50+ performance metrics and analytics
- Event-driven backtesting engine with realistic execution
- Optimal execution algorithms (VWAP, TWAP, POV, Implementation Shortfall)
- Market microstructure analysis (order book, VPIN, liquidity metrics)
- Statistical arbitrage and pairs trading
- Factor models (Fama-French, APT, custom)
- Options pricing and Greeks
- Machine learning integration
- GPU acceleration support
- Distributed computing ready

---

## Installation

### Standard Installation

```bash
pip install meridianalgo
```

### With Optional Dependencies

```bash
# Machine learning support (scikit-learn, PyTorch, statsmodels)
pip install meridianalgo[ml]

# Optimization (CVXPY, CVXOPT)
pip install meridianalgo[optimization]

# Volatility modeling (ARCH)
pip install meridianalgo[volatility]

# Alternative data (web scraping, API clients)
pip install meridianalgo[data]

# Distributed computing (Ray, Dask)
pip install meridianalgo[distributed]

# Everything
pip install meridianalgo[all]
```

---

## Quick Start Examples

### 1. Portfolio Analytics

Calculate comprehensive performance metrics including Sharpe Ratio, Drawdowns, and Value at Risk.

```python
import meridianalgo as ma
import pandas as pd
from meridianalgo.analytics import PerformanceAnalyzer

# Load your returns data
# Format: DateTime Index, Columns are asset returns
returns = pd.read_csv('returns.csv', index_col=0, parse_dates=True)

# Initialize analyzer
analyzer = PerformanceAnalyzer(returns)

# Get summary statistics
metrics = analyzer.summary()

print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.1%}")
print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
print(f"Value at Risk (95%): {metrics['var_95']:.2%}")
```

### 2. Backtesting a Strategy

Run an event-driven backtest with realistic slippage and commission models.

```python
from meridianalgo.backtesting import BacktestEngine
import yfinance as yf

# 1. Get Market Data
data = yf.download('AAPL', start='2020-01-01', end='2023-12-31')

# 2. Initialize Engine
engine = BacktestEngine(
    initial_capital=100000,
    commission=0.001,  # 0.1% commission
    slippage=0.0005    # 0.05% slippage
)

# 3. Simulate Strategy (Simple Moving Average Crossover)
short_window = 20
long_window = 50

# Calculate indicators
data['SMA_Short'] = data['Close'].rolling(window=short_window).mean()
data['SMA_Long'] = data['Close'].rolling(window=long_window).mean()

# Run Simulation
for i in range(long_window, len(data)):
    price = data['Close'].iloc[i]
    date = data.index[i]
    
    # Update engine time
    engine.update_time(date)
    
    # Check signals
    if data['SMA_Short'].iloc[i] > data['SMA_Long'].iloc[i]:
        # Buy Signal
        quantity = int(engine.cash / price)
        if quantity > 0:
            engine.execute_order('AAPL', 'market', 'buy', quantity, price)
            
    elif data['SMA_Short'].iloc[i] < data['SMA_Long'].iloc[i]:
        # Sell Signal
        if 'AAPL' in engine.positions:
            quantity = engine.positions['AAPL']
            engine.execute_order('AAPL', 'market', 'sell', quantity, price)
            
    # Record daily snapshot
    engine.record_snapshot({'AAPL': price})

# 4. Analyze Results
metrics = engine.get_performance_metrics()
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
```

### 3. Execution Algorithms (VWAP)

Simulate an optimal execution schedule using Volume Weighted Average Price.

```python
from meridianalgo.quant.execution_algorithms import VWAP

# Initialize VWAP Algo
vwap = VWAP(
    total_quantity=100000,
    start_time='09:30',
    end_time='16:00'
)

# Calculate optimal schedule based on historical volume profile
historical_volume_profile = [...] # List or array of volume fractions
schedule = vwap.calculate_schedule(historical_volume_profile)

print("Execution Schedule:")
for slice_info in schedule:
    print(f"Time: {slice_info['time']} | Quantity: {slice_info['quantity']}")
```

### 4. Market Microstructure Analysis

Analyze order book depth and liquidity metrics.

```python
from meridianalgo.liquidity import OrderBook, VPIN

# Order Book Analysis
ob = OrderBook()
ob.add_bid(price=100.0, quantity=1000)
ob.add_ask(price=100.1, quantity=1000)

print(f"Spread: {ob.spread():.4f}")
print(f"Mid Price: {ob.mid_price():.2f}")
print(f"Market Depth (5 levels): {ob.depth(levels=5)}")

# VPIN (Volume-Synchronized Probability of Informed Trading)
# Used to detect toxic order flow
vpin = VPIN(trades_data)
current_vpin = vpin.current_vpin()
print(f"Current VPIN: {current_vpin:.3f}")

if vpin.toxicity_regime() == 'HIGH':
    print("Warning: High Toxic Order Flow Detected")
```

### 5. Statistical Arbitrage (Pairs Trading)

Identify and trade cointegrated pairs.

```python
from meridianalgo.quant.statistical_arbitrage import PairsTrading, CointegrationAnalyzer

# Test for Cointegration
analyzer = CointegrationAnalyzer(asset1_prices, asset2_prices)
result = analyzer.test_cointegration()

if result['is_cointegrated']:
    print(f"Pairs are cointegrated (p-value: {result['p_value']:.4f})")
    
    # Generate Signals
    pairs = PairsTrading(asset1_prices, asset2_prices)
    signals = pairs.generate_signals(
        entry_z_score=2.0,
        exit_z_score=0.5
    )
    
    print(f"Latest Signal: {signals.iloc[-1]}")
```

### 6. Factor Risk Models

Decompose portfolio risk into factor contributions.

```python
from meridianalgo.quant.factor_models import FamaFrenchModel, FactorRiskDecomposition

# Fama-French 3-Factor Model
ff = FamaFrenchModel(portfolio_returns, market_excess, smb, hml)
alpha, beta_mkt, beta_smb, beta_hml = ff.fit()

print(f"Alpha: {alpha:.4f}")
print(f"Market Beta: {beta_mkt:.2f}")

# Risk Decomposition
decomp = FactorRiskDecomposition(portfolio_returns, factor_returns_df)
risk_contrib = decomp.factor_contribution_to_risk()

print("Risk Contribution by Factor:")
print(risk_contrib)
```

---

## Core Modules

### Analytics (`meridianalgo.analytics`)
- **PerformanceAnalyzer**: 50+ metrics (Sharpe, Sortino, Calmar, Information Ratio, etc.)
- **RiskAnalyzer**: VaR, CVaR, stress testing, scenario analysis
- **DrawdownAnalyzer**: Drawdown analysis, underwater plots, recovery metrics
- **TearSheet**: Comprehensive performance reports

### Backtesting (`meridianalgo.backtesting`)
- **Event-driven engine**: Realistic market simulation with bid-ask spreads
- **Order management**: Market, limit, stop, bracket orders
- **Execution simulation**: Market impact, slippage, commission modeling
- **Pre-built strategies**: SMA crossover, momentum, mean reversion

### Liquidity (`meridianalgo.liquidity`)
- **OrderBook**: Depth analysis, microprice, spread metrics
- **VPIN**: Volume-Synchronized Probability of Informed Trading
- **MarketImpact**: Linear, square-root, power-law impact models
- **Microstructure**: Tick data analysis, volume profiles

### Quant (`meridianalgo.quant`)
- **Execution**: VWAP, TWAP, POV, Implementation Shortfall (Almgren-Chriss)
- **Statistical Arbitrage**: Pairs trading, cointegration, mean reversion
- **Factor Models**: Fama-French, APT, custom factor models
- **High-Frequency**: Market making, latency arbitrage, order book dynamics
- **Regime Detection**: Hidden Markov Models, structural breaks, volatility regimes

### Signals (`meridianalgo.signals`)
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic, ADX, OBV
- **Signal Generation**: Multi-indicator signal generation and evaluation
- **Pattern Recognition**: Chart patterns, support/resistance levels

### Portfolio (`meridianalgo.portfolio`)
- **Optimization**: Mean-variance, risk parity, Black-Litterman
- **Rebalancing**: Calendar, threshold, and drift-based rebalancing
- **Performance Attribution**: Brinson-Fachler attribution analysis
- **Risk Management**: Position sizing, concentration limits, Greeks hedging

### Derivatives (`meridianalgo.derivatives`)
- **Options Pricing**: Black-Scholes, binomial, Monte Carlo
- **Greeks**: Delta, gamma, vega, theta, rho calculations
- **Volatility Surfaces**: Smile, skew, term structure modeling
- **Exotic Options**: Barrier, Asian, lookback options

### Data (`meridianalgo.data`)
- **Providers**: Yahoo Finance, Polygon, custom data sources
- **Processing**: OHLCV normalization, corporate actions adjustment
- **Storage**: Efficient time-series storage and retrieval
- **Streaming**: Real-time data feed integration

---

## Performance

MeridianAlgo is optimized for performance:
- **Vectorized operations**: NumPy/Pandas for fast computation
- **GPU acceleration**: CUDA support for matrix operations
- **Distributed computing**: Ray/Dask integration for parallel processing
- **Efficient memory**: Optimized data structures for large datasets

Benchmark results on typical workloads:
- Portfolio analytics: 10,000+ assets in <1 second
- Backtesting: 10 years of daily data in <5 seconds
- Factor model fitting: 1,000+ factors in <10 seconds

---

## Documentation

Full documentation available at: https://meridianalgo.readthedocs.io

- [API Reference](docs/API_REFERENCE.md)
- [User Guide](docs/README.md)
- [Examples](examples/)
- [Benchmarks](docs/benchmarks.md)

---

## Citation

If you use MeridianAlgo in your research, please cite:

```bibtex
@software{meridianalgo2026,
  title = {MeridianAlgo: The Complete Quantitative Finance Platform},
  author = {Meridian Algorithmic Research Team},
  year = {2026},
  version = {6.2.1},
  url = {https://github.com/MeridianAlgo/Python-Packages}
}
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MeridianAlgo is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/MeridianAlgo/Python-Packages/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MeridianAlgo/Python-Packages/discussions)
- **Email**: support@meridianalgo.com

---

## Disclaimer

MeridianAlgo is provided for educational and research purposes. Past performance does not guarantee future results. Always conduct thorough testing and validation before deploying trading strategies in production.
