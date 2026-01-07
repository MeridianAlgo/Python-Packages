# MeridianAlgo v6.0.0

## The Complete Quantitative Finance Platform

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI Version](https://img.shields.io/badge/pypi-6.0.0-orange.svg)](https://pypi.org/project/meridianalgo/)
[![Tests](https://img.shields.io/badge/tests-300%2B%20passing-brightgreen.svg)](tests/)
[![Code Quality](https://img.shields.io/badge/code%20quality-A-brightgreen.svg)]()

**The All-in-One Python Library for Quantitative Finance**

MeridianAlgo is the most comprehensive Python platform for institutional quantitative finance. From trading research to portfolio analytics, from liquidity analysis to options pricing — everything you need in one professional-grade package.

---

## 🎯 Why MeridianAlgo?

| Feature | MeridianAlgo | QuantLib | Zipline | Pyfolio |
|---------|--------------|----------|---------|---------|
| Portfolio Analytics | ✅ | ❌ | ⚠️ | ✅ |
| Options Pricing | ✅ | ✅ | ❌ | ❌ |
| Market Microstructure | ✅ | ❌ | ❌ | ❌ |
| Backtesting | ✅ | ❌ | ✅ | ❌ |
| Execution Algorithms | ✅ | ❌ | ⚠️ | ❌ |
| Risk Management | ✅ | ✅ | ❌ | ⚠️ |
| Factor Models | ✅ | ❌ | ⚠️ | ❌ |
| Machine Learning | ✅ | ❌ | ❌ | ❌ |
| Liquidity Analysis | ✅ | ❌ | ❌ | ❌ |
| Tear Sheets | ✅ | ❌ | ❌ | ✅ |

---

## 🚀 Quick Start

### Installation

```bash
# Standard installation
pip install meridianalgo

# With machine learning support
pip install meridianalgo[ml]

# Full installation (recommended)
pip install meridianalgo[full]

# Everything including distributed computing
pip install meridianalgo[all]
```

### Basic Usage

```python
import meridianalgo as ma

# Quick analysis of any asset
data = ma.get_market_data_quick(['AAPL', 'MSFT', 'GOOGL'], start='2023-01-01')
analysis = ma.quick_analysis(data['AAPL']['Close'])

print(f"Sharpe Ratio: {analysis['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {analysis['max_drawdown']:.1%}")
print(f"Win Rate: {analysis['win_rate']:.1%}")
```

---

## 📦 Core Modules

### 📊 Analytics (Pyfolio-Style)

Generate comprehensive performance tear sheets:

```python
from meridianalgo.analytics import TearSheet, create_full_tear_sheet

# Create full performance tear sheet
ts = TearSheet(returns, benchmark=spy_returns)
ts.create_full_tear_sheet(filename='report.pdf')

# Print summary statistics
ts.print_summary()

# Get all metrics as DataFrame
metrics = ts.get_metrics_summary()
```

**Features:**
- Cumulative returns visualization
- Rolling Sharpe ratio analysis
- Monthly returns heatmap
- Drawdown analysis & underwater chart
- Distribution analysis with VaR
- Benchmark comparison

### 💼 Portfolio Optimization

Multiple optimization methods:

```python
from meridianalgo.portfolio import (
    PortfolioOptimizer, RiskParity, 
    BlackLitterman, EfficientFrontier
)

# Mean-variance optimization
optimizer = PortfolioOptimizer(returns)
weights = optimizer.optimize(method='sharpe')

# Risk parity portfolio
rp = RiskParity(returns)
rp_weights = rp.optimize()

# Black-Litterman with views
bl = BlackLitterman(returns, market_caps)
bl_weights = bl.optimize_with_views({'AAPL': 0.15, 'MSFT': 0.12})

# Efficient frontier
ef = EfficientFrontier(returns)
frontier = ef.calculate_frontier(n_portfolios=100)
```

### 📈 Liquidity Analysis

Comprehensive market microstructure:

```python
from meridianalgo.liquidity import (
    OrderBookAnalyzer, MarketMicrostructure,
    VPIN, MarketImpact, VolumeProfile
)

# Order book analysis
analyzer = OrderBookAnalyzer()
analyzer.update(order_book)

imbalance = analyzer.order_imbalance()
toxicity = analyzer.flow_toxicity()
kyle_lambda = analyzer.kyle_lambda()

# VPIN (Volume-Synchronized PIN)
vpin = VPIN(trades)
current_toxicity = vpin.current_vpin()
regime = vpin.toxicity_regime()

# Market impact estimation
impact = MarketImpact(daily_volume=1e6, volatility=0.02)
cost = impact.estimate_total_cost(order_size=10000, price=150)

# Volume profile analysis
vp = VolumeProfile(trades)
poc = vp.point_of_control()  # Price with highest volume
va_low, va_high = vp.value_area(0.70)  # Value area
```

### 📉 Risk Management

Enterprise-grade risk analytics:

```python
from meridianalgo.risk import (
    VaRCalculator, CVaRCalculator, 
    StressTest, DrawdownAnalyzer
)
from meridianalgo.analytics import RiskAnalyzer

# Risk analyzer
risk = RiskAnalyzer(returns)

# VaR & CVaR (multiple methods)
var_95 = risk.value_at_risk(0.95, method='historical')
var_99 = risk.value_at_risk(0.99, method='cornish_fisher')
cvar = risk.conditional_var(0.95)

# GARCH volatility
garch_vol = risk.garch_volatility()

# Stress testing
stress = risk.stress_test({
    'Market Crash': -0.20,
    'Flash Crash': -0.10,
    'Black Swan': -0.40
})

# Comprehensive summary
risk_summary = risk.summary()
```

### 🎰 Derivatives & Options

Full options pricing suite:

```python
from meridianalgo.derivatives import (
    OptionsPricer, VolatilitySurface,
    BlackScholes, GreeksCalculator
)

# Options pricing
pricer = OptionsPricer()

# Black-Scholes
price = pricer.black_scholes(S=100, K=105, T=0.5, r=0.05, sigma=0.2)

# Binomial tree (American options)
price = pricer.binomial_tree(S=100, K=105, T=0.5, r=0.05, sigma=0.2, 
                             american=True, n_steps=100)

# Monte Carlo
price, std = pricer.monte_carlo_pricing(S=100, K=105, T=0.5, r=0.05, 
                                        sigma=0.2, n_simulations=10000)

# Greeks calculation
greeks = pricer.calculate_greeks(S=100, K=105, T=0.5, r=0.05, sigma=0.2)
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
print(f"Vega: {greeks['vega']:.4f}")
print(f"Theta: {greeks['theta']:.4f}")

# Implied volatility
iv = pricer.calculate_implied_volatility(S=100, K=105, T=0.5, r=0.05, 
                                         market_price=8.50)
```

### ⚡ Execution Algorithms

Institutional-grade execution:

```python
from meridianalgo.execution import (
    VWAP, TWAP, ImplementationShortfall, POV
)

# VWAP execution
vwap = VWAP(total_quantity=10000, start_time='09:30', end_time='16:00')
schedule = vwap.calculate_schedule(historical_volume)

# Implementation Shortfall (Almgren-Chriss)
is_algo = ImplementationShortfall(
    total_quantity=50000,
    total_time=1.0,
    volatility=0.02,
    risk_aversion=1e-6
)
trajectory = is_algo.calculate_optimal_trajectory()
costs = is_algo.calculate_expected_cost()
```

### 📐 Factor Models

Multi-factor analysis:

```python
from meridianalgo.factors import (
    FamaFrench, FactorModel, FactorRiskDecomposition
)

# Fama-French analysis
ff = FamaFrench(model_type='five_factor')
results = ff.fit(returns, factor_data)

print(f"Alpha: {results['alpha']:.4f} (t={results['alpha_t_stat']:.2f})")
print(f"Market Beta: {results['coefficients']['MKT']:.2f}")
print(f"SMB Beta: {results['coefficients']['SMB']:.2f}")

# Factor risk decomposition
decomp = FactorRiskDecomposition.decompose_variance(
    weights, factor_exposures, factor_covariance, specific_variances
)
```

### 🔄 Statistical Arbitrage

Pairs trading and mean reversion:

```python
from meridianalgo.quant import (
    PairsTrading, CointegrationAnalyzer, 
    OrnsteinUhlenbeck, HiddenMarkovModel
)

# Cointegration test
coint = CointegrationAnalyzer()
result = coint.engle_granger_test(stock1, stock2)

# Pairs trading strategy
pt = PairsTrading(entry_threshold=2.0, exit_threshold=0.5)
hedge_ratio = pt.calculate_hedge_ratio(stock1, stock2)
signals = pt.generate_signals(stock1, stock2)

# Mean reversion dynamics (OU process)
ou = OrnsteinUhlenbeck()
params = ou.fit(spread)
print(f"Half-life: {params['half_life']:.1f} days")

# Regime detection
hmm = HiddenMarkovModel(n_states=2)
results = hmm.fit(returns)
current_regime = hmm.predict_state(returns).iloc[-1]
```

---

## 🎓 Use Cases

### **Hedge Funds & Prop Trading**
- Statistical arbitrage strategies
- High-frequency signal generation
- Multi-factor alpha models
- Risk-adjusted portfolio construction

### **Asset Managers**
- Factor-based investing
- Portfolio optimization (MVO, Black-Litterman, Risk Parity)
- Transaction cost analysis
- Performance attribution

### **Quantitative Researchers**
- Market microstructure analysis
- Regime detection and forecasting
- Cointegration and mean reversion testing
- VPIN and flow toxicity analysis

### **Risk Managers**
- VaR and CVaR calculation
- Stress testing and scenario analysis
- Drawdown risk monitoring
- Tail risk analysis

---

## ⚙️ Configuration

```python
import meridianalgo as ma

# Configure the library
ma.set_config(
    data_provider='yahoo',       # Data source
    cache_enabled=True,          # Enable caching
    parallel_processing=True,    # Use multiprocessing
    risk_free_rate=0.05,         # Default risk-free rate
    trading_days_per_year=252,   # Trading days
)

# Enable GPU acceleration (if available)
ma.enable_gpu_acceleration()

# Enable distributed computing
ma.enable_distributed_computing(backend='ray')

# Get system info
info = ma.get_system_info()
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_analytics.py -v
pytest tests/test_liquidity.py -v

# Run with coverage
pytest tests/ --cov=meridianalgo --cov-report=html

# Run performance benchmarks
pytest tests/benchmarks/ -v
```

---

## 📚 Documentation

- **API Reference**: [docs.meridianalgo.com](https://docs.meridianalgo.com)
- **Tutorials**: [tutorials/](tutorials/)
- **Examples**: [examples/](examples/)
- **Cookbook**: [cookbook/](docs/cookbook/)

---

## 🏗️ Architecture

```
meridianalgo/
├── analytics/           # Pyfolio-style analytics & tear sheets
│   ├── performance.py   # Performance metrics (50+ measures)
│   ├── risk_analytics.py # Risk analysis (VaR, CVaR, GARCH)
│   ├── tear_sheets.py   # Visual tear sheet generation
│   ├── attribution.py   # Performance attribution (Brinson, Factor)
│   └── drawdown.py      # Drawdown analysis
│
├── liquidity/           # Market microstructure & liquidity
│   ├── order_book.py    # Order book analysis, microprice
│   ├── microstructure.py # PIN, VPIN, spread decomposition
│   ├── spread.py        # Effective & realized spread
│   ├── volume.py        # Volume profile, institutional flow
│   ├── impact.py        # Market impact models (Almgren-Chriss)
│   └── metrics.py       # Amihud, turnover, liquidity ratios
│
├── portfolio/           # Portfolio optimization
│   ├── optimization.py  # MVO, Black-Litterman, HRP
│   ├── risk_parity.py   # Risk parity strategies
│   └── rebalancing.py   # Rebalancing algorithms
│
├── risk/                # Risk management
│   ├── var.py           # VaR calculations
│   ├── stress_test.py   # Stress testing
│   └── scenario.py      # Scenario analysis
│
├── derivatives/         # Options & derivatives
│   ├── pricing.py       # Black-Scholes, Binomial, Monte Carlo
│   ├── greeks.py        # Greeks calculation
│   └── volatility.py    # Vol surface, local vol
│
├── execution/           # Execution algorithms
│   ├── vwap.py          # VWAP execution
│   ├── twap.py          # TWAP execution
│   └── impact.py        # Implementation shortfall
│
├── quant/               # Quantitative strategies
│   ├── pairs_trading.py # Pairs trading, cointegration
│   ├── regime.py        # Regime detection (HMM)
│   └── arbitrage.py     # Statistical arbitrage
│
├── factors/             # Factor models
│   ├── fama_french.py   # Fama-French models
│   └── factor_risk.py   # Factor risk decomposition
│
├── ml/                  # Machine learning
├── signals/             # Technical indicators
├── backtesting/         # Backtesting engine
├── data/                # Data management
└── fixed_income/        # Fixed income analytics
```

---

## 🌟 What's New in v6.0.0

### Major Features
- **Pyfolio-Style Analytics**: Complete tear sheet generation with 50+ metrics
- **Comprehensive Liquidity Module**: Order book, VPIN, market impact, spread decomposition
- **Modern Architecture**: Lazy loading, configuration management, GPU support
- **Type Hints Throughout**: Full typing for better IDE support

### Improvements
- Modular design for better code organization
- Enhanced error handling and validation
- Performance optimizations across all modules
- Extended documentation and examples

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/MeridianAlgo/Python-Packages/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MeridianAlgo/Python-Packages/discussions)
- **Email**: support@meridianalgo.com

---

## 🌟 Citation

```bibtex
@software{meridianalgo2025,
  title = {MeridianAlgo: The Complete Quantitative Finance Platform},
  author = {Meridian Algorithmic Research Team},
  year = {2025},
  version = {6.0.0},
  url = {https://github.com/MeridianAlgo/Python-Packages}
}
```

---

**MeridianAlgo v6.0.0** — *The Complete Quantitative Finance Platform*

*Built by students for quantitative finance.*