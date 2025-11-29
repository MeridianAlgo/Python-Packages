# MeridianAlgo v5.0.0
## Advanced Quantitative Development Platform

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI Version](https://img.shields.io/badge/pypi-5.0.0-orange.svg)](https://pypi.org/project/meridianalgo/)
[![Tests](https://img.shields.io/badge/tests-200%2B%20passing-brightgreen.svg)](tests/)
[![Code Quality](https://img.shields.io/badge/code%20quality-A-brightgreen.svg)]()

**Enterprise-Grade Quantitative Finance Platform for Professional Developers**

MeridianAlgo is the most comprehensive Python platform for institutional quantitative finance, trusted by hedge funds, asset managers, and quantitative researchers worldwide. Features cutting-edge algorithms for market microstructure, statistical arbitrage, optimal execution, high-frequency trading, factor models, and advanced risk management.

---

## 🎯 What Makes MeridianAlgo Different

### **Production-Ready Algorithms**
Every algorithm is implemented to institutional standards with proper error handling, parameter validation, and performance optimization.

### **Academic Rigor**
Based on peer-reviewed research from leading academics and practitioners (Almgren & Chriss, Avellaneda & Stoikov, Fama & French, and others).

### **Comprehensive Testing**
Over 200 test cases ensuring reliability in production environments.

### **Professional Documentation**
Clear, complete documentation with mathematical formulations and real-world examples.

---

## 🚀 Quick Start

### Installation

```bash
# Standard installation
pip install meridianalgo

# With machine learning support
pip install meridianalgo[ml]

# Complete installation (recommended for professionals)
pip install meridianalgo[all]

# Development installation
git clone https://github.com/MeridianAlgo/Python-Packages.git
cd Python-Packages
pip install -e .[dev]
```

### Basic Usage

```python
import meridianalgo as ma

# Get market data
data = ma.api_get_market_data(['AAPL', 'GOOGL'], '2023-01-01', '2023-12-31')

# Calculate technical indicators
rsi = ma.RSI(data['AAPL'], period=14)
macd_line, signal_line, histogram = ma.MACD(data['AAPL'])

# Risk analysis
metrics = ma.api_calculate_risk_metrics(data['AAPL'].pct_change())
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"VaR (95%): {metrics['var_95']:.2%}")
```

---

## 💼 Advanced Quantitative Development

### 1. **Market Microstructure Analysis**

Professional tools for analyzing market microstructure and order flow:

```python
from meridianalgo.quant import OrderFlowImbalance, RealizedVolatility, MarketImpactModel

# Order flow analysis
ofi = OrderFlowImbalance()
imbalance = ofi.calculate_ofi(bid_volumes, ask_volumes, bid_prices, ask_prices)

# Realized volatility (institutional standard)
rv = RealizedVolatility.rv_5min(high_freq_prices, freq='5min')
bipower_var = RealizedVolatility.bipower_variation(returns)

# Market impact estimation
impact_model = MarketImpactModel()
expected_impact = impact_model.square_root_law(
    order_size=10000, daily_volume=500000, sigma=0.02
)
```

### 2. **Statistical Arbitrage**

Complete framework for statistical arbitrage strategies:

```python
from meridianalgo.quant import PairsTrading, CointegrationAnalyzer, OrnsteinUhlenbeck

# Pairs trading with dynamic hedge ratio
pt = PairsTrading(entry_threshold=2.0, exit_threshold=0.5)
hedge_ratio = pt.calculate_hedge_ratio(stock1, stock2, method='tls')
signals = pt.generate_signals(stock1, stock2, window=20)

# Test for cointegration
analyzer = CointegrationAnalyzer()
result = analyzer.engle_granger_test(stock1, stock2)

# Ornstein-Uhlenbeck process modeling
ou = OrnsteinUhlenbeck()
params = ou.fit(spread_series)
print(f"Half-life: {params['half_life']:.1f} days")
```

### 3. **Optimal Execution Algorithms**

Institutional-grade execution algorithms:

```python
from meridianalgo.quant import VWAP, TWAP, ImplementationShortfall

# VWAP execution
vwap = VWAP(total_quantity=10000, start_time='09:30', end_time='16:00')
schedule = vwap.calculate_schedule(historical_volume_profile)

# Implementation Shortfall (Almgren-Chriss)
is_algo = ImplementationShortfall(
    total_quantity=50000, total_time=1.0, volatility=0.02, risk_aversion=1e-6
)
trajectory = is_algo.calculate_optimal_trajectory()
cost_analysis = is_algo.calculate_expected_cost()
```

### 4. **High-Frequency Trading**

Professional HFT strategies:

```python
from meridianalgo.quant import MarketMaking, LatencyArbitrage, HFTSignalGenerator

# Market making (Avellaneda-Stoikov model)
mm = MarketMaking(target_spread_bps=5.0, max_inventory=1000)
bid_price, ask_price = mm.calculate_quotes(mid_price=100, volatility=0.02)

# Latency arbitrage detection
arb = LatencyArbitrage(latency_threshold_us=100.0, min_profit_bps=1.0)
opportunity = arb.detect_opportunity(venue1_price, venue1_time, venue2_price, venue2_time)
```

### 5. **Factor Models**

Multi-factor models for portfolio construction:

```python
from meridianalgo.quant import FamaFrenchModel, APTModel, FactorRiskDecomposition

# Fama-French three-factor model
ff = FamaFrenchModel(model_type='three_factor')
results = ff.fit(asset_returns, factor_data)

# Factor risk decomposition
decomp = FactorRiskDecomposition.decompose_variance(
    portfolio_weights, factor_exposures, factor_covariance, specific_variances
)
```

### 6. **Regime Detection**

Advanced regime detection and market state classification:

```python
from meridianalgo.quant import HiddenMarkovModel, StructuralBreakDetection

# Hidden Markov Model for regime detection
hmm = HiddenMarkovModel(n_states=2)
results = hmm.fit(returns)
current_regime = hmm.predict_state(recent_returns).iloc[-1]

# Detect structural breaks
sbd = StructuralBreakDetection()
breaks = sbd.cusum_test(returns)
```

---

## 📊 Portfolio Management & Risk

### Portfolio Optimization

```python
# Maximum Sharpe Ratio
sharpe_weights = ma.api_optimize_portfolio(returns, method='sharpe')

# Black-Litterman Model
bl_model = ma.BlackLitterman(returns, market_caps)
bl_weights = bl_model.optimize_with_views({'AAPL': 0.15, 'MSFT': 0.12})

# Risk Parity
rp_model = ma.RiskParity(returns)
rp_weights = rp_model.optimize()

# Efficient Frontier
frontier = ma.EfficientFrontier(returns)
frontier_data = frontier.calculate_frontier(target_returns)
```

### Risk Management

```python
# Value at Risk (Multiple Methods)
historical_var = ma.HistoricalVaR(returns, confidence_level=0.95)
parametric_var = ma.ParametricVaR(returns, confidence_level=0.99)
monte_carlo_var = ma.MonteCarloVaR(returns, n_simulations=10000)

# Expected Shortfall (CVaR)
es = ma.calculate_expected_shortfall(returns, confidence_level=0.95)

# Comprehensive risk metrics
metrics = ma.api_calculate_risk_metrics(returns)
```

---

## 🤖 Machine Learning for Trading

```python
# Feature engineering
engineer = ma.FeatureEngineer()
features = engineer.create_features(prices)

# LSTM for price prediction
predictor = ma.LSTMPredictor(sequence_length=60, epochs=100)
X, y = ma.prepare_data_for_lstm(prices.values)
predictor.fit(X_train, y_train)
predictions = predictor.predict(X_test)
```

---

## 📦 Package Structure

```
meridianalgo/
├── quant/                          # Advanced Quantitative Algorithms
│   ├── market_microstructure.py   # Order flow, realized vol, market impact
│   ├── statistical_arbitrage.py   # Pairs trading, cointegration
│   ├── execution_algorithms.py    # VWAP, TWAP, POV, Implementation Shortfall
│   ├── high_frequency.py          # Market making, latency arbitrage
│   ├── factor_models.py           # Fama-French, APT, custom factors
│   └── regime_detection.py        # HMM, structural breaks
├── portfolio_management/          # Portfolio optimization
├── risk_analysis/                 # Risk management
├── backtesting/                   # Backtesting engine
├── technical_indicators/          # 200+ technical indicators
├── ml/                           # Machine learning models
├── derivatives/                   # Options & derivatives
├── fixed_income/                  # Bond pricing
├── forex/                        # FX analysis
└── crypto/                       # Cryptocurrency tools
```

---

## 🎓 Use Cases by Professional Type

### **Hedge Funds**
- Statistical arbitrage strategies
- Multi-factor alpha generation
- High-frequency trading
- Risk-adjusted portfolio construction

### **Asset Managers**
- Factor-based investing
- Portfolio optimization (Markowitz, Black-Litterman, Risk Parity)
- Transaction cost analysis
- Performance attribution

### **Quantitative Researchers**
- Market microstructure analysis
- Regime detection and forecasting
- Cointegration and mean reversion testing
- Factor model development

### **Proprietary Trading Firms**
- Optimal execution algorithms
- Market making strategies
- Latency arbitrage
- Real-time risk monitoring

---

## 🧪 Testing & Quality

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_quant.py -v

# Run with coverage
pytest tests/ --cov=meridianalgo --cov-report=html

# Run integration tests
pytest tests/integration/ -v
```

**Test Coverage**: 200+ test cases | 90%+ code coverage

---

## 📚 Documentation

- **API Reference**: `docs/api/`
- **User Guide**: `docs/user_guide/`
- **Tutorials**: `docs/tutorials/`
- **Examples**: `examples/`

### Examples

```bash
python examples/quant_examples.py              # Quant algorithms demo
python examples/advanced_trading_strategy.py   # Trading strategy
python examples/basic_usage.py                 # Getting started
```

---

## 🏗️ Development

### Setup Development Environment

```bash
git clone https://github.com/MeridianAlgo/Python-Packages.git
cd Python-Packages
pip install -e .[dev]

# Run tests
pytest tests/

# Format code
black meridianalgo/
flake8 meridianalgo/
```

### Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 💬 Support & Community

- **GitHub Issues**: [Report bugs](https://github.com/MeridianAlgo/Python-Packages/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/MeridianAlgo/Python-Packages/discussions)
- **Email**: support@meridianalgo.com
- **Documentation**: [Full docs](https://meridianalgo.readthedocs.io)

---

## 🌟 Citation

```bibtex
@software{meridianalgo2024,
  title = {MeridianAlgo: Advanced Quantitative Development Platform},
  author = {Meridian Algorithmic Research Team},
  year = {2024},
  version = {5.0.0},
  url = {https://github.com/MeridianAlgo/Python-Packages}
}
```

---

## 🚀 Changelog

### v5.0.0 (2024-11-29) - "Advanced Quantitative Development Edition"

#### ✨ New Features
- **Professional Quant Module**: Complete suite of institutional-grade algorithms
  - Market microstructure analysis (order flow, VPIN, realized volatility)
  - Statistical arbitrage (pairs trading, cointegration, OU process)
  - Execution algorithms (VWAP, TWAP, POV, Implementation Shortfall)
  - High-frequency trading (market making, latency arbitrage)
  - Factor models (Fama-French, APT, custom factors)
  - Regime detection (HMM, structural breaks, market classification)

#### 🔧 Improvements
- Reorganized package structure for better clarity
- Enhanced documentation with 100+ examples
- Complete test coverage (200+ tests)
- Improved error handling and validation
- Performance optimizations throughout

#### 📚 Documentation
- New comprehensive README
- Professional examples for all modules  
- Updated API documentation
- Real-world use case guides

#### 🧪 Testing
- 200+ unit tests
- Integration test suite
- Comprehensive test coverage
- Mock data generators

---

## 🎯 Roadmap

### v5.1.0 (Q1 2025)
- GPU acceleration for ML models
- Real-time data streaming
- Enhanced visualization tools
- Additional execution algorithms

### v5.2.0 (Q2 2025)
- Distributed computing support
- Cloud deployment tools
- Advanced options pricing models
- ESG factor integration

---

**MeridianAlgo v5.0.0 - Advanced Quantitative Development Platform**

*Built by quantitative professionals, for quantitative professionals.*

**Empowering the next generation of quantitative finance.**