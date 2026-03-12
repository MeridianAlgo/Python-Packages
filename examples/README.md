# MeridianAlgo Examples

This directory contains comprehensive examples demonstrating the capabilities of MeridianAlgo for quantitative finance and algorithmic trading.

## Getting Started

Run the examples in order to learn MeridianAlgo progressively:

### 01. Getting Started
**File:** `01_getting_started.py`

Your first steps with MeridianAlgo. Learn how to:
- Fetch market data
- Calculate returns and basic statistics
- Analyze risk metrics
- Understand correlation between assets
- Build a simple portfolio

Perfect for beginners who are new to quantitative finance.

### 02. Basic Usage
**File:** `02_basic_usage.py`

Core functionality for everyday quantitative analysis:
- Portfolio optimization using Modern Portfolio Theory
- Time series analysis and performance metrics
- Advanced risk measurement (VaR, Expected Shortfall, Hurst exponent)
- Statistical arbitrage and pairs trading
- Machine learning for price prediction

### 03. Advanced Trading Strategy
**File:** `03_advanced_trading_strategy.py`

Build and backtest a complete trading strategy:
- Mean-reversion strategy implementation
- Signal generation using z-scores
- Backtesting framework
- Performance evaluation
- Risk analysis of strategies

### 04. Comprehensive Examples
**File:** `04_comprehensive_examples.py`

Showcase of all major features:
- Portfolio analytics (Pyfolio-style)
- Liquidity analysis and market microstructure
- Technical indicators and signals
- Derivatives pricing (Black-Scholes, Greeks)
- Factor models and attribution
- Drawdown analysis

### 05. Quantitative Strategies
**File:** `05_quant_examples.py`

Professional quantitative finance algorithms:
- Market microstructure analysis
- Statistical arbitrage and pairs trading
- Optimal execution algorithms (TWAP, VWAP, Implementation Shortfall)
- High-frequency trading strategies
- Factor models (Fama-French)
- Regime detection with Hidden Markov Models

### 06. Transaction Cost Optimization
**File:** `06_transaction_cost_optimization.py`

Advanced portfolio management with transaction costs:
- Execution algorithms comparison
- Market impact models
- Tax-loss harvesting
- Transaction-cost-aware portfolio optimization
- Rebalancing frequency optimization

## Running the Examples

Each example is self-contained and can be run independently:

```bash
python examples/01_getting_started.py
python examples/02_basic_usage.py
python examples/03_advanced_trading_strategy.py
python examples/04_comprehensive_examples.py
python examples/05_quant_examples.py
python examples/06_transaction_cost_optimization.py
```

## Requirements

All examples require the base MeridianAlgo installation:

```bash
pip install meridianalgo
```

Some examples may require additional dependencies:

```bash
pip install meridianalgo[all]  # Install all optional dependencies
```

## Learning Path

1. Start with `01_getting_started.py` to understand the basics
2. Move to `02_basic_usage.py` for core functionality
3. Try `03_advanced_trading_strategy.py` to build your first strategy
4. Explore `04_comprehensive_examples.py` to see all features
5. Study `05_quant_examples.py` for professional techniques
6. Master `06_transaction_cost_optimization.py` for real-world trading

## Support

For questions or issues:
- Documentation: See the `docs/` folder
- GitHub Issues: Report bugs or request features
- Examples: All examples include detailed comments explaining each step

## Contributing

Found a bug or want to add an example? Contributions are welcome! Please ensure:
- Code is well-commented
- Examples are self-contained
- All code passes linting (ruff, isort, black)
