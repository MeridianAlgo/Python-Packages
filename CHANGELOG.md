# Changelog

All notable changes to this project will be documented in this file.

## [3.1.0] - 2025-01-27

### Added
- **Comprehensive Technical Indicators Module** (50+ indicators)
  - Momentum indicators: RSI, Stochastic, Williams %R, ROC, Momentum
  - Trend indicators: SMA, EMA, MACD, ADX, Aroon, Parabolic SAR, Ichimoku Cloud
  - Volatility indicators: Bollinger Bands, ATR, Keltner Channels, Donchian Channels
  - Volume indicators: OBV, AD Line, Chaikin Oscillator, Money Flow Index, Ease of Movement
  - Overlay indicators: Pivot Points, Fibonacci Retracement, Support and Resistance

- **Advanced Portfolio Management Module**
  - Modern Portfolio Theory (MPT) optimization
  - Black-Litterman model implementation
  - Risk Parity portfolio optimization
  - Efficient Frontier calculation
  - Portfolio rebalancing strategies

- **Comprehensive Risk Analysis Module**
  - Value at Risk (VaR) - Historical, Parametric, Monte Carlo methods
  - Expected Shortfall (CVaR) calculation
  - Stress testing and scenario analysis
  - Risk metrics: Sharpe, Sortino, Calmar ratios
  - Drawdown analysis and tail risk metrics
  - Market regime detection

- **Data Processing Module**
  - Data cleaning and validation utilities
  - Feature engineering for financial data
  - Market data providers with caching
  - Outlier detection and missing data handling

- **Modular Package Structure**
  - Organized codebase with specialized modules
  - Clean separation of concerns
  - Easy to import specific functionality

### Enhanced
- **Improved Documentation**
  - Comprehensive README with detailed examples
  - API reference for all modules
  - Usage examples and tutorials
  - Performance metrics and benchmarks

- **Better Testing**
  - Comprehensive test suite (40+ tests)
  - Unit tests for all modules
  - Integration tests for end-to-end functionality
  - Demo script showcasing all features

- **Code Quality**
  - Removed duplicate dependencies
  - Fixed import issues and circular dependencies
  - Improved error handling and validation
  - Better code organization and structure

### Fixed
- Fixed market data fetching compatibility with yfinance updates
- Fixed LSTM model inheritance issues
- Fixed volatility calculation tests
- Fixed import issues in statistics module
- Removed duplicate code and dependencies

### Technical Details
- **Dependencies**: Updated to latest versions with proper version constraints
- **Python Support**: Python 3.7+ (tested on 3.7-3.11)
- **Performance**: Optimized calculations and memory usage
- **Compatibility**: Works with latest versions of NumPy, Pandas, PyTorch

### Credits and Acknowledgments
- **Quant Analytics Integration**: Portions of this library integrate concepts and methodologies from the [quant-analytics](https://pypi.org/project/quant-analytics/) package by Anthony Baxter
- **Open Source Libraries**: Built on NumPy, Pandas, SciPy, Scikit-learn, PyTorch
- **Community**: Inspired by quantitative finance best practices and community feedback

## [3.0.0] - 2024-12-15

### Added
- Initial release with core functionality
- Portfolio optimization using Modern Portfolio Theory
- Time series analysis and technical indicators
- Machine learning integration with LSTM models
- Statistical analysis tools
- Risk metrics calculation
- Yahoo Finance data integration

### Features
- Portfolio optimization and efficient frontier calculation
- Time series analysis with returns and volatility calculation
- Risk management with VaR and Expected Shortfall
- Machine learning with LSTM prediction models
- Statistical arbitrage and correlation analysis
- Market data fetching from Yahoo Finance

---

## Installation

```bash
# Install latest version
pip install meridianalgo

# Install specific version
pip install meridianalgo==3.1.0

# Install with development dependencies
pip install meridianalgo[dev]
```

## Migration Guide

### From 3.0.0 to 3.1.0

The new version is fully backward compatible. Existing code will continue to work without changes. New features are available through additional imports:

```python
# New technical indicators
from meridianalgo import RSI, MACD, BollingerBands

# New portfolio management
from meridianalgo import EfficientFrontier, BlackLitterman

# New risk analysis
from meridianalgo import VaRCalculator, StressTester
```

## Breaking Changes

None in this release. All existing functionality remains unchanged.

## Deprecations

None in this release.

## Security

No security issues identified in this release.

## Performance

- Improved calculation speed for technical indicators
- Optimized memory usage for large datasets
- Better handling of missing data and edge cases
- Enhanced parallel processing capabilities

## Documentation

- Complete API documentation
- Comprehensive examples and tutorials
- Performance benchmarks
- Best practices guide