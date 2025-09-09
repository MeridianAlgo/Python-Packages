# Changelog

All notable changes to the MeridianAlgo package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2025-09-08

### Added
- **Machine Learning Enhancements**
  - Added support for PyTorch-based LSTM models
  - Implemented feature engineering pipeline
  - Added data preprocessing and scaling utilities

### Changed
- **Dependencies**
  - Updated required Python version to 3.8+
  - Added PyTorch as a core dependency
  - Updated all existing dependencies to their latest stable versions

### Fixed
- **Bug Fixes**
  - Resolved issues with empty data handling in ML pipelines
  - Fixed compatibility issues with newer versions of dependencies
  - Improved error handling and logging throughout the codebase

## [2.2.1] - 2025-09-08

### Added
- **Documentation Overhaul**
  - Completely redesigned README with better organization and visual hierarchy
  - Added comprehensive installation and quick start guides
  - Included detailed feature documentation with code examples
  - Added performance metrics and system requirements

### Changed
- **Package Structure**
  - Updated version to 2.2.1 to reflect documentation improvements
  - Enhanced module imports and organization
  - Improved error messages and logging

### Fixed
- **Documentation**
  - Fixed broken links and outdated information
  - Corrected code examples and usage instructions
  - Ensured all API references are up-to-date

## [2.2.0] - 2025-09-08

### Added
- **Advanced Statistical Analysis**
  - New `StatisticalArbitrage` class for pairs trading strategies
  - Cointegration tests and correlation analysis
  - Rolling correlation calculations
  - Hurst exponent for mean reversion/trend detection

- **Risk Metrics**
  - Value at Risk (VaR) calculation
  - Expected Shortfall (CVaR) implementation
  - Maximum Drawdown analysis
  - Comprehensive input validation and error handling

- **Performance Metrics**
  - Sharpe Ratio calculation
  - Sortino Ratio implementation
  - Risk-adjusted return metrics

## [2.1.0] - 2025-08-02

### Bug Fix Release
- Enhanced Ara AI integration with latest improvements
- Updated ensemble ML models with better accuracy
- Improved GPU support and performance optimizations
- Enhanced caching system and prediction validation
- Updated documentation and examples

## [2.0.0] - 2024-01-29

### Major Release - Complete System Overhaul
- Added ensemble ML system with Random Forest, Gradient Boosting, and LSTM
- Implemented 50+ technical indicators
- Added multi-GPU support (NVIDIA, AMD, Intel, Apple)
- Comprehensive prediction validation and accuracy tracking
- Professional console output with rich formatting
