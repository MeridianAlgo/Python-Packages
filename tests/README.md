# MeridianAlgo Test Suite

This directory contains comprehensive tests for the MeridianAlgo package.

## Test Files

### test_core.py
Tests for core functionality:
- Statistics calculations
- Performance metrics
- Risk measures
- Time series analysis

### test_machine_learning.py
Tests for machine learning features:
- Feature engineering
- LSTM predictors
- Model validation
- Data preprocessing

### test_comprehensive_suite.py
Comprehensive integration tests covering:
- Portfolio optimization
- Risk analytics
- Backtesting
- Data handling
- All major modules

## Running Tests

Run all tests:
```bash
pytest tests/
```

Run specific test file:
```bash
pytest tests/test_core.py
pytest tests/test_machine_learning.py
pytest tests/test_comprehensive_suite.py
```

Run with verbose output:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=meridianalgo --cov-report=html
```

## Test Configuration

Test configuration is in `pytest.ini` at the project root.

## Writing Tests

When adding new features, please include tests that:
- Cover normal use cases
- Test edge cases
- Validate error handling
- Check numerical accuracy
- Ensure backward compatibility

## Continuous Integration

Tests run automatically on:
- Every push to main/develop branches
- All pull requests
- Before PyPI publishing

See `.github/workflows/ci.yml` for CI configuration.
