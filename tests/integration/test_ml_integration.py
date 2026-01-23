"""
Integration test for machine learning models and feature engineering.
"""

import os
import sys

import numpy as np
import pandas as pd

# Add the package to path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def create_sample_data():
    """Create sample financial data for testing."""
    np.random.seed(42)
    n_samples = 500
    n_features = 10

    # Create sample features (technical indicators, etc.)
    X = np.random.randn(n_samples, n_features)

    # Create target (next period return) with some signal
    y = (
        0.1 * X[:, 0]
        + 0.05 * X[:, 1]
        - 0.03 * X[:, 2]
        + np.random.randn(n_samples) * 0.02
    )

    return X, y


def create_sample_price_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2023-06-30", freq="D")

    # Generate realistic price data
    n_periods = len(dates)
    returns = np.random.normal(0.0008, 0.015, n_periods)
    prices = 100 * np.exp(np.cumsum(returns))

    # Add intraday variation
    high_low_range = np.random.uniform(0.005, 0.03, n_periods)

    data = pd.DataFrame(
        {
            "Open": prices * (1 + np.random.uniform(-0.005, 0.005, n_periods)),
            "High": prices * (1 + high_low_range),
            "Low": prices * (1 - high_low_range),
            "Close": prices,
            "Volume": np.random.randint(100000, 1000000, n_periods),
        },
        index=dates,
    )

    return data


def test_feature_engineering():
    """Test financial feature engineering."""
    print("Testing Financial Feature Engineering...")

    try:
        from meridianalgo.machine_learning.feature_engineering import \
            TechnicalIndicatorFeatures

        data = create_sample_price_data()
        generator = TechnicalIndicatorFeatures(periods=[5, 10, 20])

        features = generator.generate_features(data)

        assert len(features) == len(data), "Feature length should match data length"
        assert len(features.columns) > 0, "Should generate some features"

        print(" Feature engineering working correctly")
        print(f"  Generated {len(features.columns)} features")
        print(f"  Data shape: {features.shape}")

        return True

    except Exception as e:
        print(f" Feature engineering test failed: {e}")
        return False


def test_ml_models():
    """Test machine learning models."""
    print("\nTesting ML Models...")

    try:
        from meridianalgo.machine_learning.models import (ModelConfig,
                                                          ModelFactory)

        X, y = create_sample_data()

        # Test Random Forest
        try:
            config = ModelConfig(model_type="random_forest", n_estimators=10)
            model = ModelFactory.create_model(config)
            model.fit(X, y)
            predictions = model.predict(X)

            assert len(predictions) == len(y), "Predictions should match target length"
            print(" Random Forest model working")

        except ImportError:
            print(" Random Forest skipped (scikit-learn not available)")

        return True

    except Exception as e:
        print(f" ML models test failed: {e}")
        return False


def test_time_series_validation():
    """Test time-series validation methods."""
    print("\nTesting Time-Series Validation...")

    try:
        from meridianalgo.machine_learning.validation import \
            WalkForwardValidator

        X, y = create_sample_data()

        # Test walk-forward validation
        validator = WalkForwardValidator(n_splits=3, test_size=50)
        splits = list(validator.split(X, y))

        assert len(splits) == 3, "Should generate correct number of splits"

        for train_idx, test_idx in splits:
            assert len(train_idx) > 0, "Training set should not be empty"
            assert len(test_idx) > 0, "Test set should not be empty"
            assert max(train_idx) < min(test_idx), "Training should come before test"

        print(" Time-series validation working correctly")
        print(f"  Generated {len(splits)} validation splits")

        return True

    except Exception as e:
        print(f" Time-series validation test failed: {e}")
        return False


def main():
    """Run all ML integration tests."""
    print("Machine Learning Integration Tests")
    print("=" * 50)

    tests = [test_feature_engineering, test_ml_models, test_time_series_validation]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n{'=' * 50}")
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print(" All ML integration tests passed!")
        return True
    else:
        print(" Some ML integration tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
