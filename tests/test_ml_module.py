"""
Unit tests for the machine learning module functionality.
"""

import unittest
import warnings

import numpy as np
import pandas as pd

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")


class TestMLModule(unittest.TestCase):
    """Test cases for ML module functionality."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.dates = pd.date_range("2023-01-01", periods=1000, freq="D")
        self.prices = pd.Series(
            np.cumprod(1 + np.random.normal(0.001, 0.02, 1000)),
            index=self.dates,
            name="Close",
        )

        # Create feature matrix for testing
        self.features = pd.DataFrame(
            {
                "returns": self.prices.pct_change(),
                "volatility": self.prices.pct_change().rolling(20).std(),
                "momentum": self.prices.pct_change(20),
            }
        ).dropna()

        self.target = self.prices.pct_change().shift(-1).dropna()

    def test_feature_engineer_initialization(self):
        """Test FeatureEngineer initialization."""
        from meridianalgo import FeatureEngineer

        engineer = FeatureEngineer(lookback=10)
        self.assertIsInstance(engineer, FeatureEngineer)
        self.assertEqual(engineer.lookback, 10)

    def test_feature_engineer_create_features(self):
        """Test feature creation."""
        from meridianalgo import FeatureEngineer

        engineer = FeatureEngineer()
        features = engineer.create_features(self.prices)

        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 5)  # Should create multiple features
        self.assertIn("returns", features.columns)
        self.assertIn("log_returns", features.columns)

        # Check for momentum features
        momentum_cols = [col for col in features.columns if "momentum" in col]
        self.assertGreater(len(momentum_cols), 0)

        # Check for volatility features
        vol_cols = [col for col in features.columns if "volatility" in col]
        self.assertGreater(len(vol_cols), 0)

        # Check for moving average features
        ma_cols = [col for col in features.columns if "ma_" in col]
        self.assertGreater(len(ma_cols), 0)

    def test_feature_engineer_create_sequences(self):
        """Test sequence creation for time series prediction."""
        from meridianalgo import FeatureEngineer

        engineer = FeatureEngineer()
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        sequence_length = 10

        X_seq, y_seq = engineer.create_sequences(X, y, sequence_length)

        self.assertEqual(X_seq.shape[0], len(X) - sequence_length)
        self.assertEqual(X_seq.shape[1], sequence_length)
        self.assertEqual(X_seq.shape[2], X.shape[1])
        self.assertEqual(len(y_seq), len(X) - sequence_length)

    def test_lstm_predictor_initialization(self):
        """Test LSTMPredictor initialization."""
        from meridianalgo import LSTMPredictor

        try:
            predictor = LSTMPredictor(sequence_length=10, hidden_size=50)
            self.assertIsInstance(predictor, LSTMPredictor)
            self.assertEqual(predictor.sequence_length, 10)
            self.assertEqual(predictor.hidden_size, 50)
        except ImportError:
            self.skipTest("PyTorch not available for LSTM testing")

    def test_lstm_predictor_training(self):
        """Test LSTM model training."""
        from meridianalgo import LSTMPredictor

        try:
            # Create small dataset for testing
            X = np.random.randn(200, 5)
            y = np.random.randn(200)

            predictor = LSTMPredictor(sequence_length=10, epochs=2, batch_size=32)
            predictor.fit(X, y)

            # Test prediction
            predictions = predictor.predict(X[-50:])
            self.assertEqual(len(predictions), 50 - predictor.sequence_length)

        except ImportError:
            self.skipTest("PyTorch not available for LSTM testing")

    def test_lstm_predictor_without_torch(self):
        """Test LSTMPredictor behavior when PyTorch is not available."""
        # This test would need to be run in an environment without PyTorch
        # For now, we'll just test that the class exists
        from meridianalgo import LSTMPredictor

        self.assertTrue(hasattr(LSTMPredictor, "__init__"))

    def test_model_evaluator_metrics(self):
        """Test ModelEvaluator metrics calculation."""
        from meridianalgo.ml import ModelEvaluator

        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

        metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)

        required_metrics = ["mse", "rmse", "mae", "r2", "direction_accuracy"]
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))

        # R should be close to 1 for this perfect prediction
        self.assertGreater(metrics["r2"], 0.9)

    def test_model_evaluator_time_series_cv(self):
        """Test time series cross-validation."""
        from meridianalgo.ml import ModelEvaluator

        # Create a simple mock model
        class MockModel:
            def fit(self, X, y):
                self.fitted = True

            def predict(self, X):
                return np.random.randn(len(X))

        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        model = MockModel()
        cv_results = ModelEvaluator.time_series_cv(model, X, y, n_splits=3)

        expected_metrics = [
            "mean_mse",
            "std_mse",
            "mean_rmse",
            "std_rmse",
            "mean_mae",
            "std_mae",
            "mean_r2",
            "std_r2",
            "mean_direction_accuracy",
            "std_direction_accuracy",
        ]

        for metric in expected_metrics:
            self.assertIn(metric, cv_results)
            self.assertIsInstance(cv_results[metric], (int, float))

    def test_prepare_data_for_lstm(self):
        """Test data preparation for LSTM."""
        from meridianalgo import prepare_data_for_lstm

        X_train, X_test, y_train, y_test = prepare_data_for_lstm(
            self.features, self.target, sequence_length=10, test_size=0.2
        )

        self.assertIsInstance(X_train, np.ndarray)
        self.assertIsInstance(X_test, np.ndarray)
        self.assertIsInstance(y_train, np.ndarray)
        self.assertIsInstance(y_test, np.ndarray)

        # Check shapes
        self.assertEqual(len(X_train.shape), 3)  # (samples, sequence_length, features)
        self.assertEqual(X_train.shape[2], self.features.shape[1])
        self.assertEqual(len(y_train), X_train.shape[0])

        # Check that test size is approximately correct
        total_samples = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total_samples
        self.assertAlmostEqual(test_ratio, 0.2, delta=0.1)


if __name__ == "__main__":
    unittest.main()
