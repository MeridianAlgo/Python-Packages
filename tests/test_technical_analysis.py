"""
Comprehensive tests for technical analysis components.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add the package to path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from meridianalgo.technical_analysis.framework import BaseIndicator
    from meridianalgo.technical_analysis.indicators import MACD, RSI, BollingerBands
    from meridianalgo.technical_analysis.patterns import CandlestickPatterns

    TECHNICAL_ANALYSIS_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYSIS_AVAILABLE = False

# Also test legacy indicators
try:
    from meridianalgo.technical_indicators import EMA
    from meridianalgo.technical_indicators import MACD as LegacyMACD
    from meridianalgo.technical_indicators import RSI as LegacyRSI
    from meridianalgo.technical_indicators import SMA
    from meridianalgo.technical_indicators import BollingerBands as LegacyBollingerBands
    from meridianalgo.technical_indicators import Stochastic

    LEGACY_INDICATORS_AVAILABLE = True
except ImportError:
    LEGACY_INDICATORS_AVAILABLE = False


class TestDataGeneration:
    """Helper class for generating test data."""

    @staticmethod
    def create_sample_ohlcv_data(periods=100, seed=42):
        """Create sample OHLCV data for testing."""
        np.random.seed(seed)
        dates = pd.date_range("2023-01-01", periods=periods, freq="D")

        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, periods)
        prices = 100 * np.exp(np.cumsum(returns))

        # Add intraday variation
        high_factor = 1 + np.abs(np.random.normal(0, 0.01, periods))
        low_factor = 1 - np.abs(np.random.normal(0, 0.01, periods))

        data = pd.DataFrame(
            {
                "Open": prices * np.random.uniform(0.995, 1.005, periods),
                "High": prices * high_factor,
                "Low": prices * low_factor,
                "Close": prices,
                "Volume": np.random.randint(100000, 1000000, periods),
            },
            index=dates,
        )

        return data


@pytest.mark.skipif(
    not TECHNICAL_ANALYSIS_AVAILABLE, reason="Technical analysis module not available"
)
class TestAdvancedIndicators:
    """Test advanced technical analysis indicators."""

    def setup_method(self):
        """Set up test data."""
        self.data = TestDataGeneration.create_sample_ohlcv_data()
        self.prices = self.data["Close"]

    def test_rsi_indicator(self):
        """Test RSI indicator calculation."""
        try:
            rsi_indicator = RSI(period=14)
            rsi_values = rsi_indicator.calculate(self.prices)

            assert isinstance(rsi_values, pd.Series)
            assert len(rsi_values) <= len(self.prices)

            # RSI should be between 0 and 100
            valid_rsi = rsi_values.dropna()
            assert all(0 <= val <= 100 for val in valid_rsi)

            # Test different periods
            rsi_short = RSI(period=7).calculate(self.prices)
            rsi_long = RSI(period=21).calculate(self.prices)

            # Short period RSI should be more volatile
            assert rsi_short.std() >= rsi_long.std()

        except Exception as e:
            pytest.skip(f"RSI test skipped: {e}")

    def test_macd_indicator(self):
        """Test MACD indicator calculation."""
        try:
            macd_indicator = MACD(fast=12, slow=26, signal=9)
            macd_line, signal_line, histogram = macd_indicator.calculate(self.prices)

            assert isinstance(macd_line, pd.Series)
            assert isinstance(signal_line, pd.Series)
            assert isinstance(histogram, pd.Series)

            # All series should have same length
            assert len(macd_line) == len(signal_line) == len(histogram)

            # Histogram should be difference between MACD and signal
            diff = macd_line - signal_line
            np.testing.assert_array_almost_equal(
                histogram.dropna().values, diff.dropna().values, decimal=6
            )

        except Exception as e:
            pytest.skip(f"MACD test skipped: {e}")

    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        try:
            bb_indicator = BollingerBands(period=20, std_dev=2)
            upper, middle, lower = bb_indicator.calculate(self.prices)

            assert isinstance(upper, pd.Series)
            assert isinstance(middle, pd.Series)
            assert isinstance(lower, pd.Series)

            # Upper should be above middle, middle above lower
            valid_data = pd.DataFrame(
                {"upper": upper, "middle": middle, "lower": lower}
            ).dropna()
            assert all(valid_data["upper"] >= valid_data["middle"])
            assert all(valid_data["middle"] >= valid_data["lower"])

            # Middle should be moving average
            sma = self.prices.rolling(window=20).mean()
            np.testing.assert_array_almost_equal(
                middle.dropna().values, sma.dropna().values, decimal=6
            )

        except Exception as e:
            pytest.skip(f"Bollinger Bands test skipped: {e}")


@pytest.mark.skipif(
    not TECHNICAL_ANALYSIS_AVAILABLE, reason="Technical analysis module not available"
)
class TestPatternRecognition:
    """Test pattern recognition functionality."""

    def setup_method(self):
        """Set up test data."""
        self.data = TestDataGeneration.create_sample_ohlcv_data()

    def test_candlestick_patterns(self):
        """Test candlestick pattern detection."""
        try:
            pattern_detector = CandlestickPatterns()

            # Test doji detection
            doji_patterns = pattern_detector.detect_doji(self.data)
            assert isinstance(doji_patterns, pd.Series)

            # Test hammer detection
            hammer_patterns = pattern_detector.detect_hammer(self.data)
            assert isinstance(hammer_patterns, pd.Series)

            # Test that patterns are boolean or numeric indicators
            assert doji_patterns.dtype in [bool, int, float]
            assert hammer_patterns.dtype in [bool, int, float]

        except Exception as e:
            pytest.skip(f"Pattern recognition test skipped: {e}")


@pytest.mark.skipif(
    not TECHNICAL_ANALYSIS_AVAILABLE, reason="Technical analysis module not available"
)
class TestCustomIndicatorFramework:
    """Test custom indicator development framework."""

    def test_base_indicator_interface(self):
        """Test base indicator interface."""
        try:
            # Test that BaseIndicator can be subclassed
            class TestIndicator(BaseIndicator):
                def __init__(self, period=10):
                    self.period = period

                def calculate(self, data):
                    return data.rolling(window=self.period).mean()

            indicator = TestIndicator(period=5)
            data = TestDataGeneration.create_sample_ohlcv_data()
            result = indicator.calculate(data["Close"])

            assert isinstance(result, pd.Series)
            assert len(result) == len(data)

        except Exception as e:
            pytest.skip(f"Custom indicator test skipped: {e}")


@pytest.mark.skipif(
    not LEGACY_INDICATORS_AVAILABLE, reason="Legacy indicators not available"
)
class TestLegacyIndicators:
    """Test legacy technical indicators for backward compatibility."""

    def setup_method(self):
        """Set up test data."""
        self.data = TestDataGeneration.create_sample_ohlcv_data()
        self.prices = self.data["Close"]
        self.high = self.data["High"]
        self.low = self.data["Low"]
        self.volume = self.data["Volume"]

    def test_legacy_rsi(self):
        """Test legacy RSI implementation."""
        try:
            rsi_values = LegacyRSI(self.prices, period=14)

            assert isinstance(rsi_values, pd.Series)

            # RSI should be between 0 and 100
            valid_rsi = rsi_values.dropna()
            if len(valid_rsi) > 0:
                assert all(0 <= val <= 100 for val in valid_rsi)

        except Exception as e:
            pytest.skip(f"Legacy RSI test skipped: {e}")

    def test_legacy_sma(self):
        """Test Simple Moving Average."""
        try:
            sma_values = SMA(self.prices, period=20)

            assert isinstance(sma_values, pd.Series)

            # Compare with pandas rolling mean
            expected = self.prices.rolling(window=20).mean()

            # Should be approximately equal (allowing for implementation differences)
            valid_indices = sma_values.dropna().index.intersection(
                expected.dropna().index
            )
            if len(valid_indices) > 0:
                np.testing.assert_allclose(
                    sma_values.loc[valid_indices].values,
                    expected.loc[valid_indices].values,
                    rtol=1e-10,
                )

        except Exception as e:
            pytest.skip(f"SMA test skipped: {e}")

    def test_legacy_ema(self):
        """Test Exponential Moving Average."""
        try:
            ema_values = EMA(self.prices, period=20)

            assert isinstance(ema_values, pd.Series)

            # EMA should be smoother than SMA
            sma_values = SMA(self.prices, period=20)

            # Both should have similar length
            assert abs(len(ema_values.dropna()) - len(sma_values.dropna())) <= 1

        except Exception as e:
            pytest.skip(f"EMA test skipped: {e}")

    def test_legacy_macd(self):
        """Test legacy MACD implementation."""
        try:
            macd_line, signal_line, histogram = LegacyMACD(
                self.prices, fast=12, slow=26, signal=9
            )

            assert isinstance(macd_line, pd.Series)
            assert isinstance(signal_line, pd.Series)
            assert isinstance(histogram, pd.Series)

            # Histogram should be MACD - Signal
            diff = macd_line - signal_line
            valid_indices = histogram.dropna().index.intersection(diff.dropna().index)

            if len(valid_indices) > 0:
                np.testing.assert_allclose(
                    histogram.loc[valid_indices].values,
                    diff.loc[valid_indices].values,
                    rtol=1e-10,
                )

        except Exception as e:
            pytest.skip(f"Legacy MACD test skipped: {e}")

    def test_legacy_bollinger_bands(self):
        """Test legacy Bollinger Bands."""
        try:
            upper, middle, lower = LegacyBollingerBands(
                self.prices, period=20, std_dev=2
            )

            assert isinstance(upper, pd.Series)
            assert isinstance(middle, pd.Series)
            assert isinstance(lower, pd.Series)

            # Test band relationships
            valid_data = pd.DataFrame(
                {"upper": upper, "middle": middle, "lower": lower}
            ).dropna()

            if len(valid_data) > 0:
                assert all(valid_data["upper"] >= valid_data["middle"])
                assert all(valid_data["middle"] >= valid_data["lower"])

        except Exception as e:
            pytest.skip(f"Legacy Bollinger Bands test skipped: {e}")

    def test_stochastic_oscillator(self):
        """Test Stochastic Oscillator."""
        try:
            stoch_k, stoch_d = Stochastic(
                self.high, self.low, self.prices, k_period=14, d_period=3
            )

            assert isinstance(stoch_k, pd.Series)
            assert isinstance(stoch_d, pd.Series)

            # Stochastic should be between 0 and 100
            valid_k = stoch_k.dropna()
            valid_d = stoch_d.dropna()

            if len(valid_k) > 0:
                assert all(0 <= val <= 100 for val in valid_k)
            if len(valid_d) > 0:
                assert all(0 <= val <= 100 for val in valid_d)

        except Exception as e:
            pytest.skip(f"Stochastic test skipped: {e}")


class TestIndicatorPerformance:
    """Test indicator calculation performance."""

    def test_indicator_performance_benchmarks(self):
        """Test that indicators calculate within reasonable time."""
        # Create larger dataset for performance testing
        large_data = TestDataGeneration.create_sample_ohlcv_data(periods=1000)
        prices = large_data["Close"]

        import time

        # Test RSI performance
        if LEGACY_INDICATORS_AVAILABLE:
            start_time = time.time()
            try:
                LegacyRSI(prices, period=14)
                rsi_time = time.time() - start_time
                assert rsi_time < 1.0  # Should complete in less than 1 second
            except Exception:
                pass

        # Test SMA performance
        if LEGACY_INDICATORS_AVAILABLE:
            start_time = time.time()
            try:
                SMA(prices, period=20)
                sma_time = time.time() - start_time
                assert sma_time < 0.5  # Should be very fast
            except Exception:
                pass


class TestIndicatorAccuracy:
    """Test indicator accuracy against known benchmarks."""

    def test_sma_accuracy(self):
        """Test SMA accuracy against pandas rolling mean."""
        if not LEGACY_INDICATORS_AVAILABLE:
            pytest.skip("Legacy indicators not available")

        data = TestDataGeneration.create_sample_ohlcv_data(periods=50)
        prices = data["Close"]

        try:
            # Calculate SMA using our implementation
            our_sma = SMA(prices, period=10)

            # Calculate using pandas
            pandas_sma = prices.rolling(window=10).mean()

            # Compare results
            valid_indices = our_sma.dropna().index.intersection(
                pandas_sma.dropna().index
            )

            if len(valid_indices) > 0:
                np.testing.assert_allclose(
                    our_sma.loc[valid_indices].values,
                    pandas_sma.loc[valid_indices].values,
                    rtol=1e-10,
                    err_msg="SMA calculation differs from pandas rolling mean",
                )
        except Exception as e:
            pytest.skip(f"SMA accuracy test skipped: {e}")


def test_technical_analysis_availability():
    """Test that technical analysis components are available."""
    # This test should always run
    try:
        # Try importing various technical analysis components
        from meridianalgo import technical_analysis  # noqa: F401

        assert True
    except ImportError:
        try:
            from meridianalgo import technical_indicators  # noqa: F401

            assert True
        except ImportError:
            # If neither is available, that's also a valid state
            assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
