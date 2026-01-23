"""
Tests for meridianalgo.signals module
"""

import numpy as np
import pandas as pd
import pytest


class TestIndicators:
    """Tests for technical indicators."""

    @pytest.fixture
    def price_data(self):
        """Generate sample price data."""
        np.random.seed(42)
        n = 100
        close = pd.Series(100 + np.cumsum(np.random.randn(n) * 1))
        high = close + np.random.uniform(0.5, 2, n)
        low = close - np.random.uniform(0.5, 2, n)
        volume = pd.Series(np.random.randint(10000, 100000, n))
        return {"close": close, "high": high, "low": low, "volume": volume}

    def test_sma(self, price_data):
        """Test SMA calculation."""
        from meridianalgo.signals import SMA

        result = SMA(price_data["close"], 20)
        assert len(result) == len(price_data["close"])
        assert result.iloc[-1] > 0

    def test_ema(self, price_data):
        """Test EMA calculation."""
        from meridianalgo.signals import EMA

        result = EMA(price_data["close"], 20)
        assert len(result) == len(price_data["close"])
        assert result.iloc[-1] > 0

    def test_rsi(self, price_data):
        """Test RSI is between 0 and 100."""
        from meridianalgo.signals import RSI

        result = RSI(price_data["close"], 14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_macd(self, price_data):
        """Test MACD returns three series."""
        from meridianalgo.signals import MACD

        macd, signal, hist = MACD(price_data["close"])
        assert len(macd) == len(price_data["close"])
        assert len(signal) == len(price_data["close"])
        assert len(hist) == len(price_data["close"])

    def test_bollinger_bands(self, price_data):
        """Test Bollinger Bands ordering."""
        from meridianalgo.signals import BollingerBands

        upper, middle, lower = BollingerBands(price_data["close"])
        # After warmup period, upper > middle > lower
        assert upper.iloc[-1] > middle.iloc[-1] > lower.iloc[-1]

    def test_atr(self, price_data):
        """Test ATR is positive."""
        from meridianalgo.signals import ATR

        result = ATR(price_data["high"], price_data["low"], price_data["close"], 14)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_stochastic(self, price_data):
        """Test Stochastic oscillator."""
        from meridianalgo.signals import Stochastic

        k, d = Stochastic(price_data["high"], price_data["low"], price_data["close"])
        valid_k = k.dropna()
        assert (valid_k >= 0).all() and (valid_k <= 100).all()

    def test_adx(self, price_data):
        """Test ADX calculation."""
        from meridianalgo.signals import ADX

        adx, plus_di, minus_di = ADX(
            price_data["high"], price_data["low"], price_data["close"]
        )
        assert len(adx) == len(price_data["close"])

    def test_obv(self, price_data):
        """Test OBV calculation."""
        from meridianalgo.signals import OBV

        result = OBV(price_data["close"], price_data["volume"])
        assert len(result) == len(price_data["close"])


class TestSignalGenerator:
    """Tests for SignalGenerator class."""

    @pytest.fixture
    def data(self):
        """Generate sample data with indicators."""
        np.random.seed(42)
        close = pd.Series(100 + np.cumsum(np.random.randn(100) * 1))
        from meridianalgo.signals import RSI

        rsi = RSI(close, 14)
        return pd.DataFrame({"close": close, "rsi": rsi})

    @pytest.fixture
    def generator(self, data):
        """Create SignalGenerator instance."""
        from meridianalgo.signals import SignalGenerator

        gen = SignalGenerator(data)
        gen.add_rule("rsi_oversold", lambda d: d["rsi"] < 30, weight=1.0)
        gen.add_rule(
            "rsi_overbought", lambda d: d["rsi"] > 70, weight=1.0, signal_type="short"
        )
        return gen

    def test_generate_signals(self, generator):
        """Test signal generation."""
        signals = generator.generate()
        assert "signal" in signals.columns
        assert "long_score" in signals.columns
        assert "short_score" in signals.columns

    def test_signal_values(self, generator):
        """Test signal values are valid."""
        signals = generator.generate()
        assert set(signals["signal"].unique()).issubset({-1, 0, 1})


class TestTechnicalAnalyzer:
    """Tests for TechnicalAnalyzer class."""

    @pytest.fixture
    def price_data(self):
        """Generate sample price data."""
        np.random.seed(42)
        n = 100
        close = pd.Series(100 + np.cumsum(np.random.randn(n) * 1))
        high = close + np.random.uniform(0.5, 2, n)
        low = close - np.random.uniform(0.5, 2, n)
        volume = pd.Series(np.random.randint(10000, 100000, n))
        return {"close": close, "high": high, "low": low, "volume": volume}

    @pytest.fixture
    def analyzer(self, price_data):
        """Create TechnicalAnalyzer instance."""
        from meridianalgo.signals import TechnicalAnalyzer

        return TechnicalAnalyzer(
            price_data["high"],
            price_data["low"],
            price_data["close"],
            price_data["volume"],
        )

    def test_calculate_all(self, analyzer):
        """Test calculating all indicators."""
        result = analyzer.calculate_all()
        assert isinstance(result, pd.DataFrame)
        assert "rsi" in result.columns
        assert "macd" in result.columns

    def test_summary(self, analyzer):
        """Test summary generation."""
        result = analyzer.summary()
        assert isinstance(result, dict)
        assert "rsi" in result
        assert "trend" in result
