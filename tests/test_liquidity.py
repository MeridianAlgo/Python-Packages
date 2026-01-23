"""
Tests for meridianalgo.liquidity module
"""

import numpy as np
import pandas as pd
import pytest


class TestOrderBook:
    """Tests for OrderBook class."""

    @pytest.fixture
    def order_book(self):
        """Create sample order book."""
        from meridianalgo.liquidity import OrderBook

        ob = OrderBook()
        ob.bids = [(100.00, 500), (99.95, 1000), (99.90, 1500)]
        ob.asks = [(100.05, 600), (100.10, 900), (100.15, 1200)]
        return ob

    def test_best_bid(self, order_book):
        """Test best bid."""
        assert order_book.best_bid[0] == 100.00
        assert order_book.best_bid[1] == 500

    def test_best_ask(self, order_book):
        """Test best ask."""
        assert order_book.best_ask[0] == 100.05
        assert order_book.best_ask[1] == 600

    def test_mid_price(self, order_book):
        """Test mid price calculation."""
        expected = (100.00 + 100.05) / 2
        assert abs(order_book.mid_price - expected) < 0.001

    def test_spread(self, order_book):
        """Test spread calculation."""
        assert abs(order_book.spread - 0.05) < 1e-10

    def test_spread_bps(self, order_book):
        """Test spread in basis points."""
        assert order_book.spread_bps > 0

    def test_microprice(self, order_book):
        """Test microprice calculation."""
        assert order_book.microprice > 0

    def test_depth(self, order_book):
        """Test depth calculation."""
        depth = order_book.depth(3)
        assert "bid_depth" in depth
        assert "ask_depth" in depth
        assert depth["bid_depth"] > 0
        assert depth["ask_depth"] > 0

    def test_price_impact(self, order_book):
        """Test price impact estimation."""
        impact = order_book.price_impact(1000, "buy")
        assert impact >= 0


class TestMarketImpact:
    """Tests for MarketImpact class."""

    @pytest.fixture
    def market_impact(self):
        """Create MarketImpact instance."""
        from meridianalgo.liquidity import MarketImpact

        return MarketImpact(daily_volume=1000000, volatility=0.02, spread_bps=5.0)

    def test_square_root_law(self, market_impact):
        """Test square root law impact."""
        result = market_impact.square_root_law(10000)
        assert result > 0

    def test_linear_impact(self, market_impact):
        """Test linear impact."""
        result = market_impact.linear_impact(10000)
        assert result > 0

    def test_estimate_total_cost(self, market_impact):
        """Test total cost estimation."""
        costs = market_impact.estimate_total_cost(10000, 100)
        assert "total_cost_dollars" in costs
        assert "impact_cost_bps" in costs
        assert costs["total_cost_dollars"] > 0


class TestVPIN:
    """Tests for VPIN class."""

    @pytest.fixture
    def trades(self):
        """Generate sample trades."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "price": 100 + np.cumsum(np.random.randn(500) * 0.01),
                "size": np.random.randint(10, 100, 500),
                "side": np.random.choice(["buy", "sell"], 500),
            }
        )

    @pytest.fixture
    def vpin(self, trades):
        """Create VPIN instance."""
        from meridianalgo.liquidity import VPIN

        return VPIN(trades)

    def test_current_vpin(self, vpin):
        """Test current VPIN is between 0 and 1."""
        result = vpin.current_vpin()
        assert 0 <= result <= 1

    def test_average_vpin(self, vpin):
        """Test average VPIN."""
        result = vpin.average_vpin()
        assert 0 <= result <= 1

    def test_toxicity_regime(self, vpin):
        """Test toxicity regime classification."""
        result = vpin.toxicity_regime()
        assert result in ["low", "normal", "elevated", "high"]

    def test_summary(self, vpin):
        """Test summary dictionary."""
        result = vpin.summary()
        assert isinstance(result, dict)
        assert "current_vpin" in result


class TestVolumeProfile:
    """Tests for VolumeProfile class."""

    @pytest.fixture
    def data(self):
        """Generate sample data."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "price": 100 + np.random.randn(500) * 2,
                "volume": np.random.randint(100, 1000, 500),
            }
        )

    @pytest.fixture
    def volume_profile(self, data):
        """Create VolumeProfile instance."""
        from meridianalgo.liquidity import VolumeProfile

        return VolumeProfile(data)

    def test_vwap(self, volume_profile):
        """Test VWAP calculation."""
        result = volume_profile.vwap()
        assert 90 < result < 110

    def test_point_of_control(self, volume_profile):
        """Test POC calculation."""
        result = volume_profile.point_of_control()
        assert 90 < result < 110

    def test_value_area(self, volume_profile):
        """Test value area calculation."""
        low, high = volume_profile.value_area()
        assert low < high
