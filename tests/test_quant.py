"""
Comprehensive test suite for the quant module.

Tests all quantitative algorithms including:
- Market microstructure
- Statistical arbitrage
- Execution algorithms
- High-frequency trading
- Factor models
- Regime detection
"""

import os

# Import quant modules
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meridianalgo.quant.execution_algorithms import (
    POV,
    TWAP,
    VWAP,
    ImplementationShortfall,
)
from meridianalgo.quant.factor_models import (
    APTModel,
    CustomFactorModel,
    FactorRiskDecomposition,
    FamaFrenchModel,
)
from meridianalgo.quant.high_frequency import (
    LatencyArbitrage,
    MarketMaking,
    MicropriceEstimator,
    OrderBook,
    OrderBookLevel,
)
from meridianalgo.quant.market_microstructure import (
    MarketImpactModel,
    OrderFlowImbalance,
    RealizedVolatility,
    calculate_microprice,
)
from meridianalgo.quant.regime_detection import (
    HiddenMarkovModel,
    MarketStateClassifier,
    RegimeSwitchingModel,
    StructuralBreakDetection,
    VolatilityRegimeDetector,
)
from meridianalgo.quant.statistical_arbitrage import (
    CointegrationAnalyzer,
    MeanReversionTester,
    OrnsteinUhlenbeck,
    PairsTrading,
)


class TestMarketMicrostructure:
    """Test market microstructure algorithms."""

    def test_order_flow_imbalance(self):
        """Test OFI calculation."""
        ofi = OrderFlowImbalance()

        bid_volume = np.array([100, 150, 120, 180])
        ask_volume = np.array([90, 100, 110, 150])
        bid_price = np.array([100.0, 100.05, 100.10, 100.15])
        ask_price = np.array([100.05, 100.10, 100.15, 100.20])

        result = ofi.calculate_ofi(bid_volume, ask_volume, bid_price, ask_price)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(bid_volume)

    def test_volume_imbalance_ratio(self):
        """Test VIR calculation."""
        ofi = OrderFlowImbalance()

        bid_volume = np.array([100, 150, 120])
        ask_volume = np.array([90, 100, 110])

        result = ofi.volume_imbalance_ratio(bid_volume, ask_volume)

        assert isinstance(result, np.ndarray)
        assert all(-1 <= x <= 1 for x in result)

    def test_realized_volatility(self):
        """Test realized volatility calculation."""
        # Generate sample price data
        dates = pd.date_range("2024-01-01", periods=100, freq="5min")
        prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.1), index=dates)

        rv = RealizedVolatility.rv_5min(prices)

        assert isinstance(rv, float)
        assert rv > 0

    def test_market_impact(self):
        """Test market impact models."""
        model = MarketImpactModel()

        impact = model.almgren_chriss_temporary_impact(
            volume=1000, daily_volume=100000, volatility=0.02
        )

        assert isinstance(impact, float)

    def test_microprice(self):
        """Test microprice calculation."""
        microprice = calculate_microprice(
            bid_price=100.0, ask_price=100.10, bid_volume=500, ask_volume=400
        )

        assert 100.0 <= microprice <= 100.10


class TestStatisticalArbitrage:
    """Test statistical arbitrage algorithms."""

    def test_pairs_trading(self):
        """Test pairs trading strategy."""
        # Generate cointegrated price series
        np.random.seed(42)
        t = np.arange(100)
        series1 = pd.Series(100 + t * 0.1 + np.random.randn(100) * 0.5)
        series2 = pd.Series(50 + t * 0.05 + np.random.randn(100) * 0.3)

        pt = PairsTrading()
        hedge_ratio = pt.calculate_hedge_ratio(series1, series2)

        assert isinstance(hedge_ratio, float)
        assert hedge_ratio > 0

        spread = pt.calculate_spread(series1, series2)
        assert isinstance(spread, pd.Series)
        assert len(spread) == len(series1)

    def test_cointegration_analyzer(self):
        """Test cointegration analysis."""
        # Generate cointegrated series
        np.random.seed(42)
        series1 = pd.Series(np.cumsum(np.random.randn(100)))
        series2 = series1 + np.random.randn(100) * 0.1

        analyzer = CointegrationAnalyzer()
        result = analyzer.engle_granger_test(series1, series2)

        assert "test_statistic" in result
        assert "pvalue" in result
        assert "is_cointegrated" in result

    def test_ornstein_uhlenbeck(self):
        """Test OU process."""
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5))

        ou = OrnsteinUhlenbeck()
        params = ou.fit(prices)

        assert "theta" in params
        assert "mu" in params
        assert "sigma" in params
        assert params["theta"] >= 0

    def test_mean_reversion_tester(self):
        """Test mean reversion tests."""
        np.random.seed(42)
        series = pd.Series(np.random.randn(100))

        tester = MeanReversionTester()
        adf_result = tester.adf_test(series)

        assert "test_statistic" in adf_result
        assert "pvalue" in adf_result
        assert "is_stationary" in adf_result


class TestExecutionAlgorithms:
    """Test execution algorithms."""

    def test_vwap(self):
        """Test VWAP algorithm."""
        vwap = VWAP(
            total_quantity=10000,
            start_time="2024-01-01 09:30:00",
            end_time="2024-01-01 16:00:00",
        )

        # Create historical volume profile
        dates = pd.date_range("2024-01-01 09:30", "2024-01-01 16:00", freq="5min")
        historical_volume = pd.Series(
            np.random.randint(100, 1000, len(dates)), index=dates
        )

        schedule = vwap.calculate_schedule(historical_volume)

        assert isinstance(schedule, pd.Series)
        assert np.isclose(schedule.sum(), 10000, rtol=0.01)

    def test_twap(self):
        """Test TWAP algorithm."""
        twap = TWAP(total_quantity=10000, duration_minutes=60)

        schedule = twap.get_schedule()

        assert isinstance(schedule, np.ndarray)
        assert np.isclose(schedule.sum(), 10000, rtol=0.01)

    def test_pov(self):
        """Test POV algorithm."""
        pov = POV(total_quantity=10000, target_pov=0.10)

        execution = pov.execute(
            market_volume=1000, market_price=100.0, time_remaining_pct=0.5
        )

        assert "quantity" in execution
        assert "realized_pov" in execution
        assert execution["quantity"] > 0

    def test_implementation_shortfall(self):
        """Test Implementation Shortfall algorithm."""
        is_algo = ImplementationShortfall(
            total_quantity=10000, total_time=1.0, volatility=0.02, risk_aversion=1e-6
        )

        trajectory = is_algo.calculate_optimal_trajectory(n_intervals=10)

        assert isinstance(trajectory, pd.DataFrame)
        assert "holdings" in trajectory.columns
        assert "trades" in trajectory.columns


class TestHighFrequency:
    """Test high-frequency trading algorithms."""

    def test_market_making(self):
        """Test market making strategy."""
        mm = MarketMaking(target_spread_bps=5.0, max_inventory=1000)

        bid_price, ask_price = mm.calculate_quotes(
            mid_price=100.0, volatility=0.02, order_flow_imbalance=0.1
        )

        assert bid_price < 100.0
        assert ask_price > 100.0
        assert ask_price > bid_price

    def test_latency_arbitrage(self):
        """Test latency arbitrage detection."""
        arb = LatencyArbitrage(latency_threshold_us=100.0, min_profit_bps=1.0)

        opportunity = arb.detect_opportunity(
            venue1_price=100.0,
            venue1_time=1000.0,
            venue2_price=100.05,
            venue2_time=1050.0,
        )

        if opportunity:
            assert "profit_bps" in opportunity
            assert "latency_us" in opportunity

    def test_order_book(self):
        """Test order book operations."""
        ob = OrderBook()
        ob.bids = [OrderBookLevel(100.0, 500, 1000.0)]
        ob.asks = [OrderBookLevel(100.05, 400, 1000.0)]

        mid_price = ob.get_mid_price()
        spread = ob.get_spread()

        assert abs(mid_price - 100.025) < 1e-10
        assert abs(spread - 0.05) < 1e-10

    def test_microprice_estimator(self):
        """Test microprice estimation."""
        microprice = MicropriceEstimator.simple_microprice(
            bid_price=100.0, ask_price=100.10, bid_volume=500, ask_volume=400
        )

        assert 100.0 <= microprice <= 100.10


class TestFactorModels:
    """Test factor models."""

    def test_fama_french_model(self):
        """Test Fama-French model."""
        np.random.seed(42)

        # Generate sample data
        returns = pd.Series(np.random.randn(100) * 0.01)
        factor_data = pd.DataFrame(
            {
                "MKT": np.random.randn(100) * 0.015,
                "SMB": np.random.randn(100) * 0.01,
                "HML": np.random.randn(100) * 0.01,
            }
        )

        ff = FamaFrenchModel(model_type="three_factor")
        results = ff.fit(returns, factor_data)

        assert "alpha" in results
        assert "coefficients" in results
        assert "r_squared" in results
        assert all(
            factor in results["coefficients"] for factor in ["Market", "Size", "Value"]
        )

    def test_apt_model(self):
        """Test APT model."""
        np.random.seed(42)

        # Generate return matrix
        return_matrix = pd.DataFrame(
            np.random.randn(100, 10) * 0.01, columns=[f"Asset{i}" for i in range(10)]
        )

        apt = APTModel(n_factors=3)
        results = apt.extract_factors(return_matrix)

        assert "factors" in results
        assert "loadings" in results
        assert "explained_variance" in results
        assert len(results["factors"].columns) == 3

    def test_custom_factor_model(self):
        """Test custom factor model."""
        np.random.seed(42)

        returns = pd.DataFrame(
            np.random.randn(100, 5) * 0.01, columns=["A", "B", "C", "D", "E"]
        )

        factors = pd.DataFrame(
            {
                "Value": np.random.randn(100) * 0.01,
                "Momentum": np.random.randn(100) * 0.01,
            }
        )

        cfm = CustomFactorModel(factor_names=["Value", "Momentum"])
        results = cfm.fit(returns, factors)

        assert len(results) == 5  # One result per asset
        assert all("alpha" in result for result in results.values())

    def test_factor_risk_decomposition(self):
        """Test factor risk decomposition."""
        weights = np.array([0.2, 0.3, 0.5])
        factor_exposures = np.array([[0.8, 0.2], [1.0, -0.1], [0.9, 0.3]])
        factor_covariance = np.array([[0.04, 0.01], [0.01, 0.02]])
        specific_variance = np.array([0.01, 0.015, 0.012])

        decomp = FactorRiskDecomposition.decompose_variance(
            weights, factor_exposures, factor_covariance, specific_variance
        )

        assert "total_variance" in decomp
        assert "factor_variance" in decomp
        assert "specific_variance" in decomp


class TestRegimeDetection:
    """Test regime detection algorithms."""

    def test_hidden_markov_model(self):
        """Test HMM for regime detection."""
        np.random.seed(42)

        # Generate returns with regime changes
        returns1 = np.random.normal(0.001, 0.01, 50)
        returns2 = np.random.normal(-0.002, 0.03, 50)
        returns = pd.Series(np.concatenate([returns1, returns2]))

        hmm = HiddenMarkovModel(n_states=2)
        results = hmm.fit(returns, max_iter=20)

        assert "means" in results
        assert "std_devs" in results
        assert "transition_matrix" in results
        assert len(results["means"]) == 2

    def test_regime_switching_model(self):
        """Test Markov regime-switching model."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.01)

        rsm = RegimeSwitchingModel(n_regimes=2)
        results = rsm.fit(returns)

        assert "regime_params" in results
        assert "current_regime" in results
        assert "regime_statistics" in results

    def test_structural_break_detection(self):
        """Test structural break detection."""
        np.random.seed(42)

        # Generate data with structural break
        data1 = np.random.normal(0, 1, 50)
        data2 = np.random.normal(2, 1, 50)
        data = pd.Series(np.concatenate([data1, data2]))

        # Chow test
        sbd = StructuralBreakDetection()
        chow_result = sbd.chow_test(data, breakpoint=50)

        assert "f_statistic" in chow_result
        assert "p_value" in chow_result
        assert "significant" in chow_result

    def test_market_state_classifier(self):
        """Test market state classification."""
        np.random.seed(42)

        prices = pd.Series(100 + np.cumsum(np.random.randn(300) * 0.5))
        returns = prices.pct_change().dropna()

        vol_regime = MarketStateClassifier.classify_volatility_regime(returns)

        assert isinstance(vol_regime, pd.Series)
        assert all(
            state in ["low_volatility", "normal", "high_volatility"]
            for state in vol_regime.dropna().unique()
        )

    def test_volatility_regime_detector(self):
        """Test volatility regime detection."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(200) * 0.01)

        vrd = VolatilityRegimeDetector()
        results = vrd.garch_volatility_regimes(returns, n_regimes=2)

        assert "regimes" in results
        assert "regime_statistics" in results
        assert "current_regime" in results


class TestIntegration:
    """Integration tests combining multiple modules."""

    def test_pairs_trading_with_execution(self):
        """Test pairs trading with execution algorithm."""
        np.random.seed(42)

        # Generate cointegrated series
        series1 = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5))
        series2 = pd.Series(50 + np.cumsum(np.random.randn(100) * 0.3))

        # Pairs trading
        pt = PairsTrading()
        signals = pt.generate_signals(series1, series2)

        # Calculate total quantity to execute
        total_quantity = abs(signals["position"].diff().sum())

        # Use TWAP for execution
        if total_quantity > 0:
            twap = TWAP(total_quantity=total_quantity, duration_minutes=60)
            schedule = twap.get_schedule()

            assert schedule.sum() > 0

    def test_factor_model_with_regime_detection(self):
        """Test factor model combined with regime detection."""
        np.random.seed(42)

        # Generate returns and factors
        returns = pd.Series(np.random.randn(200) * 0.01)
        factors = pd.DataFrame(
            {
                "MKT": np.random.randn(200) * 0.015,
                "SMB": np.random.randn(200) * 0.01,
                "HML": np.random.randn(200) * 0.01,
            }
        )

        # Fit factor model
        ff = FamaFrenchModel()
        ff_results = ff.fit(returns, factors)

        # Detect regimes
        hmm = HiddenMarkovModel(n_states=2)
        hmm_results = hmm.fit(returns, max_iter=10)

        assert "alpha" in ff_results
        assert "means" in hmm_results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
