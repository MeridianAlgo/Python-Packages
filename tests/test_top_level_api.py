"""
Tests for top-level meridianalgo API.

Verifies that every symbol advertised in __all__ and README is importable,
correctly typed, and callable with minimal inputs.
"""

import os

import numpy as np
import pandas as pd
import pytest

os.environ["MERIDIANALGO_QUIET"] = "1"


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def returns_df() -> pd.DataFrame:
    np.random.seed(0)
    return pd.DataFrame(
        np.random.randn(252, 4) * 0.01,
        columns=["AAPL", "MSFT", "GOOGL", "AMZN"],
    )


@pytest.fixture()
def returns_series(returns_df: pd.DataFrame) -> pd.Series:
    return returns_df["AAPL"]


# ============================================================================
# Import tests
# ============================================================================


class TestTopLevelImports:
    def test_version(self) -> None:
        import meridianalgo as ma

        assert ma.__version__ == "7.0.0"

    def test_no_stdout_on_import(self, capsys: pytest.CaptureFixture) -> None:
        import importlib
        import sys

        os.environ["MERIDIANALGO_QUIET"] = "1"
        if "meridianalgo" in sys.modules:
            importlib.reload(sys.modules["meridianalgo"])
        captured = capsys.readouterr()
        assert "INITIALIZED" not in captured.out
        assert "Institutional" not in captured.out

    def test_module_registry_status(self) -> None:
        import meridianalgo as ma

        status = ma.ModuleRegistry.status()
        assert isinstance(status, dict)
        assert "core" in status
        assert status["core"] is True

    def test_core_exports(self) -> None:
        from meridianalgo import (
            PortfolioOptimizer,
            StatisticalArbitrage,
            TimeSeriesAnalyzer,
            calculate_macd,
            calculate_metrics,
            calculate_returns,
            calculate_rsi,
            get_market_data,
        )

        assert all(
            x is not None
            for x in [
                PortfolioOptimizer,
                StatisticalArbitrage,
                TimeSeriesAnalyzer,
                calculate_macd,
                calculate_metrics,
                calculate_returns,
                calculate_rsi,
                get_market_data,
            ]
        )

    def test_financial_function_exports(self) -> None:
        from meridianalgo import (
            calculate_bollinger_bands,
            calculate_calmar_ratio,
            calculate_expected_shortfall,
            calculate_max_drawdown,
            calculate_sharpe_ratio,
            calculate_sortino_ratio,
        )

        assert all(
            callable(f)
            for f in [
                calculate_bollinger_bands,
                calculate_calmar_ratio,
                calculate_expected_shortfall,
                calculate_max_drawdown,
                calculate_sharpe_ratio,
                calculate_sortino_ratio,
            ]
        )

    def test_portfolio_exports(self) -> None:
        from meridianalgo import (
            BlackLitterman,
            HierarchicalRiskParity,
            KellyCriterion,
            MeanVariance,
            RiskParity,
        )

        assert all(x is not None for x in [BlackLitterman, HierarchicalRiskParity, KellyCriterion, MeanVariance, RiskParity])

    def test_risk_exports_and_aliases(self) -> None:
        from meridianalgo import (
            CVaRCalculator,
            RiskAnalyzer,
            RiskBudgeting,
            RiskMetrics,
            StressTesting,
            VaRCalculator,
        )

        assert RiskMetrics is RiskAnalyzer

    def test_ml_exports_and_aliases(self) -> None:
        from meridianalgo import (
            LSTMPredictor,
            ModelSelector,
            ModelTrainer,
            ModelValidator,
            TimeSeriesCV,
            WalkForwardOptimizer,
            WalkForwardValidator,
        )

        assert ModelValidator is WalkForwardValidator

    def test_derivatives_exports(self) -> None:
        from meridianalgo import (
            BlackScholes,
            GreeksCalculator,
            ImpliedVolatility,
            MonteCarloPricer,
            OptionChain,
        )

        assert all(x is not None for x in [BlackScholes, GreeksCalculator, ImpliedVolatility, MonteCarloPricer, OptionChain])

    def test_execution_exports(self) -> None:
        from meridianalgo import POV, TWAP, VWAP, ImplementationShortfall

        assert all(x is not None for x in [POV, TWAP, VWAP, ImplementationShortfall])

    def test_strategy_exports(self) -> None:
        from meridianalgo import (
            Backtest,
            Backtester,
            BacktestEngine,
            BollingerBandsStrategy,
            MACDCrossover,
            MomentumStrategy,
            PairsTrading,
            RSIMeanReversion,
            Strategy,
        )

        assert Backtester is BacktestEngine

    def test_fixed_income_exports(self) -> None:
        from meridianalgo import BondPricer, CreditSpreadAnalyzer, YieldCurve

        assert all(x is not None for x in [BondPricer, CreditSpreadAnalyzer, YieldCurve])


# ============================================================================
# Functional tests
# ============================================================================


class TestFinancialFunctions:
    def test_calculate_sharpe_ratio(self, returns_series: pd.Series) -> None:
        from meridianalgo import calculate_sharpe_ratio

        result = calculate_sharpe_ratio(returns_series)
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_calculate_sortino_ratio(self, returns_series: pd.Series) -> None:
        from meridianalgo import calculate_sortino_ratio

        result = calculate_sortino_ratio(returns_series)
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_calculate_calmar_ratio(self, returns_series: pd.Series) -> None:
        from meridianalgo import calculate_calmar_ratio

        result = calculate_calmar_ratio(returns_series)
        assert isinstance(result, float)

    def test_calculate_max_drawdown(self, returns_series: pd.Series) -> None:
        from meridianalgo import calculate_max_drawdown

        result = calculate_max_drawdown(returns_series)
        assert isinstance(result, float)
        assert result <= 0.0  # drawdown is non-positive

    def test_calculate_max_drawdown_monotone_increase(self) -> None:
        from meridianalgo import calculate_max_drawdown

        returns = pd.Series([0.01] * 100)
        result = calculate_max_drawdown(returns)
        assert result == 0.0

    def test_calculate_expected_shortfall(self, returns_series: pd.Series) -> None:
        from meridianalgo import calculate_expected_shortfall

        result = calculate_expected_shortfall(returns_series)
        assert isinstance(result, float)
        assert result <= 0.0

    def test_calculate_bollinger_bands(self, returns_series: pd.Series) -> None:
        from meridianalgo import calculate_bollinger_bands

        prices = (1 + returns_series).cumprod() * 100
        upper, middle, lower = calculate_bollinger_bands(prices, period=20)
        assert len(upper) == len(prices)
        assert len(middle) == len(prices)
        assert len(lower) == len(prices)
        valid = upper.dropna()
        assert (valid >= middle.dropna()).all()
        assert (middle.dropna() >= lower.dropna()).all()

    def test_calculate_rsi_range(self, returns_series: pd.Series) -> None:
        from meridianalgo import calculate_rsi

        prices = (1 + returns_series).cumprod() * 100
        rsi = calculate_rsi(prices)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()


class TestDerivativesAPI:
    def test_blackscholes_call(self) -> None:
        from meridianalgo import BlackScholes

        result = BlackScholes(S=100, K=105, T=0.25, r=0.05, sigma=0.2, option_type="call")
        assert isinstance(result, dict)
        assert "price" in result
        assert result["price"] > 0

    def test_blackscholes_put(self) -> None:
        from meridianalgo import BlackScholes

        result = BlackScholes(S=100, K=95, T=0.25, r=0.05, sigma=0.2, option_type="put")
        assert isinstance(result, dict)
        assert result["price"] > 0

    def test_put_call_parity(self) -> None:
        from meridianalgo import BlackScholes

        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        call = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma, option_type="call")["price"]
        put = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma, option_type="put")["price"]
        # Put-call parity: C - P = S - K*exp(-rT)
        lhs = call - put
        rhs = S - K * np.exp(-r * T)
        assert abs(lhs - rhs) < 0.01
