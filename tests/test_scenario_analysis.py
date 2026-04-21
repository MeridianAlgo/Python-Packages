"""Tests for scenario analysis and stress testing."""

import numpy as np
import pandas as pd
import pytest

from meridianalgo.risk.scenario import (
    HISTORICAL_SCENARIOS,
    CorrelationScenario,
    ScenarioAnalyzer,
)


@pytest.fixture
def simple_portfolio():
    weights = pd.Series({"equity": 0.50, "bonds": 0.30, "commodities": 0.20})
    sensitivities = pd.DataFrame(
        {
            "equity": [1.0, 0.0, 0.1],
            "bonds": [0.0, 1.0, 0.0],
            "commodities": [0.1, 0.0, 1.0],
            "usd": [-0.1, 0.05, -0.2],
        },
        index=["equity", "bonds", "commodities"],
    )
    return ScenarioAnalyzer(weights, sensitivities, portfolio_value=1_000_000)


class TestScenarioAnalyzer:
    def test_apply_scenario_returns_result(self, simple_portfolio):
        result = simple_portfolio.apply_scenario({"equity": -0.30, "bonds": 0.05})
        assert result.scenario_name is not None
        assert isinstance(result.portfolio_return, float)

    def test_crash_scenario_negative_return(self, simple_portfolio):
        result = simple_portfolio.apply_scenario(
            {"equity": -0.50, "bonds": 0.05, "commodities": -0.30},
            scenario_name="crash",
        )
        assert result.portfolio_return < 0

    def test_bond_rally_positive_return(self, simple_portfolio):
        weights = pd.Series({"bonds": 1.0})
        sensitivities = pd.DataFrame({"bonds": [1.0]}, index=["bonds"])
        bond_portfolio = ScenarioAnalyzer(weights, sensitivities, portfolio_value=100_000)
        result = bond_portfolio.apply_scenario({"bonds": 0.10})
        assert result.portfolio_return > 0

    def test_severity_classification_severe(self, simple_portfolio):
        result = simple_portfolio.apply_scenario(
            {"equity": -0.60, "bonds": 0.0, "commodities": -0.40}
        )
        assert result.severity == "severe"

    def test_severity_classification_benign(self, simple_portfolio):
        result = simple_portfolio.apply_scenario({"equity": 0.05, "bonds": 0.02})
        assert result.severity == "benign"

    def test_run_all_historical_returns_dict(self, simple_portfolio):
        results = simple_portfolio.run_all_historical()
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_all_historical_scenarios_have_results(self, simple_portfolio):
        results = simple_portfolio.run_all_historical()
        for name, r in results.items():
            assert isinstance(r.portfolio_return, float)

    def test_gfc_scenario_negative(self, simple_portfolio):
        results = simple_portfolio.run_all_historical()
        gfc = results.get("gfc_2008_2009")
        assert gfc is not None
        assert gfc.portfolio_return < 0

    def test_summary_table_sorted(self, simple_portfolio):
        df = simple_portfolio.summary_table()
        assert isinstance(df, pd.DataFrame)
        returns = df["portfolio_return"].values
        assert all(returns[i] <= returns[i + 1] for i in range(len(returns) - 1))

    def test_custom_scenario(self, simple_portfolio):
        result = simple_portfolio.run_custom_scenario(
            "equity_crash", equity_shock=-0.30, bond_shock=0.05
        )
        assert result.portfolio_return < 0

    def test_reverse_stress_test(self, simple_portfolio):
        target = -0.10
        shock = simple_portfolio.reverse_stress_test(
            target_loss=target, factor="equity"
        )
        assert isinstance(shock, float)

    def test_unknown_factor_raises(self, simple_portfolio):
        with pytest.raises(ValueError):
            simple_portfolio.reverse_stress_test(-0.10, factor="nonexistent_factor")

    def test_pnl_is_return_times_value(self, simple_portfolio):
        result = simple_portfolio.apply_scenario({"equity": -0.30})
        assert abs(result.portfolio_pnl - result.portfolio_return * 1_000_000) < 1e-6

    def test_historical_scenarios_defined(self):
        assert len(HISTORICAL_SCENARIOS) >= 5
        assert "gfc_2008_2009" in HISTORICAL_SCENARIOS
        assert "covid_crash_march_2020" in HISTORICAL_SCENARIOS


class TestCorrelationScenario:
    @pytest.fixture
    def correlation_setup(self):
        assets = ["equity", "bonds", "gold"]
        mean_returns = pd.Series([0.0003, 0.0001, 0.0002], index=assets)
        volatilities = pd.Series([0.012, 0.004, 0.007], index=assets)
        weights = pd.Series([0.60, 0.30, 0.10], index=assets)
        corr = pd.DataFrame(
            [[1.0, -0.30, 0.10], [-0.30, 1.0, 0.05], [0.10, 0.05, 1.0]],
            index=assets,
            columns=assets,
        )
        return CorrelationScenario(mean_returns, corr, volatilities, weights)

    def test_generate_returns_dict(self, correlation_setup):
        result = correlation_setup.generate(n_scenarios=1000)
        assert "portfolio_return" in result
        assert "var_95" in result
        assert "cvar_95" in result

    def test_portfolio_return_shape(self, correlation_setup):
        result = correlation_setup.generate(n_scenarios=1000)
        assert len(result["portfolio_return"]) == 1000

    def test_var_negative(self, correlation_setup):
        result = correlation_setup.generate(n_scenarios=10_000)
        assert result["var_95"] < 0

    def test_cvar_worse_than_var(self, correlation_setup):
        result = correlation_setup.generate(n_scenarios=10_000)
        assert result["cvar_95"] <= result["var_95"]

    def test_stressed_correlation_increases_tail_risk(self, correlation_setup):
        normal = correlation_setup.generate(n_scenarios=20_000, stress_correlation=False)
        stressed = correlation_setup.generate(
            n_scenarios=20_000, stress_correlation=True, stress_factor=0.8
        )
        assert stressed["var_95"] <= normal["var_95"]
