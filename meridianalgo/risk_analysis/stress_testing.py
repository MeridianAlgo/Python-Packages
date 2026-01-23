"""
Stress Testing Module

This module provides stress testing capabilities for portfolios.
"""

from typing import Dict, List

import numpy as np
import pandas as pd


class StressTester:
    """Portfolio Stress Tester."""

    def __init__(self, portfolio_returns: pd.Series):
        self.portfolio_returns = portfolio_returns

    def historical_stress_test(self, stress_periods: List[str]) -> Dict[str, float]:
        """Perform historical stress test."""
        results = {}
        for period in stress_periods:
            # Simplified stress test - in practice, you'd filter by specific periods
            results[period] = self.portfolio_returns.min()
        return results

    def scenario_stress_test(self, scenarios: Dict[str, float]) -> Dict[str, float]:
        """Perform scenario-based stress test."""
        results = {}
        for scenario_name, shock in scenarios.items():
            shocked_returns = self.portfolio_returns * (1 + shock)
            results[scenario_name] = shocked_returns.min()
        return results


class ScenarioAnalysis:
    """Scenario Analysis for portfolios."""

    def __init__(self, portfolio_returns: pd.Series):
        self.portfolio_returns = portfolio_returns

    def analyze_scenarios(
        self, scenarios: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze multiple scenarios."""
        results = {}
        for scenario_name, scenario_params in scenarios.items():
            # Apply scenario parameters
            shocked_returns = self.portfolio_returns.copy()
            for param, value in scenario_params.items():
                if param == "volatility_multiplier":
                    shocked_returns = shocked_returns * value
                elif param == "return_adjustment":
                    shocked_returns = shocked_returns + value

            results[scenario_name] = {
                "min_return": shocked_returns.min(),
                "max_drawdown": self.calculate_max_drawdown(shocked_returns),
                "var_95": np.percentile(shocked_returns, 5),
            }
        return results

    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdowns = (cumulative - rolling_max) / rolling_max
        return drawdowns.min()


class HistoricalStressTest:
    """Historical stress testing implementation."""

    def __init__(self, portfolio_returns: pd.Series):
        self.portfolio_returns = portfolio_returns

    def test_crisis_periods(self) -> Dict[str, float]:
        """Test portfolio performance during historical crisis periods."""
        # This is a simplified implementation
        # In practice, you'd identify specific crisis periods
        return {
            "2008_financial_crisis": self.portfolio_returns.min(),
            "covid_19_crash": self.portfolio_returns.min(),
            "dot_com_bubble": self.portfolio_returns.min(),
        }
