"""
Value at Risk and Expected Shortfall Module

This module provides various VaR and ES calculation methods.
"""

import numpy as np
import pandas as pd


class VaRCalculator:
    """Value at Risk Calculator with multiple methods."""

    def __init__(self, returns: pd.Series):
        self.returns = returns

    def historical_var(self, confidence_level: float = 0.95) -> float:
        """Calculate historical VaR."""
        return np.percentile(self.returns, (1 - confidence_level) * 100)

    def parametric_var(self, confidence_level: float = 0.95) -> float:
        """Calculate parametric VaR assuming normal distribution."""
        mean = self.returns.mean()
        std = self.returns.std()
        z_score = np.percentile(
            np.random.normal(0, 1, 10000), (1 - confidence_level) * 100
        )
        return mean + std * z_score

    def monte_carlo_var(
        self, confidence_level: float = 0.95, n_simulations: int = 10000
    ) -> float:
        """Calculate VaR using Monte Carlo simulation."""
        mean = self.returns.mean()
        std = self.returns.std()
        simulated_returns = np.random.normal(mean, std, n_simulations)
        return np.percentile(simulated_returns, (1 - confidence_level) * 100)


class ExpectedShortfall:
    """Expected Shortfall Calculator."""

    def __init__(self, returns: pd.Series):
        self.returns = returns

    def calculate_es(self, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall."""
        var = np.percentile(self.returns, (1 - confidence_level) * 100)
        return self.returns[self.returns <= var].mean()


class HistoricalVaR(VaRCalculator):
    """Historical VaR implementation."""

    def calculate(self, confidence_level: float = 0.95) -> float:
        """Calculate historical VaR."""
        return self.historical_var(confidence_level)


class ParametricVaR(VaRCalculator):
    """Parametric VaR implementation."""

    def calculate(self, confidence_level: float = 0.95) -> float:
        """Calculate parametric VaR."""
        return self.parametric_var(confidence_level)


class MonteCarloVaR(VaRCalculator):
    """Monte Carlo VaR implementation."""

    def calculate(
        self, confidence_level: float = 0.95, n_simulations: int = 10000
    ) -> float:
        """Calculate Monte Carlo VaR."""
        return self.monte_carlo_var(confidence_level, n_simulations)
