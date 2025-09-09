"""
Risk Management Module for Portfolio Management

This module provides risk management tools for portfolios.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

class RiskManager:
    """Portfolio Risk Manager."""
    
    def __init__(self, portfolio_returns: pd.Series):
        self.portfolio_returns = portfolio_returns
    
    def calculate_var(self, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk."""
        return np.percentile(self.portfolio_returns, (1 - confidence_level) * 100)
    
    def calculate_expected_shortfall(self, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall."""
        var = self.calculate_var(confidence_level)
        return self.portfolio_returns[self.portfolio_returns <= var].mean()

class VaRCalculator:
    """Value at Risk Calculator."""
    
    def __init__(self, returns: pd.Series):
        self.returns = returns
    
    def historical_var(self, confidence_level: float = 0.95) -> float:
        """Calculate historical VaR."""
        return np.percentile(self.returns, (1 - confidence_level) * 100)
    
    def parametric_var(self, confidence_level: float = 0.95) -> float:
        """Calculate parametric VaR."""
        mean = self.returns.mean()
        std = self.returns.std()
        return mean + std * np.percentile(np.random.normal(0, 1, 10000), (1 - confidence_level) * 100)

class StressTester:
    """Portfolio Stress Tester."""
    
    def __init__(self, portfolio_returns: pd.Series):
        self.portfolio_returns = portfolio_returns
    
    def historical_stress_test(self, stress_periods: List[str]) -> Dict[str, float]:
        """Perform historical stress test."""
        results = {}
        for period in stress_periods:
            # Simplified stress test
            results[period] = self.portfolio_returns.min()
        return results

class RiskMetrics:
    """Portfolio Risk Metrics Calculator."""
    
    def __init__(self, returns: pd.Series):
        self.returns = returns
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        return {
            'volatility': self.returns.std() * np.sqrt(252),
            'sharpe_ratio': self.returns.mean() / self.returns.std() * np.sqrt(252),
            'max_drawdown': self.calculate_max_drawdown(),
            'var_95': np.percentile(self.returns, 5)
        }
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + self.returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdowns = (cumulative - rolling_max) / rolling_max
        return drawdowns.min()
