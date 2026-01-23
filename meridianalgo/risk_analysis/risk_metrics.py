"""
Risk Metrics Module

This module provides comprehensive risk metrics for portfolios.
"""

from typing import Dict, List, Union

import numpy as np
import pandas as pd


class RiskMetrics:
    """Comprehensive Risk Metrics Calculator."""

    def __init__(self, returns: pd.Series):
        self.returns = returns

    def calculate_all_metrics(self) -> Dict[str, float]:
        """Calculate all risk metrics."""
        return {
            "volatility": self.calculate_volatility(),
            "sharpe_ratio": self.calculate_sharpe_ratio(),
            "sortino_ratio": self.calculate_sortino_ratio(),
            "calmar_ratio": self.calculate_calmar_ratio(),
            "max_drawdown": self.calculate_max_drawdown(),
            "var_95": self.calculate_var(0.95),
            "var_99": self.calculate_var(0.99),
            "expected_shortfall_95": self.calculate_expected_shortfall(0.95),
            "skewness": self.returns.skew(),
            "kurtosis": self.returns.kurtosis(),
        }

    def calculate_volatility(self) -> float:
        """Calculate annualized volatility."""
        return self.returns.std() * np.sqrt(252)

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        excess_return = self.returns.mean() * 252 - risk_free_rate
        return excess_return / self.calculate_volatility()

    def calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        excess_return = self.returns.mean() * 252 - risk_free_rate
        downside_returns = self.returns[self.returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        return excess_return / downside_volatility if downside_volatility > 0 else 0

    def calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio."""
        annual_return = (1 + self.returns).prod() ** (252 / len(self.returns)) - 1
        max_dd = self.calculate_max_drawdown()
        return annual_return / abs(max_dd) if max_dd != 0 else 0

    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + self.returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdowns = (cumulative - rolling_max) / rolling_max
        return drawdowns.min()

    def calculate_var(self, confidence_level: float) -> float:
        """Calculate Value at Risk."""
        return np.percentile(self.returns, (1 - confidence_level) * 100)

    def calculate_expected_shortfall(self, confidence_level: float) -> float:
        """Calculate Expected Shortfall."""
        var = self.calculate_var(confidence_level)
        return self.returns[self.returns <= var].mean()


class DrawdownAnalysis:
    """Drawdown Analysis for portfolios."""

    def __init__(self, returns: pd.Series):
        self.returns = returns

    def calculate_drawdowns(self) -> pd.Series:
        """Calculate drawdown series."""
        cumulative = (1 + self.returns).cumprod()
        rolling_max = cumulative.cummax()
        return (cumulative - rolling_max) / rolling_max

    def get_drawdown_periods(self) -> List[Dict]:
        """Get drawdown periods."""
        drawdowns = self.calculate_drawdowns()
        periods = []
        in_drawdown = False
        start_date = None

        for date, dd in drawdowns.items():
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_date = date
            elif dd == 0 and in_drawdown:
                in_drawdown = False
                periods.append(
                    {
                        "start": start_date,
                        "end": date,
                        "max_drawdown": drawdowns.loc[start_date:date].min(),
                    }
                )

        return periods


class TailRisk:
    """Tail Risk Analysis."""

    def __init__(self, returns: pd.Series):
        self.returns = returns

    def calculate_tail_risk_metrics(self) -> Dict[str, float]:
        """Calculate tail risk metrics."""
        return {
            "skewness": self.returns.skew(),
            "kurtosis": self.returns.kurtosis(),
            "excess_kurtosis": self.returns.kurtosis() - 3,
            "tail_ratio": self.calculate_tail_ratio(),
            "conditional_var": self.calculate_conditional_var(),
        }

    def calculate_tail_ratio(self) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)."""
        p95 = np.percentile(self.returns, 95)
        p5 = np.percentile(self.returns, 5)
        return abs(p95 / p5) if p5 != 0 else 0

    def calculate_conditional_var(self) -> float:
        """Calculate conditional Value at Risk."""
        return self.returns[self.returns <= np.percentile(self.returns, 5)].mean()


class CorrelationAnalysis:
    """Correlation Analysis for portfolios."""

    def __init__(self, returns: pd.DataFrame):
        self.returns = returns

    def calculate_correlation_metrics(self) -> Dict[str, Union[float, pd.DataFrame]]:
        """Calculate correlation metrics."""
        return {
            "correlation_matrix": self.returns.corr(),
            "average_correlation": self.returns.corr().mean().mean(),
            "max_correlation": self.returns.corr().max().max(),
            "min_correlation": self.returns.corr().min().min(),
        }

    def calculate_rolling_correlation(self, window: int = 21) -> pd.DataFrame:
        """Calculate rolling correlation."""
        return self.returns.rolling(window=window).corr()
