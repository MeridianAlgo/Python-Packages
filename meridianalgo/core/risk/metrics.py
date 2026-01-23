"""
Risk metrics module.

This module provides tools for calculating various financial risk metrics.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ...config import get_config


def calculate_metrics(
    returns: pd.Series, risk_free_rate: Optional[float] = None
) -> Dict[str, float]:
    """Calculate key performance metrics for a return series.

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: from config)

    Returns:
        Dictionary of performance metrics
    """
    if risk_free_rate is None:
        risk_free_rate = get_config("risk_free_rate", 0.0)

    # Convert annual risk-free rate to daily if needed
    daily_risk_free = (1 + risk_free_rate) ** (1 / 252) - 1
    excess_returns = returns - daily_risk_free

    metrics = {
        "total_return": (1 + returns).prod() - 1,
        "annualized_return": (1 + returns).prod() ** (252 / len(returns)) - 1,
        "volatility": returns.std() * np.sqrt(252),
        "annualized_volatility": returns.std() * np.sqrt(252),
        "sharpe_ratio": (
            (excess_returns.mean() / returns.std()) * np.sqrt(252)
            if returns.std() > 0
            else 0
        ),
        "max_drawdown": calculate_max_drawdown(returns),
        "sortino_ratio": calculate_sortino_ratio(returns, risk_free_rate),
        "calmar_ratio": calculate_calmar_ratio(returns),
    }

    return metrics


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from a return series.

    Args:
        returns: Series of returns

    Returns:
        Maximum drawdown as a decimal
    """
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns - running_max) / running_max
    return drawdowns.min()


def calculate_value_at_risk(
    returns: pd.Series, confidence_level: float = 0.95
) -> float:
    """Calculate Value at Risk (VaR) using historical simulation.

    Args:
        returns: Series of returns
        confidence_level: Confidence level for VaR (default: 0.95)

    Returns:
        Value at Risk as a decimal (negative for losses)
    """
    if not (0 <= confidence_level <= 1):
        raise ValueError("confidence_level must be between 0 and 1")

    return np.percentile(returns, (1 - confidence_level) * 100)


def calculate_expected_shortfall(
    returns: pd.Series, confidence_level: float = 0.95
) -> float:
    """Calculate Expected Shortfall (CVaR) using historical simulation.

    Args:
        returns: Series of returns
        confidence_level: Confidence level for ES (default: 0.95)

    Returns:
        Expected Shortfall as a decimal (negative for losses)
    """
    if not (0 <= confidence_level <= 1):
        raise ValueError("confidence_level must be between 0 and 1")

    var = calculate_value_at_risk(returns, confidence_level)
    return returns[returns <= var].mean()


def calculate_sortino_ratio(
    returns: pd.Series, risk_free_rate: Optional[float] = None
) -> float:
    """Calculate the Sortino ratio.

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: from config)

    Returns:
        Sortino ratio
    """
    if risk_free_rate is None:
        risk_free_rate = get_config("risk_free_rate", 0.0)

    # Convert annual risk-free rate to daily if needed
    daily_risk_free = (1 + risk_free_rate) ** (1 / 252) - 1
    returns - daily_risk_free

    # Calculate downside deviation
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return float("inf")

    downside_deviation = np.sqrt((downside_returns**2).mean()) * np.sqrt(252)

    if downside_deviation == 0:
        return float("inf")

    return (returns.mean() * 252 - risk_free_rate) / downside_deviation


def calculate_calmar_ratio(returns: pd.Series, period: str = "daily") -> float:
    """Calculate the Calmar ratio (return/max drawdown).

    Args:
        returns: Series of returns
        period: Period of returns ('daily', 'monthly', 'annual')

    Returns:
        Calmar ratio
    """
    max_dd = abs(calculate_max_drawdown(returns))
    if max_dd == 0:
        return float("inf")

    # Annualize returns based on period
    if period == "daily":
        annual_factor = 252
    elif period == "monthly":
        annual_factor = 12
    elif period == "annual":
        annual_factor = 1
    else:
        raise ValueError("Period must be 'daily', 'monthly', or 'annual'")

    annualized_return = (1 + returns).prod() ** (annual_factor / len(returns)) - 1
    return annualized_return / max_dd
