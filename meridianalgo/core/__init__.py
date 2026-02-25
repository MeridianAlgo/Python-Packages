"""
Core module for MeridianAlgo.

Provides base financial primitives, optimization engines, and statistical tools.
"""

from .base import (
    PortfolioOptimizer,
    TimeSeriesAnalyzer,
    calculate_calmar_ratio,
    calculate_expected_shortfall,
    calculate_macd,
    calculate_max_drawdown,
    calculate_metrics,
    calculate_returns,
    calculate_rsi,
    calculate_sortino_ratio,
    calculate_value_at_risk,
    get_market_data,
)
from .statistics import (
    StatisticalArbitrage,
    calculate_autocorrelation,
    calculate_correlation_matrix,
    calculate_half_life,
    calculate_hurst_exponent,
    calculate_rolling_correlation,
    hurst_exponent,
)

# rolling_volatility is actually in risk.core
try:
    from ..risk.core import rolling_volatility
except ImportError:
    # Fallback if risk is not available
    def rolling_volatility(*args, **kwargs):
        raise ImportError("risk module required for rolling_volatility")


# Import specialized components if they exist
try:
    from .portfolio.optimization import (
        PortfolioOptimizer as BaseOptimizer,  # noqa: F401
    )
    from .risk.metrics import calculate_metrics as BaseMetrics  # noqa: F401
except ImportError:
    pass

__all__ = [
    "PortfolioOptimizer",
    "TimeSeriesAnalyzer",
    "get_market_data",
    "calculate_metrics",
    "calculate_max_drawdown",
    "calculate_value_at_risk",
    "calculate_expected_shortfall",
    "calculate_sortino_ratio",
    "calculate_calmar_ratio",
    "calculate_macd",
    "calculate_returns",
    "calculate_rsi",
    "StatisticalArbitrage",
    "calculate_correlation_matrix",
    "calculate_rolling_correlation",
    "calculate_hurst_exponent",
    "calculate_half_life",
    "calculate_autocorrelation",
    "hurst_exponent",
    "rolling_volatility",
]
