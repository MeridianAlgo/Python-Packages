"""
Core module for MeridianAlgo.

This module provides the core functionality for quantitative finance and algorithmic trading.
"""

from .portfolio.optimization import PortfolioOptimizer
from .time_series.analysis import TimeSeriesAnalyzer, get_market_data
from .risk.metrics import (
    calculate_metrics,
    calculate_max_drawdown,
    calculate_value_at_risk,
    calculate_expected_shortfall,
    calculate_sortino_ratio,
    calculate_calmar_ratio
)
from .statistics import (
    StatisticalArbitrage,
    calculate_correlation_matrix,
    calculate_rolling_correlation,
    calculate_hurst_exponent,
    calculate_half_life,
    calculate_autocorrelation,
    hurst_exponent,
    rolling_volatility
)

__all__ = [
    'PortfolioOptimizer',
    'TimeSeriesAnalyzer',
    'get_market_data',
    'calculate_metrics',
    'calculate_max_drawdown',
    'calculate_value_at_risk',
    'calculate_expected_shortfall',
    'calculate_sortino_ratio',
    'calculate_calmar_ratio',
    'StatisticalArbitrage',
    'calculate_correlation_matrix',
    'calculate_rolling_correlation',
    'calculate_hurst_exponent',
    'calculate_half_life',
    'calculate_autocorrelation',
    'hurst_exponent',
    'rolling_volatility'
]