"""
MeridianAlgo Analytics Module

Comprehensive portfolio analytics and tear sheet generation inspired by pyfolio.
Provides institutional-grade performance analysis, risk decomposition, and visualization.
"""

from .attribution import BrinsonAttribution, FactorAttribution, PerformanceAttribution
from .drawdown import DrawdownAnalyzer, calculate_drawdown_series
from .performance import PerformanceAnalyzer, calculate_returns_metrics
from .risk_analytics import RiskAnalyzer, calculate_risk_metrics
from .tear_sheets import (
    TearSheet,
    create_bayesian_tear_sheet,
    create_full_tear_sheet,
    create_position_tear_sheet,
    create_returns_tear_sheet,
    create_round_trip_tear_sheet,
)

__all__ = [
    "PerformanceAnalyzer",
    "RiskAnalyzer",
    "TearSheet",
    "create_full_tear_sheet",
    "create_returns_tear_sheet",
    "create_position_tear_sheet",
    "create_round_trip_tear_sheet",
    "create_bayesian_tear_sheet",
    "PerformanceAttribution",
    "BrinsonAttribution",
    "FactorAttribution",
    "DrawdownAnalyzer",
    "calculate_returns_metrics",
    "calculate_risk_metrics",
    "calculate_drawdown_series",
]
