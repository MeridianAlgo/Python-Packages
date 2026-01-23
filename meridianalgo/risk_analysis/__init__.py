"""
Risk Analysis Module for MeridianAlgo

This module provides comprehensive risk analysis tools including VaR, ES,
stress testing, and risk metrics.
"""

from .regime_analysis import MarketRegime, RegimeDetector, VolatilityRegime
from .risk_metrics import CorrelationAnalysis, DrawdownAnalysis, RiskMetrics, TailRisk
from .stress_testing import HistoricalStressTest, ScenarioAnalysis, StressTester
from .var_es import (
    ExpectedShortfall,
    HistoricalVaR,
    MonteCarloVaR,
    ParametricVaR,
    VaRCalculator,
)

__all__ = [
    # VaR and ES
    "VaRCalculator",
    "ExpectedShortfall",
    "HistoricalVaR",
    "ParametricVaR",
    "MonteCarloVaR",
    # Stress Testing
    "StressTester",
    "ScenarioAnalysis",
    "HistoricalStressTest",
    # Risk Metrics
    "RiskMetrics",
    "DrawdownAnalysis",
    "TailRisk",
    "CorrelationAnalysis",
    # Regime Analysis
    "RegimeDetector",
    "VolatilityRegime",
    "MarketRegime",
]
