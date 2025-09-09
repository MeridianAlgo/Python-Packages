"""
Risk Analysis Module for MeridianAlgo

This module provides comprehensive risk analysis tools including VaR, ES,
stress testing, and risk metrics.
"""

from .var_es import (
    VaRCalculator, ExpectedShortfall, HistoricalVaR, ParametricVaR, MonteCarloVaR
)

from .stress_testing import (
    StressTester, ScenarioAnalysis, HistoricalStressTest
)

from .risk_metrics import (
    RiskMetrics, DrawdownAnalysis, TailRisk, CorrelationAnalysis
)

from .regime_analysis import (
    RegimeDetector, VolatilityRegime, MarketRegime
)

__all__ = [
    # VaR and ES
    'VaRCalculator', 'ExpectedShortfall', 'HistoricalVaR', 'ParametricVaR', 'MonteCarloVaR',
    
    # Stress Testing
    'StressTester', 'ScenarioAnalysis', 'HistoricalStressTest',
    
    # Risk Metrics
    'RiskMetrics', 'DrawdownAnalysis', 'TailRisk', 'CorrelationAnalysis',
    
    # Regime Analysis
    'RegimeDetector', 'VolatilityRegime', 'MarketRegime'
]
