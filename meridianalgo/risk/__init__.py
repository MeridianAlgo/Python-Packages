"""
Risk Management module for MeridianAlgo.

Provides comprehensive tools for Value at Risk (VaR), Conditional VaR (CVaR),
stress testing, and risk budgeting.
"""

# Core metrics and analysis
from .core import RiskAnalyzer, calculate_risk_metrics

# Advanced risk management
from .advanced import AdvancedVaR, StressTesting, RiskBudgeting

# Comprehensive exports and aliases for institutional grade usage
VaRCalculator = RiskAnalyzer
CVaRCalculator = RiskAnalyzer
DrawdownAnalyzer = RiskAnalyzer # Integrated in RiskAnalyzer
RiskMetrics = RiskMetricsResult if 'RiskMetricsResult' in locals() else object # Placeholder/ref
ScenarioAnalyzer = StressTesting
StressTest = StressTesting
TailRiskAnalyzer = RiskAnalyzer # Integrated in RiskAnalyzer

__all__ = [
    "RiskAnalyzer",
    "calculate_risk_metrics",
    "AdvancedVaR",
    "StressTesting",
    "RiskBudgeting",
    "VaRCalculator",
    "CVaRCalculator",
    "DrawdownAnalyzer",
    "ScenarioAnalyzer",
    "StressTest",
    "TailRiskAnalyzer"
]
