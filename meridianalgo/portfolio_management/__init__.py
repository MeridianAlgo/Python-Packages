"""
Portfolio Management Module for MeridianAlgo

This module provides comprehensive portfolio management tools including optimization,
risk management, and performance analysis.
"""

from .optimization import (
    PortfolioOptimizer, EfficientFrontier, BlackLitterman, RiskParity
)

from .risk_management import (
    RiskManager, VaRCalculator, StressTester, RiskMetrics
)

from .performance import (
    PerformanceAnalyzer, AttributionAnalysis, BenchmarkComparison
)

from .rebalancing import (
    Rebalancer, CalendarRebalancer, ThresholdRebalancer
)

__all__ = [
    # Optimization
    'PortfolioOptimizer', 'EfficientFrontier', 'BlackLitterman', 'RiskParity',
    
    # Risk Management
    'RiskManager', 'VaRCalculator', 'StressTester', 'RiskMetrics',
    
    # Performance
    'PerformanceAnalyzer', 'AttributionAnalysis', 'BenchmarkComparison',
    
    # Rebalancing
    'Rebalancer', 'CalendarRebalancer', 'ThresholdRebalancer'
]
