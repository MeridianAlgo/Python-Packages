"""
Institutional-grade portfolio management module for MeridianAlgo.

This module provides comprehensive portfolio management capabilities including:
- Advanced optimization algorithms (Black-Litterman, Risk Parity, HRP)
- Risk management system with VaR, CVaR, and stress testing
- Transaction cost optimization and tax-loss harvesting
- Performance attribution and factor analysis
"""

from .optimization import (
    PortfolioOptimizer,
    BlackLittermanOptimizer,
    RiskParityOptimizer,
    HierarchicalRiskParityOptimizer,
    FactorModelOptimizer,
    OptimizationResult
)

try:
    from .risk_management import (
        RiskManager,
        VaRCalculator,
        StressTester,
        RiskMetrics
    )
    RISK_MANAGEMENT_AVAILABLE = True
except ImportError:
    RISK_MANAGEMENT_AVAILABLE = False

try:
    from .performance import (
        PerformanceAnalyzer,
        AttributionAnalyzer,
        FactorAnalyzer
    )
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False

try:
    from .transaction_costs import (
        TransactionCostOptimizer,
        TaxLossHarvester,
        LinearImpactModel,
        SquareRootImpactModel
    )
    TRANSACTION_COSTS_AVAILABLE = True
except ImportError:
    TRANSACTION_COSTS_AVAILABLE = False

try:
    from .rebalancing import (
        Rebalancer,
        CalendarRebalancer,
        ThresholdRebalancer,
        OptimalRebalancer
    )
    REBALANCING_AVAILABLE = True
except ImportError:
    REBALANCING_AVAILABLE = False

__all__ = [
    # Optimization
    'PortfolioOptimizer',
    'BlackLittermanOptimizer',
    'RiskParityOptimizer', 
    'HierarchicalRiskParityOptimizer',
    'OptimizationResult'
]

# Add available modules to __all__
if RISK_MANAGEMENT_AVAILABLE:
    __all__.extend(['RiskManager', 'VaRCalculator', 'StressTester', 'RiskMetrics'])

if PERFORMANCE_AVAILABLE:
    __all__.extend(['PerformanceAnalyzer', 'AttributionAnalyzer', 'FactorAnalyzer'])

if TRANSACTION_COSTS_AVAILABLE:
    __all__.extend(['TransactionCostOptimizer', 'TaxLossHarvester', 'LinearImpactModel', 'SquareRootImpactModel'])

if REBALANCING_AVAILABLE:
    __all__.extend(['Rebalancer', 'CalendarRebalancer', 'ThresholdRebalancer', 'OptimalRebalancer'])