"""
Institutional-grade portfolio management module for MeridianAlgo.

This module provides comprehensive portfolio management capabilities including:
- Advanced optimization algorithms (Black-Litterman, Risk Parity, HRP)
- Risk management system with VaR, CVaR, and stress testing
- Transaction cost optimization and tax-loss harvesting
- Performance attribution and factor analysis
"""

from .optimization import (BlackLittermanOptimizer, FactorModelOptimizer,
                           HierarchicalRiskParityOptimizer, OptimizationResult,
                           PortfolioOptimizer, RiskParityOptimizer)

# Aliases for package-wide consistency
BlackLitterman = BlackLittermanOptimizer
RiskParity = RiskParityOptimizer
HierarchicalRiskParity = HierarchicalRiskParityOptimizer
MeanVariance = PortfolioOptimizer
EfficientFrontier = PortfolioOptimizer

__all__ = [
    # Optimization
    "PortfolioOptimizer",
    "BlackLittermanOptimizer",
    "RiskParityOptimizer",
    "HierarchicalRiskParityOptimizer",
    "FactorModelOptimizer",
    "OptimizationResult",
    "BlackLitterman",
    "RiskParity",
    "HierarchicalRiskParity",
    "MeanVariance",
    "EfficientFrontier",
]

# Risk Management
try:
    from .risk_management import (RiskManager, RiskMetrics,  # noqa: F401
                                  StressTester, VaRCalculator)

    __all__.extend(["RiskManager", "VaRCalculator", "StressTester", "RiskMetrics"])
    RISK_MANAGEMENT_AVAILABLE = True
except ImportError:
    RISK_MANAGEMENT_AVAILABLE = False

# Performance Analysis
try:
    from .performance import (AttributionAnalyzer,  # noqa: F401
                              FactorAnalyzer, PerformanceAnalyzer)

    __all__.extend(["PerformanceAnalyzer", "AttributionAnalyzer", "FactorAnalyzer"])
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False

# Transaction Costs
try:
    from .transaction_costs import (LinearImpactModel,  # noqa: F401
                                    SquareRootImpactModel, TaxLossHarvester,
                                    TransactionCostOptimizer)

    __all__.extend(
        [
            "TransactionCostOptimizer",
            "TaxLossHarvester",
            "LinearImpactModel",
            "SquareRootImpactModel",
        ]
    )
    TRANSACTION_COSTS_AVAILABLE = True
except ImportError:
    TRANSACTION_COSTS_AVAILABLE = False

# Rebalancing
try:
    from .rebalancing import (CalendarRebalancer,  # noqa: F401
                              OptimalRebalancer, Rebalancer,
                              ThresholdRebalancer)

    __all__.extend(
        ["Rebalancer", "CalendarRebalancer", "ThresholdRebalancer", "OptimalRebalancer"]
    )
    REBALANCING_AVAILABLE = True
except ImportError:
    REBALANCING_AVAILABLE = False
