"""
Institutional-grade portfolio management module for MeridianAlgo.

This module provides comprehensive portfolio management capabilities including:
- Advanced optimization algorithms (Black-Litterman, Risk Parity, HRP)
- Risk management system with VaR, CVaR, and stress testing
- Transaction cost optimization and tax-loss harvesting
- Performance attribution and factor analysis
"""

from .optimization import (
    BlackLittermanOptimizer,
    FactorModelOptimizer,
    HierarchicalRiskParityOptimizer,
    OptimizationResult,
    PortfolioOptimizer,
    RiskParityOptimizer,
)

__all__ = [
    # Optimization
    "PortfolioOptimizer",
    "BlackLittermanOptimizer",
    "RiskParityOptimizer",
    "HierarchicalRiskParityOptimizer",
    "FactorModelOptimizer",
    "OptimizationResult",
]

# Risk Management
try:
    from .risk_management import (  # noqa: F401
        RiskManager,
        RiskMetrics,
        StressTester,
        VaRCalculator,
    )

    __all__.extend(["RiskManager", "VaRCalculator", "StressTester", "RiskMetrics"])
    RISK_MANAGEMENT_AVAILABLE = True
except ImportError:
    RISK_MANAGEMENT_AVAILABLE = False

# Performance Analysis
try:
    from .performance import (  # noqa: F401
        AttributionAnalyzer,
        FactorAnalyzer,
        PerformanceAnalyzer,
    )

    __all__.extend(["PerformanceAnalyzer", "AttributionAnalyzer", "FactorAnalyzer"])
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False

# Transaction Costs
try:
    from .transaction_costs import (  # noqa: F401
        LinearImpactModel,
        SquareRootImpactModel,
        TaxLossHarvester,
        TransactionCostOptimizer,
    )

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
    from .rebalancing import (  # noqa: F401
        CalendarRebalancer,
        OptimalRebalancer,
        Rebalancer,
        ThresholdRebalancer,
    )

    __all__.extend(
        ["Rebalancer", "CalendarRebalancer", "ThresholdRebalancer", "OptimalRebalancer"]
    )
    REBALANCING_AVAILABLE = True
except ImportError:
    REBALANCING_AVAILABLE = False
