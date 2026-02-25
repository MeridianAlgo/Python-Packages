"""
MeridianAlgo v6.2.1 - The Complete Quantitative Finance Platform

A comprehensive, institutional-grade Python library for quantitative finance
covering everything from trading research to portfolio analytics to derivatives.
"""

__version__ = "6.2.2"

import os
import warnings
from typing import Any, Dict

# Configure logging
from .utils.logging import setup_logger

logger = setup_logger("meridianalgo")

# Suppress warnings
warnings.filterwarnings("ignore")

# ============================================================================
# MODULE REGISTRY
# ============================================================================


class ModuleRegistry:
    """Registry for managing package modules and their availability."""

    _modules: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, name: str, available: bool, error: str = "") -> None:
        cls._modules[name] = {"available": available, "error": error}

    @classmethod
    def is_available(cls, name: str) -> bool:
        return cls._modules.get(name, {}).get("available", False)


# ============================================================================
# CORE EXPORTS
# ============================================================================

# Core Primitives
try:
    from .core import (
        PortfolioOptimizer,
        StatisticalArbitrage,
        TimeSeriesAnalyzer,
        calculate_macd,
        calculate_metrics,
        calculate_returns,
        calculate_rsi,
        get_market_data,
    )

    ModuleRegistry.register("core", True)
except ImportError as e:
    ModuleRegistry.register("core", False, str(e))

# Portfolio Management
try:
    from .portfolio import (
        BlackLitterman,
        HierarchicalRiskParity,
        MeanVariance,
        RiskParity,
    )

    ModuleRegistry.register("portfolio", True)
except ImportError as e:
    ModuleRegistry.register("portfolio", False, str(e))

# Risk Management
try:
    from .risk import (
        CVaRCalculator,
        RiskAnalyzer,
        RiskBudgeting,
        StressTesting,
        VaRCalculator,
    )

    ModuleRegistry.register("risk", True)
except ImportError as e:
    ModuleRegistry.register("risk", False, str(e))

# Machine Learning
try:
    from .ml import (
        LSTMPredictor,
        ModelSelector,
        ModelTrainer,
        TimeSeriesCV,
        WalkForwardOptimizer,
        prepare_data_for_lstm,
    )

    ModuleRegistry.register("ml", True)
except ImportError as e:
    ModuleRegistry.register("ml", False, str(e))

# Analytics
try:
    from .analytics import PerformanceAnalyzer

    ModuleRegistry.register("analytics", True)
except ImportError as e:
    ModuleRegistry.register("analytics", False, str(e))

# Fixed Income
try:
    from .fixed_income import BondPricer, CreditSpreadAnalyzer, YieldCurve

    ModuleRegistry.register("fixed_income", True)
except ImportError as e:
    ModuleRegistry.register("fixed_income", False, str(e))

# Derivatives
try:
    from .derivatives import (
        BlackScholes,
        GreeksCalculator,
        MonteCarloPricer,
        OptionChain,
    )

    ModuleRegistry.register("derivatives", True)
except ImportError as e:
    ModuleRegistry.register("derivatives", False, str(e))

# Execution
try:
    from .execution import POV, TWAP, VWAP, ImplementationShortfall

    ModuleRegistry.register("execution", True)
except ImportError as e:
    ModuleRegistry.register("execution", False, str(e))

# Strategies
try:
    from .strategies import (
        BollingerBandsStrategy,
        MACDCrossover,
        MomentumStrategy,
        PairsTrading,
        RSIMeanReversion,
    )

    ModuleRegistry.register("strategies", True)
except ImportError as e:
    ModuleRegistry.register("strategies", False, str(e))

# Backtesting
try:
    from .backtesting import Backtest, BacktestEngine

    ModuleRegistry.register("backtesting", True)
except ImportError as e:
    ModuleRegistry.register("backtesting", False, str(e))

# Welcome Message
if not os.getenv("MERIDIANALGO_QUIET", "0") == "1":
    print(f"MeridianAlgo v{__version__} INITIALIZED")
    print("Institutional Edition - Made with love by MeridianAlgo")

__all__ = [
    "PortfolioOptimizer",
    "TimeSeriesAnalyzer",
    "StatisticalArbitrage",
    "calculate_metrics",
    "get_market_data",
    "calculate_macd",
    "calculate_returns",
    "calculate_rsi",
    "BlackLitterman",
    "RiskParity",
    "MeanVariance",
    "HierarchicalRiskParity",
    "RiskAnalyzer",
    "VaRCalculator",
    "CVaRCalculator",
    "StressTesting",
    "RiskBudgeting",
    "LSTMPredictor",
    "ModelTrainer",
    "ModelSelector",
    "TimeSeriesCV",
    "WalkForwardOptimizer",
    "prepare_data_for_lstm",
    "PerformanceAnalyzer",
    "BondPricer",
    "YieldCurve",
    "CreditSpreadAnalyzer",
    "BlackScholes",
    "GreeksCalculator",
    "OptionChain",
    "MonteCarloPricer",
    "VWAP",
    "TWAP",
    "POV",
    "ImplementationShortfall",
    "MomentumStrategy",
    "RSIMeanReversion",
    "MACDCrossover",
    "PairsTrading",
    "BollingerBandsStrategy",
    "BacktestEngine",
    "Backtest",
]
