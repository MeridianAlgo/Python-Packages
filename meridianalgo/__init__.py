"""
MeridianAlgo v7.0.0 - The Complete Quantitative Finance Platform

Institutional-grade Python library for quantitative finance covering portfolio
optimization, risk management, derivatives pricing, backtesting, machine learning,
statistical arbitrage, execution algorithms, fixed income analytics, credit risk,
volatility modeling, Monte Carlo simulation, and portfolio insurance.
"""

__version__ = "7.0.0"

import os
from typing import Any, Dict

# Configure logging
from .utils.logging import setup_logger

logger = setup_logger("meridianalgo")


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

    @classmethod
    def status(cls) -> Dict[str, bool]:
        return {k: v["available"] for k, v in cls._modules.items()}


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

# Core Financial Functions
try:
    from .core.base import (
        calculate_calmar_ratio,
        calculate_expected_shortfall,
        calculate_max_drawdown,
        calculate_sortino_ratio,
    )

    ModuleRegistry.register("core_functions", True)
except ImportError as e:
    ModuleRegistry.register("core_functions", False, str(e))

# Technical Indicators
try:
    from .signals.indicators import BollingerBands as calculate_bollinger_bands

    ModuleRegistry.register("indicators", True)
except ImportError as e:
    ModuleRegistry.register("indicators", False, str(e))

# Sharpe Ratio
try:
    from .strategies.algorithmic import calculate_sharpe_ratio

    ModuleRegistry.register("sharpe", True)
except ImportError as e:
    ModuleRegistry.register("sharpe", False, str(e))

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

# Kelly Criterion
try:
    from .portfolio.kelly import KellyCriterion

    ModuleRegistry.register("kelly", True)
except ImportError as e:
    ModuleRegistry.register("kelly", False, str(e))

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
        WalkForwardValidator,
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
        ImpliedVolatility,
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
    from .backtesting import Backtest, BacktestEngine, Strategy

    ModuleRegistry.register("backtesting", True)
except ImportError as e:
    ModuleRegistry.register("backtesting", False, str(e))

# Credit Risk
try:
    from .credit import (
        CreditDefaultSwap,
        CreditRiskAnalyzer,
        MertonModel,
        ZSpreadCalculator,
    )

    ModuleRegistry.register("credit", True)
except ImportError as e:
    ModuleRegistry.register("credit", False, str(e))

# Volatility Models
try:
    from .volatility import (
        GARCHModel,
        RealizedVolatility,
        VolatilityForecaster,
        VolatilityRegimeDetector,
        VolatilityTermStructure,
    )

    ModuleRegistry.register("volatility", True)
except ImportError as e:
    ModuleRegistry.register("volatility", False, str(e))

# Monte Carlo Simulation
try:
    from .monte_carlo import (
        CIRModel,
        GeometricBrownianMotion,
        HestonModel,
        JumpDiffusionModel,
        MonteCarloEngine,
        QuasiRandomSampler,
    )

    ModuleRegistry.register("monte_carlo", True)
except ImportError as e:
    ModuleRegistry.register("monte_carlo", False, str(e))

# Portfolio Insurance
try:
    from .portfolio.insurance import CPPI, TimeInvariantCPPI

    ModuleRegistry.register("portfolio_insurance", True)
except ImportError as e:
    ModuleRegistry.register("portfolio_insurance", False, str(e))

# Benchmark Analytics
try:
    from .analytics.benchmark import (
        ActiveShare,
        BenchmarkAnalytics,
        BrinsonAttribution,
    )

    ModuleRegistry.register("benchmark_analytics", True)
except ImportError as e:
    ModuleRegistry.register("benchmark_analytics", False, str(e))

# Scenario Analysis
try:
    from .risk.scenario import (
        CorrelationScenario,
        ScenarioAnalyzer,
    )

    ModuleRegistry.register("scenario_analysis", True)
except ImportError as e:
    ModuleRegistry.register("scenario_analysis", False, str(e))

# ============================================================================
# ALIASES — backward-compatible and README-documented shortcuts
# ============================================================================

# Risk aliases
try:
    RiskMetrics = RiskAnalyzer  # type: ignore[name-defined]
except NameError:
    pass

# Backtesting aliases
try:
    Backtester = BacktestEngine  # type: ignore[name-defined]
except NameError:
    pass

# ML aliases
try:
    ModelValidator = WalkForwardValidator  # type: ignore[name-defined]
except NameError:
    pass

__all__ = [
    # Core primitives
    "PortfolioOptimizer",
    "TimeSeriesAnalyzer",
    "StatisticalArbitrage",
    "calculate_metrics",
    "get_market_data",
    "calculate_macd",
    "calculate_returns",
    "calculate_rsi",
    # Financial functions
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_calmar_ratio",
    "calculate_max_drawdown",
    "calculate_expected_shortfall",
    "calculate_bollinger_bands",
    # Portfolio
    "BlackLitterman",
    "RiskParity",
    "MeanVariance",
    "HierarchicalRiskParity",
    "KellyCriterion",
    # Risk
    "RiskAnalyzer",
    "RiskMetrics",
    "VaRCalculator",
    "CVaRCalculator",
    "StressTesting",
    "RiskBudgeting",
    # Machine learning
    "LSTMPredictor",
    "ModelTrainer",
    "ModelSelector",
    "TimeSeriesCV",
    "WalkForwardOptimizer",
    "WalkForwardValidator",
    "ModelValidator",
    "prepare_data_for_lstm",
    # Analytics
    "PerformanceAnalyzer",
    # Fixed income
    "BondPricer",
    "YieldCurve",
    "CreditSpreadAnalyzer",
    # Derivatives
    "BlackScholes",
    "GreeksCalculator",
    "ImpliedVolatility",
    "OptionChain",
    "MonteCarloPricer",
    # Execution
    "VWAP",
    "TWAP",
    "POV",
    "ImplementationShortfall",
    # Strategies
    "MomentumStrategy",
    "RSIMeanReversion",
    "MACDCrossover",
    "PairsTrading",
    "BollingerBandsStrategy",
    # Backtesting
    "BacktestEngine",
    "Backtest",
    "Backtester",
    "Strategy",
    # Credit risk
    "MertonModel",
    "CreditDefaultSwap",
    "CreditRiskAnalyzer",
    "ZSpreadCalculator",
    # Volatility models
    "GARCHModel",
    "RealizedVolatility",
    "VolatilityForecaster",
    "VolatilityTermStructure",
    "VolatilityRegimeDetector",
    # Monte Carlo
    "GeometricBrownianMotion",
    "HestonModel",
    "JumpDiffusionModel",
    "CIRModel",
    "MonteCarloEngine",
    "QuasiRandomSampler",
    # Portfolio insurance
    "CPPI",
    "TimeInvariantCPPI",
    # Benchmark analytics
    "BenchmarkAnalytics",
    "ActiveShare",
    "BrinsonAttribution",
    # Scenario analysis
    "ScenarioAnalyzer",
    "CorrelationScenario",
    # Registry
    "ModuleRegistry",
]
