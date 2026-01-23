"""
MeridianAlgo Quantitative Algorithms Module

Advanced quantitative algorithms for institutional-grade trading and research.
Includes market microstructure, high-frequency trading, statistical arbitrage,
and advanced execution algorithms.
"""

from .advanced_signals import (
    calculate_z_score,
    fractional_difference,
    get_half_life,
    hurst_exponent,
    information_coefficient,
)
from .execution_algorithms import (
    ImplementationShortfall,
    POV,
    TWAP,
    VWAP,
)
from .factor_models import (
    AlphaCapture,
    APTModel,
    CustomFactorModel,
    FamaFrenchModel,
    FactorRiskDecomposition,
)
from .high_frequency import (
    HFTSignalGenerator,
    LatencyArbitrage,
    LiquidityProvision,
    MarketMaking,
    MicropriceEstimator,
)
from .market_microstructure import (
    MarketImpactModel,
    OrderFlowImbalance,
    RealizedVolatility,
    TickDataAnalyzer,
    VolumeWeightedSpread,
)
from .regime_detection import (
    HiddenMarkovModel,
    MarketStateClassifier,
    RegimeSwitchingModel,
    StructuralBreakDetection,
    VolatilityRegimeDetector,
)
from .statistical_arbitrage import (
    CointegrationAnalyzer,
    MeanReversionTester,
    OrnsteinUhlenbeck,
    PairsTrading,
    SpreadAnalyzer,
)

__all__ = [
    # Market Microstructure
    "OrderFlowImbalance",
    "VolumeWeightedSpread",
    "RealizedVolatility",
    "MarketImpactModel",
    "TickDataAnalyzer",
    # Statistical Arbitrage
    "PairsTrading",
    "CointegrationAnalyzer",
    "OrnsteinUhlenbeck",
    "MeanReversionTester",
    "SpreadAnalyzer",
    # Execution Algorithms
    "VWAP",
    "TWAP",
    "POV",
    "ImplementationShortfall",
    # High Frequency
    "LatencyArbitrage",
    "MarketMaking",
    "LiquidityProvision",
    "HFTSignalGenerator",
    "MicropriceEstimator",
    # Factor Models
    "FamaFrenchModel",
    "APTModel",
    "CustomFactorModel",
    "FactorRiskDecomposition",
    "AlphaCapture",
    # Regime Detection
    "HiddenMarkovModel",
    "RegimeSwitchingModel",
    "StructuralBreakDetection",
    "MarketStateClassifier",
    "VolatilityRegimeDetector",
    # Advanced Signals
    "hurst_exponent",
    "fractional_difference",
    "calculate_z_score",
    "get_half_life",
    "information_coefficient",
]
