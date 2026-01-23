"""
MeridianAlgo Quantitative Algorithms Module

Advanced quantitative algorithms for institutional-grade trading and research.
Includes market microstructure, high-frequency trading, statistical arbitrage,
and advanced execution algorithms.
"""

from .advanced_signals import *
from .execution_algorithms import *
from .factor_models import *
from .high_frequency import *
from .market_microstructure import *
from .regime_detection import *
from .statistical_arbitrage import *

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
    "AlmanacExecution",
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
