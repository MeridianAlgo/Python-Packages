"""
MeridianAlgo Quantitative Algorithms Module

Advanced quantitative algorithms for institutional-grade trading and research.
Includes market microstructure, high-frequency trading, statistical arbitrage,
and advanced execution algorithms.
"""

from .market_microstructure import *
from .statistical_arbitrage import *
from .execution_algorithms import *
from .high_frequency import *
from .factor_models import *
from .regime_detection import *

__all__ = [
    # Market Microstructure
    'OrderFlowImbalance',
    'VolumeWeightedSpread',
    'RealizedVolatility',
    'MarketImpactModel',
    'TickDataAnalyzer',
    
    # Statistical Arbitrage
    'PairsTrading',
    'CointegrationAnalyzer',
    'OrnsteinUhlenbeck',
    'MeanReversionTester',
    'SpreadAnalyzer',
    
    # Execution Algorithms
    'VWAP',
    'TWAP',
    'POV',
    'ImplementationShortfall',
    'AlmanacExecution',
    
    # High Frequency
    'LatencyArbitrage',
    'MarketMaking',
    'LiquidityProvision',
    'HFTSignalGenerator',
    'MicropriceEstimator',
    
    # Factor Models
    'FamaFrenchModel',
    'APTModel',
    'CustomFactorModel',
    'FactorRiskDecomposition',
    'AlphaCapture',
    
    # Regime Detection
    'HiddenMarkovModel',
    'RegimeSwitchingModel',
    'StructuralBreakDetection',
    'MarketStateClassifier',
    'VolatilityRegimeDetector',
]
