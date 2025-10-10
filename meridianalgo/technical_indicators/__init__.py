"""
Technical Indicators Module for MeridianAlgo

This module provides comprehensive technical analysis indicators for financial data.
Includes momentum, trend, volatility, and volume indicators.
"""

from .momentum import (
    RSI, Stochastic, WilliamsR, ROC, Momentum
)

from .trend import (
    SMA, EMA, MACD, ADX, Aroon, ParabolicSAR, Ichimoku
)

from .volatility import (
    BollingerBands, ATR, KeltnerChannels, DonchianChannels
)

from .volume import (
    OBV, ADLine, ChaikinOscillator, MoneyFlowIndex, EaseOfMovement
)

from .overlay import (
    PivotPoints, FibonacciRetracement, SupportResistance
)

# Import TA library integration
try:
    from ..technical_analysis.ta_integration import (
        TAIntegration,
        add_all_ta_features,
        get_ta_volume_indicators,
        get_ta_volatility_indicators,
        get_ta_trend_indicators,
        get_ta_momentum_indicators,
        get_all_ta_indicators
    )
    TA_INTEGRATION_AVAILABLE = True
except ImportError:
    TA_INTEGRATION_AVAILABLE = False

__all__ = [
    # Momentum indicators
    'RSI', 'Stochastic', 'WilliamsR', 'ROC', 'Momentum',
    
    # Trend indicators
    'SMA', 'EMA', 'MACD', 'ADX', 'Aroon', 'ParabolicSAR', 'Ichimoku',
    
    # Volatility indicators
    'BollingerBands', 'ATR', 'KeltnerChannels', 'DonchianChannels',
    
    # Volume indicators
    'OBV', 'ADLine', 'ChaikinOscillator', 'MoneyFlowIndex', 'EaseOfMovement',
    
    # Overlay indicators
    'PivotPoints', 'FibonacciRetracement', 'SupportResistance'
]

# Add TA integration to exports if available
if TA_INTEGRATION_AVAILABLE:
    __all__.extend([
        'TAIntegration',
        'add_all_ta_features',
        'get_ta_volume_indicators',
        'get_ta_volatility_indicators',
        'get_ta_trend_indicators',
        'get_ta_momentum_indicators',
        'get_all_ta_indicators'
    ])
