"""
MeridianAlgo Signals Module

Comprehensive technical indicators and signal generation for trading strategies.
"""

from .indicators import (
    # Trend Indicators
    SMA, EMA, WMA, DEMA, TEMA, KAMA,
    MACD, ADX, Aroon, ParabolicSAR,
    Supertrend, Ichimoku,
    
    # Momentum Indicators
    RSI, Stochastic, WilliamsR, CCI, ROC, Momentum,
    MFI, TSI, UltimateOscillator,
    
    # Volatility Indicators
    BollingerBands, ATR, KeltnerChannels, DonchianChannels,
    StandardDeviation, AverageTrueRange,
    
    # Volume Indicators
    OBV, VWAP, ChaikinMoneyFlow, AccumulationDistribution,
    ForceIndex, EaseOfMovement,
    
    # Support/Resistance
    PivotPoints, FibonacciRetracement, FibonacciExtension
)

from .generator import SignalGenerator, TechnicalAnalyzer
from .evaluation import SignalEvaluator

__all__ = [
    # Trend
    'SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'KAMA',
    'MACD', 'ADX', 'Aroon', 'ParabolicSAR',
    'Supertrend', 'Ichimoku',
    
    # Momentum
    'RSI', 'Stochastic', 'WilliamsR', 'CCI', 'ROC', 'Momentum',
    'MFI', 'TSI', 'UltimateOscillator',
    
    # Volatility
    'BollingerBands', 'ATR', 'KeltnerChannels', 'DonchianChannels',
    'StandardDeviation', 'AverageTrueRange',
    
    # Volume
    'OBV', 'VWAP', 'ChaikinMoneyFlow', 'AccumulationDistribution',
    'ForceIndex', 'EaseOfMovement',
    
    # S/R
    'PivotPoints', 'FibonacciRetracement', 'FibonacciExtension',
    
    # Generators/Evaluation
    'SignalGenerator', 'TechnicalAnalyzer', 'SignalEvaluator'
]
