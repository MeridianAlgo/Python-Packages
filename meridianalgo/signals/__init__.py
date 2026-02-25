"""
MeridianAlgo Signals Module

Comprehensive technical indicators and signal generation for trading strategies.
"""

from .evaluation import SignalEvaluator
from .generator import SignalGenerator, TechnicalAnalyzer
from .indicators import (  # Trend Indicators; Momentum Indicators; Volatility Indicators; Volume Indicators; Support/Resistance
    ADX,
    ATR,
    CCI,
    DEMA,
    EMA,
    KAMA,
    MACD,
    MFI,
    OBV,
    ROC,
    RSI,
    SMA,
    TEMA,
    TSI,
    VWAP,
    WMA,
    AccumulationDistribution,
    Aroon,
    AverageTrueRange,
    BollingerBands,
    ChaikinMoneyFlow,
    DonchianChannels,
    EaseOfMovement,
    FibonacciExtension,
    FibonacciRetracement,
    ForceIndex,
    Ichimoku,
    KeltnerChannels,
    Momentum,
    ParabolicSAR,
    PivotPoints,
    StandardDeviation,
    Stochastic,
    Supertrend,
    UltimateOscillator,
    WilliamsR,
)

__all__ = [
    # Trend
    "SMA",
    "EMA",
    "WMA",
    "DEMA",
    "TEMA",
    "KAMA",
    "MACD",
    "ADX",
    "Aroon",
    "ParabolicSAR",
    "Supertrend",
    "Ichimoku",
    # Momentum
    "RSI",
    "Stochastic",
    "WilliamsR",
    "CCI",
    "ROC",
    "Momentum",
    "MFI",
    "TSI",
    "UltimateOscillator",
    # Volatility
    "BollingerBands",
    "ATR",
    "KeltnerChannels",
    "DonchianChannels",
    "StandardDeviation",
    "AverageTrueRange",
    # Volume
    "OBV",
    "VWAP",
    "ChaikinMoneyFlow",
    "AccumulationDistribution",
    "ForceIndex",
    "EaseOfMovement",
    # S/R
    "PivotPoints",
    "FibonacciRetracement",
    "FibonacciExtension",
    # Generators/Evaluation
    "SignalGenerator",
    "TechnicalAnalyzer",
    "SignalEvaluator",
]
