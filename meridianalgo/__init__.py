"""
MeridianAlgo - Advanced Algorithmic Trading and Statistical Analysis

A comprehensive Python library for algorithmic trading, featuring advanced 
statistical analysis, machine learning integration, and financial modeling 
tools for quantitative finance.

Version: 3.1.0
"""

__version__ = '3.1.0'

# Core modules
from .core import (
    PortfolioOptimizer,
    TimeSeriesAnalyzer,
    get_market_data,
    calculate_metrics,
    calculate_max_drawdown
)

# Statistics modules
from .statistics import (
    StatisticalArbitrage,
    calculate_value_at_risk,
    calculate_expected_shortfall,
    hurst_exponent,
    calculate_autocorrelation,
    rolling_volatility
)

# ML modules
from .ml import (
    FeatureEngineer,
    LSTMPredictor,
    prepare_data_for_lstm
)

# Technical Indicators
from .technical_indicators import (
    RSI, Stochastic, WilliamsR, ROC, Momentum,
    SMA, EMA, MACD, ADX, Aroon, ParabolicSAR, Ichimoku,
    BollingerBands, ATR, KeltnerChannels, DonchianChannels,
    OBV, ADLine, ChaikinOscillator, MoneyFlowIndex, EaseOfMovement,
    PivotPoints, FibonacciRetracement, SupportResistance
)

# Portfolio Management
from .portfolio_management import (
    PortfolioOptimizer as PM_PortfolioOptimizer,
    EfficientFrontier, BlackLitterman, RiskParity
)

# Risk Analysis
from .risk_analysis import (
    VaRCalculator, ExpectedShortfall as Risk_ExpectedShortfall,
    HistoricalVaR, ParametricVaR, MonteCarloVaR
)

# Data Processing
from .data_processing import (
    DataCleaner, OutlierDetector, MissingDataHandler,
    FeatureEngineer as DP_FeatureEngineer, TechnicalFeatures,
    DataValidator, MarketDataProvider
)

__all__ = [
    # Core
    'PortfolioOptimizer',
    'TimeSeriesAnalyzer',
    'get_market_data',
    'calculate_metrics',
    'calculate_max_drawdown',
    
    # Statistics
    'StatisticalArbitrage',
    'calculate_value_at_risk',
    'calculate_expected_shortfall',
    'hurst_exponent',
    'calculate_autocorrelation',
    'rolling_volatility',
    
    # ML
    'FeatureEngineer',
    'LSTMPredictor',
    'prepare_data_for_lstm',
    
    # Technical Indicators
    'RSI', 'Stochastic', 'WilliamsR', 'ROC', 'Momentum',
    'SMA', 'EMA', 'MACD', 'ADX', 'Aroon', 'ParabolicSAR', 'Ichimoku',
    'BollingerBands', 'ATR', 'KeltnerChannels', 'DonchianChannels',
    'OBV', 'ADLine', 'ChaikinOscillator', 'MoneyFlowIndex', 'EaseOfMovement',
    'PivotPoints', 'FibonacciRetracement', 'SupportResistance',
    
    # Portfolio Management
    'PM_PortfolioOptimizer', 'EfficientFrontier', 'BlackLitterman', 'RiskParity',
    
    # Risk Analysis
    'VaRCalculator', 'Risk_ExpectedShortfall', 'HistoricalVaR', 'ParametricVaR', 'MonteCarloVaR',
    
    # Data Processing
    'DataCleaner', 'OutlierDetector', 'MissingDataHandler',
    'DP_FeatureEngineer', 'TechnicalFeatures', 'DataValidator', 'MarketDataProvider'
]