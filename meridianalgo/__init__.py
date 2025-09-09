"""
MeridianAlgo - Advanced Algorithmic Trading and Statistical Analysis

A comprehensive Python library for algorithmic trading, featuring advanced 
statistical analysis, machine learning integration, and financial modeling 
tools for quantitative finance.

Version: 3.0.0
"""

__version__ = '3.0.0'

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
    'prepare_data_for_lstm'
]