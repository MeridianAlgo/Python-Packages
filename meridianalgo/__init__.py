"""
MeridianAlgo v4.0.3 - Quantum Edition 🚀

Meridian Quant: The Ultimate Quantitative Development Platform

The most advanced Python platform for quantitative finance, integrating cutting-edge 
machine learning, institutional-grade portfolio management, and high-performance computing.
Built for quantitative analysts, portfolio managers, algorithmic traders, and financial researchers.

Version: 4.1.0
"""

__version__ = '4.1.0'

import warnings
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import unified API
try:
    from .api import (
        MeridianAlgoAPI,
        get_api,
        get_market_data as api_get_market_data,
        optimize_portfolio as api_optimize_portfolio,
        calculate_risk_metrics as api_calculate_risk_metrics,
        calculate_rsi as api_calculate_rsi,
        calculate_macd as api_calculate_macd,
        price_option
    )
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

# Core functionality that always works
def get_system_info():
    """Get system information."""
    import platform
    return {
        'python_version': sys.version,
        'platform': platform.platform(),
        'package_version': __version__
    }

# Configuration
config = {
    'data_provider': 'yahoo',
    'cache_enabled': True,
    'parallel_processing': True,
    'gpu_acceleration': False,
    'distributed_computing': False
}

def set_config(**kwargs):
    """Set global configuration options."""
    global config
    config.update(kwargs)

def get_config():
    """Get current configuration."""
    return config.copy()

def enable_gpu_acceleration():
    """Enable GPU acceleration if available."""
    config['gpu_acceleration'] = True

def enable_distributed_computing():
    """Enable distributed computing if available."""
    config['distributed_computing'] = True

# Import modules with error handling
try:
    from .statistics import (
        StatisticalArbitrage,
        calculate_value_at_risk,
        calculate_expected_shortfall,
        hurst_exponent,
        calculate_autocorrelation,
        rolling_volatility,
        calculate_metrics as stats_calculate_metrics
    )
    STATISTICS_AVAILABLE = True
except ImportError as e:
    STATISTICS_AVAILABLE = False
    print(f"Statistics module not fully available: {e}")

try:
    from .core import (
        PortfolioOptimizer,
        TimeSeriesAnalyzer,
        get_market_data,
        calculate_max_drawdown
    )
    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    print(f"Core module not fully available: {e}")

try:
    from .technical_indicators import (
        RSI, SMA, EMA, MACD, BollingerBands, Stochastic, WilliamsR,
        ROC, Momentum, ADX, Aroon, ParabolicSAR, Ichimoku,
        ATR, KeltnerChannels, DonchianChannels,
        OBV, ADLine, ChaikinOscillator, MoneyFlowIndex, EaseOfMovement,
        PivotPoints, FibonacciRetracement, SupportResistance
    )
    TECHNICAL_INDICATORS_AVAILABLE = True
except ImportError as e:
    TECHNICAL_INDICATORS_AVAILABLE = False
    print(f"Technical indicators not fully available: {e}")

try:
    from .ml import (
        FeatureEngineer,
        LSTMPredictor,
        prepare_data_for_lstm
    )
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    print(f"ML module not fully available: {e}")

try:
    from .portfolio_management import (
        PortfolioOptimizer as PM_PortfolioOptimizer,
        EfficientFrontier, BlackLitterman, RiskParity
    )
    PORTFOLIO_MANAGEMENT_AVAILABLE = True
except ImportError as e:
    PORTFOLIO_MANAGEMENT_AVAILABLE = False
    print(f"Portfolio management not fully available: {e}")

try:
    from .risk_analysis import (
        VaRCalculator, ExpectedShortfall as Risk_ExpectedShortfall,
        HistoricalVaR, ParametricVaR, MonteCarloVaR
    )
    RISK_ANALYSIS_AVAILABLE = True
except ImportError as e:
    RISK_ANALYSIS_AVAILABLE = False
    print(f"Risk analysis not fully available: {e}")

try:
    from .backtesting import BacktestEngine
    BACKTESTING_AVAILABLE = True
except ImportError as e:
    BACKTESTING_AVAILABLE = False
    print(f"Backtesting not fully available: {e}")

# Simple API class
class MeridianAlgoAPI:
    """Unified API for MeridianAlgo functionality."""
    
    def __init__(self):
        self.available_modules = {
            'statistics': STATISTICS_AVAILABLE,
            'core': CORE_AVAILABLE,
            'technical_indicators': TECHNICAL_INDICATORS_AVAILABLE,
            'ml': ML_AVAILABLE,
            'portfolio_management': PORTFOLIO_MANAGEMENT_AVAILABLE,
            'risk_analysis': RISK_ANALYSIS_AVAILABLE,
            'backtesting': BACKTESTING_AVAILABLE
        }
    
    def get_available_modules(self):
        """Get available modules."""
        return self.available_modules
    
    def get_system_info(self):
        """Get system information."""
        return get_system_info()

# Global API instance
_api_instance = None

def get_api():
    """Get the global API instance."""
    global _api_instance
    if _api_instance is None:
        _api_instance = MeridianAlgoAPI()
    return _api_instance

# Build __all__ dynamically
__all__ = ['__version__', 'get_api', 'MeridianAlgoAPI', 'get_system_info', 
           'config', 'set_config', 'get_config', 'enable_gpu_acceleration', 
           'enable_distributed_computing']

# Add API functions if available
if API_AVAILABLE:
    __all__.extend([
        'api_get_market_data',
        'api_optimize_portfolio', 
        'api_calculate_risk_metrics',
        'api_calculate_rsi',
        'api_calculate_macd',
        'price_option'
    ])

if STATISTICS_AVAILABLE:
    __all__.extend(['StatisticalArbitrage', 'calculate_value_at_risk', 
                   'calculate_expected_shortfall', 'hurst_exponent',
                   'calculate_autocorrelation', 'rolling_volatility', 'calculate_metrics',
                   'stats_calculate_metrics'])

if CORE_AVAILABLE:
    __all__.extend(['PortfolioOptimizer', 'TimeSeriesAnalyzer', 'get_market_data', 'calculate_max_drawdown'])

if TECHNICAL_INDICATORS_AVAILABLE:
    __all__.extend(['RSI', 'SMA', 'EMA', 'MACD', 'BollingerBands', 'Stochastic', 'WilliamsR',
                   'ROC', 'Momentum', 'ADX', 'Aroon', 'ParabolicSAR', 'Ichimoku',
                   'ATR', 'KeltnerChannels', 'DonchianChannels',
                   'OBV', 'ADLine', 'ChaikinOscillator', 'MoneyFlowIndex', 'EaseOfMovement',
                   'PivotPoints', 'FibonacciRetracement', 'SupportResistance'])

if ML_AVAILABLE:
    __all__.extend(['FeatureEngineer', 'LSTMPredictor', 'prepare_data_for_lstm'])

if PORTFOLIO_MANAGEMENT_AVAILABLE:
    __all__.extend(['PM_PortfolioOptimizer', 'EfficientFrontier', 'BlackLitterman', 'RiskParity'])

if RISK_ANALYSIS_AVAILABLE:
    __all__.extend(['VaRCalculator', 'Risk_ExpectedShortfall', 'HistoricalVaR', 'ParametricVaR', 'MonteCarloVaR'])

if BACKTESTING_AVAILABLE:
    __all__.extend(['BacktestEngine'])

# Welcome message
def _show_welcome():
    """Show welcome message on first import."""
    print("🚀 MeridianAlgo v4.0.3 - Quantum Edition")
    print("⚡ Meridian Quant: The Ultimate Quantitative Development Platform")
    print("📊 Ready for institutional-grade quantitative finance!")
    
    try:
        api = get_api()
        available_modules = api.get_available_modules()
        enabled_count = sum(available_modules.values())
        total_count = len(available_modules)
        
        print(f"✅ {enabled_count}/{total_count} modules available")
        
        if enabled_count < total_count:
            print("💡 Some modules may need optional dependencies")
    except:
        print("✅ Core functionality ready")

# Show welcome message on import (can be disabled)
import os
if os.getenv('MERIDIANALGO_QUIET') != '1':
    try:
        _show_welcome()
    except:
        pass  # Silently fail if there are issues