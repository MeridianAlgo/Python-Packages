"""
MeridianAlgo v4.0.0 - Quantum Edition 🚀

Meridian Quant: The Ultimate Quantitative Development Platform

The most advanced Python platform for quantitative finance, integrating cutting-edge 
machine learning, institutional-grade portfolio management, and high-performance computing.
Built for quantitative analysts, portfolio managers, algorithmic traders, and financial researchers.

Version: 4.0.0-quantum
"""

__version__ = '4.0.0-quantum'

# Import unified API
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

# TA Library Integration (if available)
try:
    from .technical_indicators import (
        TAIntegration, add_all_ta_features,
        get_ta_volume_indicators, get_ta_volatility_indicators,
        get_ta_trend_indicators, get_ta_momentum_indicators,
        get_all_ta_indicators
    )
    TA_LIBRARY_AVAILABLE = True
except ImportError:
    TA_LIBRARY_AVAILABLE = False

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

def get_system_info():
    """Get system information and available modules."""
    api = get_api()
    return api.get_system_info()

__all__ = [
    # Unified API (v4.0)
    'MeridianAlgoAPI',
    'get_api',
    'api_get_market_data',
    'api_optimize_portfolio',
    'api_calculate_risk_metrics',
    'api_calculate_rsi',
    'api_calculate_macd',
    'price_option',
    
    # Core (backward compatibility)
    'PortfolioOptimizer',
    'TimeSeriesAnalyzer',
    'get_market_data',
    'calculate_metrics',
    'calculate_max_drawdown',
    
    # Statistics (backward compatibility)
    'StatisticalArbitrage',
    'calculate_value_at_risk',
    'calculate_expected_shortfall',
    'hurst_exponent',
    'calculate_autocorrelation',
    'rolling_volatility',
    
    # ML (backward compatibility)
    'FeatureEngineer',
    'LSTMPredictor',
    'prepare_data_for_lstm',
    
    # Technical Indicators (backward compatibility)
    'RSI', 'Stochastic', 'WilliamsR', 'ROC', 'Momentum',
    'SMA', 'EMA', 'MACD', 'ADX', 'Aroon', 'ParabolicSAR', 'Ichimoku',
    'BollingerBands', 'ATR', 'KeltnerChannels', 'DonchianChannels',
    'OBV', 'ADLine', 'ChaikinOscillator', 'MoneyFlowIndex', 'EaseOfMovement',
    'PivotPoints', 'FibonacciRetracement', 'SupportResistance',
    
    # Portfolio Management (backward compatibility)
    'PM_PortfolioOptimizer', 'EfficientFrontier', 'BlackLitterman', 'RiskParity',
    
    # Risk Analysis (backward compatibility)
    'VaRCalculator', 'Risk_ExpectedShortfall', 'HistoricalVaR', 'ParametricVaR', 'MonteCarloVaR',
    
    # Data Processing (backward compatibility)
    'DataCleaner', 'OutlierDetector', 'MissingDataHandler',
    'DP_FeatureEngineer', 'TechnicalFeatures', 'DataValidator', 'MarketDataProvider',
    
    # Configuration
    'config',
    'set_config',
    'get_config',
    'enable_gpu_acceleration',
    'enable_distributed_computing',
    'get_system_info'
]

# Add TA integration to exports if available
if TA_LIBRARY_AVAILABLE:
    __all__.extend([
        'TAIntegration',
        'add_all_ta_features',
        'get_ta_volume_indicators',
        'get_ta_volatility_indicators', 
        'get_ta_trend_indicators',
        'get_ta_momentum_indicators',
        'get_all_ta_indicators'
    ])

# Welcome message
def _show_welcome():
    """Show welcome message on first import."""
    print("🚀 MeridianAlgo v4.0.0 - Quantum Edition")
    print("⚡ Meridian Quant: The Ultimate Quantitative Development Platform")
    print("📊 Ready for institutional-grade quantitative finance!")
    
    try:
        available_modules = get_api().get_available_modules()
        enabled_count = sum(available_modules.values())
        total_count = len(available_modules)
        
        print(f"✅ {enabled_count}/{total_count} modules available")
        
        if enabled_count < total_count:
            print("💡 Install optional dependencies for full functionality:")
            if not available_modules.get('data'):
                print("   pip install yfinance alpha_vantage")
            if not available_modules.get('machine_learning'):
                print("   pip install torch scikit-learn")
            if not available_modules.get('hpc'):
                print("   pip install dask ray cupy")
    except:
        print("✅ Core functionality ready")

# Show welcome message on import (can be disabled)
import os
if os.getenv('MERIDIANALGO_QUIET') != '1':
    try:
        _show_welcome()
    except:
        pass  # Silently fail if there are issues