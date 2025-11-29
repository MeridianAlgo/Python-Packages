"""
MeridianAlgo v5.0.0 - Advanced Quantitative Development Platform

Enterprise-Grade Quantitative Finance for Professional Developers

Features cutting-edge algorithms for:
- Market Microstructure Analysis
- Statistical Arbitrage & Pairs Trading
- Optimal Execution Algorithms
- High-Frequency Trading Strategies
- Multi-Factor Models & Risk Decomposition  
- Regime Detection & Structural Breaks
- Advanced Portfolio Management
- Machine Learning for Trading
- Comprehensive Risk Analysis

Built with love by MeridianAlgo, for quantitative professionals.

Version: 5.0.0 "Advanced Quantitative Development Edition"
"""

__version__ = '5.0.0'

import warnings
import sys
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Suppress import error messages unless debug mode
_DEBUG = os.getenv('MERIDIANALGO_DEBUG', '0') == '1'

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
    """Get system information."""
    import platform
    return {
        'python_version': sys.version,
        'platform': platform.platform(),
        'package_version': __version__
    }

# Import modules with graceful error handling
def _safe_import(module_name, items):
    """Safely import items from a module."""
    available = {}
    try:
        mod = __import__(f'meridianalgo.{module_name}', fromlist=items)
        for item in items:
            try:
                available[item] = getattr(mod, item)
            except AttributeError:
                pass
        return True, available
    except Exception as e:
        if _DEBUG:
            print(f"{module_name} module: {e}")
        return False, {}

# Import API functions
API_AVAILABLE = False
try:
    from .api import (
        get_market_data as api_get_market_data,
        optimize_portfolio as api_optimize_portfolio,
        calculate_risk_metrics as api_calculate_risk_metrics,
    )
    API_AVAILABLE = True
except:
    pass

# Import quant module
QUANT_AVAILABLE = False
try:
    from .quant import (
        # Market Microstructure
        OrderFlowImbalance, VolumeWeightedSpread, RealizedVolatility,
        MarketImpactModel, TickDataAnalyzer,
        # Statistical Arbitrage
        PairsTrading, CointegrationAnalyzer, OrnsteinUhlenbeck,
        MeanReversionTester, SpreadAnalyzer,
        # Execution Algorithms
        VWAP, TWAP, POV, ImplementationShortfall, AdaptiveExecution,
        # High Frequency Trading
        MarketMaking, LatencyArbitrage, LiquidityProvision,
        HFTSignalGenerator, MicropriceEstimator,
        # Factor Models
        FamaFrenchModel, APTModel, CustomFactorModel,
        FactorRiskDecomposition, AlphaCapture,
        # Regime Detection
        HiddenMarkovModel, RegimeSwitchingModel, StructuralBreakDetection,
        MarketStateClassifier, VolatilityRegimeDetector
    )
    QUANT_AVAILABLE = True
except Exception as e:
    if _DEBUG:
        print(f"Quant module: {e}")

# Import technical indicators
TECHNICAL_INDICATORS_AVAILABLE = False
try:
    from .technical_indicators import (
        RSI, SMA, EMA, MACD, BollingerBands, Stochastic, WilliamsR,
        ROC, Momentum, ADX, Aroon, ParabolicSAR,
        ATR, KeltnerChannels, DonchianChannels,
        OBV, ChaikinOscillator, MoneyFlowIndex,
        PivotPoints, FibonacciRetracement
    )
    TECHNICAL_INDICATORS_AVAILABLE = True
except:
    pass

# Other module flags (graceful degradation)
STATISTICS_AVAILABLE = False
CORE_AVAILABLE = False
ML_AVAILABLE = False
PORTFOLIO_MANAGEMENT_AVAILABLE = False
RISK_ANALYSIS_AVAILABLE = False
BACKTESTING_AVAILABLE = False
FOREX_AVAILABLE = False
CRYPTO_AVAILABLE = False
DIVERSIFICATION_AVAILABLE = False
DERIVATIVES_AVAILABLE = False
FIXED_INCOME_AVAILABLE = False
ALGORITHMIC_TRADING_AVAILABLE = False
ADVANCED_RISK_AVAILABLE = False

# Try to import other modules silently
try:
    from . import statistics
    STATISTICS_AVAILABLE = True
except:
    pass

try:
    from . import core
    CORE_AVAILABLE = True
except:
    pass

try:
    from . import ml
    ML_AVAILABLE = True
except:
    pass

try:
    from . import portfolio_management
    PORTFOLIO_MANAGEMENT_AVAILABLE = True
except:
    pass

try:
    from . import risk_analysis
    RISK_ANALYSIS_AVAILABLE = True
except:
    pass

try:
    from . import backtesting
    BACKTESTING_AVAILABLE = True
except:
    pass

try:
    from . import forex
    FOREX_AVAILABLE = True
except:
    pass

try:
    from . import crypto
    CRYPTO_AVAILABLE = True
except:
    pass

try:
    from . import diversification
    DIVERSIFICATION_AVAILABLE = True
except:
    pass

try:
    from . import derivatives
    DERIVATIVES_AVAILABLE = True
except:
    pass

try:
    from . import fixed_income
    FIXED_INCOME_AVAILABLE = True
except:
    pass

try:
    from . import algorithmic_trading
    ALGORITHMIC_TRADING_AVAILABLE = True
except:
    pass

try:
    from . import advanced_risk
    ADVANCED_RISK_AVAILABLE = True
except:
    pass

# Simple API class
class MeridianAlgoAPI:
    """Unified API for MeridianAlgo functionality."""
    
    def __init__(self):
        self.available_modules = {
            'api': API_AVAILABLE,
            'quant': QUANT_AVAILABLE,
            'statistics': STATISTICS_AVAILABLE,
            'core': CORE_AVAILABLE,
            'technical_indicators': TECHNICAL_INDICATORS_AVAILABLE,
            'ml': ML_AVAILABLE,
            'portfolio_management': PORTFOLIO_MANAGEMENT_AVAILABLE,
            'risk_analysis': RISK_ANALYSIS_AVAILABLE,
            'backtesting': BACKTESTING_AVAILABLE,
            'forex': FOREX_AVAILABLE,
            'crypto': CRYPTO_AVAILABLE,
            'diversification': DIVERSIFICATION_AVAILABLE,
            'derivatives': DERIVATIVES_AVAILABLE,
            'fixed_income': FIXED_INCOME_AVAILABLE,
            'algorithmic_trading': ALGORITHMIC_TRADING_AVAILABLE,
            'advanced_risk': ADVANCED_RISK_AVAILABLE,
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

# Build __all__ list
__all__ = [
    '__version__', 'get_api', 'get_system_info', 
    'config', 'set_config', 'get_config',
    'enable_gpu_acceleration', 'enable_distributed_computing'
]

# Add exports for available modules
if API_AVAILABLE:
    __all__.extend(['api_get_market_data', 'api_optimize_portfolio', 'api_calculate_risk_metrics'])

if QUANT_AVAILABLE:
    __all__.extend([
        'OrderFlowImbalance', 'VolumeWeightedSpread', 'RealizedVolatility',
        'MarketImpactModel', 'TickDataAnalyzer',
        'PairsTrading', 'CointegrationAnalyzer', 'OrnsteinUhlenbeck',
        'MeanReversionTester', 'SpreadAnalyzer',
        'VWAP', 'TWAP', 'POV', 'ImplementationShortfall', 'AdaptiveExecution',
        'MarketMaking', 'LatencyArbitrage', 'LiquidityProvision',
        'HFTSignalGenerator', 'MicropriceEstimator',
        'FamaFrenchModel', 'APTModel', 'CustomFactorModel',
        'FactorRiskDecomposition', 'AlphaCapture',
        'HiddenMarkovModel', 'RegimeSwitchingModel', 'StructuralBreakDetection',
        'MarketStateClassifier', 'VolatilityRegimeDetector'
    ])

if TECHNICAL_INDICATORS_AVAILABLE:
    __all__.extend([
        'RSI', 'SMA', 'EMA', 'MACD', 'BollingerBands', 'Stochastic', 'WilliamsR',
        'ROC', 'Momentum', 'ADX', 'Aroon', 'ParabolicSAR',
        'ATR', 'KeltnerChannels', 'DonchianChannels',
        'OBV', 'ChaikinOscillator', 'MoneyFlowIndex',
        'PivotPoints', 'FibonacciRetracement'
    ])

# Welcome message
def _show_welcome():
    """Show welcome message on first import."""
    if os.getenv('MERIDIANALGO_QUIET') == '1':
        return
        
    print("🚀 MeridianAlgo v5.0.0 - Advanced Quantitative Development Platform")
    print("⚡ Enterprise-Grade Quantitative Finance for Professional Developers")
    print("📊 Institutional-grade algorithms for hedge funds & trading firms")
    
    try:
        api = get_api()
        modules = api.get_available_modules()
        enabled = sum(modules.values())
        total = len(modules)
        print(f"✅ {enabled}/{total} modules loaded successfully")
        
        if QUANT_AVAILABLE:
            print("� NEW: Professional quant module ready")
        
    except:
        pass

# Show welcome message on import
try:
    _show_welcome()
except:
    pass