"""
MeridianAlgo v6.0.0 - The Complete Quantitative Finance Platform

A comprehensive, institutional-grade Python library for quantitative finance
covering everything from trading research to portfolio analytics to derivatives.

Modules:
--------
- analytics: Portfolio analytics, performance attribution, tear sheets (pyfolio-style)
- backtesting: Event-driven backtesting engine with realistic execution
- data: Data acquisition, cleaning, and management from multiple sources
- derivatives: Options pricing, Greeks, volatility surfaces, exotic derivatives
- execution: Optimal execution algorithms (VWAP, TWAP, IS, POV)
- factors: Factor modeling, alpha research, and risk decomposition
- fixed_income: Bond pricing, yield curves, duration/convexity, credit analysis
- liquidity: Market microstructure, order book analysis, liquidity metrics
- ml: Machine learning models for trading and prediction
- portfolio: Portfolio optimization, risk parity, Black-Litterman
- quant: Statistical arbitrage, pairs trading, mean reversion
- risk: VaR, CVaR, stress testing, scenario analysis
- sentiment: Alternative data, news sentiment, social media signals
- signals: Technical indicators and signal generation
- strategies: Pre-built trading strategies and templates

Enterprise Features:
-------------------
- GPU acceleration support
- Distributed computing ready
- Real-time data streaming
- Cloud deployment compatible
- Comprehensive logging and monitoring

Built by MeridianAlgo for quantitative professionals.

Version: 6.0.0 "Institutional Edition"
"""

__version__ = '6.0.0'
__author__ = 'Meridian Algorithmic Research Team'
__email__ = 'support@meridianalgo.com'
__license__ = 'MIT'

import warnings
import sys
import os
import logging
from typing import Optional, Dict, Any, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('meridianalgo')

# Suppress warnings for cleaner output (can be enabled via config)
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

class Config:
    """Global configuration for MeridianAlgo."""
    
    _instance = None
    _config: Dict[str, Any] = {
        # Data settings
        'data_provider': 'yahoo',
        'cache_enabled': True,
        'cache_dir': os.path.expanduser('~/.meridianalgo/cache'),
        
        # Computation settings
        'parallel_processing': True,
        'n_jobs': -1,  # Use all available cores
        'gpu_acceleration': False,
        'gpu_device': 0,
        'distributed_computing': False,
        
        # Risk settings
        'confidence_level': 0.95,
        'risk_free_rate': 0.05,
        'trading_days_per_year': 252,
        
        # Execution settings
        'default_slippage_bps': 5,
        'default_commission_bps': 10,
        
        # Display settings
        'quiet_mode': os.getenv('MERIDIANALGO_QUIET', '0') == '1',
        'debug_mode': os.getenv('MERIDIANALGO_DEBUG', '0') == '1',
        
        # API keys (from environment)
        'alpha_vantage_key': os.getenv('ALPHA_VANTAGE_KEY'),
        'polygon_key': os.getenv('POLYGON_KEY'),
        'quandl_key': os.getenv('QUANDL_KEY'),
    }
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return cls._config.get(key, default)
    
    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Set a configuration value."""
        cls._config[key] = value
    
    @classmethod
    def update(cls, **kwargs) -> None:
        """Update multiple configuration values."""
        cls._config.update(kwargs)
    
    @classmethod
    def reset(cls) -> None:
        """Reset configuration to defaults."""
        cls._config = {
            'data_provider': 'yahoo',
            'cache_enabled': True,
            'parallel_processing': True,
        }
    
    @classmethod
    def all(cls) -> Dict[str, Any]:
        """Get all configuration values."""
        return cls._config.copy()


# Global config instance
config = Config()


def set_config(**kwargs) -> None:
    """Set global configuration options."""
    Config.update(**kwargs)


def get_config() -> Dict[str, Any]:
    """Get current configuration."""
    return Config.all()


# ============================================================================
# GPU & DISTRIBUTED COMPUTING
# ============================================================================

def enable_gpu_acceleration(device: int = 0) -> bool:
    """
    Enable GPU acceleration if available.
    
    Args:
        device: GPU device ID to use
        
    Returns:
        True if GPU is available and enabled, False otherwise
    """
    try:
        import torch
        if torch.cuda.is_available():
            Config.set('gpu_acceleration', True)
            Config.set('gpu_device', device)
            logger.info(f"GPU acceleration enabled on device {device}")
            return True
        else:
            logger.warning("CUDA not available, GPU acceleration disabled")
            return False
    except ImportError:
        logger.warning("PyTorch not installed, GPU acceleration not available")
        return False


def enable_distributed_computing(backend: str = 'ray') -> bool:
    """
    Enable distributed computing.
    
    Args:
        backend: Distributed computing backend ('ray', 'dask', 'spark')
        
    Returns:
        True if enabled successfully
    """
    Config.set('distributed_computing', True)
    Config.set('distributed_backend', backend)
    logger.info(f"Distributed computing enabled with {backend} backend")
    return True


# ============================================================================
# SYSTEM INFORMATION
# ============================================================================

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    import platform
    
    info = {
        'package_version': __version__,
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'machine': platform.machine(),
    }
    
    # Check optional dependencies
    info['optional_packages'] = {}
    
    optional_packages = [
        'torch', 'tensorflow', 'ray', 'dask', 
        'cvxpy', 'statsmodels', 'arch', 'hmmlearn'
    ]
    
    for pkg in optional_packages:
        try:
            mod = __import__(pkg)
            info['optional_packages'][pkg] = getattr(mod, '__version__', 'installed')
        except ImportError:
            info['optional_packages'][pkg] = None
    
    # GPU info
    try:
        import torch
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
    except ImportError:
        info['cuda_available'] = False
    
    return info


# ============================================================================
# MODULE AVAILABILITY TRACKING
# ============================================================================

class ModuleRegistry:
    """Tracks available modules and their status."""
    
    _modules: Dict[str, bool] = {}
    _errors: Dict[str, str] = {}
    
    @classmethod
    def register(cls, name: str, available: bool, error: str = None):
        """Register a module's availability."""
        cls._modules[name] = available
        if error:
            cls._errors[name] = error
    
    @classmethod
    def is_available(cls, name: str) -> bool:
        """Check if a module is available."""
        return cls._modules.get(name, False)
    
    @classmethod
    def get_available(cls) -> List[str]:
        """Get list of available modules."""
        return [name for name, avail in cls._modules.items() if avail]
    
    @classmethod
    def get_unavailable(cls) -> Dict[str, str]:
        """Get unavailable modules with their errors."""
        return {name: cls._errors.get(name, 'Unknown error') 
                for name, avail in cls._modules.items() if not avail}
    
    @classmethod
    def summary(cls) -> Dict[str, Any]:
        """Get module availability summary."""
        return {
            'total': len(cls._modules),
            'available': sum(cls._modules.values()),
            'modules': cls._modules.copy()
        }


# ============================================================================
# LAZY MODULE LOADING
# ============================================================================

def _lazy_import(module_name: str, items: List[str] = None):
    """Lazily import a module or specific items from it."""
    try:
        full_name = f'meridianalgo.{module_name}'
        mod = __import__(full_name, fromlist=items or [module_name.split('.')[-1]])
        ModuleRegistry.register(module_name, True)
        return mod
    except Exception as e:
        ModuleRegistry.register(module_name, False, str(e))
        if Config.get('debug_mode'):
            logger.warning(f"Failed to import {module_name}: {e}")
        return None


# ============================================================================
# CORE IMPORTS - Essential functionality always available
# ============================================================================
# Core functionality
try:
    from .core import (
        PortfolioOptimizer,
        TimeSeriesAnalyzer,
        calculate_metrics,
        calculate_max_drawdown,
        calculate_value_at_risk,
        calculate_expected_shortfall,
        calculate_sortino_ratio,
        calculate_calmar_ratio,
        StatisticalArbitrage,
        calculate_correlation_matrix,
        calculate_rolling_correlation,
        calculate_hurst_exponent,
        calculate_half_life,
        calculate_autocorrelation,
        hurst_exponent,
        rolling_volatility,
        get_market_data
    )
    ModuleRegistry.register('core', True)
except ImportError as e:
    ModuleRegistry.register('core', False, str(e))

# Analytics (pyfolio-style)
try:
    from .analytics import (
        TearSheet, PerformanceAnalyzer, RiskAnalyzer,
        create_full_tear_sheet, create_returns_tear_sheet,
        create_position_tear_sheet, create_round_trip_tear_sheet
    )
    ModuleRegistry.register('analytics', True)
except ImportError as e:
    ModuleRegistry.register('analytics', False, str(e))

# Portfolio optimization
try:
    from .portfolio import (
        EfficientFrontier, RiskParity,
        BlackLitterman, HierarchicalRiskParity,
        MeanVarianceOptimizer, MinimumVariancePortfolio,
        MaxSharpePortfolio, MaxDiversificationPortfolio
    )
    ModuleRegistry.register('portfolio', True)
except ImportError as e:
    ModuleRegistry.register('portfolio', False, str(e))

# Risk management
try:
    from .risk import (
        VaRCalculator, CVaRCalculator, StressTest,
        ScenarioAnalyzer, DrawdownAnalyzer,
        RiskMetrics, TailRiskAnalyzer
    )
    ModuleRegistry.register('risk', True)
except ImportError as e:
    ModuleRegistry.register('risk', False, str(e))

# Data management
try:
    from .data import (
        DataManager, MarketData, get_market_data,
        DataCleaner, DataAggregator
    )
    ModuleRegistry.register('data', True)
except ImportError as e:
    ModuleRegistry.register('data', False, str(e))

# Derivatives
try:
    from .derivatives import (
        OptionsPricer, VolatilitySurface, GreeksCalculator,
        BlackScholes, BinomialTree, MonteCarloPricer,
        ImpliedVolatility, OptionChain
    )
    ModuleRegistry.register('derivatives', True)
except ImportError as e:
    ModuleRegistry.register('derivatives', False, str(e))

# Execution algorithms
try:
    from .execution import (
        VWAP, TWAP, POV, ImplementationShortfall,
        AdaptiveExecution, ExecutionAnalyzer
    )
    ModuleRegistry.register('execution', True)
except ImportError as e:
    ModuleRegistry.register('execution', False, str(e))

# Factor models
try:
    from .factors import (
        FamaFrench, FactorModel, AlphaModel,
        FactorRiskDecomposition, StyleAnalysis
    )
    ModuleRegistry.register('factors', True)
except ImportError as e:
    ModuleRegistry.register('factors', False, str(e))

# Liquidity analysis
try:
    from .liquidity import (
        LiquidityAnalyzer, OrderBookAnalyzer, MarketImpact,
        SpreadAnalyzer, VolumeProfile, VPIN
    )
    ModuleRegistry.register('liquidity', True)
except ImportError as e:
    ModuleRegistry.register('liquidity', False, str(e))

# Backtesting
try:
    from .backtesting import (
        Backtest, Strategy, Portfolio as BtPortfolio,
        EventDrivenBacktest, VectorizedBacktest
    )
    ModuleRegistry.register('backtesting', True)
except ImportError as e:
    ModuleRegistry.register('backtesting', False, str(e))

# Quant strategies
try:
    from .quant import (
        PairsTrading, CointegrationAnalyzer, OrnsteinUhlenbeck,
        StatisticalArbitrage, MeanReversionStrategy,
        HiddenMarkovModel, RegimeDetector
    )
    ModuleRegistry.register('quant', True)
except ImportError as e:
    ModuleRegistry.register('quant', False, str(e))

# Technical signals
try:
    from .signals import (
        RSI, MACD, BollingerBands, SMA, EMA,
        ATR, Stochastic, WilliamsR, ADX,
        SignalGenerator, TechnicalAnalyzer
    )
    ModuleRegistry.register('signals', True)
except ImportError as e:
    ModuleRegistry.register('signals', False, str(e))

# Machine learning
try:
    from .ml import (
        FeatureEngineer, LSTMPredictor, ModelSelector,
        TimeSeriesCV, WalkForwardOptimizer, prepare_data_for_lstm
    )
    ModuleRegistry.register('ml', True)
except ImportError as e:
    ModuleRegistry.register('ml', False, str(e))

# Fixed income
try:
    from .fixed_income import (
        BondPricer, YieldCurve, DurationCalculator,
        CreditAnalyzer, SwapPricer
    )
    ModuleRegistry.register('fixed_income', True)
except ImportError as e:
    ModuleRegistry.register('fixed_income', False, str(e))


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_analysis(
    prices,
    benchmark=None,
    risk_free_rate: float = 0.05,
    show_plots: bool = True
) -> Dict[str, Any]:
    """
    Quick analysis of a price series or returns.
    
    Args:
        prices: Price series or DataFrame
        benchmark: Optional benchmark for comparison
        risk_free_rate: Risk-free rate for calculations
        show_plots: Whether to display plots
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    import pandas as pd
    import numpy as np
    
    # Convert to returns if prices
    if isinstance(prices, pd.Series):
        if prices.mean() > 1:  # Likely prices, not returns
            returns = prices.pct_change().dropna()
        else:
            returns = prices
    else:
        returns = pd.Series(prices).pct_change().dropna()
    
    # Calculate metrics
    total_return = (1 + returns).prod() - 1
    ann_return = (1 + total_return) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0
    
    # Drawdown analysis
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    results = {
        'total_return': total_return,
        'annualized_return': ann_return,
        'annualized_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'calmar_ratio': ann_return / abs(max_drawdown) if max_drawdown != 0 else 0,
        'sortino_ratio': ann_return / (returns[returns < 0].std() * np.sqrt(252)),
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis(),
        'var_95': returns.quantile(0.05),
        'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),
        'best_day': returns.max(),
        'worst_day': returns.min(),
        'win_rate': (returns > 0).mean(),
        'num_observations': len(returns)
    }
    
    return results


def get_market_data_quick(
    symbols: Union[str, List[str]],
    start: str = None,
    end: str = None,
    interval: str = '1d'
):
    """
    Quick function to get market data.
    
    Args:
        symbols: Single symbol or list of symbols
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        interval: Data interval ('1d', '1h', '5m', etc.)
        
    Returns:
        DataFrame with market data
    """
    from datetime import datetime, timedelta
    import yfinance as yf
    
    if isinstance(symbols, str):
        symbols = [symbols]
    
    if start is None:
        start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    
    data = yf.download(symbols, start=start, end=end, interval=interval, progress=False)
    
    if len(symbols) == 1:
        return data
    
    return data


# ============================================================================
# API CLASS
# ============================================================================

class MeridianAlgoAPI:
    """
    Unified API for MeridianAlgo functionality.
    
    Provides a single entry point to all package functionality with
    convenient methods for common operations.
    """
    
    def __init__(self):
        self._cache = {}
    
    @property
    def available_modules(self) -> Dict[str, bool]:
        """Get available modules."""
        return ModuleRegistry.summary()['modules']
    
    @property
    def version(self) -> str:
        """Get package version."""
        return __version__
    
    def get_data(self, symbols, start=None, end=None, **kwargs):
        """Get market data."""
        return get_market_data_quick(symbols, start, end, **kwargs)
    
    def analyze(self, data, **kwargs):
        """Perform quick analysis."""
        return quick_analysis(data, **kwargs)
    
    def optimize_portfolio(self, returns, method='sharpe', **kwargs):
        """Optimize portfolio."""
        if ModuleRegistry.is_available('portfolio'):
            opt = PortfolioOptimizer(returns, **kwargs)
            return opt.optimize(method=method)
        raise ImportError("Portfolio module not available")
    
    def calculate_risk(self, returns, **kwargs):
        """Calculate risk metrics."""
        return quick_analysis(returns, **kwargs)
    
    def price_option(self, S, K, T, r, sigma, option_type='call', **kwargs):
        """Price an option using Black-Scholes."""
        if ModuleRegistry.is_available('derivatives'):
            pricer = OptionsPricer()
            return pricer.black_scholes(S, K, T, r, sigma, option_type, **kwargs)
        raise ImportError("Derivatives module not available")
    
    def get_system_info(self):
        """Get system information."""
        return get_system_info()


# Global API instance
_api = None

def get_api() -> MeridianAlgoAPI:
    """Get the global API instance."""
    global _api
    if _api is None:
        _api = MeridianAlgoAPI()
    return _api


# ============================================================================
# WELCOME MESSAGE
# ============================================================================

def _show_welcome():
    """Show welcome message on import."""
    if Config.get('quiet_mode'):
        return
    
    summary = ModuleRegistry.summary()
    available = summary['available']
    total = summary['total']
    
    print(f"🚀 MeridianAlgo v{__version__} - The Complete Quantitative Finance Platform")
    print(f"⚡ Institutional-grade algorithms for professional quants")
    print(f"📊 {available}/{total} modules loaded successfully")
    
    if Config.get('debug_mode'):
        unavailable = ModuleRegistry.get_unavailable()
        if unavailable:
            print(f"⚠️  Unavailable modules: {', '.join(unavailable.keys())}")


# Show welcome on import
try:
    _show_welcome()
except Exception:
    pass


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Version info
    '__version__', '__author__', '__email__',
    
    # Configuration
    'Config', 'config', 'set_config', 'get_config',
    'enable_gpu_acceleration', 'enable_distributed_computing',
    
    # System
    'get_system_info', 'ModuleRegistry',
    
    # API
    'MeridianAlgoAPI', 'get_api',
    
    # Convenience functions
    'quick_analysis', 'get_market_data_quick',
    
    # Core
    'TimeSeriesAnalyzer', 'calculate_metrics', 'calculate_max_drawdown',
    'calculate_value_at_risk', 'calculate_expected_shortfall',
    'calculate_sortino_ratio', 'calculate_calmar_ratio',
    'calculate_correlation_matrix', 'calculate_rolling_correlation',
    'calculate_hurst_exponent', 'calculate_half_life',
    'calculate_autocorrelation', 'hurst_exponent', 'rolling_volatility',
    'StatisticalArbitrage',
    
    # Analytics
    'TearSheet', 'PerformanceAnalyzer', 'RiskAnalyzer',
    'create_full_tear_sheet', 'create_returns_tear_sheet',
    
    # Portfolio
    'PortfolioOptimizer', 'EfficientFrontier', 'RiskParity',
    'BlackLitterman', 'HierarchicalRiskParity',
    
    # Risk
    'VaRCalculator', 'CVaRCalculator', 'StressTest',
    'ScenarioAnalyzer', 'DrawdownAnalyzer',
    
    # Data
    'DataManager', 'MarketData', 'get_market_data',
    
    # Derivatives
    'OptionsPricer', 'VolatilitySurface', 'GreeksCalculator',
    'BlackScholes', 'ImpliedVolatility',
    
    # Execution
    'VWAP', 'TWAP', 'POV', 'ImplementationShortfall',
    
    # Factors
    'FamaFrench', 'FactorModel', 'AlphaModel',
    
    # Liquidity
    'LiquidityAnalyzer', 'OrderBookAnalyzer', 'MarketImpact',
    
    # Backtesting
    'Backtest', 'Strategy',
    
    # Quant
    'PairsTrading', 'CointegrationAnalyzer', 'OrnsteinUhlenbeck',
    'HiddenMarkovModel', 'RegimeDetector',
    
    # Signals
    'RSI', 'MACD', 'BollingerBands', 'SMA', 'EMA',
    
    # ML
    'FeatureEngineer', 'LSTMPredictor', 'ModelSelector',
    'TimeSeriesCV', 'WalkForwardOptimizer', 'prepare_data_for_lstm',
    
    # Fixed Income
    'BondPricer', 'YieldCurve',
]