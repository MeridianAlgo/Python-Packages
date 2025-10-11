"""
Unified API for MeridianAlgo - Ultimate Quantitative Development Platform.

This module provides a consistent, high-level interface to all MeridianAlgo functionality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
import warnings

# Core imports
try:
    from .core import PortfolioOptimizer, TimeSeriesAnalyzer, get_market_data
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

# Statistics imports
try:
    from .statistics import StatisticalArbitrage, calculate_metrics, calculate_value_at_risk, calculate_expected_shortfall
    STATISTICS_AVAILABLE = True
except ImportError:
    STATISTICS_AVAILABLE = False

# Technical indicators imports
try:
    from .technical_indicators import RSI, SMA, EMA, MACD, BollingerBands
    TECHNICAL_AVAILABLE = True
except ImportError:
    TECHNICAL_AVAILABLE = False

# ML imports
try:
    from .ml import FeatureEngineer, LSTMPredictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Portfolio management imports
try:
    from .portfolio_management import PortfolioOptimizer as PM_PortfolioOptimizer
    PORTFOLIO_AVAILABLE = True
except ImportError:
    PORTFOLIO_AVAILABLE = False

# Risk analysis imports
try:
    from .risk_analysis import VaRCalculator
    RISK_AVAILABLE = True
except ImportError:
    RISK_AVAILABLE = False

class MeridianAlgoAPI:
    """Unified API for all MeridianAlgo functionality."""
    
    def __init__(self):
        """Initialize the API with available modules."""
        self.available_modules = {
            'core': CORE_AVAILABLE,
            'statistics': STATISTICS_AVAILABLE,
            'technical_indicators': TECHNICAL_AVAILABLE,
            'machine_learning': ML_AVAILABLE,
            'portfolio_management': PORTFOLIO_AVAILABLE,
            'risk_analysis': RISK_AVAILABLE
        }
    
    def get_available_modules(self) -> Dict[str, bool]:
        """Get dictionary of available modules."""
        return self.available_modules.copy()
    
    def get_market_data(self, symbols: List[str], start_date: str, end_date: str = None) -> pd.DataFrame:
        """Get market data for specified symbols."""
        if not CORE_AVAILABLE:
            raise ImportError("Core module not available")
        return get_market_data(symbols, start_date, end_date)
    
    def optimize_portfolio(self, returns: pd.DataFrame, method: str = 'sharpe', **kwargs) -> Dict[str, float]:
        """Optimize portfolio using specified method."""
        if not PORTFOLIO_AVAILABLE and not CORE_AVAILABLE:
            raise ImportError("Portfolio optimization not available")
        
        try:
            if PORTFOLIO_AVAILABLE:
                optimizer = PM_PortfolioOptimizer(returns)
            else:
                optimizer = PortfolioOptimizer(returns)
            return optimizer.optimize_portfolio(method=method, **kwargs)
        except Exception as e:
            # Fallback to equal weights
            n_assets = len(returns.columns)
            return {asset: 1.0/n_assets for asset in returns.columns}
    
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        if not STATISTICS_AVAILABLE:
            raise ImportError("Statistics module not available")
        return calculate_metrics(returns)
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        if not TECHNICAL_AVAILABLE:
            raise ImportError("Technical indicators not available")
        return RSI(prices, period)
    
    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        if not TECHNICAL_AVAILABLE:
            raise ImportError("Technical indicators not available")
        return MACD(prices)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and capabilities."""
        import platform
        import sys
        
        return {
            'python_version': sys.version,
            'platform': platform.platform(),
            'available_modules': self.available_modules,
            'package_version': '4.0.2'
        }

# Global API instance
_api_instance = None

def get_api() -> MeridianAlgoAPI:
    """Get the global API instance."""
    global _api_instance
    if _api_instance is None:
        _api_instance = MeridianAlgoAPI()
    return _api_instance

# Convenience functions
def get_market_data(symbols: List[str], start_date: str, end_date: str = None) -> pd.DataFrame:
    """Get market data for specified symbols."""
    return get_api().get_market_data(symbols, start_date, end_date)

def optimize_portfolio(returns: pd.DataFrame, method: str = 'sharpe', **kwargs) -> Dict[str, float]:
    """Optimize portfolio using specified method."""
    return get_api().optimize_portfolio(returns, method, **kwargs)

def calculate_risk_metrics(returns: pd.Series) -> Dict[str, float]:
    """Calculate comprehensive risk metrics."""
    return get_api().calculate_risk_metrics(returns)

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    return get_api().calculate_rsi(prices, period)

def calculate_macd(prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD indicator."""
    return get_api().calculate_macd(prices)

def price_option(spot: float, strike: float, time_to_expiry: float, 
                risk_free_rate: float, volatility: float, option_type: str = 'call') -> float:
    """Price an option using Black-Scholes model."""
    from scipy.stats import norm
    import math
    
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
    d2 = d1 - volatility * math.sqrt(time_to_expiry)
    
    if option_type.lower() == 'call':
        price = spot * norm.cdf(d1) - strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
    else:  # put
        price = strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    
    return price