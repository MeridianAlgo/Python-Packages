"""
Statistical analysis module for financial time series data.
Includes methods for risk metrics, correlation analysis, and statistical tests.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Tuple, Dict, List, Optional
import warnings

class StatisticalArbitrage:
    """Statistical arbitrage strategies and cointegration analysis."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with price data.
        
        Args:
            data: DataFrame with datetime index and price data
        """
        self.data = data
    
    def calculate_rolling_correlation(self, window: int = 21) -> pd.DataFrame:
        """
        Calculate rolling correlation between all pairs of assets.
        
        Args:
            window: Rolling window size in periods (default: 21 for 1 month)
            
        Returns:
            DataFrame with rolling correlation coefficients
        """
        return self.data.pct_change().rolling(window=window).corr()
    
    def test_cointegration(self, x: Union[pd.Series, np.ndarray], 
                         y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Test for cointegration between two time series using the Engle-Granger test.
        
        Args:
            x: First time series
            y: Second time series
            
        Returns:
            Dictionary with test results including test statistic and p-value
        """
        if len(x) != len(y):
            raise ValueError("Input series must have the same length")
            
        if isinstance(x, pd.Series):
            x = x.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Perform Engle-Granger test
        from statsmodels.tsa.stattools import coint
        try:
            score, pvalue, _ = coint(x, y)
            return {
                'test_statistic': score,
                'p_value': pvalue,
                'is_cointegrated': pvalue < 0.05
            }
        except ImportError:
            raise ImportError("statsmodels is required for cointegration testing")

def calculate_value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) using historical simulation.
    
    Args:
        returns: Series of returns
        confidence_level: Confidence level for VaR (default: 0.95)
        
    Returns:
        Value at Risk as a decimal
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    return np.percentile(returns, (1 - confidence_level) * 100)

def calculate_expected_shortfall(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Expected Shortfall (CVaR) using historical simulation.
    
    Args:
        returns: Series of returns
        confidence_level: Confidence level for ES (default: 0.95)
        
    Returns:
        Expected Shortfall as a decimal
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    var = calculate_value_at_risk(returns, confidence_level)
    return returns[returns <= var].mean()

def hurst_exponent(time_series: Union[pd.Series, np.ndarray], 
                  max_lag: int = 20) -> float:
    """
    Calculate the Hurst exponent using R/S analysis.
    
    Args:
        time_series: Time series data
        max_lag: Maximum lag to use in the calculation (default: 20)
        
    Returns:
        Hurst exponent value
    """
    if isinstance(time_series, pd.Series):
        time_series = time_series.values
        
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    m = np.polyfit(np.log(lags), np.log(tau), 1)
    return m[0]

def calculate_autocorrelation(series: pd.Series, lag: int = 1) -> float:
    """
    Calculate the autocorrelation of a time series.
    
    Args:
        series: Time series data
        lag: Lag for autocorrelation calculation (default: 1)
        
    Returns:
        Autocorrelation coefficient
    """
    return series.autocorr(lag=lag)

def rolling_volatility(returns: pd.Series, window: int = 21, 
                     annualized: bool = True) -> pd.Series:
    """
    Calculate rolling volatility of returns.
    
    Args:
        returns: Series of returns
        window: Rolling window size in periods (default: 21 for 1 month)
        annualized: If True, annualize the volatility (default: True)
        
    Returns:
        Series of rolling volatility values
    """
    vol = returns.rolling(window=window).std()
    if annualized:
        vol = vol * np.sqrt(252)  # Assuming daily data
    return vol

def calculate_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics for a return series.
    
    Args:
        returns: Series of returns
        
    Returns:
        Dictionary containing various performance metrics
    """
    if len(returns) == 0:
        return {}
    
    # Basic statistics
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    
    # Risk-adjusted metrics
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    # Downside metrics
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Skewness and kurtosis
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    # VaR and ES
    var_95 = calculate_value_at_risk(returns, 0.95)
    es_95 = calculate_expected_shortfall(returns, 0.95)
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'var_95': var_95,
        'expected_shortfall_95': es_95,
        'win_rate': (returns > 0).mean(),
        'avg_win': returns[returns > 0].mean() if (returns > 0).any() else 0,
        'avg_loss': returns[returns < 0].mean() if (returns < 0).any() else 0,
        'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if (returns < 0).any() else float('inf')
    }