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