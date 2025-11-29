"""
Statistical analysis module.

This module provides tools for statistical analysis of financial time series.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import statsmodels.api as sm

class StatisticalArbitrage:
    """Statistical arbitrage strategy implementation."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize with price data.
        
        Args:
            data: DataFrame with price data (tickers as columns)
        """
        self.data = data
    
    def calculate_zscore(self, window: int = 21) -> pd.Series:
        """Calculate z-score of the price series."""
        rolling_mean = self.data.rolling(window=window).mean()
        rolling_std = self.data.rolling(window=window).std()
        return (self.data - rolling_mean) / rolling_std
    
    def calculate_cointegration(self, x: pd.Series, y: pd.Series) -> Dict[str, float]:
        """Test for cointegration between two time series."""
        from statsmodels.tsa.stattools import coint
        
        df = pd.DataFrame({'x': x, 'y': y}).dropna()
        score, pvalue, _ = coint(df['x'], df['y'])
        
        return {
            'score': score,
            'pvalue': pvalue,
            'is_cointegrated': pvalue < 0.05
        }

def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix for returns."""
    return returns.corr()

def calculate_rolling_correlation(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Calculate rolling correlation between assets."""
    return returns.rolling(window=window).corr()

def calculate_hurst_exponent(time_series: pd.Series, max_lag: int = 20) -> float:
    """Calculate the Hurst exponent of a time series."""
    lags = range(2, max_lag + 1)
    tau = [np.std(np.subtract(time_series[lag:].values, time_series[:-lag].values)) 
           for lag in lags]
    return np.polyfit(np.log(lags), np.log(tau), 1)[0]

def calculate_half_life(price_series: pd.Series) -> float:
    """Calculate the half-life of a mean-reverting time series."""
    delta_p = price_series.diff().dropna()
    lag_p = price_series.shift(1).dropna()
    
    if len(delta_p) != len(lag_p):
        min_len = min(len(delta_p), len(lag_p))
        delta_p = delta_p.iloc[-min_len:]
        lag_p = lag_p.iloc[-min_len:]
    
    X = sm.add_constant(lag_p)
    model = sm.OLS(delta_p, X)
    results = model.fit()
    return -np.log(2) / results.params[1]
