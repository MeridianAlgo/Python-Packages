"""
Core functionality for the MeridianAlgo trading library.

This module includes portfolio optimization, time series analysis, and trading strategies.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional
import yfinance as yf
from datetime import datetime, timedelta
import warnings

class PortfolioOptimizer:
    """Portfolio optimization using modern portfolio theory."""
    
    def __init__(self, returns: pd.DataFrame):
        """
        Initialize the portfolio optimizer with historical returns.
        
        Args:
            returns: DataFrame containing historical returns (tickers as columns)
        """
        self.returns = returns
        self.cov_matrix = self._calculate_covariance_matrix()
    
    def _calculate_covariance_matrix(self) -> pd.DataFrame:
        """Calculate the covariance matrix of returns."""
        return self.returns.cov()
    
    def calculate_efficient_frontier(self, risk_free_rate: float = 0.0, 
                                  num_portfolios: int = 1000) -> Dict[str, np.ndarray]:
        """
        Calculate the efficient frontier using Monte Carlo simulation.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 0.0)
            num_portfolios: Number of random portfolios to generate (default: 1000)
            
        Returns:
            Dictionary containing portfolio weights, returns, and volatilities
        """
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        for i in range(num_portfolios):
            weights = np.random.random(len(self.returns.columns))
            weights /= np.sum(weights)
            weights_record.append(weights)
            
            portfolio_return = np.sum(self.returns.mean() * weights) * 252
            portfolio_std_dev = np.sqrt(
                np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
            
            results[0, i] = portfolio_std_dev
            results[1, i] = portfolio_return
            results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev  # Sharpe ratio
        
        return {
            'volatility': results[0],
            'returns': results[1],
            'sharpe': results[2],
            'weights': np.array(weights_record)
        }


class TimeSeriesAnalyzer:
    """Time series analysis for financial data."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with time series data.
        
        Args:
            data: DataFrame with datetime index and price/return data
        """
        self.data = data
    
    def calculate_returns(self, log_returns: bool = False) -> pd.Series:
        """
        Calculate returns from price data.
        
        Args:
            log_returns: If True, calculate log returns (default: False)
            
        Returns:
            Series of returns
        """
        if log_returns:
            return np.log(self.data / self.data.shift(1)).dropna()
        return self.data.pct_change().dropna()
    
    def calculate_volatility(self, window: int = 21, annualized: bool = True) -> pd.Series:
        """
        Calculate rolling volatility.
        
        Args:
            window: Rolling window size in periods (default: 21 for 1 month)
            annualized: If True, annualize the volatility (default: True)
            
        Returns:
            Series of volatility values
        """
        returns = self.calculate_returns()
        vol = returns.rolling(window=window).std()
        
        if annualized:
            return vol * np.sqrt(252)  # 252 trading days in a year
        return vol


def get_market_data(tickers: List[str], start_date: str = '2020-01-01', 
                   end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch market data from Yahoo Finance.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format (default: '2020-01-01')
        end_date: End date in 'YYYY-MM-DD' format (default: today)
        
    Returns:
        DataFrame with adjusted close prices for the given tickers
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data


def calculate_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> Dict[str, float]:
    """
    Calculate key performance metrics for a return series.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        
    Returns:
        Dictionary of performance metrics
    """
    if len(returns) < 2:
        raise ValueError("At least 2 data points are required for metrics calculation")
    
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': calculate_max_drawdown(returns)
    }


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown from a return series.
    
    Args:
        returns: Series of returns
        
    Returns:
        Maximum drawdown as a decimal
    """
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdowns = (cumulative - rolling_max) / rolling_max
    return drawdowns.min()
