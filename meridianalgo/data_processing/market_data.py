"""
Market Data Module

This module provides market data providers and caching.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import yfinance as yf
from datetime import datetime, timedelta

class MarketDataProvider:
    """Base Market Data Provider."""
    
    def __init__(self):
        pass
    
    def get_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get market data for a symbol."""
        raise NotImplementedError

class YahooFinanceProvider(MarketDataProvider):
    """Yahoo Finance Data Provider."""
    
    def __init__(self):
        super().__init__()
    
    def get_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get data from Yahoo Finance."""
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            
            # Handle different data structures from yfinance
            if 'Adj Close' in data.columns:
                return data['Adj Close']
            elif hasattr(data, 'columns') and len(data.columns.levels) > 1:
                return data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
            else:
                return data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

class DataCache:
    """Data Cache for market data."""
    
    def __init__(self, cache_duration: int = 3600):  # 1 hour default
        self.cache = {}
        self.cache_duration = cache_duration
        self.cache_timestamps = {}
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get data from cache."""
        if key in self.cache:
            if datetime.now().timestamp() - self.cache_timestamps[key] < self.cache_duration:
                return self.cache[key]
            else:
                # Cache expired
                del self.cache[key]
                del self.cache_timestamps[key]
        return None
    
    def set(self, key: str, data: pd.DataFrame) -> None:
        """Set data in cache."""
        self.cache[key] = data
        self.cache_timestamps[key] = datetime.now().timestamp()
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.cache_timestamps.clear()
