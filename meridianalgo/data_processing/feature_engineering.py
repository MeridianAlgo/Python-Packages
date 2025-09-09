"""
Feature Engineering Module

This module provides feature engineering utilities for financial data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

class FeatureEngineer:
    """Financial Feature Engineer."""
    
    def __init__(self, lookback: int = 10):
        self.lookback = lookback
    
    def create_features(self, data: pd.Series) -> pd.DataFrame:
        """Create technical features from price data."""
        df = pd.DataFrame(index=data.index)
        
        # Basic features
        df['returns'] = data.pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        
        # Momentum features
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = data.pct_change(periods=period)
        
        # Volatility features
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'ma_{period}'] = data.rolling(period).mean()
            df[f'ma_ratio_{period}'] = data / df[f'ma_{period}'] - 1
        
        # Drop NaN values
        df = df.dropna()
        
        return df

class TechnicalFeatures:
    """Technical Features Creator."""
    
    def __init__(self):
        pass
    
    def create_technical_features(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """Create technical features from OHLCV data."""
        features = pd.DataFrame(index=ohlcv_data.index)
        
        # Price features
        features['close'] = ohlcv_data['Close']
        features['high'] = ohlcv_data['High']
        features['low'] = ohlcv_data['Low']
        features['volume'] = ohlcv_data['Volume']
        
        # Price ratios
        features['hl_ratio'] = ohlcv_data['High'] / ohlcv_data['Low']
        features['co_ratio'] = ohlcv_data['Close'] / ohlcv_data['Open']
        
        # Volume features
        features['volume_ma'] = ohlcv_data['Volume'].rolling(20).mean()
        features['volume_ratio'] = ohlcv_data['Volume'] / features['volume_ma']
        
        return features.dropna()

class FundamentalFeatures:
    """Fundamental Features Creator."""
    
    def __init__(self):
        pass
    
    def create_fundamental_features(self, fundamental_data: pd.DataFrame) -> pd.DataFrame:
        """Create fundamental features."""
        features = pd.DataFrame(index=fundamental_data.index)
        
        # Financial ratios
        if 'Revenue' in fundamental_data.columns and 'Market_Cap' in fundamental_data.columns:
            features['price_to_sales'] = fundamental_data['Market_Cap'] / fundamental_data['Revenue']
        
        if 'Net_Income' in fundamental_data.columns and 'Market_Cap' in fundamental_data.columns:
            features['price_to_earnings'] = fundamental_data['Market_Cap'] / fundamental_data['Net_Income']
        
        return features.dropna()

class MacroFeatures:
    """Macroeconomic Features Creator."""
    
    def __init__(self):
        pass
    
    def create_macro_features(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """Create macroeconomic features."""
        features = pd.DataFrame(index=macro_data.index)
        
        # Interest rate features
        if 'Interest_Rate' in macro_data.columns:
            features['interest_rate'] = macro_data['Interest_Rate']
            features['interest_rate_change'] = macro_data['Interest_Rate'].pct_change()
        
        # Inflation features
        if 'Inflation' in macro_data.columns:
            features['inflation'] = macro_data['Inflation']
            features['inflation_change'] = macro_data['Inflation'].pct_change()
        
        return features.dropna()
