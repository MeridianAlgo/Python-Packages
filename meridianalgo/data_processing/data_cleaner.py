"""
Data Cleaning Module

This module provides data cleaning utilities for financial data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

class DataCleaner:
    """Financial Data Cleaner."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def clean_data(self) -> pd.DataFrame:
        """Clean financial data."""
        cleaned_data = self.data.copy()
        
        # Remove duplicates
        cleaned_data = cleaned_data.drop_duplicates()
        
        # Handle missing values
        cleaned_data = cleaned_data.fillna(method='ffill').fillna(method='bfill')
        
        # Remove infinite values
        cleaned_data = cleaned_data.replace([np.inf, -np.inf], np.nan)
        cleaned_data = cleaned_data.fillna(method='ffill').fillna(method='bfill')
        
        return cleaned_data

class OutlierDetector:
    """Outlier Detection for financial data."""
    
    def __init__(self, data: pd.Series):
        self.data = data
    
    def detect_outliers(self, method: str = 'iqr') -> pd.Series:
        """Detect outliers in data."""
        if method == 'iqr':
            return self.detect_iqr_outliers()
        elif method == 'zscore':
            return self.detect_zscore_outliers()
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")
    
    def detect_iqr_outliers(self, threshold: float = 1.5) -> pd.Series:
        """Detect outliers using IQR method."""
        Q1 = self.data.quantile(0.25)
        Q3 = self.data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        return (self.data < lower_bound) | (self.data > upper_bound)
    
    def detect_zscore_outliers(self, threshold: float = 3) -> pd.Series:
        """Detect outliers using Z-score method."""
        z_scores = np.abs((self.data - self.data.mean()) / self.data.std())
        return z_scores > threshold

class MissingDataHandler:
    """Missing Data Handler for financial data."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def handle_missing_data(self, method: str = 'forward_fill') -> pd.DataFrame:
        """Handle missing data."""
        if method == 'forward_fill':
            return self.data.fillna(method='ffill')
        elif method == 'backward_fill':
            return self.data.fillna(method='bfill')
        elif method == 'interpolate':
            return self.data.interpolate()
        elif method == 'drop':
            return self.data.dropna()
        else:
            raise ValueError("Method must be 'forward_fill', 'backward_fill', 'interpolate', or 'drop'")
