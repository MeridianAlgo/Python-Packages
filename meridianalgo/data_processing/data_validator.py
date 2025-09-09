"""
Data Validation Module

This module provides data validation utilities for financial data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

class DataValidator:
    """Financial Data Validator."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def validate_data(self) -> Dict[str, bool]:
        """Validate financial data."""
        return {
            'has_missing_values': self.data.isnull().any().any(),
            'has_infinite_values': np.isinf(self.data).any().any(),
            'has_negative_prices': (self.data < 0).any().any(),
            'has_duplicate_dates': self.data.index.duplicated().any(),
            'is_monotonic_index': self.data.index.is_monotonic_increasing
        }
    
    def get_data_quality_report(self) -> Dict[str, Union[int, float]]:
        """Get data quality report."""
        return {
            'total_rows': len(self.data),
            'total_columns': len(self.data.columns),
            'missing_values_count': self.data.isnull().sum().sum(),
            'missing_values_percentage': (self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))) * 100,
            'infinite_values_count': np.isinf(self.data).sum().sum(),
            'duplicate_rows': self.data.duplicated().sum()
        }

class SchemaValidator:
    """Schema Validator for financial data."""
    
    def __init__(self, expected_columns: List[str]):
        self.expected_columns = expected_columns
    
    def validate_schema(self, data: pd.DataFrame) -> Dict[str, bool]:
        """Validate data schema."""
        return {
            'has_required_columns': all(col in data.columns for col in self.expected_columns),
            'has_extra_columns': len(data.columns) > len(self.expected_columns),
            'missing_columns': [col for col in self.expected_columns if col not in data.columns]
        }

class QualityChecker:
    """Data Quality Checker."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def check_data_quality(self) -> Dict[str, Union[bool, List[str]]]:
        """Check data quality."""
        issues = []
        
        # Check for missing values
        if self.data.isnull().any().any():
            issues.append("Data contains missing values")
        
        # Check for infinite values
        if np.isinf(self.data).any().any():
            issues.append("Data contains infinite values")
        
        # Check for negative prices
        if (self.data < 0).any().any():
            issues.append("Data contains negative prices")
        
        # Check for duplicate dates
        if self.data.index.duplicated().any():
            issues.append("Data contains duplicate dates")
        
        return {
            'is_clean': len(issues) == 0,
            'issues': issues
        }
