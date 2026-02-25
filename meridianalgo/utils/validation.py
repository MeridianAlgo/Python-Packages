"""
MeridianAlgo Validation Framework

Enterprise-grade validation utilities for financial data, parameters,
and model inputs.
"""

import functools
from typing import Union

import numpy as np
import pandas as pd


class ValidationError(Exception):
    """Custom exception for validation failures."""

    pass


def validate_input(val_type: str = "numerical"):
    """
    Decorator for input validation.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Basic validation logic based on val_type
            if val_type == "numerical":
                for arg in args[1:]:  # Skip 'self'
                    if isinstance(arg, (int, float)):
                        if np.isnan(arg):
                            raise ValidationError(
                                f"NaN value passed to {func.__name__}"
                            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


class DataValidator:
    """
    Validator for financial datasets (DataFrames, Series).
    """

    @staticmethod
    def validate_timeseries(
        df: Union[pd.DataFrame, pd.Series],
        allow_missing: bool = False,
        check_index: bool = True,
    ) -> bool:
        """
        Validate a time series dataset.
        """
        if check_index and not isinstance(df.index, pd.DatetimeIndex):
            raise ValidationError("Dataset must have a DatetimeIndex")

        if not allow_missing and df.isnull().any().any():
            raise ValidationError("Dataset contains missing values")

        if isinstance(df, pd.DataFrame) and df.empty:
            raise ValidationError("DataFrame is empty")

        return True

    @staticmethod
    def check_positive_semi_definite(matrix: np.ndarray) -> bool:
        """
        Check if a matrix (e.g. covariance) is positive semi-definite.
        """
        return np.all(np.linalg.eigvals(matrix) >= -1e-8)


def ensure_finite(arr: np.ndarray, name: str = "array"):
    """Ensure no NaNs or Infs in array."""
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"{name} contains non-finite values (NaN/Inf)")
