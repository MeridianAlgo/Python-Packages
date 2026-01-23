"""
Data processing pipeline components for cleaning, validating, and normalizing financial data.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler

from .exceptions import ValidationError

logger = logging.getLogger(__name__)


class DataProcessor(ABC):
    """Abstract base class for data processors."""

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data."""
        pass

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "DataProcessor":
        """Fit the processor to the data."""
        pass


class DataValidator(DataProcessor):
    """Validates financial data for common issues."""

    def __init__(self, strict: bool = False):
        self.strict = strict
        self.validation_rules = [
            self._check_required_columns,
            self._check_data_types,
            self._check_negative_prices,
            self._check_zero_volume,
            self._check_ohlc_consistency,
            self._check_missing_values,
            self._check_price_gaps,
            self._check_volume_spikes,
            self._check_timestamp_consistency,
        ]

    def fit(self, data: pd.DataFrame) -> "DataValidator":
        """Fit method for consistency (no-op for validator)."""
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate the data and return cleaned version."""
        validated_data = data.copy()

        for rule in self.validation_rules:
            try:
                validated_data = rule(validated_data)
            except ValidationError as e:
                if self.strict:
                    raise e
                else:
                    logger.warning(f"Validation warning: {e}")

        return validated_data

    def _check_required_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Check for required OHLCV columns."""
        required_cols = ["Open", "High", "Low", "Close"]

        # Handle MultiIndex columns (multiple symbols)
        if isinstance(data.columns, pd.MultiIndex):
            # Check if required columns exist in any level
            level_0_cols = data.columns.get_level_values(0).unique()
            missing_cols = [col for col in required_cols if col not in level_0_cols]
        else:
            missing_cols = [col for col in required_cols if col not in data.columns]

        if missing_cols:
            raise ValidationError(f"Missing required columns: {missing_cols}")

        return data

    def _check_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure numeric columns are properly typed."""
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]

        if isinstance(data.columns, pd.MultiIndex):
            # Handle MultiIndex columns
            for col in numeric_cols:
                if col in data.columns.get_level_values(0):
                    # Convert all columns with this name to numeric
                    mask = data.columns.get_level_values(0) == col
                    for column in data.columns[mask]:
                        data[column] = pd.to_numeric(data[column], errors="coerce")
        else:
            for col in numeric_cols:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors="coerce")

        return data

    def _check_negative_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        """Check for negative prices."""
        price_cols = ["Open", "High", "Low", "Close"]

        if isinstance(data.columns, pd.MultiIndex):
            for col in price_cols:
                if col in data.columns.get_level_values(0):
                    mask = data.columns.get_level_values(0) == col
                    for column in data.columns[mask]:
                        negative_mask = data[column] < 0
                        if negative_mask.any():
                            logger.warning(
                                f"Found {negative_mask.sum()} negative values in {column}"
                            )
                            data.loc[negative_mask, column] = np.nan
        else:
            for col in price_cols:
                if col in data.columns:
                    negative_mask = data[col] < 0
                    if negative_mask.any():
                        logger.warning(
                            f"Found {negative_mask.sum()} negative values in {col}"
                        )
                        data.loc[negative_mask, col] = np.nan

        return data

    def _check_zero_volume(self, data: pd.DataFrame) -> pd.DataFrame:
        """Check for zero volume (may indicate data issues)."""
        if isinstance(data.columns, pd.MultiIndex):
            if "Volume" in data.columns.get_level_values(0):
                mask = data.columns.get_level_values(0) == "Volume"
                for column in data.columns[mask]:
                    zero_volume = (data[column] == 0).sum()
                    if zero_volume > 0:
                        logger.info(
                            f"Found {zero_volume} zero volume periods in {column}"
                        )
        else:
            if "Volume" in data.columns:
                zero_volume = (data["Volume"] == 0).sum()
                if zero_volume > 0:
                    logger.info(f"Found {zero_volume} zero volume periods")

        return data

    def _check_ohlc_consistency(self, data: pd.DataFrame) -> pd.DataFrame:
        """Check OHLC consistency (High >= Low, etc.)."""
        if isinstance(data.columns, pd.MultiIndex):
            # Get symbols
            symbols = data.columns.get_level_values(1).unique()
            for symbol in symbols:
                if all(
                    col in data.columns.get_level_values(0)
                    for col in ["Open", "High", "Low", "Close"]
                ):
                    high = data[("High", symbol)]
                    low = data[("Low", symbol)]
                    data[("Open", symbol)]
                    data[("Close", symbol)]

                    # Check High >= Low
                    inconsistent = high < low
                    if inconsistent.any():
                        logger.warning(
                            f"Found {inconsistent.sum()} periods where High < Low for {symbol}"
                        )
                        # Fix by swapping values
                        data.loc[inconsistent, ("High", symbol)] = low[inconsistent]
                        data.loc[inconsistent, ("Low", symbol)] = high[inconsistent]
        else:
            if all(col in data.columns for col in ["Open", "High", "Low", "Close"]):
                # Check High >= Low
                inconsistent = data["High"] < data["Low"]
                if inconsistent.any():
                    logger.warning(
                        f"Found {inconsistent.sum()} periods where High < Low"
                    )
                    # Fix by swapping values
                    high_vals = data.loc[inconsistent, "High"].copy()
                    data.loc[inconsistent, "High"] = data.loc[inconsistent, "Low"]
                    data.loc[inconsistent, "Low"] = high_vals

        return data

    def _check_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Check for missing values."""
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            logger.info(f"Found {missing_count} missing values in dataset")

        return data

    def _check_price_gaps(self, data: pd.DataFrame) -> pd.DataFrame:
        """Check for unusual price gaps."""
        price_cols = ["Open", "High", "Low", "Close"]

        if isinstance(data.columns, pd.MultiIndex):
            symbols = data.columns.get_level_values(1).unique()
            for symbol in symbols:
                for col in price_cols:
                    if (col, symbol) in data.columns:
                        series = data[(col, symbol)]
                        if len(series) > 1:
                            pct_change = series.pct_change().abs()
                            large_gaps = pct_change > 0.5  # 50% price change
                            if large_gaps.any():
                                logger.warning(
                                    f"Found {large_gaps.sum()} large price gaps (>50%) in {col} for {symbol}"
                                )
        else:
            for col in price_cols:
                if col in data.columns:
                    series = data[col]
                    if len(series) > 1:
                        pct_change = series.pct_change().abs()
                        large_gaps = pct_change > 0.5  # 50% price change
                        if large_gaps.any():
                            logger.warning(
                                f"Found {large_gaps.sum()} large price gaps (>50%) in {col}"
                            )

        return data

    def _check_volume_spikes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Check for unusual volume spikes."""
        if isinstance(data.columns, pd.MultiIndex):
            if "Volume" in data.columns.get_level_values(0):
                symbols = data.columns.get_level_values(1).unique()
                for symbol in symbols:
                    if ("Volume", symbol) in data.columns:
                        volume = data[("Volume", symbol)]
                        if len(volume) > 10:
                            median_volume = volume.median()
                            volume_spikes = volume > (
                                median_volume * 10
                            )  # 10x median volume
                            if volume_spikes.any():
                                logger.info(
                                    f"Found {volume_spikes.sum()} volume spikes (>10x median) for {symbol}"
                                )
        else:
            if "Volume" in data.columns:
                volume = data["Volume"]
                if len(volume) > 10:
                    median_volume = volume.median()
                    volume_spikes = volume > (median_volume * 10)  # 10x median volume
                    if volume_spikes.any():
                        logger.info(
                            f"Found {volume_spikes.sum()} volume spikes (>10x median)"
                        )

        return data

    def _check_timestamp_consistency(self, data: pd.DataFrame) -> pd.DataFrame:
        """Check for timestamp consistency issues."""
        if isinstance(data.index, pd.DatetimeIndex):
            # Check for duplicate timestamps
            duplicates = data.index.duplicated()
            if duplicates.any():
                logger.warning(f"Found {duplicates.sum()} duplicate timestamps")
                if self.strict:
                    raise ValidationError(
                        f"Duplicate timestamps found: {duplicates.sum()}"
                    )

            # Check for non-monotonic timestamps
            if not data.index.is_monotonic_increasing:
                logger.warning("Timestamps are not in ascending order")
                if self.strict:
                    raise ValidationError("Timestamps must be in ascending order")

        return data


class OutlierDetector(DataProcessor):
    """Detects and handles outliers in financial data using statistical and ML methods."""

    def __init__(
        self,
        method: str = "iqr",
        threshold: float = 3.0,
        handle_method: str = "cap",
        contamination: float = 0.1,
    ):
        """
        Initialize outlier detector.

        Args:
            method: Detection method ('iqr', 'zscore', 'modified_zscore', 'isolation_forest',
                   'elliptic_envelope', 'dbscan')
            threshold: Threshold for outlier detection (statistical methods)
            handle_method: How to handle outliers ('remove', 'cap', 'flag')
            contamination: Expected proportion of outliers (ML methods)
        """
        self.method = method
        self.threshold = threshold
        self.handle_method = handle_method
        self.contamination = contamination
        self.bounds_ = {}
        self.ml_detector_ = None

    def fit(self, data: pd.DataFrame) -> "OutlierDetector":
        """Fit the outlier detector to the data."""
        self.bounds_ = {}
        self.ml_detector_ = None

        # Get numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        if self.method in ["isolation_forest", "elliptic_envelope", "dbscan"]:
            # Machine learning methods
            clean_data = data[numeric_cols].dropna()

            if len(clean_data) < 10:
                logger.warning(
                    "Insufficient data for ML outlier detection, falling back to IQR"
                )
                self.method = "iqr"
            else:
                if self.method == "isolation_forest":
                    self.ml_detector_ = IsolationForest(
                        contamination=self.contamination,
                        random_state=42,
                        n_estimators=100,
                    )
                elif self.method == "elliptic_envelope":
                    self.ml_detector_ = EllipticEnvelope(
                        contamination=self.contamination, random_state=42
                    )
                elif self.method == "dbscan":
                    # DBSCAN doesn't have contamination parameter, we'll use it differently
                    self.ml_detector_ = DBSCAN(eps=0.5, min_samples=5)

                if self.ml_detector_ is not None:
                    self.ml_detector_.fit(clean_data)
                    return self

        # Statistical methods
        for col in numeric_cols:
            if isinstance(col, tuple):  # MultiIndex
                series = data[col].dropna()
            else:
                series = data[col].dropna()

            if len(series) == 0:
                continue

            if self.method == "iqr":
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.threshold * IQR
                upper_bound = Q3 + self.threshold * IQR

            elif self.method == "zscore":
                mean = series.mean()
                std = series.std()
                lower_bound = mean - self.threshold * std
                upper_bound = mean + self.threshold * std

            elif self.method == "modified_zscore":
                median = series.median()
                mad = np.median(np.abs(series - median))
                if mad == 0:
                    # Handle case where MAD is zero
                    lower_bound = median
                    upper_bound = median
                else:
                    modified_z_scores = 0.6745 * (series - median) / mad
                    mask = np.abs(modified_z_scores) <= self.threshold
                    if mask.any():
                        lower_bound = series[mask].min()
                        upper_bound = series[mask].max()
                    else:
                        lower_bound = median
                        upper_bound = median

            else:
                raise ValueError(f"Unknown outlier detection method: {self.method}")

            self.bounds_[col] = (lower_bound, upper_bound)

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data by handling outliers."""
        result = data.copy()
        outlier_count = 0

        # Handle ML-based outlier detection
        if self.ml_detector_ is not None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            clean_data = data[numeric_cols].dropna()

            if len(clean_data) > 0:
                if self.method == "dbscan":
                    # DBSCAN returns cluster labels, -1 indicates outliers
                    labels = self.ml_detector_.fit_predict(clean_data)
                    outlier_mask = labels == -1
                else:
                    # Isolation Forest and Elliptic Envelope return -1 for outliers
                    predictions = self.ml_detector_.predict(clean_data)
                    outlier_mask = predictions == -1

                outlier_indices = clean_data.index[outlier_mask]
                outlier_count = len(outlier_indices)

                if self.handle_method == "remove":
                    result = result.drop(outlier_indices)
                elif self.handle_method == "cap":
                    # For ML methods, cap to percentiles
                    for col in numeric_cols:
                        if col in result.columns:
                            lower_bound = result[col].quantile(0.05)
                            upper_bound = result[col].quantile(0.95)
                            result.loc[outlier_indices, col] = result.loc[
                                outlier_indices, col
                            ].clip(lower_bound, upper_bound)
                elif self.handle_method == "flag":
                    result.loc[:, "outlier_flag"] = False
                    result.loc[outlier_indices, "outlier_flag"] = True

        # Handle statistical methods
        elif self.bounds_:
            for col, (lower_bound, upper_bound) in self.bounds_.items():
                if col not in result.columns:
                    continue

                outliers = (result[col] < lower_bound) | (result[col] > upper_bound)
                outlier_count += outliers.sum()

                if self.handle_method == "remove":
                    result = result[~outliers]
                elif self.handle_method == "cap":
                    result.loc[result[col] < lower_bound, col] = lower_bound
                    result.loc[result[col] > upper_bound, col] = upper_bound
                elif self.handle_method == "flag":
                    # Add a flag column
                    flag_col = (
                        f"{col}_outlier"
                        if not isinstance(col, tuple)
                        else f"{col[0]}_{col[1]}_outlier"
                    )
                    result[flag_col] = outliers
        else:
            raise ValueError("OutlierDetector must be fitted before transform")

        if outlier_count > 0:
            logger.info(
                f"Detected and handled {outlier_count} outliers using {self.method} method"
            )

        return result


class MissingDataHandler(DataProcessor):
    """Handles missing data in financial time series with multiple interpolation strategies."""

    def __init__(
        self,
        method: str = "forward_fill",
        max_consecutive: int = 5,
        interpolation_method: str = "linear",
    ):
        """
        Initialize missing data handler.

        Args:
            method: Handling method ('forward_fill', 'backward_fill', 'interpolate', 'drop',
                   'seasonal', 'kalman', 'spline')
            max_consecutive: Maximum consecutive missing values to fill
            interpolation_method: Interpolation method for 'interpolate' ('linear', 'polynomial',
                                'spline', 'akima')
        """
        self.method = method
        self.max_consecutive = max_consecutive
        self.interpolation_method = interpolation_method
        self.seasonal_params_ = {}

    def fit(self, data: pd.DataFrame) -> "MissingDataHandler":
        """Fit method for consistency (no-op for missing data handler)."""
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data by handling missing values."""
        result = data.copy()

        if self.method == "forward_fill":
            result = result.fillna(method="ffill", limit=self.max_consecutive)

        elif self.method == "backward_fill":
            result = result.fillna(method="bfill", limit=self.max_consecutive)

        elif self.method == "interpolate":
            # Use specified interpolation method
            if isinstance(result.index, pd.DatetimeIndex):
                if self.interpolation_method == "linear":
                    result = result.interpolate(
                        method="time", limit=self.max_consecutive
                    )
                elif self.interpolation_method == "polynomial":
                    result = result.interpolate(
                        method="polynomial", order=2, limit=self.max_consecutive
                    )
                elif self.interpolation_method == "spline":
                    result = result.interpolate(
                        method="spline", order=3, limit=self.max_consecutive
                    )
                elif self.interpolation_method == "akima":
                    result = result.interpolate(
                        method="akima", limit=self.max_consecutive
                    )
                else:
                    result = result.interpolate(
                        method="time", limit=self.max_consecutive
                    )
            else:
                result = result.interpolate(
                    method=self.interpolation_method, limit=self.max_consecutive
                )

        elif self.method == "seasonal":
            # Seasonal decomposition-based interpolation
            result = self._seasonal_interpolate(result)

        elif self.method == "kalman":
            # Kalman filter-based interpolation (simplified)
            result = self._kalman_interpolate(result)

        elif self.method == "spline":
            # Spline interpolation
            result = result.interpolate(
                method="spline", order=3, limit=self.max_consecutive
            )

        elif self.method == "drop":
            result = result.dropna()

        else:
            raise ValueError(f"Unknown missing data method: {self.method}")

        missing_before = data.isnull().sum().sum()
        missing_after = result.isnull().sum().sum()

        if missing_before > 0:
            logger.info(
                f"Handled missing data: {missing_before} -> {missing_after} missing values"
            )

        return result

    def _seasonal_interpolate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Interpolate using seasonal patterns."""
        result = data.copy()

        for col in data.select_dtypes(include=[np.number]).columns:
            if data[col].isnull().any():
                # Simple seasonal interpolation using day-of-week patterns
                if isinstance(data.index, pd.DatetimeIndex):
                    # Group by day of week and use median for missing values
                    dow_medians = data.groupby(data.index.dayofweek)[col].median()

                    missing_mask = data[col].isnull()
                    for idx in data.index[missing_mask]:
                        dow = idx.dayofweek
                        if not pd.isna(dow_medians.get(dow)):
                            result.loc[idx, col] = dow_medians[dow]

        return result

    def _kalman_interpolate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Simple Kalman filter-based interpolation."""
        result = data.copy()

        for col in data.select_dtypes(include=[np.number]).columns:
            if data[col].isnull().any():
                # Simple state space model: random walk with noise
                series = data[col].copy()

                # Forward pass
                for i in range(1, len(series)):
                    if pd.isna(series.iloc[i]) and not pd.isna(series.iloc[i - 1]):
                        # Predict next value as previous value (random walk)
                        series.iloc[i] = series.iloc[i - 1]

                # Backward pass for remaining missing values
                for i in range(len(series) - 2, -1, -1):
                    if pd.isna(series.iloc[i]) and not pd.isna(series.iloc[i + 1]):
                        series.iloc[i] = series.iloc[i + 1]

                result[col] = series

        return result


class DataNormalizer(DataProcessor):
    """Normalizes financial data for analysis."""

    def __init__(self, method: str = "standard", columns: Optional[List[str]] = None):
        """
        Initialize data normalizer.

        Args:
            method: Normalization method ('standard', 'robust', 'minmax')
            columns: Specific columns to normalize (None for all numeric)
        """
        self.method = method
        self.columns = columns
        self.scaler = None
        self.fitted_columns = None

    def fit(self, data: pd.DataFrame) -> "DataNormalizer":
        """Fit the normalizer to the data."""
        # Determine columns to normalize
        if self.columns is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in self.columns if col in data.columns]

        self.fitted_columns = numeric_cols

        if not numeric_cols:
            logger.warning("No numeric columns found for normalization")
            return self

        # Initialize scaler
        if self.method == "standard":
            self.scaler = StandardScaler()
        elif self.method == "robust":
            self.scaler = RobustScaler()
        elif self.method == "minmax":
            from sklearn.preprocessing import MinMaxScaler

            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        # Fit scaler
        self.scaler.fit(data[numeric_cols])

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted normalizer."""
        if self.scaler is None or self.fitted_columns is None:
            raise ValueError("DataNormalizer must be fitted before transform")

        result = data.copy()

        if self.fitted_columns:
            # Apply normalization
            normalized_data = self.scaler.transform(data[self.fitted_columns])
            result[self.fitted_columns] = normalized_data

        return result


class DataPipeline:
    """Pipeline for processing financial data through multiple stages."""

    def __init__(self, processors: Optional[List[DataProcessor]] = None):
        """
        Initialize data pipeline.

        Args:
            processors: List of data processors to apply in order
        """
        if processors is None:
            # Default pipeline
            self.processors = [
                DataValidator(strict=False),
                OutlierDetector(method="iqr", handle_method="cap"),
                MissingDataHandler(method="forward_fill"),
            ]
        else:
            self.processors = processors

    def fit(self, data: pd.DataFrame) -> "DataPipeline":
        """Fit all processors in the pipeline."""
        current_data = data

        for processor in self.processors:
            processor.fit(current_data)
            current_data = processor.transform(current_data)

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data through the entire pipeline."""
        result = data.copy()

        for processor in self.processors:
            result = processor.transform(result)

        return result

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit the pipeline and transform the data."""
        return self.fit(data).transform(data)

    def add_processor(self, processor: DataProcessor, position: Optional[int] = None):
        """Add a processor to the pipeline."""
        if position is None:
            self.processors.append(processor)
        else:
            self.processors.insert(position, processor)

    def remove_processor(self, processor_type: type):
        """Remove all processors of a given type."""
        self.processors = [
            p for p in self.processors if not isinstance(p, processor_type)
        ]
