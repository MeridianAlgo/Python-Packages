"""
Performance optimization utilities for MeridianAlgo.
"""

import functools
import multiprocessing
import warnings
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import psutil
from joblib import Memory, Parallel, delayed
from numba import jit, prange

# Setup memory caching
memory = Memory(location="./.meridianalgo_cache", verbose=0)


class PerformanceMonitor:
    """Monitor and optimize performance of operations."""

    def __init__(self):
        self.metrics = {}

    def time_function(self, func: Callable) -> Callable:
        """Decorator to time function execution."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time

            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()

            self.metrics[func.__name__] = {
                "execution_time": end - start,
                "last_run": end,
            }
            return result

        return wrapper

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics."""
        return self.metrics.copy()


# Global performance monitor
perf_monitor = PerformanceMonitor()


def cache_result(maxsize: int = 128, ttl: Optional[float] = None):
    """Cache function results with optional TTL."""

    def decorator(func: Callable) -> Callable:
        if ttl:
            return memory.cache(func, ignore=["self"], cache_validation=ttl)
        else:
            return functools.lru_cache(maxsize=maxsize)(func)

    return decorator


def parallelize(n_jobs: Optional[int] = None, backend: str = "loky"):
    """Parallelize function execution."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            actual_n_jobs = (
                n_jobs if n_jobs is not None else multiprocessing.cpu_count()
            )

            if "iterables" in kwargs:
                iterables = kwargs.pop("iterables")
                results = Parallel(n_jobs=actual_n_jobs, backend=backend)(
                    delayed(func)(*args, **{**kwargs, "item": item})
                    for item in iterables
                )
                return results
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


@jit(nopython=True, cache=True)
def fast_rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling standard deviation using Numba."""
    n = len(values)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(window - 1, n):
        window_data = values[i - window + 1 : i + 1]
        result[i] = np.std(window_data)

    return result


@jit(nopython=True, cache=True)
def fast_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling mean using Numba."""
    n = len(values)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(window - 1, n):
        window_data = values[i - window + 1 : i + 1]
        result[i] = np.mean(window_data)

    return result


@jit(nopython=True, cache=True)
def fast_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Fast correlation calculation using Numba."""
    n = len(x)
    if n == 0:
        return np.nan

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    x_centered = x - x_mean
    y_centered = y - y_mean

    numerator = np.sum(x_centered * y_centered)
    denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))

    if denominator == 0:
        return np.nan

    return numerator / denominator


@jit(nopython=True, cache=True, parallel=True)
def fast_correlation_matrix(returns: np.ndarray) -> np.ndarray:
    """Fast correlation matrix calculation using Numba."""
    n_assets = returns.shape[1]
    corr_matrix = np.empty((n_assets, n_assets))

    for i in prange(n_assets):
        for j in prange(n_assets):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                corr_matrix[i, j] = fast_correlation(returns[:, i], returns[:, j])

    return corr_matrix


class DataOptimizer:
    """Optimize data structures for performance."""

    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes for memory efficiency."""
        optimized_df = df.copy()

        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype

            if col_type != "object":
                c_min = optimized_df[col].min()
                c_max = optimized_df[col].max()

                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        optimized_df[col] = optimized_df[col].astype(np.int8)
                    elif (
                        c_min > np.iinfo(np.int16).min
                        and c_max < np.iinfo(np.int16).max
                    ):
                        optimized_df[col] = optimized_df[col].astype(np.int16)
                    elif (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        optimized_df[col] = optimized_df[col].astype(np.int32)
                else:
                    if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                    ):
                        optimized_df[col] = optimized_df[col].astype(np.float32)
                    elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        optimized_df[col] = optimized_df[col].astype(np.float32)
            else:
                if optimized_df[col].nunique() / len(optimized_df[col]) < 0.5:
                    optimized_df[col] = optimized_df[col].astype("category")

        return optimized_df

    @staticmethod
    def chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
        """Split DataFrame into chunks for memory-efficient processing."""
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i : i + chunk_size]
            chunks.append(chunk)
        return chunks


class MemoryManager:
    """Manage memory usage during operations."""

    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss": memory_info.rss / 1024 / 1024,  # MB
            "vms": memory_info.vms / 1024 / 1024,  # MB
            "percent": process.memory_percent(),
        }

    @staticmethod
    def monitor_memory_usage(threshold: float = 80.0):
        """Monitor memory usage and warn if threshold exceeded."""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                memory_before = MemoryManager.get_memory_usage()

                if memory_before["percent"] > threshold:
                    warnings.warn(
                        f"High memory usage before function: {memory_before['percent']:.1f}%",
                        UserWarning,
                    )

                result = func(*args, **kwargs)

                memory_after = MemoryManager.get_memory_usage()

                if memory_after["percent"] > threshold:
                    warnings.warn(
                        f"High memory usage after function: {memory_after['percent']:.1f}%",
                        UserWarning,
                    )

                return result

            return wrapper

        return decorator


class VectorizedOperations:
    """Collection of vectorized operations for performance."""

    @staticmethod
    @cache_result(maxsize=100)
    def calculate_returns_vectorized(prices: np.ndarray) -> np.ndarray:
        """Vectorized returns calculation."""
        return np.diff(prices) / prices[:-1]

    @staticmethod
    @cache_result(maxsize=100)
    def calculate_rolling_returns_vectorized(
        prices: np.ndarray, window: int
    ) -> np.ndarray:
        """Vectorized rolling returns calculation."""
        returns = np.zeros(len(prices))
        returns[:] = np.nan

        for i in range(window, len(prices)):
            returns[i] = (prices[i] - prices[i - window]) / prices[i - window]

        return returns

    @staticmethod
    def calculate_zscore_vectorized(values: np.ndarray, window: int) -> np.ndarray:
        """Vectorized z-score calculation."""
        rolling_mean = fast_rolling_mean(values, window)
        rolling_std = fast_rolling_std(values, window)

        zscores = np.empty_like(values)
        zscores[:] = np.nan

        valid_mask = (
            ~np.isnan(rolling_mean) & ~np.isnan(rolling_std) & (rolling_std != 0)
        )
        zscores[valid_mask] = (
            values[valid_mask] - rolling_mean[valid_mask]
        ) / rolling_std[valid_mask]

        return zscores


# Performance decorators for commonly used functions
def optimize_performance(func: Callable) -> Callable:
    """Apply multiple performance optimizations to a function."""

    @functools.wraps(func)
    @cache_result(maxsize=128)
    @perf_monitor.time_function
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


# Batch processing utilities
class BatchProcessor:
    """Process data in batches for memory efficiency."""

    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size

    def process_in_batches(
        self, data: pd.DataFrame, process_func: Callable, **kwargs
    ) -> List[Any]:
        """Process data in batches."""
        results = []

        for i in range(0, len(data), self.batch_size):
            batch = data.iloc[i : i + self.batch_size]
            result = process_func(batch, **kwargs)
            results.append(result)

        return results

    def parallel_batch_process(
        self,
        data: pd.DataFrame,
        process_func: Callable,
        n_jobs: Optional[int] = None,
        **kwargs,
    ) -> List[Any]:
        """Process data in batches in parallel."""
        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count()

        batches = [
            data.iloc[i : i + self.batch_size]
            for i in range(0, len(data), self.batch_size)
        ]

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_func)(batch, **kwargs) for batch in batches
        )

        return results
