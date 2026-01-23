"""
High-performance computing architecture with distributed computing and GPU acceleration.
"""

import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import dask
    import dask.dataframe as dd
    from dask import delayed
    from dask.distributed import Client

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    warnings.warn("Dask not available. Distributed computing will be limited.")

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    warnings.warn("Ray not available. Distributed ML will be limited.")

try:
    import cudf
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy/cuDF not available. GPU acceleration will be limited.")

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    warnings.warn("Redis not available. Caching will be limited.")

logger = logging.getLogger(__name__)


class DistributedComputing:
    """Distributed computing framework using Dask."""

    def __init__(self, scheduler_address: str = None, n_workers: int = None):
        self.scheduler_address = scheduler_address
        self.n_workers = n_workers or 4
        self.client = None

        if DASK_AVAILABLE:
            self._initialize_client()

    def _initialize_client(self):
        """Initialize Dask client."""
        try:
            if self.scheduler_address:
                self.client = Client(self.scheduler_address)
            else:
                self.client = Client(processes=True, n_workers=self.n_workers)

            logger.info(f"Dask client initialized: {self.client}")
        except Exception as e:
            logger.error(f"Failed to initialize Dask client: {e}")
            self.client = None

    def parallelize_dataframe_operation(
        self, df: pd.DataFrame, operation: Callable, partition_size: str = "100MB"
    ) -> pd.DataFrame:
        """Parallelize DataFrame operations using Dask."""

        if not DASK_AVAILABLE or self.client is None:
            logger.warning("Dask not available, running operation sequentially")
            return operation(df)

        try:
            # Convert to Dask DataFrame
            ddf = dd.from_pandas(df, chunksize=partition_size)

            # Apply operation
            result_ddf = operation(ddf)

            # Compute result
            result = result_ddf.compute()

            return result

        except Exception as e:
            logger.error(f"Error in distributed operation: {e}")
            return operation(df)

    def parallel_backtest(
        self, strategies: List[Any], data: pd.DataFrame, backtest_func: Callable
    ) -> List[Dict[str, Any]]:
        """Run multiple backtests in parallel."""

        if not DASK_AVAILABLE or self.client is None:
            logger.warning("Running backtests sequentially")
            return [backtest_func(strategy, data) for strategy in strategies]

        try:
            # Create delayed tasks
            tasks = []
            for strategy in strategies:
                task = delayed(backtest_func)(strategy, data)
                tasks.append(task)

            # Compute all tasks
            results = dask.compute(*tasks)

            return list(results)

        except Exception as e:
            logger.error(f"Error in parallel backtesting: {e}")
            return [backtest_func(strategy, data) for strategy in strategies]

    def distributed_feature_engineering(
        self, data: pd.DataFrame, feature_functions: List[Callable]
    ) -> pd.DataFrame:
        """Distribute feature engineering across workers."""

        if not DASK_AVAILABLE or self.client is None:
            logger.warning("Running feature engineering sequentially")
            features = []
            for func in feature_functions:
                features.append(func(data))
            return pd.concat(features, axis=1)

        try:
            # Create delayed tasks for each feature function
            tasks = []
            for func in feature_functions:
                task = delayed(func)(data)
                tasks.append(task)

            # Compute all features
            feature_results = dask.compute(*tasks)

            # Combine results
            return pd.concat(feature_results, axis=1)

        except Exception as e:
            logger.error(f"Error in distributed feature engineering: {e}")
            features = []
            for func in feature_functions:
                features.append(func(data))
            return pd.concat(features, axis=1)

    def close(self):
        """Close Dask client."""
        if self.client:
            self.client.close()


class GPUAcceleration:
    """GPU acceleration using CuPy and RAPIDS."""

    def __init__(self):
        self.gpu_available = CUPY_AVAILABLE

        if self.gpu_available:
            try:
                # Test GPU availability
                cp.cuda.Device(0).use()
                logger.info("GPU acceleration available")
            except Exception as e:
                logger.warning(f"GPU not available: {e}")
                self.gpu_available = False

    def to_gpu(
        self, data: Union[np.ndarray, pd.DataFrame]
    ) -> Union[cp.ndarray, "cudf.DataFrame"]:
        """Move data to GPU."""

        if not self.gpu_available:
            return data

        try:
            if isinstance(data, np.ndarray):
                return cp.asarray(data)
            elif isinstance(data, pd.DataFrame):
                return cudf.from_pandas(data)
            else:
                return data
        except Exception as e:
            logger.error(f"Error moving data to GPU: {e}")
            return data

    def to_cpu(
        self, data: Union[cp.ndarray, "cudf.DataFrame"]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Move data back to CPU."""

        try:
            if hasattr(data, "get"):  # CuPy array
                return data.get()
            elif hasattr(data, "to_pandas"):  # cuDF DataFrame
                return data.to_pandas()
            else:
                return data
        except Exception as e:
            logger.error(f"Error moving data to CPU: {e}")
            return data

    def gpu_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators on GPU."""

        if not self.gpu_available:
            return self._cpu_technical_indicators(data)

        try:
            # Move data to GPU
            gpu_data = cudf.from_pandas(data)

            # Calculate indicators on GPU
            results = {}

            # Simple Moving Average
            for period in [10, 20, 50]:
                results[f"SMA_{period}"] = (
                    gpu_data["Close"].rolling(window=period).mean()
                )

            # RSI (simplified GPU version)
            delta = gpu_data["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            results["RSI"] = 100 - (100 / (1 + rs))

            # Combine results
            result_df = cudf.concat(results.values(), axis=1, keys=results.keys())

            # Move back to CPU
            return result_df.to_pandas()

        except Exception as e:
            logger.error(f"Error in GPU technical indicators: {e}")
            return self._cpu_technical_indicators(data)

    def _cpu_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fallback CPU implementation."""
        results = {}

        # Simple Moving Average
        for period in [10, 20, 50]:
            results[f"SMA_{period}"] = data["Close"].rolling(window=period).mean()

        # RSI
        delta = data["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        results["RSI"] = 100 - (100 / (1 + rs))

        return pd.concat(results.values(), axis=1, keys=results.keys())

    def gpu_matrix_operations(
        self, matrix_a: np.ndarray, matrix_b: np.ndarray
    ) -> np.ndarray:
        """Perform matrix operations on GPU."""

        if not self.gpu_available:
            return np.dot(matrix_a, matrix_b)

        try:
            # Move to GPU
            gpu_a = cp.asarray(matrix_a)
            gpu_b = cp.asarray(matrix_b)

            # Perform operation
            result_gpu = cp.dot(gpu_a, gpu_b)

            # Move back to CPU
            return result_gpu.get()

        except Exception as e:
            logger.error(f"Error in GPU matrix operations: {e}")
            return np.dot(matrix_a, matrix_b)


class IntelligentCache:
    """Intelligent caching system with Redis."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        default_ttl: int = 3600,
    ):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.default_ttl = default_ttl
        self.redis_client = None

        if REDIS_AVAILABLE:
            self._initialize_redis()

    def _initialize_redis(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=False,
            )

            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.redis_client = None

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""

        if not self.redis_client:
            return None

        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                import pickle

                return pickle.loads(cached_data)
            return None

        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache."""

        if not self.redis_client:
            return False

        try:
            import pickle

            serialized_value = pickle.dumps(value)

            ttl = ttl or self.default_ttl
            success = self.redis_client.setex(key, ttl, serialized_value)

            return success

        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache."""

        if not self.redis_client:
            return False

        try:
            result = self.redis_client.delete(key)
            return bool(result)

        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern."""

        if not self.redis_client:
            return 0

        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                return deleted
            return 0

        except Exception as e:
            logger.error(f"Error clearing pattern: {e}")
            return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""

        if not self.redis_client:
            return {}

        try:
            info = self.redis_client.info()
            return {
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "connected_clients": info.get("connected_clients", 0),
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}


class HighPerformanceComputing:
    """Main HPC orchestrator."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Initialize components
        self.distributed = DistributedComputing(
            scheduler_address=self.config.get("dask_scheduler"),
            n_workers=self.config.get("n_workers", 4),
        )

        self.gpu = GPUAcceleration()

        self.cache = IntelligentCache(
            redis_host=self.config.get("redis_host", "localhost"),
            redis_port=self.config.get("redis_port", 6379),
            default_ttl=self.config.get("cache_ttl", 3600),
        )

        # Performance monitoring
        self.performance_stats = {
            "operations_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "gpu_operations": 0,
            "distributed_operations": 0,
        }

    def optimize_dataframe_operation(
        self,
        df: pd.DataFrame,
        operation: Callable,
        use_gpu: bool = True,
        use_distributed: bool = True,
        cache_key: str = None,
    ) -> pd.DataFrame:
        """Optimize DataFrame operation using available acceleration."""

        self.performance_stats["operations_count"] += 1

        # Check cache first
        if cache_key:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.performance_stats["cache_hits"] += 1
                logger.debug(f"Cache hit for {cache_key}")
                return cached_result
            else:
                self.performance_stats["cache_misses"] += 1

        # Determine best execution strategy
        data_size = df.memory_usage(deep=True).sum()

        # Use GPU for large datasets if available
        if use_gpu and self.gpu.gpu_available and data_size > 100_000_000:  # 100MB
            logger.debug("Using GPU acceleration")
            self.performance_stats["gpu_operations"] += 1

            try:
                # Move to GPU and execute
                gpu_df = self.gpu.to_gpu(df)
                gpu_result = operation(gpu_df)
                result = self.gpu.to_cpu(gpu_result)
            except Exception as e:
                logger.warning(f"GPU operation failed, falling back to CPU: {e}")
                result = operation(df)

        # Use distributed computing for very large datasets
        elif use_distributed and data_size > 500_000_000:  # 500MB
            logger.debug("Using distributed computing")
            self.performance_stats["distributed_operations"] += 1
            result = self.distributed.parallelize_dataframe_operation(df, operation)

        # Use regular CPU execution
        else:
            logger.debug("Using CPU execution")
            result = operation(df)

        # Cache result if key provided
        if cache_key:
            self.cache.set(cache_key, result)

        return result

    def parallel_portfolio_optimization(
        self, portfolios: List[Dict[str, Any]], optimization_func: Callable
    ) -> List[Dict[str, Any]]:
        """Optimize multiple portfolios in parallel."""

        if len(portfolios) == 1:
            return [optimization_func(portfolios[0])]

        # Use Ray for ML-heavy operations if available
        if RAY_AVAILABLE:
            try:
                if not ray.is_initialized():
                    ray.init()

                @ray.remote
                def optimize_portfolio(portfolio):
                    return optimization_func(portfolio)

                # Submit tasks
                futures = [
                    optimize_portfolio.remote(portfolio) for portfolio in portfolios
                ]

                # Get results
                results = ray.get(futures)

                return results

            except Exception as e:
                logger.error(f"Ray optimization failed: {e}")

        # Fallback to Dask
        return self.distributed.parallel_backtest(portfolios, None, optimization_func)

    def accelerated_risk_calculation(
        self, positions: pd.DataFrame, covariance_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate portfolio risk using GPU acceleration."""

        # Convert to numpy arrays
        weights = positions.values
        cov_matrix = covariance_matrix.values

        # Use GPU for matrix operations
        portfolio_variance = self.gpu.gpu_matrix_operations(
            weights.T, self.gpu.gpu_matrix_operations(cov_matrix, weights)
        )

        portfolio_volatility = np.sqrt(portfolio_variance)

        return {
            "portfolio_variance": float(portfolio_variance),
            "portfolio_volatility": float(portfolio_volatility),
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""

        stats = self.performance_stats.copy()

        # Add cache stats
        cache_stats = self.cache.get_cache_stats()
        stats.update({f"cache_{k}": v for k, v in cache_stats.items()})

        # Calculate hit rate
        total_cache_ops = stats["cache_hits"] + stats["cache_misses"]
        if total_cache_ops > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / total_cache_ops
        else:
            stats["cache_hit_rate"] = 0

        return stats

    def cleanup(self):
        """Cleanup resources."""

        if self.distributed.client:
            self.distributed.close()

        if RAY_AVAILABLE and ray.is_initialized():
            ray.shutdown()

        logger.info("HPC resources cleaned up")


# Utility functions for common HPC operations
def auto_optimize_operation(
    df: pd.DataFrame, operation: Callable, hpc: HighPerformanceComputing = None
) -> pd.DataFrame:
    """Automatically optimize operation based on data size and available resources."""

    if hpc is None:
        hpc = HighPerformanceComputing()

    return hpc.optimize_dataframe_operation(df, operation)


def parallel_apply(
    df: pd.DataFrame, func: Callable, hpc: HighPerformanceComputing = None
) -> pd.DataFrame:
    """Apply function in parallel across DataFrame."""

    if hpc is None:
        hpc = HighPerformanceComputing()

    def parallel_operation(data):
        if hasattr(data, "apply"):
            return data.apply(func)
        else:
            return func(data)

    return hpc.optimize_dataframe_operation(df, parallel_operation)


def cached_computation(
    cache_key: str,
    computation_func: Callable,
    hpc: HighPerformanceComputing = None,
    ttl: int = 3600,
) -> Any:
    """Perform cached computation."""

    if hpc is None:
        hpc = HighPerformanceComputing()

    # Check cache
    result = hpc.cache.get(cache_key)
    if result is not None:
        return result

    # Compute and cache
    result = computation_func()
    hpc.cache.set(cache_key, result, ttl)

    return result
