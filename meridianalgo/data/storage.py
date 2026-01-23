"""
Efficient data storage using Parquet format with Redis caching layer.
"""

import hashlib
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import redis

from .exceptions import DataError
from .models import DataRequest, DataResponse

logger = logging.getLogger(__name__)


class ParquetStorage:
    """Efficient storage using Parquet format with partitioning."""

    def __init__(self, base_path: str = "data/parquet"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Parquet write options for optimal compression and performance
        self.write_options = {
            "compression": "snappy",
            "use_dictionary": True,
            "write_statistics": True,
            "data_page_size": 1024 * 1024,  # 1MB
            "row_group_size": 50000,
        }

    def store_data(
        self,
        data: pd.DataFrame,
        symbol: str,
        data_type: str = "ohlcv",
        partition_cols: Optional[List[str]] = None,
    ) -> str:
        """
        Store DataFrame in Parquet format with partitioning.

        Args:
            data: DataFrame to store
            symbol: Symbol identifier
            data_type: Type of data (ohlcv, fundamentals, etc.)
            partition_cols: Columns to partition by

        Returns:
            Path where data was stored
        """
        try:
            # Create directory structure
            storage_path = self.base_path / data_type / symbol
            storage_path.mkdir(parents=True, exist_ok=True)

            # Add metadata columns
            data_with_meta = data.copy()
            data_with_meta["symbol"] = symbol
            data_with_meta["data_type"] = data_type
            data_with_meta["stored_at"] = datetime.now()

            # Add date partitioning if index is datetime
            if isinstance(data.index, pd.DatetimeIndex):
                data_with_meta["year"] = data.index.year
                data_with_meta["month"] = data.index.month
                data_with_meta["day"] = data.index.day

                if partition_cols is None:
                    partition_cols = ["year", "month"]

            # Convert to PyArrow table
            table = pa.Table.from_pandas(data_with_meta, preserve_index=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{data_type}_{timestamp}.parquet"
            file_path = storage_path / filename

            # Write with partitioning if specified
            if partition_cols:
                partition_path = storage_path / "partitioned"
                pq.write_to_dataset(
                    table,
                    root_path=str(partition_path),
                    partition_cols=partition_cols,
                    **self.write_options,
                )
                logger.info(f"Stored partitioned data for {symbol} at {partition_path}")
                return str(partition_path)
            else:
                # Write single file
                pq.write_table(table, str(file_path), **self.write_options)
                logger.info(f"Stored data for {symbol} at {file_path}")
                return str(file_path)

        except Exception as e:
            raise DataError(f"Failed to store data for {symbol}: {e}")

    def load_data(
        self,
        symbol: str,
        data_type: str = "ohlcv",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load data from Parquet storage.

        Args:
            symbol: Symbol identifier
            data_type: Type of data to load
            start_date: Start date filter
            end_date: End date filter
            columns: Specific columns to load

        Returns:
            Loaded DataFrame
        """
        try:
            storage_path = self.base_path / data_type / symbol

            if not storage_path.exists():
                raise DataError(f"No data found for {symbol} in {data_type}")

            # Check for partitioned data first
            partition_path = storage_path / "partitioned"
            if partition_path.exists():
                # Load partitioned data
                filters = []

                if start_date:
                    filters.append(("year", ">=", start_date.year))
                    if start_date.year == end_date.year if end_date else False:
                        filters.append(("month", ">=", start_date.month))

                if end_date:
                    filters.append(("year", "<=", end_date.year))
                    if start_date and start_date.year == end_date.year:
                        filters.append(("month", "<=", end_date.month))

                dataset = pq.ParquetDataset(str(partition_path), filters=filters)
                table = dataset.read(columns=columns)
                df = table.to_pandas()
            else:
                # Load from individual files
                parquet_files = list(storage_path.glob("*.parquet"))

                if not parquet_files:
                    raise DataError(f"No parquet files found for {symbol}")

                # Load most recent file or all files
                if len(parquet_files) == 1:
                    df = pd.read_parquet(parquet_files[0], columns=columns)
                else:
                    # Load and concatenate multiple files
                    dfs = []
                    for file_path in sorted(parquet_files):
                        file_df = pd.read_parquet(file_path, columns=columns)
                        dfs.append(file_df)
                    df = pd.concat(dfs, ignore_index=False)

            # Apply date filtering if needed
            if isinstance(df.index, pd.DatetimeIndex):
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]

            # Remove metadata columns
            meta_cols = ["symbol", "data_type", "stored_at", "year", "month", "day"]
            df = df.drop(
                columns=[col for col in meta_cols if col in df.columns], errors="ignore"
            )

            logger.info(f"Loaded {len(df)} records for {symbol} from {data_type}")
            return df

        except Exception as e:
            raise DataError(f"Failed to load data for {symbol}: {e}")

    def list_symbols(self, data_type: str = "ohlcv") -> List[str]:
        """List available symbols for a data type."""
        try:
            type_path = self.base_path / data_type
            if not type_path.exists():
                return []

            symbols = [d.name for d in type_path.iterdir() if d.is_dir()]
            return sorted(symbols)

        except Exception as e:
            logger.error(f"Failed to list symbols: {e}")
            return []

    def get_data_info(self, symbol: str, data_type: str = "ohlcv") -> Dict[str, Any]:
        """Get information about stored data."""
        try:
            storage_path = self.base_path / data_type / symbol

            if not storage_path.exists():
                return {}

            info = {
                "symbol": symbol,
                "data_type": data_type,
                "storage_path": str(storage_path),
                "files": [],
                "total_size": 0,
                "date_range": None,
            }

            # Check partitioned data
            partition_path = storage_path / "partitioned"
            if partition_path.exists():
                info["partitioned"] = True
                # Get partition info
                for root, dirs, files in os.walk(partition_path):
                    for file in files:
                        if file.endswith(".parquet"):
                            file_path = Path(root) / file
                            info["files"].append(str(file_path))
                            info["total_size"] += file_path.stat().st_size
            else:
                info["partitioned"] = False
                parquet_files = list(storage_path.glob("*.parquet"))
                for file_path in parquet_files:
                    info["files"].append(str(file_path))
                    info["total_size"] += file_path.stat().st_size

            # Get date range if possible
            if info["files"]:
                try:
                    sample_df = pd.read_parquet(info["files"][0], columns=[])
                    if isinstance(sample_df.index, pd.DatetimeIndex):
                        info["date_range"] = {
                            "start": sample_df.index.min(),
                            "end": sample_df.index.max(),
                        }
                except Exception:
                    pass

            return info

        except Exception as e:
            logger.error(f"Failed to get data info for {symbol}: {e}")
            return {}


class RedisCache:
    """Redis-based caching layer for fast data access."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: int = 3600,
    ):
        """
        Initialize Redis cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            default_ttl: Default time-to-live in seconds
        """
        self.default_ttl = default_ttl

        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,  # We'll handle binary data
                socket_connect_timeout=5,
                socket_timeout=5,
            )

            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")

        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.redis_client = None

    def _generate_key(
        self,
        symbol: str,
        data_type: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs,
    ) -> str:
        """Generate cache key for data request."""
        key_parts = [symbol, data_type]

        if start_date:
            key_parts.append(start_date.strftime("%Y%m%d"))
        if end_date:
            key_parts.append(end_date.strftime("%Y%m%d"))

        # Add other parameters
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")

        key_string = ":".join(key_parts)

        # Hash long keys
        if len(key_string) > 200:
            key_hash = hashlib.md5(key_string.encode()).hexdigest()
            return f"data:{key_hash}"

        return f"data:{key_string}"

    def get(
        self,
        symbol: str,
        data_type: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """Get data from cache."""
        if not self.redis_client:
            return None

        try:
            key = self._generate_key(symbol, data_type, start_date, end_date, **kwargs)
            cached_data = self.redis_client.get(key)

            if cached_data:
                # Deserialize DataFrame
                df = pickle.loads(cached_data)
                logger.debug(f"Cache hit for {symbol} ({data_type})")
                return df

            logger.debug(f"Cache miss for {symbol} ({data_type})")
            return None

        except Exception as e:
            logger.error(f"Error getting data from cache: {e}")
            return None

    def set(
        self,
        data: pd.DataFrame,
        symbol: str,
        data_type: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        ttl: Optional[int] = None,
        **kwargs,
    ) -> bool:
        """Set data in cache."""
        if not self.redis_client:
            return False

        try:
            key = self._generate_key(symbol, data_type, start_date, end_date, **kwargs)

            # Serialize DataFrame
            serialized_data = pickle.dumps(data)

            # Check size (Redis has 512MB limit per key)
            if len(serialized_data) > 100 * 1024 * 1024:  # 100MB limit
                logger.warning(
                    f"Data too large for cache: {len(serialized_data)} bytes"
                )
                return False

            ttl = ttl or self.default_ttl
            success = self.redis_client.setex(key, ttl, serialized_data)

            if success:
                logger.debug(f"Cached data for {symbol} ({data_type}), TTL: {ttl}s")

            return success

        except Exception as e:
            logger.error(f"Error setting data in cache: {e}")
            return False

    def delete(
        self,
        symbol: str,
        data_type: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs,
    ) -> bool:
        """Delete data from cache."""
        if not self.redis_client:
            return False

        try:
            key = self._generate_key(symbol, data_type, start_date, end_date, **kwargs)
            result = self.redis_client.delete(key)

            if result:
                logger.debug(f"Deleted cache entry for {symbol} ({data_type})")

            return bool(result)

        except Exception as e:
            logger.error(f"Error deleting data from cache: {e}")
            return False

    def clear_symbol(self, symbol: str) -> int:
        """Clear all cached data for a symbol."""
        if not self.redis_client:
            return 0

        try:
            pattern = f"data:*{symbol}*"
            keys = self.redis_client.keys(pattern)

            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} cache entries for {symbol}")
                return deleted

            return 0

        except Exception as e:
            logger.error(f"Error clearing cache for {symbol}: {e}")
            return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.redis_client:
            return {}

        try:
            info = self.redis_client.info()

            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}


class DataStorageManager:
    """Unified manager for data storage with Parquet and Redis caching."""

    def __init__(
        self,
        parquet_path: str = "data/parquet",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        cache_ttl: int = 3600,
        enable_cache: bool = True,
    ):
        """
        Initialize data storage manager.

        Args:
            parquet_path: Path for Parquet storage
            redis_host: Redis host
            redis_port: Redis port
            redis_db: Redis database
            redis_password: Redis password
            cache_ttl: Cache time-to-live in seconds
            enable_cache: Whether to enable Redis caching
        """
        self.parquet_storage = ParquetStorage(parquet_path)

        self.cache = None
        if enable_cache:
            self.cache = RedisCache(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                default_ttl=cache_ttl,
            )

        self.enable_cache = enable_cache and self.cache is not None

        logger.info(
            f"Data storage manager initialized (cache: {'enabled' if self.enable_cache else 'disabled'})"
        )

    def store_data_response(self, response: DataResponse) -> str:
        """Store a DataResponse object."""
        symbol = response.metadata.get("symbols", ["unknown"])[0]
        data_type = response.metadata.get("data_type", "ohlcv")

        # Store in Parquet
        storage_path = self.parquet_storage.store_data(response.data, symbol, data_type)

        # Cache the data
        if self.enable_cache:
            self.cache.set(response.data, symbol, data_type, **response.metadata)

        return storage_path

    def load_data(
        self, request: DataRequest, use_cache: bool = True
    ) -> Optional[DataResponse]:
        """
        Load data based on request, checking cache first.

        Args:
            request: Data request
            use_cache: Whether to use cache

        Returns:
            DataResponse if data found, None otherwise
        """
        symbol = request.symbols[0] if request.symbols else None
        if not symbol:
            return None

        # Try cache first
        if self.enable_cache and use_cache:
            cached_data = self.cache.get(
                symbol,
                request.data_type,
                request.start_date,
                request.end_date,
                interval=request.interval,
            )

            if cached_data is not None:
                return DataResponse(
                    data=cached_data,
                    metadata={
                        "symbols": request.symbols,
                        "data_type": request.data_type,
                        "source": "cache",
                    },
                    provider="cache",
                    timestamp=datetime.now(),
                )

        # Load from Parquet storage
        try:
            data = self.parquet_storage.load_data(
                symbol, request.data_type, request.start_date, request.end_date
            )

            response = DataResponse(
                data=data,
                metadata={
                    "symbols": request.symbols,
                    "data_type": request.data_type,
                    "source": "parquet",
                },
                provider="storage",
                timestamp=datetime.now(),
            )

            # Cache the loaded data
            if self.enable_cache:
                self.cache.set(
                    data,
                    symbol,
                    request.data_type,
                    request.start_date,
                    request.end_date,
                    interval=request.interval,
                )

            return response

        except DataError:
            return None

    def get_available_data(self) -> Dict[str, List[str]]:
        """Get available data by type."""
        available = {}

        # Check what's available in Parquet storage
        data_types = ["ohlcv", "fundamentals", "news", "economic"]

        for data_type in data_types:
            symbols = self.parquet_storage.list_symbols(data_type)
            if symbols:
                available[data_type] = symbols

        return available

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "parquet": {
                "base_path": str(self.parquet_storage.base_path),
                "available_data": self.get_available_data(),
            }
        }

        if self.enable_cache:
            stats["cache"] = self.cache.get_cache_stats()

        return stats

    def cleanup_old_data(self, days_old: int = 30) -> int:
        """Clean up old cached data."""
        if not self.enable_cache:
            return 0

        # This is a simplified cleanup - in practice, you'd want more sophisticated logic
        try:
            # Clear cache entries (Redis doesn't have built-in TTL scanning)
            # This would need to be implemented based on your key naming strategy
            logger.info(f"Cleanup requested for data older than {days_old} days")
            return 0

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0
