"""
Data infrastructure module for MeridianAlgo.

This module provides unified interfaces for accessing financial data from multiple sources,
data processing pipelines, and real-time data streaming capabilities.
"""

from .providers import (
    DataProvider,
    YahooFinanceProvider,
    AlphaVantageProvider,
    QuandlProvider,
    IEXCloudProvider,
    FREDProvider,
    DataProviderManager
)

from .processing import (
    DataPipeline,
    DataValidator,
    OutlierDetector,
    MissingDataHandler,
    DataNormalizer
)

from .models import (
    MarketData,
    DataRequest,
    DataResponse
)

from .exceptions import (
    DataError,
    ProviderError,
    ValidationError
)

from .streaming import (
    StreamingEvent,
    StreamingDataHandler,
    MarketDataHandler,
    DataBuffer,
    WebSocketStreamer,
    EventDrivenProcessor,
    StreamingDataManager,
    create_market_data_processor,
    create_data_aggregator
)

from .storage import (
    ParquetStorage,
    RedisCache,
    DataStorageManager
)

__all__ = [
    # Providers
    'DataProvider',
    'YahooFinanceProvider', 
    'AlphaVantageProvider',
    'QuandlProvider',
    'IEXCloudProvider',
    'FREDProvider',
    'DataProviderManager',
    
    # Processing
    'DataPipeline',
    'DataValidator',
    'OutlierDetector', 
    'MissingDataHandler',
    'DataNormalizer',
    
    # Models
    'MarketData',
    'DataRequest',
    'DataResponse',
    
    # Exceptions
    'DataError',
    'ProviderError',
    'ValidationError',
    
    # Streaming
    'StreamingEvent',
    'StreamingDataHandler',
    'MarketDataHandler',
    'DataBuffer',
    'WebSocketStreamer',
    'EventDrivenProcessor',
    'StreamingDataManager',
    'create_market_data_processor',
    'create_data_aggregator',
    
    # Storage
    'ParquetStorage',
    'RedisCache',
    'DataStorageManager'
]