"""
Data models for the MeridianAlgo data infrastructure.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class MarketData:
    """Represents market data for a single instrument at a point in time."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None

    def to_ohlcv(self) -> Tuple[float, float, float, float, int]:
        """Convert to OHLCV tuple."""
        return (self.open, self.high, self.low, self.close, self.volume)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "adjusted_close": self.adjusted_close,
        }


@dataclass
class DataRequest:
    """Represents a request for market data."""

    symbols: List[str]
    start_date: datetime
    end_date: datetime
    interval: str = "1d"  # 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo
    data_type: str = "ohlcv"  # ohlcv, fundamentals, news, etc.
    provider: Optional[str] = None

    def __post_init__(self):
        """Validate the request parameters."""
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")

        valid_intervals = ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]
        if self.interval not in valid_intervals:
            raise ValueError(f"interval must be one of {valid_intervals}")


@dataclass
class DataResponse:
    """Represents a response containing market data."""

    data: pd.DataFrame
    metadata: Dict[str, Any]
    provider: str
    timestamp: datetime

    def __post_init__(self):
        """Validate the response data."""
        if self.data.empty:
            raise ValueError("Response data cannot be empty")


@dataclass
class FundamentalData:
    """Represents fundamental data for a company."""

    symbol: str
    timestamp: datetime
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    eps: Optional[float] = None
    revenue: Optional[float] = None
    net_income: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "market_cap": self.market_cap,
            "pe_ratio": self.pe_ratio,
            "pb_ratio": self.pb_ratio,
            "dividend_yield": self.dividend_yield,
            "eps": self.eps,
            "revenue": self.revenue,
            "net_income": self.net_income,
        }
