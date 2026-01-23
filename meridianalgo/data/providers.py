"""
Data provider implementations for accessing financial data from multiple sources.
"""

import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

# Import existing yfinance functionality
import yfinance as yf

from .exceptions import AuthenticationError, NetworkError, ProviderError, RateLimitError
from .models import DataRequest, DataResponse, FundamentalData

logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """Abstract base class for all data providers."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self._session = requests.Session()
        self._rate_limiter = RateLimiter(
            calls_per_second=self.config.get("rate_limit", 5)
        )

    @abstractmethod
    def get_historical_data(self, request: DataRequest) -> DataResponse:
        """Get historical market data."""
        pass

    @abstractmethod
    def get_real_time_data(self, symbols: List[str]) -> DataResponse:
        """Get real-time market data."""
        pass

    @abstractmethod
    def get_fundamental_data(self, symbols: List[str]) -> Dict[str, FundamentalData]:
        """Get fundamental data for symbols."""
        pass

    def is_available(self) -> bool:
        """Check if the provider is available."""
        try:
            # Simple connectivity test
            response = self._session.get(
                self.config.get("base_url", "https://httpbin.org/status/200"), timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def _handle_rate_limit(self):
        """Handle rate limiting."""
        self._rate_limiter.wait()

    def _make_request(
        self, url: str, params: Dict[str, Any] = None
    ) -> requests.Response:
        """Make a rate-limited HTTP request."""
        self._handle_rate_limit()

        try:
            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                raise RateLimitError(f"Rate limit exceeded for {self.name}")
            elif e.response.status_code == 401:
                raise AuthenticationError(f"Authentication failed for {self.name}")
            else:
                raise ProviderError(f"HTTP error {e.response.status_code}: {e}")
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Network error for {self.name}: {e}")


class YahooFinanceProvider(DataProvider):
    """Yahoo Finance data provider using yfinance."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("yahoo_finance", config)

    def get_historical_data(self, request: DataRequest) -> DataResponse:
        """Get historical data from Yahoo Finance."""
        try:
            # Convert symbols to Yahoo Finance format
            symbols_str = " ".join(request.symbols)

            # Download data using yfinance
            data = yf.download(
                symbols_str,
                start=request.start_date.strftime("%Y-%m-%d"),
                end=request.end_date.strftime("%Y-%m-%d"),
                interval=request.interval,
                auto_adjust=True,
                prepost=True,
                threads=True,
            )

            if data.empty:
                raise ProviderError(f"No data returned for symbols: {request.symbols}")

            # Standardize column names and format
            if len(request.symbols) == 1:
                # Single symbol - ensure consistent format
                data.columns = ["Open", "High", "Low", "Close", "Volume"]
                data = data.dropna()
            else:
                # Multiple symbols - handle MultiIndex columns
                data = data.dropna()

            metadata = {
                "symbols": request.symbols,
                "interval": request.interval,
                "source": "yahoo_finance",
                "auto_adjusted": True,
            }

            return DataResponse(
                data=data,
                metadata=metadata,
                provider=self.name,
                timestamp=datetime.now(),
            )

        except Exception as e:
            raise ProviderError(f"Yahoo Finance error: {str(e)}")

    def get_real_time_data(self, symbols: List[str]) -> DataResponse:
        """Get real-time data from Yahoo Finance."""
        try:
            tickers = [yf.Ticker(symbol) for symbol in symbols]
            data_list = []

            for ticker in tickers:
                info = ticker.info
                hist = ticker.history(period="1d", interval="1m")

                if not hist.empty:
                    latest = hist.iloc[-1]
                    data_list.append(
                        {
                            "symbol": info.get("symbol", ""),
                            "price": latest["Close"],
                            "volume": latest["Volume"],
                            "timestamp": hist.index[-1],
                        }
                    )

            df = pd.DataFrame(data_list)

            metadata = {
                "symbols": symbols,
                "source": "yahoo_finance",
                "real_time": True,
            }

            return DataResponse(
                data=df, metadata=metadata, provider=self.name, timestamp=datetime.now()
            )

        except Exception as e:
            raise ProviderError(f"Yahoo Finance real-time error: {str(e)}")

    def get_fundamental_data(self, symbols: List[str]) -> Dict[str, FundamentalData]:
        """Get fundamental data from Yahoo Finance."""
        fundamentals = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info

                fundamentals[symbol] = FundamentalData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    market_cap=info.get("marketCap"),
                    pe_ratio=info.get("trailingPE"),
                    pb_ratio=info.get("priceToBook"),
                    dividend_yield=info.get("dividendYield"),
                    eps=info.get("trailingEps"),
                    revenue=info.get("totalRevenue"),
                    net_income=info.get("netIncomeToCommon"),
                )

            except Exception as e:
                logger.warning(f"Failed to get fundamentals for {symbol}: {e}")
                continue

        return fundamentals


class AlphaVantageProvider(DataProvider):
    """Alpha Vantage data provider."""

    def __init__(self, api_key: str, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config["api_key"] = api_key
        config["base_url"] = "https://www.alphavantage.co/query"
        super().__init__("alpha_vantage", config)

    def get_historical_data(self, request: DataRequest) -> DataResponse:
        """Get historical data from Alpha Vantage."""
        if len(request.symbols) > 1:
            # Handle multiple symbols with threading
            return self._get_multiple_symbols_data(request)

        symbol = request.symbols[0]

        # Map interval to Alpha Vantage format
        interval_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "60min",
            "1d": "daily",
        }

        av_interval = interval_map.get(request.interval, "daily")

        if av_interval in ["1min", "5min", "15min", "30min", "60min"]:
            function = "TIME_SERIES_INTRADAY"
            params = {
                "function": function,
                "symbol": symbol,
                "interval": av_interval,
                "apikey": self.config["api_key"],
                "outputsize": "full",
            }
        else:
            function = "TIME_SERIES_DAILY_ADJUSTED"
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": self.config["api_key"],
                "outputsize": "full",
            }

        response = self._make_request(self.config["base_url"], params)
        data_json = response.json()

        # Parse the response
        if "Error Message" in data_json:
            raise ProviderError(f"Alpha Vantage error: {data_json['Error Message']}")

        if "Note" in data_json:
            raise RateLimitError("Alpha Vantage API rate limit exceeded")

        # Extract time series data
        time_series_key = [k for k in data_json.keys() if "Time Series" in k][0]
        time_series = data_json[time_series_key]

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Standardize column names
        column_map = {
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume",
            "5. adjusted close": "Adj Close",
        }

        df = df.rename(columns=column_map)
        df = df.astype(float)

        # Filter by date range
        df = df[(df.index >= request.start_date) & (df.index <= request.end_date)]

        metadata = {
            "symbols": request.symbols,
            "interval": request.interval,
            "source": "alpha_vantage",
            "function": function,
        }

        return DataResponse(
            data=df, metadata=metadata, provider=self.name, timestamp=datetime.now()
        )

    def get_real_time_data(self, symbols: List[str]) -> DataResponse:
        """Get real-time quotes from Alpha Vantage."""
        # Alpha Vantage real-time implementation
        data_list = []

        for symbol in symbols:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.config["api_key"],
            }

            response = self._make_request(self.config["base_url"], params)
            data_json = response.json()

            if "Global Quote" in data_json:
                quote = data_json["Global Quote"]
                data_list.append(
                    {
                        "symbol": symbol,
                        "price": float(quote["05. price"]),
                        "change": float(quote["09. change"]),
                        "change_percent": quote["10. change percent"].rstrip("%"),
                        "volume": int(quote["06. volume"]),
                        "timestamp": datetime.now(),
                    }
                )

        df = pd.DataFrame(data_list)

        metadata = {"symbols": symbols, "source": "alpha_vantage", "real_time": True}

        return DataResponse(
            data=df, metadata=metadata, provider=self.name, timestamp=datetime.now()
        )

    def get_fundamental_data(self, symbols: List[str]) -> Dict[str, FundamentalData]:
        """Get fundamental data from Alpha Vantage."""
        fundamentals = {}

        for symbol in symbols:
            try:
                params = {
                    "function": "OVERVIEW",
                    "symbol": symbol,
                    "apikey": self.config["api_key"],
                }

                response = self._make_request(self.config["base_url"], params)
                data = response.json()

                if data and "Symbol" in data:
                    fundamentals[symbol] = FundamentalData(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        market_cap=self._safe_float(data.get("MarketCapitalization")),
                        pe_ratio=self._safe_float(data.get("PERatio")),
                        pb_ratio=self._safe_float(data.get("PriceToBookRatio")),
                        dividend_yield=self._safe_float(data.get("DividendYield")),
                        eps=self._safe_float(data.get("EPS")),
                        revenue=self._safe_float(data.get("RevenueTTM")),
                    )

            except Exception as e:
                logger.warning(
                    f"Failed to get Alpha Vantage fundamentals for {symbol}: {e}"
                )
                continue

        return fundamentals

    def _get_multiple_symbols_data(self, request: DataRequest) -> DataResponse:
        """Get data for multiple symbols using threading."""
        results = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for symbol in request.symbols:
                single_request = DataRequest(
                    symbols=[symbol],
                    start_date=request.start_date,
                    end_date=request.end_date,
                    interval=request.interval,
                )
                future = executor.submit(self.get_historical_data, single_request)
                futures.append((symbol, future))

            for symbol, future in futures:
                try:
                    result = future.result()
                    results.append((symbol, result.data))
                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {e}")

        # Combine results
        if results:
            combined_data = pd.concat(
                [data for _, data in results],
                keys=[symbol for symbol, _ in results],
                axis=1,
            )

            metadata = {
                "symbols": request.symbols,
                "interval": request.interval,
                "source": "alpha_vantage",
            }

            return DataResponse(
                data=combined_data,
                metadata=metadata,
                provider=self.name,
                timestamp=datetime.now(),
            )
        else:
            raise ProviderError("No data retrieved for any symbols")

    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float."""
        if value is None or value == "None" or value == "":
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None


class QuandlProvider(DataProvider):
    """Quandl data provider for economic and financial data."""

    def __init__(self, api_key: str, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config["api_key"] = api_key
        config["base_url"] = "https://www.quandl.com/api/v3"
        super().__init__("quandl", config)

    def get_historical_data(self, request: DataRequest) -> DataResponse:
        """Get historical data from Quandl."""
        try:
            # Quandl typically uses database/dataset format
            # For stocks, we'll use the WIKI database (though it's discontinued)
            # This is a basic implementation - real usage would need specific database codes

            if len(request.symbols) > 1:
                return self._get_multiple_symbols_quandl(request)

            symbol = request.symbols[0]

            # Map symbol to Quandl format (example: AAPL -> WIKI/AAPL)
            quandl_code = f"WIKI/{symbol}"

            params = {
                "api_key": self.config["api_key"],
                "start_date": request.start_date.strftime("%Y-%m-%d"),
                "end_date": request.end_date.strftime("%Y-%m-%d"),
                "order": "asc",
            }

            url = f"{self.config['base_url']}/datasets/{quandl_code}/data.json"
            response = self._make_request(url, params)
            data_json = response.json()

            if "dataset_data" not in data_json:
                raise ProviderError(f"Invalid response from Quandl for {symbol}")

            dataset = data_json["dataset_data"]
            columns = dataset["column_names"]
            data_rows = dataset["data"]

            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=columns)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()

            # Standardize column names
            column_map = {
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
                "Adj. Close": "Adj Close",
            }

            df = df.rename(columns=column_map)
            df = df.select_dtypes(include=[np.number])

            metadata = {
                "symbols": request.symbols,
                "interval": request.interval,
                "source": "quandl",
                "database": "WIKI",
            }

            return DataResponse(
                data=df, metadata=metadata, provider=self.name, timestamp=datetime.now()
            )

        except Exception as e:
            raise ProviderError(f"Quandl error: {str(e)}")

    def get_real_time_data(self, symbols: List[str]) -> DataResponse:
        """Quandl doesn't provide real-time data."""
        raise NotImplementedError("Quandl does not provide real-time data")

    def get_fundamental_data(self, symbols: List[str]) -> Dict[str, FundamentalData]:
        """Get fundamental data from Quandl."""
        # Quandl has various fundamental data sources
        # This is a basic implementation
        fundamentals = {}

        for symbol in symbols:
            try:
                # Example: Using Sharadar fundamentals (SF1 database)
                quandl_code = f"SF1/{symbol}_MRQ"  # Most recent quarter

                params = {"api_key": self.config["api_key"], "rows": 1}

                url = f"{self.config['base_url']}/datasets/{quandl_code}/data.json"
                response = self._make_request(url, params)
                data_json = response.json()

                if "dataset_data" in data_json and data_json["dataset_data"]["data"]:
                    data_row = data_json["dataset_data"]["data"][0]
                    columns = data_json["dataset_data"]["column_names"]

                    # Create a dict from the data
                    data_dict = dict(zip(columns, data_row))

                    fundamentals[symbol] = FundamentalData(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        market_cap=data_dict.get("MARKETCAP"),
                        pe_ratio=data_dict.get("PE"),
                        pb_ratio=data_dict.get("PB"),
                        eps=data_dict.get("EPS"),
                        revenue=data_dict.get("REVENUE"),
                    )

            except Exception as e:
                logger.warning(f"Failed to get Quandl fundamentals for {symbol}: {e}")
                continue

        return fundamentals

    def _get_multiple_symbols_quandl(self, request: DataRequest) -> DataResponse:
        """Get data for multiple symbols from Quandl."""
        results = []

        with ThreadPoolExecutor(
            max_workers=3
        ) as executor:  # Quandl has stricter rate limits
            futures = []
            for symbol in request.symbols:
                single_request = DataRequest(
                    symbols=[symbol],
                    start_date=request.start_date,
                    end_date=request.end_date,
                    interval=request.interval,
                )
                future = executor.submit(self.get_historical_data, single_request)
                futures.append((symbol, future))

            for symbol, future in futures:
                try:
                    result = future.result()
                    results.append((symbol, result.data))
                except Exception as e:
                    logger.warning(f"Failed to get Quandl data for {symbol}: {e}")

        if results:
            combined_data = pd.concat(
                [data for _, data in results],
                keys=[symbol for symbol, _ in results],
                axis=1,
            )

            metadata = {
                "symbols": request.symbols,
                "interval": request.interval,
                "source": "quandl",
            }

            return DataResponse(
                data=combined_data,
                metadata=metadata,
                provider=self.name,
                timestamp=datetime.now(),
            )
        else:
            raise ProviderError("No data retrieved from Quandl for any symbols")


class IEXCloudProvider(DataProvider):
    """IEX Cloud data provider for US equities."""

    def __init__(self, api_key: str, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config["api_key"] = api_key
        config["base_url"] = "https://cloud.iexapis.com/stable"
        super().__init__("iex_cloud", config)

    def get_historical_data(self, request: DataRequest) -> DataResponse:
        """Get historical data from IEX Cloud."""
        try:
            if len(request.symbols) > 1:
                return self._get_multiple_symbols_iex(request)

            symbol = request.symbols[0].upper()

            # Map interval to IEX Cloud range
            days_diff = (request.end_date - request.start_date).days

            if days_diff <= 30:
                range_param = "1m"
            elif days_diff <= 90:
                range_param = "3m"
            elif days_diff <= 365:
                range_param = "1y"
            elif days_diff <= 730:
                range_param = "2y"
            else:
                range_param = "5y"

            params = {
                "token": self.config["api_key"],
                "range": range_param,
                "includeToday": "true",
            }

            url = f"{self.config['base_url']}/stock/{symbol}/chart/{range_param}"
            response = self._make_request(url, params)
            data_json = response.json()

            if not data_json:
                raise ProviderError(f"No data returned from IEX Cloud for {symbol}")

            # Convert to DataFrame
            df = pd.DataFrame(data_json)

            # Convert date column
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

            # Standardize column names
            column_map = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }

            df = df.rename(columns=column_map)

            # Filter by date range
            df = df[(df.index >= request.start_date) & (df.index <= request.end_date)]

            # Select only OHLCV columns
            ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
            df = df[[col for col in ohlcv_cols if col in df.columns]]

            metadata = {
                "symbols": request.symbols,
                "interval": request.interval,
                "source": "iex_cloud",
                "range": range_param,
            }

            return DataResponse(
                data=df, metadata=metadata, provider=self.name, timestamp=datetime.now()
            )

        except Exception as e:
            raise ProviderError(f"IEX Cloud error: {str(e)}")

    def get_real_time_data(self, symbols: List[str]) -> DataResponse:
        """Get real-time quotes from IEX Cloud."""
        try:
            data_list = []

            for symbol in symbols:
                symbol = symbol.upper()

                params = {"token": self.config["api_key"]}

                url = f"{self.config['base_url']}/stock/{symbol}/quote"
                response = self._make_request(url, params)
                quote = response.json()

                data_list.append(
                    {
                        "symbol": symbol,
                        "price": quote.get("latestPrice"),
                        "change": quote.get("change"),
                        "change_percent": quote.get("changePercent"),
                        "volume": quote.get("latestVolume"),
                        "market_cap": quote.get("marketCap"),
                        "pe_ratio": quote.get("peRatio"),
                        "timestamp": datetime.now(),
                    }
                )

            df = pd.DataFrame(data_list)

            metadata = {"symbols": symbols, "source": "iex_cloud", "real_time": True}

            return DataResponse(
                data=df, metadata=metadata, provider=self.name, timestamp=datetime.now()
            )

        except Exception as e:
            raise ProviderError(f"IEX Cloud real-time error: {str(e)}")

    def get_fundamental_data(self, symbols: List[str]) -> Dict[str, FundamentalData]:
        """Get fundamental data from IEX Cloud."""
        fundamentals = {}

        for symbol in symbols:
            try:
                symbol = symbol.upper()

                params = {"token": self.config["api_key"]}

                # Get company stats
                url = f"{self.config['base_url']}/stock/{symbol}/stats"
                response = self._make_request(url, params)
                stats = response.json()

                # Get key stats
                url_key_stats = (
                    f"{self.config['base_url']}/stock/{symbol}/advanced-stats"
                )
                response_key = self._make_request(url_key_stats, params)
                key_stats = response_key.json()

                fundamentals[symbol] = FundamentalData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    market_cap=stats.get("marketcap"),
                    pe_ratio=key_stats.get("peRatio"),
                    pb_ratio=key_stats.get("priceToBook"),
                    dividend_yield=key_stats.get("dividendYield"),
                    eps=key_stats.get("ttmEPS"),
                    revenue=key_stats.get("revenue"),
                    net_income=key_stats.get("netIncome"),
                )

            except Exception as e:
                logger.warning(
                    f"Failed to get IEX Cloud fundamentals for {symbol}: {e}"
                )
                continue

        return fundamentals

    def _get_multiple_symbols_iex(self, request: DataRequest) -> DataResponse:
        """Get data for multiple symbols from IEX Cloud."""
        results = []

        with ThreadPoolExecutor(
            max_workers=10
        ) as executor:  # IEX Cloud has good rate limits
            futures = []
            for symbol in request.symbols:
                single_request = DataRequest(
                    symbols=[symbol],
                    start_date=request.start_date,
                    end_date=request.end_date,
                    interval=request.interval,
                )
                future = executor.submit(self.get_historical_data, single_request)
                futures.append((symbol, future))

            for symbol, future in futures:
                try:
                    result = future.result()
                    results.append((symbol, result.data))
                except Exception as e:
                    logger.warning(f"Failed to get IEX Cloud data for {symbol}: {e}")

        if results:
            combined_data = pd.concat(
                [data for _, data in results],
                keys=[symbol for symbol, _ in results],
                axis=1,
            )

            metadata = {
                "symbols": request.symbols,
                "interval": request.interval,
                "source": "iex_cloud",
            }

            return DataResponse(
                data=combined_data,
                metadata=metadata,
                provider=self.name,
                timestamp=datetime.now(),
            )
        else:
            raise ProviderError("No data retrieved from IEX Cloud for any symbols")


class DataProviderManager:
    """Manages multiple data providers with failover capabilities."""

    def __init__(self):
        self.providers: Dict[str, DataProvider] = {}
        self.primary_provider: Optional[str] = None
        self.fallback_order: List[str] = []

    def add_provider(self, provider: DataProvider, is_primary: bool = False):
        """Add a data provider."""
        self.providers[provider.name] = provider

        if is_primary or self.primary_provider is None:
            self.primary_provider = provider.name

        if provider.name not in self.fallback_order:
            self.fallback_order.append(provider.name)

    def get_historical_data(self, request: DataRequest) -> DataResponse:
        """Get historical data with automatic failover."""
        provider_order = [self.primary_provider] + [
            p for p in self.fallback_order if p != self.primary_provider
        ]

        last_error = None

        for provider_name in provider_order:
            if provider_name not in self.providers:
                continue

            provider = self.providers[provider_name]

            try:
                if provider.is_available():
                    return provider.get_historical_data(request)
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}")
                last_error = e
                continue

        raise ProviderError(f"All providers failed. Last error: {last_error}")

    def get_real_time_data(self, symbols: List[str]) -> DataResponse:
        """Get real-time data with automatic failover."""
        provider_order = [self.primary_provider] + [
            p for p in self.fallback_order if p != self.primary_provider
        ]

        last_error = None

        for provider_name in provider_order:
            if provider_name not in self.providers:
                continue

            provider = self.providers[provider_name]

            try:
                if provider.is_available():
                    return provider.get_real_time_data(symbols)
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}")
                last_error = e
                continue

        raise ProviderError(f"All providers failed. Last error: {last_error}")


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, calls_per_second: float = 5.0):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0

    def wait(self):
        """Wait if necessary to respect rate limits."""
        now = time.time()
        time_since_last = now - self.last_call

        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)

        self.last_call = time.time()


class FREDProvider(DataProvider):
    """Federal Reserve Economic Data (FRED) provider."""

    def __init__(self, api_key: str, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config["api_key"] = api_key
        config["base_url"] = "https://api.stlouisfed.org/fred"
        super().__init__("fred", config)

    def get_historical_data(self, request: DataRequest) -> DataResponse:
        """Get economic data from FRED."""
        try:
            if len(request.symbols) > 1:
                return self._get_multiple_series_fred(request)

            series_id = request.symbols[0]

            params = {
                "series_id": series_id,
                "api_key": self.config["api_key"],
                "file_type": "json",
                "observation_start": request.start_date.strftime("%Y-%m-%d"),
                "observation_end": request.end_date.strftime("%Y-%m-%d"),
            }

            url = f"{self.config['base_url']}/series/observations"
            response = self._make_request(url, params)
            data_json = response.json()

            if "observations" not in data_json:
                raise ProviderError(f"Invalid response from FRED for {series_id}")

            observations = data_json["observations"]

            # Convert to DataFrame
            df_data = []
            for obs in observations:
                if obs["value"] != ".":  # FRED uses '.' for missing values
                    df_data.append({"date": obs["date"], "value": float(obs["value"])})

            df = pd.DataFrame(df_data)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df.columns = [series_id]

            metadata = {
                "symbols": request.symbols,
                "source": "fred",
                "series_id": series_id,
            }

            return DataResponse(
                data=df, metadata=metadata, provider=self.name, timestamp=datetime.now()
            )

        except Exception as e:
            raise ProviderError(f"FRED error: {str(e)}")

    def get_real_time_data(self, symbols: List[str]) -> DataResponse:
        """FRED doesn't provide real-time data in the traditional sense."""
        # Get the most recent observation for each series
        try:
            data_list = []

            for series_id in symbols:
                params = {
                    "series_id": series_id,
                    "api_key": self.config["api_key"],
                    "file_type": "json",
                    "limit": 1,
                    "sort_order": "desc",
                }

                url = f"{self.config['base_url']}/series/observations"
                response = self._make_request(url, params)
                data_json = response.json()

                if "observations" in data_json and data_json["observations"]:
                    obs = data_json["observations"][0]
                    if obs["value"] != ".":
                        data_list.append(
                            {
                                "series_id": series_id,
                                "value": float(obs["value"]),
                                "date": obs["date"],
                                "timestamp": datetime.now(),
                            }
                        )

            df = pd.DataFrame(data_list)

            metadata = {
                "symbols": symbols,
                "source": "fred",
                "latest_observations": True,
            }

            return DataResponse(
                data=df, metadata=metadata, provider=self.name, timestamp=datetime.now()
            )

        except Exception as e:
            raise ProviderError(f"FRED latest data error: {str(e)}")

    def get_fundamental_data(self, symbols: List[str]) -> Dict[str, FundamentalData]:
        """FRED doesn't provide traditional fundamental data."""
        raise NotImplementedError("FRED does not provide fundamental company data")

    def _get_multiple_series_fred(self, request: DataRequest) -> DataResponse:
        """Get multiple economic series from FRED."""
        results = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for series_id in request.symbols:
                single_request = DataRequest(
                    symbols=[series_id],
                    start_date=request.start_date,
                    end_date=request.end_date,
                    interval=request.interval,
                )
                future = executor.submit(self.get_historical_data, single_request)
                futures.append((series_id, future))

            for series_id, future in futures:
                try:
                    result = future.result()
                    results.append((series_id, result.data))
                except Exception as e:
                    logger.warning(f"Failed to get FRED data for {series_id}: {e}")

        if results:
            combined_data = pd.concat([data for _, data in results], axis=1)

            metadata = {"symbols": request.symbols, "source": "fred"}

            return DataResponse(
                data=combined_data,
                metadata=metadata,
                provider=self.name,
                timestamp=datetime.now(),
            )
        else:
            raise ProviderError("No data retrieved from FRED for any series")
