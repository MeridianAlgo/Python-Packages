"""
Free data providers that don't require API keys for users who can't afford premium services.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from .exceptions import ProviderError

# Import existing functionality
from .models import DataRequest, DataResponse, FundamentalData
from .providers import DataProvider

logger = logging.getLogger(__name__)


class YahooFinanceFreeProvider(DataProvider):
    """Enhanced Yahoo Finance provider that works without API keys."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("yahoo_finance_free", config)
        self.base_url = "https://query1.finance.yahoo.com"

    def get_historical_data(self, request: DataRequest) -> DataResponse:
        """Get historical data from Yahoo Finance without API key."""
        try:
            import yfinance as yf

            # Use yfinance library (free)
            symbols_str = " ".join(request.symbols)

            data = yf.download(
                symbols_str,
                start=request.start_date.strftime("%Y-%m-%d"),
                end=request.end_date.strftime("%Y-%m-%d"),
                interval=request.interval,
                auto_adjust=True,
                prepost=True,
                threads=True,
                progress=False,  # Disable progress bar
            )

            if data.empty:
                raise ProviderError(f"No data returned for symbols: {request.symbols}")

            # Handle single vs multiple symbols
            if len(request.symbols) == 1:
                if isinstance(data.columns, pd.MultiIndex):
                    # Sometimes yfinance returns MultiIndex even for single symbol
                    data.columns = data.columns.droplevel(1)

                # Ensure standard column names
                expected_cols = ["Open", "High", "Low", "Close", "Volume"]
                data = data.reindex(columns=expected_cols).dropna()

            metadata = {
                "symbols": request.symbols,
                "interval": request.interval,
                "source": "yahoo_finance_free",
                "provider": "yfinance",
                "cost": "free",
            }

            return DataResponse(
                data=data,
                metadata=metadata,
                provider=self.name,
                timestamp=datetime.now(),
            )

        except ImportError:
            # Fallback to direct API calls if yfinance not available
            return self._direct_yahoo_api(request)
        except Exception as e:
            raise ProviderError(f"Yahoo Finance Free error: {str(e)}")

    def _direct_yahoo_api(self, request: DataRequest) -> DataResponse:
        """Direct Yahoo Finance API calls without yfinance."""

        if len(request.symbols) > 1:
            # Handle multiple symbols
            all_data = []
            for symbol in request.symbols:
                single_request = DataRequest(
                    symbols=[symbol],
                    start_date=request.start_date,
                    end_date=request.end_date,
                    interval=request.interval,
                )
                try:
                    single_data = self._get_single_symbol_data(single_request)
                    all_data.append((symbol, single_data))
                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {e}")
                    continue

            if all_data:
                combined_data = pd.concat(
                    [data for _, data in all_data],
                    keys=[symbol for symbol, _ in all_data],
                    axis=1,
                )

                metadata = {
                    "symbols": request.symbols,
                    "interval": request.interval,
                    "source": "yahoo_finance_direct",
                    "cost": "free",
                }

                return DataResponse(
                    data=combined_data,
                    metadata=metadata,
                    provider=self.name,
                    timestamp=datetime.now(),
                )
            else:
                raise ProviderError("No data retrieved for any symbols")
        else:
            data = self._get_single_symbol_data(request)

            metadata = {
                "symbols": request.symbols,
                "interval": request.interval,
                "source": "yahoo_finance_direct",
                "cost": "free",
            }

            return DataResponse(
                data=data,
                metadata=metadata,
                provider=self.name,
                timestamp=datetime.now(),
            )

    def _get_single_symbol_data(self, request: DataRequest) -> pd.DataFrame:
        """Get data for single symbol using direct API."""

        symbol = request.symbols[0]

        # Convert dates to timestamps
        start_ts = int(request.start_date.timestamp())
        end_ts = int(request.end_date.timestamp())

        # Map interval
        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "1d": "1d",
            "1wk": "1wk",
            "1mo": "1mo",
        }
        yahoo_interval = interval_map.get(request.interval, "1d")

        url = f"{self.base_url}/v8/finance/chart/{symbol}"
        params = {
            "period1": start_ts,
            "period2": end_ts,
            "interval": yahoo_interval,
            "includePrePost": "true",
            "events": "div,splits",
        }

        response = self._make_request(url, params)
        data_json = response.json()

        if "chart" not in data_json or not data_json["chart"]["result"]:
            raise ProviderError(f"No data returned for {symbol}")

        result = data_json["chart"]["result"][0]

        # Extract timestamps and OHLCV data
        timestamps = result["timestamp"]
        indicators = result["indicators"]["quote"][0]

        # Create DataFrame
        df_data = {
            "Open": indicators.get("open", []),
            "High": indicators.get("high", []),
            "Low": indicators.get("low", []),
            "Close": indicators.get("close", []),
            "Volume": indicators.get("volume", []),
        }

        # Convert timestamps to datetime
        dates = [datetime.fromtimestamp(ts) for ts in timestamps]

        df = pd.DataFrame(df_data, index=dates)
        df = df.dropna()

        return df

    def get_real_time_data(self, symbols: List[str]) -> DataResponse:
        """Get real-time data using free Yahoo Finance."""
        try:
            import yfinance as yf

            data_list = []
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period="1d", interval="1m")

                    if not hist.empty:
                        latest = hist.iloc[-1]
                        data_list.append(
                            {
                                "symbol": symbol,
                                "price": latest["Close"],
                                "volume": latest["Volume"],
                                "change": info.get("regularMarketChange", 0),
                                "change_percent": info.get(
                                    "regularMarketChangePercent", 0
                                ),
                                "timestamp": hist.index[-1],
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to get real-time data for {symbol}: {e}")
                    continue

            df = pd.DataFrame(data_list)

            metadata = {
                "symbols": symbols,
                "source": "yahoo_finance_free",
                "real_time": True,
                "cost": "free",
            }

            return DataResponse(
                data=df, metadata=metadata, provider=self.name, timestamp=datetime.now()
            )

        except ImportError:
            raise ProviderError("yfinance package required for free real-time data")
        except Exception as e:
            raise ProviderError(f"Yahoo Finance Free real-time error: {str(e)}")

    def get_fundamental_data(self, symbols: List[str]) -> Dict[str, FundamentalData]:
        """Get fundamental data using free Yahoo Finance."""
        fundamentals = {}

        try:
            import yfinance as yf

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

        except ImportError:
            raise ProviderError("yfinance package required for free fundamental data")


class FREDFreeProvider(DataProvider):
    """FRED provider that works without API key (limited functionality)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("fred_free", config)
        self.base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def get_historical_data(self, request: DataRequest) -> DataResponse:
        """Get FRED data without API key using CSV download."""

        if len(request.symbols) > 1:
            # Handle multiple series
            all_data = []
            for series_id in request.symbols:
                try:
                    single_request = DataRequest(
                        symbols=[series_id],
                        start_date=request.start_date,
                        end_date=request.end_date,
                        interval=request.interval,
                    )
                    single_data = self._get_single_series_csv(single_request)
                    all_data.append((series_id, single_data))
                except Exception as e:
                    logger.warning(f"Failed to get FRED data for {series_id}: {e}")
                    continue

            if all_data:
                combined_data = pd.concat([data for _, data in all_data], axis=1)
                combined_data.columns = [series_id for series_id, _ in all_data]
            else:
                raise ProviderError("No FRED data retrieved")
        else:
            combined_data = self._get_single_series_csv(request)
            combined_data.columns = request.symbols

        metadata = {
            "symbols": request.symbols,
            "interval": request.interval,
            "source": "fred_free",
            "cost": "free",
        }

        return DataResponse(
            data=combined_data,
            metadata=metadata,
            provider=self.name,
            timestamp=datetime.now(),
        )

    def _get_single_series_csv(self, request: DataRequest) -> pd.DataFrame:
        """Get single FRED series using CSV download."""

        series_id = request.symbols[0]

        params = {
            "id": series_id,
            "cosd": request.start_date.strftime("%Y-%m-%d"),
            "coed": request.end_date.strftime("%Y-%m-%d"),
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            # Parse CSV data
            from io import StringIO

            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data, index_col=0, parse_dates=True)

            # Clean data (FRED uses '.' for missing values)
            df = df.replace(".", np.nan)
            df = df.astype(float)
            df = df.dropna()

            return df

        except Exception as e:
            raise ProviderError(f"FRED Free error for {series_id}: {str(e)}")

    def get_real_time_data(self, symbols: List[str]) -> DataResponse:
        """FRED doesn't provide real-time data."""
        raise NotImplementedError("FRED does not provide real-time data")

    def get_fundamental_data(self, symbols: List[str]) -> Dict[str, FundamentalData]:
        """FRED doesn't provide fundamental data."""
        raise NotImplementedError("FRED does not provide fundamental data")


class SimulatedDataProvider(DataProvider):
    """Simulated data provider for testing and development when no internet/APIs available."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("simulated", config)
        self.random_seed = config.get("random_seed", 42) if config else 42
        np.random.seed(self.random_seed)

    def get_historical_data(self, request: DataRequest) -> DataResponse:
        """Generate simulated historical data."""

        # Generate date range
        date_range = pd.date_range(
            start=request.start_date,
            end=request.end_date,
            freq="D" if request.interval == "1d" else "H",
        )

        all_data = {}

        for symbol in request.symbols:
            # Generate realistic price data using geometric Brownian motion
            n_periods = len(date_range)

            # Parameters for simulation
            initial_price = 100.0 + np.random.uniform(-50, 50)  # Random starting price
            drift = np.random.uniform(-0.1, 0.15)  # Annual drift
            volatility = np.random.uniform(0.15, 0.4)  # Annual volatility

            # Time step
            dt = 1 / 252 if request.interval == "1d" else 1 / (252 * 24)

            # Generate price path
            returns = np.random.normal(drift * dt, volatility * np.sqrt(dt), n_periods)

            prices = [initial_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))

            prices = np.array(prices)

            # Generate OHLCV data
            # Open = previous close (with small gap)
            opens = prices * (1 + np.random.normal(0, 0.001, n_periods))

            # High/Low based on intraday volatility
            intraday_vol = volatility * 0.3  # Intraday vol is typically lower
            highs = prices * (
                1 + np.abs(np.random.normal(0, intraday_vol / 4, n_periods))
            )
            lows = prices * (
                1 - np.abs(np.random.normal(0, intraday_vol / 4, n_periods))
            )

            # Ensure OHLC consistency
            highs = np.maximum(highs, np.maximum(opens, prices))
            lows = np.minimum(lows, np.minimum(opens, prices))

            # Volume (log-normal distribution)
            base_volume = np.random.uniform(100000, 10000000)
            volumes = np.random.lognormal(np.log(base_volume), 0.5, n_periods).astype(
                int
            )

            # Create DataFrame
            symbol_data = pd.DataFrame(
                {
                    "Open": opens,
                    "High": highs,
                    "Low": lows,
                    "Close": prices,
                    "Volume": volumes,
                },
                index=date_range,
            )

            all_data[symbol] = symbol_data

        # Combine data
        if len(request.symbols) == 1:
            data = all_data[request.symbols[0]]
        else:
            data = pd.concat(all_data.values(), keys=all_data.keys(), axis=1)

        metadata = {
            "symbols": request.symbols,
            "interval": request.interval,
            "source": "simulated",
            "cost": "free",
            "random_seed": self.random_seed,
        }

        return DataResponse(
            data=data, metadata=metadata, provider=self.name, timestamp=datetime.now()
        )

    def get_real_time_data(self, symbols: List[str]) -> DataResponse:
        """Generate simulated real-time data."""

        data_list = []

        for symbol in symbols:
            # Generate realistic current price
            base_price = 100.0 + np.random.uniform(-50, 50)
            current_price = base_price * (1 + np.random.normal(0, 0.02))

            data_list.append(
                {
                    "symbol": symbol,
                    "price": current_price,
                    "volume": np.random.randint(10000, 1000000),
                    "change": np.random.normal(0, 1.0),
                    "change_percent": np.random.normal(0, 0.02),
                    "timestamp": datetime.now(),
                }
            )

        df = pd.DataFrame(data_list)

        metadata = {
            "symbols": symbols,
            "source": "simulated",
            "real_time": True,
            "cost": "free",
        }

        return DataResponse(
            data=df, metadata=metadata, provider=self.name, timestamp=datetime.now()
        )

    def get_fundamental_data(self, symbols: List[str]) -> Dict[str, FundamentalData]:
        """Generate simulated fundamental data."""

        fundamentals = {}

        for symbol in symbols:
            fundamentals[symbol] = FundamentalData(
                symbol=symbol,
                timestamp=datetime.now(),
                market_cap=np.random.uniform(1e9, 1e12),  # $1B to $1T
                pe_ratio=np.random.uniform(5, 50),
                pb_ratio=np.random.uniform(0.5, 10),
                dividend_yield=np.random.uniform(0, 0.08),
                eps=np.random.uniform(-5, 20),
                revenue=np.random.uniform(1e8, 1e11),
                net_income=np.random.uniform(-1e9, 1e10),
            )

        return fundamentals


class FreeDataProviderManager:
    """Manager for free data providers with intelligent fallbacks."""

    def __init__(self):
        self.providers = {}
        self.provider_priority = []

        # Initialize free providers
        self._setup_free_providers()

    def _setup_free_providers(self):
        """Setup free data providers in priority order."""

        # Yahoo Finance (free, most reliable)
        yahoo_free = YahooFinanceFreeProvider()
        self.add_provider(yahoo_free, priority=1)

        # FRED Free (for economic data)
        fred_free = FREDFreeProvider()
        self.add_provider(fred_free, priority=2)

        # Simulated data (always available fallback)
        simulated = SimulatedDataProvider()
        self.add_provider(simulated, priority=99)  # Lowest priority

        logger.info("Free data providers initialized")

    def add_provider(self, provider: DataProvider, priority: int = 50):
        """Add provider with priority."""
        self.providers[provider.name] = provider

        # Insert in priority order
        inserted = False
        for i, (existing_name, existing_priority) in enumerate(self.provider_priority):
            if priority < existing_priority:
                self.provider_priority.insert(i, (provider.name, priority))
                inserted = True
                break

        if not inserted:
            self.provider_priority.append((provider.name, priority))

    def get_data(self, request: DataRequest) -> DataResponse:
        """Get data with automatic fallback to free providers."""

        last_error = None

        for provider_name, priority in self.provider_priority:
            provider = self.providers[provider_name]

            try:
                logger.info(f"Trying free provider: {provider_name}")
                return provider.get_historical_data(request)

            except Exception as e:
                last_error = e
                logger.warning(f"Free provider {provider_name} failed: {e}")
                continue

        raise ProviderError(f"All free providers failed. Last error: {last_error}")

    def get_available_symbols(self) -> Dict[str, List[str]]:
        """Get commonly available free symbols."""

        return {
            "us_stocks": [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "TSLA",
                "META",
                "NVDA",
                "JPM",
                "JNJ",
                "V",
                "PG",
                "UNH",
                "HD",
                "MA",
                "DIS",
                "PYPL",
                "ADBE",
                "NFLX",
            ],
            "etfs": [
                "SPY",
                "QQQ",
                "IWM",
                "VTI",
                "VEA",
                "VWO",
                "AGG",
                "TLT",
                "GLD",
                "SLV",
            ],
            "indices": ["^GSPC", "^IXIC", "^DJI", "^RUT", "^VIX"],
            "currencies": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCAD=X", "AUDUSD=X"],
            "commodities": [
                "GC=F",
                "SI=F",
                "CL=F",
                "NG=F",
                "ZC=F",
                "ZS=F",  # Gold, Silver, Oil, Gas, Corn, Soybeans
            ],
            "economic_indicators": [
                "GDP",
                "UNRATE",
                "CPIAUCSL",
                "FEDFUNDS",
                "DGS10",
                "DGS2",
            ],
        }


# Create a comprehensive free data solution
def create_free_data_setup() -> Dict[str, Any]:
    """Create a complete free data setup for users without API access."""

    setup = {
        "providers": FreeDataProviderManager(),
        "sample_symbols": FreeDataProviderManager().get_available_symbols(),
        "installation_guide": {
            "required_packages": [
                "yfinance",  # Free Yahoo Finance data
                "pandas",
                "numpy",
                "requests",
                "matplotlib",  # For basic plotting
                "seaborn",  # For statistical plots
            ],
            "optional_packages": [
                "plotly",  # For interactive charts
                "dash",  # For dashboards
                "jupyter",  # For notebooks
                "scikit-learn",  # For ML
                "ta-lib",  # For technical indicators
            ],
            "install_commands": [
                "pip install yfinance pandas numpy requests matplotlib seaborn",
                "pip install plotly dash jupyter scikit-learn",
                "pip install TA-Lib  # May require additional setup on some systems",
            ],
        },
        "usage_examples": {
            "basic_data_fetch": """
# Get free stock data
from meridianalgo.data.free_providers import FreeDataProviderManager
from meridianalgo.data.models import DataRequest
from datetime import datetime, timedelta

# Setup free data manager
data_manager = FreeDataProviderManager()

# Create request
request = DataRequest(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date=datetime.now() - timedelta(days=365),
    end_date=datetime.now(),
    interval='1d'
)

# Get data (completely free)
data_response = data_manager.get_data(request)
print(f"Retrieved {len(data_response.data)} days of data for free!")
            """,
            "technical_analysis": """
# Free technical analysis
from meridianalgo.technical_analysis.indicators import TechnicalIndicators
from meridianalgo.technical_analysis.visualization import TechnicalChart

# Calculate indicators (no API required)
indicators = TechnicalIndicators()
rsi = indicators.rsi(data_response.data['Close'], period=14)
sma_20 = indicators.sma(data_response.data['Close'], period=20)

# Create free interactive chart
chart = TechnicalChart("Free Technical Analysis")
chart.set_data(data_response.data)
chart.add_candlestick()
chart.add_line(sma_20, "SMA 20", color="blue")
chart.show()  # Works without any paid services
            """,
            "portfolio_optimization": """
# Free portfolio optimization
from meridianalgo.portfolio.optimization import PortfolioOptimizer
import pandas as pd

# Calculate returns (free)
returns = data_response.data.pct_change().dropna()
expected_returns = returns.mean() * 252  # Annualized
cov_matrix = returns.cov() * 252  # Annualized

# Optimize portfolio (no paid services required)
optimizer = PortfolioOptimizer()
result = optimizer.optimize(
    expected_returns, 
    cov_matrix, 
    objective='max_sharpe'
)

print(f"Optimal weights: {result.weights}")
print(f"Expected Sharpe ratio: {result.sharpe_ratio:.2f}")
            """,
        },
    }

    return setup
