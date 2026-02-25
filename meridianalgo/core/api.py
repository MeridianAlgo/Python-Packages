"""
Unified API for MeridianAlgo - Ultimate Quantitative Development Platform.

This module provides a consistent, high-level interface to all MeridianAlgo functionality.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import __version__
from .config import get_config, set_config

# Core imports
from .core import (
    PortfolioOptimizer,
    TimeSeriesAnalyzer,
    calculate_calmar_ratio,
    calculate_expected_shortfall,
    calculate_max_drawdown,
    calculate_sortino_ratio,
    calculate_value_at_risk,
)
from .core.statistics import StatisticalArbitrage

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_RISK_FREE_RATE = 0.0
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_WINDOW = 21

try:
    # signals module available but not directly imported
    TECHNICAL_AVAILABLE = True
except ImportError:
    TECHNICAL_AVAILABLE = False


# Type aliases
Numeric = Union[int, float, np.number]
DateLike = Union[str, datetime, pd.Timestamp]
Symbol = Union[str, List[str]]
OptionType = Literal["call", "put"]


class MeridianAlgoAPI:
    """
    Unified API for accessing all MeridianAlgo functionality.

    This class provides a consistent interface to all features of the MeridianAlgo
    library, including data loading, portfolio optimization, risk analysis, and more.

    Example:
        >>> api = MeridianAlgoAPI()
        >>> data = api.get_market_data(['AAPL', 'MSFT'], '2020-01-01', '2021-01-01')
        >>> returns = data.pct_change().dropna()
        >>> weights = api.optimize_portfolio(returns, method='sharpe')
        >>> metrics = api.calculate_risk_metrics(returns @ pd.Series(weights))
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the API with optional configuration.

        Args:
            config: Optional dictionary of configuration parameters to override defaults.
                   See `meridianalgo.config` for available options.
        """
        if config:
            set_config(**config)

        # Initialize modules
        self._portfolio_optimizer: Optional[PortfolioOptimizer] = None
        self._time_series_analyzer: Optional[TimeSeriesAnalyzer] = None
        self._stat_arb: Optional[StatisticalArbitrage] = None

        logger.info(f"MeridianAlgo API v{__version__} initialized")

    @property
    def portfolio_optimizer(self) -> PortfolioOptimizer:
        """Lazy-loading property for portfolio optimizer."""
        if self._portfolio_optimizer is None:
            self._portfolio_optimizer = PortfolioOptimizer()
        return self._portfolio_optimizer

    @property
    def time_series_analyzer(self) -> TimeSeriesAnalyzer:
        """Lazy-loading property for time series analyzer."""
        if self._time_series_analyzer is None:
            self._time_series_analyzer = TimeSeriesAnalyzer()
        return self._time_series_analyzer

    @property
    def stat_arb(self) -> StatisticalArbitrage:
        """Lazy-loading property for statistical arbitrage."""
        if self._stat_arb is None:
            self._stat_arb = StatisticalArbitrage()
        return self._stat_arb

    def get_market_data(
        self,
        symbols: Union[str, List[str]],
        start_date: DateLike,
        end_date: Optional[DateLike] = None,
        interval: str = "1d",
        **kwargs,
    ) -> pd.DataFrame:
        """Fetch market data for the given symbols and date range.

        Args:
            symbols: Single ticker symbol or list of ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format or datetime object
            end_date: End date in 'YYYY-MM-DD' format or datetime object.
                     If None, uses current date.
            interval: Data interval ('1d', '1h', '1m', etc.)
            **kwargs: Additional arguments passed to the data provider

        Returns:
            DataFrame with adjusted close prices for the given symbols

        Raises:
            ValueError: If symbols is empty or invalid
            DataError: If data cannot be fetched
        """
        if not symbols:
            raise ValueError("At least one symbol must be provided")

        if isinstance(symbols, str):
            symbols = [symbols]

        logger.info(
            f"Fetching {interval} data for {', '.join(symbols)} from {start_date} to {end_date}"
        )

        try:
            data = get_market_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                **kwargs,
            )
            return data
        except Exception as e:
            logger.error(f"Failed to fetch market data: {str(e)}")
            raise DataError(f"Failed to fetch market data: {str(e)}") from e

    def optimize_portfolio(
        self,
        returns: pd.DataFrame,
        method: str = "sharpe",
        risk_free_rate: Optional[float] = None,
        target_return: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Optimize portfolio weights using the specified method.

        Args:
            returns: DataFrame of asset returns (tickers as columns)
            method: Optimization method ('sharpe', 'min_vol', 'efficient_risk', 'efficient_return')
            risk_free_rate: Annual risk-free rate (default: from config)
            target_return: Target return for 'efficient_risk' method
            **kwargs: Additional arguments passed to the optimizer

        Returns:
            Dictionary of {ticker: weight} pairs

        Raises:
            ValueError: If inputs are invalid
            OptimizationError: If optimization fails
        """
        if risk_free_rate is None:
            risk_free_rate = get_config("risk_free_rate", DEFAULT_RISK_FREE_RATE)

        try:
            optimizer = PortfolioOptimizer(returns)
            result = optimizer.optimize_portfolio(
                method=method,
                risk_free_rate=risk_free_rate,
                target_return=target_return,
                **kwargs,
            )
            return result["weights"]
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {str(e)}")
            raise OptimizationError(f"Portfolio optimization failed: {str(e)}") from e

    def calculate_risk_metrics(
        self,
        returns: Union[pd.Series, pd.DataFrame],
        risk_free_rate: Optional[float] = None,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Calculate comprehensive risk metrics for the given returns.

        Args:
            returns: Series or DataFrame of returns
            risk_free_rate: Annual risk-free rate (default: from config)
            confidence_level: Confidence level for VaR and CVaR (0 < level < 1)

        Returns:
            Dictionary of risk metrics

        Raises:
            ValueError: If inputs are invalid
        """
        if risk_free_rate is None:
            risk_free_rate = get_config("risk_free_rate", DEFAULT_RISK_FREE_RATE)

        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")

        if isinstance(returns, pd.DataFrame):
            # Calculate metrics for each column
            return {
                col: self.calculate_risk_metrics(
                    returns[col],
                    risk_free_rate=risk_free_rate,
                    confidence_level=confidence_level,
                )
                for col in returns.columns
            }

        metrics = {
            "annualized_return": (1 + returns).prod() ** (252 / len(returns)) - 1,
            "annualized_volatility": returns.std() * np.sqrt(252),
            "sharpe_ratio": (returns.mean() - risk_free_rate / 252)
            / returns.std()
            * np.sqrt(252),
            "sortino_ratio": calculate_sortino_ratio(
                returns, risk_free_rate=risk_free_rate
            ),
            "max_drawdown": calculate_max_drawdown(returns),
            "calmar_ratio": calculate_calmar_ratio(returns),
            "var_95": calculate_value_at_risk(
                returns, confidence_level=confidence_level
            ),
            "cvar_95": calculate_expected_shortfall(
                returns, confidence_level=confidence_level
            ),
        }

        return metrics

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        if not TECHNICAL_AVAILABLE:
            raise ImportError("Technical indicators not available")
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(
        self, prices: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        if not TECHNICAL_AVAILABLE:
            raise ImportError("Technical indicators not available")
        fast_ema = prices.ewm(span=12, adjust=False).mean()
        slow_ema = prices.ewm(span=26, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=9, adjust=False).mean()

        return macd, signal, macd - signal

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and capabilities."""
        import platform
        import sys

        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "available_modules": self.available_modules,
            "package_version": "4.0.2",
        }


# Global API instance
_api_instance = None


def get_api() -> MeridianAlgoAPI:
    """Get the global API instance."""
    global _api_instance
    if _api_instance is None:
        _api_instance = MeridianAlgoAPI()
    return _api_instance


# Convenience functions
def get_market_data(
    symbols: List[str], start_date: str, end_date: str = None
) -> pd.DataFrame:
    """Get market data for specified symbols."""
    api = get_api()
    return api.get_market_data(symbols, start_date, end_date)


def optimize_portfolio(
    returns: pd.DataFrame, method: str = "sharpe", **kwargs
) -> Dict[str, float]:
    """Optimize portfolio using specified method."""
    return get_api().optimize_portfolio(returns, method, **kwargs)


def calculate_risk_metrics(returns: pd.Series) -> Dict[str, float]:
    """Calculate comprehensive risk metrics."""
    return get_api().calculate_risk_metrics(returns)


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    return get_api().calculate_rsi(prices, period)


def calculate_macd(prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD indicator."""
    return get_api().calculate_macd(prices)


def price_option(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call",
) -> float:
    """Price an option using Black-Scholes model."""
    import math

    from scipy.stats import norm

    d1 = (
        math.log(spot / strike)
        + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry
    ) / (volatility * math.sqrt(time_to_expiry))
    d2 = d1 - volatility * math.sqrt(time_to_expiry)

    if option_type.lower() == "call":
        price = spot * norm.cdf(d1) - strike * math.exp(
            -risk_free_rate * time_to_expiry
        ) * norm.cdf(d2)
    else:  # put
        price = strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(
            -d2
        ) - spot * norm.cdf(-d1)

    return price


# Custom exceptions
class MeridianAlgoError(Exception):
    """Base class for all MeridianAlgo exceptions."""

    pass


class DataError(MeridianAlgoError):
    """Raised when there is an error with data retrieval or processing."""

    pass


class OptimizationError(MeridianAlgoError):
    """Raised when portfolio optimization fails."""

    pass


class ValidationError(MeridianAlgoError):
    """Raised when input validation fails."""

    pass
