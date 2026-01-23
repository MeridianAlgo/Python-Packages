"""
Time series analysis module.

This module provides tools for analyzing financial time series data.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class TimeSeriesAnalyzer:
    """Time series analysis for financial data."""

    def __init__(self, data: pd.DataFrame):
        """Initialize with time series data.

        Args:
            data: DataFrame with datetime index and price/return data
        """
        self.data = data

    def calculate_returns(self, log_returns: bool = False) -> pd.Series:
        """Calculate returns from price data.

        Args:
            log_returns: If True, calculate log returns (default: False)

        Returns:
            Series of returns
        """
        if log_returns:
            return np.log(self.data / self.data.shift(1)).dropna()
        return self.data.pct_change().dropna()

    def calculate_volatility(
        self, window: int = 21, annualized: bool = True
    ) -> pd.Series:
        """Calculate rolling volatility.

        Args:
            window: Rolling window size in periods (default: 21 for 1 month)
            annualized: If True, annualize the volatility (default: True)

        Returns:
            Series of volatility values
        """
        returns = self.calculate_returns()
        volatility = returns.rolling(window=window).std()

        if annualized:
            # Annualize the volatility (assuming daily data)
            volatility = volatility * np.sqrt(252)

        return volatility

    def calculate_moving_average(
        self, window: int = 20, ma_type: str = "sma"
    ) -> pd.Series:
        """Calculate moving average.

        Args:
            window: Window size for the moving average
            ma_type: Type of moving average ('sma' or 'ema')

        Returns:
            Series containing the moving average
        """
        if ma_type.lower() == "sma":
            return self.data.rolling(window=window).mean()
        elif ma_type.lower() == "ema":
            return self.data.ewm(span=window, adjust=False).mean()
        else:
            raise ValueError(f"Unsupported moving average type: {ma_type}")

    def calculate_bollinger_bands(
        self, window: int = 20, num_std: float = 2.0
    ) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands.

        Args:
            window: Window size for the moving average
            num_std: Number of standard deviations for the bands

        Returns:
            Dictionary with 'middle', 'upper', and 'lower' bands
        """
        middle_band = self.calculate_moving_average(window, "sma")
        rolling_std = self.data.rolling(window=window).std()

        return {
            "middle": middle_band,
            "upper": middle_band + (rolling_std * num_std),
            "lower": middle_band - (rolling_std * num_std),
        }


def get_market_data(
    tickers: List[str], start_date: str = "2020-01-01", end_date: Optional[str] = None
) -> pd.DataFrame:
    """Fetch market data from Yahoo Finance.

    Args:
        tickers: List of ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format (default: '2020-01-01')
        end_date: End date in 'YYYY-MM-DD' format (default: today)

    Returns:
        DataFrame with adjusted close prices for the given tickers
    """
    from datetime import datetime

    import yfinance as yf

    if not end_date:
        end_date = datetime.today().strftime("%Y-%m-%d")

    data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
    return data
