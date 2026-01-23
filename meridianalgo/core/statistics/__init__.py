"""
Statistical analysis module.

This module provides tools for statistical analysis of financial time series.
"""

from typing import Dict, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats  # noqa: F401


class StatisticalArbitrage:
    """Statistical arbitrage strategy implementation."""

    def __init__(self, data: pd.DataFrame):
        """Initialize with price data.

        Args:
            data: DataFrame with price data (tickers as columns)
        """
        self.data = data

    def calculate_zscore(self, window: int = 21) -> pd.Series:
        """Calculate z-score of the price series."""
        rolling_mean = self.data.rolling(window=window).mean()
        rolling_std = self.data.rolling(window=window).std()
        return (self.data - rolling_mean) / rolling_std

    def calculate_rolling_correlation(self, window: int = 21) -> pd.DataFrame:
        """Calculate rolling correlation between assets in the data."""
        returns = self.data.pct_change().dropna()
        return returns.rolling(window=window).corr()

    def calculate_cointegration(self, x: pd.Series, y: pd.Series) -> Dict[str, float]:
        """Test for cointegration between two time series."""
        from statsmodels.tsa.stattools import coint

        df = pd.DataFrame({"x": x, "y": y}).dropna()
        score, pvalue, _ = coint(df["x"], df["y"])

        return {
            "score": score,
            "p_value": pvalue,
            "pvalue": pvalue,
            "is_cointegrated": bool(pvalue < 0.05),
        }

    def test_cointegration(self, x: pd.Series, y: pd.Series) -> Dict[str, float]:
        """Test for cointegration between two time series (alias for calculate_cointegration)."""
        metrics = self.calculate_cointegration(x, y)
        # Add test_statistic key if not present (sometimes expected as score)
        if "score" in metrics and "test_statistic" not in metrics:
            metrics["test_statistic"] = metrics["score"]
        return metrics


def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix for returns."""
    return returns.corr()


def calculate_rolling_correlation(
    returns: pd.DataFrame, window: int = 21
) -> pd.DataFrame:
    """Calculate rolling correlation between assets."""
    return returns.rolling(window=window).corr()


def calculate_hurst_exponent(
    time_series: Union[pd.Series, np.ndarray], max_lag: int = 20
) -> float:
    """Calculate the Hurst exponent of a time series."""
    # Convert to numpy array if it's a pandas Series
    ts = (
        time_series.values
        if hasattr(time_series, "values")
        else np.asarray(time_series)
    )

    lags = range(2, max_lag + 1)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    return np.polyfit(np.log(lags), np.log(tau), 1)[0]


# Alias for backward compatibility
hurst_exponent = calculate_hurst_exponent


def calculate_half_life(price_series: pd.Series) -> float:
    """Calculate the half-life of a mean-reverting time series."""
    delta_p = price_series.diff().dropna()
    lag_p = price_series.shift(1).dropna()

    if len(delta_p) != len(lag_p):
        min_len = min(len(delta_p), len(lag_p))
        delta_p = delta_p.iloc[-min_len:]
        lag_p = lag_p.iloc[-min_len:]

    X = sm.add_constant(lag_p)
    model = sm.OLS(delta_p, X)
    results = model.fit()
    # Access the slope parameter (second parameter after const)
    slope = results.params.iloc[1] if len(results.params) > 1 else results.params[1]
    return -np.log(2) / slope


def calculate_autocorrelation(series: pd.Series, lag: int = 1) -> float:
    """Calculate autocorrelation for a given lag."""
    return series.autocorr(lag=lag)


def rolling_volatility(
    returns: pd.Series, window: int = 21, annualized: bool = True
) -> pd.Series:
    """Calculate rolling volatility.

    Args:
        returns: Series of returns
        window: Rolling window size
        annualized: Whether to annualize the result

    Returns:
        Series of rolling volatility values
    """
    vol = returns.rolling(window=window).std()
    if annualized:
        vol = vol * np.sqrt(252)
    return vol
