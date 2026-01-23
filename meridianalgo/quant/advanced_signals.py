"""
Advanced Quantitative Signals & Analytics

Provides institutional-grade technical signals and time-series analysis tools.
"""

from typing import Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def hurst_exponent(ts: Union[pd.Series, np.ndarray], max_lag: int = 20) -> float:
    """
    Calculate the Hurst exponent of a time series.

    Interpretations:
    - h < 0.5: Anti-persistent (mean reverting)
    - h = 0.5: Random walk (Brownian motion)
    - h > 0.5: Persistent (trending)

    Args:
        ts: Time series data
        max_lag: Maximum lag for R/S analysis

    Returns:
        float: Hurst exponent
    """
    if isinstance(ts, pd.Series):
        ts = ts.values

    # Remove any NaNs
    ts = ts[~np.isnan(ts)]

    if len(ts) < max_lag * 2:
        return 0.5

    lags = range(2, max_lag)
    # Calculate the variance of the differences for each lag
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]

    # Use log-log fit to find Hurst exponent
    # Var(lag) ~ lag^(2H) => Std(lag) ~ lag^H
    # log(Std) ~ H * log(lag)
    tau = np.array(tau)
    lags = np.array(lags)
    mask = tau > 0

    if not any(mask):
        return 0.5

    poly = np.polyfit(np.log(lags[mask]), np.log(tau[mask]), 1)
    return poly[0]


def fractional_difference(
    series: pd.Series, d: float, threshold: float = 1e-4
) -> pd.Series:
    """
    Apply fractional differentiation to a series.

    Used to make a series stationary while preserving as much memory as possible.
    Based on the method by Marcos Lopez de Prado.

    Args:
        series: Input series (e.g., log prices)
        d: Differentiation order (0 < d < 1)
        threshold: Minimum weight threshold to truncate the series

    Returns:
        pd.Series: Fractionally differenced series
    """
    weights = [1.0]
    for k in range(1, len(series)):
        weight = -weights[-1] * (d - k + 1) / k
        if abs(weight) < threshold:
            break
        weights.append(weight)

    weights = np.array(weights[::-1]).reshape(-1, 1)
    series_values = series.values
    res = []

    for i in range(len(weights), len(series_values) + 1):
        res.append(np.dot(weights.T, series_values[i - len(weights) : i])[0])

    return pd.Series(res, index=series.index[len(weights) - 1 :])


def calculate_z_score(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate the rolling z-score of a series.

    Args:
        series: Price or spread series
        window: Rolling window size

    Returns:
        pd.Series: Z-score series
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    return (series - rolling_mean) / rolling_std


def get_half_life(series: pd.Series) -> float:
    """
    Calculate the half-life of mean reversion based on the Ornstein-Uhlenbeck process.

    Args:
        series: Spread or price series

    Returns:
        float: Number of periods for half-life. Returns np.inf if not mean reverting.
    """
    # y_t - y_{t-1} = alpha + lambda * y_{t-1} + epsilon
    z = series.dropna()
    if len(z) < 10:
        return np.inf

    delta_z = z.diff().dropna()
    lagged_z = z.shift(1).dropna()

    # Align both
    common_idx = delta_z.index.intersection(lagged_z.index)
    delta_z = delta_z.loc[common_idx]
    lagged_z = lagged_z.loc[common_idx]

    X = lagged_z.values.reshape(-1, 1)
    y = delta_z.values

    model = LinearRegression().fit(X, y)
    lam = model.coef_[0]

    if lam >= 0:
        return np.inf

    return -np.log(2) / lam


def information_coefficient(signals: pd.Series, returns: pd.Series) -> float:
    """
    Calculate the Information Coefficient (Rank Correlation).

    Args:
        signals: Generated signals
        returns: Forward returns

    Returns:
        float: IC value
    """
    aligned = pd.concat([signals, returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    return aligned.corr(method="spearman").iloc[0, 1]
