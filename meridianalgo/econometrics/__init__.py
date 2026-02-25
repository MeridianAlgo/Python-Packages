"""
MeridianAlgo Econometrics Module

Advanced statistical and econometric tools for quantitative finance, including
time series analysis, volatility modeling, and cointegration testing.
"""

import logging
from typing import Any, Dict, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.stattools import adfuller, coint, kpss

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning(
        "statsmodels not installed, Econometrics module functionality will be limited"
    )

try:
    from arch import arch_model

    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning("arch not installed, GARCH modeling will be unavailable")


class TimeSeriesTests:
    """
    Standard statistical tests for time series data.
    """

    @staticmethod
    def check_stationarity(
        series: Union[pd.Series, np.ndarray], method: str = "adf"
    ) -> Dict[str, Any]:
        """
        Check if a time series is stationary.

        Args:
            series: Time series data
            method: Test method ('adf', 'kpss')

        Returns:
            Test results including p-value and boolean stationarity flag
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for stationarity tests")

        if method == "adf":
            result = adfuller(series)
            return {
                "method": "Augmented Dickey-Fuller",
                "statistic": result[0],
                "p_value": result[1],
                "is_stationary": result[1] < 0.05,
                "critical_values": result[4],
            }
        elif method == "kpss":
            result = kpss(series)
            return {
                "method": "KPSS",
                "statistic": result[0],
                "p_value": result[1],
                "is_stationary": result[1] > 0.05,  # KPSS null is stationarity
                "critical_values": result[3],
            }
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def test_cointegration(y: pd.Series, x: pd.Series) -> Dict[str, Any]:
        """
        Test for cointegration between two series (Engle-Granger).
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for cointegration tests")

        score, p_value, _ = coint(y, x)
        return {
            "method": "Engle-Granger",
            "score": score,
            "p_value": p_value,
            "is_cointegrated": p_value < 0.05,
        }


class VolatilityModeler:
    """
    Advanced volatility modeling (GARCH, etc.)
    """

    @staticmethod
    def fit_garch(
        returns: Union[pd.Series, np.ndarray],
        p: int = 1,
        q: int = 1,
        dist: str = "normal",
    ) -> Dict[str, Any]:
        """
        Fit a GARCH(p, q) model to returns.
        """
        if not ARCH_AVAILABLE:
            raise ImportError("arch package is required for GARCH modeling")

        model = arch_model(returns, vol="Garch", p=p, q=q, dist=dist)
        res = model.fit(disp="off")

        return {
            "params": res.params.to_dict(),
            "volatility": res.conditional_volatility,
            "forecast": res.forecast(horizon=5).variance.iloc[-1].values,
            "aic": res.aic,
            "bic": res.bic,
            "model": res,
        }


class OrderBookDynamics:
    """
    Market microstructure and order book analysis.
    """

    @staticmethod
    def calculate_vpin(
        buys: np.ndarray, sells: np.ndarray, bucket_size: int = 1000
    ) -> np.ndarray:
        """
        Calculate Volume-Synchronized Probability of Informed Trading (VPIN).
        """
        total_volume = buys + sells
        n_buckets = len(total_volume) // bucket_size

        vpin_values = []
        for i in range(n_buckets):
            start = i * bucket_size
            end = (i + 1) * bucket_size

            bucket_buys = buys[start:end].sum()
            bucket_sells = sells[start:end].sum()

            vpin = abs(bucket_buys - bucket_sells) / bucket_size
            vpin_values.append(vpin)

        return np.array(vpin_values)


__all__ = [
    "TimeSeriesTests",
    "VolatilityModeler",
    "OrderBookDynamics",
    "STATSMODELS_AVAILABLE",
    "ARCH_AVAILABLE",
]
