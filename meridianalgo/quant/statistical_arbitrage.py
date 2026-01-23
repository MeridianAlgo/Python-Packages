"""
Statistical Arbitrage Module

Advanced statistical arbitrage strategies including pairs trading,
cointegration analysis, and mean reversion models.
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


class PairsTrading:
    """
    Comprehensive pairs trading strategy implementation.
    """

    def __init__(
        self,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        stop_loss: float = 4.0,
    ):
        """
        Initialize pairs trading strategy.

        Parameters:
        -----------
        entry_threshold : float
            Z-score threshold for entering positions
        exit_threshold : float
            Z-score threshold for exiting positions
        stop_loss : float
            Z-score threshold for stop loss
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.hedge_ratio = None
        self.spread_mean = None
        self.spread_std = None

    def calculate_hedge_ratio(
        self, series1: pd.Series, series2: pd.Series, method: str = "ols"
    ) -> float:
        """
        Calculate hedge ratio between two price series.

        Parameters:
        -----------
        series1 : pd.Series
            First price series (dependent variable)
        series2 : pd.Series
            Second price series (independent variable)
        method : str
            Method for calculating hedge ratio ('ols', 'tls', 'dynamic')

        Returns:
        --------
        float
            Hedge ratio (beta)
        """
        if method == "ols":
            # Ordinary Least Squares
            from sklearn.linear_model import LinearRegression

            X = series2.values.reshape(-1, 1)
            y = series1.values
            model = LinearRegression()
            model.fit(X, y)
            return model.coef_[0]

        elif method == "tls":
            # Total Least Squares (accounts for noise in both series)
            X = np.column_stack([series2.values, np.ones(len(series2))])
            y = series1.values

            # Solve using SVD
            U, s, Vt = np.linalg.svd(np.column_stack([X, y.reshape(-1, 1)]))
            V = Vt.T
            beta = -V[:-1, -1] / V[-1, -1]
            return beta[0]

        elif method == "dynamic":
            # Kalman filter for time-varying hedge ratio
            return self._kalman_hedge_ratio(series1, series2)

        else:
            raise ValueError(f"Unknown method: {method}")

    def _kalman_hedge_ratio(self, series1: pd.Series, series2: pd.Series) -> float:
        """
        Estimate time-varying hedge ratio using Kalman filter.
        """
        # Simplified Kalman filter implementation
        n = len(series1)
        beta = np.zeros(n)
        P = np.zeros(n)

        # Initialize
        beta[0] = series1.iloc[0] / series2.iloc[0] if series2.iloc[0] != 0 else 1.0
        P[0] = 1.0

        # Kalman filter parameters
        Q = 0.0001  # Process noise
        R = 1.0  # Measurement noise

        for t in range(1, n):
            # Prediction
            beta_pred = beta[t - 1]
            P_pred = P[t - 1] + Q

            # Update
            y_pred = beta_pred * series2.iloc[t]
            innovation = series1.iloc[t] - y_pred

            S = series2.iloc[t] ** 2 * P_pred + R
            K = P_pred * series2.iloc[t] / S if S != 0 else 0

            beta[t] = beta_pred + K * innovation
            P[t] = (1 - K * series2.iloc[t]) * P_pred

        return beta[-1]  # Return most recent estimate

    def calculate_spread(
        self,
        series1: pd.Series,
        series2: pd.Series,
        hedge_ratio: Optional[float] = None,
    ) -> pd.Series:
        """
        Calculate spread between two price series.

        Spread = Series1 - (Hedge_Ratio * Series2)
        """
        if hedge_ratio is None:
            hedge_ratio = self.calculate_hedge_ratio(series1, series2)

        self.hedge_ratio = hedge_ratio
        spread = series1 - hedge_ratio * series2

        return spread

    def calculate_zscore(self, spread: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate rolling z-score of spread.

        Parameters:
        -----------
        spread : pd.Series
            Spread series
        window : int
            Rolling window for mean and std calculation

        Returns:
        --------
        pd.Series
            Z-score series
        """
        spread_mean = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()

        # Avoid division by zero
        spread_std = spread_std.replace(0, np.nan)

        zscore = (spread - spread_mean) / spread_std
        return zscore

    def generate_signals(
        self, series1: pd.Series, series2: pd.Series, window: int = 20
    ) -> pd.DataFrame:
        """
        Generate trading signals for pairs trading.

        Parameters:
        -----------
        series1 : pd.Series
            First price series
        series2 : pd.Series
            Second price series
        window : int
            Rolling window for statistics

        Returns:
        --------
        pd.DataFrame
            DataFrame with signals and spread information
        """
        # Calculate spread and z-score
        spread = self.calculate_spread(series1, series2)
        zscore = self.calculate_zscore(spread, window)

        # Initialize signals
        signals = pd.DataFrame(index=series1.index)
        signals["spread"] = spread
        signals["zscore"] = zscore
        signals["signal"] = 0  # 0: neutral, 1: long spread, -1: short spread
        signals["position"] = 0

        # Generate signals based on z-score
        position = 0
        for i in range(len(signals)):
            z = signals["zscore"].iloc[i]

            if np.isnan(z):
                signals["signal"].iloc[i] = 0
                signals["position"].iloc[i] = position
                continue

            # Entry signals
            if position == 0:
                if z > self.entry_threshold:
                    # Z-score too high, short the spread (short stock1, long stock2)
                    position = -1
                elif z < -self.entry_threshold:
                    # Z-score too low, long the spread (long stock1, short stock2)
                    position = 1

            # Exit signals
            elif position != 0:
                if abs(z) < self.exit_threshold:
                    # Spread reverted, close position
                    position = 0
                elif abs(z) > self.stop_loss:
                    # Stop loss hit
                    position = 0

            signals["signal"].iloc[i] = position
            signals["position"].iloc[i] = position

        # Calculate returns
        signals["stock1_ret"] = series1.pct_change()
        signals["stock2_ret"] = series2.pct_change()
        signals["strategy_ret"] = signals["position"].shift(1) * (
            signals["stock1_ret"] - self.hedge_ratio * signals["stock2_ret"]
        )

        return signals


class CointegrationAnalyzer:
    """
    Analyze cointegration relationships between time series.
    """

    @staticmethod
    def engle_granger_test(
        series1: pd.Series, series2: pd.Series, significance_level: float = 0.05
    ) -> Dict:
        """
        Perform Engle-Granger cointegration test.

        Parameters:
        -----------
        series1 : pd.Series
            First price series
        series2 : pd.Series
            Second price series
        significance_level : float
            Significance level for hypothesis test

        Returns:
        --------
        Dict
            Test results including test statistic, p-value, and critical values
        """
        from statsmodels.tsa.stattools import coint

        try:
            score, pvalue, crit_values = coint(series1, series2)

            is_cointegrated = pvalue < significance_level

            return {
                "test_statistic": score,
                "pvalue": pvalue,
                "critical_values": {
                    "1%": crit_values[0],
                    "5%": crit_values[1],
                    "10%": crit_values[2],
                },
                "is_cointegrated": is_cointegrated,
                "significance_level": significance_level,
            }
        except Exception as e:
            warnings.warn(f"Cointegration test failed: {e}")
            return {
                "test_statistic": np.nan,
                "pvalue": 1.0,
                "is_cointegrated": False,
                "error": str(e),
            }

    @staticmethod
    def johansen_test(
        data: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1
    ) -> Dict:
        """
        Perform Johansen cointegration test for multiple time series.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with multiple price series
        det_order : int
            Deterministic trend order (-1, 0, 1)
        k_ar_diff : int
            Number of lags for the VAR model

        Returns:
        --------
        Dict
            Test results
        """
        from statsmodels.tsa.vector_ar.vecm import coint_johansen

        try:
            result = coint_johansen(data, det_order, k_ar_diff)

            return {
                "trace_statistic": result.lr1,
                "max_eigen_statistic": result.lr2,
                "critical_values_trace": result.cvt,
                "critical_values_max_eigen": result.cvm,
                "eigenvectors": result.evec,
                "eigenvalues": result.eig,
            }
        except Exception as e:
            warnings.warn(f"Johansen test failed: {e}")
            return {"error": str(e)}

    @staticmethod
    def half_life(spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion for a spread.

        Half-life is the expected time for the spread to revert halfway
        back to its mean.

        Parameters:
        -----------
        spread : pd.Series
            Spread time series

        Returns:
        --------
        float
            Half-life in number of periods
        """
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()

        # Align indices
        common_index = spread_lag.index.intersection(spread_diff.index)
        spread_lag = spread_lag.loc[common_index]
        spread_diff = spread_diff.loc[common_index]

        # Fit AR(1) model: spread_diff = lambda * spread_lag + error
        from sklearn.linear_model import LinearRegression

        X = spread_lag.values.reshape(-1, 1)
        y = spread_diff.values

        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        lambda_param = model.coef_[0]

        if lambda_param >= 0:
            return np.inf  # No mean reversion

        half_life = -np.log(2) / lambda_param
        return half_life


class OrnsteinUhlenbeck:
    """
    Ornstein-Uhlenbeck process for mean reversion modeling.

    dX_t = ( - X_t)dt + dW_t
    """

    def __init__(self):
        self.theta = None  # Speed of mean reversion
        self.mu = None  # Long-term mean
        self.sigma = None  # Volatility

    def fit(self, prices: pd.Series) -> Dict[str, float]:
        """
        Estimate OU process parameters using Maximum Likelihood.

        Parameters:
        -----------
        prices : pd.Series
            Price series

        Returns:
        --------
        Dict[str, float]
            Estimated parameters (theta, mu, sigma)
        """
        n = len(prices)
        dt = 1  # Assuming unit time steps

        # Calculate differences
        S_x = np.sum(prices[:-1])
        S_y = np.sum(prices[1:])
        S_xx = np.sum(prices[:-1] ** 2)
        np.sum(prices[1:] ** 2)
        S_xy = np.sum(prices[:-1] * prices[1:])

        # MLE estimates
        try:
            numerator = S_y * S_xx - S_x * S_xy
            denominator = (n - 1) * (S_xx - S_xy) - (S_x**2 - S_x * S_y)
            if abs(denominator) > 1e-12:
                self.mu = numerator / denominator
            else:
                self.mu = np.mean(prices)

            # Additional calculations
            ou_den = S_xx - 2 * self.mu * S_x + (n - 1) * self.mu**2
            ou_num = S_xy - self.mu * S_x - self.mu * S_y + (n - 1) * self.mu**2

            if ou_den > 1e-12 and ou_num / ou_den > 0:
                self.theta = -np.log(ou_num / ou_den) / dt
            else:
                self.theta = 0.1  # Default value

            # Clamp theta to be non-negative
            self.theta = max(0.0001, self.theta)

            # Estimate sigma
            a = np.exp(-self.theta * dt)
            sum_squared_errors = np.sum(
                (prices[1:] - a * prices[:-1] - self.mu * (1 - a)) ** 2
            )
            sigma_sq = sum_squared_errors * 2 * self.theta / ((n - 1) * (1 - a**2))
            self.sigma = np.sqrt(max(0, sigma_sq))

        except Exception:
            self.theta = 0.1
            self.mu = np.mean(prices)
            self.sigma = np.std(prices)

        return {
            "theta": self.theta,
            "mu": self.mu,
            "sigma": self.sigma,
            "half_life": np.log(2) / self.theta if self.theta > 0 else np.inf,
        }

    def simulate(
        self, T: int, dt: float = 1.0, X0: Optional[float] = None
    ) -> np.ndarray:
        """
        Simulate OU process.

        Parameters:
        -----------
        T : int
            Number of time steps
        dt : float
            Time step size
        X0 : float, optional
            Initial value (defaults to mu)

        Returns:
        --------
        np.ndarray
            Simulated path
        """
        if self.theta is None or self.mu is None or self.sigma is None:
            raise ValueError("Parameters not fitted. Call fit() first.")

        if X0 is None:
            X0 = self.mu

        path = np.zeros(T)
        path[0] = X0

        for t in range(1, T):
            dW = np.random.normal(0, np.sqrt(dt))
            path[t] = (
                path[t - 1]
                + self.theta * (self.mu - path[t - 1]) * dt
                + self.sigma * dW
            )

        return path

    def expected_value(self, X0: float, t: float) -> float:
        """
        Calculate expected value at time t.

        E[X_t | X_0] =  + (X_0 - )e^{-t}
        """
        return self.mu + (X0 - self.mu) * np.exp(-self.theta * t)

    def variance(self, t: float) -> float:
        """
        Calculate variance at time t.

        Var[X_t] = /(2) * (1 - e^{-2t})
        """
        return (self.sigma**2 / (2 * self.theta)) * (1 - np.exp(-2 * self.theta * t))


class MeanReversionTester:
    """
    Statistical tests for mean reversion.
    """

    @staticmethod
    def adf_test(series: pd.Series, regression: str = "c") -> Dict:
        """
        Augmented Dickey-Fuller test for stationarity.

        Parameters:
        -----------
        series : pd.Series
            Time series to test
        regression : str
            Regression type ('c': constant, 'ct': constant and trend, 'n': no constant)

        Returns:
        --------
        Dict
            Test results
        """
        from statsmodels.tsa.stattools import adfuller

        result = adfuller(series.dropna(), regression=regression)

        return {
            "test_statistic": result[0],
            "pvalue": result[1],
            "n_lags": result[2],
            "n_obs": result[3],
            "critical_values": result[4],
            "is_stationary": result[1] < 0.05,
        }

    @staticmethod
    def variance_ratio_test(returns: pd.Series, lags: List[int] = [2, 5, 10]) -> Dict:
        """
        Variance ratio test for random walk hypothesis.

        VR(k) should be 1 under random walk.
        VR(k) < 1 suggests mean reversion.
        VR(k) > 1 suggests momentum.

        Parameters:
        -----------
        returns : pd.Series
            Return series
        lags : List[int]
            Lag periods to test

        Returns:
        --------
        Dict
            Variance ratios and test statistics
        """
        results = {}

        for k in lags:
            # k-period returns
            k_period_returns = returns.rolling(window=k).sum().dropna()

            # Variance ratio
            var_k = np.var(k_period_returns, ddof=1)
            var_1 = np.var(returns.dropna(), ddof=1)

            vr = var_k / (k * var_1) if var_1 != 0 else 1.0

            # Test statistic (under homoscedasticity)
            n = len(returns.dropna())
            test_stat = (vr - 1) * np.sqrt(n / k)
            pvalue = 2 * (1 - stats.norm.cdf(abs(test_stat)))

            results[f"VR({k})"] = {
                "variance_ratio": vr,
                "test_statistic": test_stat,
                "pvalue": pvalue,
                "interpretation": (
                    "mean_reversion"
                    if vr < 1
                    else ("momentum" if vr > 1 else "random_walk")
                ),
            }

        return results

    @staticmethod
    def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
        """
        Calculate Hurst exponent.

        H < 0.5: Mean reverting
        H = 0.5: Random walk (Geometric Brownian Motion)
        H > 0.5: Trending

        Parameters:
        -----------
        series : pd.Series
            Price series
        max_lag : int
            Maximum lag for calculation

        Returns:
        --------
        float
            Hurst exponent
        """
        lags = range(2, max_lag)
        tau = []

        for lag in lags:
            # Calculate standard deviation of differences
            pp = np.subtract(series[lag:].values, series[:-lag].values)
            tau.append(np.std(pp))

        # Fit power law: tau ~ lag^H
        lags_log = np.log(list(lags))
        tau_log = np.log(tau)

        # Linear regression in log-log space
        poly = np.polyfit(lags_log, tau_log, 1)
        hurst = poly[0]

        return hurst


class SpreadAnalyzer:
    """
    Analyze spread characteristics for statistical arbitrage.
    """

    @staticmethod
    def calculate_spread_statistics(spread: pd.Series, window: int = 252) -> Dict:
        """
        Calculate comprehensive spread statistics.
        """
        return {
            "mean": spread.mean(),
            "std": spread.std(),
            "min": spread.min(),
            "max": spread.max(),
            "median": spread.median(),
            "skewness": stats.skew(spread.dropna()),
            "kurtosis": stats.kurtosis(spread.dropna()),
            "rolling_mean": spread.rolling(window=window).mean().iloc[-1],
            "rolling_std": spread.rolling(window=window).std().iloc[-1],
            "current_zscore": (
                (spread.iloc[-1] - spread.mean()) / spread.std()
                if spread.std() != 0
                else 0
            ),
        }

    @staticmethod
    def bollinger_bands_on_spread(
        spread: pd.Series, window: int = 20, num_std: float = 2.0
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands on spread.
        """
        bb = pd.DataFrame(index=spread.index)
        bb["spread"] = spread
        bb["ma"] = spread.rolling(window=window).mean()
        bb["std"] = spread.rolling(window=window).std()
        bb["upper"] = bb["ma"] + num_std * bb["std"]
        bb["lower"] = bb["ma"] - num_std * bb["std"]
        bb["position"] = (spread - bb["ma"]) / bb["std"]

        return bb

    @staticmethod
    def optimal_entry_exit_thresholds(
        spread: pd.Series, transaction_cost: float = 0.001
    ) -> Dict[str, float]:
        """
        Calculate optimal entry/exit thresholds considering transaction costs.

        Uses Sharpe ratio maximization.
        """

        def calculate_sharpe(thresholds):
            entry_threshold, exit_threshold = thresholds

            # Simulate strategy
            zscore = (spread - spread.mean()) / spread.std()
            position = 0
            returns = []

            for i in range(1, len(zscore)):
                if position == 0:
                    if abs(zscore.iloc[i]) > entry_threshold:
                        position = -np.sign(zscore.iloc[i])
                elif abs(zscore.iloc[i]) < exit_threshold or np.sign(
                    zscore.iloc[i]
                ) != np.sign(zscore.iloc[i - 1]):
                    position = 0

                if position != 0:
                    ret = (
                        position * (spread.iloc[i] - spread.iloc[i - 1])
                        - transaction_cost
                    )
                    returns.append(ret)

            if len(returns) == 0:
                return -np.inf

            returns = np.array(returns)
            sharpe = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
            return -sharpe  # Negative for minimization

        # Optimize
        result = minimize(
            calculate_sharpe,
            x0=[2.0, 0.5],
            bounds=[(0.5, 4.0), (0.1, 1.5)],
            method="L-BFGS-B",
        )

        return {
            "entry_threshold": result.x[0],
            "exit_threshold": result.x[1],
            "expected_sharpe": -result.fun,
        }


__all__ = [
    "PairsTrading",
    "CointegrationAnalyzer",
    "OrnsteinUhlenbeck",
    "MeanReversionTester",
    "SpreadAnalyzer",
]
