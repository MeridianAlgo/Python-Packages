"""
Regime Detection Module

Advanced algorithms for detecting market regimes, structural breaks,
and state changes using Hidden Markov Models and other techniques.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class HiddenMarkovModel:
    """
    Hidden Markov Model for regime detection.

    Detects latent market states (e.g., bull, bear, high volatility, low volatility)
    """

    def __init__(self, n_states: int = 2):
        """
        Initialize HMM.

        Parameters:
        -----------
        n_states : int
            Number of hidden states
        """
        self.n_states = n_states
        self.transition_matrix = None
        self.means = None
        self.std_devs = None
        self.initial_probs = None
        self.states = None

    def fit(
        self, returns: pd.Series, max_iter: int = 100, tolerance: float = 1e-4
    ) -> Dict:
        """
        Fit HMM using Baum-Welch algorithm (EM).

        Parameters:
        -----------
        returns : pd.Series
            Return series
        max_iter : int
            Maximum iterations
        tolerance : float
            Convergence tolerance

        Returns:
        --------
        Dict
            Model parameters
        """
        len(returns)
        returns_array = returns.values

        # Initialize parameters
        self.means = np.linspace(
            returns_array.min(), returns_array.max(), self.n_states
        )
        self.std_devs = np.ones(self.n_states) * returns_array.std()
        self.transition_matrix = np.ones((self.n_states, self.n_states)) / self.n_states
        self.initial_probs = np.ones(self.n_states) / self.n_states

        log_likelihood_old = -np.inf

        for iteration in range(max_iter):
            # E-step: Forward-Backward algorithm
            alpha, beta, log_likelihood = self._forward_backward(returns_array)

            # Check convergence
            if abs(log_likelihood - log_likelihood_old) < tolerance:
                break
            log_likelihood_old = log_likelihood

            # M-step: Update parameters
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)

            xi = self._calculate_xi(returns_array, alpha, beta)

            # Update initial probabilities
            self.initial_probs = gamma[0]

            # Update transition matrix
            self.transition_matrix = (
                xi.sum(axis=0) / gamma[:-1].sum(axis=0, keepdims=True).T
            )

            # Update emission probabilities (mean and std for Gaussian)
            for state in range(self.n_states):
                weights = gamma[:, state]
                self.means[state] = np.sum(weights * returns_array) / np.sum(weights)
                self.std_devs[state] = np.sqrt(
                    np.sum(weights * (returns_array - self.means[state]) ** 2)
                    / np.sum(weights)
                )

        # Viterbi algorithm for most likely state sequence
        self.states = self._viterbi(returns_array)

        return {
            "means": dict(zip([f"State{i}" for i in range(self.n_states)], self.means)),
            "std_devs": dict(
                zip([f"State{i}" for i in range(self.n_states)], self.std_devs)
            ),
            "transition_matrix": pd.DataFrame(
                self.transition_matrix,
                index=[f"State{i}" for i in range(self.n_states)],
                columns=[f"State{i}" for i in range(self.n_states)],
            ),
            "log_likelihood": log_likelihood,
            "n_iterations": iteration + 1,
        }

    def _forward_backward(
        self, observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Forward-Backward algorithm."""
        n_obs = len(observations)
        alpha = np.zeros((n_obs, self.n_states))
        beta = np.zeros((n_obs, self.n_states))

        # Emission probabilities
        emission_probs = np.zeros((n_obs, self.n_states))
        for state in range(self.n_states):
            emission_probs[:, state] = stats.norm.pdf(
                observations, self.means[state], self.std_devs[state]
            )

        # Forward pass
        alpha[0] = self.initial_probs * emission_probs[0]
        alpha[0] /= alpha[0].sum()

        for t in range(1, n_obs):
            alpha[t] = emission_probs[t] * (alpha[t - 1] @ self.transition_matrix)
            alpha[t] /= alpha[t].sum()

        # Backward pass
        beta[-1] = 1
        for t in range(n_obs - 2, -1, -1):
            beta[t] = self.transition_matrix @ (emission_probs[t + 1] * beta[t + 1])
            beta[t] /= beta[t].sum()

        # Log likelihood
        log_likelihood = np.sum(np.log(alpha.sum(axis=1)))

        return alpha, beta, log_likelihood

    def _calculate_xi(
        self, observations: np.ndarray, alpha: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        """Calculate xi (probability of being in state i at time t and state j at time t+1)."""
        n_obs = len(observations)
        xi = np.zeros((n_obs - 1, self.n_states, self.n_states))

        for t in range(n_obs - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    emission_prob = stats.norm.pdf(
                        observations[t + 1], self.means[j], self.std_devs[j]
                    )
                    xi[t, i, j] = (
                        alpha[t, i]
                        * self.transition_matrix[i, j]
                        * emission_prob
                        * beta[t + 1, j]
                    )
            xi[t] /= xi[t].sum()

        return xi

    def _viterbi(self, observations: np.ndarray) -> np.ndarray:
        """Viterbi algorithm for most likely state sequence."""
        n_obs = len(observations)
        delta = np.zeros((n_obs, self.n_states))
        psi = np.zeros((n_obs, self.n_states), dtype=int)

        # Emission probabilities
        emission_probs = np.zeros((n_obs, self.n_states))
        for state in range(self.n_states):
            emission_probs[:, state] = stats.norm.pdf(
                observations, self.means[state], self.std_devs[state]
            )

        # Initialization
        delta[0] = self.initial_probs * emission_probs[0]

        # Recursion
        for t in range(1, n_obs):
            for j in range(self.n_states):
                probs = delta[t - 1] * self.transition_matrix[:, j]
                psi[t, j] = np.argmax(probs)
                delta[t, j] = np.max(probs) * emission_probs[t, j]

        # Backtracking
        states = np.zeros(n_obs, dtype=int)
        states[-1] = np.argmax(delta[-1])

        for t in range(n_obs - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def predict_state(self, returns: pd.Series) -> pd.Series:
        """
        Predict states for new data.
        """
        if self.states is None:
            raise ValueError("Model not fitted. Call fit() first.")

        states = self._viterbi(returns.values)
        return pd.Series(states, index=returns.index, name="state")


class RegimeSwitchingModel:
    """
    Markov Regime-Switching model for returns.
    """

    def __init__(self, n_regimes: int = 2):
        """
        Initialize regime-switching model.

        Parameters:
        -----------
        n_regimes : int
            Number of market regimes
        """
        self.n_regimes = n_regimes
        self.regime_params = None
        self.transition_probs = None
        self.current_regime = None

    def fit(self, returns: pd.Series) -> Dict:
        """
        Fit regime-switching model.

        Estimates parameters for each regime using EM algorithm.
        """
        # Use HMM as underlying implementation
        hmm = HiddenMarkovModel(n_states=self.n_regimes)
        results = hmm.fit(returns)

        # Store parameters
        self.regime_params = {
            "means": results["means"],
            "volatilities": results["std_devs"],
        }
        self.transition_probs = results["transition_matrix"]

        # Classify current regime
        self.current_regime = hmm.states[-1]

        # Calculate regime characteristics
        regime_stats = {}
        states = pd.Series(hmm.states, index=returns.index)

        for regime in range(self.n_regimes):
            regime_returns = returns[states == regime]
            regime_stats[f"Regime{regime}"] = {
                "count": len(regime_returns),
                "frequency": len(regime_returns) / len(returns),
                "mean_return": regime_returns.mean(),
                "volatility": regime_returns.std(),
                "sharpe": (
                    regime_returns.mean() / regime_returns.std() * np.sqrt(252)
                    if regime_returns.std() > 0
                    else 0
                ),
                "max_drawdown": (
                    regime_returns.cumsum() - regime_returns.cumsum().cummax()
                ).min(),
            }

        return {
            "regime_params": self.regime_params,
            "transition_matrix": self.transition_probs,
            "current_regime": self.current_regime,
            "regime_statistics": regime_stats,
        }

    def forecast_regime_probability(self, n_steps: int = 5) -> pd.DataFrame:
        """
        Forecast probability of being in each regime.

        Parameters:
        -----------
        n_steps : int
            Number of steps ahead to forecast

        Returns:
        --------
        pd.DataFrame
            Regime probabilities over time
        """
        if self.transition_probs is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Current state (one-hot encoded)
        current_state = np.zeros(self.n_regimes)
        current_state[self.current_regime] = 1

        # Forecast
        forecasts = []
        state = current_state

        for t in range(n_steps):
            state = state @ self.transition_probs.values
            forecasts.append(state.copy())

        forecast_df = pd.DataFrame(
            forecasts,
            columns=[f"Regime{i}" for i in range(self.n_regimes)],
            index=range(1, n_steps + 1),
        )

        return forecast_df


class StructuralBreakDetection:
    """
    Detect structural breaks in time series.
    """

    @staticmethod
    def chow_test(data: pd.Series, breakpoint: int) -> Dict:
        """
        Chow test for structural break at known breakpoint.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        breakpoint : int
            Index of suspected breakpoint

        Returns:
        --------
        Dict
            Test statistics and p-value
        """
        # Split data
        data1 = data.iloc[:breakpoint]
        data2 = data.iloc[breakpoint:]

        # Fit separate models (simple mean model)
        n1, n2 = len(data1), len(data2)
        k = 1  # Number of parameters (just mean)

        # RSS for individual regressions
        rss1 = np.sum((data1 - data1.mean()) ** 2)
        rss2 = np.sum((data2 - data2.mean()) ** 2)

        # RSS for pooled regression
        rss_pooled = np.sum((data - data.mean()) ** 2)

        # Chow F-statistic
        numerator = (rss_pooled - (rss1 + rss2)) / k
        denominator = (rss1 + rss2) / (n1 + n2 - 2 * k)

        f_stat = numerator / denominator if denominator > 0 else 0

        # P-value
        p_value = 1 - stats.f.cdf(f_stat, k, n1 + n2 - 2 * k)

        return {
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "breakpoint": breakpoint,
        }

    @staticmethod
    def cusum_test(data: pd.Series, significance_level: float = 0.05) -> Dict:
        """
        CUSUM test for detecting structural breaks.

        Parameters:
        -----------
        data : pd.Series
            Time series data
        significance_level : float
            Significance level for test

        Returns:
        --------
        Dict
            Test results and detected breaks
        """
        n = len(data)
        data_array = data.values

        # Calculate cumulative sum of residuals
        mean = data_array.mean()
        std = data_array.std()

        residuals = data_array - mean
        cusum = np.cumsum(residuals) / std if std > 0 else np.cumsum(residuals)

        # Calculate critical value
        critical_value = 0.948 * np.sqrt(n)  # 5% significance level

        # Detect breaks
        breaks = np.where(np.abs(cusum) > critical_value)[0]

        return {
            "cusum": pd.Series(cusum, index=data.index),
            "critical_value": critical_value,
            "breaks_detected": len(breaks) > 0,
            "break_points": breaks.tolist() if len(breaks) > 0 else [],
            "max_cusum": np.max(np.abs(cusum)),
            "significant": np.max(np.abs(cusum)) > critical_value,
        }

    @staticmethod
    def bai_perron_test(data: pd.Series, max_breaks: int = 5) -> Dict:
        """
        Bai-Perron test for multiple structural breaks.

        Simplified implementation for detecting multiple breaks.
        """
        n = len(data)
        data_array = data.values

        # Minimum segment size (15% of sample)
        min_segment = int(0.15 * n)

        best_breaks = []
        best_bic = np.inf

        # Try different numbers of breaks
        for n_breaks in range(1, min(max_breaks + 1, n // min_segment)):
            # Grid search for break points
            range(min_segment, n - min_segment)

            # For simplicity, use equal spacing
            break_spacing = (n - 2 * min_segment) // (n_breaks + 1)
            breaks = [min_segment + i * break_spacing for i in range(1, n_breaks + 1)]

            # Calculate BIC for this configuration
            segments = np.split(data_array, breaks)

            rss = sum(
                np.sum((seg - seg.mean()) ** 2) for seg in segments if len(seg) > 0
            )
            k = (n_breaks + 1) * 2  # parameters: mean and variance for each segment

            bic = n * np.log(rss / n) + k * np.log(n)

            if bic < best_bic:
                best_bic = bic
                best_breaks = breaks

        return {
            "n_breaks": len(best_breaks),
            "break_points": best_breaks,
            "bic": best_bic,
            "break_dates": (
                [data.index[bp] for bp in best_breaks] if len(best_breaks) > 0 else []
            ),
        }


class MarketStateClassifier:
    """
    Classify current market state using multiple indicators.
    """

    @staticmethod
    def classify_volatility_regime(returns: pd.Series, window: int = 60) -> pd.Series:
        """
        Classify volatility regime (low, normal, high).
        """
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)

        # Calculate percentiles
        vol_percentiles = rolling_vol.rank(pct=True)

        # Classify
        states = pd.Series("normal", index=returns.index)
        states[vol_percentiles < 0.33] = "low_volatility"
        states[vol_percentiles > 0.67] = "high_volatility"

        return states

    @staticmethod
    def classify_trend_regime(
        prices: pd.Series, short_window: int = 50, long_window: int = 200
    ) -> pd.Series:
        """
        Classify trend regime using moving averages.
        """
        sma_short = prices.rolling(window=short_window).mean()
        sma_long = prices.rolling(window=long_window).mean()

        states = pd.Series("neutral", index=prices.index)
        states[sma_short > sma_long * 1.02] = "strong_uptrend"
        states[(sma_short > sma_long) & (sma_short <= sma_long * 1.02)] = "uptrend"
        states[sma_short < sma_long * 0.98] = "strong_downtrend"
        states[(sma_short < sma_long) & (sma_short >= sma_long * 0.98)] = "downtrend"

        return states

    @staticmethod
    def composite_market_state(returns: pd.Series, prices: pd.Series) -> pd.DataFrame:
        """
        Create composite market state from multiple regime indicators.
        """
        states = pd.DataFrame(index=returns.index)

        # Volatility regime
        states["volatility_regime"] = MarketStateClassifier.classify_volatility_regime(
            returns
        )

        # Trend regime
        states["trend_regime"] = MarketStateClassifier.classify_trend_regime(prices)

        # Return distribution
        rolling_skew = returns.rolling(window=60).skew()
        states["return_skew"] = pd.cut(
            rolling_skew,
            bins=[-np.inf, -0.5, 0.5, np.inf],
            labels=["negative_skew", "neutral_skew", "positive_skew"],
        )

        # Market stress indicator
        rolling_max_dd = prices / prices.rolling(window=252).max() - 1
        states["stress_level"] = pd.cut(
            rolling_max_dd,
            bins=[-np.inf, -0.20, -0.10, 0],
            labels=["high_stress", "moderate_stress", "low_stress"],
        )

        return states


class VolatilityRegimeDetector:
    """
    Specialized detector for volatility regimes.
    """

    @staticmethod
    def garch_volatility_regimes(returns: pd.Series, n_regimes: int = 2) -> Dict:
        """
        Detect volatility regimes using GARCH models.

        Simplified implementation using realized volatility.
        """
        # Calculate realized volatility
        realized_vol = returns.rolling(window=20).std() * np.sqrt(252)

        # Use k-means clustering on volatility
        from sklearn.cluster import KMeans

        vol_array = realized_vol.dropna().values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        clusters = kmeans.fit_predict(vol_array)

        # Sort clusters by volatility level
        cluster_means = [vol_array[clusters == i].mean() for i in range(n_regimes)]
        cluster_mapping = np.argsort(cluster_means)

        # Map clusters to regimes (0 = low vol, 1 = high vol, etc.)
        regimes = np.zeros_like(clusters)
        for i, cluster_id in enumerate(cluster_mapping):
            regimes[clusters == cluster_id] = i

        # Create result series
        regime_series = pd.Series(np.nan, index=returns.index)
        regime_series.iloc[-len(regimes) :] = regimes

        # Calculate regime statistics
        regime_stats = {}
        for regime in range(n_regimes):
            regime_vol = vol_array[regimes == regime]
            regime_stats[f"Regime{regime}"] = {
                "mean_volatility": regime_vol.mean(),
                "min_volatility": regime_vol.min(),
                "max_volatility": regime_vol.max(),
                "frequency": len(regime_vol) / len(vol_array),
            }

        return {
            "regimes": regime_series,
            "realized_volatility": realized_vol,
            "regime_statistics": regime_stats,
            "current_regime": regimes[-1],
        }


__all__ = [
    "HiddenMarkovModel",
    "RegimeSwitchingModel",
    "StructuralBreakDetection",
    "MarketStateClassifier",
    "VolatilityRegimeDetector",
]
