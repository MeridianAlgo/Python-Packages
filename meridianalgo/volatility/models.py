"""
Volatility Models

GARCH(p,q), EGARCH, GJR-GARCH conditional volatility models, realized volatility
estimators (close-to-close, Parkinson, Garman-Klass, Rogers-Satchell), HAR-RV
forecasting model, and volatility term structure analytics.

Uses the `arch` library when available; falls back to pure NumPy GARCH(1,1).

References:
    Bollerslev (1986) - GARCH
    Nelson (1991) - EGARCH
    Glosten, Jagannathan, Runkle (1993) - GJR-GARCH
    Andersen & Bollerslev (1998) - Realized Volatility
    Corsi (2009) - HAR-RV
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

try:
    from arch import arch_model as _arch_model

    _ARCH_AVAILABLE = True
except ImportError:
    _ARCH_AVAILABLE = False


@dataclass
class GARCHResult:
    """Container for GARCH model estimation results."""

    model_type: str
    omega: float
    alpha: List[float]
    beta: List[float]
    gamma: Optional[float]
    log_likelihood: float
    aic: float
    bic: float
    conditional_volatility: pd.Series
    standardized_residuals: pd.Series
    persistence: float
    half_life: float


@dataclass
class VolatilityForecast:
    """Container for volatility forecasts."""

    point_forecast: pd.Series
    lower_bound: pd.Series
    upper_bound: pd.Series
    horizon: int
    model: str
    annualized: bool = True


class RealizedVolatility:
    """
    High-frequency realized volatility estimators.

    Implements multiple realized volatility measures from daily OHLCV data,
    ranging from simple close-to-close to the more efficient Garman-Klass
    and Rogers-Satchell estimators that exploit intraday price range.

    Example:
        >>> rv = RealizedVolatility(prices)
        >>> cc_vol = rv.close_to_close(window=21)
        >>> gk_vol = rv.garman_klass(window=21)
        >>> print(f"Close-Close (21d):  {cc_vol.iloc[-1]:.4f}")
        >>> print(f"Garman-Klass (21d): {gk_vol.iloc[-1]:.4f}")
    """

    def __init__(self, ohlcv: pd.DataFrame, trading_days: int = 252):
        """
        Parameters
        ----------
        ohlcv : pd.DataFrame
            DataFrame with columns: Open, High, Low, Close (and optionally Volume)
        trading_days : int
            Trading days per year for annualization
        """
        required = {"Open", "High", "Low", "Close"}
        missing = required - set(ohlcv.columns)
        if missing:
            raise ValueError(f"Missing OHLCV columns: {missing}")

        self.data = ohlcv.copy()
        self.trading_days = trading_days

    def close_to_close(self, window: int = 21, annualize: bool = True) -> pd.Series:
        """Standard close-to-close volatility."""
        log_returns = np.log(self.data["Close"] / self.data["Close"].shift(1))
        vol = log_returns.rolling(window).std()
        if annualize:
            vol = vol * np.sqrt(self.trading_days)
        return vol.rename("close_to_close_vol")

    def parkinson(self, window: int = 21, annualize: bool = True) -> pd.Series:
        """
        Parkinson (1980) range-based estimator.

        Uses High-Low range; 5x more efficient than close-to-close.
        Assumes zero drift; biased for trending assets.
        """
        factor = 1.0 / (4 * np.log(2))
        hl = np.log(self.data["High"] / self.data["Low"]) ** 2
        daily_var = factor * hl
        vol = np.sqrt(daily_var.rolling(window).mean())
        if annualize:
            vol = vol * np.sqrt(self.trading_days)
        return vol.rename("parkinson_vol")

    def garman_klass(self, window: int = 21, annualize: bool = True) -> pd.Series:
        """
        Garman-Klass (1980) estimator.

        Incorporates open-to-close and high-low range. More efficient than
        Parkinson. Assumes zero drift.
        """
        log_hl = np.log(self.data["High"] / self.data["Low"]) ** 2
        log_co = np.log(self.data["Close"] / self.data["Open"]) ** 2
        daily_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        vol = np.sqrt(daily_var.rolling(window).mean())
        if annualize:
            vol = vol * np.sqrt(self.trading_days)
        return vol.rename("garman_klass_vol")

    def rogers_satchell(self, window: int = 21, annualize: bool = True) -> pd.Series:
        """
        Rogers-Satchell (1991) estimator.

        Unbiased under non-zero drift. Uses all four price levels.
        """
        log_ho = np.log(self.data["High"] / self.data["Open"])
        log_hc = np.log(self.data["High"] / self.data["Close"])
        log_lo = np.log(self.data["Low"] / self.data["Open"])
        log_lc = np.log(self.data["Low"] / self.data["Close"])

        daily_var = log_ho * log_hc + log_lo * log_lc
        vol = np.sqrt(daily_var.rolling(window).mean())
        if annualize:
            vol = vol * np.sqrt(self.trading_days)
        return vol.rename("rogers_satchell_vol")

    def yang_zhang(self, window: int = 21, annualize: bool = True) -> pd.Series:
        """
        Yang-Zhang (2000) estimator.

        Handles overnight gaps. Minimum variance unbiased estimator combining
        overnight, open-to-close, and Rogers-Satchell estimators.
        """
        log_oc = np.log(self.data["Open"] / self.data["Close"].shift(1))
        log_co = np.log(self.data["Close"] / self.data["Open"])

        rs_var = (
            np.log(self.data["High"] / self.data["Open"])
            * np.log(self.data["High"] / self.data["Close"])
            + np.log(self.data["Low"] / self.data["Open"])
            * np.log(self.data["Low"] / self.data["Close"])
        )

        k = 0.34 / (1.34 + (window + 1) / (window - 1))

        overnight_var = log_oc.rolling(window).var()
        open_close_var = log_co.rolling(window).var()
        rs_var_mean = rs_var.rolling(window).mean()

        combined_var = overnight_var + k * open_close_var + (1 - k) * rs_var_mean
        vol = np.sqrt(combined_var)
        if annualize:
            vol = vol * np.sqrt(self.trading_days)
        return vol.rename("yang_zhang_vol")

    def realized_variance(
        self, returns: pd.Series, window: int = 21
    ) -> pd.Series:
        """
        Realized variance as sum of squared returns over a rolling window.

        Parameters
        ----------
        returns : pd.Series
            High-frequency or daily return series
        """
        rv = (returns**2).rolling(window).sum()
        return rv.rename("realized_variance")

    def all_estimators(self, window: int = 21) -> pd.DataFrame:
        """Compute all volatility estimators and return as DataFrame."""
        return pd.concat(
            [
                self.close_to_close(window),
                self.parkinson(window),
                self.garman_klass(window),
                self.rogers_satchell(window),
                self.yang_zhang(window),
            ],
            axis=1,
        )


class GARCHModel:
    """
    GARCH family conditional volatility models.

    Supports GARCH(p,q), EGARCH, and GJR-GARCH. Uses the `arch` library
    for maximum likelihood estimation when available; falls back to a
    pure-NumPy GARCH(1,1) implementation via conditional-sum-of-squares.

    Example:
        >>> model = GARCHModel(returns, model_type="garch", p=1, q=1)
        >>> result = model.fit()
        >>> print(f"Persistence: {result.persistence:.4f}")
        >>> print(f"Half-life:   {result.half_life:.1f} days")
        >>> forecasts = model.forecast(horizon=10)
    """

    def __init__(
        self,
        returns: pd.Series,
        model_type: str = "garch",
        p: int = 1,
        q: int = 1,
        distribution: str = "normal",
    ):
        """
        Parameters
        ----------
        returns : pd.Series
            Return series (decimal, not percentage)
        model_type : str
            'garch', 'egarch', or 'gjr'
        p : int
            ARCH order (lag of squared residuals)
        q : int
            GARCH order (lag of conditional variance)
        distribution : str
            Error distribution: 'normal', 't', or 'skewt'
        """
        valid_types = {"garch", "egarch", "gjr"}
        if model_type not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}")

        self.returns = returns.dropna() * 100
        self.model_type = model_type
        self.p = p
        self.q = q
        self.distribution = distribution
        self._fitted = None
        self._result: Optional[GARCHResult] = None

    def fit(self) -> GARCHResult:
        """
        Estimate model parameters via maximum likelihood.

        Returns
        -------
        GARCHResult
        """
        if _ARCH_AVAILABLE:
            return self._fit_arch()
        return self._fit_numpy_garch11()

    def _fit_arch(self) -> GARCHResult:
        vol_map = {"garch": "GARCH", "egarch": "EGARCH", "gjr": "GARCH"}
        dist_map = {"normal": "normal", "t": "t", "skewt": "skewt"}
        power = 2.0
        o = 1 if self.model_type == "gjr" else 0

        am = _arch_model(
            self.returns,
            vol=vol_map[self.model_type],
            p=self.p,
            o=o,
            q=self.q,
            dist=dist_map.get(self.distribution, "normal"),
            power=power,
        )
        res = am.fit(disp="off", show_warning=False)
        self._fitted = res

        params = res.params
        omega = float(params.get("omega", params.iloc[1]))
        alpha = [float(params.get(f"alpha[{i+1}]", 0)) for i in range(self.p)]
        beta = [float(params.get(f"beta[{i+1}]", 0)) for i in range(self.q)]
        gamma = float(params.get("gamma[1]", 0)) if self.model_type == "gjr" else None

        persistence = sum(alpha) + sum(beta) + (0.5 * gamma if gamma else 0)
        half_life = (
            np.log(0.5) / np.log(persistence)
            if 0 < persistence < 1
            else float("inf")
        )

        cond_vol = res.conditional_volatility / 100
        std_resid = res.std_resid

        n = len(self.returns)
        k = len(params)

        return GARCHResult(
            model_type=self.model_type.upper(),
            omega=omega,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            log_likelihood=float(res.loglikelihood),
            aic=float(res.aic),
            bic=float(res.bic),
            conditional_volatility=cond_vol,
            standardized_residuals=std_resid,
            persistence=persistence,
            half_life=half_life,
        )

    def _fit_numpy_garch11(self) -> GARCHResult:
        """Pure NumPy GARCH(1,1) via conditional-sum-of-squares optimization."""
        r = self.returns.values

        def neg_log_likelihood(params: np.ndarray) -> float:
            omega, alpha1, beta1 = params
            if omega <= 0 or alpha1 < 0 or beta1 < 0 or alpha1 + beta1 >= 1:
                return 1e10
            n = len(r)
            sigma2 = np.var(r) * np.ones(n)
            for t in range(1, n):
                sigma2[t] = omega + alpha1 * r[t - 1] ** 2 + beta1 * sigma2[t - 1]
            sigma2 = np.maximum(sigma2, 1e-8)
            ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + r**2 / sigma2)
            return -ll

        var0 = np.var(r)
        x0 = [var0 * 0.1, 0.05, 0.85]
        bounds = [(1e-8, None), (0, 0.5), (0, 0.999)]
        res = minimize(neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds)

        omega, alpha1, beta1 = res.x
        n = len(r)
        sigma2 = np.var(r) * np.ones(n)
        for t in range(1, n):
            sigma2[t] = omega + alpha1 * r[t - 1] ** 2 + beta1 * sigma2[t - 1]
        sigma2 = np.maximum(sigma2, 1e-8)

        cond_vol = pd.Series(
            np.sqrt(sigma2) / 100, index=self.returns.index, name="cond_vol"
        )
        std_resid = pd.Series(
            r / np.sqrt(sigma2), index=self.returns.index, name="std_resid"
        )

        persistence = alpha1 + beta1
        half_life = (
            np.log(0.5) / np.log(persistence) if 0 < persistence < 1 else float("inf")
        )
        k = 3
        ll = -res.fun
        aic = -2 * ll + 2 * k
        bic = -2 * ll + k * np.log(n)

        return GARCHResult(
            model_type="GARCH(1,1)-NumPy",
            omega=omega,
            alpha=[alpha1],
            beta=[beta1],
            gamma=None,
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            conditional_volatility=cond_vol,
            standardized_residuals=std_resid,
            persistence=persistence,
            half_life=half_life,
        )

    def forecast(
        self, horizon: int = 10, annualize: bool = True, trading_days: int = 252
    ) -> VolatilityForecast:
        """
        Multi-step ahead volatility forecast from fitted model.

        Parameters
        ----------
        horizon : int
            Number of periods ahead to forecast
        annualize : bool
            If True, convert to annualized volatility

        Returns
        -------
        VolatilityForecast
        """
        if self._result is None:
            self._result = self.fit()

        if _ARCH_AVAILABLE and self._fitted is not None:
            fc = self._fitted.forecast(horizon=horizon, reindex=False)
            var_forecast = fc.variance.values[-1, :]
            vol_forecast = np.sqrt(var_forecast) / 100
        else:
            omega = self._result.omega
            alpha1 = self._result.alpha[0]
            beta1 = self._result.beta[0]
            last_var = float(self._result.conditional_volatility.iloc[-1] * 100) ** 2

            persistence = alpha1 + beta1
            long_run_var = omega / (1 - persistence) if persistence < 1 else last_var

            vol_forecast = np.zeros(horizon)
            current_var = last_var
            for h in range(horizon):
                current_var = omega + persistence * current_var
                vol_forecast[h] = np.sqrt(current_var) / 100

        scale = np.sqrt(trading_days) if annualize else 1.0
        point = pd.Series(vol_forecast * scale, name="vol_forecast")

        std_err = 0.1 * point
        lower = point - 1.96 * std_err
        upper = point + 1.96 * std_err

        return VolatilityForecast(
            point_forecast=point,
            lower_bound=lower,
            upper_bound=upper,
            horizon=horizon,
            model=self._result.model_type,
            annualized=annualize,
        )


class VolatilityTermStructure:
    """
    Volatility term structure (VTS) analytics.

    Builds a volatility term structure from realized or implied volatility
    at multiple horizons, computes the slope, curvature, and VIX-style
    30-day implied volatility index.

    Example:
        >>> vts = VolatilityTermStructure(returns)
        >>> term_struct = vts.build(horizons=[5, 10, 21, 63, 126, 252])
        >>> print(term_struct)
    """

    def __init__(self, returns: pd.Series, trading_days: int = 252):
        self.returns = returns.dropna()
        self.trading_days = trading_days

    def build(self, horizons: Optional[List[int]] = None) -> pd.Series:
        """
        Build realized volatility term structure.

        Parameters
        ----------
        horizons : list of int
            Lookback windows in trading days

        Returns
        -------
        pd.Series
            Annualized volatility at each horizon
        """
        if horizons is None:
            horizons = [5, 10, 21, 63, 126, 252]

        vols = {}
        for h in horizons:
            if len(self.returns) >= h:
                recent = self.returns.iloc[-h:]
                vols[h] = recent.std() * np.sqrt(self.trading_days)
        return pd.Series(vols, name="annualized_vol")

    def slope(self, short_window: int = 21, long_window: int = 252) -> float:
        """
        Slope of the vol term structure (long - short).

        Positive slope = contango (normal), negative = backwardation (stress).
        """
        ts = self.build([short_window, long_window])
        return float(ts[long_window] - ts[short_window])

    def vix_style_index(self, window: int = 21) -> float:
        """Annualized 21-day realized volatility as a VIX-style measure."""
        recent = self.returns.iloc[-window:]
        return float(recent.std() * np.sqrt(self.trading_days) * 100)

    def vol_of_vol(self, vol_window: int = 21, vov_window: int = 63) -> float:
        """Volatility of the rolling volatility series (vol-of-vol)."""
        rolling_vol = self.returns.rolling(vol_window).std() * np.sqrt(
            self.trading_days
        )
        return float(rolling_vol.dropna().rolling(vov_window).std().iloc[-1])


class VolatilityRegimeDetector:
    """
    Detect low/medium/high volatility regimes using threshold-based clustering.

    Example:
        >>> detector = VolatilityRegimeDetector(returns)
        >>> regimes = detector.classify()
        >>> print(regimes.value_counts())
    """

    REGIME_LOW = "low_vol"
    REGIME_MEDIUM = "medium_vol"
    REGIME_HIGH = "high_vol"

    def __init__(
        self,
        returns: pd.Series,
        window: int = 21,
        trading_days: int = 252,
    ):
        self.returns = returns.dropna()
        self.window = window
        self.trading_days = trading_days

    def rolling_volatility(self) -> pd.Series:
        """Compute rolling annualized volatility."""
        return (
            self.returns.rolling(self.window).std()
            * np.sqrt(self.trading_days)
        )

    def classify(
        self,
        low_quantile: float = 0.33,
        high_quantile: float = 0.67,
    ) -> pd.Series:
        """
        Assign volatility regime labels using rolling vol quantiles.

        Parameters
        ----------
        low_quantile : float
            Vol below this percentile = low regime
        high_quantile : float
            Vol above this percentile = high regime

        Returns
        -------
        pd.Series
            Regime labels: 'low_vol', 'medium_vol', 'high_vol'
        """
        vol = self.rolling_volatility().dropna()
        low_thresh = vol.quantile(low_quantile)
        high_thresh = vol.quantile(high_quantile)

        regimes = pd.Series(self.REGIME_MEDIUM, index=vol.index, name="vol_regime")
        regimes[vol <= low_thresh] = self.REGIME_LOW
        regimes[vol >= high_thresh] = self.REGIME_HIGH

        return regimes

    def regime_statistics(self) -> pd.DataFrame:
        """
        Descriptive statistics for returns within each volatility regime.

        Returns
        -------
        pd.DataFrame
            Mean, std, Sharpe, skewness, kurtosis by regime
        """
        regimes = self.classify()
        common = regimes.index.intersection(self.returns.index)
        aligned_returns = self.returns.loc[common]
        aligned_regimes = regimes.loc[common]

        rows = []
        for regime in [self.REGIME_LOW, self.REGIME_MEDIUM, self.REGIME_HIGH]:
            mask = aligned_regimes == regime
            r = aligned_returns[mask]
            if len(r) == 0:
                continue
            ann_ret = r.mean() * self.trading_days
            ann_vol = r.std() * np.sqrt(self.trading_days)
            rows.append(
                {
                    "regime": regime,
                    "n_obs": len(r),
                    "annualized_return": ann_ret,
                    "annualized_vol": ann_vol,
                    "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
                    "skewness": float(r.skew()),
                    "excess_kurtosis": float(r.kurtosis()),
                }
            )

        return pd.DataFrame(rows).set_index("regime")


class VolatilityForecaster:
    """
    HAR-RV (Heterogeneous Autoregressive Realized Variance) model.

    Decomposes volatility into daily, weekly, and monthly components.
    Consistently outperforms GARCH in realized volatility forecasting.

    Reference:
        Corsi (2009) - A Simple Approximate Long Memory Model of Realized Volatility

    Example:
        >>> rv = returns ** 2
        >>> forecaster = VolatilityForecaster(rv)
        >>> forecaster.fit()
        >>> forecasts = forecaster.forecast(horizon=5)
    """

    def __init__(self, realized_variance: pd.Series, trading_days: int = 252):
        """
        Parameters
        ----------
        realized_variance : pd.Series
            Daily realized variance series
        """
        self.rv = realized_variance.dropna()
        self.trading_days = trading_days
        self._params: Optional[Dict] = None

    def _build_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Construct HAR-RV feature matrix: daily, weekly (5d), monthly (22d) RV."""
        rv = self.rv.values
        n = len(rv)
        rv_d = rv.copy()
        rv_w = pd.Series(rv).rolling(5).mean().values
        rv_m = pd.Series(rv).rolling(22).mean().values

        start = 22
        X = np.column_stack([
            np.ones(n - start),
            rv_d[start - 1: -1],
            rv_w[start - 1: -1],
            rv_m[start - 1: -1],
        ])
        y = rv[start:]
        return X, y

    def fit(self) -> Dict[str, float]:
        """
        Fit HAR-RV via OLS.

        Returns
        -------
        dict
            Coefficients: intercept, beta_d, beta_w, beta_m
        """
        X, y = self._build_features()
        coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)

        self._params = {
            "intercept": coeffs[0],
            "beta_d": coeffs[1],
            "beta_w": coeffs[2],
            "beta_m": coeffs[3],
        }

        y_hat = X @ coeffs
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        self._params["r_squared"] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return self._params

    def forecast(self, horizon: int = 5, annualize: bool = True) -> pd.Series:
        """
        Multi-step ahead HAR-RV forecasts.

        Parameters
        ----------
        horizon : int
            Number of trading days ahead

        Returns
        -------
        pd.Series
            Forecasted annualized volatility
        """
        if self._params is None:
            self.fit()

        rv = self.rv.values
        intercept = self._params["intercept"]
        b_d = self._params["beta_d"]
        b_w = self._params["beta_w"]
        b_m = self._params["beta_m"]

        forecasts = []
        rv_history = list(rv)

        for _ in range(horizon):
            rv_d = rv_history[-1]
            rv_w = np.mean(rv_history[-5:]) if len(rv_history) >= 5 else rv_d
            rv_m = np.mean(rv_history[-22:]) if len(rv_history) >= 22 else rv_w
            next_rv = intercept + b_d * rv_d + b_w * rv_w + b_m * rv_m
            next_rv = max(next_rv, 0)
            forecasts.append(next_rv)
            rv_history.append(next_rv)

        vol_forecasts = np.sqrt(np.array(forecasts))
        if annualize:
            vol_forecasts = vol_forecasts * np.sqrt(self.trading_days)

        return pd.Series(vol_forecasts, name="har_rv_forecast")
