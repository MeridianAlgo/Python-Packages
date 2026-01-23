"""
Risk Analytics Module

Advanced risk analysis including VaR, CVaR, stress testing, factor risk decomposition,
and tail risk analysis.
"""

from dataclasses import dataclass
from typing import Any, Dict, Union

import numpy as np
import pandas as pd


@dataclass
class RiskMetricsResult:
    """Container for risk analysis results."""

    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    volatility: float
    downside_volatility: float
    max_drawdown: float
    skewness: float
    kurtosis: float
    tail_ratio: float


class RiskAnalyzer:
    """
    Comprehensive risk analysis for portfolios and trading strategies.

    Provides institutional-grade risk metrics, VaR/CVaR calculations,
    stress testing, and factor risk decomposition.

    Example:
        >>> analyzer = RiskAnalyzer(returns)
        >>> var = analyzer.value_at_risk(confidence=0.95)
        >>> stress = analyzer.stress_test(scenarios)
    """

    def __init__(
        self, returns: Union[pd.Series, pd.DataFrame], periods_per_year: int = 252
    ):
        """
        Initialize RiskAnalyzer.

        Args:
            returns: Return series or DataFrame
            periods_per_year: Number of trading periods per year
        """
        if isinstance(returns, pd.DataFrame):
            self.returns = returns
            self.is_portfolio = True
        else:
            self.returns = pd.Series(returns).dropna()
            self.is_portfolio = False

        self.periods_per_year = periods_per_year

    # =========================================================================
    # VALUE AT RISK
    # =========================================================================

    def value_at_risk(
        self,
        confidence: float = 0.95,
        method: str = "historical",
        horizon: int = 1,
        n_simulations: int = 10000,
    ) -> float:
        """
        Calculate Value at Risk.

        Args:
            confidence: Confidence level (e.g., 0.95 for 95%)
            method: 'historical', 'parametric', 'cornish_fisher', or 'monte_carlo'
            horizon: Holding period in days
            n_simulations: Number of simulations for Monte Carlo

        Returns:
            VaR as a negative number (loss)
        """
        returns = self.returns if not self.is_portfolio else self.returns.mean(axis=1)

        if method == "historical":
            var = returns.quantile(1 - confidence)

        elif method == "parametric":
            try:
                from scipy.stats import norm
            except ImportError:
                raise ImportError("scipy required for parametric VaR")

            z = norm.ppf(1 - confidence)
            var = returns.mean() + z * returns.std()

        elif method == "cornish_fisher":
            try:
                from scipy.stats import norm
            except ImportError:
                raise ImportError("scipy required for Cornish-Fisher VaR")

            z = norm.ppf(1 - confidence)
            s = returns.skew()
            k = returns.kurtosis()

            # Cornish-Fisher expansion
            z_cf = (
                z
                + (z**2 - 1) * s / 6
                + (z**3 - 3 * z) * (k - 3) / 24
                - (2 * z**3 - 5 * z) * s**2 / 36
            )

            var = returns.mean() + z_cf * returns.std()

        elif method == "monte_carlo":
            # Simulate returns
            simulated = np.random.normal(returns.mean(), returns.std(), n_simulations)
            var = np.percentile(simulated, (1 - confidence) * 100)

        else:
            raise ValueError(f"Unknown VaR method: {method}")

        # Scale for holding period
        if horizon > 1:
            var = var * np.sqrt(horizon)

        return var

    def conditional_var(
        self, confidence: float = 0.95, method: str = "historical"
    ) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall / CVaR).

        Args:
            confidence: Confidence level
            method: 'historical' or 'gaussian'

        Returns:
            CVaR as a negative number (expected loss beyond VaR)
        """
        returns = self.returns if not self.is_portfolio else self.returns.mean(axis=1)

        if method == "historical":
            var = self.value_at_risk(confidence, "historical")
            return returns[returns <= var].mean()

        elif method == "gaussian":
            try:
                from scipy.stats import norm
            except ImportError:
                raise ImportError("scipy required for gaussian CVaR")

            mu = returns.mean()
            sigma = returns.std()
            alpha = 1 - confidence

            return mu - sigma * norm.pdf(norm.ppf(alpha)) / alpha

        else:
            raise ValueError(f"Unknown CVaR method: {method}")

    def var_breakdown(self, confidence: float = 0.95) -> Dict[str, float]:
        """
        Calculate component VaR for portfolio.

        Returns:
            Dictionary with VaR contribution by asset
        """
        if not self.is_portfolio:
            raise ValueError("VaR breakdown requires portfolio returns")

        # Calculate marginal VaR for each asset
        result = {}
        weights = np.ones(self.returns.shape[1]) / self.returns.shape[1]

        for col in self.returns.columns:
            result[col] = self.returns[col].quantile(1 - confidence) * weights[0]

        return result

    # =========================================================================
    # VOLATILITY ANALYSIS
    # =========================================================================

    def volatility(self, annualized: bool = True) -> float:
        """Calculate volatility."""
        returns = self.returns if not self.is_portfolio else self.returns.mean(axis=1)
        vol = returns.std()

        if annualized:
            vol *= np.sqrt(self.periods_per_year)

        return vol

    def downside_volatility(
        self, threshold: float = 0.0, annualized: bool = True
    ) -> float:
        """Calculate downside volatility."""
        returns = self.returns if not self.is_portfolio else self.returns.mean(axis=1)
        downside = returns[returns < threshold]

        if len(downside) == 0:
            return 0.0

        vol = downside.std()

        if annualized:
            vol *= np.sqrt(self.periods_per_year)

        return vol

    def upside_volatility(
        self, threshold: float = 0.0, annualized: bool = True
    ) -> float:
        """Calculate upside volatility."""
        returns = self.returns if not self.is_portfolio else self.returns.mean(axis=1)
        upside = returns[returns > threshold]

        if len(upside) == 0:
            return 0.0

        vol = upside.std()

        if annualized:
            vol *= np.sqrt(self.periods_per_year)

        return vol

    def volatility_ratio(self) -> float:
        """Calculate volatility ratio (upside vol / downside vol)."""
        upside = self.upside_volatility()
        downside = self.downside_volatility()

        return upside / downside if downside > 0 else np.inf

    def rolling_volatility(
        self, window: int = 21, annualized: bool = True
    ) -> pd.Series:
        """Calculate rolling volatility."""
        returns = self.returns if not self.is_portfolio else self.returns.mean(axis=1)
        vol = returns.rolling(window).std()

        if annualized:
            vol *= np.sqrt(self.periods_per_year)

        return vol

    def garch_volatility(self, p: int = 1, q: int = 1) -> pd.Series:
        """
        Calculate GARCH(p,q) volatility forecast.

        Args:
            p: Order of the GARCH term
            q: Order of the ARCH term

        Returns:
            Series of conditional volatility
        """
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError("arch package required for GARCH volatility")

        returns = self.returns if not self.is_portfolio else self.returns.mean(axis=1)
        returns = returns * 100  # Scale for numerical stability

        model = arch_model(returns, vol="Garch", p=p, q=q)
        fitted = model.fit(disp="off")

        return fitted.conditional_volatility / 100

    # =========================================================================
    # DRAWDOWN ANALYSIS
    # =========================================================================

    def max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        returns = self.returns if not self.is_portfolio else self.returns.mean(axis=1)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def drawdown_series(self) -> pd.Series:
        """Calculate drawdown series."""
        returns = self.returns if not self.is_portfolio else self.returns.mean(axis=1)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        return (cumulative - running_max) / running_max

    def average_drawdown(self) -> float:
        """Calculate average drawdown."""
        drawdown = self.drawdown_series()
        return drawdown[drawdown < 0].mean()

    def drawdown_duration(self) -> pd.DataFrame:
        """Calculate drawdown durations."""
        drawdown = self.drawdown_series()

        # Find drawdown periods
        in_drawdown = drawdown < 0

        # Calculate duration for each drawdown
        duration = []
        current_start = None

        for date, is_dd in in_drawdown.items():
            if is_dd and current_start is None:
                current_start = date
            elif not is_dd and current_start is not None:
                duration.append(
                    {
                        "start": current_start,
                        "end": date,
                        "duration": (
                            (date - current_start).days
                            if hasattr(date - current_start, "days")
                            else 1
                        ),
                        "max_drawdown": drawdown[current_start:date].min(),
                    }
                )
                current_start = None

        return pd.DataFrame(duration)

    def ulcer_index(self) -> float:
        """Calculate Ulcer Index (quadratic mean of drawdowns)."""
        drawdown = self.drawdown_series()
        return np.sqrt(np.mean(drawdown**2))

    def pain_index(self) -> float:
        """Calculate Pain Index (mean absolute drawdown)."""
        drawdown = self.drawdown_series()
        return abs(drawdown.mean())

    # =========================================================================
    # TAIL RISK
    # =========================================================================

    def skewness(self) -> float:
        """Calculate skewness."""
        returns = self.returns if not self.is_portfolio else self.returns.mean(axis=1)
        return returns.skew()

    def kurtosis(self) -> float:
        """Calculate excess kurtosis."""
        returns = self.returns if not self.is_portfolio else self.returns.mean(axis=1)
        return returns.kurtosis()

    def tail_ratio(self, percentile: float = 0.05) -> float:
        """
        Calculate tail ratio.

        Args:
            percentile: Percentile for tail calculation

        Returns:
            Ratio of right tail to left tail
        """
        returns = self.returns if not self.is_portfolio else self.returns.mean(axis=1)
        right = returns.quantile(1 - percentile)
        left = abs(returns.quantile(percentile))
        return right / left if left > 0 else np.inf

    def jarque_bera_test(self) -> Dict[str, float]:
        """
        Perform Jarque-Bera test for normality.

        Returns:
            Dictionary with test statistic and p-value
        """
        try:
            from scipy.stats import jarque_bera
        except ImportError:
            raise ImportError("scipy required for Jarque-Bera test")

        returns = self.returns if not self.is_portfolio else self.returns.mean(axis=1)
        stat, pvalue = jarque_bera(returns)

        return {"statistic": stat, "pvalue": pvalue, "is_normal": pvalue > 0.05}

    def extreme_returns(self, percentile: float = 0.01) -> Dict[str, pd.DataFrame]:
        """
        Get extreme returns (tails).

        Args:
            percentile: Percentile threshold

        Returns:
            Dictionary with 'worst' and 'best' returns
        """
        returns = self.returns if not self.is_portfolio else self.returns.mean(axis=1)

        lower_threshold = returns.quantile(percentile)
        upper_threshold = returns.quantile(1 - percentile)

        return {
            "worst": returns[returns <= lower_threshold].sort_values(),
            "best": returns[returns >= upper_threshold].sort_values(ascending=False),
        }

    # =========================================================================
    # STRESS TESTING
    # =========================================================================

    def stress_test(self, scenarios: Dict[str, float]) -> Dict[str, float]:
        """
        Perform stress test with given scenarios.

        Args:
            scenarios: Dictionary of scenario name to market shock (%)

        Returns:
            Portfolio impact for each scenario
        """
        if not self.is_portfolio:
            # For single asset, just apply the shock directly
            return {name: shock for name, shock in scenarios.items()}

        # Calculate correlation with market (first column or provided)
        market = self.returns.iloc[:, 0]
        betas = {}

        for col in self.returns.columns:
            cov = np.cov(self.returns[col], market)
            betas[col] = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1

        results = {}
        for scenario_name, market_shock in scenarios.items():
            portfolio_impact = sum(
                betas.get(col, 1) * market_shock for col in self.returns.columns
            ) / len(self.returns.columns)
            results[scenario_name] = portfolio_impact

        return results

    def historical_stress_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """
        Get returns during historical stress periods.

        Returns:
            Dictionary of scenario results
        """
        returns = self.returns if not self.is_portfolio else self.returns.mean(axis=1)

        # Define historical stress periods
        stress_periods = {
            "Black Monday (1987)": ("1987-10-19", "1987-10-20"),
            "Dot-com Crash (2000-2002)": ("2000-03-01", "2002-10-09"),
            "Financial Crisis (2008)": ("2008-09-15", "2008-11-20"),
            "COVID Crash (2020)": ("2020-02-19", "2020-03-23"),
            "Flash Crash (2010)": ("2010-05-06", "2010-05-07"),
        }

        results = {}
        if not isinstance(returns.index, pd.DatetimeIndex):
            return results

        for name, (start, end) in stress_periods.items():
            try:
                period_returns = returns[start:end]
                if len(period_returns) > 0:
                    results[name] = {
                        "return": (1 + period_returns).prod() - 1,
                        "volatility": period_returns.std() * np.sqrt(252),
                        "max_loss": period_returns.min(),
                        "days": len(period_returns),
                    }
            except Exception:
                continue

        return results

    # =========================================================================
    # COMPREHENSIVE ANALYSIS
    # =========================================================================

    def summary(self) -> Dict[str, Any]:
        """Generate comprehensive risk summary."""
        return {
            # VaR/CVaR
            "var_95_historical": self.value_at_risk(0.95, "historical"),
            "var_95_parametric": self.value_at_risk(0.95, "parametric"),
            "cvar_95": self.conditional_var(0.95),
            "var_99_historical": self.value_at_risk(0.99, "historical"),
            "cvar_99": self.conditional_var(0.99),
            # Volatility
            "volatility": self.volatility(),
            "downside_volatility": self.downside_volatility(),
            "upside_volatility": self.upside_volatility(),
            "volatility_ratio": self.volatility_ratio(),
            # Drawdown
            "max_drawdown": self.max_drawdown(),
            "average_drawdown": self.average_drawdown(),
            "ulcer_index": self.ulcer_index(),
            "pain_index": self.pain_index(),
            # Distribution
            "skewness": self.skewness(),
            "kurtosis": self.kurtosis(),
            "tail_ratio": self.tail_ratio(),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert summary to DataFrame."""
        summary = self.summary()
        return pd.DataFrame({"Value": summary.values()}, index=summary.keys())


def calculate_risk_metrics(
    returns: Union[pd.Series, pd.DataFrame], confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Convenience function to calculate all risk metrics.

    Args:
        returns: Return series or DataFrame
        confidence: Confidence level for VaR/CVaR

    Returns:
        Dictionary of all risk metrics
    """
    analyzer = RiskAnalyzer(returns)
    summary = analyzer.summary()
    summary["confidence"] = confidence
    return summary
