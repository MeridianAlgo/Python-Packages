"""
Comprehensive risk management system with VaR, CVaR, and stress testing.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    from arch import arch_model  # noqa: F401

    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    warnings.warn("ARCH package not available. GARCH models will be limited.")

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Container for risk metrics."""

    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    volatility: float
    skewness: float
    kurtosis: float
    beta: Optional[float] = None
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "VaR_95": self.var_95,
            "VaR_99": self.var_99,
            "CVaR_95": self.cvar_95,
            "CVaR_99": self.cvar_99,
            "Max_Drawdown": self.max_drawdown,
            "Volatility": self.volatility,
            "Skewness": self.skewness,
            "Kurtosis": self.kurtosis,
            "Beta": self.beta,
            "Tracking_Error": self.tracking_error,
            "Information_Ratio": self.information_ratio,
        }


@dataclass
class StressTestResult:
    """Result of stress testing."""

    scenario_name: str
    portfolio_return: float
    portfolio_value_change: float
    asset_returns: pd.Series
    risk_metrics: RiskMetrics
    metadata: Dict[str, Any] = None


class BaseRiskModel(ABC):
    """Abstract base class for risk models."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate_var(
        self, returns: pd.Series, confidence_level: float = 0.95
    ) -> float:
        """Calculate Value at Risk."""
        pass

    @abstractmethod
    def calculate_cvar(
        self, returns: pd.Series, confidence_level: float = 0.95
    ) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        pass


class HistoricalVaR(BaseRiskModel):
    """Historical simulation VaR model."""

    def __init__(self, window: int = 252):
        super().__init__("HistoricalVaR")
        self.window = window

    def calculate_var(
        self, returns: pd.Series, confidence_level: float = 0.95
    ) -> float:
        """Calculate historical VaR."""
        if len(returns) < 2:
            return 0.0

        # Use rolling window if specified
        if self.window and len(returns) > self.window:
            returns = returns.tail(self.window)

        return -np.percentile(returns.dropna(), (1 - confidence_level) * 100)

    def calculate_cvar(
        self, returns: pd.Series, confidence_level: float = 0.95
    ) -> float:
        """Calculate historical CVaR."""
        if len(returns) < 2:
            return 0.0

        # Use rolling window if specified
        if self.window and len(returns) > self.window:
            returns = returns.tail(self.window)

        var = self.calculate_var(returns, confidence_level)
        # CVaR is the mean of returns below VaR threshold
        tail_returns = returns[returns <= -var]

        return -tail_returns.mean() if len(tail_returns) > 0 else var


class ParametricVaR(BaseRiskModel):
    """Parametric (Normal) VaR model."""

    def __init__(self, distribution: str = "normal"):
        super().__init__("ParametricVaR")
        self.distribution = distribution

    def calculate_var(
        self, returns: pd.Series, confidence_level: float = 0.95
    ) -> float:
        """Calculate parametric VaR."""
        if len(returns) < 2:
            return 0.0

        returns_clean = returns.dropna()

        if self.distribution == "normal":
            mu = returns_clean.mean()
            sigma = returns_clean.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            return -(mu + z_score * sigma)

        elif self.distribution == "t":
            # Fit t-distribution
            params = stats.t.fit(returns_clean)
            df, loc, scale = params
            t_score = stats.t.ppf(1 - confidence_level, df, loc, scale)
            return -t_score

        elif self.distribution == "skewed_t":
            # Fit skewed t-distribution
            try:
                params = stats.skewt.fit(returns_clean)
                var_value = stats.skewt.ppf(1 - confidence_level, *params)
                return -var_value
            except Exception:
                # Fallback to normal distribution
                return self._normal_var(returns_clean, confidence_level)

        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def calculate_cvar(
        self, returns: pd.Series, confidence_level: float = 0.95
    ) -> float:
        """Calculate parametric CVaR."""
        if len(returns) < 2:
            return 0.0

        returns_clean = returns.dropna()

        if self.distribution == "normal":
            mu = returns_clean.mean()
            sigma = returns_clean.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            # CVaR for normal distribution
            phi_z = stats.norm.pdf(z_score)
            cvar = mu - sigma * phi_z / (1 - confidence_level)
            return -cvar

        else:
            # For non-normal distributions, use numerical integration
            var = self.calculate_var(returns_clean, confidence_level)

            def integrand(x):
                if self.distribution == "t":
                    params = stats.t.fit(returns_clean)
                    return x * stats.t.pdf(x, *params)
                elif self.distribution == "skewed_t":
                    params = stats.skewt.fit(returns_clean)
                    return x * stats.skewt.pdf(x, *params)
                else:
                    return x * stats.norm.pdf(
                        x, returns_clean.mean(), returns_clean.std()
                    )

            # Numerical integration from -inf to -VaR
            from scipy.integrate import quad

            integral, _ = quad(integrand, -np.inf, -var)
            cvar = integral / (1 - confidence_level)
            return -cvar

    def _normal_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate normal VaR as fallback."""
        mu = returns.mean()
        sigma = returns.std()
        z_score = stats.norm.ppf(1 - confidence_level)
        return -(mu + z_score * sigma)


class MonteCarloVaR(BaseRiskModel):
    """Monte Carlo simulation VaR model."""

    def __init__(self, n_simulations: int = 10000, time_horizon: int = 1):
        super().__init__("MonteCarloVaR")
        self.n_simulations = n_simulations
        self.time_horizon = time_horizon

    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = "bootstrap",
    ) -> float:
        """Calculate Monte Carlo VaR."""
        if len(returns) < 2:
            return 0.0

        returns_clean = returns.dropna()

        if method == "bootstrap":
            # Bootstrap resampling
            simulated_returns = np.random.choice(
                returns_clean.values,
                size=(self.n_simulations, self.time_horizon),
                replace=True,
            )

            # Calculate portfolio returns for each simulation
            if self.time_horizon == 1:
                portfolio_returns = simulated_returns.flatten()
            else:
                # Compound returns over time horizon
                portfolio_returns = np.prod(1 + simulated_returns, axis=1) - 1

        elif method == "parametric":
            # Parametric simulation assuming normal distribution
            mu = returns_clean.mean()
            sigma = returns_clean.std()

            if self.time_horizon == 1:
                portfolio_returns = np.random.normal(mu, sigma, self.n_simulations)
            else:
                # Geometric Brownian Motion
                dt = 1.0  # Daily time step
                portfolio_returns = []

                for _ in range(self.n_simulations):
                    path = [0]
                    for t in range(self.time_horizon):
                        dW = np.random.normal(0, np.sqrt(dt))
                        path.append(path[-1] + mu * dt + sigma * dW)

                    total_return = np.exp(path[-1]) - 1
                    portfolio_returns.append(total_return)

                portfolio_returns = np.array(portfolio_returns)

        else:
            raise ValueError(f"Unknown method: {method}")

        return -np.percentile(portfolio_returns, (1 - confidence_level) * 100)

    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = "bootstrap",
    ) -> float:
        """Calculate Monte Carlo CVaR."""
        if len(returns) < 2:
            return 0.0

        # First calculate VaR
        var = self.calculate_var(returns, confidence_level, method)

        # Then simulate again to get tail expectation
        returns_clean = returns.dropna()

        if method == "bootstrap":
            simulated_returns = np.random.choice(
                returns_clean.values,
                size=(self.n_simulations, self.time_horizon),
                replace=True,
            )

            if self.time_horizon == 1:
                portfolio_returns = simulated_returns.flatten()
            else:
                portfolio_returns = np.prod(1 + simulated_returns, axis=1) - 1

        elif method == "parametric":
            mu = returns_clean.mean()
            sigma = returns_clean.std()

            if self.time_horizon == 1:
                portfolio_returns = np.random.normal(mu, sigma, self.n_simulations)
            else:
                portfolio_returns = []
                dt = 1.0

                for _ in range(self.n_simulations):
                    path = [0]
                    for t in range(self.time_horizon):
                        dW = np.random.normal(0, np.sqrt(dt))
                        path.append(path[-1] + mu * dt + sigma * dW)

                    total_return = np.exp(path[-1]) - 1
                    portfolio_returns.append(total_return)

                portfolio_returns = np.array(portfolio_returns)

        # CVaR is the mean of returns below VaR threshold
        tail_returns = portfolio_returns[portfolio_returns <= -var]

        return -np.mean(tail_returns) if len(tail_returns) > 0 else var


class RiskManager:
    """Comprehensive risk management system."""

    def __init__(self):
        self.risk_models = {
            "historical": HistoricalVaR(),
            "parametric": ParametricVaR(),
            "monte_carlo": MonteCarloVaR(),
        }
        self.stress_scenarios = {}

    def calculate_portfolio_risk(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series = None,
        confidence_levels: List[float] = [0.95, 0.99],
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics for a portfolio."""
        if len(returns) < 2:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0)

        returns_clean = returns.dropna()

        # Basic statistics
        volatility = returns_clean.std() * np.sqrt(252)  # Annualized
        skewness = stats.skew(returns_clean)
        kurtosis = stats.kurtosis(returns_clean)

        # VaR and CVaR calculations
        var_95 = self.risk_models["historical"].calculate_var(returns_clean, 0.95)
        var_99 = self.risk_models["historical"].calculate_var(returns_clean, 0.99)
        cvar_95 = self.risk_models["historical"].calculate_cvar(returns_clean, 0.95)
        cvar_99 = self.risk_models["historical"].calculate_cvar(returns_clean, 0.99)

        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown(returns_clean)

        # Benchmark-relative metrics
        beta = None
        tracking_error = None
        information_ratio = None

        if benchmark_returns is not None:
            benchmark_clean = benchmark_returns.dropna()
            aligned_returns, aligned_benchmark = returns_clean.align(
                benchmark_clean, join="inner"
            )

            if len(aligned_returns) > 1:
                beta = self._calculate_beta(aligned_returns, aligned_benchmark)
                tracking_error = self._calculate_tracking_error(
                    aligned_returns, aligned_benchmark
                )
                information_ratio = self._calculate_information_ratio(
                    aligned_returns, aligned_benchmark
                )

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_drawdown,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            beta=beta,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
        )

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _calculate_beta(self, returns: pd.Series, benchmark: pd.Series) -> float:
        """Calculate beta relative to benchmark."""
        covariance = np.cov(returns, benchmark)[0, 1]
        benchmark_variance = np.var(benchmark)
        return covariance / benchmark_variance if benchmark_variance > 0 else 0

    def _calculate_tracking_error(
        self, returns: pd.Series, benchmark: pd.Series
    ) -> float:
        """Calculate tracking error."""
        active_returns = returns - benchmark
        return active_returns.std() * np.sqrt(252)  # Annualized

    def _calculate_information_ratio(
        self, returns: pd.Series, benchmark: pd.Series
    ) -> float:
        """Calculate information ratio."""
        active_returns = returns - benchmark
        active_return = active_returns.mean() * 252  # Annualized
        tracking_error = active_returns.std() * np.sqrt(252)  # Annualized
        return active_return / tracking_error if tracking_error > 0 else 0

    def add_stress_scenario(self, name: str, scenario: Dict[str, float]):
        """Add a stress testing scenario."""
        self.stress_scenarios[name] = scenario

    def stress_test_portfolio(
        self,
        weights: pd.Series,
        asset_returns: pd.DataFrame,
        scenarios: List[str] = None,
    ) -> List[StressTestResult]:
        """Perform stress testing on portfolio."""
        if scenarios is None:
            scenarios = list(self.stress_scenarios.keys())

        results = []

        for scenario_name in scenarios:
            if scenario_name not in self.stress_scenarios:
                logger.warning(f"Scenario {scenario_name} not found")
                continue

            scenario = self.stress_scenarios[scenario_name]

            # Apply scenario shocks to asset returns
            shocked_returns = pd.Series(index=weights.index, dtype=float)

            for asset in weights.index:
                if asset in scenario:
                    shocked_returns[asset] = scenario[asset]
                else:
                    # Use historical average if not specified in scenario
                    if asset in asset_returns.columns:
                        shocked_returns[asset] = asset_returns[asset].mean()
                    else:
                        shocked_returns[asset] = 0.0

            # Calculate portfolio return under stress
            portfolio_return = np.dot(weights.values, shocked_returns.values)

            # Calculate portfolio value change (assuming initial value of 1)
            portfolio_value_change = portfolio_return

            # Calculate risk metrics for the stressed scenario
            # For this, we'll use the shocked returns as a single observation
            stressed_portfolio_returns = pd.Series([portfolio_return])
            risk_metrics = self.calculate_portfolio_risk(stressed_portfolio_returns)

            results.append(
                StressTestResult(
                    scenario_name=scenario_name,
                    portfolio_return=portfolio_return,
                    portfolio_value_change=portfolio_value_change,
                    asset_returns=shocked_returns,
                    risk_metrics=risk_metrics,
                    metadata={"scenario_definition": scenario},
                )
            )

        return results

    def create_historical_scenarios(
        self, returns_data: pd.DataFrame, crisis_periods: List[Tuple[str, str]]
    ) -> None:
        """Create historical stress scenarios from crisis periods."""
        for start_date, end_date in crisis_periods:
            try:
                period_data = returns_data.loc[start_date:end_date]
                if len(period_data) > 0:
                    # Calculate cumulative returns during crisis
                    cumulative_returns = (1 + period_data).prod() - 1
                    scenario_name = f"Historical_{start_date}_{end_date}"
                    self.add_stress_scenario(
                        scenario_name, cumulative_returns.to_dict()
                    )
            except Exception as e:
                logger.warning(
                    f"Could not create scenario for {start_date}-{end_date}: {e}"
                )

    def monte_carlo_stress_test(
        self,
        weights: pd.Series,
        covariance_matrix: pd.DataFrame,
        n_simulations: int = 1000,
        confidence_levels: List[float] = [0.95, 0.99],
    ) -> Dict[str, Any]:
        """Perform Monte Carlo stress testing."""
        n_assets = len(weights)

        # Generate random scenarios using multivariate normal distribution
        mean_returns = np.zeros(n_assets)  # Assume zero mean for stress testing

        try:
            # Generate correlated random shocks
            random_shocks = np.random.multivariate_normal(
                mean_returns, covariance_matrix.values, n_simulations
            )
        except np.linalg.LinAlgError:
            # If covariance matrix is not positive definite, use diagonal
            logger.warning("Covariance matrix not positive definite, using diagonal")
            diag_cov = np.diag(np.diag(covariance_matrix.values))
            random_shocks = np.random.multivariate_normal(
                mean_returns, diag_cov, n_simulations
            )

        # Calculate portfolio returns for each simulation
        portfolio_returns = np.dot(random_shocks, weights.values)

        # Calculate stress test metrics
        results = {}

        for confidence_level in confidence_levels:
            var = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)
            tail_returns = portfolio_returns[portfolio_returns <= -var]
            cvar = -np.mean(tail_returns) if len(tail_returns) > 0 else var

            results[f"VaR_{int(confidence_level * 100)}"] = var
            results[f"CVaR_{int(confidence_level * 100)}"] = cvar

        results["worst_case"] = portfolio_returns.min()
        results["best_case"] = portfolio_returns.max()
        results["mean_return"] = portfolio_returns.mean()
        results["volatility"] = portfolio_returns.std()
        results["simulations"] = portfolio_returns.tolist()

        return results

    def tail_risk_analysis(
        self, returns: pd.Series, threshold_percentile: float = 0.95
    ) -> Dict[str, float]:
        """Perform tail risk analysis using Extreme Value Theory."""
        returns_clean = returns.dropna()

        if len(returns_clean) < 50:  # Need sufficient data for EVT
            logger.warning("Insufficient data for tail risk analysis")
            return {}

        # Define threshold
        threshold = np.percentile(returns_clean, threshold_percentile * 100)

        # Extract exceedances (losses beyond threshold)
        exceedances = returns_clean[returns_clean < threshold] - threshold

        if len(exceedances) < 10:
            logger.warning("Insufficient exceedances for tail risk analysis")
            return {}

        # Fit Generalized Pareto Distribution to exceedances
        try:
            # Fit GPD using method of moments or MLE
            shape, loc, scale = stats.genpareto.fit(-exceedances, floc=0)

            # Calculate tail risk metrics
            results = {
                "threshold": threshold,
                "n_exceedances": len(exceedances),
                "exceedance_rate": len(exceedances) / len(returns_clean),
                "gpd_shape": shape,
                "gpd_scale": scale,
            }

            # Calculate tail VaR and CVaR using GPD
            for confidence_level in [0.95, 0.99, 0.999]:
                if confidence_level > threshold_percentile:
                    # Tail probability
                    p = (1 - confidence_level) / (1 - threshold_percentile)

                    # GPD quantile
                    if abs(shape) > 1e-6:
                        tail_var = threshold + (scale / shape) * (p ** (-shape) - 1)
                    else:
                        tail_var = threshold + scale * np.log(1 / p)

                    results[f"tail_var_{int(confidence_level * 100)}"] = -tail_var

                    # Tail CVaR
                    if shape < 1:
                        tail_cvar = tail_var + (scale + shape * tail_var) / (1 - shape)
                        results[f"tail_cvar_{int(confidence_level * 100)}"] = -tail_cvar

            return results

        except Exception as e:
            logger.error(f"Error in tail risk analysis: {e}")
            return {}


# Pre-defined stress scenarios
DEFAULT_STRESS_SCENARIOS = {
    "market_crash": {
        "SPY": -0.20,
        "QQQ": -0.25,
        "IWM": -0.30,
        "EFA": -0.18,
        "EEM": -0.35,
        "TLT": 0.05,
        "GLD": 0.02,
        "VIX": 2.0,
    },
    "interest_rate_shock": {
        "TLT": -0.15,
        "IEF": -0.08,
        "SHY": -0.02,
        "TIPS": -0.10,
        "SPY": -0.05,
        "QQQ": -0.08,
        "XLF": -0.12,
        "REITs": -0.20,
    },
    "credit_crisis": {
        "HYG": -0.25,
        "LQD": -0.12,
        "XLF": -0.30,
        "SPY": -0.15,
        "TLT": 0.08,
        "GLD": 0.05,
        "USD": 0.03,
    },
    "inflation_shock": {
        "TIPS": 0.05,
        "GLD": 0.15,
        "DBC": 0.20,
        "TLT": -0.20,
        "SPY": -0.08,
        "QQQ": -0.12,
        "XLE": 0.10,
    },
    "geopolitical_crisis": {
        "SPY": -0.10,
        "EFA": -0.15,
        "EEM": -0.25,
        "GLD": 0.08,
        "TLT": 0.03,
        "VIX": 1.5,
        "USD": 0.02,
    },
}
