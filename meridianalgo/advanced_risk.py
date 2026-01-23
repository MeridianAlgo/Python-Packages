"""
MeridianAlgo Advanced Risk Management Module

Comprehensive risk management and analysis including advanced VaR methods,
stress testing, scenario analysis, and risk budgeting. Integrates concepts from
PyPortfolioOpt, RiskMetrics, and other leading risk management libraries.
"""

import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm
from scipy.stats import t as t_dist

# Suppress warnings
warnings.filterwarnings("ignore")

# Check for optional dependencies
try:
    # Optional sklearn imports available but not directly used
    import sklearn.covariance  # noqa: F401
    import sklearn.decomposition  # noqa: F401

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class AdvancedVaR:
    """
    Advanced Value at Risk calculation methods.

    Features:
    - Historical VaR
    - Parametric VaR (Normal, t-distribution)
    - Monte Carlo VaR
    - Conditional VaR (Expected Shortfall)
    - Component VaR
    - Marginal VaR
    - Incremental VaR
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize Advanced VaR calculator.

        Args:
            confidence_level: Confidence level for VaR calculation
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def historical_var(
        self,
        returns: pd.DataFrame,
        portfolio_weights: Optional[np.ndarray] = None,
        method: str = "simple",
    ) -> Dict[str, Union[float, pd.Series]]:
        """
        Calculate Historical VaR.

        Args:
            returns: DataFrame of asset returns
            portfolio_weights: Portfolio weights (equal weights if None)
            method: 'simple' or 'weighted' historical VaR

        Returns:
            Historical VaR results
        """
        if portfolio_weights is None:
            portfolio_weights = np.ones(returns.shape[1]) / returns.shape[1]

        # Calculate portfolio returns
        portfolio_returns = (returns * portfolio_weights).sum(axis=1)

        if method == "simple":
            # Simple historical VaR
            var = portfolio_returns.quantile(self.alpha)

            # Calculate weights for historical VaR
            weights = np.ones(len(portfolio_returns)) / len(portfolio_returns)

        elif method == "weighted":
            # Weighted historical VaR (exponential decay)
            decay_factor = 0.94
            weights = np.array(
                [(decay_factor**i) for i in range(len(portfolio_returns))][::-1]
            )
            weights = weights / weights.sum()

            # Sort returns and calculate weighted VaR
            sorted_returns = portfolio_returns.sort_values()
            sorted_weights = weights[np.argsort(portfolio_returns.values)]

            cumulative_weights = np.cumsum(sorted_weights)
            var_idx = np.argmax(cumulative_weights >= self.alpha)
            var = sorted_returns.iloc[var_idx]

        else:
            raise ValueError(f"Unknown method: {method}")

        # Expected Shortfall (CVaR)
        cvar = portfolio_returns[portfolio_returns <= var].mean()

        return {
            "var": var,
            "cvar": cvar,
            "portfolio_returns": portfolio_returns,
            "method": method,
            "weights": weights,
        }

    def parametric_var(
        self,
        returns: pd.DataFrame,
        portfolio_weights: Optional[np.ndarray] = None,
        distribution: str = "normal",
        degrees_of_freedom: Optional[float] = None,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate Parametric VaR.

        Args:
            returns: DataFrame of asset returns
            portfolio_weights: Portfolio weights
            distribution: 'normal', 't', or 'skewed_t'
            degrees_of_freedom: Degrees of freedom for t-distribution

        Returns:
            Parametric VaR results
        """
        if portfolio_weights is None:
            portfolio_weights = np.ones(returns.shape[1]) / returns.shape[1]

        # Calculate portfolio mean and variance
        portfolio_returns = (returns * portfolio_weights).sum(axis=1)
        portfolio_mean = portfolio_returns.mean()
        portfolio_var = portfolio_returns.var()

        if distribution == "normal":
            # Normal distribution VaR
            z_score = norm.ppf(self.alpha)
            var = portfolio_mean + np.sqrt(portfolio_var) * z_score

            # Expected Shortfall
            cvar = portfolio_mean + np.sqrt(portfolio_var) * (
                norm.pdf(z_score) / self.alpha
            )

        elif distribution == "t":
            # t-distribution VaR
            if degrees_of_freedom is None:
                # Estimate degrees of freedom
                dof_est = self._estimate_t_dof(portfolio_returns)
            else:
                dof_est = degrees_of_freedom

            t_score = t_dist.ppf(self.alpha, dof_est)
            scaling_factor = np.sqrt((dof_est - 2) / dof_est)
            var = portfolio_mean + np.sqrt(portfolio_var) * scaling_factor * t_score

            # Expected Shortfall for t-distribution
            if self.alpha < 0.5:
                cvar = portfolio_mean + np.sqrt(portfolio_var) * scaling_factor * (
                    t_dist.pdf(t_score, dof_est)
                    * (dof_est + t_score**2)
                    / ((dof_est - 1) * self.alpha)
                )
            else:
                cvar = portfolio_returns[portfolio_returns <= var].mean()

        elif distribution == "skewed_t":
            # Skewed t-distribution VaR (simplified)
            var = self._skewed_t_var(portfolio_returns, self.alpha)
            cvar = portfolio_returns[portfolio_returns <= var].mean()

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        return {
            "var": var,
            "cvar": cvar,
            "distribution": distribution,
            "portfolio_mean": portfolio_mean,
            "portfolio_var": portfolio_var,
            "degrees_of_freedom": degrees_of_freedom,
        }

    def monte_carlo_var(
        self,
        returns: pd.DataFrame,
        portfolio_weights: Optional[np.ndarray] = None,
        n_simulations: int = 10000,
        method: str = "gaussian",
        time_horizon: int = 1,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate Monte Carlo VaR.

        Args:
            returns: DataFrame of asset returns
            portfolio_weights: Portfolio weights
            n_simulations: Number of simulations
            method: 'gaussian', 'historical', or 'bootstrap'
            time_horizon: Time horizon in days

        Returns:
            Monte Carlo VaR results
        """
        if portfolio_weights is None:
            portfolio_weights = np.ones(returns.shape[1]) / returns.shape[1]

        returns.shape[1]

        # Calculate covariance matrix
        cov_matrix = returns.cov()
        mean_returns = returns.mean()

        if method == "gaussian":
            # Gaussian simulation
            simulated_returns = np.random.multivariate_normal(
                mean_returns.values, cov_matrix.values, n_simulations
            )

        elif method == "historical":
            # Historical bootstrap
            simulated_returns = returns.sample(n_simulations, replace=True).values

        elif method == "bootstrap":
            # Block bootstrap for time series dependence
            block_size = 20
            n_blocks = n_simulations // block_size

            simulated_returns = []
            for _ in range(n_blocks):
                start_idx = np.random.randint(0, len(returns) - block_size)
                block = returns.iloc[start_idx : start_idx + block_size].values
                simulated_returns.append(block)

            simulated_returns = np.vstack(simulated_returns)[:n_simulations]

        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate portfolio returns
        portfolio_sim_returns = (simulated_returns * portfolio_weights).sum(axis=1)

        # Scale for time horizon
        portfolio_sim_returns *= np.sqrt(time_horizon)

        # Calculate VaR and CVaR
        var = np.percentile(portfolio_sim_returns, self.alpha * 100)
        cvar = portfolio_sim_returns[portfolio_sim_returns <= var].mean()

        return {
            "var": var,
            "cvar": cvar,
            "simulated_returns": portfolio_sim_returns,
            "method": method,
            "n_simulations": n_simulations,
            "time_horizon": time_horizon,
        }

    def component_var(
        self, returns: pd.DataFrame, portfolio_weights: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate Component VaR for each asset.

        Args:
            returns: DataFrame of asset returns
            portfolio_weights: Portfolio weights

        Returns:
            Component VaR for each asset
        """
        # Calculate portfolio VaR
        portfolio_var_result = self.parametric_var(returns, portfolio_weights)
        portfolio_var = portfolio_var_result["var"]

        # Calculate marginal VaR for each asset
        cov_matrix = returns.cov()
        portfolio_vol = np.sqrt(portfolio_weights @ cov_matrix @ portfolio_weights)

        component_var = {}
        marginal_var = {}

        for i, asset in enumerate(returns.columns):
            # Marginal VaR
            marginal_var_i = cov_matrix.iloc[i, :] @ portfolio_weights / portfolio_vol
            marginal_var[asset] = marginal_var_i

            # Component VaR
            component_var_i = (
                portfolio_weights[i] * marginal_var_i * portfolio_var / portfolio_vol
            )
            component_var[asset] = component_var_i

        # Check that component VaR sums to portfolio VaR
        total_component_var = sum(component_var.values())

        return {
            "component_var": component_var,
            "marginal_var": marginal_var,
            "portfolio_var": portfolio_var,
            "total_component_var": total_component_var,
            "var_decomposition_error": abs(total_component_var - portfolio_var),
        }

    def incremental_var(
        self,
        returns: pd.DataFrame,
        current_weights: np.ndarray,
        asset_index: int,
        position_change: float,
    ) -> Dict[str, float]:
        """
        Calculate Incremental VaR for a position change.

        Args:
            returns: DataFrame of asset returns
            current_weights: Current portfolio weights
            asset_index: Index of asset to change
            position_change: Change in position (weight)

        Returns:
            Incremental VaR
        """
        # Current portfolio VaR
        current_var = self.parametric_var(returns, current_weights)["var"]

        # New weights after position change
        new_weights = current_weights.copy()
        new_weights[asset_index] += position_change

        # Rebalance to maintain sum of weights = 1
        new_weights = new_weights / new_weights.sum()

        # New portfolio VaR
        new_var = self.parametric_var(returns, new_weights)["var"]

        # Incremental VaR
        incremental_var = new_var - current_var

        return {
            "current_var": current_var,
            "new_var": new_var,
            "incremental_var": incremental_var,
            "position_change": position_change,
            "asset": returns.columns[asset_index],
        }

    def _estimate_t_dof(self, returns: pd.Series) -> float:
        """Estimate degrees of freedom for t-distribution."""
        # Method of moments estimator
        excess_kurtosis = returns.kurtosis()

        # For t-distribution: kurtosis = 6 / (dof - 4)
        if excess_kurtosis > 0:
            dof_est = 6 / excess_kurtosis + 4
        else:
            dof_est = 10  # Default value

        return max(dof_est, 2.1)  # Minimum dof > 2

    def _skewed_t_var(self, returns: pd.Series, alpha: float) -> float:
        """Simplified skewed t-distribution VaR."""
        # This is a simplified implementation
        # In practice, you would use a more sophisticated method
        normal_var = returns.quantile(alpha)

        # Adjust for skewness
        skewness = returns.skew()
        if skewness < 0:  # Left-skewed
            adjustment = 1 + abs(skewness) * 0.1
        else:
            adjustment = 1 - skewness * 0.05

        return normal_var * adjustment


class StressTesting:
    """
    Comprehensive stress testing and scenario analysis.

    Features:
    - Historical stress scenarios
    - Hypothetical stress scenarios
    - Factor stress testing
    - Reverse stress testing
    - Extreme value theory
    """

    def __init__(self):
        """Initialize stress testing framework."""
        self.historical_scenarios = {}
        self.hypothetical_scenarios = {}

    def add_historical_scenario(
        self, name: str, returns_data: pd.DataFrame, start_date: str, end_date: str
    ):
        """
        Add historical stress scenario.

        Args:
            name: Scenario name
            returns_data: Historical returns data
            start_date: Scenario start date
            end_date: Scenario end date
        """
        scenario_returns = returns_data.loc[start_date:end_date]

        self.historical_scenarios[name] = {
            "returns": scenario_returns,
            "start_date": start_date,
            "end_date": end_date,
            "duration": len(scenario_returns),
            "cumulative_returns": (1 + scenario_returns).prod() - 1,
        }

    def add_hypothetical_scenario(
        self,
        name: str,
        scenario_shocks: Dict[str, float],
        correlation_matrix: Optional[pd.DataFrame] = None,
    ):
        """
        Add hypothetical stress scenario.

        Args:
            name: Scenario name
            scenario_shocks: Dictionary of asset shocks
            correlation_matrix: Asset correlation matrix
        """
        self.hypothetical_scenarios[name] = {
            "shocks": scenario_shocks,
            "correlation_matrix": correlation_matrix,
        }

    def run_historical_stress_test(
        self, portfolio_weights: np.ndarray, current_portfolio_value: float
    ) -> Dict[str, Dict[str, float]]:
        """
        Run historical stress tests.

        Args:
            portfolio_weights: Portfolio weights
            current_portfolio_value: Current portfolio value

        Returns:
            Stress test results
        """
        results = {}

        for scenario_name, scenario_data in self.historical_scenarios.items():
            scenario_returns = scenario_data["returns"]

            # Calculate portfolio returns during scenario
            portfolio_scenario_returns = (scenario_returns * portfolio_weights).sum(
                axis=1
            )

            # Calculate portfolio value change
            cumulative_return = (1 + portfolio_scenario_returns).prod() - 1
            portfolio_value_change = current_portfolio_value * cumulative_return

            # Calculate worst loss
            worst_loss = portfolio_scenario_returns.min() * current_portfolio_value

            # Calculate maximum drawdown during scenario
            cumulative_returns = (1 + portfolio_scenario_returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min() * current_portfolio_value

            results[scenario_name] = {
                "cumulative_return": cumulative_return,
                "portfolio_value_change": portfolio_value_change,
                "worst_loss": worst_loss,
                "max_drawdown": max_drawdown,
                "volatility": portfolio_scenario_returns.std() * np.sqrt(252),
                "duration": scenario_data["duration"],
            }

        return results

    def run_hypothetical_stress_test(
        self,
        portfolio_weights: np.ndarray,
        current_portfolio_value: float,
        base_returns: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run hypothetical stress tests.

        Args:
            portfolio_weights: Portfolio weights
            current_portfolio_value: Current portfolio value
            base_returns: Base returns for correlation

        Returns:
            Stress test results
        """
        results = {}

        for scenario_name, scenario_data in self.hypothetical_scenarios.items():
            shocks = scenario_data["shocks"]
            correlation_matrix = scenario_data["correlation_matrix"]

            if correlation_matrix is not None:
                # Generate correlated shocks
                n_assets = len(portfolio_weights)
                correlated_shocks = self._generate_correlated_shocks(
                    shocks, correlation_matrix, n_assets
                )
            else:
                correlated_shocks = shocks

            # Calculate portfolio impact
            portfolio_shock = sum(
                correlated_shocks.get(asset, 0) * weight
                for asset, weight in zip(base_returns.columns, portfolio_weights)
            )

            portfolio_value_change = current_portfolio_value * portfolio_shock

            results[scenario_name] = {
                "portfolio_shock": portfolio_shock,
                "portfolio_value_change": portfolio_value_change,
                "worst_case_loss": min(portfolio_value_change, 0),
                "shocks_applied": correlated_shocks,
            }

        return results

    def factor_stress_test(
        self,
        factor_returns: pd.DataFrame,
        factor_loadings: pd.DataFrame,
        portfolio_weights: np.ndarray,
        factor_shocks: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Run factor-based stress test.

        Args:
            factor_returns: Factor returns
            factor_loadings: Asset factor loadings
            portfolio_weights: Portfolio weights
            factor_shocks: Factor shocks to apply

        Returns:
            Factor stress test results
        """
        # Calculate portfolio factor exposures
        portfolio_factor_exposures = factor_loadings.T @ portfolio_weights

        # Apply factor shocks
        factor_impact = sum(
            exposure * shock
            for factor, exposure in portfolio_factor_exposures.items()
            for shock in [factor_shocks.get(factor, 0)]
        )

        # Calculate residual impact (simplified)
        residual_risk = 0.1  # Assume 10% residual risk

        total_impact = factor_impact + residual_risk

        return {
            "factor_impact": factor_impact,
            "residual_impact": residual_risk,
            "total_impact": total_impact,
            "portfolio_exposures": portfolio_factor_exposures.to_dict(),
            "factor_shocks": factor_shocks,
        }

    def reverse_stress_test(
        self,
        portfolio_weights: np.ndarray,
        returns: pd.DataFrame,
        target_loss: float,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Reverse stress test - find scenarios that cause target loss.

        Args:
            portfolio_weights: Portfolio weights
            returns: Historical returns
            target_loss: Target loss to achieve
            confidence_level: Confidence level

        Returns:
            Reverse stress test results
        """
        # Calculate portfolio returns
        portfolio_returns = (returns * portfolio_weights).sum(axis=1)

        # Find historical periods with losses exceeding target
        loss_periods = portfolio_returns[portfolio_returns <= target_loss]

        if len(loss_periods) == 0:
            return {
                "found_scenarios": False,
                "message": "No historical scenarios found",
            }

        # Analyze characteristics of loss periods
        loss_analysis = {}

        for period in loss_periods.index:
            period_returns = returns.loc[period]

            # Calculate asset contributions to loss
            asset_contributions = period_returns * portfolio_weights

            # Calculate volatility during this period
            period_volatility = returns.rolling(20).std().loc[period]

            loss_analysis[period] = {
                "portfolio_return": portfolio_returns.loc[period],
                "asset_contributions": asset_contributions.to_dict(),
                "volatility": period_volatility.to_dict(),
                "worst_performing_asset": asset_contributions.idxmin(),
                "worst_asset_contribution": asset_contributions.min(),
            }

        # Summarize common characteristics
        common_assets = [
            analysis["worst_performing_asset"] for analysis in loss_analysis.values()
        ]
        asset_frequency = pd.Series(common_assets).value_counts()

        return {
            "found_scenarios": True,
            "n_scenarios": len(loss_periods),
            "loss_analysis": loss_analysis,
            "common_culprits": asset_frequency.head(3).to_dict(),
            "average_loss": loss_periods.mean(),
            "worst_loss": loss_periods.min(),
        }

    def extreme_value_analysis(
        self, returns: pd.Series, threshold: Optional[float] = None, method: str = "gpd"
    ) -> Dict[str, Any]:
        """
        Extreme Value Theory analysis.

        Args:
            returns: Return series
            threshold: Threshold for extreme values (None for automatic)
            method: 'gpd' (Generalized Pareto Distribution) or 'gev'

        Returns:
            EVT analysis results
        """
        if threshold is None:
            # Use 5th percentile as threshold for left tail
            threshold = returns.quantile(0.05)

        # Extract extreme values
        extreme_returns = returns[returns <= threshold]
        excesses = threshold - extreme_returns

        if len(excesses) < 30:
            return {"error": "Not enough extreme observations"}

        if method == "gpd":
            # Fit Generalized Pareto Distribution (simplified)
            # In practice, you would use maximum likelihood estimation
            shape_param = 0.1  # Simplified shape parameter
            scale_param = excesses.mean() * (1 - shape_param)

            # Calculate extreme quantiles
            extreme_var = threshold - (scale_param / shape_param) * (
                (len(returns) / len(excesses) * (1 - 0.01)) ** (-shape_param) - 1
            )

        else:  # GEV method
            # Simplified GEV analysis
            extreme_var = returns.quantile(0.01)

        return {
            "threshold": threshold,
            "n_exceedances": len(excesses),
            "exceedance_rate": len(excesses) / len(returns),
            "extreme_var": extreme_var,
            "shape_parameter": shape_param if method == "gpd" else None,
            "scale_parameter": scale_param if method == "gpd" else None,
            "method": method,
        }

    def _generate_correlated_shocks(
        self,
        base_shocks: Dict[str, float],
        correlation_matrix: pd.DataFrame,
        n_assets: int,
    ) -> Dict[str, float]:
        """Generate correlated shocks from base shocks."""
        # Simplified correlation adjustment
        correlated_shocks = base_shocks.copy()

        # Apply correlation effects (simplified)
        for asset1 in base_shocks:
            for asset2 in base_shocks:
                if (
                    asset1 != asset2
                    and asset1 in correlation_matrix.index
                    and asset2 in correlation_matrix.columns
                ):
                    correlation = correlation_matrix.loc[asset1, asset2]
                    correlated_shocks[asset1] += base_shocks[asset2] * correlation * 0.1

        return correlated_shocks


class RiskBudgeting:
    """
    Risk budgeting and allocation strategies.

    Features:
    - Equal risk contribution
    - Risk parity
    - Risk budgeting with constraints
    - Risk factor budgeting
    - Dynamic risk budgeting
    """

    def __init__(self):
        """Initialize risk budgeting framework."""
        pass

    def equal_risk_contribution(
        self, returns: pd.DataFrame, risk_free_rate: float = 0.0
    ) -> Dict[str, Union[np.ndarray, Dict[str, float]]]:
        """
        Calculate Equal Risk Contribution (ERC) portfolio.

        Args:
            returns: Asset returns
            risk_free_rate: Risk-free rate

        Returns:
            ERC portfolio weights and risk contributions
        """
        cov_matrix = returns.cov() * 252  # Annualized
        n_assets = len(returns.columns)

        # Iterative solution for ERC
        weights = np.ones(n_assets) / n_assets

        for _ in range(1000):  # Maximum iterations
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_risk = (cov_matrix @ weights) / portfolio_vol
            risk_contributions = weights * marginal_risk

            # Update weights to equalize risk contributions
            target_risk = risk_contributions.mean()
            weights = weights * (target_risk / risk_contributions)
            weights = weights / weights.sum()

            # Check convergence
            if np.std(risk_contributions) / risk_contributions.mean() < 0.01:
                break

        # Calculate final risk metrics
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        expected_return = returns.mean() * 252 @ weights
        sharpe_ratio = (expected_return - risk_free_rate) / portfolio_vol

        return {
            "weights": weights,
            "risk_contributions": risk_contributions,
            "portfolio_volatility": portfolio_vol,
            "expected_return": expected_return,
            "sharpe_ratio": sharpe_ratio,
            "risk_parity_error": np.std(risk_contributions) / risk_contributions.mean(),
        }

    def risk_parity_with_constraints(
        self,
        returns: pd.DataFrame,
        min_weights: Optional[np.ndarray] = None,
        max_weights: Optional[np.ndarray] = None,
        risk_budgets: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[np.ndarray, Dict[str, float]]]:
        """
        Risk parity with constraints.

        Args:
            returns: Asset returns
            min_weights: Minimum weight constraints
            max_weights: Maximum weight constraints
            risk_budgets: Target risk budgets for each asset

        Returns:
            Constrained risk parity weights
        """
        cov_matrix = returns.cov() * 252
        n_assets = len(returns.columns)

        if risk_budgets is None:
            risk_budgets = np.ones(n_assets) / n_assets

        # Optimization objective: minimize difference between actual and target risk contributions
        def objective(weights):
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_risk = (cov_matrix @ weights) / portfolio_vol
            risk_contributions = weights * marginal_risk

            # Normalize risk contributions
            normalized_contributions = risk_contributions / risk_contributions.sum()

            # Minimize squared difference from target risk budgets
            return np.sum((normalized_contributions - risk_budgets) ** 2)

        # Constraints
        constraints = []
        constraints.append(
            {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        )  # Weights sum to 1

        bounds = [(0, 1)] * n_assets  # Default bounds
        if min_weights is not None:
            bounds = [(min_weights[i], bounds[i][1]) for i in range(n_assets)]
        if max_weights is not None:
            bounds = [(bounds[i][0], max_weights[i]) for i in range(n_assets)]

        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets

        # Optimize
        result = optimize.minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        if result.success:
            weights = result.x

            # Calculate final metrics
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_risk = (cov_matrix @ weights) / portfolio_vol
            risk_contributions = weights * marginal_risk

            return {
                "weights": weights,
                "risk_contributions": risk_contributions,
                "portfolio_volatility": portfolio_vol,
                "optimization_success": True,
            }
        else:
            return {
                "weights": np.ones(n_assets) / n_assets,
                "risk_contributions": np.zeros(n_assets),
                "portfolio_volatility": 0,
                "optimization_success": False,
                "error": result.message,
            }

    def factor_risk_budgeting(
        self,
        factor_returns: pd.DataFrame,
        factor_loadings: pd.DataFrame,
        target_factor_risks: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[np.ndarray, Dict[str, float]]]:
        """
        Factor-based risk budgeting.

        Args:
            factor_returns: Factor returns
            factor_loadings: Asset factor loadings
            target_factor_risks: Target risk contributions for each factor

        Returns:
            Factor risk budgeted weights
        """
        n_factors = len(factor_returns.columns)
        n_assets = len(factor_loadings.index)

        if target_factor_risks is None:
            target_factor_risks = np.ones(n_factors) / n_factors

        # Calculate factor covariance matrix
        factor_cov = factor_returns.cov() * 252

        # Calculate asset covariance from factor model
        asset_cov = factor_loadings @ factor_cov @ factor_loadings.T

        # Add specific risk (simplified)
        specific_var = np.diag(asset_cov) * 0.1  # 10% specific risk
        asset_cov += np.diag(specific_var)

        # Optimize for target factor risk contributions
        def objective(weights):
            weights @ asset_cov @ weights

            # Calculate factor contributions to portfolio risk
            factor_contributions = []
            for i, factor in enumerate(factor_returns.columns):
                factor_loading = factor_loadings.iloc[:, i].values
                factor_contrib = (weights @ factor_loading) ** 2 * factor_cov.iloc[i, i]
                factor_contributions.append(factor_contrib)

            factor_contributions = np.array(factor_contributions)
            normalized_contributions = factor_contributions / factor_contributions.sum()

            return np.sum((normalized_contributions - target_factor_risks) ** 2)

        # Constraints
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, 1)] * n_assets

        # Optimize
        x0 = np.ones(n_assets) / n_assets
        result = optimize.minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        if result.success:
            weights = result.x

            # Calculate factor risk contributions
            portfolio_var = weights @ asset_cov @ weights
            factor_contributions = []
            for i, factor in enumerate(factor_returns.columns):
                factor_loading = factor_loadings.iloc[:, i].values
                factor_contrib = (weights @ factor_loading) ** 2 * factor_cov.iloc[i, i]
                factor_contributions.append(factor_contrib)

            return {
                "weights": weights,
                "factor_contributions": np.array(factor_contributions),
                "portfolio_variance": portfolio_var,
                "target_factor_risks": target_factor_risks,
                "optimization_success": True,
            }
        else:
            return {
                "weights": np.ones(n_assets) / n_assets,
                "factor_contributions": np.zeros(n_factors),
                "portfolio_variance": 0,
                "optimization_success": False,
                "error": result.message,
            }


# Utility functions
def calculate_risk_adjusted_returns(
    returns: pd.Series, risk_free_rate: float = 0.02, method: str = "sharpe"
) -> pd.Series:
    """
    Calculate risk-adjusted returns.

    Args:
        returns: Return series
        risk_free_rate: Risk-free rate
        method: 'sharpe', 'sortino', or 'information_ratio'

    Returns:
        Risk-adjusted returns
    """
    if method == "sharpe":
        excess_returns = returns - risk_free_rate / 252
        return excess_returns / excess_returns.rolling(252).std() * np.sqrt(252)

    elif method == "sortino":
        excess_returns = returns - risk_free_rate / 252
        downside_std = excess_returns[excess_returns < 0].rolling(252).std()
        return excess_returns / downside_std * np.sqrt(252)

    else:
        return returns


def calculate_rolling_var(
    returns: pd.Series, window: int = 252, confidence_level: float = 0.95
) -> pd.Series:
    """
    Calculate rolling Value at Risk.

    Args:
        returns: Return series
        window: Rolling window
        confidence_level: Confidence level

    Returns:
        Rolling VaR
    """
    return returns.rolling(window).quantile(1 - confidence_level)


# Export main classes and functions
__all__ = [
    "AdvancedVaR",
    "StressTesting",
    "RiskBudgeting",
    "calculate_risk_adjusted_returns",
    "calculate_rolling_var",
]
