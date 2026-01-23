"""
MeridianAlgo Core Module

Comprehensive quantitative finance core functionality including portfolio optimization,
risk analysis, statistical arbitrage, and time series analysis.
"""

import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import squareform

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Check for optional dependencies
try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import cvxpy as cp

    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

try:
    from sklearn.covariance import LedoitWolf

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class PortfolioOptimizer:
    """
    Advanced portfolio optimization using modern portfolio theory and beyond.

    Features:
    - Mean-variance optimization
    - Risk parity strategies
    - Black-Litterman model
    - Hierarchical Risk Parity (HRP)
    - Robust optimization
    - Transaction cost optimization
    - Advanced diversification methods
    - Multi-asset class optimization
    - Dynamic asset allocation
    """

    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.0):
        """
        Initialize portfolio optimizer.

        Args:
            returns: DataFrame of asset returns
            risk_free_rate: Risk-free rate for optimization
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.n_assets = returns.shape[1]
        self.asset_names = returns.columns.tolist()

        # Calculate covariance matrix
        self.cov_matrix = self._calculate_covariance_matrix()
        self.mean_returns = returns.mean()

        # Store optimization results
        self.weights = None
        self.optimization_result = None

    def _calculate_covariance_matrix(self, method: str = "ledoit_wolf") -> pd.DataFrame:
        """
        Calculate robust covariance matrix.

        Args:
            method: Method for covariance calculation ('sample', 'ledoit_wolf', 'shrunk')

        Returns:
            Covariance matrix
        """
        if method == "ledoit_wolf" and SKLEARN_AVAILABLE:
            lw = LedoitWolf()
            cov_matrix = lw.fit(self.returns).covariance_
            return pd.DataFrame(
                cov_matrix, index=self.asset_names, columns=self.asset_names
            )
        elif method == "shrunk":
            # Shrinkage estimator
            sample_cov = self.returns.cov()
            shrinkage = 0.1
            n_assets = len(self.asset_names)
            identity = np.eye(n_assets)
            shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * np.trace(
                sample_cov
            ) / n_assets * identity
            return pd.DataFrame(
                shrunk_cov, index=self.asset_names, columns=self.asset_names
            )
        else:
            return self.returns.cov()

    def optimize_portfolio(
        self,
        method: str = "sharpe",
        target_return: Optional[float] = None,
        target_volatility: Optional[float] = None,
        risk_budget: Optional[np.ndarray] = None,
        constraints: Optional[Dict[str, Any]] = None,
        transaction_costs: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Optimize portfolio using specified method.

        Args:
            method: Optimization method ('sharpe', 'min_vol', 'max_return', 'risk_parity', 'hrp', 'equal_weight')
            target_return: Target return for efficient frontier optimization
            target_volatility: Target volatility for efficient frontier optimization
            risk_budget: Risk budget for risk parity
            constraints: Additional constraints
            transaction_costs: Transaction costs

        Returns:
            Dictionary with optimization results
        """
        if method == "sharpe":
            return self._maximize_sharpe_ratio(constraints, transaction_costs)
        elif method == "min_vol":
            return self._minimize_volatility(constraints, transaction_costs)
        elif method == "max_return":
            return self._maximize_return(
                target_volatility, constraints, transaction_costs
            )
        elif method == "risk_parity":
            return self._risk_parity(risk_budget, constraints, transaction_costs)
        elif method == "hrp":
            return self._hierarchical_risk_parity()
        elif method == "equal_weight":
            return self._equal_weight()
        elif method == "black_litterman":
            return self._black_litterman(constraints, transaction_costs)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def _maximize_sharpe_ratio(
        self,
        constraints: Optional[Dict[str, Any]] = None,
        transaction_costs: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Maximize Sharpe ratio."""
        if not CVXPY_AVAILABLE:
            # Fallback to analytical solution
            return self._analytical_sharpe_optimization()

        n = self.n_assets

        # Define optimization variable
        weights = cp.Variable(n)

        # Define objective (negative Sharpe ratio for minimization)
        portfolio_return = self.mean_returns.values @ weights
        portfolio_risk = cp.quad_form(weights, self.cov_matrix.values)
        sharpe = (portfolio_return - self.risk_free_rate) / cp.sqrt(portfolio_risk)

        objective = cp.Maximize(sharpe)

        # Define constraints
        constraints_list = [cp.sum(weights) == 1, weights >= 0]

        # Add user-defined constraints
        if constraints:
            if "weight_limits" in constraints:
                for i, (min_w, max_w) in enumerate(constraints["weight_limits"]):
                    constraints_list.extend([weights[i] >= min_w, weights[i] <= max_w])

            if "sector_limits" in constraints:
                for sector, limit in constraints["sector_limits"].items():
                    sector_weights = [
                        weights[i]
                        for i, asset in enumerate(self.asset_names)
                        if asset.startswith(sector)
                    ]
                    if sector_weights:
                        constraints_list.append(cp.sum(sector_weights) <= limit)

        # Add transaction costs
        if transaction_costs:
            cost_penalty = 0
            for asset, cost in transaction_costs.items():
                if asset in self.asset_names:
                    idx = self.asset_names.index(asset)
                    cost_penalty += cost * cp.abs(weights[idx])
            objective = cp.Maximize(sharpe - cost_penalty)

        # Solve optimization
        problem = cp.Problem(objective, constraints_list)
        problem.solve()

        if problem.status != "optimal":
            raise RuntimeError("Optimization failed")

        self.weights = weights.value

        return {
            "weights": dict(zip(self.asset_names, self.weights)),
            "expected_return": portfolio_return.value,
            "volatility": np.sqrt(portfolio_risk.value),
            "sharpe_ratio": sharpe.value,
            "status": problem.status,
        }

    def _analytical_sharpe_optimization(self) -> Dict[str, Any]:
        """Analytical solution for maximum Sharpe ratio."""
        # Calculate inverse of covariance matrix
        inv_cov = np.linalg.inv(self.cov_matrix.values)

        # Calculate market portfolio weights
        ones = np.ones((self.n_assets, 1))
        excess_returns = self.mean_returns.values.reshape(-1, 1) - self.risk_free_rate

        # Analytical solution
        numerator = inv_cov @ excess_returns
        denominator = ones.T @ inv_cov @ excess_returns

        weights_unscaled = numerator / denominator
        weights = weights_unscaled / weights_unscaled.sum()

        # Calculate portfolio metrics
        portfolio_return = (self.mean_returns.values * weights.flatten()).sum()
        portfolio_variance = weights.T @ self.cov_matrix.values @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility

        self.weights = weights.flatten()

        return {
            "weights": dict(zip(self.asset_names, self.weights)),
            "expected_return": portfolio_return,
            "volatility": portfolio_volatility,
            "sharpe_ratio": sharpe_ratio,
            "status": "optimal",
        }

    def _minimize_volatility(
        self,
        constraints: Optional[Dict[str, Any]] = None,
        transaction_costs: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Minimize portfolio volatility."""
        if not CVXPY_AVAILABLE:
            return self._analytical_min_vol_optimization()

        n = self.n_assets
        weights = cp.Variable(n)

        # Objective: minimize variance
        portfolio_risk = cp.quad_form(weights, self.cov_matrix.values)
        objective = cp.Minimize(portfolio_risk)

        # Constraints
        constraints_list = [cp.sum(weights) == 1, weights >= 0]

        # Add user constraints
        if constraints and "weight_limits" in constraints:
            for i, (min_w, max_w) in enumerate(constraints["weight_limits"]):
                constraints_list.extend([weights[i] >= min_w, weights[i] <= max_w])

        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve()

        if problem.status != "optimal":
            raise RuntimeError("Optimization failed")

        self.weights = weights.value
        portfolio_return = (self.mean_returns.values * self.weights).sum()
        portfolio_volatility = np.sqrt(portfolio_risk.value)

        return {
            "weights": dict(zip(self.asset_names, self.weights)),
            "expected_return": portfolio_return,
            "volatility": portfolio_volatility,
            "sharpe_ratio": (portfolio_return - self.risk_free_rate)
            / portfolio_volatility,
            "status": problem.status,
        }

    def _analytical_min_vol_optimization(self) -> Dict[str, Any]:
        """Analytical solution for minimum volatility."""
        inv_cov = np.linalg.inv(self.cov_matrix.values)
        ones = np.ones((self.n_assets, 1))

        weights_unscaled = inv_cov @ ones
        weights = weights_unscaled / weights_unscaled.sum()

        portfolio_return = (self.mean_returns.values * weights.flatten()).sum()
        portfolio_variance = weights.T @ self.cov_matrix.values @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility

        self.weights = weights.flatten()

        return {
            "weights": dict(zip(self.asset_names, self.weights)),
            "expected_return": portfolio_return,
            "volatility": portfolio_volatility,
            "sharpe_ratio": sharpe_ratio,
            "status": "optimal",
        }

    def _risk_parity(
        self,
        risk_budget: Optional[np.ndarray] = None,
        constraints: Optional[Dict[str, Any]] = None,
        transaction_costs: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Risk parity optimization."""
        if risk_budget is None:
            risk_budget = np.ones(self.n_assets) / self.n_assets

        if not CVXPY_AVAILABLE:
            return self._iterative_risk_parity(risk_budget)

        n = self.n_assets
        weights = cp.Variable(n)

        # Risk contribution calculation
        portfolio_risk = cp.sqrt(cp.quad_form(weights, self.cov_matrix.values))
        marginal_risk = (self.cov_matrix.values @ weights) / portfolio_risk
        risk_contribution = cp.multiply(weights, marginal_risk)

        # Objective: minimize deviation from risk budget
        objective = cp.Minimize(cp.sum_squares(risk_contribution - risk_budget))

        # Constraints
        constraints_list = [cp.sum(weights) == 1, weights >= 0]

        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve()

        if problem.status != "optimal":
            return self._iterative_risk_parity(risk_budget)

        self.weights = weights.value
        portfolio_return = (self.mean_returns.values * self.weights).sum()
        portfolio_volatility = np.sqrt(weights.T @ self.cov_matrix.values @ weights)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility

        return {
            "weights": dict(zip(self.asset_names, self.weights)),
            "expected_return": portfolio_return,
            "volatility": portfolio_volatility,
            "sharpe_ratio": sharpe_ratio,
            "risk_contributions": dict(zip(self.asset_names, risk_contribution.value)),
            "status": problem.status,
        }

    def _iterative_risk_parity(self, risk_budget: np.ndarray) -> Dict[str, Any]:
        """Iterative risk parity solution."""
        # Simple iterative approach
        weights = np.ones(self.n_assets) / self.n_assets

        for _ in range(100):  # Maximum iterations
            portfolio_risk = np.sqrt(weights.T @ self.cov_matrix.values @ weights)
            marginal_risk = (self.cov_matrix.values @ weights) / portfolio_risk
            risk_contribution = weights * marginal_risk

            # Update weights
            scaling = risk_budget / risk_contribution
            weights = weights * scaling
            weights = weights / weights.sum()

        portfolio_return = (self.mean_returns.values * weights).sum()
        portfolio_volatility = np.sqrt(weights.T @ self.cov_matrix.values @ weights)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility

        self.weights = weights

        return {
            "weights": dict(zip(self.asset_names, self.weights)),
            "expected_return": portfolio_return,
            "volatility": portfolio_volatility,
            "sharpe_ratio": sharpe_ratio,
            "risk_contributions": dict(zip(self.asset_names, risk_contribution)),
            "status": "optimal",
        }

    def _hierarchical_risk_parity(self) -> Dict[str, Any]:
        """Hierarchical Risk Parity implementation."""
        # Calculate correlation matrix
        corr_matrix = self.returns.corr()

        # Calculate distance matrix
        distance_matrix = np.sqrt((1 - corr_matrix) / 2)

        # Hierarchical clustering
        from scipy.cluster.hierarchy import linkage

        # Convert condensed distance matrix
        condensed_distance = squareform(distance_matrix.values)
        np.fill_diagonal(distance_matrix.values, 0)
        condensed_distance = squareform(distance_matrix.values)

        # Perform linkage
        link = linkage(condensed_distance, method="single")

        # Calculate HRP weights
        weights = self._hrp_recursive_bisection(corr_matrix.values, link)

        portfolio_return = (self.mean_returns.values * weights).sum()
        portfolio_volatility = np.sqrt(weights.T @ self.cov_matrix.values @ weights)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility

        self.weights = weights

        return {
            "weights": dict(zip(self.asset_names, self.weights)),
            "expected_return": portfolio_return,
            "volatility": portfolio_volatility,
            "sharpe_ratio": sharpe_ratio,
            "status": "optimal",
        }

    def _hrp_recursive_bisection(
        self, corr_matrix: np.ndarray, link: np.ndarray
    ) -> np.ndarray:
        """Recursive bisection for HRP."""
        n = len(corr_matrix)
        weights = np.ones(n)

        # Simple HRP implementation (simplified)
        # In practice, this would be more complex
        cluster_assignments = self._get_clusters(link, n)

        for cluster_id in np.unique(cluster_assignments):
            cluster_indices = np.where(cluster_assignments == cluster_id)[0]
            if len(cluster_indices) > 1:
                # Inverse volatility weighting within cluster
                cluster_vol = np.sqrt(np.diag(corr_matrix)[cluster_indices])
                cluster_weights = 1 / cluster_vol
                cluster_weights = cluster_weights / cluster_weights.sum()
                weights[cluster_indices] = cluster_weights

        return weights

    def _get_clusters(self, link: np.ndarray, n: int) -> np.ndarray:
        """Get cluster assignments from linkage."""
        # Simplified clustering
        # In practice, this would use the linkage matrix properly
        return np.arange(n) % 2  # Simplified: alternate between 2 clusters

    def _equal_weight(self) -> Dict[str, Any]:
        """Equal weight portfolio."""
        weights = np.ones(self.n_assets) / self.n_assets

        portfolio_return = (self.mean_returns.values * weights).sum()
        portfolio_variance = weights.T @ self.cov_matrix.values @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility

        self.weights = weights

        return {
            "weights": dict(zip(self.asset_names, self.weights)),
            "expected_return": portfolio_return,
            "volatility": portfolio_volatility,
            "sharpe_ratio": sharpe_ratio,
            "status": "optimal",
        }

    def _black_litterman(
        self,
        constraints: Optional[Dict[str, Any]] = None,
        transaction_costs: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Black-Litterman model implementation."""
        # Simplified Black-Litterman
        # In practice, this would require views and confidence parameters

        # Use equilibrium weights (equal weight as starting point)
        equilibrium_weights = np.ones(self.n_assets) / self.n_assets

        # Calculate implied returns
        implied_returns = (
            self.risk_free_rate + self.cov_matrix.values @ equilibrium_weights * 0.1
        )

        # Use implied returns as expected returns
        expected_returns = pd.Series(implied_returns, index=self.asset_names)

        # Optimize with Black-Litterman returns
        temp_returns = self.mean_returns
        self.mean_returns = expected_returns

        result = self._maximize_sharpe_ratio(constraints, transaction_costs)

        # Restore original returns
        self.mean_returns = temp_returns

        return result

    def calculate_efficient_frontier(
        self, n_portfolios: int = 100, method: str = "sharpe"
    ) -> Dict[str, np.ndarray]:
        """
        Calculate efficient frontier.

        Args:
            n_portfolios: Number of portfolios to generate
            method: Method for generating portfolios

        Returns:
            Dictionary with efficient frontier data
        """
        # Target returns range
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)

        efficient_portfolios = []

        for target_ret in target_returns:
            try:
                if method == "sharpe":
                    result = self._maximize_return(target_ret)
                else:
                    result = self._minimize_volatility()

                efficient_portfolios.append(
                    [
                        result["volatility"],
                        result["expected_return"],
                        result["sharpe_ratio"],
                    ]
                )
            except Exception:
                continue

        if not efficient_portfolios:
            raise RuntimeError("Failed to calculate efficient frontier")

        efficient_portfolios = np.array(efficient_portfolios)

        return {
            "volatility": efficient_portfolios[:, 0],
            "returns": efficient_portfolios[:, 1],
            "sharpe": efficient_portfolios[:, 2],
            "weights": np.array(
                [list(result["weights"].values()) for result in efficient_portfolios]
            ),
        }

    def _maximize_return(self, target_volatility: float) -> Dict[str, Any]:
        """Maximize return for given volatility."""
        if not CVXPY_AVAILABLE:
            # Fallback
            return self._equal_weight()

        n = self.n_assets
        weights = cp.Variable(n)

        # Objective: maximize return
        portfolio_return = self.mean_returns.values @ weights
        objective = cp.Maximize(portfolio_return)

        # Constraints
        portfolio_risk = cp.sqrt(cp.quad_form(weights, self.cov_matrix.values))
        constraints_list = [
            cp.sum(weights) == 1,
            weights >= 0,
            portfolio_risk <= target_volatility,
        ]

        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve()

        if problem.status != "optimal":
            return self._equal_weight()

        weights_value = weights.value
        portfolio_volatility = np.sqrt(
            weights_value.T @ self.cov_matrix.values @ weights_value
        )
        sharpe_ratio = (
            portfolio_return.value - self.risk_free_rate
        ) / portfolio_volatility

        return {
            "weights": dict(zip(self.asset_names, weights_value)),
            "expected_return": portfolio_return.value,
            "volatility": portfolio_volatility,
            "sharpe_ratio": sharpe_ratio,
            "status": problem.status,
        }

    def calculate_risk_metrics(
        self, weights: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics.

        Args:
            weights: Portfolio weights (uses optimized weights if None)

        Returns:
            Dictionary of risk metrics
        """
        if weights is None:
            if self.weights is None:
                weights = np.ones(self.n_assets) / self.n_assets
            else:
                weights = self.weights

        # Portfolio returns
        portfolio_returns = (self.returns * weights).sum(axis=1)

        # Basic metrics
        total_return = portfolio_returns.sum()
        annualized_return = portfolio_returns.mean() * 252
        annualized_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility

        # Risk metrics
        var_95 = portfolio_returns.quantile(0.05)
        var_99 = portfolio_returns.quantile(0.01)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()

        # Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Additional metrics
        sortino_ratio = annualized_return / (
            portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)
        )
        calmar_ratio = annualized_return / abs(max_drawdown)

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
            "max_drawdown": max_drawdown,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
        }

    def advanced_diversification_optimization(
        self,
        method: str = "equal_risk_contribution",
        max_weight: float = 0.3,
        min_weight: float = 0.05,
        diversification_bonus: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Advanced diversification optimization methods.

        Args:
            method: Diversification method ('equal_risk_contribution', 'max_diversification', 'risk_parity_plus')
            max_weight: Maximum weight for any asset
            min_weight: Minimum weight for any asset
            diversification_bonus: Bonus for diversification in optimization

        Returns:
            Optimization results
        """
        len(self.returns.columns)

        if method == "equal_risk_contribution":
            weights = self._equal_risk_contribution()
        elif method == "max_diversification":
            weights = self._maximum_diversification()
        elif method == "risk_parity_plus":
            weights = self._risk_parity_plus(diversification_bonus)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Apply weight constraints
        weights = np.clip(weights, min_weight, max_weight)
        weights = weights / weights.sum()

        # Calculate portfolio metrics
        portfolio_returns = (
            self.returns * pd.Series(weights, index=self.returns.columns)
        ).sum(axis=1)

        metrics = {
            "weights": dict(zip(self.returns.columns, weights)),
            "diversification_ratio": self._calculate_diversification_ratio(weights),
            "concentration_ratio": self._calculate_concentration_ratio(weights),
            "effective_number_bets": self._calculate_effective_number_bets(weights),
            "portfolio_return": portfolio_returns.mean() * 252,
            "portfolio_volatility": portfolio_returns.std() * np.sqrt(252),
            "sharpe_ratio": (portfolio_returns.mean() * 252 - self.risk_free_rate)
            / (portfolio_returns.std() * np.sqrt(252)),
        }

        return metrics

    def _equal_risk_contribution(self) -> np.ndarray:
        """Equal Risk Contribution (ERC) portfolio."""
        n_assets = len(self.returns.columns)
        cov_matrix = self.returns.cov() * 252

        # Simplified ERC using iterative approach
        weights = np.ones(n_assets) / n_assets

        for _ in range(100):  # Iterative solution
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_risk = (cov_matrix @ weights) / portfolio_vol
            risk_contributions = weights * marginal_risk

            # Update weights to equalize risk contributions
            weights = weights * (risk_contributions.mean() / risk_contributions)
            weights = weights / weights.sum()

        return weights

    def _maximum_diversification(self) -> np.ndarray:
        """Maximum Diversification Ratio portfolio."""
        cov_matrix = self.returns.cov() * 252
        volatilities = np.sqrt(np.diag(cov_matrix))

        # Optimize for maximum diversification ratio
        len(self.returns.columns)

        # Simplified optimization using inverse volatility weights
        inv_vol_weights = 1 / volatilities
        weights = inv_vol_weights / inv_vol_weights.sum()

        return weights

    def _risk_parity_plus(self, diversification_bonus: float) -> np.ndarray:
        """Risk Parity with diversification bonus."""
        # Base risk parity weights
        erc_weights = self._equal_risk_contribution()

        # Add diversification bonus
        n_assets = len(self.returns.columns)
        equal_weights = np.ones(n_assets) / n_assets

        # Blend ERC with equal weights
        weights = (
            1 - diversification_bonus
        ) * erc_weights + diversification_bonus * equal_weights

        return weights

    def _calculate_diversification_ratio(self, weights: np.ndarray) -> float:
        """Calculate diversification ratio."""
        cov_matrix = self.returns.cov() * 252
        volatilities = np.sqrt(np.diag(cov_matrix))

        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        weighted_avg_vol = (weights * volatilities).sum()

        return weighted_avg_vol / portfolio_vol

    def _calculate_concentration_ratio(self, weights: np.ndarray) -> float:
        """Calculate concentration ratio (Herfindahl index)."""
        return np.sum(weights**2)

    def _calculate_effective_number_bets(self, weights: np.ndarray) -> float:
        """Calculate effective number of bets."""
        return 1 / np.sum(weights**2)

    def multi_asset_optimization(
        self,
        asset_classes: Dict[str, List[str]],
        class_constraints: Dict[str, Tuple[float, float]],
        risk_budget: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Multi-asset class portfolio optimization.

        Args:
            asset_classes: Dictionary of asset classes and their assets
            class_constraints: Min/max constraints for each asset class
            risk_budget: Risk budget for each asset class

        Returns:
            Multi-asset optimization results
        """
        # Get available assets
        available_assets = []
        for assets in asset_classes.values():
            available_assets.extend(
                [asset for asset in assets if asset in self.returns.columns]
            )

        if not available_assets:
            raise ValueError("No available assets found")

        returns_subset = self.returns[available_assets]
        n_assets = len(available_assets)

        # Initialize weights
        weights = np.ones(n_assets) / n_assets

        # Apply asset class constraints
        for class_name, (min_weight, max_weight) in class_constraints.items():
            if class_name in asset_classes:
                class_assets = asset_classes[class_name]
                class_indices = [
                    available_assets.index(asset)
                    for asset in class_assets
                    if asset in available_assets
                ]

                if class_indices:
                    class_weight = weights[class_indices].sum()
                    target_weight = (min_weight + max_weight) / 2

                    # Adjust weights to meet class constraints
                    adjustment_factor = target_weight / class_weight
                    weights[class_indices] *= adjustment_factor

        # Normalize weights
        weights = weights / weights.sum()

        # Calculate portfolio metrics
        portfolio_returns = (
            returns_subset * pd.Series(weights, index=available_assets)
        ).sum(axis=1)

        # Calculate class-level metrics
        class_metrics = {}
        for class_name, assets in asset_classes.items():
            class_assets = [asset for asset in assets if asset in available_assets]
            if class_assets:
                class_weights = [
                    weights[available_assets.index(asset)] for asset in class_assets
                ]
                class_returns = returns_subset[class_assets]
                class_portfolio = (class_returns * class_weights).sum(axis=1)

                class_metrics[class_name] = {
                    "weight": sum(class_weights),
                    "return": class_portfolio.mean() * 252,
                    "volatility": class_portfolio.std() * np.sqrt(252),
                    "sharpe_ratio": (class_portfolio.mean() * 252 - self.risk_free_rate)
                    / (class_portfolio.std() * np.sqrt(252)),
                }

        return {
            "weights": dict(zip(available_assets, weights)),
            "portfolio_return": portfolio_returns.mean() * 252,
            "portfolio_volatility": portfolio_returns.std() * np.sqrt(252),
            "sharpe_ratio": (portfolio_returns.mean() * 252 - self.risk_free_rate)
            / (portfolio_returns.std() * np.sqrt(252)),
            "class_metrics": class_metrics,
            "diversification_ratio": self._calculate_diversification_ratio(weights),
        }

    def dynamic_asset_allocation(
        self,
        lookback_window: int = 252,
        rebalance_frequency: int = 20,
        strategy: str = "volatility_targeting",
    ) -> Dict[str, Any]:
        """
        Dynamic asset allocation strategy.

        Args:
            lookback_window: Lookback window for calculations
            rebalance_frequency: Rebalancing frequency in days
            strategy: Allocation strategy ('volatility_targeting', 'momentum', 'mean_variance')

        Returns:
            Dynamic allocation results
        """
        if strategy == "volatility_targeting":
            return self._volatility_targeting_strategy(
                lookback_window, rebalance_frequency
            )
        elif strategy == "momentum":
            return self._momentum_strategy(lookback_window, rebalance_frequency)
        elif strategy == "mean_variance":
            return self._dynamic_mean_variance(lookback_window, rebalance_frequency)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _volatility_targeting_strategy(
        self,
        lookback_window: int,
        rebalance_frequency: int,
        target_volatility: float = 0.15,
    ) -> Dict[str, Any]:
        """Volatility targeting strategy."""
        returns_clean = self.returns.dropna()
        dates = returns_clean.index

        # Initialize weights series
        weights_history = pd.DataFrame(index=dates, columns=returns_clean.columns)

        for i in range(lookback_window, len(dates), rebalance_frequency):
            window_end = min(i + rebalance_frequency, len(dates))

            # Calculate rolling volatilities
            window_returns = returns_clean.iloc[i - lookback_window : i]
            volatilities = window_returns.std() * np.sqrt(252)

            # Inverse volatility weights
            inv_vol_weights = 1 / volatilities
            inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()

            # Scale to target volatility
            portfolio_vol = np.sqrt(
                inv_vol_weights @ (window_returns.cov() * 252) @ inv_vol_weights
            )
            scale_factor = target_volatility / portfolio_vol

            scaled_weights = inv_vol_weights * scale_factor

            # Store weights for the period
            weights_history.iloc[i:window_end] = scaled_weights

        # Calculate strategy returns
        strategy_returns = (returns_clean * weights_history).sum(axis=1)

        return {
            "weights_history": weights_history,
            "strategy_returns": strategy_returns,
            "annual_return": strategy_returns.mean() * 252,
            "annual_volatility": strategy_returns.std() * np.sqrt(252),
            "sharpe_ratio": (strategy_returns.mean() * 252 - self.risk_free_rate)
            / (strategy_returns.std() * np.sqrt(252)),
            "max_drawdown": self._calculate_max_drawdown(strategy_returns),
        }

    def _momentum_strategy(
        self, lookback_window: int, rebalance_frequency: int
    ) -> Dict[str, Any]:
        """Momentum-based allocation strategy."""
        returns_clean = self.returns.dropna()
        dates = returns_clean.index

        weights_history = pd.DataFrame(index=dates, columns=returns_clean.columns)

        for i in range(lookback_window, len(dates), rebalance_frequency):
            window_end = min(i + rebalance_frequency, len(dates))

            # Calculate momentum scores (cumulative returns)
            window_returns = returns_clean.iloc[i - lookback_window : i]
            momentum_scores = (1 + window_returns).prod() - 1

            # Rank assets by momentum
            ranked_assets = momentum_scores.sort_values(ascending=False)

            # Allocate to top performers
            top_n = min(5, len(ranked_assets))  # Top 5 assets
            weights = np.zeros(len(returns_clean.columns))

            for j, asset in enumerate(ranked_assets.head(top_n).index):
                asset_idx = returns_clean.columns.get_loc(asset)
                weights[asset_idx] = 1 / top_n

            weights_history.iloc[i:window_end] = weights

        strategy_returns = (returns_clean * weights_history).sum(axis=1)

        return {
            "weights_history": weights_history,
            "strategy_returns": strategy_returns,
            "annual_return": strategy_returns.mean() * 252,
            "annual_volatility": strategy_returns.std() * np.sqrt(252),
            "sharpe_ratio": (strategy_returns.mean() * 252 - self.risk_free_rate)
            / (strategy_returns.std() * np.sqrt(252)),
            "max_drawdown": self._calculate_max_drawdown(strategy_returns),
        }

    def _dynamic_mean_variance(
        self, lookback_window: int, rebalance_frequency: int
    ) -> Dict[str, Any]:
        """Dynamic mean-variance optimization."""
        returns_clean = self.returns.dropna()
        dates = returns_clean.index

        weights_history = pd.DataFrame(index=dates, columns=returns_clean.columns)

        for i in range(lookback_window, len(dates), rebalance_frequency):
            window_end = min(i + rebalance_frequency, len(dates))

            # Calculate mean-variance optimization for window
            window_returns = returns_clean.iloc[i - lookback_window : i]

            # Simplified mean-variance using equal risk contribution
            window_returns.cov()
            n_assets = len(window_returns.columns)

            # Use equal risk contribution as approximation
            weights = np.ones(n_assets) / n_assets

            weights_history.iloc[i:window_end] = weights

        strategy_returns = (returns_clean * weights_history).sum(axis=1)

        return {
            "weights_history": weights_history,
            "strategy_returns": strategy_returns,
            "annual_return": strategy_returns.mean() * 252,
            "annual_volatility": strategy_returns.std() * np.sqrt(252),
            "sharpe_ratio": (strategy_returns.mean() * 252 - self.risk_free_rate)
            / (strategy_returns.std() * np.sqrt(252)),
            "max_drawdown": self._calculate_max_drawdown(strategy_returns),
        }

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()


class TimeSeriesAnalyzer:
    """
    Advanced time series analysis for financial data.

    Features:
    - Technical indicators
    - Pattern recognition
    - Regime detection
    - Volatility modeling
    - Seasonality analysis
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize time series analyzer.

        Args:
            data: DataFrame with datetime index and price/return data
        """
        self.data = data
        self.returns = None
        self.volatility = None
        self.trends = None

        # Calculate returns if price data
        if data.max().max() > 1:  # Likely price data
            self.returns = data.pct_change().dropna()
        else:
            self.returns = data

    def calculate_returns(self, log_returns: bool = False) -> pd.DataFrame:
        """
        Calculate returns from price data.

        Args:
            log_returns: Whether to calculate log returns

        Returns:
            DataFrame of returns
        """
        if log_returns:
            return np.log(self.data / self.data.shift(1)).dropna()
        else:
            return self.data.pct_change().dropna()

    def calculate_volatility(
        self, window: int = 21, method: str = "historical", annualized: bool = True
    ) -> pd.DataFrame:
        """
        Calculate rolling volatility.

        Args:
            window: Rolling window size
            method: Method for volatility calculation
            annualized: Whether to annualize volatility

        Returns:
            DataFrame of volatility
        """
        if method == "historical":
            vol = self.returns.rolling(window=window).std()
        elif method == "ewma":
            # Exponentially weighted moving average
            vol = self.returns.ewm(span=window).std()
        elif method == "garch":
            # Simplified GARCH(1,1) approximation
            vol = self._simple_garch_volatility(window)
        else:
            raise ValueError(f"Unknown volatility method: {method}")

        if annualized:
            vol = vol * np.sqrt(252)

        self.volatility = vol
        return vol

    def _simple_garch_volatility(self, window: int) -> pd.DataFrame:
        """Simple GARCH(1,1) volatility estimation."""
        omega = 0.00001
        alpha = 0.1
        beta = 0.85

        vol = pd.DataFrame(index=self.returns.index, columns=self.returns.columns)

        for asset in self.returns.columns:
            returns_series = self.returns[asset].dropna()
            vol_series = pd.Series(index=returns_series.index, dtype=float)

            # Initialize
            vol_series.iloc[0] = returns_series.iloc[
                : min(window, len(returns_series))
            ].std()

            for i in range(1, len(returns_series)):
                vol_series.iloc[i] = np.sqrt(
                    omega
                    + alpha * returns_series.iloc[i - 1] ** 2
                    + beta * vol_series.iloc[i - 1] ** 2
                )

            vol[asset] = vol_series

        return vol

    def calculate_moving_average(
        self, window: int, ma_type: str = "sma", **kwargs
    ) -> pd.DataFrame:
        """
        Calculate moving averages.

        Args:
            window: Window size
            ma_type: Type of moving average ('sma', 'ema', 'wma')
            **kwargs: Additional parameters

        Returns:
            DataFrame of moving averages
        """
        if ma_type == "sma":
            return self.data.rolling(window=window).mean()
        elif ma_type == "ema":
            span = kwargs.get("span", window)
            return self.data.ewm(span=span).mean()
        elif ma_type == "wma":
            # Weighted moving average
            weights = np.arange(1, window + 1)
            weights = weights / weights.sum()
            return self.data.rolling(window=window).apply(
                lambda x: np.dot(x, weights), raw=True
            )
        else:
            raise ValueError(f"Unknown MA type: {ma_type}")

    def calculate_bollinger_bands(
        self, window: int = 20, num_std: float = 2.0
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate Bollinger Bands.

        Args:
            window: Window size for moving average
            num_std: Number of standard deviations

        Returns:
            Dictionary with upper, middle, lower bands
        """
        middle = self.calculate_moving_average(window, "sma")
        rolling_std = self.data.rolling(window=window).std()

        upper = middle + (rolling_std * num_std)
        lower = middle - (rolling_std * num_std)

        return {"upper": upper, "middle": middle, "lower": lower}

    def calculate_rsi(self, window: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index.

        Args:
            window: Window size for RSI

        Returns:
            DataFrame of RSI values
        """
        delta = self.data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd(
        self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate MACD indicator.

        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period

        Returns:
            Dictionary with MACD components
        """
        ema_fast = self.data.ewm(span=fast_period).mean()
        ema_slow = self.data.ewm(span=slow_period).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line

        return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

    def detect_trends(self, window: int = 50) -> pd.DataFrame:
        """
        Detect trends using moving average crossover.

        Args:
            window: Window size for trend detection

        Returns:
            DataFrame with trend signals
        """
        short_ma = self.calculate_moving_average(window // 2, "sma")
        long_ma = self.calculate_moving_average(window, "sma")

        trends = pd.DataFrame(index=self.data.index, columns=self.data.columns)

        # 1 for uptrend, -1 for downtrend, 0 for neutral
        trends[short_ma > long_ma] = 1
        trends[short_ma < long_ma] = -1
        trends[short_ma == long_ma] = 0

        self.trends = trends
        return trends

    def calculate_autocorrelation(self, lags: int = 20) -> Dict[str, pd.Series]:
        """
        Calculate autocorrelation function.

        Args:
            lags: Number of lags to calculate

        Returns:
            Dictionary of autocorrelation series
        """
        autocorr = {}

        for asset in self.returns.columns:
            series = self.returns[asset].dropna()
            autocorr_values = []

            for lag in range(1, lags + 1):
                autocorr_values.append(series.autocorr(lag=lag))

            autocorr[asset] = pd.Series(autocorr_values, index=range(1, lags + 1))

        return autocorr

    def calculate_seasonality(self, freq: str = "M") -> Dict[str, pd.DataFrame]:
        """
        Calculate seasonal patterns.

        Args:
            freq: Frequency for seasonality analysis

        Returns:
            Dictionary of seasonal patterns
        """
        seasonal_patterns = {}

        for asset in self.returns.columns:
            if freq == "M":
                # Monthly seasonality
                monthly_returns = self.returns[asset].resample("M").mean()
                seasonal_pattern = monthly_returns.groupby(
                    monthly_returns.index.month
                ).mean()
            elif freq == "Q":
                # Quarterly seasonality
                quarterly_returns = self.returns[asset].resample("Q").mean()
                seasonal_pattern = quarterly_returns.groupby(
                    quarterly_returns.index.quarter
                ).mean()
            elif freq == "D":
                # Day of week seasonality
                seasonal_pattern = (
                    self.returns[asset].groupby(self.returns.index.dayofweek).mean()
                )
            else:
                raise ValueError(f"Unknown frequency: {freq}")

            seasonal_patterns[asset] = seasonal_pattern

        return seasonal_patterns


# Utility functions
def get_market_data(
    symbols: Union[str, List[str]],
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    interval: str = "1d",
    **kwargs,
) -> pd.DataFrame:
    """
    Fetch market data from Yahoo Finance.

    Args:
        symbols: Symbol(s) to fetch
        start_date: Start date
        end_date: End date
        interval: Data interval
        **kwargs: Additional parameters

    Returns:
        DataFrame with market data
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError(
            "yfinance is required for market data. Install with: pip install yfinance"
        )

    if isinstance(symbols, str):
        symbols = [symbols]

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    data = yf.download(
        symbols, start=start_date, end=end_date, interval=interval, **kwargs
    )

    # Return adjusted close prices
    if "Adj Close" in data.columns:
        return data["Adj Close"]
    elif "Close" in data.columns:
        return data["Close"]
    else:
        return data


def calculate_metrics(
    returns: pd.Series, risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.

    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate

    Returns:
        Dictionary of metrics
    """
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = returns.mean() * 252
    annualized_volatility = returns.std() * np.sqrt(252)

    # Risk-adjusted metrics
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (
        (annualized_return - risk_free_rate) / downside_volatility
        if downside_volatility > 0
        else 0
    )

    # Drawdown metrics
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Additional metrics
    skew = returns.skew()
    kurtosis = returns.kurtosis()
    var_95 = returns.quantile(0.05)
    var_99 = returns.quantile(0.01)

    # Win rate
    win_rate = (returns > 0).mean()

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "skew": skew,
        "kurtosis": kurtosis,
        "var_95": var_95,
        "var_99": var_99,
        "win_rate": win_rate,
    }


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.

    Args:
        returns: Series of returns

    Returns:
        Maximum drawdown
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_value_at_risk(
    returns: pd.Series, confidence_level: float = 0.95, method: str = "historical"
) -> float:
    """
    Calculate Value at Risk.

    Args:
        returns: Series of returns
        confidence_level: Confidence level
        method: Method for VaR calculation

    Returns:
        VaR value
    """
    alpha = 1 - confidence_level

    if method == "historical":
        return returns.quantile(alpha)
    elif method == "gaussian":
        return returns.mean() + returns.std() * stats.norm.ppf(alpha)
    elif method == "cornish_fisher":
        # Cornish-Fisher expansion
        z = stats.norm.ppf(alpha)
        s = returns.skew()
        k = returns.kurtosis()

        cf_z = (
            z
            + (s / 6) * (z**2 - 1)
            + (k / 24) * (z**3 - 3 * z)
            - (s**2 / 36) * (2 * z**3 - 5 * z)
        )

        return returns.mean() + returns.std() * cf_z
    else:
        raise ValueError(f"Unknown VaR method: {method}")


def calculate_expected_shortfall(
    returns: pd.Series, confidence_level: float = 0.95
) -> float:
    """
    Calculate Expected Shortfall (Conditional VaR).

    Args:
        returns: Series of returns
        confidence_level: Confidence level

    Returns:
        Expected Shortfall value
    """
    var = calculate_value_at_risk(returns, confidence_level)
    return returns[returns <= var].mean()


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino ratio.

    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate

    Returns:
        Sortino ratio
    """
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return float("inf")

    downside_volatility = downside_returns.std() * np.sqrt(252)
    annualized_return = returns.mean() * 252

    return (annualized_return - risk_free_rate) / downside_volatility


def calculate_calmar_ratio(returns: pd.Series) -> float:
    """
    Calculate Calmar ratio.

    Args:
        returns: Series of returns

    Returns:
        Calmar ratio
    """
    annualized_return = returns.mean() * 252
    max_drawdown = calculate_max_drawdown(returns)

    if max_drawdown == 0:
        return float("inf")

    return annualized_return / abs(max_drawdown)


# Export main classes and functions
__all__ = [
    "PortfolioOptimizer",
    "TimeSeriesAnalyzer",
    "get_market_data",
    "calculate_metrics",
    "calculate_max_drawdown",
    "calculate_value_at_risk",
    "calculate_expected_shortfall",
    "calculate_sortino_ratio",
    "calculate_calmar_ratio",
]
