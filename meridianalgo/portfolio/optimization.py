"""
Advanced portfolio optimization algorithms including Black-Litterman, Risk Parity, and HRP.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

try:
    from .transaction_costs import TransactionCostOptimizer

    TRANSACTION_COSTS_AVAILABLE = True
except ImportError:
    TRANSACTION_COSTS_AVAILABLE = False
    warnings.warn("Transaction cost optimization not available.")

try:
    import cvxpy as cp

    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    warnings.warn("CVXPY not available. Some optimization features will be limited.")

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""

    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    optimization_method: str
    success: bool
    message: str
    metadata: Dict[str, Any] = None


class BaseOptimizer(ABC):
    """Abstract base class for portfolio optimizers."""

    def __init__(self, name: str):
        self.name = name
        self.last_result: Optional[OptimizationResult] = None

    @abstractmethod
    def optimize(
        self, expected_returns: pd.Series, covariance_matrix: pd.DataFrame, **kwargs
    ) -> OptimizationResult:
        """Optimize portfolio weights."""
        pass

    def validate_inputs(
        self, expected_returns: pd.Series, covariance_matrix: pd.DataFrame
    ) -> None:
        """Validate optimization inputs."""
        if len(expected_returns) != len(covariance_matrix):
            raise ValueError("Dimension mismatch between returns and covariance matrix")

        if not covariance_matrix.index.equals(covariance_matrix.columns):
            raise ValueError(
                "Covariance matrix must be square with matching index/columns"
            )

        if not expected_returns.index.equals(covariance_matrix.index):
            raise ValueError("Returns index must match covariance matrix index")

        # Check for positive semi-definite covariance matrix
        eigenvals = np.linalg.eigvals(covariance_matrix.values)
        if np.any(eigenvals < -1e-8):
            warnings.warn("Covariance matrix is not positive semi-definite")

    def calculate_portfolio_metrics(
        self,
        weights: pd.Series,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float = 0.0,
    ) -> Tuple[float, float, float]:
        """Calculate portfolio return, volatility, and Sharpe ratio."""
        portfolio_return = np.dot(weights.values, expected_returns.values)
        portfolio_variance = np.dot(
            weights.values, np.dot(covariance_matrix.values, weights.values)
        )
        portfolio_volatility = np.sqrt(portfolio_variance)

        if portfolio_volatility > 0:
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        else:
            sharpe_ratio = 0.0

        return portfolio_return, portfolio_volatility, sharpe_ratio


class PortfolioOptimizer(BaseOptimizer):
    """Standard portfolio optimizer with various objective functions."""

    def __init__(self):
        super().__init__("StandardOptimizer")

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        objective: str = "max_sharpe",
        constraints: Dict[str, Any] = None,
        bounds: Tuple[float, float] = (0.0, 1.0),
        risk_free_rate: float = 0.0,
        target_return: float = None,
        target_volatility: float = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio using various objectives.

        Args:
            expected_returns: Expected returns for assets
            covariance_matrix: Covariance matrix of asset returns
            objective: Optimization objective ('max_sharpe', 'min_volatility', 'max_return', 'target_return', 'target_volatility')
            constraints: Additional constraints
            bounds: Weight bounds for each asset
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            target_return: Target return for 'target_return' objective
            target_volatility: Target volatility for 'target_volatility' objective

        Returns:
            OptimizationResult with optimized weights and metrics
        """
        self.validate_inputs(expected_returns, covariance_matrix)

        len(expected_returns)
        constraints = constraints or {}

        # Define optimization variables
        if CVXPY_AVAILABLE:
            return self._optimize_cvxpy(
                expected_returns,
                covariance_matrix,
                objective,
                constraints,
                bounds,
                risk_free_rate,
                target_return,
                target_volatility,
            )
        else:
            return self._optimize_scipy(
                expected_returns,
                covariance_matrix,
                objective,
                constraints,
                bounds,
                risk_free_rate,
                target_return,
                target_volatility,
            )

    def _optimize_cvxpy(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        objective: str,
        constraints: Dict[str, Any],
        bounds: Tuple[float, float],
        risk_free_rate: float,
        target_return: float,
        target_volatility: float,
    ) -> OptimizationResult:
        """Optimize using CVXPY."""
        n_assets = len(expected_returns)
        w = cp.Variable(n_assets)

        # Basic constraints
        constraints_list = [cp.sum(w) == 1]  # Weights sum to 1

        # Bounds
        if bounds:
            constraints_list.extend([w >= bounds[0], w <= bounds[1]])

        # Additional constraints
        if "max_weight" in constraints:
            constraints_list.append(w <= constraints["max_weight"])
        if "min_weight" in constraints:
            constraints_list.append(w >= constraints["min_weight"])
        if "group_constraints" in constraints:
            for group_constraint in constraints["group_constraints"]:
                indices = group_constraint["assets"]
                if group_constraint["type"] == "max_weight":
                    constraints_list.append(
                        cp.sum(w[indices]) <= group_constraint["limit"]
                    )
                elif group_constraint["type"] == "min_weight":
                    constraints_list.append(
                        cp.sum(w[indices]) >= group_constraint["limit"]
                    )

        # Define objective
        mu = expected_returns.values
        Sigma = covariance_matrix.values

        if objective == "max_sharpe":
            # Maximize Sharpe ratio (equivalent to maximizing return/risk)
            portfolio_return = mu.T @ w
            portfolio_risk = cp.quad_form(w, Sigma)
            # Use auxiliary variable for Sharpe ratio maximization
            kappa = cp.Variable()
            y = cp.Variable(n_assets)

            constraints_list = [
                cp.sum(y) == 1,
                mu.T @ y - risk_free_rate * kappa == 1,
                cp.quad_form(y, Sigma) <= kappa**2,
                kappa >= 0,
            ]

            if bounds:
                constraints_list.extend(
                    [y >= bounds[0] * kappa, y <= bounds[1] * kappa]
                )

            prob = cp.Problem(cp.Maximize(kappa), constraints_list)
            prob.solve()

            if prob.status == cp.OPTIMAL:
                weights = pd.Series(y.value / kappa.value, index=expected_returns.index)
            else:
                return OptimizationResult(
                    weights=pd.Series(np.zeros(n_assets), index=expected_returns.index),
                    expected_return=0.0,
                    volatility=0.0,
                    sharpe_ratio=0.0,
                    optimization_method=self.name,
                    success=False,
                    message=f"Optimization failed: {prob.status}",
                )

        elif objective == "min_volatility":
            portfolio_risk = cp.quad_form(w, Sigma)
            prob = cp.Problem(cp.Minimize(portfolio_risk), constraints_list)
            prob.solve()

            if prob.status == cp.OPTIMAL:
                weights = pd.Series(w.value, index=expected_returns.index)
            else:
                return OptimizationResult(
                    weights=pd.Series(np.zeros(n_assets), index=expected_returns.index),
                    expected_return=0.0,
                    volatility=0.0,
                    sharpe_ratio=0.0,
                    optimization_method=self.name,
                    success=False,
                    message=f"Optimization failed: {prob.status}",
                )

        elif objective == "max_return":
            portfolio_return = mu.T @ w
            prob = cp.Problem(cp.Maximize(portfolio_return), constraints_list)
            prob.solve()

            if prob.status == cp.OPTIMAL:
                weights = pd.Series(w.value, index=expected_returns.index)
            else:
                return OptimizationResult(
                    weights=pd.Series(np.zeros(n_assets), index=expected_returns.index),
                    expected_return=0.0,
                    volatility=0.0,
                    sharpe_ratio=0.0,
                    optimization_method=self.name,
                    success=False,
                    message=f"Optimization failed: {prob.status}",
                )

        elif objective == "target_return" and target_return is not None:
            portfolio_return = mu.T @ w
            portfolio_risk = cp.quad_form(w, Sigma)
            constraints_list.append(portfolio_return >= target_return)

            prob = cp.Problem(cp.Minimize(portfolio_risk), constraints_list)
            prob.solve()

            if prob.status == cp.OPTIMAL:
                weights = pd.Series(w.value, index=expected_returns.index)
            else:
                return OptimizationResult(
                    weights=pd.Series(np.zeros(n_assets), index=expected_returns.index),
                    expected_return=0.0,
                    volatility=0.0,
                    sharpe_ratio=0.0,
                    optimization_method=self.name,
                    success=False,
                    message=f"Optimization failed: {prob.status}",
                )

        elif objective == "target_volatility" and target_volatility is not None:
            portfolio_return = mu.T @ w
            portfolio_risk = cp.quad_form(w, Sigma)
            constraints_list.append(portfolio_risk <= target_volatility**2)

            prob = cp.Problem(cp.Maximize(portfolio_return), constraints_list)
            prob.solve()

            if prob.status == cp.OPTIMAL:
                weights = pd.Series(w.value, index=expected_returns.index)
            else:
                return OptimizationResult(
                    weights=pd.Series(np.zeros(n_assets), index=expected_returns.index),
                    expected_return=0.0,
                    volatility=0.0,
                    sharpe_ratio=0.0,
                    optimization_method=self.name,
                    success=False,
                    message=f"Optimization failed: {prob.status}",
                )
        else:
            raise ValueError(f"Unknown objective: {objective}")

        # Calculate metrics
        port_return, port_vol, sharpe = self.calculate_portfolio_metrics(
            weights, expected_returns, covariance_matrix, risk_free_rate
        )

        return OptimizationResult(
            weights=weights,
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=sharpe,
            optimization_method=self.name,
            success=True,
            message="Optimization successful",
        )

    def _optimize_scipy(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        objective: str,
        constraints: Dict[str, Any],
        bounds: Tuple[float, float],
        risk_free_rate: float,
        target_return: float,
        target_volatility: float,
    ) -> OptimizationResult:
        """Optimize using scipy.optimize."""
        n_assets = len(expected_returns)
        mu = expected_returns.values
        Sigma = covariance_matrix.values

        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets

        # Bounds
        bounds_list = (
            [(bounds[0], bounds[1]) for _ in range(n_assets)] if bounds else None
        )

        # Constraints
        constraints_list = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        ]  # Weights sum to 1

        if objective == "target_return" and target_return is not None:
            constraints_list.append(
                {"type": "eq", "fun": lambda x: np.dot(x, mu) - target_return}
            )
        elif objective == "target_volatility" and target_volatility is not None:
            constraints_list.append(
                {
                    "type": "eq",
                    "fun": lambda x: np.sqrt(np.dot(x, np.dot(Sigma, x)))
                    - target_volatility,
                }
            )

        # Define objective function
        if objective == "max_sharpe":

            def objective_func(x):
                port_return = np.dot(x, mu)
                port_vol = np.sqrt(np.dot(x, np.dot(Sigma, x)))
                return (
                    -(port_return - risk_free_rate) / port_vol
                    if port_vol > 0
                    else -np.inf
                )

        elif objective == "min_volatility":

            def objective_func(x):
                return np.sqrt(np.dot(x, np.dot(Sigma, x)))

        elif objective == "max_return":

            def objective_func(x):
                return -np.dot(x, mu)

        elif objective == "target_return":

            def objective_func(x):
                return np.sqrt(np.dot(x, np.dot(Sigma, x)))

        elif objective == "target_volatility":

            def objective_func(x):
                return -np.dot(x, mu)

        else:
            raise ValueError(f"Unknown objective: {objective}")

        # Optimize
        result = minimize(
            objective_func,
            x0,
            method="SLSQP",
            bounds=bounds_list,
            constraints=constraints_list,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        if result.success:
            weights = pd.Series(result.x, index=expected_returns.index)
            port_return, port_vol, sharpe = self.calculate_portfolio_metrics(
                weights, expected_returns, covariance_matrix, risk_free_rate
            )

            return OptimizationResult(
                weights=weights,
                expected_return=port_return,
                volatility=port_vol,
                sharpe_ratio=sharpe,
                optimization_method=self.name,
                success=True,
                message="Optimization successful",
            )
        else:
            return OptimizationResult(
                weights=pd.Series(np.zeros(n_assets), index=expected_returns.index),
                expected_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                optimization_method=self.name,
                success=False,
                message=f"Optimization failed: {result.message}",
            )


class BlackLittermanOptimizer(BaseOptimizer):
    """Black-Litterman model implementation with Bayesian updating."""

    def __init__(self, tau: float = 0.05, risk_aversion: float = 3.0):
        super().__init__("BlackLitterman")
        self.tau = tau  # Scaling factor for uncertainty of prior
        self.risk_aversion = risk_aversion  # Risk aversion parameter

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        market_caps: pd.Series = None,
        views: Dict[str, Any] = None,
        view_confidences: pd.DataFrame = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Optimize using Black-Litterman model.

        Args:
            expected_returns: Market equilibrium returns (can be None, will be calculated)
            covariance_matrix: Covariance matrix of asset returns
            market_caps: Market capitalizations for equilibrium weights
            views: Dictionary of views {'assets': [list], 'returns': [list]}
            view_confidences: Omega matrix (uncertainty of views)

        Returns:
            OptimizationResult with Black-Litterman optimized weights
        """
        self.validate_inputs(expected_returns, covariance_matrix)

        n_assets = len(expected_returns)
        Sigma = covariance_matrix.values

        # Step 1: Calculate equilibrium weights (if market caps provided)
        if market_caps is not None:
            w_eq = market_caps / market_caps.sum()
        else:
            # Use equal weights as default
            w_eq = pd.Series(np.ones(n_assets) / n_assets, index=expected_returns.index)

        # Step 2: Calculate implied equilibrium returns
        Pi = self.risk_aversion * np.dot(Sigma, w_eq.values)

        # Step 3: Process views
        if views is not None and len(views.get("assets", [])) > 0:
            # Create picking matrix P
            view_assets = views["assets"]
            view_returns = np.array(views["returns"])

            P = np.zeros((len(view_assets), n_assets))
            for i, assets in enumerate(view_assets):
                if isinstance(assets, str):
                    # Single asset view
                    asset_idx = expected_returns.index.get_loc(assets)
                    P[i, asset_idx] = 1.0
                elif isinstance(assets, list):
                    # Relative view (e.g., A outperforms B)
                    if len(assets) == 2:
                        asset1_idx = expected_returns.index.get_loc(assets[0])
                        asset2_idx = expected_returns.index.get_loc(assets[1])
                        P[i, asset1_idx] = 1.0
                        P[i, asset2_idx] = -1.0
                    else:
                        # Equal weighted portfolio view
                        for asset in assets:
                            asset_idx = expected_returns.index.get_loc(asset)
                            P[i, asset_idx] = 1.0 / len(assets)

            Q = view_returns

            # Step 4: Create uncertainty matrix for views (Omega)
            if view_confidences is not None:
                Omega = view_confidences.values
            else:
                # Default: diagonal matrix with view uncertainties
                # Proportional to variance of view portfolio
                Omega = np.zeros((len(view_assets), len(view_assets)))
                for i in range(len(view_assets)):
                    view_var = np.dot(P[i], np.dot(Sigma, P[i]))
                    Omega[i, i] = self.tau * view_var

            # Step 5: Black-Litterman formula
            tau_Sigma = self.tau * Sigma

            # New expected returns (mu_bl)
            M1 = np.linalg.inv(tau_Sigma)
            M2 = np.dot(P.T, np.dot(np.linalg.inv(Omega), P))
            M3 = np.dot(np.linalg.inv(tau_Sigma), Pi)
            M4 = np.dot(P.T, np.dot(np.linalg.inv(Omega), Q))

            mu_bl = np.dot(np.linalg.inv(M1 + M2), M3 + M4)

            # New covariance matrix (Sigma_bl)
            Sigma_bl = np.linalg.inv(M1 + M2)

        else:
            # No views, use equilibrium returns
            mu_bl = Pi
            Sigma_bl = Sigma

        # Step 6: Optimize portfolio with Black-Litterman inputs
        bl_returns = pd.Series(mu_bl, index=expected_returns.index)
        bl_cov = pd.DataFrame(
            Sigma_bl, index=covariance_matrix.index, columns=covariance_matrix.columns
        )

        # Use standard optimizer with Black-Litterman inputs
        standard_optimizer = PortfolioOptimizer()
        result = standard_optimizer.optimize(
            bl_returns,
            bl_cov,
            objective=kwargs.get("objective", "max_sharpe"),
            **kwargs,
        )

        # Update result metadata
        result.optimization_method = self.name
        result.metadata = {
            "tau": self.tau,
            "risk_aversion": self.risk_aversion,
            "equilibrium_weights": w_eq.to_dict(),
            "implied_returns": Pi.tolist(),
            "bl_returns": mu_bl.tolist(),
            "views_used": views is not None and len(views.get("assets", [])) > 0,
        }

        return result


class RiskParityOptimizer(BaseOptimizer):
    """Risk Parity optimizer with multiple risk measures."""

    def __init__(self, risk_measure: str = "volatility"):
        super().__init__("RiskParity")
        self.risk_measure = risk_measure  # 'volatility', 'var', 'cvar'

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_budgets: pd.Series = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Optimize using Risk Parity approach.

        Args:
            expected_returns: Expected returns (used for metrics calculation)
            covariance_matrix: Covariance matrix of asset returns
            risk_budgets: Target risk budgets for each asset (default: equal)

        Returns:
            OptimizationResult with risk parity optimized weights
        """
        self.validate_inputs(expected_returns, covariance_matrix)

        n_assets = len(expected_returns)
        Sigma = covariance_matrix.values

        # Default equal risk budgets
        if risk_budgets is None:
            risk_budgets = pd.Series(
                np.ones(n_assets) / n_assets, index=expected_returns.index
            )

        # Risk parity optimization
        if self.risk_measure == "volatility":
            weights = self._optimize_volatility_parity(Sigma, risk_budgets.values)
        else:
            raise NotImplementedError(
                f"Risk measure {self.risk_measure} not implemented yet"
            )

        weights_series = pd.Series(weights, index=expected_returns.index)

        # Calculate metrics
        port_return, port_vol, sharpe = self.calculate_portfolio_metrics(
            weights_series, expected_returns, covariance_matrix
        )

        return OptimizationResult(
            weights=weights_series,
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=sharpe,
            optimization_method=self.name,
            success=True,
            message="Risk parity optimization successful",
            metadata={
                "risk_measure": self.risk_measure,
                "risk_budgets": risk_budgets.to_dict(),
                "risk_contributions": self._calculate_risk_contributions(
                    weights, Sigma
                ).tolist(),
            },
        )

    def _optimize_volatility_parity(
        self, Sigma: np.ndarray, risk_budgets: np.ndarray
    ) -> np.ndarray:
        """Optimize for volatility risk parity."""
        n_assets = Sigma.shape[0]

        def risk_budget_objective(weights):
            """Objective function for risk budgeting."""
            weights = np.maximum(weights, 1e-8)  # Avoid division by zero
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(Sigma, weights)))

            # Risk contributions
            marginal_contrib = np.dot(Sigma, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            contrib_pct = contrib / np.sum(contrib)

            # Sum of squared deviations from target risk budgets
            return np.sum((contrib_pct - risk_budgets) ** 2)

        # Constraints and bounds
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        bounds = [(0.001, 1.0) for _ in range(n_assets)]  # Small positive lower bound

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            risk_budget_objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        return result.x if result.success else x0

    def _calculate_risk_contributions(
        self, weights: np.ndarray, Sigma: np.ndarray
    ) -> np.ndarray:
        """Calculate risk contributions for each asset."""
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(Sigma, weights)))
        marginal_contrib = np.dot(Sigma, weights) / portfolio_vol
        contrib = weights * marginal_contrib
        return contrib / np.sum(contrib)


class HierarchicalRiskParityOptimizer(BaseOptimizer):
    """Hierarchical Risk Parity using machine learning clustering."""

    def __init__(
        self, linkage_method: str = "ward", distance_metric: str = "correlation"
    ):
        super().__init__("HierarchicalRiskParity")
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        returns_data: pd.DataFrame = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Optimize using Hierarchical Risk Parity.

        Args:
            expected_returns: Expected returns (used for metrics calculation)
            covariance_matrix: Covariance matrix of asset returns
            returns_data: Historical returns data for clustering

        Returns:
            OptimizationResult with HRP optimized weights
        """
        self.validate_inputs(expected_returns, covariance_matrix)

        # Step 1: Calculate distance matrix
        if self.distance_metric == "correlation":
            corr_matrix = covariance_matrix.div(
                np.outer(
                    np.sqrt(np.diag(covariance_matrix)),
                    np.sqrt(np.diag(covariance_matrix)),
                )
            )
            distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        else:
            # Use covariance as distance
            distance_matrix = covariance_matrix

        # Step 2: Hierarchical clustering
        condensed_distances = squareform(distance_matrix.values, checks=False)
        linkage_matrix = linkage(condensed_distances, method=self.linkage_method)

        # Step 3: Quasi-diagonalization
        sorted_indices = self._get_quasi_diag(linkage_matrix)

        # Step 4: Recursive bisection
        weights = self._get_rec_bipart(
            covariance_matrix.iloc[sorted_indices, sorted_indices]
        )

        # Map back to original order
        weights_full = np.zeros(len(expected_returns))
        for i, idx in enumerate(sorted_indices):
            weights_full[idx] = weights[i]

        weights_series = pd.Series(weights_full, index=expected_returns.index)

        # Calculate metrics
        port_return, port_vol, sharpe = self.calculate_portfolio_metrics(
            weights_series, expected_returns, covariance_matrix
        )

        return OptimizationResult(
            weights=weights_series,
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=sharpe,
            optimization_method=self.name,
            success=True,
            message="HRP optimization successful",
            metadata={
                "linkage_method": self.linkage_method,
                "distance_metric": self.distance_metric,
                "sorted_indices": sorted_indices.tolist(),
            },
        )

    def _get_quasi_diag(self, linkage_matrix: np.ndarray) -> np.ndarray:
        """Get quasi-diagonal order from linkage matrix."""
        link = linkage_matrix.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]

        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = sort_ix.append(df0).sort_index()
            sort_ix.index = range(sort_ix.shape[0])

        return sort_ix.astype(int).values

    def _get_rec_bipart(self, cov: pd.DataFrame) -> np.ndarray:
        """Recursive bisection for HRP weights."""
        w = pd.Series(1.0, index=cov.index)
        c_items = [cov.index.tolist()]

        while len(c_items) > 0:
            c_items = [
                i[j:k]
                for i in c_items
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]

            for i in range(0, len(c_items), 2):
                c_items0 = c_items[i]
                c_items1 = c_items[i + 1]

                c_var0 = self._get_cluster_var(cov, c_items0)
                c_var1 = self._get_cluster_var(cov, c_items1)

                alpha = 1 - c_var0 / (c_var0 + c_var1)

                w[c_items0] *= alpha
                w[c_items1] *= 1 - alpha

        return w.values

    def _get_cluster_var(self, cov: pd.DataFrame, c_items: List[str]) -> float:
        """Calculate cluster variance."""
        cov_slice = cov.loc[c_items, c_items]
        w = 1.0 / np.diag(cov_slice)
        w /= w.sum()
        return np.dot(w, np.dot(cov_slice.values, w))


class FactorModelOptimizer(BaseOptimizer):
    """Factor model optimization with Fama-French and custom factors."""

    def __init__(self, factor_model: str = "fama_french_3"):
        super().__init__("FactorModel")
        self.factor_model = factor_model

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        factor_exposures: pd.DataFrame = None,
        factor_covariance: pd.DataFrame = None,
        specific_risk: pd.Series = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Optimize using factor model approach.

        Args:
            expected_returns: Expected returns for assets
            covariance_matrix: Covariance matrix (can be None if factor model provided)
            factor_exposures: Factor exposures (beta matrix)
            factor_covariance: Factor covariance matrix
            specific_risk: Specific (idiosyncratic) risk for each asset

        Returns:
            OptimizationResult with factor model optimized weights
        """
        if (
            factor_exposures is not None
            and factor_covariance is not None
            and specific_risk is not None
        ):
            # Reconstruct covariance matrix from factor model
            # Cov = B * F * B' + D
            B = factor_exposures.values
            F = factor_covariance.values
            D = np.diag(specific_risk.values)

            reconstructed_cov = np.dot(B, np.dot(F, B.T)) + D
            covariance_matrix = pd.DataFrame(
                reconstructed_cov,
                index=expected_returns.index,
                columns=expected_returns.index,
            )

        # Use standard optimizer with factor-based covariance
        standard_optimizer = PortfolioOptimizer()
        result = standard_optimizer.optimize(
            expected_returns, covariance_matrix, **kwargs
        )

        # Update result metadata
        result.optimization_method = self.name
        result.metadata = {
            "factor_model": self.factor_model,
            "factor_based_covariance": factor_exposures is not None,
        }

        return result


class TransactionCostAwareOptimizer(BaseOptimizer):
    """Portfolio optimizer that incorporates transaction costs."""

    def __init__(self, base_optimizer: BaseOptimizer = None):
        super().__init__("TransactionCostAware")
        self.base_optimizer = base_optimizer or PortfolioOptimizer()

        if TRANSACTION_COSTS_AVAILABLE:
            self.cost_optimizer = TransactionCostOptimizer()
        else:
            raise ImportError("Transaction cost optimization not available")

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        current_weights: pd.Series = None,
        portfolio_value: float = 1000000.0,
        market_data: Dict[str, Dict[str, Any]] = None,
        transaction_cost_penalty: float = 1.0,
        **kwargs,
    ) -> OptimizationResult:
        """
        Optimize portfolio considering transaction costs.

        Args:
            expected_returns: Expected returns for assets
            covariance_matrix: Covariance matrix of asset returns
            current_weights: Current portfolio weights (if rebalancing)
            portfolio_value: Total portfolio value
            market_data: Market data for each asset (volume, volatility, etc.)
            transaction_cost_penalty: Penalty factor for transaction costs
            **kwargs: Additional arguments for base optimizer

        Returns:
            OptimizationResult with transaction-cost-aware weights
        """
        self.validate_inputs(expected_returns, covariance_matrix)

        # If no current weights, assume starting from cash
        if current_weights is None:
            current_weights = pd.Series(0.0, index=expected_returns.index)

        # Default market data if not provided
        if market_data is None:
            market_data = {}
            for asset in expected_returns.index:
                market_data[asset] = {
                    "volume": 1000000,
                    "volatility": 0.25,
                    "price": 100.0,
                }

        # Extract risk_aversion from kwargs before passing to base optimizer
        kwargs.pop("risk_aversion", 1.0)

        # First, get the unconstrained optimal weights
        base_result = self.base_optimizer.optimize(
            expected_returns, covariance_matrix, **kwargs
        )

        if not base_result.success:
            return base_result

        # For simplicity, use a heuristic approach that balances base optimal weights
        # with transaction costs by adjusting towards current weights

        # Calculate turnover for base optimal weights
        base_turnover = np.sum(np.abs(base_result.weights - current_weights))

        # If turnover is low, use base optimal weights
        if base_turnover < 0.1:  # Less than 10% turnover
            optimal_weights = base_result.weights
        else:
            # Blend between current and optimal weights based on transaction cost penalty
            blend_factor = min(0.9, transaction_cost_penalty / 10.0)
            optimal_weights = (
                1 - blend_factor
            ) * base_result.weights + blend_factor * current_weights

            # Ensure weights sum to 1
            optimal_weights = optimal_weights / optimal_weights.sum()

        # Calculate final metrics
        port_return, port_vol, sharpe = self.calculate_portfolio_metrics(
            optimal_weights, expected_returns, covariance_matrix
        )

        # Calculate final transaction costs
        final_cost_analysis = self.cost_optimizer.calculate_rebalancing_costs(
            current_weights, optimal_weights, portfolio_value, market_data
        )

        return OptimizationResult(
            weights=optimal_weights,
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=sharpe,
            optimization_method=self.name,
            success=True,
            message="Transaction-cost-aware optimization successful",
            metadata={
                "base_optimizer": self.base_optimizer.name,
                "transaction_costs": final_cost_analysis,
                "cost_penalty": transaction_cost_penalty,
                "turnover": np.sum(np.abs(optimal_weights - current_weights)),
                "base_turnover": base_turnover,
            },
        )


class RebalancingOptimizer:
    """Optimizer for portfolio rebalancing with cost considerations."""

    def __init__(self):
        if TRANSACTION_COSTS_AVAILABLE:
            self.cost_optimizer = TransactionCostOptimizer()
        else:
            raise ImportError("Transaction cost optimization not available")

    def optimize_rebalancing_frequency(
        self,
        returns_data: pd.DataFrame,
        target_weights: pd.Series,
        rebalancing_frequencies: List[int] = None,
        market_data: Dict[str, Dict[str, Any]] = None,
        portfolio_value: float = 1000000.0,
    ) -> Dict[str, Any]:
        """
        Optimize rebalancing frequency considering transaction costs.

        Args:
            returns_data: Historical returns data
            target_weights: Target portfolio weights
            rebalancing_frequencies: List of frequencies to test (in days)
            market_data: Market data for transaction cost calculation
            portfolio_value: Portfolio value

        Returns:
            Dictionary with optimal rebalancing frequency analysis
        """
        if rebalancing_frequencies is None:
            rebalancing_frequencies = [1, 5, 10, 20, 30, 60, 90, 180, 365]

        if market_data is None:
            market_data = {}
            for asset in target_weights.index:
                market_data[asset] = {
                    "volume": 1000000,
                    "volatility": returns_data[asset].std() * np.sqrt(252),
                    "price": 100.0,
                }

        results = {}

        for freq in rebalancing_frequencies:
            # Simulate rebalancing at this frequency
            total_costs = 0.0
            tracking_errors = []

            # Get rebalancing dates
            rebalancing_dates = returns_data.index[::freq]

            current_weights = target_weights.copy()

            for i, date in enumerate(rebalancing_dates[:-1]):
                # Calculate drift in weights due to returns
                if i > 0:
                    period_returns = returns_data.loc[
                        rebalancing_dates[i - 1] : date
                    ].iloc[1:]
                    if len(period_returns) > 0:
                        # Calculate how weights drifted
                        cumulative_returns = (1 + period_returns).prod() - 1
                        new_values = current_weights * (1 + cumulative_returns)
                        current_weights = new_values / new_values.sum()

                # Calculate rebalancing costs
                cost_analysis = self.cost_optimizer.calculate_rebalancing_costs(
                    current_weights, target_weights, portfolio_value, market_data
                )

                total_costs += cost_analysis["total_cost"]

                # Calculate tracking error
                weight_diff = current_weights - target_weights
                tracking_error = np.sqrt(np.dot(weight_diff, weight_diff))
                tracking_errors.append(tracking_error)

                # Reset to target weights after rebalancing
                current_weights = target_weights.copy()

            # Calculate metrics for this frequency
            avg_tracking_error = np.mean(tracking_errors) if tracking_errors else 0.0
            annual_cost = total_costs * (365 / freq) if freq > 0 else total_costs
            cost_percentage = annual_cost / portfolio_value * 100

            results[freq] = {
                "frequency_days": freq,
                "total_costs": total_costs,
                "annual_cost": annual_cost,
                "cost_percentage": cost_percentage,
                "avg_tracking_error": avg_tracking_error,
                "num_rebalances": len(rebalancing_dates) - 1,
                "cost_per_rebalance": total_costs / max(1, len(rebalancing_dates) - 1),
            }

        # Find optimal frequency (minimize cost + tracking error)
        optimal_freq = min(
            results.keys(),
            key=lambda f: results[f]["cost_percentage"]
            + results[f]["avg_tracking_error"] * 100,
        )

        return {
            "optimal_frequency": optimal_freq,
            "results_by_frequency": results,
            "recommendation": f"Optimal rebalancing frequency: {optimal_freq} days",
        }


# Example usage functions
def create_transaction_cost_example():
    """Create example for transaction cost optimization."""
    # Sample data
    assets = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    expected_returns = pd.Series([0.12, 0.15, 0.10, 0.18], index=assets)

    # Sample covariance matrix
    corr_matrix = np.array(
        [
            [1.00, 0.30, 0.25, 0.20],
            [0.30, 1.00, 0.35, 0.25],
            [0.25, 0.35, 1.00, 0.30],
            [0.20, 0.25, 0.30, 1.00],
        ]
    )

    volatilities = np.array([0.25, 0.30, 0.22, 0.40])
    cov_matrix = np.outer(volatilities, volatilities) * corr_matrix
    cov_df = pd.DataFrame(cov_matrix, index=assets, columns=assets)

    # Current portfolio weights (need to rebalance)
    current_weights = pd.Series([0.40, 0.20, 0.30, 0.10], index=assets)

    # Market data
    market_data = {
        "AAPL": {"volume": 50000000, "volatility": 0.25, "price": 150.0},
        "GOOGL": {"volume": 25000000, "volatility": 0.30, "price": 2800.0},
        "MSFT": {"volume": 40000000, "volatility": 0.22, "price": 300.0},
        "TSLA": {"volume": 30000000, "volatility": 0.40, "price": 800.0},
    }

    return expected_returns, cov_df, current_weights, market_data


if __name__ == "__main__":
    # Example usage
    if TRANSACTION_COSTS_AVAILABLE:
        expected_returns, cov_matrix, current_weights, market_data = (
            create_transaction_cost_example()
        )

        # Test transaction-cost-aware optimization
        optimizer = TransactionCostAwareOptimizer()
        result = optimizer.optimize(
            expected_returns,
            cov_matrix,
            current_weights=current_weights,
            market_data=market_data,
            transaction_cost_penalty=2.0,
            objective="max_sharpe",
        )

        print("Transaction-Cost-Aware Optimization:")
        print(f"Success: {result.success}")
        print(f"Expected Return: {result.expected_return:.4f}")
        print(f"Volatility: {result.volatility:.4f}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.4f}")
        print(
            f"Transaction Costs: ${result.metadata['transaction_costs']['total_cost']:,.2f}"
        )
        print(f"Turnover: {result.metadata['turnover']:.4f}")
