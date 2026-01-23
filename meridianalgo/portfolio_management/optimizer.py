"""
Portfolio optimization algorithms for MeridianAlgo.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class PortfolioOptimizer:
    """Advanced portfolio optimization using modern portfolio theory."""

    def __init__(self, returns: pd.DataFrame):
        """
        Initialize portfolio optimizer.

        Args:
            returns: DataFrame of asset returns
        """
        self.returns = returns
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.n_assets = len(returns.columns)

    def optimize_portfolio(
        self,
        method: str = "sharpe",
        target_return: float = None,
        risk_free_rate: float = 0.02,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Optimize portfolio using specified method.

        Args:
            method: Optimization method ('sharpe', 'min_vol', 'max_return')
            target_return: Target return for optimization
            risk_free_rate: Risk-free rate for Sharpe ratio calculation

        Returns:
            Dictionary of optimal weights
        """
        try:
            if method == "sharpe":
                return self._maximize_sharpe_ratio(risk_free_rate)
            elif method == "min_vol" or method == "min_volatility":
                return self._minimize_volatility()
            elif method == "max_return":
                return self._maximize_return()
            elif method == "target_return" and target_return is not None:
                return self._target_return_optimization(target_return)
            else:
                # Default to equal weights
                return self._equal_weights()
        except Exception:
            # Fallback to equal weights
            return self._equal_weights()

    def _equal_weights(self) -> Dict[str, float]:
        """Return equal weights for all assets."""
        weight = 1.0 / self.n_assets
        return {asset: weight for asset in self.returns.columns}

    def _maximize_sharpe_ratio(self, risk_free_rate: float) -> Dict[str, float]:
        """Maximize Sharpe ratio."""

        def negative_sharpe(weights):
            portfolio_return = np.sum(self.mean_returns * weights) * 252
            portfolio_vol = np.sqrt(
                np.dot(weights.T, np.dot(self.cov_matrix * 252, weights))
            )
            return -(portfolio_return - risk_free_rate) / portfolio_vol

        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1 / self.n_assets] * self.n_assets)

        try:
            result = minimize(
                negative_sharpe,
                initial_guess,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            if result.success:
                return {
                    asset: weight
                    for asset, weight in zip(self.returns.columns, result.x)
                }
        except Exception:
            pass

        return self._equal_weights()

    def _minimize_volatility(self) -> Dict[str, float]:
        """Minimize portfolio volatility."""

        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1 / self.n_assets] * self.n_assets)

        try:
            result = minimize(
                portfolio_volatility,
                initial_guess,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            if result.success:
                return {
                    asset: weight
                    for asset, weight in zip(self.returns.columns, result.x)
                }
        except Exception:
            pass

        return self._equal_weights()

    def _maximize_return(self) -> Dict[str, float]:
        """Maximize portfolio return."""

        def negative_return(weights):
            return -np.sum(self.mean_returns * weights)

        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1 / self.n_assets] * self.n_assets)

        try:
            result = minimize(
                negative_return,
                initial_guess,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            if result.success:
                return {
                    asset: weight
                    for asset, weight in zip(self.returns.columns, result.x)
                }
        except Exception:
            pass

        return self._equal_weights()

    def _target_return_optimization(self, target_return: float) -> Dict[str, float]:
        """Optimize for target return with minimum risk."""

        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
            {
                "type": "eq",
                "fun": lambda x: np.sum(self.mean_returns * x) * 252 - target_return,
            },
        ]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1 / self.n_assets] * self.n_assets)

        try:
            result = minimize(
                portfolio_volatility,
                initial_guess,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            if result.success:
                return {
                    asset: weight
                    for asset, weight in zip(self.returns.columns, result.x)
                }
        except Exception:
            pass

        return self._equal_weights()


class EfficientFrontier:
    """Calculate efficient frontier for portfolio optimization."""

    def __init__(self, returns: pd.DataFrame):
        """Initialize with returns data."""
        self.returns = returns
        self.optimizer = PortfolioOptimizer(returns)

    def calculate_frontier(self, target_returns: List[float]) -> pd.DataFrame:
        """Calculate efficient frontier points."""
        frontier_data = []

        for target_return in target_returns:
            try:
                weights = self.optimizer.optimize_portfolio(
                    method="target_return", target_return=target_return
                )

                # Calculate portfolio metrics
                weights_array = np.array(
                    [weights[asset] for asset in self.returns.columns]
                )
                portfolio_return = (
                    np.sum(self.optimizer.mean_returns * weights_array) * 252
                )
                portfolio_vol = np.sqrt(
                    np.dot(
                        weights_array.T,
                        np.dot(self.optimizer.cov_matrix * 252, weights_array),
                    )
                )

                frontier_data.append(
                    {
                        "return": portfolio_return,
                        "volatility": portfolio_vol,
                        "sharpe": (
                            portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
                        ),
                    }
                )
            except Exception:
                # Skip failed optimizations
                continue

        return pd.DataFrame(frontier_data)


class BlackLitterman:
    """Black-Litterman portfolio optimization model."""

    def __init__(self, returns: pd.DataFrame, market_caps: Dict[str, float]):
        """Initialize Black-Litterman model."""
        self.returns = returns
        self.market_caps = market_caps
        self.optimizer = PortfolioOptimizer(returns)

    def optimize_with_views(
        self, views: Dict[str, float], confidence: float = 0.5
    ) -> Dict[str, float]:
        """
        Optimize portfolio with investor views.

        Args:
            views: Dictionary of asset views (expected returns)
            confidence: Confidence in views (0-1)

        Returns:
            Optimal portfolio weights
        """
        try:
            # Simplified Black-Litterman implementation
            # In practice, this would involve more complex calculations

            # Start with market cap weights
            total_market_cap = sum(self.market_caps.values())
            market_weights = {
                asset: cap / total_market_cap for asset, cap in self.market_caps.items()
            }

            # Adjust weights based on views
            adjusted_weights = market_weights.copy()

            for asset, view in views.items():
                if asset in adjusted_weights:
                    # Simple adjustment based on view and confidence
                    adjustment = confidence * (
                        view - self.optimizer.mean_returns[asset]
                    )
                    adjusted_weights[asset] *= 1 + adjustment

            # Normalize weights
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {
                    asset: weight / total_weight
                    for asset, weight in adjusted_weights.items()
                }

            return adjusted_weights
        except Exception:
            return self.optimizer._equal_weights()


class RiskParity:
    """Risk parity portfolio optimization."""

    def __init__(self, returns: pd.DataFrame):
        """Initialize risk parity optimizer."""
        self.returns = returns
        self.cov_matrix = returns.cov()

    def optimize(self) -> Dict[str, float]:
        """Optimize for risk parity (equal risk contribution)."""
        try:
            # Simplified risk parity - inverse volatility weighting
            volatilities = np.sqrt(np.diag(self.cov_matrix))
            inv_vol_weights = 1 / volatilities
            normalized_weights = inv_vol_weights / np.sum(inv_vol_weights)

            return {
                asset: weight
                for asset, weight in zip(self.returns.columns, normalized_weights)
            }
        except Exception:
            # Fallback to equal weights
            n_assets = len(self.returns.columns)
            return {asset: 1 / n_assets for asset in self.returns.columns}
