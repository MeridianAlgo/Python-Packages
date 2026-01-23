"""
Portfolio optimization module.

This module provides tools for portfolio optimization using modern portfolio theory.
"""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from ...config import get_config


class PortfolioOptimizer:
    """Portfolio optimization using modern portfolio theory."""

    def __init__(self, returns: pd.DataFrame):
        """Initialize the portfolio optimizer with historical returns.

        Args:
            returns: DataFrame containing historical returns (tickers as columns)
        """
        if returns is None or returns.empty:
            raise ValueError("Returns data cannot be empty")
        self.returns = returns
        self.cov_matrix = self._calculate_covariance_matrix()

    def _calculate_covariance_matrix(self) -> pd.DataFrame:
        """Calculate the covariance matrix of returns."""
        return self.returns.cov()

    def calculate_efficient_frontier(
        self, risk_free_rate: Optional[float] = None, num_portfolios: int = 1000
    ) -> Dict[str, np.ndarray]:
        """Calculate the efficient frontier using Monte Carlo simulation.

        Args:
            risk_free_rate: Annual risk-free rate. If None, uses the default from config.
            num_portfolios: Number of random portfolios to generate (default: 1000)

        Returns:
            Dictionary containing portfolio weights, returns, and volatilities
        """
        if risk_free_rate is None:
            risk_free_rate = get_config("risk_free_rate", 0.0)

        results = np.zeros((3, num_portfolios))
        weights_record = []

        for i in range(num_portfolios):
            weights = np.random.random(len(self.returns.columns))
            weights /= np.sum(weights)
            weights_record.append(weights)

            # Calculate portfolio return and volatility
            portfolio_return = np.sum(self.returns.mean() * weights) * 252
            portfolio_volatility = np.sqrt(
                np.dot(weights.T, np.dot(self.cov_matrix * 252, weights))
            )

            # Store results
            results[0, i] = portfolio_volatility
            results[1, i] = portfolio_return
            results[2, i] = (
                portfolio_return - risk_free_rate
            ) / portfolio_volatility  # Sharpe ratio

        return {
            "volatility": results[0],
            "returns": results[1],
            "sharpe": results[2],
            "weights": np.array(weights_record),
        }

    def optimize_portfolio(
        self,
        target_return: Optional[float] = None,
        risk_free_rate: Optional[float] = None,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Optimize portfolio for a target return.

        Args:
            target_return: Target annualized return. If None, maximizes Sharpe ratio.
            risk_free_rate: Risk-free rate for Sharpe ratio calculation.

        Returns:
            Dictionary containing optimized weights and metrics
        """
        if risk_free_rate is None:
            risk_free_rate = get_config("risk_free_rate", 0.0)

        len(self.returns.columns)

        if target_return is None:
            # Find portfolio with maximum Sharpe ratio
            frontier = self.calculate_efficient_frontier(risk_free_rate)
            max_sharpe_idx = np.argmax(frontier["sharpe"])

            return {
                "weights": frontier["weights"][max_sharpe_idx],
                "return": frontier["returns"][max_sharpe_idx],
                "volatility": frontier["volatility"][max_sharpe_idx],
                "sharpe": frontier["sharpe"][max_sharpe_idx],
            }
        else:
            # Find portfolio with target return and minimum volatility
            frontier = self.calculate_efficient_frontier(risk_free_rate)
            return_idx = np.argmin(np.abs(frontier["returns"] - target_return))

            return {
                "weights": frontier["weights"][return_idx],
                "return": frontier["returns"][return_idx],
                "volatility": frontier["volatility"][return_idx],
                "sharpe": frontier["sharpe"][return_idx],
            }
