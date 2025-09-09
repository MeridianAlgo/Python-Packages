"""
Portfolio Optimization Module

This module contains various portfolio optimization strategies including
Modern Portfolio Theory, Black-Litterman, and Risk Parity.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy.optimize import minimize
import warnings

class PortfolioOptimizer:
    """Enhanced Portfolio Optimizer with multiple optimization strategies."""
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize portfolio optimizer.
        
        Args:
            returns: DataFrame of asset returns
            risk_free_rate: Risk-free rate (default: 0.02)
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.n_assets = len(returns.columns)
    
    def calculate_efficient_frontier(self, num_portfolios: int = 1000) -> Dict[str, np.ndarray]:
        """
        Calculate efficient frontier using Monte Carlo simulation.
        
        Args:
            num_portfolios: Number of portfolios to generate
            
        Returns:
            Dictionary with portfolio statistics
        """
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        for i in range(num_portfolios):
            weights = np.random.random(self.n_assets)
            weights /= np.sum(weights)
            weights_record.append(weights)
            
            portfolio_return = np.sum(self.mean_returns * weights) * 252
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
            
            results[0, i] = portfolio_std
            results[1, i] = portfolio_return
            results[2, i] = (portfolio_return - self.risk_free_rate) / portfolio_std
        
        return {
            'volatility': results[0],
            'returns': results[1],
            'sharpe': results[2],
            'weights': np.array(weights_record)
        }
    
    def optimize_portfolio(self, objective: str = 'sharpe', 
                          constraints: Optional[Dict] = None) -> Dict:
        """
        Optimize portfolio using specified objective.
        
        Args:
            objective: Optimization objective ('sharpe', 'min_vol', 'max_return')
            constraints: Additional constraints
            
        Returns:
            Dictionary with optimal portfolio
        """
        if constraints is None:
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Initial guess
        x0 = np.array([1/self.n_assets] * self.n_assets)
        
        # Bounds for weights (0 to 1)
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        if objective == 'sharpe':
            def neg_sharpe(weights):
                portfolio_return = np.sum(self.mean_returns * weights) * 252
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
                return -(portfolio_return - self.risk_free_rate) / portfolio_std
            
            result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        elif objective == 'min_vol':
            def portfolio_volatility(weights):
                return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
            
            result = minimize(portfolio_volatility, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        elif objective == 'max_return':
            def neg_return(weights):
                return -(np.sum(self.mean_returns * weights) * 252)
            
            result = minimize(neg_return, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        else:
            raise ValueError("Objective must be 'sharpe', 'min_vol', or 'max_return'")
        
        optimal_weights = result.x
        portfolio_return = np.sum(self.mean_returns * optimal_weights) * 252
        portfolio_std = np.sqrt(np.dot(optimal_weights.T, np.dot(self.cov_matrix * 252, optimal_weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
        
        return {
            'weights': optimal_weights,
            'return': portfolio_return,
            'volatility': portfolio_std,
            'sharpe_ratio': sharpe_ratio
        }

class EfficientFrontier:
    """Efficient Frontier Calculator."""
    
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.n_assets = len(returns.columns)
    
    def calculate_frontier(self, target_returns: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate efficient frontier for given target returns.
        
        Args:
            target_returns: Array of target returns
            
        Returns:
            Dictionary with frontier data
        """
        frontier_weights = []
        frontier_volatilities = []
        
        for target_return in target_returns:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(self.mean_returns * x) * 252 - target_return}
            ]
            
            bounds = tuple((0, 1) for _ in range(self.n_assets))
            x0 = np.array([1/self.n_assets] * self.n_assets)
            
            def portfolio_volatility(weights):
                return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
            
            result = minimize(portfolio_volatility, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                frontier_weights.append(result.x)
                frontier_volatilities.append(result.fun)
            else:
                frontier_weights.append(np.full(self.n_assets, np.nan))
                frontier_volatilities.append(np.nan)
        
        return {
            'weights': np.array(frontier_weights),
            'volatilities': np.array(frontier_volatilities),
            'returns': target_returns
        }

class BlackLitterman:
    """Black-Litterman Model Implementation."""
    
    def __init__(self, returns: pd.DataFrame, market_caps: Optional[pd.Series] = None):
        self.returns = returns
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.n_assets = len(returns.columns)
        
        if market_caps is not None:
            self.market_caps = market_caps
        else:
            # Equal weight market portfolio
            self.market_caps = pd.Series(1/self.n_assets, index=returns.columns)
    
    def calculate_implied_returns(self, risk_aversion: float = 3) -> pd.Series:
        """
        Calculate implied equilibrium returns.
        
        Args:
            risk_aversion: Risk aversion parameter
            
        Returns:
            Implied equilibrium returns
        """
        return risk_aversion * np.dot(self.cov_matrix * 252, self.market_caps)
    
    def optimize_with_views(self, views: Dict, confidence: float = 0.1) -> Dict:
        """
        Optimize portfolio with Black-Litterman views.
        
        Args:
            views: Dictionary with view specifications
            confidence: Confidence level for views
            
        Returns:
            Optimized portfolio weights
        """
        # This is a simplified implementation
        # Full implementation would require more complex matrix operations
        implied_returns = self.calculate_implied_returns()
        
        # For now, return equal weights as placeholder
        weights = np.array([1/self.n_assets] * self.n_assets)
        
        return {
            'weights': weights,
            'implied_returns': implied_returns
        }

class RiskParity:
    """Risk Parity Portfolio Optimization."""
    
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.cov_matrix = returns.cov()
        self.n_assets = len(returns.columns)
    
    def optimize(self) -> Dict:
        """
        Optimize portfolio using risk parity approach.
        
        Returns:
            Risk parity portfolio weights
        """
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
            risk_contributions = (weights * np.dot(self.cov_matrix * 252, weights)) / portfolio_vol
            target_contributions = np.ones(self.n_assets) / self.n_assets
            return np.sum((risk_contributions - target_contributions) ** 2)
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        x0 = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(risk_parity_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return {
            'weights': result.x,
            'risk_contributions': (result.x * np.dot(self.cov_matrix * 252, result.x)) / 
                                np.sqrt(np.dot(result.x.T, np.dot(self.cov_matrix * 252, result.x)))
        }
