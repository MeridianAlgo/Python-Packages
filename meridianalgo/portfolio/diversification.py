"""
MeridianAlgo Diversification Module

Advanced diversification strategies and analysis including asset class correlation,
geographic diversification, sector allocation, and alternative investment strategies.
"""

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Suppress warnings
warnings.filterwarnings("ignore")


class DiversificationAnalyzer:
    """
    Comprehensive diversification analysis and optimization.

    Features:
    - Asset class diversification analysis
    - Geographic diversification
    - Sector allocation optimization
    - Alternative investment strategies
    - Correlation clustering
    - Diversification metrics calculation
    """

    def __init__(self, returns: pd.DataFrame, asset_info: Dict[str, Dict[str, str]]):
        """
        Initialize diversification analyzer.

        Args:
            returns: DataFrame of asset returns
            asset_info: Dictionary with asset information (sector, region, type, etc.)
        """
        self.returns = returns
        self.asset_info = asset_info
        self.correlation_matrix = None
        self.diversification_metrics = None

    def calculate_correlation_clusters(
        self, n_clusters: int = 5, clustering_method: str = "kmeans"
    ) -> Dict[str, Any]:
        """
        Identify correlation-based clusters for diversification.

        Args:
            n_clusters: Number of clusters to identify
            clustering_method: Clustering method ('kmeans', 'hierarchical')

        Returns:
            Clustering results
        """
        # Calculate correlation matrix
        self.correlation_matrix = self.returns.corr()

        # Convert correlation to distance
        distance_matrix = np.sqrt((1 - self.correlation_matrix) / 2)

        if clustering_method == "kmeans":
            # Use k-means clustering on correlation-based features
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(distance_matrix)
        else:
            # Hierarchical clustering (simplified)
            from scipy.cluster.hierarchy import fcluster, linkage

            # Convert to condensed distance matrix
            condensed_distance = []
            n = len(self.correlation_matrix)
            for i in range(n):
                for j in range(i + 1, n):
                    condensed_distance.append(distance_matrix.iloc[i, j])

            linkage_matrix = linkage(condensed_distance, method="ward")
            cluster_labels = (
                fcluster(linkage_matrix, n_clusters, criterion="maxclust") - 1
            )

        # Organize results
        clusters = {}
        for i, asset in enumerate(self.returns.columns):
            cluster_id = cluster_labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(asset)

        # Calculate cluster statistics
        cluster_stats = {}
        for cluster_id, assets in clusters.items():
            cluster_returns = self.returns[assets]
            cluster_stats[cluster_id] = {
                "assets": assets,
                "avg_correlation": self.correlation_matrix.loc[assets, assets]
                .values[
                    np.triu_indices_from(
                        self.correlation_matrix.loc[assets, assets].values, k=1
                    )
                ]
                .mean(),
                "volatility": cluster_returns.std().mean() * np.sqrt(252),
                "return": cluster_returns.mean().mean() * 252,
                "sharpe_ratio": (cluster_returns.mean().mean() * 252)
                / (cluster_returns.std().mean() * np.sqrt(252)),
            }

        return {
            "clusters": clusters,
            "cluster_stats": cluster_stats,
            "cluster_labels": dict(zip(self.returns.columns, cluster_labels)),
        }

    def calculate_diversification_ratio(
        self, weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate diversification ratio.

        Args:
            weights: Portfolio weights (equal weights if None)

        Returns:
            Diversification ratio
        """
        if self.correlation_matrix is None:
            self.correlation_matrix = self.returns.corr()

        if weights is None:
            weights = np.ones(len(self.returns.columns)) / len(self.returns.columns)

        volatilities = self.returns.std() * np.sqrt(252)
        weighted_avg_vol = (weights * volatilities).sum()

        portfolio_vol = np.sqrt(weights @ (self.returns.cov() * 252) @ weights)

        return weighted_avg_vol / portfolio_vol

    def calculate_effective_number_bets(
        self, weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate effective number of independent bets.

        Args:
            weights: Portfolio weights

        Returns:
            Effective number of bets
        """
        if weights is None:
            weights = np.ones(len(self.returns.columns)) / len(self.returns.columns)

        # Using principal components to estimate independent factors
        pca = PCA()
        pca.fit(self.returns.dropna())

        # Calculate contribution of each factor to portfolio variance
        explained_variance_ratio = pca.explained_variance_ratio_

        # Effective number of factors accounting for 95% of variance
        cumulative_variance = np.cumsum(explained_variance_ratio)
        n_factors = np.argmax(cumulative_variance >= 0.95) + 1

        return min(n_factors, len(weights))

    def analyze_geographic_diversification(self) -> Dict[str, Any]:
        """
        Analyze geographic diversification of portfolio.

        Returns:
            Geographic diversification analysis
        """
        # Group assets by geographic region
        geographic_groups = {}
        for asset, info in self.asset_info.items():
            region = info.get("region", "Unknown")
            if region not in geographic_groups:
                geographic_groups[region] = []
            if asset in self.returns.columns:
                geographic_groups[region].append(asset)

        # Calculate metrics for each region
        region_metrics = {}
        for region, assets in geographic_groups.items():
            if assets:
                region_returns = self.returns[assets]
                region_metrics[region] = {
                    "assets": assets,
                    "weight": len(assets) / len(self.returns.columns),
                    "return": region_returns.mean().mean() * 252,
                    "volatility": region_returns.std().mean() * np.sqrt(252),
                    "sharpe_ratio": (region_returns.mean().mean() * 252)
                    / (region_returns.std().mean() * np.sqrt(252)),
                }

        # Calculate geographic concentration
        geographic_weights = [metrics["weight"] for metrics in region_metrics.values()]
        geographic_concentration = np.sum(np.array(geographic_weights) ** 2)

        return {
            "region_metrics": region_metrics,
            "geographic_concentration": geographic_concentration,
            "effective_regions": 1 / geographic_concentration,
        }

    def analyze_sector_diversification(self) -> Dict[str, Any]:
        """
        Analyze sector diversification of portfolio.

        Returns:
            Sector diversification analysis
        """
        # Group assets by sector
        sector_groups = {}
        for asset, info in self.asset_info.items():
            sector = info.get("sector", "Unknown")
            if sector not in sector_groups:
                sector_groups[sector] = []
            if asset in self.returns.columns:
                sector_groups[sector].append(asset)

        # Calculate metrics for each sector
        sector_metrics = {}
        for sector, assets in sector_groups.items():
            if assets:
                sector_returns = self.returns[assets]
                sector_metrics[sector] = {
                    "assets": assets,
                    "weight": len(assets) / len(self.returns.columns),
                    "return": sector_returns.mean().mean() * 252,
                    "volatility": sector_returns.std().mean() * np.sqrt(252),
                    "sharpe_ratio": (sector_returns.mean().mean() * 252)
                    / (sector_returns.std().mean() * np.sqrt(252)),
                }

        # Calculate sector concentration
        sector_weights = [metrics["weight"] for metrics in sector_metrics.values()]
        sector_concentration = np.sum(np.array(sector_weights) ** 2)

        return {
            "sector_metrics": sector_metrics,
            "sector_concentration": sector_concentration,
            "effective_sectors": 1 / sector_concentration,
        }

    def optimize_diversified_portfolio(
        self,
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        objective: str = "max_diversification",
    ) -> Dict[str, Any]:
        """
        Optimize portfolio for maximum diversification.

        Args:
            constraints: Constraints for asset groups
            objective: Optimization objective

        Returns:
            Optimized portfolio weights
        """
        n_assets = len(self.returns.columns)

        if objective == "max_diversification":
            # Maximum diversification ratio
            weights = self._max_diversification_optimization()
        elif objective == "risk_parity":
            # Risk parity
            weights = self._risk_parity_optimization()
        elif objective == "equal_weight":
            # Equal weight
            weights = np.ones(n_assets) / n_assets
        else:
            raise ValueError(f"Unknown objective: {objective}")

        # Apply constraints
        if constraints:
            weights = self._apply_constraints(weights, constraints)

        # Calculate portfolio metrics
        portfolio_returns = (
            self.returns * pd.Series(weights, index=self.returns.columns)
        ).sum(axis=1)

        return {
            "weights": dict(zip(self.returns.columns, weights)),
            "diversification_ratio": self.calculate_diversification_ratio(weights),
            "effective_number_bets": self.calculate_effective_number_bets(weights),
            "portfolio_return": portfolio_returns.mean() * 252,
            "portfolio_volatility": portfolio_returns.std() * np.sqrt(252),
            "sharpe_ratio": (portfolio_returns.mean() * 252)
            / (portfolio_returns.std() * np.sqrt(252)),
        }

    def _max_diversification_optimization(self) -> np.ndarray:
        """Maximum diversification optimization."""
        cov_matrix = self.returns.cov() * 252
        volatilities = np.sqrt(np.diag(cov_matrix))

        # Simplified optimization using inverse volatility weights
        inv_vol_weights = 1 / volatilities
        weights = inv_vol_weights / inv_vol_weights.sum()

        return weights

    def _risk_parity_optimization(self) -> np.ndarray:
        """Risk parity optimization."""
        n_assets = len(self.returns.columns)
        cov_matrix = self.returns.cov() * 252

        # Simplified risk parity using iterative approach
        weights = np.ones(n_assets) / n_assets

        for _ in range(100):
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_risk = (cov_matrix @ weights) / portfolio_vol
            risk_contributions = weights * marginal_risk

            # Update weights to equalize risk contributions
            weights = weights * (risk_contributions.mean() / risk_contributions)
            weights = weights / weights.sum()

        return weights

    def _apply_constraints(
        self, weights: np.ndarray, constraints: Dict[str, Tuple[float, float]]
    ) -> np.ndarray:
        """Apply constraints to portfolio weights."""
        # This is a simplified constraint application
        # In practice, you'd want to use optimization libraries for complex constraints

        for asset_group, (min_weight, max_weight) in constraints.items():
            # Find assets in this group (simplified)
            group_assets = [
                asset
                for asset in self.returns.columns
                if asset_group.lower() in asset.lower()
            ]

            if group_assets:
                group_indices = [
                    self.returns.columns.get_loc(asset) for asset in group_assets
                ]
                current_group_weight = weights[group_indices].sum()

                # Adjust weights to meet constraints
                if current_group_weight < min_weight:
                    scale_factor = min_weight / current_group_weight
                    weights[group_indices] *= scale_factor
                elif current_group_weight > max_weight:
                    scale_factor = max_weight / current_group_weight
                    weights[group_indices] *= scale_factor

        # Normalize weights
        weights = weights / weights.sum()
        return weights


class AlternativeInvestmentAnalyzer:
    """
    Analysis of alternative investment strategies for diversification.

    Features:
    - Real estate investment analysis
    - Commodities diversification
    - Private equity strategies
    - Hedge fund strategies
    - Infrastructure investments
    """

    def __init__(self):
        """Initialize alternative investment analyzer."""
        self.alternative_returns = None
        self.alternative_correlations = None

    def analyze_alternative_investments(
        self, traditional_returns: pd.DataFrame, alternative_returns: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze diversification benefits of alternative investments.

        Args:
            traditional_returns: Traditional asset returns
            alternative_returns: Alternative investment returns

        Returns:
            Alternative investment analysis
        """
        # Combine returns
        all_returns = pd.concat(
            [traditional_returns, alternative_returns], axis=1
        ).dropna()

        # Calculate correlations
        correlations = all_returns.corr()

        # Calculate diversification benefits
        traditional_only = traditional_returns.mean(axis=1)
        combined = all_returns.mean(axis=1)

        # Calculate metrics
        traditional_vol = traditional_only.std() * np.sqrt(252)
        combined_vol = combined.std() * np.sqrt(252)
        diversification_benefit = (traditional_vol - combined_vol) / traditional_vol

        return {
            "correlations": correlations,
            "diversification_benefit": diversification_benefit,
            "traditional_volatility": traditional_vol,
            "combined_volatility": combined_vol,
            "alternative_correlation_avg": correlations.loc[
                traditional_returns.columns, alternative_returns.columns
            ]
            .mean()
            .mean(),
        }

    def calculate_alternative_portfolio_allocation(
        self,
        traditional_returns: pd.DataFrame,
        alternative_returns: pd.DataFrame,
        target_alternative_allocation: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Calculate optimal allocation to alternative investments.

        Args:
            traditional_returns: Traditional asset returns
            alternative_returns: Alternative investment returns
            target_alternative_allocation: Target allocation to alternatives

        Returns:
            Portfolio allocation results
        """
        # Calculate risk-adjusted returns
        (traditional_returns.mean() * 252) / (traditional_returns.std() * np.sqrt(252))
        (alternative_returns.mean() * 252) / (alternative_returns.std() * np.sqrt(252))

        # Optimize allocation (simplified)
        n_alternatives = len(alternative_returns.columns)
        alternative_weights = (
            np.ones(n_alternatives) / n_alternatives * target_alternative_allocation
        )

        n_traditional = len(traditional_returns.columns)
        traditional_weights = (
            np.ones(n_traditional) / n_traditional * (1 - target_alternative_allocation)
        )

        # Combine weights
        all_weights = np.concatenate([traditional_weights, alternative_weights])
        all_returns = pd.concat(
            [traditional_returns, alternative_returns], axis=1
        ).dropna()

        # Calculate portfolio metrics
        portfolio_returns = (all_returns * all_weights).sum(axis=1)

        return {
            "traditional_weights": dict(
                zip(traditional_returns.columns, traditional_weights)
            ),
            "alternative_weights": dict(
                zip(alternative_returns.columns, alternative_weights)
            ),
            "portfolio_return": portfolio_returns.mean() * 252,
            "portfolio_volatility": portfolio_returns.std() * np.sqrt(252),
            "sharpe_ratio": (portfolio_returns.mean() * 252)
            / (portfolio_returns.std() * np.sqrt(252)),
            "alternative_allocation": target_alternative_allocation,
        }


# Utility functions
def get_asset_class_template() -> Dict[str, Dict[str, str]]:
    """Get template for asset information structure."""
    return {
        "AAPL": {
            "sector": "Technology",
            "region": "US",
            "type": "Equity",
            "market_cap": "Large",
        },
        "MSFT": {
            "sector": "Technology",
            "region": "US",
            "type": "Equity",
            "market_cap": "Large",
        },
        "GOOGL": {
            "sector": "Technology",
            "region": "US",
            "type": "Equity",
            "market_cap": "Large",
        },
        "AMZN": {
            "sector": "Consumer Discretionary",
            "region": "US",
            "type": "Equity",
            "market_cap": "Large",
        },
        "TSLA": {
            "sector": "Consumer Discretionary",
            "region": "US",
            "type": "Equity",
            "market_cap": "Large",
        },
        "BTC": {
            "sector": "Cryptocurrency",
            "region": "Global",
            "type": "Crypto",
            "market_cap": "Large",
        },
        "ETH": {
            "sector": "Cryptocurrency",
            "region": "Global",
            "type": "Crypto",
            "market_cap": "Large",
        },
        "EUR": {
            "sector": "Currency",
            "region": "Europe",
            "type": "Forex",
            "market_cap": "N/A",
        },
        "GBP": {
            "sector": "Currency",
            "region": "UK",
            "type": "Forex",
            "market_cap": "N/A",
        },
        "JPY": {
            "sector": "Currency",
            "region": "Japan",
            "type": "Forex",
            "market_cap": "N/A",
        },
        "Gold": {
            "sector": "Commodity",
            "region": "Global",
            "type": "Commodity",
            "market_cap": "N/A",
        },
        "Oil": {
            "sector": "Commodity",
            "region": "Global",
            "type": "Commodity",
            "market_cap": "N/A",
        },
    }


def calculate_herfindahl_index(weights: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Herfindahl-Hirschman Index for concentration.

    Args:
        weights: Portfolio weights

    Returns:
        HHI value
    """
    if isinstance(weights, pd.Series):
        weights = weights.values

    return np.sum(weights**2)


def calculate_concentration_ratio(
    weights: Union[np.ndarray, pd.Series], n: int = 5
) -> float:
    """
    Calculate concentration ratio for top n assets.

    Args:
        weights: Portfolio weights
        n: Number of top assets to consider

    Returns:
        Concentration ratio
    """
    if isinstance(weights, pd.Series):
        weights = weights.values

    sorted_weights = np.sort(weights)[::-1]
    return np.sum(sorted_weights[:n])


# Export main classes and functions
__all__ = [
    "DiversificationAnalyzer",
    "AlternativeInvestmentAnalyzer",
    "get_asset_class_template",
    "calculate_herfindahl_index",
    "calculate_concentration_ratio",
]
