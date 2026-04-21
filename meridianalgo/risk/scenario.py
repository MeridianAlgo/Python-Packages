"""
Scenario Analysis and Stress Testing

Pre-built historical scenarios (dot-com crash, GFC, COVID-19, 2022 rate shock),
user-defined factor shocks, correlated multi-factor stress scenarios, and
reverse stress testing (find the scenario that causes a target loss).

References:
    BCBS (2009) - Principles for Sound Stress Testing Practices
    Rebonato (2010) - Coherent Stress Testing
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    """Container for scenario analysis output."""

    scenario_name: str
    portfolio_return: float
    portfolio_pnl: float
    asset_returns: pd.Series
    worst_asset: str
    best_asset: str
    severity: str


HISTORICAL_SCENARIOS: Dict[str, Dict[str, float]] = {
    "dot_com_crash_2000_2002": {
        "equity": -0.49,
        "technology": -0.78,
        "bonds": 0.12,
        "gold": 0.08,
        "real_estate": 0.03,
        "commodities": -0.10,
        "usd": 0.04,
        "credit_spread_change_bps": 150,
    },
    "gfc_2008_2009": {
        "equity": -0.55,
        "technology": -0.52,
        "bonds": 0.06,
        "gold": 0.02,
        "real_estate": -0.40,
        "commodities": -0.35,
        "usd": 0.12,
        "credit_spread_change_bps": 400,
        "vix_change": 60,
    },
    "covid_crash_march_2020": {
        "equity": -0.34,
        "technology": -0.28,
        "bonds": 0.04,
        "gold": -0.02,
        "real_estate": -0.40,
        "commodities": -0.25,
        "usd": 0.05,
        "credit_spread_change_bps": 300,
        "vix_change": 50,
    },
    "rate_shock_2022": {
        "equity": -0.19,
        "technology": -0.33,
        "bonds": -0.13,
        "gold": -0.01,
        "real_estate": -0.25,
        "commodities": 0.18,
        "usd": 0.15,
        "credit_spread_change_bps": 130,
    },
    "black_monday_1987": {
        "equity": -0.34,
        "technology": -0.35,
        "bonds": 0.04,
        "gold": 0.02,
        "usd": -0.08,
    },
    "asian_crisis_1997_1998": {
        "equity": -0.15,
        "bonds": 0.04,
        "usd": 0.05,
        "commodities": -0.12,
    },
    "euro_debt_crisis_2011": {
        "equity": -0.20,
        "bonds": 0.08,
        "credit_spread_change_bps": 180,
        "usd": 0.07,
    },
    "flash_crash_may_2010": {
        "equity": -0.07,
        "bonds": 0.01,
        "vix_change": 25,
    },
    "taper_tantrum_2013": {
        "equity": -0.06,
        "bonds": -0.05,
        "usd": 0.04,
    },
    "russia_ukraine_2022_initial": {
        "equity": -0.10,
        "commodities": 0.15,
        "energy": 0.25,
        "bonds": -0.03,
        "usd": 0.03,
    },
}


class ScenarioAnalyzer:
    """
    Apply stress scenarios to portfolios with factor-sensitivity mapping.

    Maps portfolio assets to macro factors, applies scenario shocks, and
    computes portfolio P&L. Supports historical scenarios, user-defined
    shocks, and reverse stress testing.

    Example:
        >>> analyzer = ScenarioAnalyzer(
        ...     portfolio_weights=weights,
        ...     factor_sensitivities=sensitivities,
        ...     portfolio_value=1_000_000,
        ... )
        >>> results = analyzer.run_all_historical()
        >>> for name, r in results.items():
        ...     print(f"{name}: {r.portfolio_return:.2%}")
    """

    def __init__(
        self,
        portfolio_weights: pd.Series,
        factor_sensitivities: pd.DataFrame,
        portfolio_value: float = 1.0,
    ):
        """
        Parameters
        ----------
        portfolio_weights : pd.Series
            Portfolio asset weights (sum to 1)
        factor_sensitivities : pd.DataFrame
            Asset-to-factor sensitivity matrix (assets x factors).
            Each cell is the expected return of that asset for a 100% shock
            to the given factor (i.e., a beta/loading).
        portfolio_value : float
            Total portfolio value for P&L computation
        """
        self.weights = portfolio_weights
        self.sensitivities = factor_sensitivities
        self.portfolio_value = portfolio_value

    def apply_scenario(
        self,
        factor_shocks: Dict[str, float],
        scenario_name: str = "custom",
    ) -> ScenarioResult:
        """
        Apply factor shocks to the portfolio.

        Parameters
        ----------
        factor_shocks : dict
            Factor name -> shock magnitude (decimal return)
        scenario_name : str
            Label for this scenario

        Returns
        -------
        ScenarioResult
        """
        available_factors = [
            f for f in factor_shocks if f in self.sensitivities.columns
        ]

        if not available_factors:
            asset_returns = pd.Series(0.0, index=self.weights.index)
        else:
            shock_vector = np.array([factor_shocks[f] for f in available_factors])
            sens_matrix = self.sensitivities[available_factors].reindex(
                self.weights.index
            ).fillna(0).values
            asset_returns = pd.Series(
                sens_matrix @ shock_vector,
                index=self.weights.index,
            )

        portfolio_return = float((self.weights * asset_returns).sum())
        portfolio_pnl = portfolio_return * self.portfolio_value

        if len(asset_returns) > 0:
            worst_asset = asset_returns.idxmin()
            best_asset = asset_returns.idxmax()
        else:
            worst_asset = best_asset = "N/A"

        if portfolio_return < -0.20:
            severity = "severe"
        elif portfolio_return < -0.10:
            severity = "moderate"
        elif portfolio_return < -0.05:
            severity = "mild"
        else:
            severity = "benign"

        return ScenarioResult(
            scenario_name=scenario_name,
            portfolio_return=portfolio_return,
            portfolio_pnl=portfolio_pnl,
            asset_returns=asset_returns,
            worst_asset=str(worst_asset),
            best_asset=str(best_asset),
            severity=severity,
        )

    def run_all_historical(self) -> Dict[str, ScenarioResult]:
        """
        Apply all built-in historical scenarios.

        Returns
        -------
        dict
            scenario_name -> ScenarioResult
        """
        results = {}
        for name, shocks in HISTORICAL_SCENARIOS.items():
            decimal_shocks = {
                k: v for k, v in shocks.items()
                if not k.endswith("_bps") and not k.endswith("_change")
            }
            results[name] = self.apply_scenario(decimal_shocks, name)
        return results

    def run_custom_scenario(
        self,
        name: str,
        equity_shock: float = 0.0,
        bond_shock: float = 0.0,
        credit_spread_shock: float = 0.0,
        fx_shock: float = 0.0,
        commodity_shock: float = 0.0,
    ) -> ScenarioResult:
        """
        Run a scenario with intuitive macro-factor inputs.

        Parameters
        ----------
        equity_shock : float
            Change in equity factor (e.g., -0.20 for -20% equity crash)
        bond_shock : float
            Change in bond/duration factor
        credit_spread_shock : float
            Change in credit spreads in decimal (e.g., 0.02 = +200bps)
        fx_shock : float
            Change in USD index (+ = USD strengthens)
        commodity_shock : float
            Change in commodity index
        """
        shocks = {
            "equity": equity_shock,
            "bonds": bond_shock,
            "credit": -credit_spread_shock,
            "usd": fx_shock,
            "commodities": commodity_shock,
        }
        return self.apply_scenario(shocks, name)

    def summary_table(
        self, results: Optional[Dict[str, ScenarioResult]] = None
    ) -> pd.DataFrame:
        """
        Format scenario results as a summary DataFrame.

        Parameters
        ----------
        results : dict, optional
            If None, runs all historical scenarios

        Returns
        -------
        pd.DataFrame
            Sorted by portfolio return (worst first)
        """
        if results is None:
            results = self.run_all_historical()

        rows = []
        for name, r in results.items():
            rows.append(
                {
                    "scenario": name,
                    "portfolio_return": r.portfolio_return,
                    "portfolio_pnl": r.portfolio_pnl,
                    "worst_asset": r.worst_asset,
                    "best_asset": r.best_asset,
                    "severity": r.severity,
                }
            )

        df = pd.DataFrame(rows).sort_values("portfolio_return").reset_index(drop=True)
        return df

    def reverse_stress_test(
        self,
        target_loss: float,
        factor: str,
        other_shocks: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Find the factor shock magnitude that produces a target portfolio loss.

        Parameters
        ----------
        target_loss : float
            Target portfolio loss in decimal (e.g., -0.10 for 10% loss)
        factor : str
            Factor to shock (must be in sensitivities columns)
        other_shocks : dict, optional
            Additional factor shocks to apply alongside the primary shock

        Returns
        -------
        float
            Required shock magnitude for the target factor
        """
        if factor not in self.sensitivities.columns:
            raise ValueError(f"Factor '{factor}' not in sensitivities matrix")

        base_shocks = other_shocks or {}

        from scipy.optimize import brentq

        def objective(shock_magnitude: float) -> float:
            shocks = {**base_shocks, factor: shock_magnitude}
            result = self.apply_scenario(shocks, "reverse_stress")
            return result.portfolio_return - target_loss

        try:
            required_shock = brentq(objective, -1.0, 1.0, xtol=1e-6)
        except ValueError:
            result_at_min = self.apply_scenario({**base_shocks, factor: -1.0}, "test")
            result_at_max = self.apply_scenario({**base_shocks, factor: 1.0}, "test")
            if result_at_min.portfolio_return > target_loss:
                required_shock = -1.0
                logger.warning(
                    "Target loss not achievable even at maximum shock (-100%%)"
                )
            else:
                required_shock = 1.0
                logger.warning(
                    "Target loss requires positive factor shock"
                )

        return required_shock


class CorrelationScenario:
    """
    Generate correlated multi-asset stress scenarios using Cholesky decomposition.

    Useful for estimating portfolio loss distributions under stressed
    correlation regimes (correlations tend to increase during crises).

    Example:
        >>> gen = CorrelationScenario(mean_returns, correlation_matrix, volatilities)
        >>> scenarios = gen.generate(n_scenarios=10_000)
        >>> var_95 = np.percentile(scenarios['portfolio_return'], 5)
    """

    def __init__(
        self,
        mean_returns: pd.Series,
        correlation_matrix: pd.DataFrame,
        volatilities: pd.Series,
        weights: pd.Series,
    ):
        """
        Parameters
        ----------
        mean_returns : pd.Series
            Expected daily returns per asset
        correlation_matrix : pd.DataFrame
            Correlation matrix (assets x assets)
        volatilities : pd.Series
            Daily volatilities per asset
        weights : pd.Series
            Portfolio weights
        """
        assets = mean_returns.index
        self.mean_returns = mean_returns.reindex(assets)
        self.volatilities = volatilities.reindex(assets)
        self.weights = weights.reindex(assets).fillna(0)
        self.correlation = correlation_matrix.reindex(index=assets, columns=assets)
        self.n_assets = len(assets)

    def generate(
        self,
        n_scenarios: int = 10_000,
        horizon_days: int = 1,
        stress_correlation: bool = False,
        stress_factor: float = 0.5,
        seed: int = 42,
    ) -> Dict[str, np.ndarray]:
        """
        Generate multivariate normal scenarios.

        Parameters
        ----------
        n_scenarios : int
            Number of scenarios to generate
        horizon_days : int
            Holding period in days
        stress_correlation : bool
            If True, apply correlation stress (blend toward 1)
        stress_factor : float
            Blend factor for correlation stress (0=no stress, 1=full correlation)
        seed : int
            Random seed

        Returns
        -------
        dict
            Keys: asset_returns (n_scenarios x n_assets), portfolio_return
        """
        rng = np.random.default_rng(seed)

        corr = self.correlation.values.copy()
        if stress_correlation:
            ones = np.ones_like(corr)
            np.fill_diagonal(ones, 1.0)
            corr = (1 - stress_factor) * corr + stress_factor * ones

        corr = np.clip(corr, -1, 1)
        np.fill_diagonal(corr, 1.0)

        try:
            L = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            corr += np.eye(self.n_assets) * 1e-6
            L = np.linalg.cholesky(corr)

        vol_sqrt_t = self.volatilities.values * np.sqrt(horizon_days)
        mean_t = self.mean_returns.values * horizon_days

        Z = rng.standard_normal((n_scenarios, self.n_assets))
        corr_Z = Z @ L.T
        asset_returns = mean_t + vol_sqrt_t * corr_Z

        portfolio_returns = asset_returns @ self.weights.values

        return {
            "asset_returns": asset_returns,
            "portfolio_return": portfolio_returns,
            "var_95": float(np.percentile(portfolio_returns, 5)),
            "var_99": float(np.percentile(portfolio_returns, 1)),
            "cvar_95": float(
                portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean()
            ),
            "mean": float(np.mean(portfolio_returns)),
            "std": float(np.std(portfolio_returns)),
        }
