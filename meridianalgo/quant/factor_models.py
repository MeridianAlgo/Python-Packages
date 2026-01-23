"""
Factor Models Module

Multi-factor models for risk decomposition, alpha generation,
and portfolio construction.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge


class FamaFrenchModel:
    """
    Fama-French three-factor and five-factor models.
    """

    def __init__(self, model_type: str = "three_factor"):
        """
        Initialize Fama-French model.

        Parameters:
        -----------
        model_type : str
            'three_factor' or 'five_factor'
        """
        self.model_type = model_type
        self.coefficients = None
        self.alpha = None
        self.r_squared = None

    def fit(
        self,
        returns: pd.Series,
        factor_data: pd.DataFrame,
        risk_free_rate: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Fit Fama-French model.

        Parameters:
        -----------
        returns : pd.Series
            Asset returns
        factor_data : pd.DataFrame
            Factor returns (MKT, SMB, HML, RMW, CMA)
        risk_free_rate : pd.Series, optional
            Risk-free rate

        Returns:
        --------
        Dict
            Model results
        """
        # Calculate excess returns
        if risk_free_rate is not None:
            excess_returns = returns - risk_free_rate
        else:
            excess_returns = returns

        # Align data
        common_index = excess_returns.index.intersection(factor_data.index)
        y = excess_returns.loc[common_index].values

        if self.model_type == "three_factor":
            X = factor_data.loc[common_index, ["MKT", "SMB", "HML"]].values
            factor_names = ["Market", "Size", "Value"]
        else:  # five_factor
            X = factor_data.loc[
                common_index, ["MKT", "SMB", "HML", "RMW", "CMA"]
            ].values
            factor_names = ["Market", "Size", "Value", "Profitability", "Investment"]

        # Fit regression
        model = LinearRegression()
        model.fit(X, y)

        self.alpha = model.intercept_
        self.coefficients = dict(zip(factor_names, model.coef_))

        # Calculate R-squared
        predictions = model.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # T-statistics
        residuals = y - predictions
        mse = np.sum(residuals**2) / (len(y) - X.shape[1] - 1)
        var_beta = mse * np.linalg.inv(X.T @ X).diagonal()
        t_stats = model.coef_ / np.sqrt(var_beta)

        # Alpha t-stat
        alpha_var = mse * (
            1 / len(y)
            + np.mean(X, axis=0) @ np.linalg.inv(X.T @ X) @ np.mean(X, axis=0).T
        )
        alpha_t_stat = self.alpha / np.sqrt(alpha_var) if alpha_var > 0 else 0

        return {
            "alpha": self.alpha,
            "coefficients": self.coefficients,
            "r_squared": self.r_squared,
            "alpha_t_stat": alpha_t_stat,
            "t_stats": dict(zip(factor_names, t_stats)),
            "significant_alpha": abs(alpha_t_stat) > 1.96,  # 95% confidence
        }

    def predict(self, factor_data: pd.DataFrame) -> pd.Series:
        """
        Predict returns using fitted model.
        """
        if self.coefficients is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.model_type == "three_factor":
            factors = factor_data[["MKT", "SMB", "HML"]].values
        else:
            factors = factor_data[["MKT", "SMB", "HML", "RMW", "CMA"]].values

        coef_values = np.array(list(self.coefficients.values()))
        predictions = self.alpha + factors @ coef_values

        return pd.Series(predictions, index=factor_data.index)


class APTModel:
    """
    Arbitrage Pricing Theory (APT) model with statistical factors.
    """

    def __init__(self, n_factors: int = 5):
        """
        Initialize APT model.

        Parameters:
        -----------
        n_factors : int
            Number of factors to extract
        """
        self.n_factors = n_factors
        self.factor_loadings = None
        self.factors = None
        self.pca = None

    def extract_factors(self, return_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Extract statistical factors using PCA.

        Parameters:
        -----------
        return_matrix : pd.DataFrame
            Returns for multiple assets (rows=dates, cols=assets)

        Returns:
        --------
        pd.DataFrame
            Extracted factors
        """
        # Standardize returns
        standardized_returns = (
            return_matrix - return_matrix.mean()
        ) / return_matrix.std()

        # Apply PCA
        self.pca = PCA(n_components=self.n_factors)
        factor_returns = self.pca.fit_transform(standardized_returns.fillna(0))

        # Create factor DataFrame
        self.factors = pd.DataFrame(
            factor_returns,
            index=return_matrix.index,
            columns=[f"Factor{i + 1}" for i in range(self.n_factors)],
        )

        # Factor loadings
        self.factor_loadings = pd.DataFrame(
            self.pca.components_.T,
            index=return_matrix.columns,
            columns=[f"Factor{i + 1}" for i in range(self.n_factors)],
        )

        # Explained variance
        explained_var = pd.Series(
            self.pca.explained_variance_ratio_,
            index=[f"Factor{i + 1}" for i in range(self.n_factors)],
        )

        return {
            "factors": self.factors,
            "loadings": self.factor_loadings,
            "explained_variance": explained_var,
            "cumulative_variance": explained_var.cumsum(),
        }

    def fit_asset(self, asset_returns: pd.Series) -> Dict:
        """
        Fit APT model for a single asset.

        Parameters:
        -----------
        asset_returns : pd.Series
            Asset returns

        Returns:
        --------
        Dict
            Factor exposures and statistics
        """
        if self.factors is None:
            raise ValueError("Extract factors first using extract_factors()")

        # Align data
        common_index = asset_returns.index.intersection(self.factors.index)
        y = asset_returns.loc[common_index].values
        X = self.factors.loc[common_index].values

        # Fit regression
        model = LinearRegression()
        model.fit(X, y)

        alpha = model.intercept_
        betas = dict(zip(self.factors.columns, model.coef_))

        # Calculate R-squared
        predictions = model.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            "alpha": alpha,
            "betas": betas,
            "r_squared": r_squared,
            "predicted_return": alpha
            + np.sum(list(betas.values()) * self.factors.iloc[-1].values),
        }


class CustomFactorModel:
    """
    Custom factor model using user-defined factors.
    """

    def __init__(self, factor_names: List[str]):
        """
        Initialize custom factor model.

        Parameters:
        -----------
        factor_names : List[str]
            Names of custom factors
        """
        self.factor_names = factor_names
        self.models = {}

    def fit(
        self, returns: pd.DataFrame, factors: pd.DataFrame, regularization: float = 0.0
    ) -> Dict:
        """
        Fit custom factor model for multiple assets.

        Parameters:
        -----------
        returns : pd.DataFrame
            Asset returns (columns = assets)
        factors : pd.DataFrame
            Factor values (columns = factors)
        regularization : float
            L2 regularization parameter (0 = no regularization)

        Returns:
        --------
        Dict
            Model results for all assets
        """
        results = {}

        # Align data
        common_index = returns.index.intersection(factors.index)

        for asset in returns.columns:
            y = returns.loc[common_index, asset].values
            X = factors.loc[common_index, self.factor_names].values

            # Fit with regularization if specified
            if regularization > 0:
                model = Ridge(alpha=regularization)
            else:
                model = LinearRegression()

            model.fit(X, y)

            alpha = model.intercept_
            betas = dict(zip(self.factor_names, model.coef_))

            # Calculate metrics
            predictions = model.predict(X)
            residuals = y - predictions

            # R-squared
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            # Information Ratio (assuming specific return = alpha)
            ir = (
                alpha / np.std(residuals) * np.sqrt(252) if np.std(residuals) > 0 else 0
            )

            results[asset] = {
                "alpha": alpha,
                "betas": betas,
                "r_squared": r_squared,
                "information_ratio": ir,
                "residual_volatility": np.std(residuals),
            }

            self.models[asset] = model

        return results

    def calculate_factor_contributions(
        self, asset: str, factor_returns: pd.Series
    ) -> pd.Series:
        """
        Calculate contribution of each factor to asset return.
        """
        if asset not in self.models:
            raise ValueError(f"Model not fitted for {asset}")

        model = self.models[asset]
        contributions = {}

        for i, factor_name in enumerate(self.factor_names):
            beta = model.coef_[i]
            factor_return = factor_returns.get(factor_name, 0)
            contributions[factor_name] = beta * factor_return

        contributions["Alpha"] = model.intercept_

        return pd.Series(contributions)


class FactorRiskDecomposition:
    """
    Decompose portfolio risk into factor components.
    """

    @staticmethod
    def decompose_variance(
        weights: np.ndarray,
        factor_exposures: np.ndarray,
        factor_covariance: np.ndarray,
        specific_variance: np.ndarray,
    ) -> Dict:
        """
        Decompose portfolio variance into factor and specific risk.

        Variance = B' F B + D
        where B = factor exposures, F = factor covariance, D = specific variance

        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        factor_exposures : np.ndarray
            Factor exposures (assets x factors)
        factor_covariance : np.ndarray
            Factor covariance matrix
        specific_variance : np.ndarray
            Asset-specific variances

        Returns:
        --------
        Dict
            Variance decomposition
        """
        # Portfolio factor exposures
        portfolio_exposures = weights @ factor_exposures

        # Factor contribution to variance
        factor_variance = (
            portfolio_exposures @ factor_covariance @ portfolio_exposures.T
        )

        # Specific risk contribution
        specific_risk_variance = weights @ np.diag(specific_variance) @ weights.T

        # Total variance
        total_variance = factor_variance + specific_risk_variance

        # Individual factor contributions
        factor_contributions = {}
        for i in range(factor_exposures.shape[1]):
            exposure_i = portfolio_exposures[i]
            var_i = exposure_i**2 * factor_covariance[i, i]
            factor_contributions[f"Factor{i + 1}"] = (
                var_i / total_variance if total_variance > 0 else 0
            )

        return {
            "total_variance": total_variance,
            "total_volatility": np.sqrt(total_variance),
            "factor_variance": factor_variance,
            "specific_variance": specific_risk_variance,
            "factor_risk_pct": (
                factor_variance / total_variance if total_variance > 0 else 0
            ),
            "specific_risk_pct": (
                specific_risk_variance / total_variance if total_variance > 0 else 0
            ),
            "factor_contributions": factor_contributions,
        }

    @staticmethod
    def marginal_factor_contributions(
        weights: np.ndarray, factor_exposures: np.ndarray, factor_covariance: np.ndarray
    ) -> pd.DataFrame:
        """
        Calculate marginal contribution of each asset to factor risks.
        """
        portfolio_exposures = weights @ factor_exposures

        # Marginal contributions
        marginal_contribs = factor_exposures @ factor_covariance @ portfolio_exposures.T

        # Percentage contributions
        total_risk = np.sqrt(
            portfolio_exposures @ factor_covariance @ portfolio_exposures.T
        )
        pct_contributions = (
            (weights * marginal_contribs) / total_risk
            if total_risk > 0
            else np.zeros_like(weights)
        )

        return pd.DataFrame(
            {
                "weight": weights,
                "marginal_contribution": marginal_contribs,
                "percent_contribution": pct_contributions,
            }
        )


class AlphaCapture:
    """
    Alpha generation and capture using factor models.
    """

    @staticmethod
    def calculate_pure_alpha(
        returns: pd.Series,
        factor_returns: pd.DataFrame,
        factor_loadings: Dict[str, float],
    ) -> pd.Series:
        """
        Calculate pure alpha by removing factor contributions.

        Alpha = Return - (beta_i * factor_return_i)
        """
        factor_contribution = pd.Series(0, index=returns.index)

        for factor_name, beta in factor_loadings.items():
            if factor_name in factor_returns.columns:
                factor_contribution += beta * factor_returns[factor_name]

        pure_alpha = returns - factor_contribution
        return pure_alpha

    @staticmethod
    def information_coefficient(
        predicted_returns: pd.Series, actual_returns: pd.Series
    ) -> float:
        """
        Calculate Information Coefficient (IC).

        IC = correlation between predicted and actual returns
        """
        # Align series
        common_index = predicted_returns.index.intersection(actual_returns.index)

        if len(common_index) == 0:
            return 0.0

        pred = predicted_returns.loc[common_index]
        actual = actual_returns.loc[common_index]

        # Remove NaN values
        valid_mask = ~(pred.isna() | actual.isna())
        pred = pred[valid_mask]
        actual = actual[valid_mask]

        if len(pred) < 2:
            return 0.0

        ic = stats.spearmanr(pred, actual)[0]
        return ic

    @staticmethod
    def fundamental_factor_scores(
        price_data: pd.DataFrame, fundamental_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate factor scores from fundamental data.

        Common factors: Value, Growth, Quality, Momentum
        """
        scores = pd.DataFrame(index=price_data.index)

        # Value score (e.g., P/E, P/B ratios - lower is better)
        if "PE" in fundamental_data.columns:
            scores["Value"] = -fundamental_data["PE"].rank(pct=True)

        # Growth score (e.g., earnings growth - higher is better)
        if "EPS_Growth" in fundamental_data.columns:
            scores["Growth"] = fundamental_data["EPS_Growth"].rank(pct=True)

        # Quality score (e.g., ROE, debt ratios)
        if "ROE" in fundamental_data.columns:
            scores["Quality"] = fundamental_data["ROE"].rank(pct=True)

        # Momentum score (recent price performance)
        momentum = price_data.pct_change(periods=60).iloc[-1]  # 60-day momentum
        scores["Momentum"] = momentum.rank(pct=True)

        return scores

    @staticmethod
    def optimize_factor_tilts(
        expected_factor_returns: Dict[str, float],
        factor_covariance: np.ndarray,
        factor_names: List[str],
        max_tilt: float = 0.5,
    ) -> Dict[str, float]:
        """Optimize portfolio tilts towards high-alpha factors.

        Parameters:
        -----------
        expected_factor_returns : Dict[str, float]
            Expected return for each factor
        factor_covariance : np.ndarray
            Covariance matrix of factors
        factor_names : List[str]
            Names of factors
        max_tilt : float
            Maximum tilt (exposure) to any single factor

        Returns:
        --------
        Dict[str, float]
            Optimal factor tilts
        """
        n_factors = len(factor_names)

        # Expected returns vector
        mu = np.array([expected_factor_returns.get(name, 0) for name in factor_names])

        # Optimization: maximize Sharpe ratio
        def neg_sharpe(tilts):
            portfolio_return = tilts @ mu
            portfolio_vol = np.sqrt(tilts @ factor_covariance @ tilts.T)
            return -portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

        # Constraints
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]  # Sum to 1

        bounds = [(-max_tilt, max_tilt) for _ in range(n_factors)]

        # Initial guess: equal weight
        x0 = np.ones(n_factors) / n_factors

        # Optimize
        result = minimize(
            neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        optimal_tilts = dict(zip(factor_names, result.x))

        return optimal_tilts


__all__ = [
    "FamaFrenchModel",
    "APTModel",
    "CustomFactorModel",
    "FactorRiskDecomposition",
    "AlphaCapture",
]
