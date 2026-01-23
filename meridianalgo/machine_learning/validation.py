"""
Proper time-series cross-validation and model evaluation for financial data.
Implements walk-forward analysis, purged cross-validation, and advanced validation techniques.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import BaseCrossValidator  # noqa: F401

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn(
        "Scikit-learn not available. Some validation features will be limited."
    )

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result from time-series validation."""

    scores: Dict[str, List[float]]
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]
    fold_results: List[Dict[str, Any]]
    validation_method: str
    n_splits: int
    success: bool
    message: str
    metadata: Dict[str, Any] = None


class BaseTimeSeriesValidator(ABC):
    """Abstract base class for time-series validators."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def split(
        self, X: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits for time-series data."""
        pass

    @abstractmethod
    def get_n_splits(
        self, X: np.ndarray = None, y: np.ndarray = None, groups: np.ndarray = None
    ) -> int:
        """Get number of splits."""
        pass


class WalkForwardValidator(BaseTimeSeriesValidator):
    """Walk-forward analysis for time-series data."""

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
        expanding_window: bool = False,
    ):
        """
        Initialize walk-forward validator.

        Args:
            n_splits: Number of splits to generate
            test_size: Size of test set (if None, uses equal splits)
            gap: Gap between train and test sets to prevent data leakage
            expanding_window: If True, use expanding window (train size grows)
        """
        super().__init__("WalkForward")
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.expanding_window = expanding_window

    def split(
        self, X: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward splits."""
        n_samples = len(X)

        if self.test_size is None:
            # Equal splits
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size

        for i in range(self.n_splits):
            # Calculate test indices
            test_start = n_samples - (self.n_splits - i) * test_size
            test_end = test_start + test_size

            if test_end > n_samples:
                test_end = n_samples

            # Calculate train indices
            if self.expanding_window:
                train_start = 0
            else:
                # Fixed window size
                train_size = test_start - self.gap
                train_start = max(
                    0, train_size - test_size * 3
                )  # Use 3x test size for training

            train_end = test_start - self.gap

            if train_end <= train_start:
                continue

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices

    def get_n_splits(
        self, X: np.ndarray = None, y: np.ndarray = None, groups: np.ndarray = None
    ) -> int:
        """Get number of splits."""
        return self.n_splits


class PurgedCrossValidator(BaseTimeSeriesValidator):
    """Purged cross-validation to prevent data leakage in time-series."""

    def __init__(self, n_splits: int = 5, purge_length: int = 1):
        """
        Initialize purged cross-validator.

        Args:
            n_splits: Number of splits
            purge_length: Number of samples to purge around test set
        """
        super().__init__("PurgedCV")
        self.n_splits = n_splits
        self.purge_length = purge_length

    def split(
        self, X: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate purged cross-validation splits."""
        n_samples = len(X)
        test_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            # Test set
            test_start = i * test_size
            test_end = min((i + 1) * test_size, n_samples)

            # Purge around test set
            purge_start = max(0, test_start - self.purge_length)
            purge_end = min(n_samples, test_end + self.purge_length)

            # Train set (excluding test and purged samples)
            train_indices = np.concatenate(
                [np.arange(0, purge_start), np.arange(purge_end, n_samples)]
            )

            test_indices = np.arange(test_start, test_end)

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

    def get_n_splits(
        self, X: np.ndarray = None, y: np.ndarray = None, groups: np.ndarray = None
    ) -> int:
        """Get number of splits."""
        return self.n_splits


class CombinatorialPurgedCV(BaseTimeSeriesValidator):
    """Combinatorial Purged Cross-Validation (CPCV) for advanced validation."""

    def __init__(
        self, n_splits: int = 5, n_test_groups: int = 2, purge_length: int = 1
    ):
        """
        Initialize combinatorial purged cross-validator.

        Args:
            n_splits: Number of groups to create
            n_test_groups: Number of groups to use for testing in each split
            purge_length: Number of samples to purge around test groups
        """
        super().__init__("CombinatorialPurgedCV")
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.purge_length = purge_length

    def split(
        self, X: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate combinatorial purged splits."""
        n_samples = len(X)
        group_size = n_samples // self.n_splits

        # Create all possible combinations of test groups
        from itertools import combinations

        for test_groups in combinations(range(self.n_splits), self.n_test_groups):
            test_indices = []
            purged_indices = set()

            # Collect test indices and purged regions
            for group_idx in test_groups:
                group_start = group_idx * group_size
                group_end = min((group_idx + 1) * group_size, n_samples)

                test_indices.extend(range(group_start, group_end))

                # Add purged regions
                purge_start = max(0, group_start - self.purge_length)
                purge_end = min(n_samples, group_end + self.purge_length)
                purged_indices.update(range(purge_start, purge_end))

            # Train indices (all samples except test and purged)
            all_indices = set(range(n_samples))
            train_indices = list(all_indices - purged_indices)

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield np.array(train_indices), np.array(test_indices)

    def get_n_splits(
        self, X: np.ndarray = None, y: np.ndarray = None, groups: np.ndarray = None
    ) -> int:
        """Get number of splits."""
        from math import comb

        return comb(self.n_splits, self.n_test_groups)


class TimeSeriesGroupCV(BaseTimeSeriesValidator):
    """Time-series cross-validation with custom groups."""

    def __init__(self, groups: np.ndarray, n_splits: int = None):
        """
        Initialize group-based time-series cross-validator.

        Args:
            groups: Array of group labels for each sample
            n_splits: Number of splits (if None, uses number of unique groups - 1)
        """
        super().__init__("TimeSeriesGroupCV")
        self.groups = groups
        self.unique_groups = np.unique(groups)
        self.n_splits = n_splits or len(self.unique_groups) - 1

    def split(
        self, X: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate group-based splits."""
        if groups is not None:
            self.groups = groups

        unique_groups = np.unique(self.groups)

        for i in range(min(self.n_splits, len(unique_groups) - 1)):
            # Use first i+1 groups for training, next group for testing
            train_groups = unique_groups[: i + 1]
            test_group = unique_groups[i + 1]

            train_indices = np.where(np.isin(self.groups, train_groups))[0]
            test_indices = np.where(self.groups == test_group)[0]

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

    def get_n_splits(
        self, X: np.ndarray = None, y: np.ndarray = None, groups: np.ndarray = None
    ) -> int:
        """Get number of splits."""
        return self.n_splits


class FinancialMetrics:
    """Financial-specific evaluation metrics."""

    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy."""
        if len(y_true) < 2:
            return 0.0

        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))

        return np.mean(true_direction == pred_direction)

    @staticmethod
    def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate information coefficient (correlation)."""
        if len(y_true) < 2:
            return 0.0

        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

    @staticmethod
    def hit_rate(
        y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.0
    ) -> float:
        """Calculate hit rate (percentage of correct predictions above threshold)."""
        correct_predictions = np.sum((y_pred > threshold) == (y_true > threshold))
        return correct_predictions / len(y_true)

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from returns."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate
        return (
            np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        )  # Annualized

    @staticmethod
    def maximum_drawdown(returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns."""
        if len(returns) == 0:
            return 0.0

        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak

        return np.min(drawdown)

    @staticmethod
    def calmar_ratio(returns: np.ndarray) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        if len(returns) == 0:
            return 0.0

        annual_return = np.mean(returns) * 252
        max_dd = FinancialMetrics.maximum_drawdown(returns)

        return annual_return / abs(max_dd) if max_dd != 0 else 0.0


class TimeSeriesValidator:
    """Main time-series validation class."""

    def __init__(self, validator: BaseTimeSeriesValidator = None):
        self.validator = validator or WalkForwardValidator()
        self.metrics = FinancialMetrics()

    def cross_validate(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        scoring: List[str] = None,
        return_predictions: bool = False,
        **fit_params,
    ) -> ValidationResult:
        """
        Perform cross-validation with time-series aware splits.

        Args:
            model: Model to validate (must have fit and predict methods)
            X: Feature matrix
            y: Target vector
            scoring: List of metrics to compute
            return_predictions: Whether to return predictions for each fold
            **fit_params: Additional parameters for model fitting

        Returns:
            ValidationResult with scores and metadata
        """
        if scoring is None:
            scoring = [
                "mse",
                "mae",
                "r2",
                "directional_accuracy",
                "information_coefficient",
            ]

        scores = {metric: [] for metric in scoring}
        fold_results = []
        all_predictions = []
        all_actuals = []

        try:
            for fold_idx, (train_idx, test_idx) in enumerate(
                self.validator.split(X, y)
            ):
                # Split data
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Fit model
                model_copy = self._clone_model(model)
                model_copy.fit(X_train, y_train, **fit_params)

                # Predict
                y_pred = model_copy.predict(X_test)

                # Calculate metrics
                fold_scores = {}
                for metric in scoring:
                    score = self._calculate_metric(metric, y_test, y_pred)
                    scores[metric].append(score)
                    fold_scores[metric] = score

                fold_result = {
                    "fold": fold_idx,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    "scores": fold_scores,
                }

                if return_predictions:
                    fold_result["predictions"] = y_pred
                    fold_result["actuals"] = y_test
                    all_predictions.extend(y_pred)
                    all_actuals.extend(y_test)

                fold_results.append(fold_result)

                logger.info(
                    f"Fold {fold_idx + 1}/{self.validator.get_n_splits()}: "
                    f"R = {fold_scores.get('r2', 0):.4f}"
                )

            # Calculate summary statistics
            mean_scores = {metric: np.mean(scores[metric]) for metric in scoring}
            std_scores = {metric: np.std(scores[metric]) for metric in scoring}

            # Overall metrics if predictions available
            metadata = {}
            if return_predictions and len(all_predictions) > 0:
                metadata["overall_scores"] = {}
                for metric in scoring:
                    overall_score = self._calculate_metric(
                        metric, np.array(all_actuals), np.array(all_predictions)
                    )
                    metadata["overall_scores"][metric] = overall_score

            return ValidationResult(
                scores=scores,
                mean_scores=mean_scores,
                std_scores=std_scores,
                fold_results=fold_results,
                validation_method=self.validator.name,
                n_splits=len(fold_results),
                success=True,
                message="Cross-validation completed successfully",
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return ValidationResult(
                scores={},
                mean_scores={},
                std_scores={},
                fold_results=[],
                validation_method=self.validator.name,
                n_splits=0,
                success=False,
                message=f"Cross-validation failed: {str(e)}",
            )

    def _clone_model(self, model):
        """Clone model for cross-validation."""
        # Simple cloning - in practice, you might want more sophisticated cloning
        try:
            # Try sklearn-style cloning
            from sklearn.base import clone

            return clone(model)
        except Exception:
            # Fallback: return the same model (not ideal but works for testing)
            return model

    def _calculate_metric(
        self, metric: str, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Calculate a specific metric."""
        try:
            if metric == "mse":
                return mean_squared_error(y_true, y_pred)
            elif metric == "mae":
                return mean_absolute_error(y_true, y_pred)
            elif metric == "rmse":
                return np.sqrt(mean_squared_error(y_true, y_pred))
            elif metric == "r2":
                return r2_score(y_true, y_pred)
            elif metric == "directional_accuracy":
                return self.metrics.directional_accuracy(y_true, y_pred)
            elif metric == "information_coefficient":
                return self.metrics.information_coefficient(y_true, y_pred)
            elif metric == "hit_rate":
                return self.metrics.hit_rate(y_true, y_pred)
            elif metric == "sharpe_ratio":
                # Treat predictions as returns for Sharpe calculation
                return self.metrics.sharpe_ratio(y_pred)
            elif metric == "max_drawdown":
                return self.metrics.maximum_drawdown(y_pred)
            elif metric == "calmar_ratio":
                return self.metrics.calmar_ratio(y_pred)
            else:
                logger.warning(f"Unknown metric: {metric}")
                return 0.0
        except Exception as e:
            logger.warning(f"Error calculating {metric}: {e}")
            return 0.0

    def validate_model_stability(
        self, model, X: np.ndarray, y: np.ndarray, n_runs: int = 10, **fit_params
    ) -> Dict[str, Any]:
        """
        Validate model stability across multiple runs.

        Args:
            model: Model to validate
            X: Feature matrix
            y: Target vector
            n_runs: Number of validation runs
            **fit_params: Additional parameters for model fitting

        Returns:
            Dictionary with stability metrics
        """
        all_scores = []

        for run in range(n_runs):
            # Add some randomness by shuffling the validation order
            np.random.seed(run)
            result = self.cross_validate(model, X, y, **fit_params)

            if result.success:
                all_scores.append(result.mean_scores)

        if not all_scores:
            return {"success": False, "message": "No successful validation runs"}

        # Calculate stability metrics
        stability_metrics = {}
        for metric in all_scores[0].keys():
            metric_values = [scores[metric] for scores in all_scores]
            stability_metrics[f"{metric}_mean"] = np.mean(metric_values)
            stability_metrics[f"{metric}_std"] = np.std(metric_values)
            stability_metrics[f"{metric}_cv"] = (
                np.std(metric_values) / abs(np.mean(metric_values))
                if np.mean(metric_values) != 0
                else np.inf
            )

        return {
            "success": True,
            "n_runs": len(all_scores),
            "stability_metrics": stability_metrics,
            "all_scores": all_scores,
        }


# Model Selection Framework
class ModelSelector:
    """Framework for model selection with proper time-series validation."""

    def __init__(self, validator: BaseTimeSeriesValidator = None):
        self.validator = validator or WalkForwardValidator()
        self.ts_validator = TimeSeriesValidator(self.validator)

    def select_best_model(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        scoring_metric: str = "r2",
        **fit_params,
    ) -> Dict[str, Any]:
        """
        Select best model using time-series cross-validation.

        Args:
            models: Dictionary of {name: model} pairs
            X: Feature matrix
            y: Target vector
            scoring_metric: Metric to use for selection
            **fit_params: Additional parameters for model fitting

        Returns:
            Dictionary with selection results
        """
        results = {}

        for name, model in models.items():
            logger.info(f"Evaluating model: {name}")

            result = self.ts_validator.cross_validate(model, X, y, **fit_params)
            results[name] = result

        # Find best model
        best_model = None
        best_score = (
            float("-inf")
            if scoring_metric in ["r2", "information_coefficient"]
            else float("inf")
        )

        for name, result in results.items():
            if result.success:
                score = result.mean_scores.get(scoring_metric, 0)

                if scoring_metric in [
                    "r2",
                    "information_coefficient",
                    "directional_accuracy",
                    "hit_rate",
                ]:
                    # Higher is better
                    if score > best_score:
                        best_score = score
                        best_model = name
                else:
                    # Lower is better (MSE, MAE, etc.)
                    if score < best_score:
                        best_score = score
                        best_model = name

        return {
            "best_model": best_model,
            "best_score": best_score,
            "all_results": results,
            "scoring_metric": scoring_metric,
        }

    def hyperparameter_search(
        self,
        model_class,
        param_grid: Dict[str, List],
        X: np.ndarray,
        y: np.ndarray,
        scoring_metric: str = "r2",
        **fit_params,
    ) -> Dict[str, Any]:
        """
        Hyperparameter search with time-series validation.

        Args:
            model_class: Model class to instantiate
            param_grid: Dictionary of parameter names and values to try
            X: Feature matrix
            y: Target vector
            scoring_metric: Metric to use for selection
            **fit_params: Additional parameters for model fitting

        Returns:
            Dictionary with search results
        """
        from itertools import product

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        best_params = None
        best_score = (
            float("-inf")
            if scoring_metric in ["r2", "information_coefficient"]
            else float("inf")
        )
        all_results = []

        for param_combination in product(*param_values):
            params = dict(zip(param_names, param_combination))

            try:
                # Create model with these parameters
                model = model_class(**params)

                # Validate
                result = self.ts_validator.cross_validate(model, X, y, **fit_params)

                if result.success:
                    score = result.mean_scores.get(scoring_metric, 0)

                    result_dict = {"params": params, "score": score, "result": result}
                    all_results.append(result_dict)

                    # Check if this is the best
                    if scoring_metric in [
                        "r2",
                        "information_coefficient",
                        "directional_accuracy",
                        "hit_rate",
                    ]:
                        if score > best_score:
                            best_score = score
                            best_params = params
                    else:
                        if score < best_score:
                            best_score = score
                            best_params = params

                    logger.info(f"Params {params}: {scoring_metric} = {score:.4f}")

            except Exception as e:
                logger.warning(f"Failed to evaluate params {params}: {e}")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": all_results,
            "scoring_metric": scoring_metric,
        }


if __name__ == "__main__":
    # Example usage
    print("Time-Series Validation Example")
    print("=" * 40)

    # Generate sample time-series data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5

    # Create time-series with trend and noise
    time_trend = np.linspace(0, 10, n_samples)
    X = np.random.randn(n_samples, n_features)
    X[:, 0] = (
        time_trend + np.random.randn(n_samples) * 0.1
    )  # Add trend to first feature

    # Target with some dependency on features and time
    y = (
        0.5 * X[:, 0]
        + 0.3 * X[:, 1]
        + np.sin(time_trend)
        + np.random.randn(n_samples) * 0.1
    )

    # Simple model for testing
    class SimpleModel:
        def __init__(self):
            self.coef_ = None

        def fit(self, X, y):
            # Simple linear regression
            self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
            return self

        def predict(self, X):
            return X @ self.coef_

    # Test different validators
    validators = {
        "Walk-Forward": WalkForwardValidator(n_splits=5),
        "Purged CV": PurgedCrossValidator(n_splits=5, purge_length=10),
        "Combinatorial Purged": CombinatorialPurgedCV(
            n_splits=6, n_test_groups=2, purge_length=5
        ),
    }

    for name, validator in validators.items():
        print(f"\nTesting {name}:")

        ts_validator = TimeSeriesValidator(validator)
        model = SimpleModel()

        result = ts_validator.cross_validate(model, X, y)

        if result.success:
            print(
                f"  R Score: {result.mean_scores['r2']:.4f}  {result.std_scores['r2']:.4f}"
            )
            print(
                f"  Directional Accuracy: {result.mean_scores['directional_accuracy']:.4f}"
            )
            print(f"  Number of folds: {result.n_splits}")
        else:
            print(f"  Validation failed: {result.message}")

    print("\nTime-series validation implementation completed!")
