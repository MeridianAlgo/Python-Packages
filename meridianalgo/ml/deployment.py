"""
Model deployment and monitoring system for financial ML models.
Implements model versioning, A/B testing, performance monitoring, and automated retraining.
"""

import hashlib
import json
import logging
import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import joblib

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    warnings.warn("Joblib not available. Model serialization will be limited.")

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Model version information."""

    version_id: str
    model_name: str
    created_at: datetime
    model_hash: str
    performance_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = False
    deployment_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionLog:
    """Log entry for model predictions."""

    timestamp: datetime
    model_version: str
    input_features: Dict[str, Any]
    prediction: Union[float, np.ndarray]
    confidence: Optional[float] = None
    actual_outcome: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance monitoring alert."""

    alert_id: str
    timestamp: datetime
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    metrics: Dict[str, float]
    threshold_breached: Dict[str, float]
    model_version: str


class ModelRegistry:
    """Registry for managing model versions and metadata."""

    def __init__(self, registry_path: str = "model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        self.models_path = self.registry_path / "models"
        self.models_path.mkdir(exist_ok=True)
        self.metadata_file = self.registry_path / "registry.json"

        # Load existing registry
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")

        return {"models": {}, "versions": {}}

    def _save_registry(self) -> None:
        """Save registry to file."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.registry, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def register_model(
        self,
        model_name: str,
        model: Any,
        performance_metrics: Dict[str, float],
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Register a new model version.

        Args:
            model_name: Name of the model
            model: The model object to register
            performance_metrics: Performance metrics for this model
            metadata: Additional metadata

        Returns:
            Version ID of the registered model
        """
        # Generate version ID
        timestamp = datetime.now()
        version_id = f"{model_name}_v{timestamp.strftime('%Y%m%d_%H%M%S')}"

        # Calculate model hash
        model_hash = self._calculate_model_hash(model)

        # Save model
        model_path = self.models_path / f"{version_id}.pkl"
        try:
            if JOBLIB_AVAILABLE:
                joblib.dump(model, model_path)
            else:
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

        # Create version info
        ModelVersion(
            version_id=version_id,
            model_name=model_name,
            created_at=timestamp,
            model_hash=model_hash,
            performance_metrics=performance_metrics,
            metadata=metadata or {},
        )

        # Update registry
        if model_name not in self.registry["models"]:
            self.registry["models"][model_name] = {
                "versions": [],
                "active_version": None,
            }

        self.registry["models"][model_name]["versions"].append(version_id)
        self.registry["versions"][version_id] = {
            "model_name": model_name,
            "created_at": timestamp.isoformat(),
            "model_hash": model_hash,
            "performance_metrics": performance_metrics,
            "metadata": metadata or {},
            "is_active": False,
            "model_path": str(model_path),
        }

        self._save_registry()

        logger.info(f"Registered model {model_name} version {version_id}")
        return version_id

    def load_model(self, version_id: str) -> Any:
        """Load a model by version ID."""
        if version_id not in self.registry["versions"]:
            raise ValueError(f"Model version {version_id} not found")

        model_path = Path(self.registry["versions"][version_id]["model_path"])

        try:
            if JOBLIB_AVAILABLE:
                return joblib.load(model_path)
            else:
                with open(model_path, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load model {version_id}: {e}")
            raise

    def set_active_version(self, model_name: str, version_id: str) -> None:
        """Set the active version for a model."""
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model {model_name} not found")

        if version_id not in self.registry["versions"]:
            raise ValueError(f"Version {version_id} not found")

        # Deactivate current active version
        current_active = self.registry["models"][model_name]["active_version"]
        if current_active:
            self.registry["versions"][current_active]["is_active"] = False

        # Set new active version
        self.registry["models"][model_name]["active_version"] = version_id
        self.registry["versions"][version_id]["is_active"] = True

        self._save_registry()
        logger.info(f"Set {version_id} as active version for {model_name}")

    def get_active_model(self, model_name: str) -> Tuple[str, Any]:
        """Get the active model version."""
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model {model_name} not found")

        active_version = self.registry["models"][model_name]["active_version"]
        if not active_version:
            raise ValueError(f"No active version for model {model_name}")

        model = self.load_model(active_version)
        return active_version, model

    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self.registry["models"].keys())

    def list_versions(self, model_name: str) -> List[str]:
        """List all versions for a model."""
        if model_name not in self.registry["models"]:
            return []
        return self.registry["models"][model_name]["versions"]

    def get_version_info(self, version_id: str) -> Dict[str, Any]:
        """Get information about a specific version."""
        if version_id not in self.registry["versions"]:
            raise ValueError(f"Version {version_id} not found")
        return self.registry["versions"][version_id]

    def _calculate_model_hash(self, model: Any) -> str:
        """Calculate hash of model for versioning."""
        try:
            model_bytes = pickle.dumps(model)
            return hashlib.md5(model_bytes).hexdigest()
        except Exception:
            # Fallback to timestamp-based hash
            return hashlib.md5(str(datetime.now()).encode()).hexdigest()


class ABTestManager:
    """A/B testing framework for model deployment."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.active_tests: Dict[str, Dict[str, Any]] = {}
        self.test_results: List[Dict[str, Any]] = []

    def create_ab_test(
        self,
        test_name: str,
        model_a: str,
        model_b: str,
        traffic_split: float = 0.5,
        duration_days: int = 7,
        success_metric: str = "accuracy",
    ) -> str:
        """
        Create an A/B test between two model versions.

        Args:
            test_name: Name of the test
            model_a: Version ID of model A (control)
            model_b: Version ID of model B (treatment)
            traffic_split: Fraction of traffic to send to model B
            duration_days: Duration of test in days
            success_metric: Metric to optimize for

        Returns:
            Test ID
        """
        test_id = f"{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        test_config = {
            "test_id": test_id,
            "test_name": test_name,
            "model_a": model_a,
            "model_b": model_b,
            "traffic_split": traffic_split,
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(days=duration_days),
            "success_metric": success_metric,
            "status": "active",
            "results": {"model_a": [], "model_b": []},
        }

        self.active_tests[test_id] = test_config
        logger.info(f"Created A/B test {test_id}: {model_a} vs {model_b}")

        return test_id

    def route_prediction(self, test_id: str, input_data: Any) -> Tuple[str, Any]:
        """
        Route prediction request to appropriate model in A/B test.

        Args:
            test_id: ID of the A/B test
            input_data: Input data for prediction

        Returns:
            Tuple of (model_version, prediction)
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        test_config = self.active_tests[test_id]

        # Check if test is still active
        if datetime.now() > test_config["end_time"]:
            test_config["status"] = "completed"
            return self._get_winner_model(test_id), None

        # Route based on traffic split
        if np.random.random() < test_config["traffic_split"]:
            model_version = test_config["model_b"]
        else:
            model_version = test_config["model_a"]

        # Load and run model
        model = self.registry.load_model(model_version)
        prediction = model.predict(input_data)

        return model_version, prediction

    def record_result(
        self, test_id: str, model_version: str, prediction: Any, actual: Any
    ) -> None:
        """Record result for A/B test analysis."""
        if test_id not in self.active_tests:
            return

        test_config = self.active_tests[test_id]

        # Calculate metric
        metric_value = self._calculate_metric(
            test_config["success_metric"], actual, prediction
        )

        # Record result
        if model_version == test_config["model_a"]:
            test_config["results"]["model_a"].append(metric_value)
        elif model_version == test_config["model_b"]:
            test_config["results"]["model_b"].append(metric_value)

    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get current results of A/B test."""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        test_config = self.active_tests[test_id]
        results_a = test_config["results"]["model_a"]
        results_b = test_config["results"]["model_b"]

        analysis = {
            "test_id": test_id,
            "status": test_config["status"],
            "model_a": {
                "version": test_config["model_a"],
                "samples": len(results_a),
                "mean_metric": np.mean(results_a) if results_a else 0,
                "std_metric": np.std(results_a) if results_a else 0,
            },
            "model_b": {
                "version": test_config["model_b"],
                "samples": len(results_b),
                "mean_metric": np.mean(results_b) if results_b else 0,
                "std_metric": np.std(results_b) if results_b else 0,
            },
        }

        # Statistical significance test
        if len(results_a) > 10 and len(results_b) > 10:
            try:
                from scipy import stats

                t_stat, p_value = stats.ttest_ind(results_a, results_b)
                analysis["statistical_significance"] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                }
            except ImportError:
                analysis["statistical_significance"] = None

        return analysis

    def _get_winner_model(self, test_id: str) -> str:
        """Determine winner of A/B test."""
        results = self.get_test_results(test_id)

        if results["model_b"]["mean_metric"] > results["model_a"]["mean_metric"]:
            return results["model_b"]["version"]
        else:
            return results["model_a"]["version"]

    def _calculate_metric(
        self, metric_name: str, actual: Any, prediction: Any
    ) -> float:
        """Calculate metric for A/B test."""
        try:
            if metric_name == "accuracy":
                return float(actual == prediction)
            elif metric_name == "mse":
                return float((actual - prediction) ** 2)
            elif metric_name == "mae":
                return float(abs(actual - prediction))
            else:
                return 0.0
        except Exception:
            return 0.0


class ModelMonitor:
    """Real-time model performance monitoring."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.prediction_logs: List[PredictionLog] = []
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.alert_thresholds: Dict[str, Dict[str, float]] = {}
        self.alerts: List[PerformanceAlert] = []

    def set_alert_thresholds(
        self, model_name: str, thresholds: Dict[str, float]
    ) -> None:
        """Set performance alert thresholds for a model."""
        self.alert_thresholds[model_name] = thresholds
        logger.info(f"Set alert thresholds for {model_name}: {thresholds}")

    def log_prediction(
        self,
        model_version: str,
        input_features: Dict[str, Any],
        prediction: Union[float, np.ndarray],
        confidence: Optional[float] = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Log a model prediction."""
        log_entry = PredictionLog(
            timestamp=datetime.now(),
            model_version=model_version,
            input_features=input_features,
            prediction=prediction,
            confidence=confidence,
            metadata=metadata or {},
        )

        self.prediction_logs.append(log_entry)

        # Keep only recent logs (last 10000)
        if len(self.prediction_logs) > 10000:
            self.prediction_logs = self.prediction_logs[-10000:]

        return f"log_{len(self.prediction_logs)}"

    def update_actual_outcome(self, log_id: str, actual_outcome: float) -> None:
        """Update prediction log with actual outcome."""
        try:
            log_index = int(log_id.split("_")[1]) - 1
            if 0 <= log_index < len(self.prediction_logs):
                self.prediction_logs[log_index].actual_outcome = actual_outcome

                # Trigger performance evaluation
                self._evaluate_performance(self.prediction_logs[log_index])
        except (ValueError, IndexError):
            logger.warning(f"Invalid log ID: {log_id}")

    def _evaluate_performance(self, log_entry: PredictionLog) -> None:
        """Evaluate performance and check for alerts."""
        model_version = log_entry.model_version
        version_info = self.registry.get_version_info(model_version)
        model_name = version_info["model_name"]

        # Calculate recent performance
        recent_logs = [
            log
            for log in self.prediction_logs[-100:]  # Last 100 predictions
            if log.model_version == model_version and log.actual_outcome is not None
        ]

        if len(recent_logs) < 10:  # Need minimum samples
            return

        # Calculate metrics
        predictions = [log.prediction for log in recent_logs]
        actuals = [log.actual_outcome for log in recent_logs]

        metrics = self._calculate_performance_metrics(actuals, predictions)

        # Store performance history
        if model_version not in self.performance_history:
            self.performance_history[model_version] = []

        self.performance_history[model_version].append(
            {
                "timestamp": datetime.now(),
                "metrics": metrics,
                "sample_size": len(recent_logs),
            }
        )

        # Check for alerts
        self._check_alerts(model_name, model_version, metrics)

    def _calculate_performance_metrics(
        self, actuals: List[float], predictions: List[float]
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        actuals = np.array(actuals)
        predictions = np.array(predictions)

        metrics = {}

        try:
            # Basic metrics
            metrics["mse"] = np.mean((actuals - predictions) ** 2)
            metrics["mae"] = np.mean(np.abs(actuals - predictions))
            metrics["rmse"] = np.sqrt(metrics["mse"])

            # R-squared
            ss_res = np.sum((actuals - predictions) ** 2)
            ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
            metrics["r2"] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            # Directional accuracy
            if len(actuals) > 1:
                actual_direction = np.sign(np.diff(actuals))
                pred_direction = np.sign(np.diff(predictions))
                metrics["directional_accuracy"] = np.mean(
                    actual_direction == pred_direction
                )

            # Information coefficient
            correlation = np.corrcoef(actuals, predictions)[0, 1]
            metrics["information_coefficient"] = (
                correlation if not np.isnan(correlation) else 0
            )

        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")

        return metrics

    def _check_alerts(
        self, model_name: str, model_version: str, metrics: Dict[str, float]
    ) -> None:
        """Check if any alert thresholds are breached."""
        if model_name not in self.alert_thresholds:
            return

        thresholds = self.alert_thresholds[model_name]
        breached = {}

        for metric, threshold in thresholds.items():
            if metric in metrics:
                current_value = metrics[metric]

                # Check if threshold is breached
                if metric in ["mse", "mae", "rmse"]:  # Lower is better
                    if current_value > threshold:
                        breached[metric] = current_value
                else:  # Higher is better
                    if current_value < threshold:
                        breached[metric] = current_value

        if breached:
            # Determine severity
            severity = "medium"
            if len(breached) > 2:
                severity = "high"
            if any(
                abs(metrics[m] - thresholds[m]) / thresholds[m] > 0.5 for m in breached
            ):
                severity = "critical"

            alert = PerformanceAlert(
                alert_id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                alert_type="performance_degradation",
                severity=severity,
                message=f"Performance degradation detected for {model_name}",
                metrics=metrics,
                threshold_breached=breached,
                model_version=model_version,
            )

            self.alerts.append(alert)
            logger.warning(f"Performance alert: {alert.message}")

    def get_model_performance(
        self, model_version: str, hours_back: int = 24
    ) -> Dict[str, Any]:
        """Get performance metrics for a model version."""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        # Get recent performance history
        if model_version in self.performance_history:
            recent_history = [
                entry
                for entry in self.performance_history[model_version]
                if entry["timestamp"] > cutoff_time
            ]
        else:
            recent_history = []

        # Get recent prediction logs
        recent_logs = [
            log
            for log in self.prediction_logs
            if (
                log.model_version == model_version
                and log.timestamp > cutoff_time
                and log.actual_outcome is not None
            )
        ]

        if not recent_logs:
            return {
                "model_version": model_version,
                "period_hours": hours_back,
                "predictions_count": 0,
                "performance_metrics": {},
                "performance_trend": [],
            }

        # Calculate overall metrics
        predictions = [log.prediction for log in recent_logs]
        actuals = [log.actual_outcome for log in recent_logs]
        overall_metrics = self._calculate_performance_metrics(actuals, predictions)

        return {
            "model_version": model_version,
            "period_hours": hours_back,
            "predictions_count": len(recent_logs),
            "performance_metrics": overall_metrics,
            "performance_trend": recent_history,
            "latest_predictions": recent_logs[-10:] if recent_logs else [],
        }

    def get_active_alerts(
        self, severity: Optional[str] = None
    ) -> List[PerformanceAlert]:
        """Get active performance alerts."""
        alerts = self.alerts

        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]

        # Return recent alerts (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        return [alert for alert in alerts if alert.timestamp > cutoff_time]


class AutoRetrainer:
    """Automated model retraining system."""

    def __init__(self, registry: ModelRegistry, monitor: ModelMonitor):
        self.registry = registry
        self.monitor = monitor
        self.retraining_configs: Dict[str, Dict[str, Any]] = {}
        self.retraining_history: List[Dict[str, Any]] = []

    def configure_retraining(
        self,
        model_name: str,
        trigger_conditions: Dict[str, Any],
        retraining_pipeline: Callable,
        validation_threshold: float = 0.05,
    ) -> None:
        """
        Configure automated retraining for a model.

        Args:
            model_name: Name of the model
            trigger_conditions: Conditions that trigger retraining
            retraining_pipeline: Function that retrains the model
            validation_threshold: Minimum improvement required for deployment
        """
        self.retraining_configs[model_name] = {
            "trigger_conditions": trigger_conditions,
            "retraining_pipeline": retraining_pipeline,
            "validation_threshold": validation_threshold,
            "last_retrain": None,
            "enabled": True,
        }

        logger.info(f"Configured automated retraining for {model_name}")

    def check_retraining_triggers(self, model_name: str) -> bool:
        """Check if retraining should be triggered."""
        if model_name not in self.retraining_configs:
            return False

        config = self.retraining_configs[model_name]
        if not config["enabled"]:
            return False

        triggers = config["trigger_conditions"]

        # Get current active model version
        try:
            active_version, _ = self.registry.get_active_model(model_name)
            performance = self.monitor.get_model_performance(
                active_version, hours_back=24
            )
        except Exception:
            return False

        # Check performance degradation
        if "performance_threshold" in triggers:
            threshold = triggers["performance_threshold"]
            current_metrics = performance["performance_metrics"]

            for metric, min_value in threshold.items():
                if metric in current_metrics:
                    if current_metrics[metric] < min_value:
                        logger.info(
                            f"Retraining triggered for {model_name}: {metric} below threshold"
                        )
                        return True

        # Check time-based triggers
        if "max_age_days" in triggers:
            max_age = triggers["max_age_days"]
            last_retrain = config["last_retrain"]

            if last_retrain is None:
                # Use model creation date
                version_info = self.registry.get_version_info(active_version)
                last_retrain = datetime.fromisoformat(version_info["created_at"])

            if (datetime.now() - last_retrain).days > max_age:
                logger.info(
                    f"Retraining triggered for {model_name}: model age exceeded"
                )
                return True

        # Check data drift (simplified)
        if (
            "data_drift_threshold" in triggers
            and performance["predictions_count"] > 100
        ):
            # Simple drift detection based on prediction distribution
            recent_predictions = [
                log.prediction for log in performance["latest_predictions"]
            ]
            if len(recent_predictions) > 10:
                pred_std = np.std(recent_predictions)
                if pred_std > triggers["data_drift_threshold"]:
                    logger.info(
                        f"Retraining triggered for {model_name}: data drift detected"
                    )
                    return True

        return False

    def trigger_retraining(self, model_name: str) -> Optional[str]:
        """Trigger automated retraining for a model."""
        if model_name not in self.retraining_configs:
            logger.error(f"No retraining config for {model_name}")
            return None

        config = self.retraining_configs[model_name]
        retraining_pipeline = config["retraining_pipeline"]

        try:
            logger.info(f"Starting automated retraining for {model_name}")

            # Run retraining pipeline
            new_model, performance_metrics, training_data = retraining_pipeline()

            # Register new model version
            new_version_id = self.registry.register_model(
                model_name=model_name,
                model=new_model,
                performance_metrics=performance_metrics,
                metadata={
                    "retraining_trigger": "automated",
                    "training_timestamp": datetime.now().isoformat(),
                    "training_data_size": (
                        len(training_data)
                        if hasattr(training_data, "__len__")
                        else "unknown"
                    ),
                },
            )

            # Validate new model performance
            if self._validate_new_model(model_name, new_version_id):
                # Deploy new model
                self.registry.set_active_version(model_name, new_version_id)
                config["last_retrain"] = datetime.now()

                logger.info(
                    f"Successfully retrained and deployed {model_name} version {new_version_id}"
                )

                # Record retraining event
                self.retraining_history.append(
                    {
                        "timestamp": datetime.now(),
                        "model_name": model_name,
                        "new_version": new_version_id,
                        "trigger": "automated",
                        "success": True,
                        "performance_metrics": performance_metrics,
                    }
                )

                return new_version_id
            else:
                logger.warning(
                    f"New model version {new_version_id} did not meet validation criteria"
                )
                return None

        except Exception as e:
            logger.error(f"Retraining failed for {model_name}: {e}")

            # Record failed retraining
            self.retraining_history.append(
                {
                    "timestamp": datetime.now(),
                    "model_name": model_name,
                    "new_version": None,
                    "trigger": "automated",
                    "success": False,
                    "error": str(e),
                }
            )

            return None

    def _validate_new_model(self, model_name: str, new_version_id: str) -> bool:
        """Validate that new model meets deployment criteria."""
        try:
            # Get current active model performance
            active_version, _ = self.registry.get_active_model(model_name)
            current_performance = self.registry.get_version_info(active_version)[
                "performance_metrics"
            ]

            # Get new model performance
            new_performance = self.registry.get_version_info(new_version_id)[
                "performance_metrics"
            ]

            # Check if new model is significantly better
            threshold = self.retraining_configs[model_name]["validation_threshold"]

            # Compare key metrics (assuming higher is better for most metrics)
            for metric in ["accuracy", "r2", "information_coefficient"]:
                if metric in current_performance and metric in new_performance:
                    improvement = new_performance[metric] - current_performance[metric]
                    if improvement > threshold:
                        return True

            # For error metrics (lower is better)
            for metric in ["mse", "mae", "rmse"]:
                if metric in current_performance and metric in new_performance:
                    improvement = current_performance[metric] - new_performance[metric]
                    relative_improvement = improvement / current_performance[metric]
                    if relative_improvement > threshold:
                        return True

            return False

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False

    def get_retraining_history(
        self, model_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get retraining history."""
        if model_name:
            return [
                event
                for event in self.retraining_history
                if event["model_name"] == model_name
            ]
        return self.retraining_history


class ModelExplainer:
    """Model explainability and interpretability tools."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def explain_prediction(
        self,
        model_version: str,
        input_features: Dict[str, Any],
        method: str = "feature_importance",
    ) -> Dict[str, Any]:
        """
        Explain a model prediction.

        Args:
            model_version: Version ID of the model
            input_features: Input features for the prediction
            method: Explanation method ('feature_importance', 'shap', 'lime')

        Returns:
            Explanation results
        """
        model = self.registry.load_model(model_version)

        if method == "feature_importance":
            return self._feature_importance_explanation(model, input_features)
        elif method == "shap":
            return self._shap_explanation(model, input_features)
        elif method == "lime":
            return self._lime_explanation(model, input_features)
        else:
            raise ValueError(f"Unknown explanation method: {method}")

    def _feature_importance_explanation(
        self, model: Any, input_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Basic feature importance explanation."""
        try:
            # For tree-based models
            if hasattr(model, "feature_importances_"):
                feature_names = list(input_features.keys())
                importances = model.feature_importances_

                feature_importance = dict(zip(feature_names, importances))

                return {
                    "method": "feature_importance",
                    "feature_importance": feature_importance,
                    "top_features": sorted(
                        feature_importance.items(), key=lambda x: x[1], reverse=True
                    )[:5],
                }

            # For linear models
            elif hasattr(model, "coef_"):
                feature_names = list(input_features.keys())
                coefficients = model.coef_

                if len(coefficients.shape) > 1:
                    coefficients = coefficients[0]  # Take first class for multi-class

                feature_importance = dict(zip(feature_names, np.abs(coefficients)))

                return {
                    "method": "linear_coefficients",
                    "coefficients": dict(zip(feature_names, coefficients)),
                    "feature_importance": feature_importance,
                    "top_features": sorted(
                        feature_importance.items(), key=lambda x: x[1], reverse=True
                    )[:5],
                }

            else:
                return {
                    "method": "feature_importance",
                    "error": "Model does not support feature importance",
                    "model_type": type(model).__name__,
                }

        except Exception as e:
            return {"method": "feature_importance", "error": str(e)}

    def _shap_explanation(
        self, model: Any, input_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """SHAP-based explanation (requires shap library)."""
        try:
            import shap

            # Convert input to array
            feature_names = list(input_features.keys())
            input_array = np.array([list(input_features.values())])

            # Create explainer based on model type
            if hasattr(model, "predict_proba"):
                explainer = shap.Explainer(model.predict_proba, input_array)
            else:
                explainer = shap.Explainer(model.predict, input_array)

            # Calculate SHAP values
            shap_values = explainer(input_array)

            # Extract values for single prediction
            if len(shap_values.shape) > 2:
                values = shap_values[0, :, 0]  # First sample, all features, first class
            else:
                values = shap_values[0, :]  # First sample, all features

            feature_contributions = dict(zip(feature_names, values))

            return {
                "method": "shap",
                "feature_contributions": feature_contributions,
                "top_positive": sorted(
                    [(k, v) for k, v in feature_contributions.items() if v > 0],
                    key=lambda x: x[1],
                    reverse=True,
                )[:3],
                "top_negative": sorted(
                    [(k, v) for k, v in feature_contributions.items() if v < 0],
                    key=lambda x: x[1],
                )[:3],
            }

        except ImportError:
            return {
                "method": "shap",
                "error": "SHAP library not available. Install with: pip install shap",
            }
        except Exception as e:
            return {"method": "shap", "error": str(e)}

    def _lime_explanation(
        self, model: Any, input_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """LIME-based explanation (requires lime library)."""
        try:
            import lime  # noqa: F401

            # This is a simplified implementation
            # In practice, you'd need training data for LIME
            return {
                "method": "lime",
                "error": "LIME explanation requires training data. Use SHAP instead.",
            }

        except ImportError:
            return {
                "method": "lime",
                "error": "LIME library not available. Install with: pip install lime",
            }
        except Exception as e:
            return {"method": "lime", "error": str(e)}

    def generate_model_report(self, model_version: str) -> Dict[str, Any]:
        """Generate comprehensive model interpretability report."""
        version_info = self.registry.get_version_info(model_version)
        model = self.registry.load_model(model_version)

        report = {
            "model_version": model_version,
            "model_name": version_info["model_name"],
            "created_at": version_info["created_at"],
            "model_type": type(model).__name__,
            "performance_metrics": version_info["performance_metrics"],
        }

        # Add model-specific interpretability information
        if hasattr(model, "feature_importances_"):
            report["has_feature_importance"] = True
            report["feature_count"] = len(model.feature_importances_)

        if hasattr(model, "coef_"):
            report["has_coefficients"] = True
            report["is_linear_model"] = True

        # Add complexity metrics
        if hasattr(model, "n_estimators"):
            report["n_estimators"] = model.n_estimators

        if hasattr(model, "max_depth"):
            report["max_depth"] = model.max_depth

        return report


class ModelDeploymentPipeline:
    """Complete model deployment pipeline orchestrator."""

    def __init__(self, registry_path: str = "model_registry"):
        self.registry = ModelRegistry(registry_path)
        self.ab_test_manager = ABTestManager(self.registry)
        self.monitor = ModelMonitor(self.registry)
        self.auto_retrainer = AutoRetrainer(self.registry, self.monitor)
        self.explainer = ModelExplainer(self.registry)

    def deploy_model(
        self,
        model_name: str,
        model: Any,
        performance_metrics: Dict[str, float],
        deployment_config: Dict[str, Any] = None,
    ) -> str:
        """
        Deploy a model with full monitoring and management setup.

        Args:
            model_name: Name of the model
            model: The trained model
            performance_metrics: Performance metrics
            deployment_config: Deployment configuration

        Returns:
            Version ID of deployed model
        """
        # Register model
        version_id = self.registry.register_model(
            model_name=model_name,
            model=model,
            performance_metrics=performance_metrics,
            metadata={"deployment_config": deployment_config or {}},
        )

        # Set as active version
        self.registry.set_active_version(model_name, version_id)

        # Set up monitoring thresholds
        if deployment_config and "monitoring_thresholds" in deployment_config:
            self.monitor.set_alert_thresholds(
                model_name, deployment_config["monitoring_thresholds"]
            )

        # Set up automated retraining
        if deployment_config and "auto_retrain" in deployment_config:
            retrain_config = deployment_config["auto_retrain"]
            if "retraining_pipeline" in retrain_config:
                self.auto_retrainer.configure_retraining(
                    model_name=model_name,
                    trigger_conditions=retrain_config.get("triggers", {}),
                    retraining_pipeline=retrain_config["retraining_pipeline"],
                    validation_threshold=retrain_config.get(
                        "validation_threshold", 0.05
                    ),
                )

        logger.info(f"Successfully deployed model {model_name} version {version_id}")
        return version_id

    def predict_with_monitoring(
        self,
        model_name: str,
        input_features: Dict[str, Any],
        log_prediction: bool = True,
    ) -> Dict[str, Any]:
        """Make prediction with full monitoring and logging."""
        # Get active model
        version_id, model = self.registry.get_active_model(model_name)

        # Make prediction
        if hasattr(model, "predict_proba"):
            prediction = model.predict_proba([list(input_features.values())])[0]
            confidence = np.max(prediction)
            prediction = np.argmax(prediction)
        else:
            prediction = model.predict([list(input_features.values())])[0]
            confidence = None

        # Log prediction
        log_id = None
        if log_prediction:
            log_id = self.monitor.log_prediction(
                model_version=version_id,
                input_features=input_features,
                prediction=prediction,
                confidence=confidence,
            )

        # Check for retraining triggers
        if self.auto_retrainer.check_retraining_triggers(model_name):
            # Trigger retraining in background (in practice, use async/celery)
            logger.info(f"Retraining triggered for {model_name}")

        return {
            "prediction": prediction,
            "confidence": confidence,
            "model_version": version_id,
            "log_id": log_id,
            "timestamp": datetime.now(),
        }

    def get_deployment_status(self, model_name: str) -> Dict[str, Any]:
        """Get comprehensive deployment status for a model."""
        try:
            version_id, _ = self.registry.get_active_model(model_name)
            version_info = self.registry.get_version_info(version_id)
            performance = self.monitor.get_model_performance(version_id)
            alerts = self.monitor.get_active_alerts()

            # Filter alerts for this model
            model_alerts = [
                alert for alert in alerts if alert.model_version == version_id
            ]

            return {
                "model_name": model_name,
                "active_version": version_id,
                "deployment_date": version_info["created_at"],
                "performance_metrics": version_info["performance_metrics"],
                "recent_performance": performance,
                "active_alerts": len(model_alerts),
                "alert_details": model_alerts,
                "retraining_configured": model_name
                in self.auto_retrainer.retraining_configs,
                "status": "healthy" if len(model_alerts) == 0 else "degraded",
            }

        except Exception as e:
            return {"model_name": model_name, "status": "error", "error": str(e)}
