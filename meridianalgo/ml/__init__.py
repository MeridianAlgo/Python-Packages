"""
Machine learning module for MeridianAlgo.

Unified access to forecasting models, validation frameworks, and feature engineering.
"""

from .core import (
                   EnsemblePredictor,
                   FeatureEngineer,
                   LSTMPredictor,
                   ModelEvaluator,
                   create_ml_models,
                   prepare_data_for_lstm,
)
from .deployment import (
                   AutoRetrainer,
                   ModelDeploymentPipeline,
                   ModelMonitor,
                   ModelRegistry,
)
from .feature_engineering import (
                   BaseFeatureGenerator,
                   ComprehensiveFeatureEngineer,
                   FeatureConfig,
                   FeatureSelector,
)

# Import from the unified ml models directory
from .models import (
                   GRUModel,
                   LSTMModel,
                   ModelConfig,
                   ModelFactory,
                   ModelTrainer,
                   TraditionalMLModel,
                   TransformerModel,
)
from .validation import (
                   CombinatorialPurgedCV,
                   ModelSelector,
                   PurgedCrossValidator,
                   TimeSeriesValidator,
                   WalkForwardValidator,
)

# Aliases for backward compatibility and institutional standards
WalkForwardOptimizer = WalkForwardValidator
TimeSeriesCV = PurgedCrossValidator
FeatureGenerator = BaseFeatureGenerator
TechnicalFeatureEngineer = ComprehensiveFeatureEngineer
ModelDeploymentHandler = ModelDeploymentPipeline  # Alias

__all__ = [
    "FeatureEngineer",
    "LSTMPredictor",
    "EnsemblePredictor",
    "ModelEvaluator",
    "prepare_data_for_lstm",
    "create_ml_models",
    "WalkForwardOptimizer",
    "TimeSeriesCV",
    "ModelSelector",
    "ModelTrainer",
    "ModelFactory",
    "ModelConfig",
    "TransformerModel",
    "LSTMModel",
    "GRUModel",
    "TraditionalMLModel",
    "FeatureGenerator",
    "TechnicalFeatureEngineer",
    "ModelDeploymentPipeline",
    "ModelDeploymentHandler",
    "ModelRegistry",
    "ModelMonitor",
    "AutoRetrainer",
    "ComprehensiveFeatureEngineer",
    "FeatureConfig",
    "FeatureSelector",
    "WalkForwardValidator",
    "PurgedCrossValidator",
    "TimeSeriesValidator",
    "CombinatorialPurgedCV",
]
