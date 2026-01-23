"""
Machine learning module for MeridianAlgo.
"""

from .core import (
    EnsemblePredictor,
    FeatureEngineer,
    LSTMPredictor,
    ModelEvaluator,
    create_ml_models,
    prepare_data_for_lstm,
)

# Import from the newer machine_learning directory if helpful,
# or provide aliases for things expected by the top-level __init__.py
try:
    from ..machine_learning.models import (  # noqa: F401
        GRUModel,
        LSTMModel,
        ModelFactory,
        ModelTrainer,
        TraditionalMLModel,
        TransformerModel,
    )
    from ..machine_learning.validation import (  # noqa: F401
        CombinatorialPurgedCV,
        ModelSelector,
        PurgedCrossValidator,
        TimeSeriesValidator,
        WalkForwardValidator,
    )

    # Aliases
    WalkForwardOptimizer = WalkForwardValidator
    TimeSeriesCV = PurgedCrossValidator  # Or WalkForwardValidator
except ImportError:
    pass

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
]
