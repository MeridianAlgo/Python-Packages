"""
Machine learning module for MeridianAlgo.
"""

from .core import (
    FeatureEngineer,
    LSTMPredictor,
    EnsemblePredictor,
    ModelEvaluator,
    prepare_data_for_lstm,
    create_ml_models
)

# Import from the newer machine_learning directory if helpful, 
# or provide aliases for things expected by the top-level __init__.py
try:
    from ..machine_learning.models import (
        LSTMModel, GRUModel, TransformerModel, 
        TraditionalMLModel, ModelFactory, ModelTrainer
    )
    from ..machine_learning.validation import (
        WalkForwardValidator, PurgedCrossValidator, 
        CombinatorialPurgedCV, TimeSeriesValidator,
        ModelSelector
    )
    # Aliases
    WalkForwardOptimizer = WalkForwardValidator
    TimeSeriesCV = PurgedCrossValidator  # Or WalkForwardValidator
except ImportError:
    pass

__all__ = [
    'FeatureEngineer',
    'LSTMPredictor',
    'EnsemblePredictor',
    'ModelEvaluator',
    'prepare_data_for_lstm',
    'create_ml_models',
    'WalkForwardOptimizer',
    'TimeSeriesCV',
    'ModelSelector'
]
