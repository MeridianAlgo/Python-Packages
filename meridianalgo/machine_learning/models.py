"""
Advanced ML models for financial time series prediction and trading strategies.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Deep learning models will be limited.")

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Traditional ML models will be limited.")

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available.")

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available.")

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for ML models."""

    model_type: str
    hyperparameters: Dict[str, Any]
    training_config: Dict[str, Any]
    validation_config: Dict[str, Any]


@dataclass
class ModelResult:
    """Result from model training/prediction."""

    predictions: np.ndarray
    actual: np.ndarray
    metrics: Dict[str, float]
    model_info: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]] = None


class BaseFinancialModel(ABC):
    """Abstract base class for financial ML models."""

    def __init__(self, name: str, config: ModelConfig = None):
        self.name = name
        self.config = config
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.feature_names = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "BaseFinancialModel":
        """Fit the model to training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        predictions = self.predict(X)

        metrics = {
            "mse": mean_squared_error(y, predictions),
            "mae": mean_absolute_error(y, predictions),
            "rmse": np.sqrt(mean_squared_error(y, predictions)),
            "r2": r2_score(y, predictions),
        }

        # Financial-specific metrics
        if len(predictions) > 1:
            # Directional accuracy
            actual_direction = np.sign(y[1:] - y[:-1])
            pred_direction = np.sign(predictions[1:] - predictions[:-1])
            directional_accuracy = np.mean(actual_direction == pred_direction)
            metrics["directional_accuracy"] = directional_accuracy

            # Information coefficient (correlation between predictions and actual)
            ic = (
                np.corrcoef(predictions, y)[0, 1]
                if not np.isnan(np.corrcoef(predictions, y)[0, 1])
                else 0.0
            )
            metrics["information_coefficient"] = ic

        return metrics

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        return None


class LSTMModel(BaseFinancialModel):
    """LSTM model for time series prediction."""

    def __init__(self, config: ModelConfig = None):
        super().__init__("LSTM", config)

        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTM model")

        # Default hyperparameters
        self.hidden_size = (
            config.hyperparameters.get("hidden_size", 64) if config else 64
        )
        self.num_layers = config.hyperparameters.get("num_layers", 2) if config else 2
        self.dropout = config.hyperparameters.get("dropout", 0.2) if config else 0.2
        self.sequence_length = (
            config.hyperparameters.get("sequence_length", 20) if config else 20
        )

        # Training parameters
        self.epochs = config.training_config.get("epochs", 100) if config else 100
        self.batch_size = config.training_config.get("batch_size", 32) if config else 32
        self.learning_rate = (
            config.training_config.get("learning_rate", 0.001) if config else 0.001
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_sequences(
        self, data: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []

        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length : i])
            y.append(target[i])

        return np.array(X), np.array(y)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "LSTMModel":
        """Fit LSTM model."""

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y)

        if len(X_seq) == 0:
            raise ValueError("Insufficient data for sequence creation")

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)

        # Create model
        input_size = X_seq.shape[2]
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with LSTM."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))

        if len(X_seq) == 0:
            # Return zeros if insufficient data
            return np.zeros(len(X))

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().squeeze()

        # Pad predictions to match input length
        full_predictions = np.zeros(len(X))
        full_predictions[
            self.sequence_length : self.sequence_length + len(predictions)
        ] = predictions

        return full_predictions


class LSTMNetwork(nn.Module):
    """LSTM neural network architecture."""

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2
    ):
        super(LSTMNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Take the last output
        last_output = lstm_out[:, -1, :]

        # Apply dropout and final layer
        output = self.dropout(last_output)
        output = self.fc(output)

        return output


class GRUModel(BaseFinancialModel):
    """GRU model for time series prediction."""

    def __init__(self, config: ModelConfig = None):
        super().__init__("GRU", config)

        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GRU model")

        # Similar to LSTM but with GRU
        self.hidden_size = (
            config.hyperparameters.get("hidden_size", 64) if config else 64
        )
        self.num_layers = config.hyperparameters.get("num_layers", 2) if config else 2
        self.dropout = config.hyperparameters.get("dropout", 0.2) if config else 0.2
        self.sequence_length = (
            config.hyperparameters.get("sequence_length", 20) if config else 20
        )

        self.epochs = config.training_config.get("epochs", 100) if config else 100
        self.batch_size = config.training_config.get("batch_size", 32) if config else 32
        self.learning_rate = (
            config.training_config.get("learning_rate", 0.001) if config else 0.001
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_sequences(
        self, data: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for GRU training."""
        X, y = [], []

        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length : i])
            y.append(target[i])

        return np.array(X), np.array(y)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "GRUModel":
        """Fit GRU model."""

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y)

        if len(X_seq) == 0:
            raise ValueError("Insufficient data for sequence creation")

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)

        # Create model
        input_size = X_seq.shape[2]
        self.model = GRUNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with GRU."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))

        if len(X_seq) == 0:
            return np.zeros(len(X))

        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().squeeze()

        full_predictions = np.zeros(len(X))
        full_predictions[
            self.sequence_length : self.sequence_length + len(predictions)
        ] = predictions

        return full_predictions


class GRUNetwork(nn.Module):
    """GRU neural network architecture."""

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2
    ):
        super(GRUNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        output = self.dropout(last_output)
        output = self.fc(output)
        return output


class TransformerModel(BaseFinancialModel):
    """Transformer model for sequence modeling."""

    def __init__(self, config: ModelConfig = None):
        super().__init__("Transformer", config)

        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Transformer model")

        # Transformer hyperparameters
        self.d_model = config.hyperparameters.get("d_model", 64) if config else 64
        self.nhead = config.hyperparameters.get("nhead", 8) if config else 8
        self.num_layers = config.hyperparameters.get("num_layers", 3) if config else 3
        self.dropout = config.hyperparameters.get("dropout", 0.1) if config else 0.1
        self.sequence_length = (
            config.hyperparameters.get("sequence_length", 50) if config else 50
        )

        self.epochs = config.training_config.get("epochs", 100) if config else 100
        self.batch_size = config.training_config.get("batch_size", 32) if config else 32
        self.learning_rate = (
            config.training_config.get("learning_rate", 0.001) if config else 0.001
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_sequences(
        self, data: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for Transformer training."""
        X, y = [], []

        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length : i])
            y.append(target[i])

        return np.array(X), np.array(y)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "TransformerModel":
        """Fit Transformer model."""

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        X_seq, y_seq = self._create_sequences(X_scaled, y)

        if len(X_seq) == 0:
            raise ValueError("Insufficient data for sequence creation")

        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)

        input_size = X_seq.shape[2]
        self.model = TransformerNetwork(
            input_size=input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
            sequence_length=self.sequence_length,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with Transformer."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))

        if len(X_seq) == 0:
            return np.zeros(len(X))

        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().squeeze()

        full_predictions = np.zeros(len(X))
        full_predictions[
            self.sequence_length : self.sequence_length + len(predictions)
        ] = predictions

        return full_predictions


class TransformerNetwork(nn.Module):
    """Transformer neural network architecture."""

    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        sequence_length: int,
    ):
        super(TransformerNetwork, self).__init__()

        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout, sequence_length)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoding
        transformer_out = self.transformer(x)

        # Take the last output
        last_output = transformer_out[:, -1, :]

        # Final prediction
        output = self.dropout(last_output)
        output = self.fc(output)

        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class EnsembleModel(BaseFinancialModel):
    """Ensemble model combining multiple model types."""

    def __init__(self, models: List[BaseFinancialModel], weights: List[float] = None):
        super().__init__("Ensemble")
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "EnsembleModel":
        """Fit all models in the ensemble."""

        for i, model in enumerate(self.models):
            logger.info(f"Training model {i + 1}/{len(self.models)}: {model.name}")
            model.fit(X, y, **kwargs)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        predictions = []

        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        # Weighted average
        ensemble_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            ensemble_pred += self.weights[i] * pred

        return ensemble_pred

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get averaged feature importance from models that support it."""
        importances = []

        for model in self.models:
            importance = model.get_feature_importance()
            if importance:
                importances.append(importance)

        if not importances:
            return None

        # Average importances
        all_features = set()
        for imp in importances:
            all_features.update(imp.keys())

        avg_importance = {}
        for feature in all_features:
            values = [imp.get(feature, 0) for imp in importances]
            avg_importance[feature] = np.mean(values)

        return avg_importance


class TraditionalMLModel(BaseFinancialModel):
    """Wrapper for traditional ML models from scikit-learn."""

    def __init__(self, model_type: str, config: ModelConfig = None):
        super().__init__(f"Traditional_{model_type}", config)

        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for traditional ML models")

        self.model_type = model_type
        self._create_model()

    def _create_model(self):
        """Create the underlying model."""
        hyperparams = self.config.hyperparameters if self.config else {}

        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(**hyperparams)
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(**hyperparams)
        elif self.model_type == "linear":
            self.model = LinearRegression(**hyperparams)
        elif self.model_type == "ridge":
            self.model = Ridge(**hyperparams)
        elif self.model_type == "lasso":
            self.model = Lasso(**hyperparams)
        elif self.model_type == "elastic_net":
            self.model = ElasticNet(**hyperparams)
        elif self.model_type == "svr":
            self.model = SVR(**hyperparams)
        elif self.model_type == "xgboost" and XGBOOST_AVAILABLE:
            self.model = xgb.XGBRegressor(**hyperparams)
        elif self.model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMRegressor(**hyperparams)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "TraditionalMLModel":
        """Fit the traditional ML model."""

        # Scale features for some models
        if self.model_type in ["svr", "linear", "ridge", "lasso", "elastic_net"]:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        self.model.fit(X_scaled, y)
        self.is_fitted = True
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        if not self.is_fitted:
            return None

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
            return dict(zip(self.feature_names, importance))
        elif hasattr(self.model, "coef_"):
            # For linear models, use absolute coefficients
            importance = np.abs(self.model.coef_)
            return dict(zip(self.feature_names, importance))
        else:
            return None


class ModelFactory:
    """Factory for creating ML models."""

    @staticmethod
    def create_model(model_type: str, config: ModelConfig = None) -> BaseFinancialModel:
        """Create a model of the specified type."""

        if model_type == "lstm":
            return LSTMModel(config)
        elif model_type == "gru":
            return GRUModel(config)
        elif model_type == "transformer":
            return TransformerModel(config)
        elif model_type in [
            "random_forest",
            "gradient_boosting",
            "linear",
            "ridge",
            "lasso",
            "elastic_net",
            "svr",
            "xgboost",
            "lightgbm",
        ]:
            return TraditionalMLModel(model_type, config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def create_ensemble(
        model_configs: List[Tuple[str, ModelConfig]], weights: List[float] = None
    ) -> EnsembleModel:
        """Create an ensemble model."""

        models = []
        for model_type, config in model_configs:
            model = ModelFactory.create_model(model_type, config)
            models.append(model)

        return EnsembleModel(models, weights)


class ModelTrainer:
    """Utility class for training and evaluating models."""

    def __init__(self):
        self.trained_models = {}
        self.evaluation_results = {}

    def train_model(
        self,
        model: BaseFinancialModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> ModelResult:
        """Train a model and evaluate it."""

        # Train the model
        logger.info(f"Training {model.name} model...")
        model.fit(X_train, y_train)

        # Make predictions
        train_pred = model.predict(X_train)

        # Evaluate on training set
        train_metrics = model.evaluate(X_train, y_train)

        # Evaluate on validation set if provided
        val_metrics = {}
        val_pred = None
        if X_val is not None and y_val is not None:
            val_pred = model.predict(X_val)
            val_metrics = model.evaluate(X_val, y_val)

        # Get feature importance
        feature_importance = model.get_feature_importance()

        # Create result
        result = ModelResult(
            predictions=val_pred if val_pred is not None else train_pred,
            actual=y_val if y_val is not None else y_train,
            metrics={"train": train_metrics, "validation": val_metrics},
            model_info={
                "name": model.name,
                "type": type(model).__name__,
                "is_fitted": model.is_fitted,
            },
            feature_importance=feature_importance,
        )

        # Store results
        self.trained_models[model.name] = model
        self.evaluation_results[model.name] = result

        logger.info(f"Training completed for {model.name}")
        if val_metrics:
            logger.info(f"Validation R: {val_metrics.get('r2', 0):.4f}")

        return result

    def compare_models(self, results: List[ModelResult]) -> pd.DataFrame:
        """Compare multiple model results."""

        comparison_data = []

        for result in results:
            model_name = result.model_info["name"]

            # Get validation metrics if available, otherwise training metrics
            metrics = result.metrics.get("validation", result.metrics.get("train", {}))

            comparison_data.append(
                {
                    "Model": model_name,
                    "R": metrics.get("r2", 0),
                    "RMSE": metrics.get("rmse", 0),
                    "MAE": metrics.get("mae", 0),
                    "Directional_Accuracy": metrics.get("directional_accuracy", 0),
                    "Hit_Rate": metrics.get("hit_rate", 0),
                }
            )

        return pd.DataFrame(comparison_data).sort_values("R", ascending=False)


# Reinforcement Learning for Trading Strategies
class TradingEnvironment:
    """Trading environment for reinforcement learning."""

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,
    ):
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reset()

    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 0: no position, 1: long, -1: short
        self.portfolio_value = self.initial_balance
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        """Get current observation."""
        if self.current_step >= len(self.data):
            return np.zeros(
                self.data.shape[1] + 3
            )  # +3 for balance, position, portfolio_value

        market_data = self.data.iloc[self.current_step].values
        portfolio_data = np.array([self.balance, self.position, self.portfolio_value])
        return np.concatenate([market_data, portfolio_data])

    def step(self, action):
        """Execute action and return next state, reward, done."""
        if self.current_step >= len(self.data) - 1:
            self.done = True
            return self._get_observation(), 0, self.done, {}

        current_price = self.data.iloc[self.current_step]["Close"]
        next_price = self.data.iloc[self.current_step + 1]["Close"]

        # Actions: 0=hold, 1=buy, 2=sell
        reward = 0

        if action == 1 and self.position <= 0:  # Buy
            if self.position == -1:  # Close short position
                profit = (current_price - next_price) * abs(self.position)
                self.balance += profit - (self.transaction_cost * current_price)

            # Open long position
            shares = self.balance / current_price
            self.position = shares
            self.balance = 0

        elif action == 2 and self.position >= 0:  # Sell
            if self.position > 0:  # Close long position
                profit = (next_price - current_price) * self.position
                self.balance += (
                    (self.position * current_price)
                    + profit
                    - (self.transaction_cost * current_price)
                )
                self.position = 0

        # Calculate portfolio value
        if self.position > 0:
            self.portfolio_value = self.position * next_price
        elif self.position < 0:
            self.portfolio_value = self.balance - (abs(self.position) * next_price)
        else:
            self.portfolio_value = self.balance

        # Reward is change in portfolio value
        reward = (self.portfolio_value - self.initial_balance) / self.initial_balance

        self.current_step += 1

        return self._get_observation(), reward, self.done, {}


class DQNAgent:
    """Deep Q-Network agent for trading."""

    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQN agent")

        # Neural networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Copy weights to target network
        self.update_target_network()

    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:  # Limit memory size
            self.memory.pop(0)

    def act(self, state):
        """Choose action using epsilon-greedy policy."""
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self, batch_size=32):
        """Train the model on a batch of experiences."""
        if len(self.memory) < batch_size:
            return

        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states = torch.FloatTensor([self.memory[i][0] for i in batch]).to(self.device)
        actions = torch.LongTensor([self.memory[i][1] for i in batch]).to(self.device)
        rewards = torch.FloatTensor([self.memory[i][2] for i in batch]).to(self.device)
        next_states = torch.FloatTensor([self.memory[i][3] for i in batch]).to(
            self.device
        )
        dones = torch.BoolTensor([self.memory[i][4] for i in batch]).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class DQNNetwork(nn.Module):
    """Deep Q-Network architecture."""

    def __init__(self, state_size: int, action_size: int):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ReinforcementLearningTrader(BaseFinancialModel):
    """Reinforcement Learning trading model."""

    def __init__(self, config: ModelConfig = None):
        super().__init__("RLTrader", config)
        self.episodes = config.hyperparameters.get("episodes", 1000) if config else 1000
        self.agent = None
        self.env = None

    def fit(
        self, X: pd.DataFrame, y: pd.Series = None, **kwargs
    ) -> "ReinforcementLearningTrader":
        """Train RL agent on trading environment."""
        # Create trading environment
        self.env = TradingEnvironment(X)

        # Create DQN agent
        state_size = len(self.env._get_observation())
        action_size = 3  # hold, buy, sell
        self.agent = DQNAgent(state_size, action_size)

        # Training loop
        scores = []
        for episode in range(self.episodes):
            state = self.env.reset()
            total_reward = 0

            while not self.env.done:
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if len(self.agent.memory) > 32:
                    self.agent.replay()

            scores.append(total_reward)

            # Update target network periodically
            if episode % 100 == 0:
                self.agent.update_target_network()
                logger.info(
                    f"Episode {episode}, Average Score: {np.mean(scores[-100:]):.4f}"
                )

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate trading signals using trained RL agent."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Create environment for prediction
        env = TradingEnvironment(X)
        state = env.reset()
        actions = []

        while not env.done:
            action = self.agent.act(state)
            actions.append(action)
            state, _, done, _ = env.step(action)

        return np.array(actions)


# Ensemble Methods
class EnsembleModel(BaseFinancialModel):
    """Ensemble model combining multiple base models."""

    def __init__(
        self, models: List[BaseFinancialModel], weights: Optional[List[float]] = None
    ):
        super().__init__("Ensemble")
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "EnsembleModel":
        """Fit all models in the ensemble."""
        for model in self.models:
            try:
                model.fit(X, y, **kwargs)
                logger.info(f"Fitted {model.name} successfully")
            except Exception as e:
                logger.warning(f"Failed to fit {model.name}: {e}")

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        predictions = []
        for model in self.models:
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Failed to predict with {model.name}: {e}")
                # Use zeros as fallback
                predictions.append(np.zeros(len(X)))

        # Weighted average of predictions
        ensemble_pred = np.zeros(len(X))
        total_weight = 0

        for pred, weight in zip(predictions, self.weights):
            if len(pred) == len(X):  # Ensure prediction length matches
                ensemble_pred += pred * weight
                total_weight += weight

        if total_weight > 0:
            ensemble_pred /= total_weight

        return ensemble_pred

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get aggregated feature importance from ensemble models."""
        importance_dict = {}

        for model, weight in zip(self.models, self.weights):
            model_importance = model.get_feature_importance()
            if model_importance:
                for feature, importance in model_importance.items():
                    if feature in importance_dict:
                        importance_dict[feature] += importance * weight
                    else:
                        importance_dict[feature] = importance * weight

        return importance_dict if importance_dict else None


# Traditional ML Models with Financial Enhancements
class EnhancedRandomForest(BaseFinancialModel):
    """Enhanced Random Forest with financial-specific features."""

    def __init__(self, config: ModelConfig = None):
        super().__init__("EnhancedRandomForest", config)

        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for Random Forest")

        # Default parameters optimized for financial data
        self.n_estimators = (
            config.hyperparameters.get("n_estimators", 200) if config else 200
        )
        self.max_depth = config.hyperparameters.get("max_depth", 10) if config else 10
        self.min_samples_split = (
            config.hyperparameters.get("min_samples_split", 5) if config else 5
        )
        self.min_samples_leaf = (
            config.hyperparameters.get("min_samples_leaf", 2) if config else 2
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "EnhancedRandomForest":
        """Fit Random Forest model."""
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
            n_jobs=-1,
        )

        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from Random Forest."""
        if not self.is_fitted or self.feature_names is None:
            return None

        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))


class XGBoostModel(BaseFinancialModel):
    """XGBoost model optimized for financial data."""

    def __init__(self, config: ModelConfig = None):
        super().__init__("XGBoost", config)

        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required for this model")

        # Default parameters
        self.n_estimators = (
            config.hyperparameters.get("n_estimators", 200) if config else 200
        )
        self.max_depth = config.hyperparameters.get("max_depth", 6) if config else 6
        self.learning_rate = (
            config.hyperparameters.get("learning_rate", 0.1) if config else 0.1
        )
        self.subsample = config.hyperparameters.get("subsample", 0.8) if config else 0.8

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "XGBoostModel":
        """Fit XGBoost model."""
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            random_state=42,
            n_jobs=-1,
        )

        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from XGBoost."""
        if not self.is_fitted or self.feature_names is None:
            return None

        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))


if __name__ == "__main__":
    # Example usage
    print("Advanced ML Models for Finance Example")
    print("=" * 50)

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    # Create sample features and target
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :5], axis=1) + np.random.randn(n_samples) * 0.1

    # Test different models
    models_to_test = ["rf", "xgboost"]

    for model_type in models_to_test:
        try:
            print(f"\nTesting {model_type.upper()} model:")

            # Create model
            model = ModelFactory.create_model(model_type)

            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Fit and evaluate
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            metrics = model.evaluate(X_test, y_test)

            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  R: {metrics['r2']:.4f}")
            print(
                f"  Directional Accuracy: {metrics.get('directional_accuracy', 'N/A')}"
            )

        except ImportError as e:
            print(f"  Skipping {model_type}: {e}")
        except Exception as e:
            print(f"  Error with {model_type}: {e}")

    # Test ensemble
    try:
        print("\nTesting Ensemble model:")
        ensemble = ModelFactory.create_ensemble(["rf", "xgboost"])
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)
        metrics = ensemble.evaluate(X_test, y_test)

        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  R: {metrics['r2']:.4f}")

    except Exception as e:
        print(f"  Error with ensemble: {e}")

    print("\nAdvanced ML models implementation completed!")
