"""
MeridianAlgo Machine Learning Module

Comprehensive machine learning functionality for financial applications including
time series forecasting, feature engineering, model evaluation, and deployment.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVR

# Suppress warnings
warnings.filterwarnings("ignore")

# Check for optional dependencies
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from sklearn.neural_network import MLPRegressor  # noqa: F401

    SKLEARN_ML_AVAILABLE = True
except ImportError:
    SKLEARN_ML_AVAILABLE = False


class FeatureEngineer:
    """
    Advanced feature engineering for financial time series.

    Features:
    - Technical indicators
    - Lag features
    - Rolling statistics
    - Fourier transforms
    - Regime features
    - Sentiment features (if available)
    """

    def __init__(
        self, lookback: int = 10, lookback_periods: Optional[List[int]] = None
    ):
        """
        Initialize feature engineer.

        Args:
            lookback: Default lookback period
            lookback_periods: List of lookback periods for features
        """
        self.lookback = lookback
        self.lookback_periods = lookback_periods or [5, 10, 20, 50]
        self.scalers = {}
        self.feature_names = []
        self.is_fitted = False

    def create_technical_features(
        self, prices: pd.DataFrame, volumes: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create technical indicator features.

        Args:
            prices: DataFrame of prices
            volumes: Optional DataFrame of volumes

        Returns:
            DataFrame of technical features
        """
        features = pd.DataFrame(index=prices.index)

        for asset in prices.columns:
            price_series = prices[asset]

            # Returns
            features[f"{asset}_returns"] = price_series.pct_change()
            features[f"{asset}_log_returns"] = np.log(
                price_series / price_series.shift(1)
            )

            # Moving averages
            for period in self.lookback_periods:
                features[f"{asset}_ma_{period}"] = price_series.rolling(period).mean()
                features[f"{asset}_ema_{period}"] = price_series.ewm(span=period).mean()

            # Volatility
            features[f"{asset}_volatility"] = (
                price_series.pct_change().rolling(self.lookback).std()
            )
            for period in self.lookback_periods:
                features[f"{asset}_vol_{period}"] = (
                    price_series.pct_change().rolling(period).std()
                )

            # Momentum
            features[f"{asset}_momentum"] = price_series.pct_change(self.lookback)

            # RSI
            features[f"{asset}_rsi_14"] = self._calculate_rsi(price_series, 14)

            # MACD
            macd_data = self._calculate_macd(price_series)
            features[f"{asset}_macd"] = macd_data["macd"]
            features[f"{asset}_macd_signal"] = macd_data["signal"]
            features[f"{asset}_macd_hist"] = macd_data["histogram"]

            # Bollinger Bands
            bb_data = self._calculate_bollinger_bands(price_series)
            features[f"{asset}_bb_upper"] = bb_data["upper"]
            features[f"{asset}_bb_middle"] = bb_data["middle"]
            features[f"{asset}_bb_lower"] = bb_data["lower"]
            features[f"{asset}_bb_width"] = (
                bb_data["upper"] - bb_data["lower"]
            ) / bb_data["middle"]
            features[f"{asset}_bb_position"] = (price_series - bb_data["lower"]) / (
                bb_data["upper"] - bb_data["lower"]
            )

            # Price patterns
            for period in [5, 10, 20]:
                features[f"{asset}_high_{period}"] = price_series.rolling(period).max()
                features[f"{asset}_low_{period}"] = price_series.rolling(period).min()
                features[f"{asset}_rank_{period}"] = price_series.rolling(period).rank(
                    pct=True
                )

            # Volume features (if available)
            if volumes is not None and asset in volumes.columns:
                vol_series = volumes[asset]
                for period in [5, 10, 20]:
                    features[f"{asset}_vol_ma_{period}"] = vol_series.rolling(
                        period
                    ).mean()
                    features[f"{asset}_vol_std_{period}"] = vol_series.rolling(
                        period
                    ).std()

                # Price-volume correlation
                features[f"{asset}_price_volume_corr_20"] = (
                    price_series.pct_change().rolling(20).corr(vol_series.pct_change())
                )

        self.feature_names = features.columns.tolist()
        return features

    def create_features(self, prices: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Alias for create_technical_features for simplified usage."""
        if isinstance(prices, pd.Series):
            name = prices.name or "price"
            df = pd.DataFrame({name: prices})
            result = self.create_technical_features(df)
            # Remove the name prefix if it's a single series to match test expectations
            result.columns = [col.replace(f"{name}_", "") for col in result.columns]
            return result
        else:
            return self.create_technical_features(prices)

    def create_sequences(
        self, X: np.ndarray, y: np.ndarray, sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction models."""
        X_seq, y_seq = [], []
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i - sequence_length : i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    def create_lag_features(
        self, data: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        """
        Create lag features.

        Args:
            data: DataFrame of base features
            lags: List of lag periods

        Returns:
            DataFrame with lag features
        """
        lag_features = pd.DataFrame(index=data.index)

        for col in data.columns:
            for lag in lags:
                lag_features[f"{col}_lag_{lag}"] = data[col].shift(lag)

        return lag_features

    def create_rolling_features(
        self,
        data: pd.DataFrame,
        windows: List[int] = [5, 10, 20],
        functions: List[str] = ["mean", "std", "min", "max", "skew", "kurt"],
    ) -> pd.DataFrame:
        """
        Create rolling window features.

        Args:
            data: DataFrame of base features
            windows: List of window sizes
            functions: List of functions to apply

        Returns:
            DataFrame with rolling features
        """
        rolling_features = pd.DataFrame(index=data.index)

        for col in data.columns:
            for window in windows:
                for func in functions:
                    if func == "mean":
                        rolling_features[f"{col}_rolling_{func}_{window}"] = (
                            data[col].rolling(window).mean()
                        )
                    elif func == "std":
                        rolling_features[f"{col}_rolling_{func}_{window}"] = (
                            data[col].rolling(window).std()
                        )
                    elif func == "min":
                        rolling_features[f"{col}_rolling_{func}_{window}"] = (
                            data[col].rolling(window).min()
                        )
                    elif func == "max":
                        rolling_features[f"{col}_rolling_{func}_{window}"] = (
                            data[col].rolling(window).max()
                        )
                    elif func == "skew":
                        rolling_features[f"{col}_rolling_{func}_{window}"] = (
                            data[col].rolling(window).skew()
                        )
                    elif func == "kurt":
                        rolling_features[f"{col}_rolling_{func}_{window}"] = (
                            data[col].rolling(window).kurt()
                        )

        return rolling_features

    def create_fourier_features(
        self, data: pd.DataFrame, frequencies: List[int] = [1, 2, 3, 4, 5]
    ) -> pd.DataFrame:
        """
        Create Fourier transform features.

        Args:
            data: DataFrame of time series data
            frequencies: List of frequencies for Fourier transform

        Returns:
            DataFrame with Fourier features
        """
        fourier_features = pd.DataFrame(index=data.index)

        for col in data.columns:
            series = data[col].dropna()
            if len(series) > 0:
                fft = np.fft.fft(series.values)

                for freq in frequencies:
                    if freq < len(fft):
                        fourier_features[f"{col}_fft_real_{freq}"] = np.real(fft[freq])
                        fourier_features[f"{col}_fft_imag_{freq}"] = np.imag(fft[freq])
                        fourier_features[f"{col}_fft_power_{freq}"] = (
                            np.abs(fft[freq]) ** 2
                        )

        return fourier_features

    def create_regime_features(
        self, data: pd.DataFrame, window: int = 50
    ) -> pd.DataFrame:
        """
        Create market regime features.

        Args:
            data: DataFrame of returns
            window: Window for regime detection

        Returns:
            DataFrame with regime features
        """
        regime_features = pd.DataFrame(index=data.index)

        for col in data.columns:
            returns = data[col]

            # Volatility regime
            rolling_vol = returns.rolling(window).std()
            vol_regime = pd.qcut(
                rolling_vol, q=3, labels=["low_vol", "med_vol", "high_vol"]
            )
            regime_features[f"{col}_vol_regime"] = vol_regime.cat.codes

            # Trend regime
            rolling_mean = returns.rolling(window).mean()
            trend_regime = pd.qcut(
                rolling_mean, q=3, labels=["downtrend", "sideways", "uptrend"]
            )
            regime_features[f"{col}_trend_regime"] = trend_regime.cat.codes

            # Correlation regime (for multiple assets)
            if len(data.columns) > 1:
                corr_matrix = data.rolling(window).corr()
                avg_corr = corr_matrix.mean().mean(axis=1)
                corr_regime = pd.qcut(
                    avg_corr, q=3, labels=["low_corr", "med_corr", "high_corr"]
                )
                regime_features[f"{col}_corr_regime"] = corr_regime.cat.codes

        return regime_features

    def fit_scalers(self, data: pd.DataFrame, method: str = "standard") -> None:
        """
        Fit scalers for feature normalization.

        Args:
            data: DataFrame of features
            method: Scaling method ('standard', 'minmax', 'robust')
        """
        if method == "standard":
            scaler_class = StandardScaler
        elif method == "minmax":
            scaler_class = MinMaxScaler
        elif method == "robust":
            scaler_class = RobustScaler
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        self.scalers = {}
        for col in data.columns:
            scaler = scaler_class()
            scaler.fit(data[[col]].dropna())
            self.scalers[col] = scaler

        self.is_fitted = True

    def transform_features(
        self, data: pd.DataFrame, method: str = "standard"
    ) -> pd.DataFrame:
        """
        Transform features using fitted scalers.

        Args:
            data: DataFrame of features
            method: Scaling method

        Returns:
            Scaled DataFrame
        """
        if not self.is_fitted:
            self.fit_scalers(data, method)

        scaled_data = data.copy()

        for col in data.columns:
            if col in self.scalers:
                scaled_data[col] = self.scalers[col].transform(data[[col]])

        return scaled_data

    def fit_transform(
        self, data: pd.DataFrame, method: str = "standard"
    ) -> pd.DataFrame:
        """
        Fit scalers and transform features.

        Args:
            data: DataFrame of features
            method: Scaling method

        Returns:
            Scaled DataFrame
        """
        self.fit_scalers(data, method)
        return self.transform_features(data, method)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Dict[str, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line

        return {"macd": macd, "signal": signal_line, "histogram": histogram}

    def _calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, std: float = 2
    ) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(period).mean()
        rolling_std = prices.rolling(period).std()
        upper = middle + (rolling_std * std)
        lower = middle - (rolling_std * std)

        return {"upper": upper, "middle": middle, "lower": lower}


class LSTMPredictor:
    """
    LSTM-based time series predictor using PyTorch.

    Features:
    - Multi-layer LSTM architecture
    - Dropout regularization
    - Early stopping
    - Hyperparameter optimization
    """

    def __init__(
        self,
        sequence_length: int = 20,
        hidden_size: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 10,
    ):
        """
        Initialize LSTM predictor.

        Args:
            sequence_length: Length of input sequences
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Number of training epochs
            early_stopping_patience: Early stopping patience
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTMPredictor")

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience

        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_fitted = False

        # Training history
        self.train_losses = []
        self.val_losses = []

    def _create_model(self, input_size: int) -> nn.Module:
        """Create LSTM model."""

        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers

                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                )

                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                lstm_out = self.dropout(lstm_out[:, -1, :])  # Use last output
                output = self.fc(lstm_out)
                return output

        return LSTMModel(
            input_size, self.hidden_size, self.num_layers, self.dropout
        ).to(self.device)

    def prepare_data(
        self, features: pd.DataFrame, target: pd.Series, test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training.

        Args:
            features: DataFrame of features
            target: Series of target values
            test_size: Fraction of data for testing

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Create sequences
        X, y = [], []

        for i in range(self.sequence_length, len(features_scaled)):
            X.append(features_scaled[i - self.sequence_length : i])
            # Handle both pandas and numpy for target
            target_val = target.iloc[i] if hasattr(target, "iloc") else target[i]
            y.append(target_val)

        X = np.array(X)
        y = np.array(y)

        # Split data
        split_idx = int(len(X) * (1 - test_size))

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return X_train, X_test, y_train, y_test

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        validation_split: float = 0.2,
        verbose: bool = True,
    ) -> "LSTMPredictor":
        """
        Train LSTM model.

        Args:
            features: DataFrame of features
            target: Series of target values
            validation_split: Fraction of data for validation
            verbose: Whether to print training progress

        Returns:
            Self
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(features, target)

        # Create validation split
        val_split_idx = int(len(X_train) * (1 - validation_split))
        X_val, y_val = X_train[val_split_idx:], y_train[val_split_idx:]
        X_train, y_train = X_train[:val_split_idx], y_train[:val_split_idx]

        # Create model
        self.model = self._create_model(X_train.shape[2])

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), "best_lstm_model.pth")
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

        # Load best model
        self.model.load_state_dict(torch.load("best_lstm_model.pth"))
        self.is_fitted = True

        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            features: DataFrame of features

        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Create sequences
        X = []
        for i in range(self.sequence_length, len(features_scaled)):
            X.append(features_scaled[i - self.sequence_length : i])

        X = np.array(X)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()

        return predictions.flatten()

    def evaluate(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            features: DataFrame of features
            target: Series of target values

        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(features)

        # Align predictions with target
        y_true = target.iloc[self.sequence_length :].values
        y_pred = predictions

        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)

        # Directional accuracy
        direction_true = np.diff(y_true) > 0
        direction_pred = np.diff(y_pred) > 0
        directional_accuracy = (direction_true == direction_pred).mean()

        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "directional_accuracy": directional_accuracy,
        }


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple models.

    Features:
    - Weighted ensemble
    - Stacking ensemble
    - Cross-validation for weights
    - Model selection
    """

    def __init__(
        self, models: List[BaseEstimator], weights: Optional[List[float]] = None
    ):
        """
        Initialize ensemble predictor.

        Args:
            models: List of base models
            weights: Optional weights for weighted ensemble
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.is_fitted = False
        self.meta_model = None

    def fit(
        self, X: pd.DataFrame, y: pd.Series, method: str = "weighted", cv_folds: int = 5
    ) -> "EnsemblePredictor":
        """
        Fit ensemble models.

        Args:
            X: Features
            y: Target
            method: Ensemble method ('weighted', 'stacking', 'voting')
            cv_folds: Number of CV folds

        Returns:
            Self
        """
        if method == "stacking":
            self._fit_stacking_ensemble(X, y, cv_folds)
        else:
            # Fit individual models
            for model in self.models:
                model.fit(X, y)

        self.is_fitted = True
        return self

    def _fit_stacking_ensemble(
        self, X: pd.DataFrame, y: pd.Series, cv_folds: int
    ) -> None:
        """Fit stacking ensemble."""
        from sklearn.model_selection import KFold

        # Generate out-of-fold predictions
        oof_predictions = np.zeros((len(X), len(self.models)))

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]

            for i, model in enumerate(self.models):
                model.fit(X_train, y_train)
                oof_predictions[val_idx, i] = model.predict(X_val)

        # Fit meta-model on out-of-fold predictions
        self.meta_model = LinearRegression()
        self.meta_model.fit(oof_predictions, y)

        # Fit base models on full data
        for model in self.models:
            model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before making predictions")

        if self.meta_model is not None:
            # Stacking ensemble
            predictions = np.column_stack([model.predict(X) for model in self.models])
            return self.meta_model.predict(predictions)
        else:
            # Weighted ensemble
            predictions = np.zeros(len(X))
            for model, weight in zip(self.models, self.weights):
                predictions += weight * model.predict(X)
            return predictions

    def get_feature_importance(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get feature importance from models.

        Returns:
            Dictionary of feature importances by model
        """
        importance_dict = {}

        for i, model in enumerate(self.models):
            if hasattr(model, "feature_importances_"):
                importance_dict[f"model_{i}"] = model.feature_importances_

        return importance_dict if importance_dict else None


class ModelEvaluator:
    """
    Comprehensive model evaluation for financial ML.

    Features:
    - Time series cross-validation
    - Financial metrics
    - Feature importance analysis
    - Model comparison
    """

    def __init__(self):
        """Initialize model evaluator."""
        self.results = {}

    def evaluate_model(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        cv_method: str = "time_series",
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Evaluate model with time series cross-validation.

        Args:
            model: Model to evaluate
            X: Features
            y: Target
            cv_method: Cross-validation method
            cv_folds: Number of CV folds

        Returns:
            Dictionary of evaluation results
        """
        if cv_method == "time_series":
            cv = TimeSeriesSplit(n_splits=cv_folds)
        else:
            from sklearn.model_selection import KFold

            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # Cross-validation predictions and per-fold metrics
        predictions = np.zeros(len(y))
        fold_metrics = []

        for train_idx, val_idx in cv.split(X):
            # Handle both pandas and numpy
            if hasattr(X, "iloc"):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_train, X_val = X[train_idx], X[val_idx]

            if hasattr(y, "iloc"):
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            else:
                y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            predictions[val_idx] = y_pred
            fold_metrics.append(self.calculate_metrics(y_val, y_pred))

        # Calculate summary metrics
        summary_results = {}
        if fold_metrics:
            metric_keys = fold_metrics[0].keys()
            for key in metric_keys:
                values = [m[key] for m in fold_metrics]
                summary_results[f"mean_{key}"] = np.mean(values)
                summary_results[f"std_{key}"] = np.std(values)

        # Overall metrics
        metrics = self.calculate_metrics(y, predictions)
        summary_results["metrics"] = metrics
        summary_results["predictions"] = predictions
        summary_results["model"] = model

        # Feature importance
        if hasattr(model, "feature_importances_") and hasattr(X, "columns"):
            summary_results["feature_importance"] = dict(
                zip(X.columns, model.feature_importances_)
            )
        else:
            summary_results["feature_importance"] = None

        return summary_results

    @staticmethod
    def calculate_metrics(
        y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        # Convert to numpy for calculations
        y_true_np = np.asarray(y_true)
        y_pred_np = np.asarray(y_pred)

        # Basic regression metrics
        mse = mean_squared_error(y_true_np, y_pred_np)
        mae = mean_absolute_error(y_true_np, y_pred_np)
        r2 = r2_score(y_true_np, y_pred_np)
        rmse = np.sqrt(mse)

        # Financial metrics
        # Directional accuracy
        if len(y_true_np) > 1:
            direction_true = np.diff(y_true_np) > 0
            direction_pred = np.diff(y_pred_np) > 0
            direction_accuracy = (direction_true == direction_pred).mean()
        else:
            direction_accuracy = 1.0

        # Information ratio (if returns)
        if y_true_np.std() > 0:
            information_ratio = (y_pred_np - y_true_np).mean() / (
                y_pred_np - y_true_np
            ).std()
        else:
            information_ratio = 0

        # Hit rate (correct sign prediction)
        hit_rate = (np.sign(y_true_np) == np.sign(y_pred_np)).mean()

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "direction_accuracy": direction_accuracy,
            "directional_accuracy": direction_accuracy,
            "information_ratio": information_ratio,
            "hit_rate": hit_rate,
        }

    _calculate_metrics = calculate_metrics

    @staticmethod
    def time_series_cv(
        model: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        """Alias for evaluate_model with time_series method."""
        evaluator = ModelEvaluator()
        return evaluator.evaluate_model(
            model, X, y, cv_method="time_series", cv_folds=n_splits
        )

    def compare_models(
        self, models: Dict[str, BaseEstimator], X: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """
        Compare multiple models.

        Args:
            models: Dictionary of models
            X: Features
            y: Target

        Returns:
            DataFrame of comparison results
        """
        results = []

        for name, model in models.items():
            result = self.evaluate_model(model, X, y)
            metrics = result["metrics"]
            metrics["model"] = name
            results.append(metrics)

        return pd.DataFrame(results).set_index("model")


# Utility functions
def prepare_data_for_lstm(
    features: pd.DataFrame,
    target: pd.Series,
    sequence_length: int = 20,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for LSTM models.

    Args:
        features: DataFrame of features
        target: Series of target values
        sequence_length: Length of input sequences
        test_size: Fraction of data for testing

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Create sequences
    X, y = [], []

    for i in range(sequence_length, len(features_scaled)):
        X.append(features_scaled[i - sequence_length : i])
        y.append(target.iloc[i])

    X = np.array(X)
    y = np.array(y)

    # Split data
    split_idx = int(len(X) * (1 - test_size))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test


def create_ml_models(
    model_types: List[str] = ["rf", "gb", "xgb", "lgb", "linear"], **kwargs
) -> Dict[str, BaseEstimator]:
    """
    Create a dictionary of ML models.

    Args:
        model_types: List of model types to create
        **kwargs: Additional parameters for models

    Returns:
        Dictionary of models
    """
    models = {}

    for model_type in model_types:
        if model_type == "rf":
            models["random_forest"] = RandomForestRegressor(
                n_estimators=kwargs.get("n_estimators", 100), random_state=42
            )
        elif model_type == "gb":
            models["gradient_boosting"] = GradientBoostingRegressor(
                n_estimators=kwargs.get("n_estimators", 100), random_state=42
            )
        elif model_type == "xgb" and XGBOOST_AVAILABLE:
            models["xgboost"] = xgb.XGBRegressor(
                n_estimators=kwargs.get("n_estimators", 100), random_state=42
            )
        elif model_type == "lgb" and LIGHTGBM_AVAILABLE:
            models["lightgbm"] = lgb.LGBMRegressor(
                n_estimators=kwargs.get("n_estimators", 100), random_state=42
            )
        elif model_type == "linear":
            models["linear"] = LinearRegression()
        elif model_type == "ridge":
            models["ridge"] = Ridge(alpha=kwargs.get("alpha", 1.0))
        elif model_type == "lasso":
            models["lasso"] = Lasso(alpha=kwargs.get("alpha", 1.0))
        elif model_type == "svr":
            models["svr"] = SVR(kernel="rbf")

    return models


# Export main classes and functions
__all__ = [
    "FeatureEngineer",
    "LSTMPredictor",
    "EnsemblePredictor",
    "ModelEvaluator",
    "prepare_data_for_lstm",
    "create_ml_models",
]
