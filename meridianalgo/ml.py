"""
Machine learning module for financial time series prediction.
Includes feature engineering, model training, and prediction utilities.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Union, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

class FeatureEngineer:
    """Feature engineering for financial time series data."""
    
    def __init__(self, lookback: int = 10):
        """
        Initialize feature engineer.
        
        Args:
            lookback: Number of periods to look back for feature creation
        """
        self.lookback = lookback
    
    def create_features(self, data: pd.Series) -> pd.DataFrame:
        """
        Create technical features from price/return data.
        
        Args:
            data: Time series data (prices or returns)
            
        Returns:
            DataFrame with engineered features
        """
        df = pd.DataFrame(index=data.index)
        
        # Basic features
        df['returns'] = data.pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        
        # Momentum features
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = data.pct_change(periods=period)
        
        # Volatility features
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'ma_{period}'] = data.rolling(period).mean()
            df[f'ma_ratio_{period}'] = data / df[f'ma_{period}'] - 1
        
        # Drop NaN values created by rolling windows
        df = df.dropna()
        
        return df
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray, 
                        sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            data: Feature data
            target: Target values
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X, y) arrays for model training
        """
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(target[i + sequence_length])
        return np.array(X), np.array(y)


class LSTMPredictor:
    """LSTM-based time series predictor using PyTorch."""
    
    def __init__(self, sequence_length: int = 10, hidden_size: int = 50, 
                 num_layers: int = 2, epochs: int = 50, batch_size: int = 32, 
                 learning_rate: float = 0.001, dropout: float = 0.2):
        """
        Initialize LSTM predictor with PyTorch backend.
        
        Args:
            sequence_length: Number of time steps to look back
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            dropout: Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the LSTM model.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).unsqueeze(1).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Build and train model
        self.model = LSTMModel(
            input_size=X_seq.shape[2],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = self.model(batch_x)
                    val_loss += criterion(outputs, batch_y).item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss/len(val_loader):.6f}')
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predicted values (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        import torch
        
        X_scaled = self.scaler.transform(X)
        X_seq = self._create_sequences(X_scaled, np.zeros(len(X)))[0]  # Dummy y
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy().flatten()
        
        return y_pred
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input."""
        Xs, ys = [], []
        for i in range(len(X) - self.sequence_length):
            Xs.append(X[i:(i + self.sequence_length)])
            ys.append(y[i + self.sequence_length])
        return np.array(Xs), np.array(ys)


class LSTMModel:
    """PyTorch LSTM model for time series prediction."""
    
    def __init__(self, input_size: int, hidden_size: int, 
                 num_layers: int, dropout: float = 0.2):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate (applied between LSTM layers if num_layers > 1)
        """
        import torch
        import torch.nn as nn
        
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def to(self, device):
        """Move model to specified device (CPU/GPU)."""
        self.device = device
        self.lstm = self.lstm.to(device)
        self.fc = self.fc.to(device)
        return self
    
    def forward(self, x):
        """Forward pass through the network."""
        import torch
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class ModelEvaluator:
    """Evaluate time series prediction models."""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'direction_accuracy': np.mean(np.sign(y_true[1:] - y_true[:-1]) == 
                                        np.sign(y_pred[1:] - y_pred[:-1]))
        }
    
    @staticmethod
    def time_series_cv(model, X: np.ndarray, y: np.ndarray, 
                      n_splits: int = 5) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation.
        
        Args:
            model: Model implementing fit/predict interface
            X: Feature matrix
            y: Target vector
            n_splits: Number of CV folds
            
        Returns:
            Dictionary of evaluation metrics across folds
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics = {'mse': [], 'rmse': [], 'mae': [], 'r2': [], 'direction_accuracy': []}
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            fold_metrics = ModelEvaluator.calculate_metrics(y_test, y_pred)
            for k, v in fold_metrics.items():
                metrics[k].append(v)
        
        # Calculate mean and std of metrics
        result = {}
        for k, v in metrics.items():
            result[f'mean_{k}'] = np.mean(v)
            result[f'std_{k}'] = np.std(v)
        
        return result


def prepare_data_for_lstm(features: pd.DataFrame, target: pd.Series, 
                        sequence_length: int = 10, 
                        test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, 
                                                       np.ndarray, np.ndarray]:
    """
    Prepare data for LSTM models.
    
    Args:
        features: DataFrame of features
        target: Series of target values
        sequence_length: Length of input sequences
        test_size: Fraction of data to use for testing
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_features) - sequence_length):
        X.append(scaled_features[i:(i + sequence_length)])
        y.append(target.iloc[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train/test
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test
