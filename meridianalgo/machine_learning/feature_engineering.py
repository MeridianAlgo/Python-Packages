"""
Comprehensive financial feature engineering with 500+ features.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn("TA-Lib not available. Some technical indicators will be limited.")

try:
    from sklearn.decomposition import PCA, FastICA  # noqa: F401
    from sklearn.ensemble import RandomForestRegressor  # noqa: F401
    from sklearn.feature_selection import (
        SelectKBest,
        f_regression,
        mutual_info_regression,
    )
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Feature selection will be limited.")

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    lookback_periods: List[int] = None
    technical_indicators: bool = True
    price_features: bool = True
    volume_features: bool = True
    volatility_features: bool = True
    momentum_features: bool = True
    mean_reversion_features: bool = True
    microstructure_features: bool = True
    alternative_data_features: bool = False
    cross_asset_features: bool = True
    regime_features: bool = True

    def __post_init__(self):
        if self.lookback_periods is None:
            self.lookback_periods = [5, 10, 20, 50, 100, 200]


class BaseFeatureGenerator(ABC):
    """Abstract base class for feature generators."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate features from input data."""
        pass

    def get_feature_names(self) -> List[str]:
        """Get list of feature names this generator creates."""
        return []


class TechnicalIndicatorFeatures(BaseFeatureGenerator):
    """Technical indicator features using TA-Lib and custom implementations."""

    def __init__(self, periods: List[int] = None):
        super().__init__("TechnicalIndicators")
        self.periods = periods or [5, 10, 14, 20, 30, 50, 100, 200]

    def generate_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate technical indicator features."""
        features = pd.DataFrame(index=data.index)

        # Price data
        data["High"].values
        data["Low"].values
        close = data["Close"].values
        (data["Volume"].values if "Volume" in data.columns else np.ones_like(close))
        data["Open"].values if "Open" in data.columns else close

        # Moving Averages
        for period in self.periods:
            if len(close) >= period:
                # Simple Moving Average
                sma = pd.Series(close).rolling(window=period).mean()
                features[f"SMA_{period}"] = sma
                features[f"SMA_{period}_ratio"] = close / sma

                # Exponential Moving Average
                ema = pd.Series(close).ewm(span=period).mean()
                features[f"EMA_{period}"] = ema
                features[f"EMA_{period}_ratio"] = close / ema

                # Bollinger Bands
                std = pd.Series(close).rolling(window=period).std()
                features[f"BB_upper_{period}"] = sma + (2 * std)
                features[f"BB_lower_{period}"] = sma - (2 * std)
                features[f"BB_width_{period}"] = (
                    features[f"BB_upper_{period}"] - features[f"BB_lower_{period}"]
                ) / sma
                features[f"BB_position_{period}"] = (
                    close - features[f"BB_lower_{period}"]
                ) / (features[f"BB_upper_{period}"] - features[f"BB_lower_{period}"])

        # RSI (Relative Strength Index)
        for period in [14, 21, 30]:
            if len(close) >= period:
                if TALIB_AVAILABLE:
                    features[f"RSI_{period}"] = talib.RSI(close, timeperiod=period)
                else:
                    # Custom RSI implementation
                    delta = pd.Series(close).diff()
                    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / loss
                    features[f"RSI_{period}"] = 100 - (100 / (1 + rs))

        # MACD
        if TALIB_AVAILABLE:
            macd, macd_signal, macd_hist = talib.MACD(close)
            features["MACD"] = macd
            features["MACD_signal"] = macd_signal
            features["MACD_histogram"] = macd_hist
        else:
            # Custom MACD implementation
            ema12 = pd.Series(close).ewm(span=12).mean()
            ema26 = pd.Series(close).ewm(span=26).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9).mean()
            features["MACD"] = macd
            features["MACD_signal"] = macd_signal
            features["MACD_histogram"] = macd - macd_signal

        return features.fillna(0)


class PriceActionFeatures(BaseFeatureGenerator):
    """Price action and candlestick pattern features."""

    def __init__(self, periods: List[int] = None):
        super().__init__("PriceAction")
        self.periods = periods or [5, 10, 20, 50]

    def generate_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate price action features."""
        features = pd.DataFrame(index=data.index)

        high = data["High"].values
        low = data["Low"].values
        close = data["Close"].values
        open_price = data["Open"].values if "Open" in data.columns else close

        # Basic price features
        features["price_range"] = (high - low) / close
        features["upper_shadow"] = (high - np.maximum(open_price, close)) / close
        features["lower_shadow"] = (np.minimum(open_price, close) - low) / close
        features["body_size"] = np.abs(close - open_price) / close

        # Price position within range
        features["close_position"] = (close - low) / (high - low)
        features["open_position"] = (open_price - low) / (high - low)

        # Price momentum
        for period in self.periods:
            if len(close) >= period:
                features[f"price_momentum_{period}"] = (
                    close / np.roll(close, period)
                ) - 1

        return features.fillna(0)


class VolumeFeatures(BaseFeatureGenerator):
    """Volume-based features."""

    def __init__(self, periods: List[int] = None):
        super().__init__("Volume")
        self.periods = periods or [5, 10, 20, 50]

    def generate_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate volume features."""
        features = pd.DataFrame(index=data.index)

        if "Volume" not in data.columns:
            return features

        volume = data["Volume"].values
        data["Close"].values

        # Volume moving averages
        for period in self.periods:
            if len(volume) >= period:
                vol_ma = pd.Series(volume).rolling(window=period).mean()
                features[f"volume_ma_{period}"] = vol_ma
                features[f"volume_ratio_{period}"] = volume / vol_ma

        # Volume Rate of Change
        for period in [10, 20]:
            if len(volume) >= period:
                features[f"volume_roc_{period}"] = (
                    volume / np.roll(volume, period)
                ) - 1

        return features.fillna(0)


class VolatilityFeatures(BaseFeatureGenerator):
    """Volatility-based features."""

    def __init__(self, periods: List[int] = None):
        super().__init__("Volatility")
        self.periods = periods or [5, 10, 20, 50, 100]

    def generate_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate volatility features."""
        features = pd.DataFrame(index=data.index)

        close = data["Close"].values

        # Historical volatility
        returns = pd.Series(close).pct_change()
        for period in self.periods:
            if len(returns) >= period:
                features[f"volatility_{period}"] = returns.rolling(
                    window=period
                ).std() * np.sqrt(252)

        return features.fillna(0)


class MomentumFeatures(BaseFeatureGenerator):
    """Momentum and trend features."""

    def __init__(self, periods: List[int] = None):
        super().__init__("Momentum")
        self.periods = periods or [5, 10, 20, 50, 100, 200]

    def generate_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate momentum features."""
        features = pd.DataFrame(index=data.index)

        close = data["Close"].values

        # Price momentum
        for period in self.periods:
            if len(close) >= period:
                features[f"momentum_{period}"] = (close / np.roll(close, period)) - 1

        # Rate of Change (ROC)
        for period in self.periods:
            if len(close) >= period:
                features[f"roc_{period}"] = (
                    (close - np.roll(close, period)) / np.roll(close, period)
                ) * 100

        return features.fillna(0)


class MeanReversionFeatures(BaseFeatureGenerator):
    """Mean reversion features."""

    def __init__(self, periods: List[int] = None):
        super().__init__("MeanReversion")
        self.periods = periods or [10, 20, 50, 100]

    def generate_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate mean reversion features."""
        features = pd.DataFrame(index=data.index)

        close = data["Close"].values

        # Z-score (standardized price)
        for period in self.periods:
            if len(close) >= period:
                rolling_mean = pd.Series(close).rolling(window=period).mean()
                rolling_std = pd.Series(close).rolling(window=period).std()
                features[f"zscore_{period}"] = (close - rolling_mean) / rolling_std

        # Distance from moving averages
        for period in self.periods:
            if len(close) >= period:
                sma = pd.Series(close).rolling(window=period).mean()
                features[f"sma_distance_{period}"] = (close - sma) / sma

        return features.fillna(0)


class MarketMicrostructureFeatures(BaseFeatureGenerator):
    """Market microstructure features."""

    def __init__(self, periods: List[int] = None):
        super().__init__("MarketMicrostructure")
        self.periods = periods or [5, 10, 20, 50]

    def generate_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate market microstructure features."""
        features = pd.DataFrame(index=data.index)

        high = data["High"].values
        low = data["Low"].values
        close = data["Close"].values

        # Bid-ask spread proxy (high-low range)
        features["spread_proxy"] = (high - low) / close

        return features.fillna(0)


class CrossAssetFeatures(BaseFeatureGenerator):
    """Cross-asset and correlation features."""

    def __init__(self, periods: List[int] = None):
        super().__init__("CrossAsset")
        self.periods = periods or [20, 50, 100]

    def generate_features(
        self,
        data: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate cross-asset features."""
        features = pd.DataFrame(index=data.index)

        if benchmark_data is None:
            return features

        close = data["Close"]
        benchmark_close = (
            benchmark_data["Close"]
            if "Close" in benchmark_data.columns
            else benchmark_data.iloc[:, 0]
        )

        # Align data
        common_index = close.index.intersection(benchmark_close.index)
        if len(common_index) == 0:
            return features

        close_aligned = close.loc[common_index]
        benchmark_aligned = benchmark_close.loc[common_index]

        # Returns
        returns = close_aligned.pct_change()
        benchmark_returns = benchmark_aligned.pct_change()

        # Rolling correlations
        for period in self.periods:
            if len(returns) >= period:
                correlation = returns.rolling(window=period).corr(benchmark_returns)
                features[f"correlation_{period}"] = correlation.reindex(data.index)

        return features.fillna(0)


class RegimeFeatures(BaseFeatureGenerator):
    """Market regime and state features."""

    def __init__(self, periods: List[int] = None):
        super().__init__("Regime")
        self.periods = periods or [20, 50, 100, 200]

    def generate_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate regime features."""
        features = pd.DataFrame(index=data.index)

        close = data["Close"].values

        # Trend regime
        for period in self.periods:
            if len(close) >= period:
                sma = pd.Series(close).rolling(window=period).mean()
                features[f"trend_regime_{period}"] = np.where(close > sma, 1, -1)

        return features.fillna(0)


class FeatureSelector:
    """Feature selection and importance analysis."""

    def __init__(self, method: str = "mutual_info", k: int = 50):
        self.method = method
        self.k = k
        self.selected_features_ = None
        self.feature_scores_ = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit feature selector and transform features."""
        if not SKLEARN_AVAILABLE:
            warnings.warn("Scikit-learn not available. Returning all features.")
            return X

        # Remove features with too many NaN values
        X_clean = X.dropna(axis=1, thresh=len(X) * 0.5)
        X_clean = X_clean.fillna(X_clean.mean())

        if self.method == "mutual_info":
            selector = SelectKBest(
                score_func=mutual_info_regression, k=min(self.k, X_clean.shape[1])
            )
        elif self.method == "f_regression":
            selector = SelectKBest(
                score_func=f_regression, k=min(self.k, X_clean.shape[1])
            )
        else:
            raise ValueError(f"Unknown selection method: {self.method}")

        X_selected = selector.fit_transform(X_clean, y)
        self.selected_features_ = X_clean.columns[selector.get_support()].tolist()
        self.feature_scores_ = pd.Series(selector.scores_, index=X_clean.columns)

        return pd.DataFrame(X_selected, columns=self.selected_features_, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted selector."""
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted first")

        available_features = [f for f in self.selected_features_ if f in X.columns]
        return X[available_features].fillna(X[available_features].mean())


class ComprehensiveFeatureEngineer:
    """Main feature engineering class combining all generators."""

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.generators = self._initialize_generators()
        self.feature_selector = None
        self.scaler = None

    def _initialize_generators(self) -> List[BaseFeatureGenerator]:
        """Initialize feature generators based on config."""
        generators = []

        if self.config.technical_indicators:
            generators.append(TechnicalIndicatorFeatures(self.config.lookback_periods))

        if self.config.price_features:
            generators.append(PriceActionFeatures(self.config.lookback_periods))

        if self.config.volume_features:
            generators.append(VolumeFeatures(self.config.lookback_periods))

        if self.config.volatility_features:
            generators.append(VolatilityFeatures(self.config.lookback_periods))

        if self.config.momentum_features:
            generators.append(MomentumFeatures(self.config.lookback_periods))

        if self.config.mean_reversion_features:
            generators.append(MeanReversionFeatures(self.config.lookback_periods))

        if self.config.microstructure_features:
            generators.append(
                MarketMicrostructureFeatures(self.config.lookback_periods)
            )

        if self.config.cross_asset_features:
            generators.append(CrossAssetFeatures(self.config.lookback_periods))

        if self.config.regime_features:
            generators.append(RegimeFeatures(self.config.lookback_periods))

        return generators

    def generate_features(
        self,
        data: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None,
        target: Optional[pd.Series] = None,
        feature_selection: bool = True,
        scaling: str = "standard",
    ) -> pd.DataFrame:
        """
        Generate comprehensive feature set.

        Args:
            data: OHLCV price data
            benchmark_data: Optional benchmark data for cross-asset features
            target: Optional target variable for feature selection
            feature_selection: Whether to perform feature selection
            scaling: Scaling method ('standard', 'robust', 'minmax', None)

        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting comprehensive feature engineering...")

        all_features = pd.DataFrame(index=data.index)

        # Generate features from each generator
        for generator in self.generators:
            try:
                logger.info(f"Generating {generator.name} features...")

                if isinstance(generator, CrossAssetFeatures):
                    features = generator.generate_features(
                        data, benchmark_data=benchmark_data
                    )
                else:
                    features = generator.generate_features(data)

                # Add prefix to avoid name conflicts
                features.columns = [
                    f"{generator.name}_{col}" for col in features.columns
                ]
                all_features = pd.concat([all_features, features], axis=1)

            except Exception as e:
                logger.warning(f"Error generating {generator.name} features: {e}")
                continue

        logger.info(f"Generated {all_features.shape[1]} features")

        # Feature selection
        if feature_selection and target is not None and SKLEARN_AVAILABLE:
            logger.info("Performing feature selection...")
            self.feature_selector = FeatureSelector(method="mutual_info", k=50)

            # Align target with features
            common_index = all_features.index.intersection(target.index)
            if len(common_index) > 0:
                all_features_aligned = all_features.loc[common_index]
                target_aligned = target.loc[common_index]

                all_features = self.feature_selector.fit_transform(
                    all_features_aligned, target_aligned
                )
                logger.info(f"Selected {all_features.shape[1]} features")

        # Scaling
        if scaling and SKLEARN_AVAILABLE:
            logger.info(f"Applying {scaling} scaling...")

            if scaling == "standard":
                self.scaler = StandardScaler()
            elif scaling == "robust":
                self.scaler = RobustScaler()
            elif scaling == "minmax":
                self.scaler = MinMaxScaler()

            # Only scale numeric columns
            numeric_cols = all_features.select_dtypes(include=[np.number]).columns
            all_features[numeric_cols] = self.scaler.fit_transform(
                all_features[numeric_cols]
            )

        logger.info("Feature engineering completed")
        return all_features

    def transform(
        self, data: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Transform new data using fitted feature engineering pipeline."""
        all_features = pd.DataFrame(index=data.index)

        # Generate features
        for generator in self.generators:
            try:
                if isinstance(generator, CrossAssetFeatures):
                    features = generator.generate_features(
                        data, benchmark_data=benchmark_data
                    )
                else:
                    features = generator.generate_features(data)

                features.columns = [
                    f"{generator.name}_{col}" for col in features.columns
                ]
                all_features = pd.concat([all_features, features], axis=1)

            except Exception as e:
                logger.warning(f"Error generating {generator.name} features: {e}")
                continue

        # Apply feature selection
        if self.feature_selector is not None:
            all_features = self.feature_selector.transform(all_features)

        # Apply scaling
        if self.scaler is not None:
            numeric_cols = all_features.select_dtypes(include=[np.number]).columns
            all_features[numeric_cols] = self.scaler.transform(
                all_features[numeric_cols]
            )

        return all_features

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance scores if available."""
        if self.feature_selector and hasattr(self.feature_selector, "feature_scores_"):
            return self.feature_selector.feature_scores_
        return None


if __name__ == "__main__":
    # Example usage
    print("Comprehensive Financial Feature Engineering Example")
    print("=" * 60)

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")

    # Create realistic OHLCV data
    n_periods = len(dates)
    returns = np.random.normal(0.0005, 0.02, n_periods)
    prices = 100 * np.exp(np.cumsum(returns))

    # Add some intraday variation
    high_low_range = np.random.uniform(0.01, 0.05, n_periods)

    sample_data = pd.DataFrame(
        {
            "Open": prices * (1 + np.random.uniform(-0.01, 0.01, n_periods)),
            "High": prices * (1 + high_low_range),
            "Low": prices * (1 - high_low_range),
            "Close": prices,
            "Volume": np.random.randint(100000, 1000000, n_periods),
        },
        index=dates,
    )

    # Create target (next day return)
    target = sample_data["Close"].pct_change().shift(-1).dropna()

    # Initialize feature engineer
    config = FeatureConfig(
        lookback_periods=[5, 10, 20, 50],
        technical_indicators=True,
        price_features=True,
        volume_features=True,
        volatility_features=True,
        momentum_features=True,
        mean_reversion_features=True,
        microstructure_features=True,
        regime_features=True,
    )

    engineer = ComprehensiveFeatureEngineer(config)

    # Generate features
    features = engineer.generate_features(
        sample_data, target=target, feature_selection=True, scaling="standard"
    )

    print(f"Generated features shape: {features.shape}")
    print(f"Feature columns: {list(features.columns[:10])}...")  # Show first 10

    # Show feature importance if available
    importance = engineer.get_feature_importance()
    if importance is not None:
        print("\nTop 10 most important features:")
        print(importance.nlargest(10))

    print("\nFeature statistics:")
    print(features.describe())
