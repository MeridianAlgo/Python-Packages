"""
MeridianAlgo Algorithmic Trading Module

Comprehensive algorithmic trading strategies and execution systems.
Integrates concepts from zipline, backtrader, and other leading algorithmic trading libraries.
"""

import warnings
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# Check for optional dependencies
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt  # noqa: F401

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class TradingStrategy:
    """
    Base class for trading strategies.

    Features:
    - Signal generation
    - Position sizing
    - Risk management
    - Performance tracking
    """

    def __init__(self, name: str, initial_capital: float = 100000):
        """
        Initialize trading strategy.

        Args:
            name: Strategy name
            initial_capital: Initial capital
        """
        self.name = name
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.signals = []

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals.

        Args:
            data: Market data

        Returns:
            DataFrame with signals
        """
        raise NotImplementedError("Subclasses must implement generate_signals")

    def calculate_position_size(
        self, signal: float, price: float, volatility: float
    ) -> float:
        """
        Calculate position size based on signal and risk.

        Args:
            signal: Trading signal
            price: Current price
            volatility: Price volatility

        Returns:
            Position size
        """
        # Simple position sizing based on volatility
        risk_per_trade = 0.02  # 2% risk per trade
        stop_loss_pct = 2 * volatility  # 2 standard deviations

        if abs(signal) > 0.5:  # Only trade on strong signals
            position_size = (self.current_capital * risk_per_trade) / (
                price * stop_loss_pct
            )
            return position_size * np.sign(signal)

        return 0.0

    def execute_trade(
        self, symbol: str, signal: float, price: float, timestamp: datetime
    ):
        """
        Execute a trade.

        Args:
            symbol: Trading symbol
            signal: Trading signal
            price: Execution price
            timestamp: Trade timestamp
        """
        current_position = self.positions.get(symbol, 0)

        if signal > 0 and current_position <= 0:  # Buy signal
            position_size = self.calculate_position_size(signal, price, 0.02)
            if position_size > 0:
                cost = position_size * price
                if cost <= self.current_capital:
                    self.positions[symbol] = current_position + position_size
                    self.current_capital -= cost
                    self.trades.append(
                        {
                            "timestamp": timestamp,
                            "symbol": symbol,
                            "side": "BUY",
                            "quantity": position_size,
                            "price": price,
                            "cost": cost,
                        }
                    )

        elif signal < 0 and current_position > 0:  # Sell signal
            proceeds = current_position * price
            self.positions[symbol] = 0
            self.current_capital += proceeds
            self.trades.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "side": "SELL",
                    "quantity": current_position,
                    "price": price,
                    "proceeds": proceeds,
                }
            )

    def calculate_performance(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate strategy performance metrics.

        Args:
            price_data: Price data for portfolio valuation

        Returns:
            Performance metrics
        """
        if not self.trades:
            return {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}

        # Calculate portfolio value over time
        portfolio_values = []
        for timestamp in price_data.index:
            portfolio_value = self.current_capital
            for symbol, position in self.positions.items():
                if symbol in price_data.columns and timestamp in price_data.index:
                    portfolio_value += position * price_data.loc[timestamp, symbol]
            portfolio_values.append(portfolio_value)

        portfolio_values = pd.Series(portfolio_values, index=price_data.index)
        returns = portfolio_values.pct_change().dropna()

        # Calculate metrics
        total_return = (
            portfolio_values.iloc[-1] - self.initial_capital
        ) / self.initial_capital
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate and profit factor
        winning_trades = [t for t in self.trades if t.get("profit", 0) > 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": len(self.trades),
        }


class MeanReversionStrategy(TradingStrategy):
    """
    Mean reversion trading strategy.

    Features:
    - Bollinger Bands mean reversion
    - RSI mean reversion
    - Statistical arbitrage
    - Pairs trading
    """

    def __init__(
        self,
        name: str = "MeanReversion",
        initial_capital: float = 100000,
        lookback_period: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
    ):
        """
        Initialize mean reversion strategy.

        Args:
            name: Strategy name
            initial_capital: Initial capital
            lookback_period: Lookback period for mean/std calculation
            entry_threshold: Entry threshold in standard deviations
            exit_threshold: Exit threshold in standard deviations
        """
        super().__init__(name, initial_capital)
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mean reversion signals using Bollinger Bands.

        Args:
            data: Market data with OHLCV

        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index, columns=data.columns, data=0.0)

        for symbol in data.columns:
            if isinstance(data[symbol], pd.DataFrame):
                # Handle OHLCV data
                prices = (
                    data[symbol]["close"]
                    if "close" in data[symbol].columns
                    else data[symbol].iloc[:, 0]
                )
            else:
                prices = data[symbol]

            # Calculate rolling mean and standard deviation
            rolling_mean = prices.rolling(window=self.lookback_period).mean()
            rolling_std = prices.rolling(window=self.lookback_period).std()

            # Calculate Bollinger Bands
            upper_band = rolling_mean + (rolling_std * self.entry_threshold)
            lower_band = rolling_mean - (rolling_std * self.entry_threshold)
            exit_upper = rolling_mean + (rolling_std * self.exit_threshold)
            exit_lower = rolling_mean - (rolling_std * self.exit_threshold)

            # Generate signals
            # Buy when price crosses below lower band
            buy_signal = (prices < lower_band) & (
                prices.shift(1) >= lower_band.shift(1)
            )
            # Sell when price crosses above upper band
            sell_signal = (prices > upper_band) & (
                prices.shift(1) <= upper_band.shift(1)
            )
            # Exit long when price crosses above exit upper band
            exit_long = (prices > exit_upper) & (prices.shift(1) <= exit_upper.shift(1))
            # Exit short when price crosses below exit lower band
            exit_short = (prices < exit_lower) & (
                prices.shift(1) >= exit_lower.shift(1)
            )

            # Combine signals
            signals[symbol] = np.where(
                buy_signal,
                1.0,
                np.where(
                    sell_signal,
                    -1.0,
                    np.where(exit_long, 0.0, np.where(exit_short, 0.0, 0.0)),
                ),
            )

        return signals

    def pairs_trading_signals(
        self, data: pd.DataFrame, pair: Tuple[str, str]
    ) -> pd.Series:
        """
        Generate pairs trading signals.

        Args:
            data: Price data for both assets
            pair: Tuple of two symbols to pair trade

        Returns:
            Series with pair trading signals
        """
        symbol1, symbol2 = pair

        if symbol1 not in data.columns or symbol2 not in data.columns:
            return pd.Series(index=data.index, data=0.0)

        # Calculate price ratio
        ratio = data[symbol1] / data[symbol2]

        # Calculate rolling mean and std of ratio
        ratio_mean = ratio.rolling(window=self.lookback_period).mean()
        ratio_std = ratio.rolling(window=self.lookback_period).std()

        # Calculate z-score of ratio
        z_score = (ratio - ratio_mean) / ratio_std

        # Generate signals
        signals = pd.Series(index=data.index, data=0.0)

        # Long pair (buy symbol1, sell symbol2) when z-score is low
        signals[z_score < -self.entry_threshold] = 1.0
        # Short pair (sell symbol1, buy symbol2) when z-score is high
        signals[z_score > self.entry_threshold] = -1.0
        # Exit when z-score returns to normal
        signals[(z_score > -self.exit_threshold) & (z_score < self.exit_threshold)] = (
            0.0
        )

        return signals


class MomentumStrategy(TradingStrategy):
    """
    Momentum trading strategy.

    Features:
    - Price momentum
    - Earnings momentum
    - Sector momentum
    - Time series momentum
    """

    def __init__(
        self,
        name: str = "Momentum",
        initial_capital: float = 100000,
        lookback_period: int = 252,
        formation_period: int = 63,
        holding_period: int = 21,
    ):
        """
        Initialize momentum strategy.

        Args:
            name: Strategy name
            initial_capital: Initial capital
            lookback_period: Lookback period for momentum calculation
            formation_period: Formation period for signal generation
            holding_period: Holding period after signal generation
        """
        super().__init__(name, initial_capital)
        self.lookback_period = lookback_period
        self.formation_period = formation_period
        self.holding_period = holding_period
        self.last_signal_date = {}

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate momentum signals.

        Args:
            data: Market data

        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index, columns=data.columns, data=0.0)

        for symbol in data.columns:
            if isinstance(data[symbol], pd.DataFrame):
                prices = (
                    data[symbol]["close"]
                    if "close" in data[symbol].columns
                    else data[symbol].iloc[:, 0]
                )
            else:
                prices = data[symbol]

            # Calculate momentum (past returns)
            momentum = prices.pct_change(period=self.lookback_period)

            # Calculate momentum strength (ranking)
            momentum_rank = momentum.rolling(window=self.formation_period).rank(
                pct=True
            )

            # Generate signals based on momentum strength
            # Buy top momentum (top 20%)
            buy_signal = momentum_rank > 0.8
            # Sell bottom momentum (bottom 20%)
            sell_signal = momentum_rank < 0.2

            # Apply holding period constraint
            if symbol in self.last_signal_date:
                days_since_signal = (data.index - self.last_signal_date[symbol]).days
                can_trade = days_since_signal >= self.holding_period
                buy_signal = buy_signal & can_trade
                sell_signal = sell_signal & can_trade

            # Update last signal date
            new_signals = buy_signal | sell_signal
            if new_signals.any():
                self.last_signal_date[symbol] = data.index[new_signals].max()

            signals[symbol] = np.where(
                buy_signal, 1.0, np.where(sell_signal, -1.0, 0.0)
            )

        return signals

    def sector_momentum_signals(
        self, data: pd.DataFrame, sector_mapping: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Generate sector momentum signals.

        Args:
            data: Market data
            sector_mapping: Mapping from symbols to sectors

        Returns:
            DataFrame with sector momentum signals
        """
        # Calculate sector returns
        sector_returns = {}
        for sector, symbols in sector_mapping.items():
            sector_symbols = [s for s in symbols if s in data.columns]
            if sector_symbols:
                sector_data = data[sector_symbols]
                sector_returns[sector] = sector_data.pct_change().mean(axis=1)

        sector_returns_df = pd.DataFrame(sector_returns)

        # Calculate sector momentum
        sector_momentum = sector_returns_df.rolling(window=self.lookback_period).mean()

        # Rank sectors by momentum
        sector_rank = sector_momentum.rank(axis=1, pct=True)

        # Generate signals for top and bottom sectors
        signals = pd.DataFrame(index=data.index, columns=data.columns, data=0.0)

        for date in data.index:
            if date in sector_rank.index:
                top_sectors = sector_rank.loc[date][sector_rank.loc[date] > 0.7].index
                bottom_sectors = sector_rank.loc[date][
                    sector_rank.loc[date] < 0.3
                ].index

                for sector in top_sectors:
                    sector_symbols = [
                        s for s in sector_mapping[sector] if s in data.columns
                    ]
                    for symbol in sector_symbols:
                        signals.loc[date, symbol] = 1.0

                for sector in bottom_sectors:
                    sector_symbols = [
                        s for s in sector_mapping[sector] if s in data.columns
                    ]
                    for symbol in sector_symbols:
                        signals.loc[date, symbol] = -1.0

        return signals


class MachineLearningStrategy(TradingStrategy):
    """
    Machine learning-based trading strategy.

    Features:
    - Random Forest classifier
    - Gradient Boosting
    - Neural networks
    - Feature engineering
    - Model training and validation
    """

    def __init__(
        self,
        name: str = "MLStrategy",
        initial_capital: float = 100000,
        model_type: str = "RandomForest",
        lookback_period: int = 252,
    ):
        """
        Initialize machine learning strategy.

        Args:
            name: Strategy name
            initial_capital: Initial capital
            model_type: Type of ML model
            lookback_period: Lookback period for features
        """
        super().__init__(name, initial_capital)
        self.model_type = model_type
        self.lookback_period = lookback_period
        self.models = {}
        self.scalers = {}
        self.feature_columns = []

        if not SKLEARN_AVAILABLE:
            print("Warning: sklearn not available, using simplified ML approach")

    def create_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Create features for machine learning.

        Args:
            data: Market data
            symbol: Symbol to create features for

        Returns:
            DataFrame with features
        """
        if isinstance(data[symbol], pd.DataFrame):
            prices = (
                data[symbol]["close"]
                if "close" in data[symbol].columns
                else data[symbol].iloc[:, 0]
            )
            volumes = (
                data[symbol]["volume"] if "volume" in data[symbol].columns else None
            )
        else:
            prices = data[symbol]
            volumes = None

        features = pd.DataFrame(index=prices.index)

        # Price-based features
        features["returns"] = prices.pct_change()
        features["returns_5"] = prices.pct_change(5)
        features["returns_20"] = prices.pct_change(20)
        features["returns_60"] = prices.pct_change(60)

        # Moving averages
        features["ma_5"] = prices.rolling(5).mean()
        features["ma_20"] = prices.rolling(20).mean()
        features["ma_60"] = prices.rolling(60).mean()

        # Price relative to moving averages
        features["price_ma_5_ratio"] = prices / features["ma_5"]
        features["price_ma_20_ratio"] = prices / features["ma_20"]
        features["price_ma_60_ratio"] = prices / features["ma_60"]

        # Volatility features
        features["volatility_5"] = features["returns"].rolling(5).std()
        features["volatility_20"] = features["returns"].rolling(20).std()
        features["volatility_60"] = features["returns"].rolling(60).std()

        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features["rsi"] = 100 - (100 / (1 + rs))

        # Volume features (if available)
        if volumes is not None:
            features["volume"] = volumes
            features["volume_ma_5"] = volumes.rolling(5).mean()
            features["volume_ratio"] = volumes / features["volume_ma_5"]

        # Lagged returns
        for lag in [1, 2, 3, 5, 10]:
            features[f"return_lag_{lag}"] = features["returns"].shift(lag)

        return features

    def create_labels(
        self, data: pd.DataFrame, symbol: str, horizon: int = 5
    ) -> pd.Series:
        """
        Create labels for supervised learning.

        Args:
            data: Market data
            symbol: Symbol to create labels for
            horizon: Return horizon for labels

        Returns:
            Series with labels
        """
        if isinstance(data[symbol], pd.DataFrame):
            prices = (
                data[symbol]["close"]
                if "close" in data[symbol].columns
                else data[symbol].iloc[:, 0]
            )
        else:
            prices = data[symbol]

        future_returns = prices.pct_change(horizon).shift(-horizon)

        # Create binary labels: 1 for positive returns, 0 for negative
        labels = (future_returns > 0).astype(int)

        return labels

    def train_model(self, data: pd.DataFrame, symbol: str):
        """
        Train machine learning model.

        Args:
            data: Training data
            symbol: Symbol to train model for
        """
        if not SKLEARN_AVAILABLE:
            # Simplified approach without sklearn
            self.models[symbol] = {"type": "simple", "threshold": 0.0}
            return

        # Create features and labels
        features = self.create_features(data, symbol)
        labels = self.create_labels(data, symbol)

        # Remove NaN values
        valid_data = features.dropna()
        valid_labels = labels.loc[valid_data.index]

        if len(valid_data) < 100:
            return  # Not enough data

        # Split into train and validation
        split_point = int(len(valid_data) * 0.8)
        X_train = valid_data.iloc[:split_point]
        y_train = valid_labels.iloc[:split_point]
        X_val = valid_data.iloc[split_point:]
        y_val = valid_labels.iloc[split_point:]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train model
        if self.model_type == "RandomForest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == "GradientBoosting":
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:  # Logistic Regression
            model = LogisticRegression(random_state=42, max_iter=1000)

        model.fit(X_train_scaled, y_train)

        # Validate model
        train_score = model.score(X_train_scaled, y_train)
        val_score = model.score(X_val_scaled, y_val)

        # Store model and scaler
        self.models[symbol] = {
            "model": model,
            "scaler": scaler,
            "feature_columns": valid_data.columns.tolist(),
            "train_score": train_score,
            "val_score": val_score,
        }

        self.scalers[symbol] = scaler

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals using trained models.

        Args:
            data: Market data

        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index, columns=data.columns, data=0.0)

        for symbol in data.columns:
            if symbol not in self.models:
                continue

            # Create features
            features = self.create_features(data, symbol)

            if not SKLEARN_AVAILABLE:
                # Simple momentum-based signals
                if isinstance(data[symbol], pd.DataFrame):
                    prices = (
                        data[symbol]["close"]
                        if "close" in data[symbol].columns
                        else data[symbol].iloc[:, 0]
                    )
                else:
                    prices = data[symbol]

                returns = prices.pct_change(20)
                signals[symbol] = np.where(
                    returns > 0.02, 1.0, np.where(returns < -0.02, -1.0, 0.0)
                )
                continue

            # Use trained model
            model_info = self.models[symbol]
            model = model_info["model"]
            scaler = model_info["scaler"]
            feature_cols = model_info["feature_columns"]

            # Get valid features
            valid_features = features[feature_cols].dropna()

            if len(valid_features) == 0:
                continue

            # Scale features
            X_scaled = scaler.transform(valid_features)

            # Predict
            predictions = model.predict_proba(X_scaled)[
                :, 1
            ]  # Probability of positive return

            # Generate signals based on predictions
            threshold = 0.6  # High confidence threshold
            signals.loc[valid_features.index, symbol] = np.where(
                predictions > threshold,
                1.0,
                np.where(predictions < (1 - threshold), -1.0, 0.0),
            )

        return signals


class HighFrequencyStrategy(TradingStrategy):
    """
    High-frequency trading strategy.

    Features:
    - Order book analysis
    - Microstructure modeling
    - Latency arbitrage
    - Market making
    """

    def __init__(
        self,
        name: str = "HFTStrategy",
        initial_capital: float = 100000,
        max_position_size: float = 1000,
        inventory_target: float = 0.0,
    ):
        """
        Initialize high-frequency strategy.

        Args:
            name: Strategy name
            initial_capital: Initial capital
            max_position_size: Maximum position size
            inventory_target: Target inventory level
        """
        super().__init__(name, initial_capital)
        self.max_position_size = max_position_size
        self.inventory_target = inventory_target
        self.order_book = {}
        self.spreads = {}

    def update_order_book(
        self,
        symbol: str,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        timestamp: datetime,
    ):
        """
        Update order book for a symbol.

        Args:
            symbol: Trading symbol
            bids: List of (price, quantity) bids
            asks: List of (price, quantity) asks
            timestamp: Update timestamp
        """
        self.order_book[symbol] = {"bids": bids, "asks": asks, "timestamp": timestamp}

        # Calculate spread
        if bids and asks:
            best_bid = max(bids, key=lambda x: x[0])[0]
            best_ask = min(asks, key=lambda x: x[0])[0]
            self.spreads[symbol] = best_ask - best_bid

    def market_making_signals(self, symbol: str) -> Dict[str, float]:
        """
        Generate market making signals.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with market making signals
        """
        if symbol not in self.order_book or symbol not in self.spreads:
            return {"bid": 0.0, "ask": 0.0, "spread": 0.0}

        order_book = self.order_book[symbol]
        spread = self.spreads[symbol]

        # Get best bid and ask
        best_bid = max(order_book["bids"], key=lambda x: x[0])[0]
        best_ask = min(order_book["asks"], key=lambda x: x[0])[0]

        # Calculate our quotes (simplified)
        our_bid = best_bid - spread * 0.1
        our_ask = best_ask + spread * 0.1

        # Adjust for inventory
        current_position = self.positions.get(symbol, 0)
        inventory_skew = (
            current_position - self.inventory_target
        ) / self.max_position_size

        # Widen spreads if we have inventory
        if inventory_skew > 0:  # Long inventory
            our_bid -= spread * 0.2 * inventory_skew
            our_ask += spread * 0.1 * inventory_skew
        else:  # Short inventory
            our_bid -= spread * 0.1 * abs(inventory_skew)
            our_ask += spread * 0.2 * abs(inventory_skew)

        return {
            "bid": our_bid,
            "ask": our_ask,
            "spread": our_ask - our_bid,
            "inventory_skew": inventory_skew,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate high-frequency signals.

        Args:
            data: Market data (not typically used in HFT)

        Returns:
            DataFrame with signals
        """
        # HFT strategies typically use order book data, not price data
        # This is a placeholder implementation
        signals = pd.DataFrame(index=data.index, columns=data.columns, data=0.0)

        for symbol in data.columns:
            if symbol in self.order_book:
                self.market_making_signals(symbol)
                # Simplified: place orders at bid/ask
                # In practice, this would be more sophisticated
                signals[symbol] = 0.0  # Market making doesn't use traditional signals

        return signals


class BacktestEngine:
    """
    Comprehensive backtesting engine.

    Features:
    - Strategy backtesting
    - Performance analysis
    - Risk metrics
    - Benchmark comparison
    - Transaction costs
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0001,
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Initial capital
            commission_rate: Commission rate per trade
            slippage_rate: Slippage rate per trade
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.results = {}

    def run_backtest(
        self, strategy: TradingStrategy, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Run backtest for a strategy.

        Args:
            strategy: Trading strategy to backtest
            data: Market data

        Returns:
            Backtest results
        """
        # Reset strategy
        strategy.current_capital = self.initial_capital
        strategy.positions = {}
        strategy.trades = []

        # Generate signals
        signals = strategy.generate_signals(data)

        # Execute trades
        for timestamp in signals.index:
            for symbol in signals.columns:
                signal = signals.loc[timestamp, symbol]

                if signal != 0:
                    # Get price
                    if isinstance(data[symbol], pd.DataFrame):
                        price = (
                            data[symbol].loc[timestamp, "close"]
                            if "close" in data[symbol].columns
                            else data[symbol].loc[timestamp, 0]
                        )
                    else:
                        price = data[symbol].loc[timestamp]

                    # Apply slippage
                    if signal > 0:  # Buy
                        execution_price = price * (1 + self.slippage_rate)
                    else:  # Sell
                        execution_price = price * (1 - self.slippage_rate)

                    # Execute trade
                    strategy.execute_trade(symbol, signal, execution_price, timestamp)

        # Calculate performance
        performance = strategy.calculate_performance(data)

        # Additional analysis
        trades_df = pd.DataFrame(strategy.trades)

        if not trades_df.empty:
            # Calculate trade-level metrics
            trades_df["profit"] = 0.0

            # Pair buy and sell trades
            for symbol in trades_df["symbol"].unique():
                symbol_trades = trades_df[trades_df["symbol"] == symbol].sort_values(
                    "timestamp"
                )

                position = 0
                for i, trade in symbol_trades.iterrows():
                    if trade["side"] == "BUY":
                        position += trade["quantity"]
                    else:  # SELL
                        if position > 0:
                            # Calculate profit for this position
                            avg_cost = (
                                symbol_trades[
                                    (symbol_trades["symbol"] == symbol)
                                    & (symbol_trades["side"] == "BUY")
                                ]["cost"].sum()
                                / position
                            )
                            profit = (trade["price"] - avg_cost) * min(
                                position, trade["quantity"]
                            )
                            trades_df.loc[i, "profit"] = profit
                            position -= trade["quantity"]

            total_profit = trades_df["profit"].sum()
            profit_trades = trades_df[trades_df["profit"] > 0]
            loss_trades = trades_df[trades_df["profit"] < 0]

            profit_factor = (
                profit_trades["profit"].sum() / abs(loss_trades["profit"].sum())
                if not loss_trades.empty
                else float("inf")
            )
            avg_win = profit_trades["profit"].mean() if not profit_trades.empty else 0
            avg_loss = abs(loss_trades["profit"].mean()) if not loss_trades.empty else 0
        else:
            total_profit = 0
            profit_factor = 0
            avg_win = 0
            avg_loss = 0

        self.results[strategy.name] = {
            "performance": performance,
            "trades": trades_df,
            "total_profit": total_profit,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_trades": len(trades_df),
            "winning_trades": (
                len(trades_df[trades_df["profit"] > 0]) if not trades_df.empty else 0
            ),
        }

        return self.results[strategy.name]


# Utility functions
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate

    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / 252
    return (
        excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        if excess_returns.std() > 0
        else 0
    )


def calculate_information_ratio(
    returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    """
    Calculate Information Ratio.

    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns

    Returns:
        Information ratio
    """
    active_returns = returns - benchmark_returns
    return (
        active_returns.mean() / active_returns.std() * np.sqrt(252)
        if active_returns.std() > 0
        else 0
    )


def calculate_calmar_ratio(returns: pd.Series) -> float:
    """
    Calculate Calmar ratio.

    Args:
        returns: Series of returns

    Returns:
        Calmar ratio
    """
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    annual_return = returns.mean() * 252
    return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0


# Export main classes and functions
__all__ = [
    "TradingStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "MachineLearningStrategy",
    "HighFrequencyStrategy",
    "BacktestEngine",
    "calculate_sharpe_ratio",
    "calculate_information_ratio",
    "calculate_calmar_ratio",
]
