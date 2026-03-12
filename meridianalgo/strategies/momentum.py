"""
Momentum-based trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import pandas as pd

from ..core import calculate_macd, calculate_returns, calculate_rsi


class BaseStrategy(ABC):
    """Base class for trading strategies."""

    def __init__(self, name: str):
        """Initialize strategy with name."""
        self.name = name
        self.positions = None
        self.signals = None

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from market data."""
        pass

    @abstractmethod
    def calculate_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Calculate positions from signals."""
        pass

    def backtest(
        self, data: pd.DataFrame, initial_capital: float = 100000
    ) -> Dict[str, pd.Series]:
        """Run backtest on historical data."""
        # Generate signals
        self.signals = self.generate_signals(data)

        # Calculate positions
        self.positions = self.calculate_positions(self.signals)

        # Calculate returns
        returns = calculate_returns(data)

        # Calculate strategy returns
        strategy_returns = (self.positions.shift(1) * returns).sum(axis=1)

        # Calculate portfolio value
        portfolio_value = initial_capital * (1 + strategy_returns).cumprod()

        return {
            "returns": strategy_returns,
            "portfolio_value": portfolio_value,
            "signals": self.signals,
            "positions": self.positions,
        }


class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy."""

    def __init__(
        self,
        lookback_period: int = 252,
        formation_period: int = 63,
        holding_period: int = 21,
        n_assets: int = 10,
        long_short: bool = True,
    ):
        """Initialize momentum strategy.

        Args:
            lookback_period: Period for calculating momentum
            formation_period: Period for forming portfolios
            holding_period: Period to hold positions
            n_assets: Number of assets to trade
            long_short: Whether to use long-short or long-only
        """
        super().__init__("Momentum")
        self.lookback_period = lookback_period
        self.formation_period = formation_period
        self.holding_period = holding_period
        self.n_assets = n_assets
        self.long_short = long_short

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum signals."""
        # Calculate returns over lookback period
        returns = calculate_returns(data)

        # Calculate cumulative returns over lookback period
        momentum = returns.rolling(window=self.lookback_period).apply(
            lambda x: (1 + x).prod() - 1, raw=False
        )

        # Rank assets by momentum
        ranked = momentum.rank(axis=1, ascending=False)

        # Generate signals
        signals = pd.DataFrame(0, index=momentum.index, columns=data.columns)

        for i in range(self.lookback_period, len(momentum)):
            # Get top n_assets for long
            top_assets = ranked.iloc[i].nlargest(self.n_assets).index

            # Get bottom n_assets for short (if long-short)
            if self.long_short:
                bottom_assets = ranked.iloc[i].nsmallest(self.n_assets).index
                signals.loc[momentum.index[i], bottom_assets] = -1

            signals.loc[momentum.index[i], top_assets] = 1

        return signals

    def calculate_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Calculate positions from signals with rebalancing."""
        positions = pd.DataFrame(0, index=signals.index, columns=signals.columns)

        for i in range(self.holding_period, len(signals)):
            # Check if it's a rebalancing date
            if (i - self.holding_period) % self.formation_period == 0:
                # Get current signals
                current_signals = signals.iloc[i]

                # Calculate equal weights
                long_assets = current_signals[current_signals > 0]
                short_assets = current_signals[current_signals < 0]

                if len(long_assets) > 0:
                    long_weight = 1.0 / len(long_assets)
                    positions.loc[signals.index[i], long_assets.index] = long_weight

                if self.long_short and len(short_assets) > 0:
                    short_weight = -1.0 / len(short_assets)
                    positions.loc[signals.index[i], short_assets.index] = short_weight
            else:
                # Carry forward previous positions
                positions.iloc[i] = positions.iloc[i - 1]

        return positions


class RSIMeanReversion(BaseStrategy):
    """RSI-based mean reversion strategy."""

    def __init__(
        self,
        rsi_period: int = 14,
        oversold_threshold: float = 30,
        overbought_threshold: float = 70,
        exit_threshold: float = 50,
    ):
        """Initialize RSI mean reversion strategy.

        Args:
            rsi_period: Period for RSI calculation
            oversold_threshold: RSI level for oversold condition
            overbought_threshold: RSI level for overbought condition
            exit_threshold: RSI level for exit condition
        """
        super().__init__("RSI Mean Reversion")
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.exit_threshold = exit_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI-based signals."""
        signals = pd.DataFrame(0, index=data.index, columns=data.columns)

        for asset in data.columns:
            prices = data[asset]
            rsi = calculate_rsi(prices, window=self.rsi_period)

            # Generate signals based on RSI levels
            signals.loc[rsi < self.oversold_threshold, asset] = 1  # Buy when oversold
            signals.loc[
                rsi > self.overbought_threshold, asset
            ] = -1  # Sell when overbought
            signals.loc[
                (rsi >= self.exit_threshold) & (rsi <= 100 - self.exit_threshold), asset
            ] = 0  # Exit at neutral

        return signals

    def calculate_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Calculate positions from signals."""
        # Simple position calculation - equal weight for all active signals
        positions = signals.copy()

        # Normalize positions to equal weight
        for i in range(len(positions)):
            active_signals = positions.iloc[i][positions.iloc[i] != 0]
            if len(active_signals) > 0:
                weight = 1.0 / len(active_signals)
                positions.iloc[i] = positions.iloc[i] * weight

        return positions


class MACDCrossover(BaseStrategy):
    """MACD crossover strategy."""

    def __init__(
        self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ):
        """Initialize MACD crossover strategy.

        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
        """
        super().__init__("MACD Crossover")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate MACD crossover signals."""
        signals = pd.DataFrame(0, index=data.index, columns=data.columns)

        for asset in data.columns:
            prices = data[asset]
            macd_data = calculate_macd(
                prices,
                fast_period=self.fast_period,
                slow_period=self.slow_period,
                signal_period=self.signal_period,
            )

            # Generate crossover signals
            macd_line = macd_data["macd"]
            signal_line = macd_data["signal"]

            # Buy signal: MACD crosses above signal
            buy_signal = (macd_line > signal_line) & (
                macd_line.shift(1) <= signal_line.shift(1)
            )

            # Sell signal: MACD crosses below signal
            sell_signal = (macd_line < signal_line) & (
                macd_line.shift(1) >= signal_line.shift(1)
            )

            signals.loc[buy_signal, asset] = 1
            signals.loc[sell_signal, asset] = -1

        return signals

    def calculate_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Calculate positions from signals."""
        # Hold position until opposite signal
        positions = signals.copy()

        for asset in positions.columns:
            # Forward fill positions
            positions[asset] = positions[asset].replace(0, method="ffill").fillna(0)

        return positions


class PairsTrading(BaseStrategy):
    """Statistical pairs trading strategy."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        lookback_period: int = 252,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
    ):
        """Initialize pairs trading strategy.

        Args:
            pairs: List of asset pairs to trade
            lookback_period: Period for calculating spread statistics
            z_entry: Z-score threshold for entry
            z_exit: Z-score threshold for exit
        """
        super().__init__("Pairs Trading")
        self.pairs = pairs
        self.lookback_period = lookback_period
        self.z_entry = z_entry
        self.z_exit = z_exit

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate pairs trading signals."""
        signals = pd.DataFrame(0, index=data.index, columns=data.columns)

        for asset1, asset2 in self.pairs:
            if asset1 in data.columns and asset2 in data.columns:
                # Calculate spread
                prices1 = data[asset1]
                prices2 = data[asset2]

                # Calculate hedge ratio using rolling regression
                spread = self._calculate_spread(prices1, prices2)

                # Calculate z-score of spread
                spread_mean = spread.rolling(window=self.lookback_period).mean()
                spread_std = spread.rolling(window=self.lookback_period).std()
                z_score = (spread - spread_mean) / spread_std

                # Generate signals
                # Long spread (buy asset1, sell asset2) when z-score is negative
                long_spread = z_score < -self.z_entry
                # Short spread (sell asset1, buy asset2) when z-score is positive
                short_spread = z_score > self.z_entry

                # Exit positions when z-score reverts
                exit_long = (z_score > -self.z_exit) & (z_score < self.z_exit)
                exit_short = (z_score > -self.z_exit) & (z_score < self.z_exit)

                # Apply signals
                signals.loc[long_spread, asset1] = 1
                signals.loc[long_spread, asset2] = -1
                signals.loc[short_spread, asset1] = -1
                signals.loc[short_spread, asset2] = 1
                signals.loc[exit_long, [asset1, asset2]] = 0
                signals.loc[exit_short, [asset1, asset2]] = 0

        return signals

    def _calculate_spread(self, prices1: pd.Series, prices2: pd.Series) -> pd.Series:
        """Calculate spread between two price series."""
        # Simple ratio spread
        return prices1 / prices2

    def calculate_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Calculate positions from signals."""
        # Forward fill positions
        positions = signals.copy()

        for asset in positions.columns:
            positions[asset] = positions[asset].replace(0, method="ffill").fillna(0)

        return positions


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands mean reversion strategy."""

    def __init__(
        self, window: int = 20, num_std: float = 2.0, exit_at_middle: bool = True
    ):
        """Initialize Bollinger Bands strategy.

        Args:
            window: Window for moving average and standard deviation
            num_std: Number of standard deviations for bands
            exit_at_middle: Whether to exit at middle band or opposite band
        """
        super().__init__("Bollinger Bands")
        self.window = window
        self.num_std = num_std
        self.exit_at_middle = exit_at_middle

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Bollinger Bands signals."""
        signals = pd.DataFrame(0, index=data.index, columns=data.columns)

        for asset in data.columns:
            prices = data[asset]

            # Calculate Bollinger Bands
            ma = prices.rolling(window=self.window).mean()
            std = prices.rolling(window=self.window).std()
            upper_band = ma + (std * self.num_std)
            lower_band = ma - (std * self.num_std)

            # Generate signals
            # Buy when price touches lower band
            buy_signal = prices <= lower_band
            # Sell when price touches upper band
            sell_signal = prices >= upper_band

            # Exit conditions
            if self.exit_at_middle:
                exit_buy = prices >= ma
                exit_sell = prices <= ma
            else:
                exit_buy = prices >= upper_band
                exit_sell = prices <= lower_band

            # Apply signals
            signals.loc[buy_signal, asset] = 1
            signals.loc[sell_signal, asset] = -1
            signals.loc[exit_buy, asset] = 0
            signals.loc[exit_sell, asset] = 0

        return signals

    def calculate_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Calculate positions from signals."""
        # Forward fill positions
        positions = signals.copy()

        for asset in positions.columns:
            positions[asset] = positions[asset].replace(0, method="ffill").fillna(0)

        return positions


# Strategy factory for easy creation
def create_strategy(strategy_type: str, **kwargs) -> BaseStrategy:
    """Create a strategy instance.

    Args:
        strategy_type: Type of strategy to create
        **kwargs: Strategy-specific parameters

    Returns:
        Strategy instance
    """
    strategies = {
        "momentum": MomentumStrategy,
        "rsi_mean_reversion": RSIMeanReversion,
        "macd_crossover": MACDCrossover,
        "pairs_trading": PairsTrading,
        "bollinger_bands": BollingerBandsStrategy,
    }

    if strategy_type not in strategies:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    return strategies[strategy_type](**kwargs)
