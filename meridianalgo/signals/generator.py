"""
Signal Generator Module

Generate trading signals from technical indicators and custom rules.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd


class SignalGenerator:
    """
    Generate trading signals from multiple indicators.

    Combines multiple technical indicators into a unified
    signal with configurable weights and thresholds.

    Example:
        >>> gen = SignalGenerator(data)
        >>> gen.add_rule('rsi_oversold', lambda d: d['rsi'] < 30, weight=1.5)
        >>> gen.add_rule('macd_cross', lambda d: d['macd'] > d['signal'], weight=1.0)
        >>> signals = gen.generate()
    """

    def __init__(self, data: pd.DataFrame, default_threshold: float = 0.5):
        """
        Initialize SignalGenerator.

        Args:
            data: DataFrame with OHLCV and indicator data
            default_threshold: Default threshold for signal generation
        """
        self.data = data.copy()
        self.rules: List[Dict[str, Any]] = []
        self.default_threshold = default_threshold

    def add_rule(
        self,
        name: str,
        condition: Callable[[pd.DataFrame], pd.Series],
        weight: float = 1.0,
        signal_type: str = "long",  # 'long', 'short', or 'both'
    ):
        """
        Add a trading rule.

        Args:
            name: Rule name
            condition: Function that returns boolean Series
            weight: Rule weight (higher = more important)
            signal_type: Type of signal ('long', 'short', 'both')
        """
        self.rules.append(
            {
                "name": name,
                "condition": condition,
                "weight": weight,
                "signal_type": signal_type,
            }
        )

    def generate(self, threshold: float = None) -> pd.DataFrame:
        """
        Generate trading signals.

        Args:
            threshold: Signal threshold (default: self.default_threshold)

        Returns:
            DataFrame with signals and scores
        """
        threshold = threshold or self.default_threshold

        long_score = pd.Series(0.0, index=self.data.index)
        short_score = pd.Series(0.0, index=self.data.index)
        total_long_weight = 0
        total_short_weight = 0

        for rule in self.rules:
            try:
                condition_result = rule["condition"](self.data).astype(float)
            except Exception:
                condition_result = pd.Series(0.0, index=self.data.index)

            if rule["signal_type"] in ["long", "both"]:
                long_score += condition_result * rule["weight"]
                total_long_weight += rule["weight"]

            if rule["signal_type"] in ["short", "both"]:
                short_score += (~condition_result.astype(bool)).astype(float) * rule[
                    "weight"
                ]
                total_short_weight += rule["weight"]

        # Normalize scores
        if total_long_weight > 0:
            long_score = long_score / total_long_weight
        if total_short_weight > 0:
            short_score = short_score / total_short_weight

        # Generate signals
        result = pd.DataFrame(
            {"long_score": long_score, "short_score": short_score, "signal": 0},
            index=self.data.index,
        )

        result.loc[long_score >= threshold, "signal"] = 1
        result.loc[short_score >= threshold, "signal"] = -1

        return result

    def backtest_signals(
        self, signals: pd.DataFrame, returns: pd.Series, transaction_cost: float = 0.001
    ) -> Dict[str, Any]:
        """
        Backtest generated signals.

        Args:
            signals: Signal DataFrame from generate()
            returns: Asset returns
            transaction_cost: Transaction cost per trade

        Returns:
            Dictionary with backtest results
        """
        # Calculate strategy returns
        position = signals["signal"]
        trades = position.diff().abs()
        costs = trades * transaction_cost

        strategy_returns = position.shift(1) * returns - costs
        strategy_returns = strategy_returns.fillna(0)

        # Calculate metrics
        total_return = (1 + strategy_returns).prod() - 1
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)

        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.cummax()
        max_dd = ((cumulative - running_max) / running_max).min()

        return {
            "total_return": total_return,
            "annualized_return": (1 + total_return) ** (252 / len(returns)) - 1,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": (strategy_returns > 0).mean(),
            "num_trades": trades.sum() / 2,
            "avg_trade_return": strategy_returns[trades.shift(1) > 0].mean(),
        }


class TechnicalAnalyzer:
    """
    Comprehensive technical analysis with automatic indicator calculation.

    Calculates all major technical indicators and provides
    summary analysis and signals.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: Optional[pd.Series] = None,
    ):
        """
        Initialize TechnicalAnalyzer.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Trading volume (optional)
        """
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

        self._indicators_calculated = False
        self.indicators = {}

    def calculate_all(self) -> pd.DataFrame:
        """
        Calculate all technical indicators.

        Returns:
            DataFrame with all indicators
        """
        from .indicators import (
            ADX,
            ATR,
            EMA,
            MACD,
            OBV,
            RSI,
            SMA,
            BollingerBands,
            Stochastic,
        )

        df = pd.DataFrame(index=self.close.index)

        # Moving Averages
        df["sma_20"] = SMA(self.close, 20)
        df["sma_50"] = SMA(self.close, 50)
        df["sma_200"] = SMA(self.close, 200)
        df["ema_12"] = EMA(self.close, 12)
        df["ema_26"] = EMA(self.close, 26)

        # MACD
        macd, signal, hist = MACD(self.close)
        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_hist"] = hist

        # RSI
        df["rsi"] = RSI(self.close, 14)

        # Bollinger Bands
        upper, middle, lower = BollingerBands(self.close)
        df["bb_upper"] = upper
        df["bb_middle"] = middle
        df["bb_lower"] = lower
        df["bb_pct"] = (self.close - lower) / (upper - lower)

        # ATR
        df["atr"] = ATR(self.high, self.low, self.close, 14)
        df["atr_pct"] = df["atr"] / self.close

        # Stochastic
        k, d = Stochastic(self.high, self.low, self.close)
        df["stoch_k"] = k
        df["stoch_d"] = d

        # ADX
        adx, plus_di, minus_di = ADX(self.high, self.low, self.close)
        df["adx"] = adx
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di

        # Volume indicators
        if self.volume is not None:
            df["obv"] = OBV(self.close, self.volume)
            df["volume_sma"] = SMA(self.volume, 20)
            df["relative_volume"] = self.volume / df["volume_sma"]

        self.indicators = df
        self._indicators_calculated = True

        return df

    def get_signals(self) -> pd.DataFrame:
        """
        Generate signals based on indicator values.

        Returns:
            DataFrame with signals for each indicator
        """
        if not self._indicators_calculated:
            self.calculate_all()

        df = self.indicators.copy()
        signals = pd.DataFrame(index=df.index)

        # RSI signals
        signals["rsi_signal"] = 0
        signals.loc[df["rsi"] < 30, "rsi_signal"] = 1  # Oversold
        signals.loc[df["rsi"] > 70, "rsi_signal"] = -1  # Overbought

        # MACD signals
        signals["macd_signal"] = np.where(df["macd"] > df["macd_signal"], 1, -1)

        # Bollinger Band signals
        signals["bb_signal"] = 0
        signals.loc[df["bb_pct"] < 0, "bb_signal"] = 1  # Below lower band
        signals.loc[df["bb_pct"] > 1, "bb_signal"] = -1  # Above upper band

        # Stochastic signals
        signals["stoch_signal"] = 0
        signals.loc[(df["stoch_k"] < 20) & (df["stoch_d"] < 20), "stoch_signal"] = 1
        signals.loc[(df["stoch_k"] > 80) & (df["stoch_d"] > 80), "stoch_signal"] = -1

        # Trend signals (MA cross)
        signals["trend_signal"] = np.where(
            (df["sma_20"] > df["sma_50"]) & (df["sma_50"] > df["sma_200"]),
            1,
            np.where(
                (df["sma_20"] < df["sma_50"]) & (df["sma_50"] < df["sma_200"]), -1, 0
            ),
        )

        # ADX trend strength
        signals["trend_strength"] = np.where(df["adx"] > 25, "strong", "weak")

        # Combined signal
        signal_cols = [
            "rsi_signal",
            "macd_signal",
            "bb_signal",
            "stoch_signal",
            "trend_signal",
        ]
        signals["combined"] = signals[signal_cols].sum(axis=1)
        signals["combined_signal"] = np.sign(signals["combined"])

        return signals

    def summary(self) -> Dict[str, Any]:
        """
        Generate analysis summary.

        Returns:
            Dictionary with current indicator values and signals
        """
        if not self._indicators_calculated:
            self.calculate_all()

        df = self.indicators
        signals = self.get_signals()

        current = {
            "price": self.close.iloc[-1],
            "rsi": df["rsi"].iloc[-1],
            "macd": df["macd"].iloc[-1],
            "macd_signal": df["macd_signal"].iloc[-1],
            "macd_histogram": df["macd_hist"].iloc[-1],
            "bb_upper": df["bb_upper"].iloc[-1],
            "bb_lower": df["bb_lower"].iloc[-1],
            "bb_position": df["bb_pct"].iloc[-1],
            "stoch_k": df["stoch_k"].iloc[-1],
            "stoch_d": df["stoch_d"].iloc[-1],
            "adx": df["adx"].iloc[-1],
            "atr": df["atr"].iloc[-1],
            "atr_pct": df["atr_pct"].iloc[-1],
            "sma_20": df["sma_20"].iloc[-1],
            "sma_50": df["sma_50"].iloc[-1],
            "sma_200": df["sma_200"].iloc[-1],
        }

        current["trend"] = (
            "bullish" if current["sma_20"] > current["sma_50"] else "bearish"
        )
        current["volatility"] = "high" if current["atr_pct"] > 0.02 else "low"
        current["momentum"] = (
            "positive" if current["macd"] > current["macd_signal"] else "negative"
        )
        current["overbought"] = current["rsi"] > 70
        current["oversold"] = current["rsi"] < 30
        current["combined_signal"] = signals["combined_signal"].iloc[-1]

        return current
