"""Candlestick pattern detection."""

import pandas as pd


def _col(data: pd.DataFrame, name: str) -> pd.Series:
    """Fetch an OHLC column case-insensitively."""
    for candidate in (name, name.capitalize(), name.upper(), name.lower()):
        if candidate in data.columns:
            return data[candidate]
    raise KeyError(f"Column '{name}' not found in data")


class CandlestickPatterns:
    """Detect common single-candle patterns from an OHLC(V) frame."""

    def detect_doji(self, data: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
        """Doji: open and close nearly equal relative to the day's range."""
        open_, high, low, close = (
            _col(data, "open"),
            _col(data, "high"),
            _col(data, "low"),
            _col(data, "close"),
        )
        body = (close - open_).abs()
        rng = (high - low).replace(0, pd.NA)
        return (body / rng).fillna(1.0) <= threshold

    def detect_hammer(self, data: pd.DataFrame) -> pd.Series:
        """Hammer: small body near the top with a long lower shadow."""
        open_, high, low, close = (
            _col(data, "open"),
            _col(data, "high"),
            _col(data, "low"),
            _col(data, "close"),
        )
        body = (close - open_).abs()
        rng = (high - low).replace(0, pd.NA)
        lower_shadow = pd.concat([open_, close], axis=1).min(axis=1) - low
        upper_shadow = high - pd.concat([open_, close], axis=1).max(axis=1)
        is_hammer = (
            (lower_shadow >= 2 * body) & (upper_shadow <= body) & (body / rng <= 0.4)
        )
        return is_hammer.fillna(False)
