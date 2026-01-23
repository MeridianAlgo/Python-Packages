"""
Advanced pattern recognition system for candlestick and chart patterns.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import talib
from scipy import signal
from scipy.stats import linregress

try:
    import numba  # noqa: F401
    from numba import jit, njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


logger = logging.getLogger(__name__)


@dataclass
class PatternResult:
    """Result of pattern recognition."""

    pattern_name: str
    start_index: int
    end_index: int
    confidence: float
    strength: float
    direction: str  # 'bullish', 'bearish', 'neutral'
    metadata: Dict[str, any] = None


class BasePattern(ABC):
    """Abstract base class for pattern recognition."""

    def __init__(self, name: str, min_periods: int = 5):
        self.name = name
        self.min_periods = min_periods

    @abstractmethod
    def detect(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect pattern in the data."""
        pass

    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data."""
        if len(data) < self.min_periods:
            raise ValueError(f"Insufficient data for {self.name} pattern detection")

        required_columns = ["Open", "High", "Low", "Close"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")


class CandlestickPatterns:
    """Candlestick pattern recognition using TA-Lib and custom implementations."""

    def __init__(self):
        self.talib_patterns = self._get_talib_patterns()
        self.custom_patterns = self._get_custom_patterns()

    def _get_talib_patterns(self) -> Dict[str, str]:
        """Get available TA-Lib candlestick patterns."""
        return {
            "DOJI": "CDL_DOJI",
            "HAMMER": "CDL_HAMMER",
            "HANGING_MAN": "CDL_HANGINGMAN",
            "INVERTED_HAMMER": "CDL_INVERTEDHAMMER",
            "SHOOTING_STAR": "CDL_SHOOTINGSTAR",
            "MARUBOZU": "CDL_MARUBOZU",
            "SPINNING_TOP": "CDL_SPINNINGTOP",
            "ENGULFING": "CDL_ENGULFING",
            "HARAMI": "CDL_HARAMI",
            "PIERCING": "CDL_PIERCING",
            "DARK_CLOUD": "CDL_DARKCLOUD",
            "MORNING_STAR": "CDL_MORNINGSTAR",
            "EVENING_STAR": "CDL_EVENINGSTAR",
            "THREE_WHITE_SOLDIERS": "CDL_3WHITESOLDIERS",
            "THREE_BLACK_CROWS": "CDL_3BLACKCROWS",
            "INSIDE": "CDL_3INSIDE",
            "OUTSIDE": "CDL_3OUTSIDE",
            "ABANDONED_BABY": "CDL_ABANDONEDBABY",
            "ADVANCE_BLOCK": "CDL_ADVANCEBLOCK",
            "BELT_HOLD": "CDL_BELTHOLD",
            "BREAKAWAY": "CDL_BREAKAWAY",
            "CLOSING_MARUBOZU": "CDL_CLOSINGMARUBOZU",
            "CONCEALING_BABY_SWALLOW": "CDL_CONCEALBABYSWALLOW",
            "COUNTERATTACK": "CDL_COUNTERATTACK",
            "DRAGONFLY_DOJI": "CDL_DRAGONFLYDOJI",
            "GAPSIDE_WHITE": "CDL_GAPSIDESIDEWHITE",
            "GRAVESTONE_DOJI": "CDL_GRAVESTONEDOJI",
            "HOMING_PIGEON": "CDL_HOMINGPIGEON",
            "IDENTICAL_THREE_CROWS": "CDL_IDENTICAL3CROWS",
            "IN_NECK": "CDL_INNECK",
            "KICKING": "CDL_KICKING",
            "LADDER_BOTTOM": "CDL_LADDERBOTTOM",
            "LONG_LEGGED_DOJI": "CDL_LONGLEGGEDDOJI",
            "MATCHING_LOW": "CDL_MATCHINGLOW",
            "ON_NECK": "CDL_ONNECK",
            "RICKSHAW_MAN": "CDL_RICKSHAWMAN",
            "RISING_FALLING_THREE": "CDL_RISEFALL3METHODS",
            "SEPARATING_LINES": "CDL_SEPARATINGLINES",
            "STALLED_PATTERN": "CDL_STALLEDPATTERN",
            "STICK_SANDWICH": "CDL_STICKSANDWICH",
            "TAKURI": "CDL_TAKURI",
            "TASUKI_GAP": "CDL_TASUKIGAP",
            "THRUSTING": "CDL_THRUSTING",
            "TRISTAR": "CDL_TRISTAR",
            "UNIQUE_THREE_RIVER": "CDL_UNIQUE3RIVER",
            "UPSIDE_GAP_TWO_CROWS": "CDL_UPSIDEGAP2CROWS",
            "XSIDE_GAP_THREE_METHODS": "CDL_XSIDEGAP3METHODS",
        }

    def _get_custom_patterns(self) -> Dict[str, callable]:
        """Get custom pattern detection functions."""
        return {
            "BULLISH_REVERSAL_COMBO": self._detect_bullish_reversal_combo,
            "BEARISH_REVERSAL_COMBO": self._detect_bearish_reversal_combo,
            "STRONG_TREND_CONTINUATION": self._detect_trend_continuation,
            "INDECISION_CLUSTER": self._detect_indecision_cluster,
        }

    def detect_pattern(
        self, data: pd.DataFrame, pattern_name: str
    ) -> List[PatternResult]:
        """Detect a specific candlestick pattern."""
        pattern_name = pattern_name.upper()

        if pattern_name in self.talib_patterns:
            return self._detect_talib_pattern(data, pattern_name)
        elif pattern_name in self.custom_patterns:
            return self.custom_patterns[pattern_name](data)
        else:
            raise ValueError(f"Unknown pattern: {pattern_name}")

    def detect_all_patterns(
        self, data: pd.DataFrame, min_confidence: float = 0.5
    ) -> List[PatternResult]:
        """Detect all available patterns."""
        all_results = []

        # Detect TA-Lib patterns
        for pattern_name in self.talib_patterns:
            try:
                results = self._detect_talib_pattern(data, pattern_name)
                all_results.extend(
                    [r for r in results if r.confidence >= min_confidence]
                )
            except Exception as e:
                logger.warning(f"Failed to detect {pattern_name}: {e}")

        # Detect custom patterns
        for pattern_name in self.custom_patterns:
            try:
                results = self.custom_patterns[pattern_name](data)
                all_results.extend(
                    [r for r in results if r.confidence >= min_confidence]
                )
            except Exception as e:
                logger.warning(f"Failed to detect {pattern_name}: {e}")

        return sorted(all_results, key=lambda x: x.confidence, reverse=True)

    def _detect_talib_pattern(
        self, data: pd.DataFrame, pattern_name: str
    ) -> List[PatternResult]:
        """Detect TA-Lib candlestick pattern."""
        talib_func_name = self.talib_patterns[pattern_name]
        talib_func = getattr(talib, talib_func_name)

        # Calculate pattern
        pattern_values = talib_func(
            data["Open"].values,
            data["High"].values,
            data["Low"].values,
            data["Close"].values,
        )

        results = []
        for i, value in enumerate(pattern_values):
            if value != 0:  # Pattern detected
                confidence = abs(value) / 100.0  # TA-Lib returns -100 to 100
                direction = "bullish" if value > 0 else "bearish"

                results.append(
                    PatternResult(
                        pattern_name=pattern_name,
                        start_index=i,
                        end_index=i,
                        confidence=confidence,
                        strength=confidence,
                        direction=direction,
                        metadata={"talib_value": value},
                    )
                )

        return results

    def _detect_bullish_reversal_combo(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect bullish reversal combination patterns."""
        results = []

        # Look for hammer followed by bullish engulfing
        hammer = talib.CDLHAMMER(data["Open"], data["High"], data["Low"], data["Close"])
        engulfing = talib.CDLENGULFING(
            data["Open"], data["High"], data["Low"], data["Close"]
        )

        for i in range(1, len(data)):
            if hammer[i - 1] > 0 and engulfing[i] > 0:
                confidence = 0.8  # High confidence for combo pattern
                results.append(
                    PatternResult(
                        pattern_name="BULLISH_REVERSAL_COMBO",
                        start_index=i - 1,
                        end_index=i,
                        confidence=confidence,
                        strength=confidence,
                        direction="bullish",
                        metadata={"components": ["HAMMER", "ENGULFING"]},
                    )
                )

        return results

    def _detect_bearish_reversal_combo(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect bearish reversal combination patterns."""
        results = []

        # Look for shooting star followed by bearish engulfing
        shooting_star = talib.CDLSHOOTINGSTAR(
            data["Open"], data["High"], data["Low"], data["Close"]
        )
        engulfing = talib.CDLENGULFING(
            data["Open"], data["High"], data["Low"], data["Close"]
        )

        for i in range(1, len(data)):
            if shooting_star[i - 1] > 0 and engulfing[i] < 0:
                confidence = 0.8
                results.append(
                    PatternResult(
                        pattern_name="BEARISH_REVERSAL_COMBO",
                        start_index=i - 1,
                        end_index=i,
                        confidence=confidence,
                        strength=confidence,
                        direction="bearish",
                        metadata={"components": ["SHOOTING_STAR", "BEARISH_ENGULFING"]},
                    )
                )

        return results

    def _detect_trend_continuation(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect strong trend continuation patterns."""
        results = []

        # Calculate trend strength using moving averages
        sma_short = talib.SMA(data["Close"], timeperiod=5)
        sma_long = talib.SMA(data["Close"], timeperiod=20)

        # Look for three consecutive candles in trend direction
        for i in range(2, len(data)):
            if i < 20:  # Need enough data for moving averages
                continue

            # Check for uptrend continuation
            if (
                sma_short[i] > sma_long[i]
                and data["Close"].iloc[i]
                > data["Close"].iloc[i - 1]
                > data["Close"].iloc[i - 2]
                and data["Low"].iloc[i]
                > data["Low"].iloc[i - 1]
                > data["Low"].iloc[i - 2]
            ):
                confidence = 0.7
                results.append(
                    PatternResult(
                        pattern_name="STRONG_TREND_CONTINUATION",
                        start_index=i - 2,
                        end_index=i,
                        confidence=confidence,
                        strength=confidence,
                        direction="bullish",
                        metadata={"trend_type": "uptrend"},
                    )
                )

            # Check for downtrend continuation
            elif (
                sma_short[i] < sma_long[i]
                and data["Close"].iloc[i]
                < data["Close"].iloc[i - 1]
                < data["Close"].iloc[i - 2]
                and data["High"].iloc[i]
                < data["High"].iloc[i - 1]
                < data["High"].iloc[i - 2]
            ):
                confidence = 0.7
                results.append(
                    PatternResult(
                        pattern_name="STRONG_TREND_CONTINUATION",
                        start_index=i - 2,
                        end_index=i,
                        confidence=confidence,
                        strength=confidence,
                        direction="bearish",
                        metadata={"trend_type": "downtrend"},
                    )
                )

        return results

    def _detect_indecision_cluster(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect clusters of indecision patterns."""
        results = []

        # Detect doji patterns
        doji = talib.CDLDOJI(data["Open"], data["High"], data["Low"], data["Close"])
        spinning_top = talib.CDLSPINNINGTOP(
            data["Open"], data["High"], data["Low"], data["Close"]
        )

        # Look for clusters of indecision patterns
        window_size = 5
        for i in range(window_size, len(data)):
            window_doji = doji[i - window_size : i]
            window_spinning = spinning_top[i - window_size : i]

            indecision_count = np.sum(np.abs(window_doji) > 0) + np.sum(
                np.abs(window_spinning) > 0
            )

            if indecision_count >= 3:  # At least 3 indecision patterns in window
                confidence = min(0.9, indecision_count / window_size)
                results.append(
                    PatternResult(
                        pattern_name="INDECISION_CLUSTER",
                        start_index=i - window_size,
                        end_index=i - 1,
                        confidence=confidence,
                        strength=confidence,
                        direction="neutral",
                        metadata={
                            "indecision_count": indecision_count,
                            "window_size": window_size,
                        },
                    )
                )

        return results


class ChartPatterns:
    """Chart pattern recognition for technical analysis."""

    def __init__(self):
        self.patterns = {
            "TRIANGLE": self._detect_triangle,
            "HEAD_AND_SHOULDERS": self._detect_head_and_shoulders,
            "DOUBLE_TOP": self._detect_double_top,
            "DOUBLE_BOTTOM": self._detect_double_bottom,
            "FLAG": self._detect_flag,
            "PENNANT": self._detect_pennant,
            "WEDGE": self._detect_wedge,
            "CHANNEL": self._detect_channel,
            "CUP_AND_HANDLE": self._detect_cup_and_handle,
            "RECTANGLE": self._detect_rectangle,
        }

    def detect_pattern(
        self, data: pd.DataFrame, pattern_name: str, min_periods: int = 20
    ) -> List[PatternResult]:
        """Detect a specific chart pattern."""
        pattern_name = pattern_name.upper()

        if pattern_name not in self.patterns:
            raise ValueError(f"Unknown chart pattern: {pattern_name}")

        if len(data) < min_periods:
            return []

        return self.patterns[pattern_name](data)

    def detect_all_patterns(
        self, data: pd.DataFrame, min_periods: int = 20, min_confidence: float = 0.6
    ) -> List[PatternResult]:
        """Detect all chart patterns."""
        all_results = []

        for pattern_name in self.patterns:
            try:
                results = self.detect_pattern(data, pattern_name, min_periods)
                all_results.extend(
                    [r for r in results if r.confidence >= min_confidence]
                )
            except Exception as e:
                logger.warning(f"Failed to detect {pattern_name}: {e}")

        return sorted(all_results, key=lambda x: x.confidence, reverse=True)

    def _detect_triangle(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect triangle patterns (ascending, descending, symmetrical)."""
        results = []
        window_size = 20

        for i in range(window_size, len(data)):
            window_data = data.iloc[i - window_size : i]

            # Find peaks and troughs
            peaks = self._find_peaks(window_data["High"].values)
            troughs = self._find_troughs(window_data["Low"].values)

            if len(peaks) >= 2 and len(troughs) >= 2:
                # Calculate trend lines
                peak_slope = self._calculate_slope(peaks)
                trough_slope = self._calculate_slope(troughs)

                # Classify triangle type
                if abs(peak_slope) < 0.001 and trough_slope > 0.001:
                    # Ascending triangle
                    triangle_type = "ascending"
                    direction = "bullish"
                    confidence = 0.7
                elif peak_slope < -0.001 and abs(trough_slope) < 0.001:
                    # Descending triangle
                    triangle_type = "descending"
                    direction = "bearish"
                    confidence = 0.7
                elif peak_slope < -0.001 and trough_slope > 0.001:
                    # Symmetrical triangle
                    triangle_type = "symmetrical"
                    direction = "neutral"
                    confidence = 0.6
                else:
                    continue

                results.append(
                    PatternResult(
                        pattern_name="TRIANGLE",
                        start_index=i - window_size,
                        end_index=i - 1,
                        confidence=confidence,
                        strength=confidence,
                        direction=direction,
                        metadata={
                            "triangle_type": triangle_type,
                            "peak_slope": peak_slope,
                            "trough_slope": trough_slope,
                        },
                    )
                )

        return results

    def _detect_head_and_shoulders(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect head and shoulders pattern."""
        results = []
        window_size = 30

        for i in range(window_size, len(data)):
            window_data = data.iloc[i - window_size : i]
            peaks = self._find_peaks(window_data["High"].values, min_distance=5)

            if len(peaks) >= 3:
                # Check for head and shoulders formation
                peak_heights = [window_data["High"].iloc[p] for p in peaks[-3:]]

                # Head should be higher than both shoulders
                if (
                    peak_heights[1] > peak_heights[0]
                    and peak_heights[1] > peak_heights[2]
                    and abs(peak_heights[0] - peak_heights[2]) / peak_heights[1] < 0.05
                ):
                    confidence = 0.8
                    results.append(
                        PatternResult(
                            pattern_name="HEAD_AND_SHOULDERS",
                            start_index=i - window_size,
                            end_index=i - 1,
                            confidence=confidence,
                            strength=confidence,
                            direction="bearish",
                            metadata={
                                "left_shoulder": peak_heights[0],
                                "head": peak_heights[1],
                                "right_shoulder": peak_heights[2],
                            },
                        )
                    )

        return results

    def _detect_double_top(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect double top pattern."""
        results = []
        window_size = 25

        for i in range(window_size, len(data)):
            window_data = data.iloc[i - window_size : i]
            peaks = self._find_peaks(window_data["High"].values, min_distance=8)

            if len(peaks) >= 2:
                # Check last two peaks for double top
                last_two_peaks = peaks[-2:]
                peak_heights = [window_data["High"].iloc[p] for p in last_two_peaks]

                # Peaks should be approximately equal
                if abs(peak_heights[0] - peak_heights[1]) / max(peak_heights) < 0.03:
                    # Check for valley between peaks
                    valley_start = last_two_peaks[0]
                    valley_end = last_two_peaks[1]
                    valley_low = window_data["Low"].iloc[valley_start:valley_end].min()

                    # Valley should be significantly lower than peaks
                    if (max(peak_heights) - valley_low) / max(peak_heights) > 0.05:
                        confidence = 0.75
                        results.append(
                            PatternResult(
                                pattern_name="DOUBLE_TOP",
                                start_index=i - window_size + last_two_peaks[0],
                                end_index=i - 1,
                                confidence=confidence,
                                strength=confidence,
                                direction="bearish",
                                metadata={
                                    "first_peak": peak_heights[0],
                                    "second_peak": peak_heights[1],
                                    "valley_low": valley_low,
                                },
                            )
                        )

        return results

    def _detect_double_bottom(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect double bottom pattern."""
        results = []
        window_size = 25

        for i in range(window_size, len(data)):
            window_data = data.iloc[i - window_size : i]
            troughs = self._find_troughs(window_data["Low"].values, min_distance=8)

            if len(troughs) >= 2:
                # Check last two troughs for double bottom
                last_two_troughs = troughs[-2:]
                trough_lows = [window_data["Low"].iloc[t] for t in last_two_troughs]

                # Troughs should be approximately equal
                if abs(trough_lows[0] - trough_lows[1]) / min(trough_lows) < 0.03:
                    # Check for peak between troughs
                    peak_start = last_two_troughs[0]
                    peak_end = last_two_troughs[1]
                    peak_high = window_data["High"].iloc[peak_start:peak_end].max()

                    # Peak should be significantly higher than troughs
                    if (peak_high - min(trough_lows)) / peak_high > 0.05:
                        confidence = 0.75
                        results.append(
                            PatternResult(
                                pattern_name="DOUBLE_BOTTOM",
                                start_index=i - window_size + last_two_troughs[0],
                                end_index=i - 1,
                                confidence=confidence,
                                strength=confidence,
                                direction="bullish",
                                metadata={
                                    "first_trough": trough_lows[0],
                                    "second_trough": trough_lows[1],
                                    "peak_high": peak_high,
                                },
                            )
                        )

        return results

    def _detect_flag(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect flag pattern."""
        results = []
        window_size = 15

        for i in range(window_size, len(data)):
            window_data = data.iloc[i - window_size : i]

            # Flag pattern: strong move followed by consolidation
            # Check for strong initial move
            initial_move = abs(
                window_data["Close"].iloc[-1] - window_data["Close"].iloc[0]
            )
            avg_range = window_data["High"].sub(window_data["Low"]).mean()

            if initial_move > 3 * avg_range:
                # Check for consolidation (flag pole)
                consolidation_data = window_data.iloc[-10:]  # Last 10 periods
                consolidation_range = (
                    consolidation_data["High"].max() - consolidation_data["Low"].min()
                )

                if consolidation_range < 0.5 * initial_move:
                    direction = (
                        "bullish"
                        if window_data["Close"].iloc[-1] > window_data["Close"].iloc[0]
                        else "bearish"
                    )
                    confidence = 0.65

                    results.append(
                        PatternResult(
                            pattern_name="FLAG",
                            start_index=i - window_size,
                            end_index=i - 1,
                            confidence=confidence,
                            strength=confidence,
                            direction=direction,
                            metadata={
                                "initial_move": initial_move,
                                "consolidation_range": consolidation_range,
                            },
                        )
                    )

        return results

    def _detect_pennant(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect pennant pattern."""
        # Similar to flag but with converging trend lines
        return self._detect_flag(data)  # Simplified implementation

    def _detect_wedge(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect wedge pattern."""
        results = []
        window_size = 20

        for i in range(window_size, len(data)):
            window_data = data.iloc[i - window_size : i]

            # Calculate trend lines for highs and lows
            x = np.arange(len(window_data))
            high_slope, _, _, _, _ = linregress(x, window_data["High"].values)
            low_slope, _, _, _, _ = linregress(x, window_data["Low"].values)

            # Wedge: both trend lines slope in same direction and converge
            if (high_slope > 0 and low_slope > 0 and high_slope < low_slope) or (
                high_slope < 0 and low_slope < 0 and high_slope > low_slope
            ):
                wedge_type = "rising" if high_slope > 0 else "falling"
                direction = "bearish" if wedge_type == "rising" else "bullish"
                confidence = 0.6

                results.append(
                    PatternResult(
                        pattern_name="WEDGE",
                        start_index=i - window_size,
                        end_index=i - 1,
                        confidence=confidence,
                        strength=confidence,
                        direction=direction,
                        metadata={
                            "wedge_type": wedge_type,
                            "high_slope": high_slope,
                            "low_slope": low_slope,
                        },
                    )
                )

        return results

    def _detect_channel(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect channel pattern."""
        results = []
        window_size = 25

        for i in range(window_size, len(data)):
            window_data = data.iloc[i - window_size : i]

            # Calculate parallel trend lines
            x = np.arange(len(window_data))
            high_slope, high_intercept, high_r, _, _ = linregress(
                x, window_data["High"].values
            )
            low_slope, low_intercept, low_r, _, _ = linregress(
                x, window_data["Low"].values
            )

            # Channel: parallel trend lines with good fit
            if (
                abs(high_slope - low_slope) < 0.001
                and abs(high_r) > 0.7
                and abs(low_r) > 0.7
            ):
                channel_type = (
                    "ascending"
                    if high_slope > 0.001
                    else "descending"
                    if high_slope < -0.001
                    else "horizontal"
                )
                direction = (
                    "bullish"
                    if channel_type == "ascending"
                    else "bearish"
                    if channel_type == "descending"
                    else "neutral"
                )
                confidence = min(abs(high_r), abs(low_r))

                results.append(
                    PatternResult(
                        pattern_name="CHANNEL",
                        start_index=i - window_size,
                        end_index=i - 1,
                        confidence=confidence,
                        strength=confidence,
                        direction=direction,
                        metadata={
                            "channel_type": channel_type,
                            "high_slope": high_slope,
                            "low_slope": low_slope,
                            "high_r_squared": high_r**2,
                            "low_r_squared": low_r**2,
                        },
                    )
                )

        return results

    def _detect_cup_and_handle(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect cup and handle pattern."""
        results = []
        window_size = 40  # Longer window for cup and handle

        for i in range(window_size, len(data)):
            window_data = data.iloc[i - window_size : i]

            # Simplified cup and handle detection
            # Look for U-shaped bottom followed by small consolidation
            mid_point = len(window_data) // 2
            left_side = window_data.iloc[:mid_point]
            right_side = window_data.iloc[mid_point:]

            # Cup: declining then rising
            left_trend = left_side["Close"].iloc[-1] < left_side["Close"].iloc[0]
            right_trend = right_side["Close"].iloc[-1] > right_side["Close"].iloc[0]

            if left_trend and right_trend:
                # Check for handle (small consolidation at end)
                handle_data = window_data.iloc[-10:]
                handle_range = handle_data["High"].max() - handle_data["Low"].min()
                cup_range = window_data["High"].max() - window_data["Low"].min()

                if handle_range < 0.3 * cup_range:
                    confidence = 0.7
                    results.append(
                        PatternResult(
                            pattern_name="CUP_AND_HANDLE",
                            start_index=i - window_size,
                            end_index=i - 1,
                            confidence=confidence,
                            strength=confidence,
                            direction="bullish",
                            metadata={
                                "cup_range": cup_range,
                                "handle_range": handle_range,
                            },
                        )
                    )

        return results

    def _detect_rectangle(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect rectangle pattern."""
        results = []
        window_size = 20

        for i in range(window_size, len(data)):
            window_data = data.iloc[i - window_size : i]

            # Rectangle: horizontal support and resistance
            resistance = window_data["High"].quantile(0.95)
            support = window_data["Low"].quantile(0.05)

            # Check if price stays within rectangle
            breakouts = (
                (window_data["High"] > resistance * 1.02)
                | (window_data["Low"] < support * 0.98)
            ).sum()

            if breakouts <= 2:  # Allow few breakouts
                range_size = (resistance - support) / support
                if 0.03 < range_size < 0.15:  # Reasonable range
                    confidence = 0.6
                    results.append(
                        PatternResult(
                            pattern_name="RECTANGLE",
                            start_index=i - window_size,
                            end_index=i - 1,
                            confidence=confidence,
                            strength=confidence,
                            direction="neutral",
                            metadata={
                                "resistance": resistance,
                                "support": support,
                                "range_size": range_size,
                            },
                        )
                    )

        return results

    def _find_peaks(self, data: np.ndarray, min_distance: int = 3) -> List[int]:
        """Find peaks in data."""
        peaks, _ = signal.find_peaks(data, distance=min_distance)
        return peaks.tolist()

    def _find_troughs(self, data: np.ndarray, min_distance: int = 3) -> List[int]:
        """Find troughs in data."""
        troughs, _ = signal.find_peaks(-data, distance=min_distance)
        return troughs.tolist()

    def _calculate_slope(self, points: List[Tuple[int, float]]) -> float:
        """Calculate slope of trend line through points."""
        if len(points) < 2:
            return 0.0

        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]

        slope, _, _, _, _ = linregress(x_vals, y_vals)
        return slope


class PatternRecognizer:
    """Main pattern recognition system combining candlestick and chart patterns."""

    def __init__(self):
        self.candlestick_patterns = CandlestickPatterns()
        self.chart_patterns = ChartPatterns()

    def analyze_patterns(
        self,
        data: pd.DataFrame,
        pattern_types: List[str] = None,
        min_confidence: float = 0.5,
    ) -> Dict[str, List[PatternResult]]:
        """
        Comprehensive pattern analysis.

        Args:
            data: OHLCV DataFrame
            pattern_types: Types to analyze ('candlestick', 'chart', or both)
            min_confidence: Minimum confidence threshold

        Returns:
            Dictionary of pattern results by type
        """
        if pattern_types is None:
            pattern_types = ["candlestick", "chart"]

        results = {}

        if "candlestick" in pattern_types:
            candlestick_results = self.candlestick_patterns.detect_all_patterns(
                data, min_confidence
            )
            results["candlestick"] = candlestick_results

        if "chart" in pattern_types:
            chart_results = self.chart_patterns.detect_all_patterns(
                data, min_confidence=min_confidence
            )
            results["chart"] = chart_results

        return results

    def get_pattern_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of all detected patterns."""
        all_patterns = self.analyze_patterns(data)

        summary = {
            "total_patterns": 0,
            "bullish_patterns": 0,
            "bearish_patterns": 0,
            "neutral_patterns": 0,
            "avg_confidence": 0.0,
            "pattern_types": {},
            "strongest_patterns": [],
        }

        all_results = []
        for pattern_type, patterns in all_patterns.items():
            summary["pattern_types"][pattern_type] = len(patterns)
            all_results.extend(patterns)

        if all_results:
            summary["total_patterns"] = len(all_results)
            summary["bullish_patterns"] = len(
                [p for p in all_results if p.direction == "bullish"]
            )
            summary["bearish_patterns"] = len(
                [p for p in all_results if p.direction == "bearish"]
            )
            summary["neutral_patterns"] = len(
                [p for p in all_results if p.direction == "neutral"]
            )
            summary["avg_confidence"] = np.mean([p.confidence for p in all_results])

            # Get top 5 strongest patterns
            sorted_patterns = sorted(
                all_results, key=lambda x: x.strength, reverse=True
            )
            summary["strongest_patterns"] = sorted_patterns[:5]

        return summary
