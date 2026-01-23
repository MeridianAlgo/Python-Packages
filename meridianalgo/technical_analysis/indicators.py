"""
Technical indicators with TA-Lib integration and performance optimization.
"""

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import talib

try:
    import numba  # noqa: F401
    from numba import jit, njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Create dummy decorators if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


logger = logging.getLogger(__name__)


def performance_optimized(func):
    """Decorator to add performance optimization to indicator functions."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Add timing and optimization logic
        import time

        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        execution_time = end_time - start_time

        if execution_time > 1.0:  # Log slow operations
            logger.debug(f"{func.__name__} took {execution_time:.3f} seconds")

        return result

    return wrapper


class BaseIndicator(ABC):
    """Abstract base class for all technical indicators."""

    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None):
        self.name = name
        self.parameters = parameters or {}
        self.compiled_func = None

    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate the indicator values."""
        pass

    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data."""
        if data.empty:
            raise ValueError("Input data cannot be empty")

        required_columns = self.get_required_columns()
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def get_required_columns(self) -> List[str]:
        """Get required columns for this indicator."""
        return ["Close"]  # Default requirement

    def optimize_parameters(
        self,
        data: pd.DataFrame,
        parameter_ranges: Dict[str, Tuple[float, float]],
        optimization_metric: str = "sharpe",
    ) -> Dict[str, Any]:
        """Optimize indicator parameters."""
        # This is a placeholder for parameter optimization
        # In a full implementation, this would use techniques like grid search,
        # genetic algorithms, or Bayesian optimization
        return self.parameters


class CustomIndicator(BaseIndicator):
    """Custom indicator with JIT compilation support."""

    def __init__(
        self,
        name: str,
        calculation_func: Callable,
        parameters: Optional[Dict[str, Any]] = None,
        required_columns: Optional[List[str]] = None,
        enable_jit: bool = True,
    ):
        super().__init__(name, parameters)
        self.calculation_func = calculation_func
        self.required_columns_list = required_columns or ["Close"]
        self.enable_jit = enable_jit and NUMBA_AVAILABLE

        if self.enable_jit:
            try:
                self.compiled_func = jit(nopython=True)(calculation_func)
                logger.info(f"JIT compiled indicator: {name}")
            except Exception as e:
                logger.warning(f"JIT compilation failed for {name}: {e}")
                self.compiled_func = calculation_func
        else:
            self.compiled_func = calculation_func

    def calculate(self, data: pd.DataFrame, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate custom indicator."""
        self.validate_data(data)

        # Prepare input arrays
        inputs = {}
        for col in self.required_columns_list:
            if col in data.columns:
                inputs[col.lower()] = data[col].values

        # Merge parameters with kwargs
        params = {**self.parameters, **kwargs}

        # Calculate using compiled function
        if self.compiled_func:
            result = self.compiled_func(**inputs, **params)
        else:
            result = self.calculation_func(**inputs, **params)

        # Convert result to pandas Series/DataFrame
        if isinstance(result, np.ndarray):
            if result.ndim == 1:
                return pd.Series(result, index=data.index, name=self.name)
            else:
                columns = [f"{self.name}_{i}" for i in range(result.shape[1])]
                return pd.DataFrame(result, index=data.index, columns=columns)

        return result

    def get_required_columns(self) -> List[str]:
        return self.required_columns_list


class TALibIndicators:
    """Wrapper for TA-Lib indicators with pandas DataFrame interface."""

    def __init__(self):
        self.indicator_groups = self._get_indicator_groups()
        self.available_indicators = self._get_available_indicators()

    def _get_indicator_groups(self) -> Dict[str, List[str]]:
        """Get TA-Lib indicators grouped by category."""
        return {
            "overlap": [
                "BBANDS",
                "DEMA",
                "EMA",
                "HT_TRENDLINE",
                "KAMA",
                "MA",
                "MAMA",
                "MAVP",
                "MIDPOINT",
                "MIDPRICE",
                "SAR",
                "SAREXT",
                "SMA",
                "T3",
                "TEMA",
                "TRIMA",
                "WMA",
            ],
            "momentum": [
                "ADX",
                "ADXR",
                "APO",
                "AROON",
                "AROONOSC",
                "BOP",
                "CCI",
                "CMO",
                "DX",
                "MACD",
                "MACDEXT",
                "MACDFIX",
                "MFI",
                "MINUS_DI",
                "MINUS_DM",
                "MOM",
                "PLUS_DI",
                "PLUS_DM",
                "PPO",
                "ROC",
                "ROCP",
                "ROCR",
                "ROCR100",
                "RSI",
                "STOCH",
                "STOCHF",
                "STOCHRSI",
                "TRIX",
                "ULTOSC",
                "WILLR",
            ],
            "volume": ["AD", "ADOSC", "OBV"],
            "volatility": ["ATR", "NATR", "TRANGE"],
            "price_transform": ["AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"],
            "cycle": [
                "HT_DCPERIOD",
                "HT_DCPHASE",
                "HT_PHASOR",
                "HT_SINE",
                "HT_TRENDMODE",
            ],
            "math_transform": [
                "ACOS",
                "ASIN",
                "ATAN",
                "CEIL",
                "COS",
                "COSH",
                "EXP",
                "FLOOR",
                "LN",
                "LOG10",
                "SIN",
                "SINH",
                "SQRT",
                "TAN",
                "TANH",
            ],
            "math_operators": [
                "ADD",
                "DIV",
                "MAX",
                "MAXINDEX",
                "MIN",
                "MININDEX",
                "MINMAX",
                "MINMAXINDEX",
                "MULT",
                "SUB",
                "SUM",
            ],
            "statistic": [
                "BETA",
                "CORREL",
                "LINEARREG",
                "LINEARREG_ANGLE",
                "LINEARREG_INTERCEPT",
                "LINEARREG_SLOPE",
                "STDDEV",
                "TSF",
                "VAR",
            ],
        }

    def _get_available_indicators(self) -> List[str]:
        """Get list of all available TA-Lib indicators."""
        all_indicators = []
        for group_indicators in self.indicator_groups.values():
            all_indicators.extend(group_indicators)
        return sorted(all_indicators)

    @performance_optimized
    def calculate_indicator(
        self, data: pd.DataFrame, indicator_name: str, **kwargs
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate a single TA-Lib indicator.

        Args:
            data: OHLCV DataFrame
            indicator_name: Name of the TA-Lib indicator
            **kwargs: Indicator parameters

        Returns:
            Calculated indicator values
        """
        indicator_name = indicator_name.upper()

        if indicator_name not in self.available_indicators:
            raise ValueError(f"Unknown indicator: {indicator_name}")

        # Get the TA-Lib function
        talib_func = getattr(talib, indicator_name)

        # Prepare input data
        inputs = self._prepare_inputs(data, indicator_name)

        try:
            # Calculate indicator
            result = talib_func(**inputs, **kwargs)

            # Convert to pandas format
            return self._format_output(result, data.index, indicator_name)

        except Exception as e:
            raise ValueError(f"Error calculating {indicator_name}: {e}")

    def _prepare_inputs(
        self, data: pd.DataFrame, indicator_name: str
    ) -> Dict[str, np.ndarray]:
        """Prepare input arrays for TA-Lib functions."""
        inputs = {}

        # Standard OHLCV inputs
        if "Open" in data.columns:
            inputs["open"] = data["Open"].values
        if "High" in data.columns:
            inputs["high"] = data["High"].values
        if "Low" in data.columns:
            inputs["low"] = data["Low"].values
        if "Close" in data.columns:
            inputs["close"] = data["Close"].values
        if "Volume" in data.columns:
            inputs["volume"] = data["Volume"].values

        # Some indicators need specific input preparation
        if indicator_name in ["AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"]:
            # These only need OHLC
            pass
        elif indicator_name in ["AD", "ADOSC", "OBV"]:
            # Volume indicators
            if "Volume" not in inputs:
                raise ValueError(f"{indicator_name} requires Volume data")

        return inputs

    def _format_output(
        self,
        result: Union[np.ndarray, Tuple[np.ndarray, ...]],
        index: pd.Index,
        indicator_name: str,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Format TA-Lib output to pandas format."""
        if isinstance(result, tuple):
            # Multiple outputs (e.g., MACD returns 3 arrays)
            if indicator_name == "MACD":
                columns = ["MACD", "MACD_Signal", "MACD_Hist"]
            elif indicator_name == "STOCH":
                columns = ["SlowK", "SlowD"]
            elif indicator_name == "STOCHF":
                columns = ["FastK", "FastD"]
            elif indicator_name == "AROON":
                columns = ["AroonDown", "AroonUp"]
            elif indicator_name == "BBANDS":
                columns = ["BB_Upper", "BB_Middle", "BB_Lower"]
            elif indicator_name == "HT_PHASOR":
                columns = ["InPhase", "Quadrature"]
            elif indicator_name == "HT_SINE":
                columns = ["Sine", "LeadSine"]
            elif indicator_name == "MAMA":
                columns = ["MAMA", "FAMA"]
            elif indicator_name == "MINMAX":
                columns = ["Min", "Max"]
            elif indicator_name == "MINMAXINDEX":
                columns = ["MinIdx", "MaxIdx"]
            else:
                columns = [f"{indicator_name}_{i}" for i in range(len(result))]

            df_data = {}
            for i, (col, arr) in enumerate(zip(columns, result)):
                df_data[col] = arr

            return pd.DataFrame(df_data, index=index)
        else:
            # Single output
            return pd.Series(result, index=index, name=indicator_name)

    @performance_optimized
    def calculate_multiple(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, Dict[str, Any]],
        parallel: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate multiple indicators efficiently.

        Args:
            data: OHLCV DataFrame
            indicators: Dict of {indicator_name: parameters}
            parallel: Whether to use parallel processing

        Returns:
            DataFrame with all calculated indicators
        """
        if parallel and len(indicators) > 1:
            return self._calculate_parallel(data, indicators)
        else:
            return self._calculate_sequential(data, indicators)

    def _calculate_sequential(
        self, data: pd.DataFrame, indicators: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """Calculate indicators sequentially."""
        results = {}

        for indicator_name, params in indicators.items():
            try:
                result = self.calculate_indicator(data, indicator_name, **params)

                if isinstance(result, pd.Series):
                    results[indicator_name] = result
                elif isinstance(result, pd.DataFrame):
                    for col in result.columns:
                        results[col] = result[col]

            except Exception as e:
                logger.error(f"Failed to calculate {indicator_name}: {e}")
                continue

        return pd.DataFrame(results, index=data.index)

    def _calculate_parallel(
        self, data: pd.DataFrame, indicators: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """Calculate indicators in parallel."""
        results = {}

        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all indicator calculations
            future_to_indicator = {}
            for indicator_name, params in indicators.items():
                future = executor.submit(
                    self.calculate_indicator, data, indicator_name, **params
                )
                future_to_indicator[future] = indicator_name

            # Collect results
            for future in as_completed(future_to_indicator):
                indicator_name = future_to_indicator[future]
                try:
                    result = future.result()

                    if isinstance(result, pd.Series):
                        results[indicator_name] = result
                    elif isinstance(result, pd.DataFrame):
                        for col in result.columns:
                            results[col] = result[col]

                except Exception as e:
                    logger.error(f"Failed to calculate {indicator_name}: {e}")
                    continue

        return pd.DataFrame(results, index=data.index)

    def get_indicator_info(self, indicator_name: str) -> Dict[str, Any]:
        """Get information about a specific indicator."""
        indicator_name = indicator_name.upper()

        if indicator_name not in self.available_indicators:
            return {}

        # Find which group the indicator belongs to
        group = None
        for group_name, indicators in self.indicator_groups.items():
            if indicator_name in indicators:
                group = group_name
                break

        # Get function info from TA-Lib
        try:
            getattr(talib, indicator_name)
            func_info = talib.abstract.Function(indicator_name).info

            return {
                "name": indicator_name,
                "group": group,
                "display_name": func_info.get("display_name", indicator_name),
                "function_flags": func_info.get("function_flags", []),
                "input_names": func_info.get("input_names", {}),
                "output_names": func_info.get("output_names", {}),
                "parameters": func_info.get("parameters", {}),
            }
        except Exception:
            return {"name": indicator_name, "group": group}

    def list_indicators_by_group(self, group: str) -> List[str]:
        """List indicators in a specific group."""
        return self.indicator_groups.get(group, [])


class IndicatorManager:
    """Manager for technical indicators with caching and optimization."""

    def __init__(self):
        self.talib_indicators = TALibIndicators()
        self.custom_indicators: Dict[str, CustomIndicator] = {}
        self.cache: Dict[str, pd.DataFrame] = {}
        self.cache_enabled = True
        self.max_cache_size = 100

    def add_custom_indicator(self, indicator: CustomIndicator) -> None:
        """Add a custom indicator."""
        self.custom_indicators[indicator.name] = indicator
        logger.info(f"Added custom indicator: {indicator.name}")

    def remove_custom_indicator(self, name: str) -> None:
        """Remove a custom indicator."""
        if name in self.custom_indicators:
            del self.custom_indicators[name]
            logger.info(f"Removed custom indicator: {name}")

    def calculate(
        self, data: pd.DataFrame, indicator_name: str, **kwargs
    ) -> Union[pd.Series, pd.DataFrame]:
        """Calculate any indicator (TA-Lib or custom)."""
        # Check cache first
        cache_key = self._generate_cache_key(data, indicator_name, kwargs)
        if self.cache_enabled and cache_key in self.cache:
            logger.debug(f"Cache hit for {indicator_name}")
            return self.cache[cache_key]

        # Calculate indicator
        if indicator_name in self.custom_indicators:
            result = self.custom_indicators[indicator_name].calculate(data, **kwargs)
        else:
            result = self.talib_indicators.calculate_indicator(
                data, indicator_name, **kwargs
            )

        # Cache result
        if self.cache_enabled:
            self._cache_result(cache_key, result)

        return result

    def calculate_batch(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, Dict[str, Any]],
        parallel: bool = True,
    ) -> pd.DataFrame:
        """Calculate multiple indicators efficiently."""
        talib_indicators = {}
        custom_indicators = {}

        # Separate TA-Lib and custom indicators
        for name, params in indicators.items():
            if name in self.custom_indicators:
                custom_indicators[name] = params
            else:
                talib_indicators[name] = params

        results = {}

        # Calculate TA-Lib indicators
        if talib_indicators:
            talib_results = self.talib_indicators.calculate_multiple(
                data, talib_indicators, parallel
            )
            results.update(talib_results.to_dict("series"))

        # Calculate custom indicators
        for name, params in custom_indicators.items():
            try:
                result = self.custom_indicators[name].calculate(data, **params)
                if isinstance(result, pd.Series):
                    results[name] = result
                elif isinstance(result, pd.DataFrame):
                    results.update(result.to_dict("series"))
            except Exception as e:
                logger.error(f"Failed to calculate custom indicator {name}: {e}")

        return pd.DataFrame(results, index=data.index)

    def _generate_cache_key(
        self, data: pd.DataFrame, indicator_name: str, params: Dict[str, Any]
    ) -> str:
        """Generate cache key for indicator calculation."""
        # Simple cache key based on data hash and parameters
        data_hash = hash(tuple(data.index.tolist() + data.values.flatten().tolist()))
        params_hash = hash(tuple(sorted(params.items())))
        return f"{indicator_name}_{data_hash}_{params_hash}"

    def _cache_result(
        self, cache_key: str, result: Union[pd.Series, pd.DataFrame]
    ) -> None:
        """Cache calculation result."""
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[cache_key] = result

    def clear_cache(self) -> None:
        """Clear indicator cache."""
        self.cache.clear()
        logger.info("Indicator cache cleared")

    def get_available_indicators(self) -> Dict[str, List[str]]:
        """Get all available indicators."""
        return {
            "talib": self.talib_indicators.available_indicators,
            "custom": list(self.custom_indicators.keys()),
        }

    def enable_cache(self) -> None:
        """Enable result caching."""
        self.cache_enabled = True

    def disable_cache(self) -> None:
        """Disable result caching."""
        self.cache_enabled = False
        self.clear_cache()


# Pre-defined custom indicators with JIT compilation


@njit
def rsi_numba(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Numba-optimized RSI calculation."""
    n = len(close)
    rsi = np.full(n, np.nan)

    if n < period + 1:
        return rsi

    # Calculate price changes
    deltas = np.diff(close)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Calculate initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    # Calculate RSI for remaining periods
    for i in range(period + 1, n):
        gain = gains[i - 1]
        loss = losses[i - 1]

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


@njit
def sma_numba(close: np.ndarray, period: int) -> np.ndarray:
    """Numba-optimized Simple Moving Average."""
    n = len(close)
    sma = np.full(n, np.nan)

    if n < period:
        return sma

    # Calculate first SMA
    sma[period - 1] = np.mean(close[:period])

    # Calculate remaining SMAs
    for i in range(period, n):
        sma[i] = sma[i - 1] + (close[i] - close[i - period]) / period

    return sma


@njit
def ema_numba(close: np.ndarray, period: int) -> np.ndarray:
    """Numba-optimized Exponential Moving Average."""
    n = len(close)
    ema = np.full(n, np.nan)

    if n == 0:
        return ema

    alpha = 2.0 / (period + 1)
    ema[0] = close[0]

    for i in range(1, n):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i - 1]

    return ema


# Create pre-defined custom indicators
def create_optimized_indicators() -> Dict[str, CustomIndicator]:
    """Create optimized custom indicators."""
    indicators = {}

    if NUMBA_AVAILABLE:
        # RSI with Numba optimization
        indicators["RSI_FAST"] = CustomIndicator(
            name="RSI_FAST",
            calculation_func=lambda close, period=14: rsi_numba(close, period),
            parameters={"period": 14},
            required_columns=["Close"],
            enable_jit=False,  # Already compiled
        )

        # SMA with Numba optimization
        indicators["SMA_FAST"] = CustomIndicator(
            name="SMA_FAST",
            calculation_func=lambda close, period=20: sma_numba(close, period),
            parameters={"period": 20},
            required_columns=["Close"],
            enable_jit=False,  # Already compiled
        )

        # EMA with Numba optimization
        indicators["EMA_FAST"] = CustomIndicator(
            name="EMA_FAST",
            calculation_func=lambda close, period=20: ema_numba(close, period),
            parameters={"period": 20},
            required_columns=["Close"],
            enable_jit=False,  # Already compiled
        )

    return indicators
