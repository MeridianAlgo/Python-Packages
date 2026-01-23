"""
Integration with the TA (Technical Analysis) library.

This module integrates the comprehensive TA library to provide even more
technical analysis indicators and functionality.
"""

import warnings
from typing import Dict

import pandas as pd

try:
    import ta

    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    warnings.warn("TA library not available. Install with: pip install ta")


class TAIntegration:
    """Integration class for TA library functionality."""

    def __init__(self):
        if not TA_AVAILABLE:
            raise ImportError("TA library not available. Install with: pip install ta")

    @staticmethod
    def add_all_ta_features(
        df: pd.DataFrame,
        high_col: str = "High",
        low_col: str = "Low",
        close_col: str = "Close",
        volume_col: str = "Volume",
        open_col: str = "Open",
    ) -> pd.DataFrame:
        """
        Add all TA library features to a DataFrame.

        Args:
            df: DataFrame with OHLCV data
            high_col: Name of high price column
            low_col: Name of low price column
            close_col: Name of close price column
            volume_col: Name of volume column
            open_col: Name of open price column

        Returns:
            DataFrame with all TA features added
        """
        if not TA_AVAILABLE:
            return df

        df_with_ta = df.copy()

        try:
            # Add all TA features
            df_with_ta = ta.add_all_ta_features(
                df_with_ta,
                open=open_col,
                high=high_col,
                low=low_col,
                close=close_col,
                volume=volume_col,
                fillna=True,
            )
        except Exception as e:
            warnings.warn(f"Error adding TA features: {e}")

        return df_with_ta

    @staticmethod
    def get_volume_indicators(
        df: pd.DataFrame,
        high_col: str = "High",
        low_col: str = "Low",
        close_col: str = "Close",
        volume_col: str = "Volume",
    ) -> Dict[str, pd.Series]:
        """Get all volume-based indicators from TA library."""
        if not TA_AVAILABLE:
            return {}

        indicators = {}

        try:
            # Accumulation/Distribution Index
            indicators["adi"] = ta.volume.AccDistIndexIndicator(
                high=df[high_col],
                low=df[low_col],
                close=df[close_col],
                volume=df[volume_col],
            ).acc_dist_index()

            # On-Balance Volume
            indicators["obv"] = ta.volume.OnBalanceVolumeIndicator(
                close=df[close_col], volume=df[volume_col]
            ).on_balance_volume()

            # Chaikin Money Flow
            indicators["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(
                high=df[high_col],
                low=df[low_col],
                close=df[close_col],
                volume=df[volume_col],
            ).chaikin_money_flow()

            # Force Index
            indicators["fi"] = ta.volume.ForceIndexIndicator(
                close=df[close_col], volume=df[volume_col]
            ).force_index()

            # Ease of Movement
            indicators["eom"] = ta.volume.EaseOfMovementIndicator(
                high=df[high_col], low=df[low_col], volume=df[volume_col]
            ).ease_of_movement()

            # Volume-Price Trend
            indicators["vpt"] = ta.volume.VolumePriceTrendIndicator(
                close=df[close_col], volume=df[volume_col]
            ).volume_price_trend()

            # Negative Volume Index
            indicators["nvi"] = ta.volume.NegativeVolumeIndexIndicator(
                close=df[close_col], volume=df[volume_col]
            ).negative_volume_index()

            # Volume Weighted Average Price
            indicators["vwap"] = ta.volume.VolumeSMAIndicator(
                close=df[close_col], volume=df[volume_col]
            ).volume_sma()

        except Exception as e:
            warnings.warn(f"Error calculating volume indicators: {e}")

        return indicators

    @staticmethod
    def get_volatility_indicators(
        df: pd.DataFrame,
        high_col: str = "High",
        low_col: str = "Low",
        close_col: str = "Close",
    ) -> Dict[str, pd.Series]:
        """Get all volatility indicators from TA library."""
        if not TA_AVAILABLE:
            return {}

        indicators = {}

        try:
            # Average True Range
            indicators["atr"] = ta.volatility.AverageTrueRange(
                high=df[high_col], low=df[low_col], close=df[close_col]
            ).average_true_range()

            # Bollinger Bands
            bb = ta.volatility.BollingerBands(
                close=df[close_col], window=20, window_dev=2
            )
            indicators["bb_high"] = bb.bollinger_hband()
            indicators["bb_low"] = bb.bollinger_lband()
            indicators["bb_mid"] = bb.bollinger_mavg()
            indicators["bb_width"] = bb.bollinger_wband()
            indicators["bb_pband"] = bb.bollinger_pband()

            # Keltner Channel
            kc = ta.volatility.KeltnerChannel(
                high=df[high_col], low=df[low_col], close=df[close_col]
            )
            indicators["kc_high"] = kc.keltner_channel_hband()
            indicators["kc_low"] = kc.keltner_channel_lband()
            indicators["kc_mid"] = kc.keltner_channel_mband()
            indicators["kc_width"] = kc.keltner_channel_wband()
            indicators["kc_pband"] = kc.keltner_channel_pband()

            # Donchian Channel
            dc = ta.volatility.DonchianChannel(
                high=df[high_col], low=df[low_col], close=df[close_col]
            )
            indicators["dc_high"] = dc.donchian_channel_hband()
            indicators["dc_low"] = dc.donchian_channel_lband()
            indicators["dc_mid"] = dc.donchian_channel_mband()
            indicators["dc_width"] = dc.donchian_channel_wband()
            indicators["dc_pband"] = dc.donchian_channel_pband()

            # Ulcer Index
            indicators["ui"] = ta.volatility.UlcerIndex(
                close=df[close_col]
            ).ulcer_index()

        except Exception as e:
            warnings.warn(f"Error calculating volatility indicators: {e}")

        return indicators

    @staticmethod
    def get_trend_indicators(
        df: pd.DataFrame,
        high_col: str = "High",
        low_col: str = "Low",
        close_col: str = "Close",
    ) -> Dict[str, pd.Series]:
        """Get all trend indicators from TA library."""
        if not TA_AVAILABLE:
            return {}

        indicators = {}

        try:
            # Simple Moving Average
            indicators["sma_10"] = ta.trend.SMAIndicator(
                close=df[close_col], window=10
            ).sma_indicator()
            indicators["sma_20"] = ta.trend.SMAIndicator(
                close=df[close_col], window=20
            ).sma_indicator()
            indicators["sma_50"] = ta.trend.SMAIndicator(
                close=df[close_col], window=50
            ).sma_indicator()

            # Exponential Moving Average
            indicators["ema_10"] = ta.trend.EMAIndicator(
                close=df[close_col], window=10
            ).ema_indicator()
            indicators["ema_20"] = ta.trend.EMAIndicator(
                close=df[close_col], window=20
            ).ema_indicator()

            # MACD
            macd = ta.trend.MACD(
                close=df[close_col], window_slow=26, window_fast=12, window_sign=9
            )
            indicators["macd"] = macd.macd()
            indicators["macd_signal"] = macd.macd_signal()
            indicators["macd_diff"] = macd.macd_diff()

            # Average Directional Movement Index
            adx = ta.trend.ADXIndicator(
                high=df[high_col], low=df[low_col], close=df[close_col]
            )
            indicators["adx"] = adx.adx()
            indicators["adx_pos"] = adx.adx_pos()
            indicators["adx_neg"] = adx.adx_neg()

            # Vortex Indicator
            vi = ta.trend.VortexIndicator(
                high=df[high_col], low=df[low_col], close=df[close_col]
            )
            indicators["vi_pos"] = vi.vortex_indicator_pos()
            indicators["vi_neg"] = vi.vortex_indicator_neg()
            indicators["vi_diff"] = vi.vortex_indicator_diff()

            # TRIX
            indicators["trix"] = ta.trend.TRIXIndicator(close=df[close_col]).trix()

            # Mass Index
            indicators["mi"] = ta.trend.MassIndex(
                high=df[high_col], low=df[low_col]
            ).mass_index()

            # Commodity Channel Index
            indicators["cci"] = ta.trend.CCIIndicator(
                high=df[high_col], low=df[low_col], close=df[close_col]
            ).cci()

            # Detrended Price Oscillator
            indicators["dpo"] = ta.trend.DPOIndicator(close=df[close_col]).dpo()

            # KST Oscillator
            indicators["kst"] = ta.trend.KSTIndicator(close=df[close_col]).kst()
            indicators["kst_sig"] = ta.trend.KSTIndicator(close=df[close_col]).kst_sig()

            # Ichimoku
            ichimoku = ta.trend.IchimokuIndicator(high=df[high_col], low=df[low_col])
            indicators["ichimoku_conv"] = ichimoku.ichimoku_conversion_line()
            indicators["ichimoku_base"] = ichimoku.ichimoku_base_line()
            indicators["ichimoku_a"] = ichimoku.ichimoku_a()
            indicators["ichimoku_b"] = ichimoku.ichimoku_b()

            # Parabolic SAR
            indicators["psar"] = ta.trend.PSARIndicator(
                high=df[high_col], low=df[low_col], close=df[close_col]
            ).psar()

            # Aroon
            aroon = ta.trend.AroonIndicator(high=df[high_col], low=df[low_col])
            indicators["aroon_up"] = aroon.aroon_up()
            indicators["aroon_down"] = aroon.aroon_down()
            indicators["aroon_ind"] = aroon.aroon_indicator()

        except Exception as e:
            warnings.warn(f"Error calculating trend indicators: {e}")

        return indicators

    @staticmethod
    def get_momentum_indicators(
        df: pd.DataFrame,
        high_col: str = "High",
        low_col: str = "Low",
        close_col: str = "Close",
        volume_col: str = "Volume",
    ) -> Dict[str, pd.Series]:
        """Get all momentum indicators from TA library."""
        if not TA_AVAILABLE:
            return {}

        indicators = {}

        try:
            # RSI
            indicators["rsi"] = ta.momentum.RSIIndicator(close=df[close_col]).rsi()

            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(
                high=df[high_col], low=df[low_col], close=df[close_col]
            )
            indicators["stoch_k"] = stoch.stoch()
            indicators["stoch_d"] = stoch.stoch_signal()

            # Williams %R
            indicators["williams_r"] = ta.momentum.WilliamsRIndicator(
                high=df[high_col], low=df[low_col], close=df[close_col]
            ).williams_r()

            # Awesome Oscillator
            indicators["ao"] = ta.momentum.AwesomeOscillatorIndicator(
                high=df[high_col], low=df[low_col]
            ).awesome_oscillator()

            # KAMA
            indicators["kama"] = ta.momentum.KAMAIndicator(close=df[close_col]).kama()

            # Rate of Change
            indicators["roc"] = ta.momentum.ROCIndicator(close=df[close_col]).roc()

            # Stochastic RSI
            stoch_rsi = ta.momentum.StochRSIIndicator(close=df[close_col])
            indicators["stoch_rsi"] = stoch_rsi.stochrsi()
            indicators["stoch_rsi_k"] = stoch_rsi.stochrsi_k()
            indicators["stoch_rsi_d"] = stoch_rsi.stochrsi_d()

            # Percentage Price Oscillator
            indicators["ppo"] = ta.momentum.PercentagePriceOscillator(
                close=df[close_col]
            ).ppo()
            indicators["ppo_signal"] = ta.momentum.PercentagePriceOscillator(
                close=df[close_col]
            ).ppo_signal()
            indicators["ppo_hist"] = ta.momentum.PercentagePriceOscillator(
                close=df[close_col]
            ).ppo_hist()

            # Percentage Volume Oscillator
            indicators["pvo"] = ta.momentum.PercentageVolumeOscillator(
                volume=df[volume_col]
            ).pvo()
            indicators["pvo_signal"] = ta.momentum.PercentageVolumeOscillator(
                volume=df[volume_col]
            ).pvo_signal()
            indicators["pvo_hist"] = ta.momentum.PercentageVolumeOscillator(
                volume=df[volume_col]
            ).pvo_hist()

        except Exception as e:
            warnings.warn(f"Error calculating momentum indicators: {e}")

        return indicators

    @staticmethod
    def get_others_indicators(
        df: pd.DataFrame,
        high_col: str = "High",
        low_col: str = "Low",
        close_col: str = "Close",
        volume_col: str = "Volume",
    ) -> Dict[str, pd.Series]:
        """Get other indicators from TA library."""
        if not TA_AVAILABLE:
            return {}

        indicators = {}

        try:
            # Daily Return
            indicators["daily_return"] = ta.others.DailyReturnIndicator(
                close=df[close_col]
            ).daily_return()

            # Daily Log Return
            indicators["daily_log_return"] = ta.others.DailyLogReturnIndicator(
                close=df[close_col]
            ).daily_log_return()

            # Cumulative Return
            indicators["cumulative_return"] = ta.others.CumulativeReturnIndicator(
                close=df[close_col]
            ).cumulative_return()

        except Exception as e:
            warnings.warn(f"Error calculating other indicators: {e}")

        return indicators

    @staticmethod
    def get_all_indicators(
        df: pd.DataFrame,
        high_col: str = "High",
        low_col: str = "Low",
        close_col: str = "Close",
        volume_col: str = "Volume",
        open_col: str = "Open",
    ) -> Dict[str, Dict[str, pd.Series]]:
        """Get all indicators organized by category."""
        if not TA_AVAILABLE:
            return {}

        all_indicators = {
            "volume": TAIntegration.get_volume_indicators(
                df, high_col, low_col, close_col, volume_col
            ),
            "volatility": TAIntegration.get_volatility_indicators(
                df, high_col, low_col, close_col
            ),
            "trend": TAIntegration.get_trend_indicators(
                df, high_col, low_col, close_col
            ),
            "momentum": TAIntegration.get_momentum_indicators(
                df, high_col, low_col, close_col, volume_col
            ),
            "others": TAIntegration.get_others_indicators(
                df, high_col, low_col, close_col, volume_col
            ),
        }

        return all_indicators


# Convenience functions for easy access
def add_all_ta_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Add all TA library features to a DataFrame."""
    return TAIntegration.add_all_ta_features(df, **kwargs)


def get_ta_volume_indicators(df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
    """Get volume indicators from TA library."""
    return TAIntegration.get_volume_indicators(df, **kwargs)


def get_ta_volatility_indicators(df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
    """Get volatility indicators from TA library."""
    return TAIntegration.get_volatility_indicators(df, **kwargs)


def get_ta_trend_indicators(df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
    """Get trend indicators from TA library."""
    return TAIntegration.get_trend_indicators(df, **kwargs)


def get_ta_momentum_indicators(df: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
    """Get momentum indicators from TA library."""
    return TAIntegration.get_momentum_indicators(df, **kwargs)


def get_all_ta_indicators(
    df: pd.DataFrame, **kwargs
) -> Dict[str, Dict[str, pd.Series]]:
    """Get all TA indicators organized by category."""
    return TAIntegration.get_all_indicators(df, **kwargs)
