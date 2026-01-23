"""
Market Microstructure Analysis

Advanced algorithms for analyzing market microstructure, order flow,
and high-frequency market dynamics.
"""

from typing import Dict

import numpy as np
import pandas as pd


class OrderFlowImbalance:
    """
    Calculate order flow imbalance metrics for high-frequency trading.

    Order flow imbalance is a key indicator in market microstructure that
    measures the imbalance between buy and sell orders.
    """

    @staticmethod
    def calculate_ofi(
        bid_volume: np.ndarray,
        ask_volume: np.ndarray,
        bid_price: np.ndarray,
        ask_price: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate Order Flow Imbalance (OFI).

        Parameters:
        -----------
        bid_volume : np.ndarray
            Bid side volume at each time step
        ask_volume : np.ndarray
            Ask side volume at each time step
        bid_price : np.ndarray
            Bid price at each time step
        ask_price : np.ndarray
            Ask price at each time step

        Returns:
        --------
        np.ndarray
            Order flow imbalance values
        """
        # Change in bid volume
        delta_bid_volume = np.diff(bid_volume, prepend=bid_volume[0])
        delta_ask_volume = np.diff(ask_volume, prepend=ask_volume[0])

        # Price changes
        np.diff(bid_price, prepend=bid_price[0])
        np.diff(ask_price, prepend=ask_price[0])

        # OFI calculation
        ofi = (
            (bid_price >= np.roll(bid_price, 1)) * delta_bid_volume
            - (bid_price <= np.roll(bid_price, 1)) * np.roll(bid_volume, 1)
            - (ask_price <= np.roll(ask_price, 1)) * delta_ask_volume
            + (ask_price >= np.roll(ask_price, 1)) * np.roll(ask_volume, 1)
        )

        return ofi

    @staticmethod
    def volume_imbalance_ratio(
        bid_volume: np.ndarray, ask_volume: np.ndarray
    ) -> np.ndarray:
        """
        Calculate simple volume imbalance ratio.

        VIR = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume)
        """
        total_volume = bid_volume + ask_volume
        # Avoid division by zero
        total_volume = np.where(total_volume == 0, 1e-10, total_volume)

        vir = (bid_volume - ask_volume) / total_volume
        return vir

    @staticmethod
    def weighted_pressure(
        bid_volume: np.ndarray,
        ask_volume: np.ndarray,
        bid_depth: int = 5,
        ask_depth: int = 5,
    ) -> float:
        """
        Calculate weighted order book pressure using depth.

        Parameters:
        -----------
        bid_volume : np.ndarray
            Volumes at different bid levels
        ask_volume : np.ndarray
            Volumes at different ask levels
        bid_depth : int
            Number of bid levels to consider
        ask_depth : int
            Number of ask levels to consider

        Returns:
        --------
        float
            Weighted pressure metric
        """
        # Exponential weights (closer levels have more weight)
        bid_weights = np.exp(-np.arange(min(bid_depth, len(bid_volume))))
        ask_weights = np.exp(-np.arange(min(ask_depth, len(ask_volume))))

        weighted_bid = np.sum(
            bid_volume[:bid_depth] * bid_weights[: len(bid_volume[:bid_depth])]
        )
        weighted_ask = np.sum(
            ask_volume[:ask_depth] * ask_weights[: len(ask_volume[:ask_depth])]
        )

        total_weighted = weighted_bid + weighted_ask
        if total_weighted == 0:
            return 0.0

        return (weighted_bid - weighted_ask) / total_weighted


class VolumeWeightedSpread:
    """
    Calculate volume-weighted bid-ask spread metrics.
    """

    @staticmethod
    def calculate_vw_spread(
        bid_price: np.ndarray,
        ask_price: np.ndarray,
        bid_volume: np.ndarray,
        ask_volume: np.ndarray,
    ) -> float:
        """
        Calculate volume-weighted spread.

        Parameters:
        -----------
        bid_price : np.ndarray
            Bid prices
        ask_price : np.ndarray
            Ask prices
        bid_volume : np.ndarray
            Bid volumes
        ask_volume : np.ndarray
            Ask volumes

        Returns:
        --------
        float
            Volume-weighted spread
        """
        total_volume = bid_volume + ask_volume
        total_volume = np.where(total_volume == 0, 1e-10, total_volume)

        weights = (bid_volume + ask_volume) / np.sum(total_volume)
        spread = ask_price - bid_price

        vw_spread = np.sum(spread * weights)
        return vw_spread

    @staticmethod
    def effective_spread(trade_price: float, mid_price: float, side: str) -> float:
        """
        Calculate effective spread for a trade.

        Effective Spread = 2 * |Trade Price - Mid Price| * Sign
        """
        sign = 1 if side.lower() == "buy" else -1
        return 2 * abs(trade_price - mid_price) * sign

    @staticmethod
    def realized_spread(
        trade_price: float, future_mid_price: float, side: str
    ) -> float:
        """
        Calculate realized spread (measures temporary price impact).
        """
        sign = 1 if side.lower() == "buy" else -1
        return 2 * (trade_price - future_mid_price) * sign


class RealizedVolatility:
    """
    Calculate realized volatility using high-frequency data.
    """

    @staticmethod
    def rv_5min(prices: pd.Series, freq: str = "5min") -> float:
        """
        Calculate 5-minute realized volatility.

        Parameters:
        -----------
        prices : pd.Series
            Price series with datetime index
        freq : str
            Sampling frequency

        Returns:
        --------
        float
            Realized volatility (annualized)
        """
        returns = prices.resample(freq).last().pct_change().dropna()
        rv = np.sqrt(np.sum(returns**2) * (252 * 78))  # 78 = 5-min intervals per day
        return rv

    @staticmethod
    def bipower_variation(returns: np.ndarray) -> float:
        """
        Calculate bipower variation (robust to jumps).

        BPV = (/2) * |r_t| * |r_{t-1}|
        """
        abs_returns = np.abs(returns)
        bpv = (np.pi / 2) * np.sum(abs_returns[1:] * abs_returns[:-1])
        return bpv

    @staticmethod
    def realized_kernel(returns: np.ndarray, bandwidth: int = 5) -> float:
        """
        Calculate realized kernel (noise-robust estimator).

        Parameters:
        -----------
        returns : np.ndarray
            High-frequency returns
        bandwidth : int
            Kernel bandwidth parameter

        Returns:
        --------
        float
            Realized kernel estimate
        """
        len(returns)
        gamma = np.zeros(bandwidth + 1)

        # Calculate autocovariances
        for h in range(bandwidth + 1):
            if h == 0:
                gamma[h] = np.sum(returns**2)
            else:
                gamma[h] = np.sum(returns[h:] * returns[:-h])

        # Parzen kernel weights
        weights = np.zeros(bandwidth + 1)
        for h in range(bandwidth + 1):
            x = h / (bandwidth + 1)
            if x <= 0.5:
                weights[h] = 1 - 6 * x**2 + 6 * x**3
            else:
                weights[h] = 2 * (1 - x) ** 3

        rk = gamma[0] + 2 * np.sum(weights[1:] * gamma[1:])
        return np.sqrt(rk * 252)  # Annualized


class MarketImpactModel:
    """
    Models for estimating market impact of trades.
    """

    @staticmethod
    def almgren_chriss_temporary_impact(
        volume: float, daily_volume: float, volatility: float, gamma: float = 0.1
    ) -> float:
        """
        Almgren-Chriss temporary market impact model.

        Parameters:
        -----------
        volume : float
            Trade volume
        daily_volume : float
            Average daily volume
        volatility : float
            Daily volatility
        gamma : float
            Impact parameter

        Returns:
        --------
        float
            Temporary price impact
        """
        participation_rate = volume / daily_volume
        impact = gamma * volatility * np.sign(volume) * (participation_rate**0.5)
        return impact

    @staticmethod
    def permanent_impact(
        volume: float, daily_volume: float, volatility: float, eta: float = 0.05
    ) -> float:
        """
        Permanent market impact (price doesn't revert).

        Parameters:
        -----------
        volume : float
            Trade volume
        daily_volume : float
            Average daily volume
        volatility : float
            Daily volatility
        eta : float
            Permanent impact parameter

        Returns:
        --------
        float
            Permanent price impact
        """
        participation_rate = volume / daily_volume
        impact = eta * volatility * np.sign(volume) * participation_rate
        return impact

    @staticmethod
    def square_root_law(
        order_size: float, daily_volume: float, sigma: float, alpha: float = 0.314
    ) -> float:
        """
        Square-root law for market impact (empirical model).

        Impact =  *  * (Order Size / Daily Volume)^(1/2)

        Parameters:
        -----------
        order_size : float
            Size of the order
        daily_volume : float
            Average daily volume
        sigma : float
            Volatility
        alpha : float
            Calibrated parameter (typically 0.1 to 0.5)

        Returns:
        --------
        float
            Expected market impact
        """
        participation = order_size / daily_volume
        impact = alpha * sigma * np.sqrt(participation)
        return impact

    def optimize_execution_schedule(
        self,
        total_volume: float,
        T: float,
        daily_volume: float,
        volatility: float,
        risk_aversion: float = 1e-6,
    ) -> np.ndarray:
        """
        Optimize execution schedule using Almgren-Chriss framework.

        Parameters:
        -----------
        total_volume : float
            Total volume to execute
        T : float
            Time horizon (in days)
        daily_volume : float
            Average daily volume
        volatility : float
            Volatility
        risk_aversion : float
            Risk aversion parameter (lambda)

        Returns:
        --------
        np.ndarray
            Optimal execution schedule
        """
        n_intervals = int(T * 78)  # 5-minute intervals
        t = np.linspace(0, T, n_intervals)

        # Simplified optimal trajectory (linear with adjustments)
        # For full Almgren-Chriss, solve the differential equation
        kappa = np.sqrt(risk_aversion * volatility**2 / 2)

        remaining_volume = total_volume * (
            np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
        )
        execution_schedule = -np.diff(remaining_volume, prepend=total_volume)

        return execution_schedule


class TickDataAnalyzer:
    """
    Analyze tick-by-tick market data.
    """

    @staticmethod
    def lee_ready_algorithm(
        trade_price: float, bid_price: float, ask_price: float
    ) -> int:
        """
        Lee-Ready algorithm for trade classification.

        Classifies trades as buyer-initiated (+1) or seller-initiated (-1).

        Parameters:
        -----------
        trade_price : float
            Price of the trade
        bid_price : float
            Prevailing bid price
        ask_price : float
            Prevailing ask price

        Returns:
        --------
        int
            +1 for buyer-initiated, -1 for seller-initiated, 0 for midpoint
        """
        mid_price = (bid_price + ask_price) / 2

        if trade_price > mid_price:
            return 1  # Buyer-initiated
        elif trade_price < mid_price:
            return -1  # Seller-initiated
        else:
            # Tick test: compare with previous trade price
            return 0  # Would need previous price for full implementation

    @staticmethod
    def vpin(
        buy_volume: np.ndarray, sell_volume: np.ndarray, bucket_size: int = 50
    ) -> np.ndarray:
        """
        Calculate Volume-Synchronized Probability of Informed Trading (VPIN).

        VPIN is a high-frequency measure of order flow toxicity.

        Parameters:
        -----------
        buy_volume : np.ndarray
            Buyer-initiated volume
        sell_volume : np.ndarray
            Seller-initiated volume
        bucket_size : int
            Number of buckets for volume synchronization

        Returns:
        --------
        np.ndarray
            VPIN values
        """
        total_volume = buy_volume + sell_volume
        volume_imbalance = np.abs(buy_volume - sell_volume)

        # Create volume buckets
        np.cumsum(total_volume)
        np.max(total_volume) * bucket_size

        # Calculate VPIN for each bucket
        vpin_values = []
        for i in range(len(total_volume)):
            window_imbalance = volume_imbalance[max(0, i - bucket_size) : i + 1]
            window_volume = total_volume[max(0, i - bucket_size) : i + 1]

            if np.sum(window_volume) > 0:
                vpin_val = np.sum(window_imbalance) / np.sum(window_volume)
            else:
                vpin_val = 0

            vpin_values.append(vpin_val)

        return np.array(vpin_values)

    @staticmethod
    def roll_spread_estimator(price_changes: np.ndarray) -> float:
        """
        Roll's estimator for effective spread.

        Estimates the bid-ask spread from price changes (assuming random walk).

        Parameters:
        -----------
        price_changes : np.ndarray
            Series of price changes

        Returns:
        --------
        float
            Estimated effective spread
        """
        # Calculate serial covariance
        covariance = np.cov(price_changes[1:], price_changes[:-1])[0, 1]

        if covariance >= 0:
            return 0.0  # No spread if positive serial correlation

        spread = 2 * np.sqrt(-covariance)
        return spread

    @staticmethod
    def trade_duration_analysis(timestamps: pd.DatetimeIndex) -> Dict[str, float]:
        """
        Analyze trade duration (time between trades).

        Parameters:
        -----------
        timestamps : pd.DatetimeIndex
            Timestamps of trades

        Returns:
        --------
        Dict[str, float]
            Statistics on trade duration
        """
        durations = (
            np.diff(timestamps).astype("timedelta64[ms]").astype(float) / 1000
        )  # in seconds

        return {
            "mean_duration": np.mean(durations),
            "median_duration": np.median(durations),
            "std_duration": np.std(durations),
            "min_duration": np.min(durations),
            "max_duration": np.max(durations),
            "trade_intensity": (
                1 / np.mean(durations) if np.mean(durations) > 0 else 0
            ),  # trades per second
        }


# Utility functions
def calculate_microprice(
    bid_price: float, ask_price: float, bid_volume: float, ask_volume: float
) -> float:
    """
    Calculate microprice (volume-weighted mid-price).

    Microprice = (Ask Volume * Bid Price + Bid Volume * Ask Price) / (Bid Volume + Ask Volume)

    Parameters:
    -----------
    bid_price : float
        Best bid price
    ask_price : float
        Best ask price
    bid_volume : float
        Volume at best bid
    ask_volume : float
        Volume at best ask

    Returns:
    --------
    float
        Microprice estimate
    """
    total_volume = bid_volume + ask_volume
    if total_volume == 0:
        return (bid_price + ask_price) / 2

    microprice = (ask_volume * bid_price + bid_volume * ask_price) / total_volume
    return microprice


def adverse_selection_cost(
    trade_price: float, future_mid_price: float, initial_mid_price: float
) -> float:
    """
    Calculate adverse selection cost component.

    Measures the cost of trading against informed traders.
    """
    return future_mid_price - initial_mid_price


__all__ = [
    "OrderFlowImbalance",
    "VolumeWeightedSpread",
    "RealizedVolatility",
    "MarketImpactModel",
    "TickDataAnalyzer",
    "calculate_microprice",
    "adverse_selection_cost",
]
