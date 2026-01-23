"""
Liquidity Metrics Module

Key liquidity measures including Amihud illiquidity, turnover,
and various liquidity ratios.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


class LiquidityMetrics:
    """
    Calculate various liquidity metrics.

    Provides:
    - Amihud illiquidity
    - Roll measure
    - Turnover ratio
    - Bid-ask spread proxies
    - Volume-based measures
    """

    def __init__(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        market_cap: Optional[float] = None,
        shares_outstanding: Optional[float] = None,
    ):
        """
        Initialize LiquidityMetrics.

        Args:
            prices: Price series
            volumes: Volume series (in shares or dollars)
            market_cap: Market capitalization
            shares_outstanding: Total shares outstanding
        """
        self.prices = prices
        self.volumes = volumes
        self.market_cap = market_cap
        self.shares_outstanding = shares_outstanding

        # Calculate returns
        self.returns = prices.pct_change().dropna()

    def calculate_all(self) -> Dict[str, float]:
        """Calculate all liquidity metrics."""
        return {
            "amihud_illiquidity": self.amihud_illiquidity(),
            "roll_measure": self.roll_measure(),
            "turnover_ratio": self.turnover_ratio(),
            "avg_volume": self.volumes.mean(),
            "volume_volatility": self.volumes.std() / self.volumes.mean(),
            "zero_volume_days": (self.volumes == 0).mean(),
            "kyle_lambda": self.kyles_lambda(),
            "volume_return_correlation": self.volume_return_correlation(),
        }


class AmmihudIlliquidity:
    """
    Amihud (2002) illiquidity measure.

    ILLIQ = |R| / Volume

    Measures price impact per unit of trading volume.
    Higher values indicate lower liquidity.
    """

    def __init__(
        self, returns: pd.Series, volumes: pd.Series, dollar_volume: bool = True
    ):
        """
        Initialize AmmihudIlliquidity.

        Args:
            returns: Return series
            volumes: Volume series
            dollar_volume: If True, volumes are in dollars; else in shares
        """
        self.returns = returns
        self.volumes = volumes
        self.dollar_volume = dollar_volume

    def calculate(self) -> float:
        """
        Calculate average Amihud illiquidity.

        Returns:
            Amihud illiquidity ratio
        """
        # Filter out zero volume days
        valid = self.volumes > 0

        if not valid.any():
            return np.inf

        illiq = np.abs(self.returns[valid]) / self.volumes[valid]

        # Scale for readability (multiply by 10^6 for typical values)
        return illiq.mean() * 1e6

    def rolling(self, window: int = 21) -> pd.Series:
        """
        Calculate rolling Amihud illiquidity.

        Args:
            window: Rolling window size

        Returns:
            Rolling illiquidity series
        """
        abs_returns = np.abs(self.returns)

        # Rolling ratio
        return (abs_returns / self.volumes).rolling(window).mean() * 1e6

    def monthly(self) -> pd.Series:
        """Calculate monthly average Amihud illiquidity."""
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            return pd.Series()

        daily_illiq = np.abs(self.returns) / self.volumes.replace(0, np.nan)
        return daily_illiq.resample("M").mean() * 1e6

    def standardized(self) -> float:
        """
        Calculate standardized Amihud ratio.

        Adjusted for market-wide liquidity.
        """
        illiq = self.calculate()

        # Standardize by historical mean and std
        hist_mean = self.rolling(252).mean()
        hist_std = self.rolling(252).std()

        if hist_std > 0:
            return (illiq - hist_mean) / hist_std
        return 0


class TurnoverRatio:
    """
    Calculate turnover ratio and related liquidity measures.

    Turnover = Trading Volume / Shares Outstanding

    Measures how frequently shares change hands.
    """

    def __init__(self, volumes: pd.Series, shares_outstanding: float):
        """
        Initialize TurnoverRatio.

        Args:
            volumes: Daily trading volume series
            shares_outstanding: Total shares outstanding
        """
        self.volumes = volumes
        self.shares_outstanding = shares_outstanding

    def daily_turnover(self) -> pd.Series:
        """Calculate daily turnover ratio."""
        return self.volumes / self.shares_outstanding

    def average_turnover(self) -> float:
        """Calculate average daily turnover."""
        return self.daily_turnover().mean()

    def annualized_turnover(self) -> float:
        """Calculate annualized turnover (assuming 252 trading days)."""
        return self.average_turnover() * 252

    def rolling_turnover(self, window: int = 21) -> pd.Series:
        """Calculate rolling average turnover."""
        return self.daily_turnover().rolling(window).mean()

    def turnover_velocity(self) -> float:
        """
        Calculate turnover velocity.

        Days to trade entire float.
        """
        avg_daily = self.average_turnover()
        if avg_daily == 0:
            return np.inf
        return 1 / avg_daily

    def turnover_percentile(self, current: Optional[float] = None) -> float:
        """
        Get percentile rank of turnover.

        Args:
            current: Current turnover value (default: latest)

        Returns:
            Percentile (0-100)
        """
        daily = self.daily_turnover()

        if current is None:
            current = daily.iloc[-1]

        return (daily < current).mean() * 100

    def summary(self) -> Dict[str, float]:
        """Generate turnover summary."""
        return {
            "avg_daily_turnover": self.average_turnover(),
            "annualized_turnover": self.annualized_turnover(),
            "turnover_velocity_days": self.turnover_velocity(),
            "turnover_volatility": self.daily_turnover().std(),
            "current_turnover": self.daily_turnover().iloc[-1],
            "turnover_percentile": self.turnover_percentile(),
        }


# ============================================================================
# ADDITIONAL LIQUIDITY FUNCTIONS
# ============================================================================


def roll_measure(prices: pd.Series) -> float:
    """
    Calculate Roll (1984) effective spread measure.

    Based on serial covariance of price changes.

    Args:
        prices: Price series

    Returns:
        Estimated effective spread
    """
    returns = prices.pct_change().dropna()

    # Serial covariance
    cov = returns.autocorr(1)

    if cov >= 0:
        return 0  # No meaningful estimate when covariance is positive

    # Roll measure
    return 2 * np.sqrt(-cov) * prices.mean()


def corwin_schultz_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Calculate Corwin-Schultz (2012) spread estimator.

    Uses high-low prices to estimate bid-ask spread.

    Args:
        high: High price series
        low: Low price series

    Returns:
        Estimated spread series
    """
    # Beta calculation
    log_hl = np.log(high / low)
    log_hl_sq = log_hl**2

    # Two-period high and low
    high_2 = high.rolling(2).max()
    low_2 = low.rolling(2).min()

    gamma = np.log(high_2 / low_2) ** 2
    beta = log_hl_sq + log_hl_sq.shift(1)

    # Spread estimate
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(
        gamma / (3 - 2 * np.sqrt(2))
    )

    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

    # Clean up negative values
    spread = spread.clip(lower=0)

    return spread


def pastor_stambaugh_liquidity(
    returns: pd.Series, market_returns: pd.Series, volumes: pd.Series
) -> float:
    """
    Calculate Pastor-Stambaugh (2003) liquidity measure.

    Measures return reversal following volume shocks.

    Args:
        returns: Asset returns
        market_returns: Market returns
        volumes: Trading volumes

    Returns:
        Liquidity gamma coefficient
    """
    # Align all series
    data = pd.DataFrame(
        {"ret": returns, "mkt": market_returns, "vol": volumes}
    ).dropna()

    if len(data) < 30:
        return 0

    # Lagged terms
    data["ret_lag"] = data["ret"].shift(1)
    data["vol_lag"] = data["vol"].shift(1)
    data["signed_vol"] = data["vol_lag"] * np.sign(data["ret_lag"])

    # Regression: r_t = alpha + beta * mkt_t + gamma * sign(r_t-1) * vol_t-1 + epsilon
    data = data.dropna()

    X = np.column_stack(
        [np.ones(len(data)), data["mkt"].values, data["signed_vol"].values]
    )
    y = data["ret"].values

    try:
        betas = np.linalg.lstsq(X, y, rcond=None)[0]
        gamma = betas[2]  # Liquidity coefficient
    except Exception:
        gamma = 0

    return gamma


def zero_return_days(returns: pd.Series) -> float:
    """
    Calculate fraction of zero return days.

    Higher values indicate lower liquidity.

    Args:
        returns: Return series

    Returns:
        Fraction of days with zero returns
    """
    return (returns == 0).mean()


def liu_liquidity(
    returns: pd.Series, volumes: pd.Series, turnover: pd.Series, months: int = 12
) -> float:
    """
    Calculate Liu (2006) liquidity measure.

    Combines trading speed, trading quantity, and trading cost.

    Args:
        returns: Return series
        volumes: Volume series
        turnover: Turnover series
        months: Number of months for calculation

    Returns:
        Standardized Liu liquidity measure
    """
    # Number of zero volume days
    zero_days = (volumes == 0).sum()

    # 1/turnover deflator
    turnover_12m = turnover.tail(months * 21).mean() * 252
    deflator = 1 / turnover_12m if turnover_12m > 0 else 0

    # LM measure
    lm = zero_days + deflator / 11000

    return lm
