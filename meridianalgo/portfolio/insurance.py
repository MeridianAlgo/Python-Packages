"""
Portfolio Insurance Strategies

CPPI (Constant Proportion Portfolio Insurance) and dynamic floor management.
Protects a minimum portfolio value while participating in upside returns.

References:
    Black & Jones (1987) - Simplifying Portfolio Insurance
    Perold & Sharpe (1988) - Dynamic Strategies for Asset Allocation
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CPPIResult:
    """Container for CPPI backtest results."""

    portfolio_value: pd.Series
    floor_value: pd.Series
    cushion: pd.Series
    risky_allocation: pd.Series
    safe_allocation: pd.Series
    risky_weight: pd.Series
    total_return: float
    annualized_return: float
    annualized_volatility: float
    max_drawdown: float
    floor_breaches: int
    multiplier: float
    floor_rate: float


class CPPI:
    """
    Constant Proportion Portfolio Insurance (CPPI).

    Dynamically allocates between a risky asset (growth) and a safe asset
    (floor protection) based on the cushion = portfolio value - floor value.
    The risky allocation = multiplier * cushion.

    When the cushion approaches zero, the strategy shifts entirely to the
    safe asset, locking in the floor protection.

    Example:
        >>> cppi = CPPI(multiplier=3.0, floor_pct=0.80, safe_rate=0.04)
        >>> result = cppi.run(
        ...     risky_returns=equity_returns,
        ...     initial_value=100_000,
        ... )
        >>> print(f"Total Return:    {result.total_return:.2%}")
        >>> print(f"Floor Breaches:  {result.floor_breaches}")
        >>> print(f"Max Drawdown:    {result.max_drawdown:.2%}")
    """

    def __init__(
        self,
        multiplier: float = 3.0,
        floor_pct: float = 0.80,
        safe_rate: float = 0.02,
        rebalance_frequency: int = 1,
        max_leverage: float = 1.0,
    ):
        """
        Parameters
        ----------
        multiplier : float
            CPPI multiplier (higher = more aggressive participation)
        floor_pct : float
            Floor as fraction of initial value (e.g., 0.80 = 80% protection)
        safe_rate : float
            Annual return on safe asset (money market / T-bills)
        rebalance_frequency : int
            Rebalancing interval in periods (1 = daily)
        max_leverage : float
            Maximum risky asset weight (1.0 = no leverage)
        """
        if multiplier <= 0:
            raise ValueError("multiplier must be positive")
        if not 0 < floor_pct <= 1:
            raise ValueError("floor_pct must be in (0, 1]")

        self.multiplier = multiplier
        self.floor_pct = floor_pct
        self.safe_rate = safe_rate
        self.rebalance_frequency = rebalance_frequency
        self.max_leverage = max_leverage

    def run(
        self,
        risky_returns: pd.Series,
        initial_value: float = 1.0,
        safe_returns: Optional[pd.Series] = None,
        trading_days: int = 252,
    ) -> CPPIResult:
        """
        Run CPPI strategy on historical returns.

        Parameters
        ----------
        risky_returns : pd.Series
            Return series for the risky (growth) asset
        initial_value : float
            Starting portfolio value
        safe_returns : pd.Series, optional
            Return series for the safe asset. If None, uses constant safe_rate/trading_days.
        trading_days : int
            Trading days per year (for annualization)

        Returns
        -------
        CPPIResult
        """
        returns = risky_returns.dropna()
        n = len(returns)

        if safe_returns is None:
            safe_ret = pd.Series(
                self.safe_rate / trading_days,
                index=returns.index,
            )
        else:
            safe_ret = safe_returns.reindex(returns.index).fillna(
                self.safe_rate / trading_days
            )

        floor_initial = initial_value * self.floor_pct

        portfolio_values = np.zeros(n + 1)
        floor_values = np.zeros(n + 1)
        cushions = np.zeros(n + 1)
        risky_allocs = np.zeros(n + 1)
        safe_allocs = np.zeros(n + 1)
        risky_weights = np.zeros(n + 1)

        portfolio_values[0] = initial_value
        floor_values[0] = floor_initial
        cushion = initial_value - floor_initial
        cushions[0] = cushion

        risky_alloc = min(self.multiplier * cushion, self.max_leverage * initial_value)
        risky_alloc = max(risky_alloc, 0)
        safe_alloc = initial_value - risky_alloc
        risky_allocs[0] = risky_alloc
        safe_allocs[0] = safe_alloc
        risky_weights[0] = risky_alloc / initial_value if initial_value > 0 else 0

        floor_breaches = 0

        for t, (r_ret, s_ret) in enumerate(zip(returns.values, safe_ret.values)):
            risky_alloc = risky_allocs[t]
            safe_alloc = safe_allocs[t]
            portfolio_val = portfolio_values[t]

            risky_alloc_new = risky_alloc * (1 + r_ret)
            safe_alloc_new = safe_alloc * (1 + s_ret)
            new_portfolio = risky_alloc_new + safe_alloc_new

            new_floor = floor_values[t] * (1 + s_ret)

            if new_portfolio < new_floor:
                floor_breaches += 1

            portfolio_values[t + 1] = new_portfolio
            floor_values[t + 1] = new_floor
            cushion = max(new_portfolio - new_floor, 0)
            cushions[t + 1] = cushion

            if (t + 1) % self.rebalance_frequency == 0:
                risky_alloc = min(
                    self.multiplier * cushion,
                    self.max_leverage * new_portfolio,
                )
                risky_alloc = max(risky_alloc, 0)
                safe_alloc = new_portfolio - risky_alloc
            else:
                risky_alloc = risky_alloc_new
                safe_alloc = safe_alloc_new

            risky_allocs[t + 1] = risky_alloc
            safe_allocs[t + 1] = safe_alloc
            risky_weights[t + 1] = risky_alloc / new_portfolio if new_portfolio > 0 else 0

        index_full = pd.Index([returns.index[0]] + list(returns.index))
        pv = pd.Series(portfolio_values, index=index_full, name="portfolio_value")
        fv = pd.Series(floor_values, index=index_full, name="floor_value")
        cv = pd.Series(cushions, index=index_full, name="cushion")
        ra = pd.Series(risky_allocs, index=index_full, name="risky_allocation")
        sa = pd.Series(safe_allocs, index=index_full, name="safe_allocation")
        rw = pd.Series(risky_weights, index=index_full, name="risky_weight")

        total_return = (portfolio_values[-1] / initial_value) - 1
        n_years = n / trading_days
        ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        pv_series = pd.Series(portfolio_values[1:], index=returns.index)
        pv_returns = pv_series.pct_change().dropna()
        ann_vol = pv_returns.std() * np.sqrt(trading_days)

        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = float(np.min(drawdown))

        return CPPIResult(
            portfolio_value=pv,
            floor_value=fv,
            cushion=cv,
            risky_allocation=ra,
            safe_allocation=sa,
            risky_weight=rw,
            total_return=total_return,
            annualized_return=ann_return,
            annualized_volatility=ann_vol,
            max_drawdown=max_drawdown,
            floor_breaches=floor_breaches,
            multiplier=self.multiplier,
            floor_rate=self.floor_pct,
        )

    def sensitivity_analysis(
        self,
        risky_returns: pd.Series,
        multipliers: Optional[List[float]] = None,
        floor_pcts: Optional[List[float]] = None,
        initial_value: float = 1.0,
        trading_days: int = 252,
    ) -> pd.DataFrame:
        """
        Analyze CPPI performance across multiplier and floor combinations.

        Returns
        -------
        pd.DataFrame
            Total return, max drawdown, floor breaches for each parameter combo
        """
        if multipliers is None:
            multipliers = [1.0, 2.0, 3.0, 4.0, 5.0]
        if floor_pcts is None:
            floor_pcts = [0.70, 0.80, 0.90]

        rows = []
        for m in multipliers:
            for f in floor_pcts:
                cppi = CPPI(multiplier=m, floor_pct=f, safe_rate=self.safe_rate)
                result = cppi.run(risky_returns, initial_value, trading_days=trading_days)
                rows.append(
                    {
                        "multiplier": m,
                        "floor_pct": f,
                        "total_return": result.total_return,
                        "ann_return": result.annualized_return,
                        "ann_vol": result.annualized_volatility,
                        "max_drawdown": result.max_drawdown,
                        "floor_breaches": result.floor_breaches,
                    }
                )

        return pd.DataFrame(rows)


class TimeInvariantCPPI:
    """
    Time-Invariant Portfolio Protection (TIPP).

    Variant of CPPI where the floor is a fraction of the running maximum
    portfolio value rather than a fixed initial floor. Ratchets up the
    floor as the portfolio grows, locking in profits.

    Example:
        >>> tipp = TimeInvariantCPPI(multiplier=3.0, floor_pct=0.80)
        >>> result = tipp.run(equity_returns, initial_value=100_000)
    """

    def __init__(
        self,
        multiplier: float = 3.0,
        floor_pct: float = 0.80,
        safe_rate: float = 0.02,
    ):
        self.multiplier = multiplier
        self.floor_pct = floor_pct
        self.safe_rate = safe_rate

    def run(
        self,
        risky_returns: pd.Series,
        initial_value: float = 1.0,
        trading_days: int = 252,
    ) -> CPPIResult:
        """
        Run TIPP strategy. Floor ratchets up with portfolio peaks.

        Parameters
        ----------
        risky_returns : pd.Series
            Return series for the risky asset
        initial_value : float
            Starting portfolio value
        """
        returns = risky_returns.dropna()
        n = len(returns)
        safe_daily = self.safe_rate / trading_days

        portfolio_values = np.zeros(n + 1)
        floor_values = np.zeros(n + 1)
        cushions = np.zeros(n + 1)
        risky_allocs = np.zeros(n + 1)
        safe_allocs = np.zeros(n + 1)
        risky_weights = np.zeros(n + 1)

        portfolio_values[0] = initial_value
        running_max = initial_value
        floor_values[0] = running_max * self.floor_pct
        cushion = initial_value - floor_values[0]
        cushions[0] = cushion

        risky_alloc = min(self.multiplier * cushion, initial_value)
        risky_alloc = max(risky_alloc, 0)
        safe_alloc = initial_value - risky_alloc
        risky_allocs[0] = risky_alloc
        safe_allocs[0] = safe_alloc
        risky_weights[0] = risky_alloc / initial_value

        floor_breaches = 0

        for t, r_ret in enumerate(returns.values):
            risky_alloc_new = risky_allocs[t] * (1 + r_ret)
            safe_alloc_new = safe_allocs[t] * (1 + safe_daily)
            new_portfolio = risky_alloc_new + safe_alloc_new

            running_max = max(running_max, new_portfolio)
            new_floor = running_max * self.floor_pct

            if new_portfolio < new_floor:
                floor_breaches += 1

            portfolio_values[t + 1] = new_portfolio
            floor_values[t + 1] = new_floor
            cushion = max(new_portfolio - new_floor, 0)
            cushions[t + 1] = cushion

            risky_alloc = min(self.multiplier * cushion, new_portfolio)
            risky_alloc = max(risky_alloc, 0)
            safe_alloc = new_portfolio - risky_alloc
            risky_allocs[t + 1] = risky_alloc
            safe_allocs[t + 1] = safe_alloc
            risky_weights[t + 1] = risky_alloc / new_portfolio if new_portfolio > 0 else 0

        index_full = pd.Index([returns.index[0]] + list(returns.index))
        pv = pd.Series(portfolio_values, index=index_full, name="portfolio_value")
        fv = pd.Series(floor_values, index=index_full, name="floor_value")

        total_return = (portfolio_values[-1] / initial_value) - 1
        n_years = n / trading_days
        ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        pv_series = pd.Series(portfolio_values[1:], index=returns.index)
        pv_returns = pv_series.pct_change().dropna()
        ann_vol = pv_returns.std() * np.sqrt(trading_days)

        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = float(np.min(drawdown))

        return CPPIResult(
            portfolio_value=pv,
            floor_value=fv,
            cushion=pd.Series(cushions, index=index_full, name="cushion"),
            risky_allocation=pd.Series(risky_allocs, index=index_full),
            safe_allocation=pd.Series(safe_allocs, index=index_full),
            risky_weight=pd.Series(risky_weights, index=index_full),
            total_return=total_return,
            annualized_return=ann_return,
            annualized_volatility=ann_vol,
            max_drawdown=max_drawdown,
            floor_breaches=floor_breaches,
            multiplier=self.multiplier,
            floor_rate=self.floor_pct,
        )
