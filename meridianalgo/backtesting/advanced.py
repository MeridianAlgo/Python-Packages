"""
Advanced backtesting engine with realistic market simulation.
"""

import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..core import calculate_metrics
from ..strategies import BaseStrategy
from ..utils.performance import monitor_memory_usage


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    initial_capital: float = 100000
    commission: float = 0.001
    slippage: float = 0.0001
    borrow_rate: float = 0.0001
    short_selling_allowed: bool = True
    leverage: float = 1.0
    rebalance_frequency: str = "daily"
    benchmark: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@dataclass
class Trade:
    """Record of a single trade."""

    timestamp: datetime
    asset: str
    action: str  # 'buy' or 'sell'
    quantity: float
    price: float
    commission: float
    slippage: float
    total_cost: float


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state at a point in time."""

    timestamp: datetime
    cash: float
    positions: Dict[str, float]
    portfolio_value: float
    total_return: float
    daily_return: float


class BacktestEngine:
    """Advanced backtesting engine with realistic market simulation."""

    def __init__(self, config: BacktestConfig):
        """Initialize backtesting engine.

        Args:
            config: Backtest configuration
        """
        self.config = config
        self.reset()

    def reset(self):
        """Reset backtesting state."""
        self.cash = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        self.current_date = None
        self.market_data = None
        self.benchmark_returns = None

    @monitor_memory_usage(threshold=80.0)
    def run_backtest(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Run backtest with given strategy and data.

        Args:
            strategy: Trading strategy to backtest
            data: Market data for backtesting
            benchmark_data: Optional benchmark data

        Returns:
            Dictionary with backtest results
        """
        self.reset()
        self.market_data = data

        # Filter data by date range if specified
        if self.config.start_date:
            data = data[data.index >= self.config.start_date]
        if self.config.end_date:
            data = data[data.index <= self.config.end_date]

        # Set benchmark data
        if benchmark_data is not None:
            self.benchmark_returns = benchmark_data.pct_change().dropna()

        # Generate strategy signals
        signals = strategy.generate_signals(data)
        positions = strategy.calculate_positions(signals)

        # Run simulation
        for date, row in data.iterrows():
            self.current_date = date
            self._process_date(
                date, row, positions.loc[date] if date in positions.index else None
            )

        # Calculate final results
        results = self._calculate_results()

        return results

    def _process_date(
        self,
        date: datetime,
        market_data: pd.Series,
        target_positions: Optional[pd.Series],
    ):
        """Process a single day in the backtest."""
        # Update portfolio value
        portfolio_value = self._calculate_portfolio_value(market_data)

        # Calculate daily return
        if len(self.portfolio_history) > 0:
            prev_value = self.portfolio_history[-1].portfolio_value
            daily_return = (portfolio_value - prev_value) / prev_value
        else:
            daily_return = 0.0

        # Rebalance if target positions provided
        if target_positions is not None:
            self._rebalance(target_positions, market_data)

        # Record portfolio snapshot
        snapshot = PortfolioSnapshot(
            timestamp=date,
            cash=self.cash,
            positions=self.positions.copy(),
            portfolio_value=portfolio_value,
            total_return=(portfolio_value - self.config.initial_capital)
            / self.config.initial_capital,
            daily_return=daily_return,
        )
        self.portfolio_history.append(snapshot)

    def _calculate_portfolio_value(self, market_data: pd.Series) -> float:
        """Calculate current portfolio value."""
        portfolio_value = self.cash

        for asset, quantity in self.positions.items():
            if asset in market_data and quantity != 0:
                portfolio_value += quantity * market_data[asset]

        return portfolio_value

    def _rebalance(self, target_positions: pd.Series, market_data: pd.Series):
        """Rebalance portfolio to target positions."""
        # Calculate current portfolio value
        portfolio_value = self._calculate_portfolio_value(market_data)

        # Calculate target quantities
        for asset, target_weight in target_positions.items():
            if asset not in market_data:
                continue

            current_quantity = self.positions.get(asset, 0)
            target_value = portfolio_value * target_weight * self.config.leverage
            target_quantity = target_value / market_data[asset]

            # Calculate trade quantity
            trade_quantity = target_quantity - current_quantity

            if abs(trade_quantity) > 0.01:  # Minimum trade size
                self._execute_trade(asset, trade_quantity, market_data[asset])

    def _execute_trade(self, asset: str, quantity: float, price: float):
        """Execute a trade with realistic costs."""
        # Apply slippage
        if quantity > 0:  # Buy
            execution_price = price * (1 + self.config.slippage)
        else:  # Sell
            execution_price = price * (1 - self.config.slippage)

        # Calculate commission
        trade_value = abs(quantity * execution_price)
        commission = trade_value * self.config.commission

        # Check short selling constraints
        if quantity < 0 and not self.config.short_selling_allowed:
            warnings.warn(f"Short selling not allowed for {asset}")
            return

        # Execute trade
        total_cost = quantity * execution_price + commission

        if quantity > 0:  # Buy
            if self.cash >= total_cost:
                self.cash -= total_cost
                self.positions[asset] = self.positions.get(asset, 0) + quantity
            else:
                warnings.warn(f"Insufficient cash to buy {asset}")
        else:  # Sell
            current_position = self.positions.get(asset, 0)
            if abs(quantity) <= current_position:
                self.cash -= total_cost  # Note: total_cost is negative for sells
                self.positions[asset] = current_position + quantity
                if abs(self.positions[asset]) < 0.01:
                    del self.positions[asset]
            else:
                warnings.warn(f"Insufficient position to sell {asset}")

        # Record trade
        trade = Trade(
            timestamp=self.current_date,
            asset=asset,
            action="buy" if quantity > 0 else "sell",
            quantity=quantity,
            price=execution_price,
            commission=commission,
            slippage=abs(price - execution_price),
            total_cost=total_cost,
        )
        self.trades.append(trade)

    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate comprehensive backtest results."""
        if not self.portfolio_history:
            return {}

        # Extract time series
        dates = [s.timestamp for s in self.portfolio_history]
        portfolio_values = [s.portfolio_value for s in self.portfolio_history]
        daily_returns = [s.daily_return for s in self.portfolio_history]

        # Create Series
        returns_series = pd.Series(daily_returns, index=dates)

        # Calculate metrics
        metrics = calculate_metrics(returns_series)

        # Add additional metrics
        metrics.update(
            {
                "total_trades": len(self.trades),
                "win_rate": self._calculate_win_rate(),
                "profit_factor": self._calculate_profit_factor(),
                "average_trade": self._calculate_average_trade(),
                "max_consecutive_losses": self._calculate_max_consecutive_losses(),
                "turnover": self._calculate_turnover(),
            }
        )

        # Calculate benchmark metrics if available
        benchmark_metrics = {}
        if self.benchmark_returns is not None:
            benchmark_returns_aligned = self.benchmark_returns.reindex(
                dates, method="ffill"
            )
            benchmark_metrics = calculate_metrics(benchmark_returns_aligned.dropna())

        return {
            "metrics": metrics,
            "benchmark_metrics": benchmark_metrics,
            "portfolio_history": self.portfolio_history,
            "trades": self.trades,
            "returns_series": returns_series,
            "portfolio_values": pd.Series(portfolio_values, index=dates),
        }

    def _calculate_win_rate(self) -> float:
        """Calculate win rate of trades."""
        if not self.trades:
            return 0.0

        winning_trades = 0
        for trade in self.trades:
            # Simplified win calculation (would need more complex logic for real P&L)
            if trade.action == "sell" and trade.total_cost < 0:
                winning_trades += 1

        return winning_trades / len(self.trades)

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        # Simplified calculation
        gross_profit = sum(
            t.total_cost for t in self.trades if t.action == "sell" and t.total_cost < 0
        )
        gross_loss = sum(
            t.total_cost for t in self.trades if t.action == "sell" and t.total_cost > 0
        )

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 1.0

        return abs(gross_profit / gross_loss)

    def _calculate_average_trade(self) -> float:
        """Calculate average trade P&L."""
        if not self.trades:
            return 0.0

        total_pnl = sum(t.total_cost for t in self.trades if t.action == "sell")
        return total_pnl / len(self.trades)

    def _calculate_max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losses."""
        max_consecutive = 0
        current_consecutive = 0

        for trade in self.trades:
            if trade.action == "sell" and trade.total_cost > 0:  # Loss
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _calculate_turnover(self) -> float:
        """Calculate portfolio turnover."""
        if not self.trades:
            return 0.0

        total_traded = sum(abs(t.quantity * t.price) for t in self.trades)
        avg_portfolio_value = np.mean(
            [s.portfolio_value for s in self.portfolio_history]
        )

        return total_traded / (avg_portfolio_value * len(self.portfolio_history) / 252)


class MultiStrategyBacktester:
    """Backtest multiple strategies simultaneously."""

    def __init__(self, config: BacktestConfig):
        """Initialize multi-strategy backtester."""
        self.config = config
        self.results = {}

    def run_backtests(
        self,
        strategies: Dict[str, BaseStrategy],
        data: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run backtests for multiple strategies."""
        for name, strategy in strategies.items():
            print(f"Running backtest for {name}...")
            engine = BacktestEngine(self.config)
            self.results[name] = engine.run_backtest(strategy, data, benchmark_data)

        return self.results

    def compare_strategies(self) -> pd.DataFrame:
        """Compare performance of all strategies."""
        comparison_data = []

        for name, result in self.results.items():
            metrics = result["metrics"]
            comparison_data.append(
                {
                    "Strategy": name,
                    "Total Return": metrics.get("total_return", 0),
                    "Annualized Return": metrics.get("annualized_return", 0),
                    "Annualized Volatility": metrics.get("annualized_volatility", 0),
                    "Sharpe Ratio": metrics.get("sharpe_ratio", 0),
                    "Max Drawdown": metrics.get("max_drawdown", 0),
                    "Win Rate": metrics.get("win_rate", 0),
                    "Total Trades": metrics.get("total_trades", 0),
                }
            )

        return pd.DataFrame(comparison_data).set_index("Strategy")

    def get_best_strategy(
        self, metric: str = "sharpe_ratio"
    ) -> Tuple[str, Dict[str, Any]]:
        """Get the best performing strategy by metric."""
        best_name = max(
            self.results.keys(),
            key=lambda x: self.results[x]["metrics"].get(metric, -float("inf")),
        )
        return best_name, self.results[best_name]
