"""
Simple backtesting engine for MeridianAlgo.
"""

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class BacktestEngine:
    """Simple backtesting engine for strategy testing."""

    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ):
        """
        Initialize backtesting engine.

        Args:
            initial_capital: Starting capital
            commission: Commission rate (default: 0.1%)
            slippage: Slippage rate (default: 0.05%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.cash = initial_capital
        self.positions: Dict[str, int] = {}
        self.transaction_log: List[Dict] = []
        self.history: List[Dict] = []
        self.current_time: Optional[datetime] = None

    def update_time(self, timestamp: datetime):
        """Update current time for the engine."""
        self.current_time = timestamp

    def execute_order(
        self, symbol: str, order_type: str, side: str, quantity: int, price: float
    ) -> bool:
        """
        Execute a trading order.

        Args:
            symbol: Trading symbol
            order_type: Order type ('market', 'limit')
            side: Order side ('buy', 'sell')
            quantity: Number of shares
            price: Order price

        Returns:
            True if order executed successfully
        """
        try:
            trade_value = quantity * price
            total_cost = trade_value * (1 + self.commission + self.slippage)

            if side.lower() == "buy":
                if self.cash >= total_cost:
                    self.cash -= total_cost
                    self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                    self._log_transaction(symbol, side, quantity, price, total_cost)
                    return True
                return False

            elif side.lower() == "sell":
                if self.positions.get(symbol, 0) >= quantity:
                    proceeds = trade_value * (1 - self.commission - self.slippage)
                    self.cash += proceeds
                    self.positions[symbol] = self.positions.get(symbol, 0) - quantity
                    if self.positions[symbol] == 0:
                        del self.positions[symbol]
                    self._log_transaction(symbol, side, quantity, price, proceeds)
                    return True
                return False

            return False
        except Exception:
            return False

    def _log_transaction(
        self, symbol: str, side: str, quantity: int, price: float, value: float
    ):
        """Log a transaction."""
        self.transaction_log.append(
            {
                "timestamp": self.current_time or datetime.now(),
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "value": value,
                "commission": value * self.commission,
            }
        )

    def record_snapshot(self, current_prices: Dict[str, float]):
        """Record portfolio snapshot for performance tracking."""
        portfolio_value = self.get_portfolio_value(current_prices)
        self.history.append(
            {
                "timestamp": self.current_time or datetime.now(),
                "cash": self.cash,
                "portfolio_value": portfolio_value,
                "positions_value": portfolio_value - self.cash,
            }
        )

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value."""
        position_value = sum(
            self.positions.get(symbol, 0) * current_prices.get(symbol, 0)
            for symbol in self.positions
        )
        return self.cash + position_value

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics."""
        if not self.history:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
            }

        df = pd.DataFrame(self.history)
        df.set_index("timestamp", inplace=True)

        # Calculate returns
        df["returns"] = df["portfolio_value"].pct_change().fillna(0)

        # Calculate metrics
        total_return = (df["portfolio_value"].iloc[-1] / self.initial_capital) - 1

        # Annualized metrics (assuming daily data for simplicity, can be adjusted)
        annualized_return = df["returns"].mean() * 252
        annualized_volatility = df["returns"].std() * np.sqrt(252)

        sharpe_ratio = 0.0
        if annualized_volatility > 0:
            sharpe_ratio = annualized_return / annualized_volatility

        # Drawdown
        cumulative = (1 + df["returns"]).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate from transactions
        winning_trades = 0
        total_trades = 0

        # Simple win rate approximation based on completed round trips would be better
        # but for now we look at daily returns
        win_rate = (df["returns"] > 0).mean()

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
        }
