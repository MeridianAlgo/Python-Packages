"""
Simple backtesting engine for MeridianAlgo.
"""

from typing import Dict


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
        self.positions = {}
        self.transaction_log = []

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
                    return True
                return False

            elif side.lower() == "sell":
                if self.positions.get(symbol, 0) >= quantity:
                    proceeds = trade_value * (1 - self.commission - self.slippage)
                    self.cash += proceeds
                    self.positions[symbol] = self.positions.get(symbol, 0) - quantity
                    if self.positions[symbol] == 0:
                        del self.positions[symbol]
                    return True
                return False

            return False
        except Exception:
            return False

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value."""
        position_value = sum(
            self.positions.get(symbol, 0) * current_prices.get(symbol, 0)
            for symbol in self.positions
        )
        return self.cash + position_value

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get basic performance metrics."""
        # This is a simplified version
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
        }
