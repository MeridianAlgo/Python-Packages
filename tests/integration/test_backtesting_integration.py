"""
Integration test for event-driven backtesting framework.
"""

import os
import sys

import numpy as np
import pandas as pd

# Add the package to path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def test_events_system():
    """Test the events system."""
    print("Testing Events System...")

    try:
        from datetime import datetime

        from meridianalgo.backtesting.events import (EventQueue, MarketEvent,
                                                     SignalEvent, SignalType)

        # Test event creation
        market_event = MarketEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            open_price=150.0,
            high_price=152.0,
            low_price=149.0,
            close_price=151.0,
            volume=1000000,
        )

        signal_event = SignalEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.BUY,
            target_quantity=100,
        )

        print(" Events created successfully")

        # Test event queue
        queue = EventQueue()
        queue.put(market_event)
        queue.put(signal_event)

        assert queue.size() == 2
        assert not queue.empty()

        event1 = queue.get()
        event2 = queue.get()

        assert queue.empty()
        print(" Event queue working correctly")

        return True

    except Exception as e:
        print(f" Events system test failed: {e}")
        return False


def test_market_simulator():
    """Test the market simulator."""
    print("\nTesting Market Simulator...")

    try:
        from datetime import datetime

        from meridianalgo.backtesting.events import (MarketEvent, OrderEvent,
                                                     OrderSide, OrderType)
        from meridianalgo.backtesting.market_simulator import MarketSimulator

        # Create simulator
        simulator = MarketSimulator()

        # Create market event
        market_event = MarketEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            open_price=150.0,
            high_price=152.0,
            low_price=149.0,
            close_price=151.0,
            volume=1000000,
        )

        # Update market state
        simulator.update_market_state(market_event)

        # Create and execute order
        order = OrderEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
        )

        fills = simulator.execute_order(order)

        assert len(fills) == 1
        assert fills[0].fill_quantity == 100
        print(" Market simulator working correctly")

        return True

    except Exception as e:
        print(f" Market simulator test failed: {e}")
        return False


def test_backtesting_framework():
    """Test the complete backtesting framework."""
    print("\nTesting Backtesting Framework...")

    try:
        from meridianalgo.backtesting.backtester import (BuyAndHoldStrategy,
                                                         EventDrivenBacktester,
                                                         PandasDataHandler)

        # Create sample data
        dates = pd.date_range("2023-01-01", "2023-01-31", freq="D")
        np.random.seed(42)

        symbols = ["AAPL"]
        data = {}

        for symbol in symbols:
            prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
            data[f"{symbol}_Close"] = prices
            data[f"{symbol}_Volume"] = np.random.randint(100000, 1000000, len(dates))

        market_data = pd.DataFrame(data, index=dates)

        # Create backtester
        backtester = EventDrivenBacktester(initial_capital=10000)

        # Set data handler and strategy
        data_handler = PandasDataHandler(market_data, symbols)
        strategy = BuyAndHoldStrategy(symbols)

        backtester.set_data_handler(data_handler)
        backtester.set_strategy(strategy)

        # Run backtest
        result = backtester.run_backtest()

        assert result.success
        assert result.initial_capital == 10000
        assert result.final_value > 0
        assert result.total_trades >= 0

        print(" Backtesting framework working correctly")
        print(f"  Initial Capital: ${result.initial_capital:,.2f}")
        print(f"  Final Value: ${result.final_value:,.2f}")
        print(f"  Total Return: {result.total_return:.2%}")
        print(f"  Total Trades: {result.total_trades}")

        return True

    except Exception as e:
        print(f" Backtesting framework test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_portfolio_tracking():
    """Test portfolio position tracking."""
    print("\nTesting Portfolio Tracking...")

    try:
        from datetime import datetime

        from meridianalgo.backtesting.backtester import Portfolio
        from meridianalgo.backtesting.events import (FillEvent, FillStatus,
                                                     OrderSide)

        # Create portfolio
        portfolio = Portfolio(initial_cash=10000)

        # Create fill event
        fill = FillEvent(
            timestamp=datetime.now(),
            order_id="TEST_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=150.0,
            fill_quantity=100,
            remaining_quantity=0,
            commission=1.0,
            fill_status=FillStatus.FILLED,
        )

        # Process fill
        portfolio.process_fill(fill)

        # Check position
        position = portfolio.get_position("AAPL")
        assert position.quantity == 100
        assert position.avg_cost == 150.0
        assert portfolio.cash == 10000 - (100 * 150.0) - 1.0

        print(" Portfolio tracking working correctly")
        print(f"  Position: {position.quantity} shares @ ${position.avg_cost:.2f}")
        print(f"  Remaining cash: ${portfolio.cash:.2f}")

        return True

    except Exception as e:
        print(f" Portfolio tracking test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Event-Driven Backtesting Framework Integration Tests")
    print("=" * 60)

    tests = [
        test_events_system,
        test_market_simulator,
        test_portfolio_tracking,
        test_backtesting_framework,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n{'=' * 60}")
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print(" All integration tests passed!")
        return True
    else:
        print(" Some integration tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
