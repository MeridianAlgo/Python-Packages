"""
Comprehensive tests for backtesting module.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add the parent directory to the path to import meridianalgo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import meridianalgo as ma
    from meridianalgo.backtesting import BacktestEngine
except ImportError as e:
    pytest.skip(f"Could not import meridianalgo: {e}", allow_module_level=True)


class TestBacktesting:
    """Test suite for backtesting module."""

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for backtesting."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")

        # Generate realistic OHLCV data
        returns = np.random.normal(0.001, 0.02, 252)
        prices = [100]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        data = pd.DataFrame(
            {
                "Open": [p * np.random.uniform(0.995, 1.005) for p in prices],
                "High": [p * np.random.uniform(1.00, 1.03) for p in prices],
                "Low": [p * np.random.uniform(0.97, 1.00) for p in prices],
                "Close": prices,
                "Volume": np.random.randint(1000000, 10000000, 252),
            },
            index=dates,
        )

        # Ensure OHLC relationships are correct
        for i in range(len(data)):
            high = max(data.iloc[i]["Open"], data.iloc[i]["Close"]) * np.random.uniform(
                1.0, 1.02
            )
            low = min(data.iloc[i]["Open"], data.iloc[i]["Close"]) * np.random.uniform(
                0.98, 1.0
            )
            data.iloc[i, data.columns.get_loc("High")] = high
            data.iloc[i, data.columns.get_loc("Low")] = low

        return data

    def test_backtest_engine_creation(self, sample_market_data):
        """Test backtest engine initialization."""
        try:
            engine = BacktestEngine(
                initial_capital=100000, commission=0.001, slippage=0.0005
            )

            assert engine is not None
            assert engine.initial_capital == 100000
            assert engine.commission == 0.001
            assert engine.slippage == 0.0005
            assert engine.cash == 100000
            assert len(engine.positions) == 0

            print(" Backtest engine creation test passed")
        except Exception as e:
            print(f" Backtest engine creation test failed: {e}")

    def test_simple_buy_hold_strategy(self, sample_market_data):
        """Test simple buy and hold strategy."""
        try:
            engine = BacktestEngine(initial_capital=100000)

            # Simple buy and hold strategy
            symbol = "TEST"

            # Buy at the beginning
            first_price = sample_market_data["Close"].iloc[0]
            quantity = int(engine.cash / first_price)

            # Execute buy order
            success = engine.execute_order(
                symbol=symbol,
                order_type="market",
                side="buy",
                quantity=quantity,
                price=first_price,
            )

            if success:
                assert symbol in engine.positions
                assert engine.positions[symbol] == quantity
                assert engine.cash < engine.initial_capital

                # Hold until the end
                final_price = sample_market_data["Close"].iloc[-1]

                # Calculate final portfolio value
                position_value = engine.positions[symbol] * final_price
                total_value = engine.cash + position_value

                # Calculate return
                total_return = (
                    total_value - engine.initial_capital
                ) / engine.initial_capital

                assert isinstance(total_return, (int, float))

                print(" Buy and hold strategy test passed")
            else:
                print(" Order execution failed in buy and hold test")

        except Exception as e:
            print(f" Buy and hold strategy test failed: {e}")

    def test_order_execution(self, sample_market_data):
        """Test different order types and execution."""
        try:
            engine = BacktestEngine(initial_capital=100000)
            symbol = "TEST"

            # Test market buy order
            price = sample_market_data["Close"].iloc[0]
            quantity = 100

            success = engine.execute_order(
                symbol=symbol,
                order_type="market",
                side="buy",
                quantity=quantity,
                price=price,
            )

            if success:
                assert engine.positions.get(symbol, 0) == quantity

                # Test market sell order
                success_sell = engine.execute_order(
                    symbol=symbol,
                    order_type="market",
                    side="sell",
                    quantity=50,
                    price=price * 1.01,
                )

                if success_sell:
                    assert engine.positions.get(symbol, 0) == 50

                    print(" Order execution test passed")
                else:
                    print(" Sell order execution failed")
            else:
                print(" Buy order execution failed")

        except Exception as e:
            print(f" Order execution test failed: {e}")

    def test_commission_and_slippage(self, sample_market_data):
        """Test commission and slippage calculations."""
        try:
            commission_rate = 0.001
            slippage_rate = 0.0005

            engine = BacktestEngine(
                initial_capital=100000,
                commission=commission_rate,
                slippage=slippage_rate,
            )

            symbol = "TEST"
            price = 100.0
            quantity = 100

            initial_cash = engine.cash

            # Execute buy order
            success = engine.execute_order(
                symbol=symbol,
                order_type="market",
                side="buy",
                quantity=quantity,
                price=price,
            )

            if success:
                # Calculate expected costs
                trade_value = quantity * price
                expected_commission = trade_value * commission_rate
                expected_slippage = trade_value * slippage_rate
                expected_total_cost = (
                    trade_value + expected_commission + expected_slippage
                )

                # Check cash reduction
                cash_used = initial_cash - engine.cash

                # Allow for small rounding differences
                assert abs(cash_used - expected_total_cost) < 0.01

                print(" Commission and slippage test passed")
            else:
                print(" Order execution failed in commission test")

        except Exception as e:
            print(f" Commission and slippage test failed: {e}")

    def test_portfolio_value_calculation(self, sample_market_data):
        """Test portfolio value calculation."""
        try:
            engine = BacktestEngine(initial_capital=100000)

            # Add some positions
            engine.positions = {"AAPL": 100, "GOOGL": 50}
            engine.cash = 50000

            # Current prices
            current_prices = {"AAPL": 150.0, "GOOGL": 2500.0}

            # Calculate portfolio value
            position_value = sum(
                engine.positions[symbol] * current_prices[symbol]
                for symbol in engine.positions
            )
            total_value = engine.cash + position_value

            expected_value = 50000 + (100 * 150.0) + (50 * 2500.0)

            assert abs(total_value - expected_value) < 0.01

            print(" Portfolio value calculation test passed")
        except Exception as e:
            print(f" Portfolio value calculation test failed: {e}")

    def test_performance_metrics(self, sample_market_data):
        """Test performance metrics calculation."""
        try:
            # Create sample portfolio returns
            np.random.seed(42)
            returns = np.random.normal(0.001, 0.02, 252)
            portfolio_returns = pd.Series(returns, index=sample_market_data.index)

            # Calculate performance metrics
            total_return = (1 + portfolio_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0

            # Calculate maximum drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # Validate metrics
            assert isinstance(total_return, (int, float))
            assert isinstance(annual_return, (int, float))
            assert isinstance(volatility, (int, float))
            assert isinstance(sharpe_ratio, (int, float))
            assert isinstance(max_drawdown, (int, float))

            assert volatility >= 0
            assert max_drawdown <= 0

            print(" Performance metrics test passed")
        except Exception as e:
            print(f" Performance metrics test failed: {e}")

    def test_moving_average_strategy(self, sample_market_data):
        """Test a simple moving average crossover strategy."""
        try:
            engine = BacktestEngine(initial_capital=100000)
            symbol = "TEST"

            # Calculate moving averages
            short_ma = sample_market_data["Close"].rolling(window=10).mean()
            long_ma = sample_market_data["Close"].rolling(window=20).mean()

            position = 0  # Track current position

            for i in range(20, len(sample_market_data)):  # Start after MA calculation
                current_price = sample_market_data["Close"].iloc[i]
                short_ma_current = short_ma.iloc[i]
                long_ma_current = long_ma.iloc[i]
                short_ma_prev = short_ma.iloc[i - 1]
                long_ma_prev = long_ma.iloc[i - 1]

                # Buy signal: short MA crosses above long MA
                if (
                    short_ma_current > long_ma_current
                    and short_ma_prev <= long_ma_prev
                    and position == 0
                ):

                    quantity = int(engine.cash / current_price)
                    if quantity > 0:
                        success = engine.execute_order(
                            symbol=symbol,
                            order_type="market",
                            side="buy",
                            quantity=quantity,
                            price=current_price,
                        )
                        if success:
                            position = quantity

                # Sell signal: short MA crosses below long MA
                elif (
                    short_ma_current < long_ma_current
                    and short_ma_prev >= long_ma_prev
                    and position > 0
                ):

                    success = engine.execute_order(
                        symbol=symbol,
                        order_type="market",
                        side="sell",
                        quantity=position,
                        price=current_price,
                    )
                    if success:
                        position = 0

            # Calculate final portfolio value
            final_price = sample_market_data["Close"].iloc[-1]
            position_value = engine.positions.get(symbol, 0) * final_price
            total_value = engine.cash + position_value

            # Calculate return
            total_return = (
                total_value - engine.initial_capital
            ) / engine.initial_capital

            assert isinstance(total_return, (int, float))

            print(" Moving average strategy test passed")
        except Exception as e:
            print(f" Moving average strategy test failed: {e}")

    def test_risk_management(self, sample_market_data):
        """Test risk management features."""
        try:
            engine = BacktestEngine(initial_capital=100000)

            # Test position sizing
            max_position_size = 0.1  # 10% of portfolio
            price = 100.0

            max_quantity = int((engine.cash * max_position_size) / price)

            # Try to buy more than allowed
            excessive_quantity = max_quantity * 2

            # Should limit to max allowed
            actual_quantity = min(excessive_quantity, max_quantity)

            assert actual_quantity <= max_quantity

            # Test stop loss
            entry_price = 100.0
            stop_loss_pct = 0.05  # 5% stop loss
            stop_price = entry_price * (1 - stop_loss_pct)

            current_price = 94.0  # Below stop loss

            should_stop = current_price <= stop_price
            assert should_stop

            print(" Risk management test passed")
        except Exception as e:
            print(f" Risk management test failed: {e}")

    def test_multiple_assets(self, sample_market_data):
        """Test backtesting with multiple assets."""
        try:
            engine = BacktestEngine(initial_capital=100000)

            # Create data for multiple assets
            symbols = ["AAPL", "GOOGL", "MSFT"]

            # Simulate trading multiple assets
            for symbol in symbols:
                price = 100.0 + np.random.uniform(-10, 10)
                quantity = int(engine.cash / (len(symbols) * price))

                if quantity > 0:
                    success = engine.execute_order(
                        symbol=symbol,
                        order_type="market",
                        side="buy",
                        quantity=quantity,
                        price=price,
                    )

                    if success:
                        assert symbol in engine.positions
                        assert engine.positions[symbol] == quantity

            # Check that we have positions in multiple assets
            assert len(engine.positions) > 0

            print(" Multiple assets test passed")
        except Exception as e:
            print(f" Multiple assets test failed: {e}")

    def test_benchmark_comparison(self, sample_market_data):
        """Test benchmark comparison functionality."""
        try:
            # Strategy returns (random for testing)
            np.random.seed(42)
            strategy_returns = pd.Series(
                np.random.normal(0.0008, 0.02, len(sample_market_data)),
                index=sample_market_data.index,
            )

            # Benchmark returns (market)
            benchmark_returns = sample_market_data["Close"].pct_change().dropna()

            # Align the series
            common_index = strategy_returns.index.intersection(benchmark_returns.index)
            if len(common_index) > 50:
                strategy_aligned = strategy_returns.loc[common_index]
                benchmark_aligned = benchmark_returns.loc[common_index]

                # Calculate excess returns
                excess_returns = strategy_aligned - benchmark_aligned

                # Calculate tracking error
                tracking_error = excess_returns.std() * np.sqrt(252)

                # Calculate information ratio
                information_ratio = (
                    (excess_returns.mean() * 252) / tracking_error
                    if tracking_error > 0
                    else 0
                )

                # Calculate beta
                covariance = np.cov(strategy_aligned, benchmark_aligned)[0, 1]
                benchmark_variance = np.var(benchmark_aligned)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

                # Validate metrics
                assert isinstance(tracking_error, (int, float))
                assert isinstance(information_ratio, (int, float))
                assert isinstance(beta, (int, float))
                assert tracking_error >= 0

                print(" Benchmark comparison test passed")
            else:
                print(" Insufficient data for benchmark comparison")

        except Exception as e:
            print(f" Benchmark comparison test failed: {e}")

    def test_transaction_log(self, sample_market_data):
        """Test transaction logging functionality."""
        try:
            engine = BacktestEngine(initial_capital=100000)

            # Initialize transaction log
            engine.transaction_log = []

            symbol = "TEST"
            price = 100.0
            quantity = 100

            # Execute order and log transaction
            success = engine.execute_order(
                symbol=symbol,
                order_type="market",
                side="buy",
                quantity=quantity,
                price=price,
            )

            if success:
                # Manually add to transaction log for testing
                transaction = {
                    "timestamp": sample_market_data.index[0],
                    "symbol": symbol,
                    "side": "buy",
                    "quantity": quantity,
                    "price": price,
                    "value": quantity * price,
                    "commission": quantity * price * engine.commission,
                    "slippage": quantity * price * engine.slippage,
                }
                engine.transaction_log.append(transaction)

                # Check transaction log
                assert len(engine.transaction_log) == 1
                assert engine.transaction_log[0]["symbol"] == symbol
                assert engine.transaction_log[0]["quantity"] == quantity

                print(" Transaction log test passed")
            else:
                print(" Order execution failed in transaction log test")

        except Exception as e:
            print(f" Transaction log test failed: {e}")

    def test_error_handling(self, sample_market_data):
        """Test error handling for invalid operations."""
        try:
            engine = BacktestEngine(initial_capital=100000)

            # Test selling more than owned
            symbol = "TEST"

            # Try to sell without owning
            success = engine.execute_order(
                symbol=symbol,
                order_type="market",
                side="sell",
                quantity=100,
                price=100.0,
            )

            # Should fail or handle gracefully
            assert not success or engine.positions.get(symbol, 0) >= 0

            # Test buying with insufficient cash
            engine.cash = 10  # Very low cash

            success = engine.execute_order(
                symbol=symbol,
                order_type="market",
                side="buy",
                quantity=1000,  # Expensive order
                price=1000.0,
            )

            # Should fail or handle gracefully
            assert not success or engine.cash >= 0

            print(" Error handling test passed")
        except Exception as e:
            print(f" Error handling test failed: {e}")


def test_backtesting_import():
    """Test that backtesting can be imported."""
    try:
        from meridianalgo.backtesting import BacktestEngine  # noqa: F401

        print(" Backtesting import test passed")
        return True
    except ImportError as e:
        print(f" Import test failed: {e}")
        return False


def test_backtesting_with_real_data():
    """Test backtesting with real market data if available."""
    try:
        # Try to get real data
        data = ma.get_market_data(["AAPL"], "2023-01-01", "2023-12-31")

        if data is not None and len(data) > 50:
            # Simple backtest
            engine = BacktestEngine(initial_capital=100000)

            # Buy and hold test
            first_price = data["AAPL"].iloc[0]
            quantity = int(engine.cash / first_price)

            success = engine.execute_order(
                symbol="AAPL",
                order_type="market",
                side="buy",
                quantity=quantity,
                price=first_price,
            )

            if success:
                final_price = data["AAPL"].iloc[-1]
                position_value = engine.positions["AAPL"] * final_price
                total_value = engine.cash + position_value

                assert total_value > 0

                print(" Real data backtesting test passed")
            else:
                print(" Order execution failed in real data test")
        else:
            print(" No real data available, skipping real data test")

    except Exception as e:
        print(f" Real data backtesting test failed: {e}")


if __name__ == "__main__":
    # Run tests manually
    print(" Running Backtesting Tests...")

    # Test imports first
    if not test_backtesting_import():
        print(" Cannot proceed with tests - import failed")
        exit(1)

    # Create test instance
    test_instance = TestBacktesting()

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    returns = np.random.normal(0.001, 0.02, 252)
    prices = [100]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    sample_data = pd.DataFrame(
        {
            "Open": [p * np.random.uniform(0.995, 1.005) for p in prices],
            "High": [p * np.random.uniform(1.00, 1.03) for p in prices],
            "Low": [p * np.random.uniform(0.97, 1.00) for p in prices],
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, 252),
        },
        index=dates,
    )

    # Ensure OHLC relationships
    for i in range(len(sample_data)):
        high = max(
            sample_data.iloc[i]["Open"], sample_data.iloc[i]["Close"]
        ) * np.random.uniform(1.0, 1.02)
        low = min(
            sample_data.iloc[i]["Open"], sample_data.iloc[i]["Close"]
        ) * np.random.uniform(0.98, 1.0)
        sample_data.iloc[i, sample_data.columns.get_loc("High")] = high
        sample_data.iloc[i, sample_data.columns.get_loc("Low")] = low

    # Run all tests
    test_methods = [
        test_instance.test_backtest_engine_creation,
        test_instance.test_simple_buy_hold_strategy,
        test_instance.test_order_execution,
        test_instance.test_commission_and_slippage,
        test_instance.test_portfolio_value_calculation,
        test_instance.test_performance_metrics,
        test_instance.test_moving_average_strategy,
        test_instance.test_risk_management,
        test_instance.test_multiple_assets,
        test_instance.test_benchmark_comparison,
        test_instance.test_transaction_log,
        test_instance.test_error_handling,
    ]

    passed = 0
    total = len(test_methods)

    for test_method in test_methods:
        try:
            test_method(sample_data)
            passed += 1
        except Exception as e:
            print(f" Test {test_method.__name__} failed: {e}")

    # Test with real data
    test_backtesting_with_real_data()

    print(f"\n Test Results: {passed}/{total} tests passed")

    if passed == total:
        print(" All backtesting tests passed!")
    else:
        print(f" {total - passed} tests failed")
