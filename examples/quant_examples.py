"""
MeridianAlgo v5.0.0 - Professional Quant Module Examples

This file demonstrates the new quantitative finance algorithms including:
- Market microstructure analysis
- Statistical arbitrage
- Optimal execution
- High-frequency trading
- Factor models
- Regime detection
"""

import numpy as np
import pandas as pd

from meridianalgo.quant.execution_algorithms import TWAP, VWAP, ImplementationShortfall
from meridianalgo.quant.factor_models import FactorRiskDecomposition, FamaFrenchModel
from meridianalgo.quant.high_frequency import HFTSignalGenerator, MarketMaking

# Import MeridianAlgo quant modules
from meridianalgo.quant.market_microstructure import (
    MarketImpactModel,
    OrderFlowImbalance,
    RealizedVolatility,
)
from meridianalgo.quant.regime_detection import HiddenMarkovModel, MarketStateClassifier
from meridianalgo.quant.statistical_arbitrage import (
    CointegrationAnalyzer,
    OrnsteinUhlenbeck,
    PairsTrading,
)


def example_market_microstructure():
    """Example: Market microstructure analysis."""
    print("=" * 80)
    print("MARKET MICROSTRUCTURE ANALYSIS")
    print("=" * 80)

    # Generate sample order book data
    np.random.seed(42)
    n_obs = 100

    bid_volumes = np.random.randint(100, 1000, n_obs)
    ask_volumes = np.random.randint(100, 1000, n_obs)
    bid_prices = 100 + np.cumsum(np.random.randn(n_obs) * 0.01)
    ask_prices = bid_prices + 0.05

    # Order Flow Imbalance
    print("\n1. Order Flow Imbalance (OFI)")
    ofi = OrderFlowImbalance()
    imbalance = ofi.calculate_ofi(bid_volumes, ask_volumes, bid_prices, ask_prices)
    print(f"   Current OFI: {imbalance[-1]:.2f}")
    print(f"   Average OFI: {np.mean(imbalance):.2f}")

    # Volume Imbalance Ratio
    vir = ofi.volume_imbalance_ratio(bid_volumes, ask_volumes)
    print("\n2. Volume Imbalance Ratio")
    print(f"   Current VIR: {vir[-1]:.4f}")
    print(f"   Interpretation: {'Buy pressure' if vir[-1] > 0 else 'Sell pressure'}")

    # Market Impact
    print("\n3. Market Impact Estimation")
    impact_model = MarketImpactModel()
    impact = impact_model.square_root_law(
        order_size=10000, daily_volume=500000, sigma=0.02
    )
    print(f"   Expected market impact: {impact:.4f} ({impact*10000:.2f} bps)")

    # Realized Volatility
    print("\n4. Realized Volatility")
    dates = pd.date_range("2024-01-01", periods=n_obs, freq="5min")
    prices = pd.Series(100 + np.cumsum(np.random.randn(n_obs) * 0.1), index=dates)

    rv = RealizedVolatility.rv_5min(prices)
    print(f"   5-minute Realized Volatility: {rv:.2%} (annualized)")


def example_statistical_arbitrage():
    """Example: Statistical arbitrage strategy."""
    print("\n" + "=" * 80)
    print("STATISTICAL ARBITRAGE")
    print("=" * 80)

    # Generate cointegrated stock prices
    np.random.seed(42)
    t = np.arange(200)
    common_trend = t * 0.1

    stock1 = pd.Series(100 + common_trend + np.random.randn(200) * 2, name="STOCK1")
    stock2 = pd.Series(
        50 + common_trend * 0.5 + np.random.randn(200) * 1.5, name="STOCK2"
    )

    # Test for cointegration
    print("\n1. Cointegration Test")
    analyzer = CointegrationAnalyzer()
    coint_result = analyzer.engle_granger_test(stock1, stock2)

    print(f"   Test Statistic: {coint_result['test_statistic']:.4f}")
    print(f"   P-value: {coint_result['pvalue']:.4f}")
    print(f"   Cointegrated: {coint_result['is_cointegrated']}")

    # Pairs Trading Strategy
    print("\n2. Pairs Trading Strategy")
    pt = PairsTrading(entry_threshold=2.0, exit_threshold=0.5, stop_loss=4.0)

    hedge_ratio = pt.calculate_hedge_ratio(stock1, stock2, method="ols")
    print(f"   Hedge Ratio (): {hedge_ratio:.4f}")

    # Generate signals
    signals = pt.generate_signals(stock1, stock2, window=20)

    current_signal = signals["signal"].iloc[-1]
    current_zscore = signals["zscore"].iloc[-1]
    print(f"   Current Z-score: {current_zscore:.2f}")
    print(
        f"   Current Signal: {current_signal} ({'Long spread' if current_signal > 0 else 'Short spread' if current_signal < 0 else 'Neutral'})"
    )

    # Calculate strategy returns
    strategy_returns = signals["strategy_ret"].dropna()
    total_return = (1 + strategy_returns).prod() - 1
    sharpe = (
        strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        if strategy_returns.std() > 0
        else 0
    )

    print("\n3. Strategy Performance")
    print(f"   Total Return: {total_return:.2%}")
    print(f"   Sharpe Ratio: {sharpe:.2f}")
    print(f"   Number of Trades: {signals['signal'].diff().abs().sum() / 2:.0f}")

    # Ornstein-Uhlenbeck Process
    print("\n4. Mean Reversion Analysis (OU Process)")
    spread = stock1 - hedge_ratio * stock2
    ou = OrnsteinUhlenbeck()
    params = ou.fit(spread)

    print(f"   Speed of Mean Reversion (): {params['theta']:.4f}")
    print(f"   Long-term Mean (): {params['mu']:.2f}")
    print(f"   Volatility (): {params['sigma']:.2f}")
    print(f"   Half-life: {params['half_life']:.1f} periods")


def example_optimal_execution():
    """Example: Optimal execution algorithms."""
    print("\n" + "=" * 80)
    print("OPTIMAL EXECUTION ALGORITHMS")
    print("=" * 80)

    # VWAP Execution
    print("\n1. VWAP (Volume-Weighted Average Price)")
    vwap = VWAP(
        total_quantity=10000,
        start_time="2024-01-01 09:30:00",
        end_time="2024-01-01 16:00:00",
    )

    # Historical volume profile
    dates = pd.date_range("2024-01-01 09:30", "2024-01-01 16:00", freq="5min")
    historical_volume = pd.Series(np.random.randint(500, 2000, len(dates)), index=dates)

    schedule = vwap.calculate_schedule(historical_volume)
    print(f"   Total Quantity: {vwap.total_quantity:,} shares")
    print(f"   Number of Slices: {len(schedule)}")
    print(f"   First 5 slices: {schedule.head().values}")

    # TWAP Execution
    print("\n2. TWAP (Time-Weighted Average Price)")
    twap = TWAP(total_quantity=10000, duration_minutes=60, slice_interval_minutes=5)
    print(f"   Total Quantity: {twap.total_quantity:,} shares")
    print(f"   Duration: {twap.duration_minutes} minutes")
    print(f"   Number of Slices: {twap.n_slices}")
    print(f"   Quantity per Slice: {twap.quantity_per_slice:.0f} shares")

    # Implementation Shortfall (Almgren-Chriss)
    print("\n3. Implementation Shortfall (Almgren-Chriss)")
    is_algo = ImplementationShortfall(
        total_quantity=50000,
        total_time=1.0,
        volatility=0.02,
        risk_aversion=1e-6,
        permanent_impact=0.1,
        temporary_impact=0.01,
    )

    trajectory = is_algo.calculate_optimal_trajectory(n_intervals=10)
    cost_analysis = is_algo.calculate_expected_cost()

    print(f"   Total Quantity: {is_algo.Q:,} shares")
    print(f"   Expected Total Cost: ${cost_analysis['total_cost']:,.2f}")
    print(f"   Cost per Share: ${cost_analysis['cost_per_share']:.4f}")
    print(f"   Timing Risk:  ${cost_analysis['timing_risk']:,.2f}")
    print(
        f"   Market Impact: ${cost_analysis['permanent_impact'] + cost_analysis['temporary_impact']:,.2f}"
    )

    print("\n   Optimal Execution Schedule (first 5 intervals):")
    for i in range(min(5, len(trajectory))):
        row = trajectory.iloc[i]
        print(
            f"   Interval {i}: Trade {row['trades']:.0f} shares, Holdings {row['holdings']:.0f}"
        )


def example_high_frequency_trading():
    """Example: High-frequency trading strategies."""
    print("\n" + "=" * 80)
    print("HIGH-FREQUENCY TRADING")
    print("=" * 80)

    # Market Making (Avellaneda-Stoikov)
    print("\n1. Market Making Strategy (Avellaneda-Stoikov)")
    mm = MarketMaking(
        target_spread_bps=5.0,
        max_inventory=1000,
        inventory_penalty=0.01,
        tick_size=0.01,
    )

    mid_price = 100.0
    volatility = 0.02

    # Calculate optimal quotes
    bid_price, ask_price = mm.calculate_quotes(
        mid_price=mid_price, volatility=volatility, order_flow_imbalance=0.1
    )

    bid_size, ask_size = mm.calculate_quote_sizes(base_size=100)

    print(f"   Mid Price: ${mid_price:.2f}")
    print(f"   Optimal Bid: ${bid_price:.2f} (size: {bid_size})")
    print(f"   Optimal Ask: ${ask_price:.2f} (size: {ask_size})")
    print(
        f"   Spread: ${ask_price - bid_price:.2f} ({(ask_price - bid_price)/mid_price * 10000:.1f} bps)"
    )
    print(f"   Current Inventory: {mm.position}")

    # Simulate some fills
    print("\n   Simulating trades...")
    mm.on_fill("buy", 100.00, 50)
    mm.on_fill("sell", 100.05, 30)

    pnl = mm.calculate_pnl(current_price=100.02)
    print(f"   Position after trades: {mm.position}")
    print(f"   Cash: ${mm.cash:.2f}")
    print(f"   Total P&L: ${pnl['total_pnl']:.2f}")

    # Order Flow Toxicity
    print("\n2. Order Flow Toxicity (Informed Trading Detection)")

    # Generate sample trade data
    np.random.seed(42)
    n_trades = 200
    trade_prices = 100 + np.cumsum(np.random.randn(n_trades) * 0.05)
    trade_volumes = np.random.randint(10, 100, n_trades)
    trade_sides = np.random.choice([-1, 1], n_trades)  # -1: sell, 1: buy

    toxicity = HFTSignalGenerator.order_flow_toxicity(
        trade_prices, trade_volumes, trade_sides, window=50
    )

    current_toxicity = toxicity[-1]
    print(f"   Current Toxicity Level: {current_toxicity:.4f}")
    print(f"   Average Toxicity: {np.mean(toxicity[toxicity > 0]):.4f}")
    print(
        f"   Interpretation: {'High informed trading' if current_toxicity > 0.5 else 'Low informed trading'}"
    )


def example_factor_models():
    """Example: Factor models for portfolio analysis."""
    print("\n" + "=" * 80)
    print("FACTOR MODELS")
    print("=" * 80)

    # Generate sample data
    np.random.seed(42)
    n_obs = 250

    returns = pd.Series(np.random.randn(n_obs) * 0.01 + 0.0003, name="Asset")

    factor_data = pd.DataFrame(
        {
            "MKT": np.random.randn(n_obs) * 0.015 + 0.0004,
            "SMB": np.random.randn(n_obs) * 0.01,
            "HML": np.random.randn(n_obs) * 0.01,
        }
    )

    # Fama-French Three-Factor Model
    print("\n1. Fama-French Three-Factor Model")
    ff = FamaFrenchModel(model_type="three_factor")
    results = ff.fit(returns, factor_data)

    print("\n   Regression Results:")
    print(f"   Alpha: {results['alpha']:.6f} ({results['alpha']*252:.4%} annualized)")
    print(f"   Alpha t-stat: {results['alpha_t_stat']:.2f}")
    print(f"   Significant Alpha: {results['significant_alpha']}")
    print(f"   R-squared: {results['r_squared']:.4f}")

    print("\n   Factor Exposures:")
    for factor, beta in results["coefficients"].items():
        t_stat = results["t_stats"][factor]
        significant = abs(t_stat) > 1.96
        print(
            f"   {factor:12s}:  = {beta:7.4f}  (t = {t_stat:6.2f}) {'***' if significant else ''}"
        )

    # Factor Risk Decomposition
    print("\n2. Factor Risk Decomposition")

    # Portfolio weights
    weights = np.array([0.2, 0.3, 0.15, 0.25, 0.1])

    # Factor exposures (assets x factors)
    factor_exposures = np.array(
        [
            [1.0, 0.5, 0.2],
            [0.9, 0.3, -0.1],
            [1.1, 0.1, 0.5],
            [0.8, 0.8, 0.3],
            [1.0, -0.2, 0.4],
        ]
    )

    # Factor covariance
    factor_covariance = np.array(
        [[0.04, 0.01, 0.005], [0.01, 0.02, 0.003], [0.005, 0.003, 0.015]]
    )

    # Specific variances
    specific_variance = np.array([0.01, 0.012, 0.015, 0.011, 0.013])

    decomp = FactorRiskDecomposition.decompose_variance(
        weights, factor_exposures, factor_covariance, specific_variance
    )

    print(f"   Total Risk: {decomp['total_volatility']:.2%}")
    print(
        f"   Factor Risk: {decomp['factor_risk_pct']:.1%} ({decomp['total_volatility'] * decomp['factor_risk_pct']:.2%})"
    )
    print(
        f"   Specific Risk: {decomp['specific_risk_pct']:.1%} ({decomp['total_volatility'] * decomp['specific_risk_pct']:.2%})"
    )

    print("\n   Individual Factor Contributions:")
    for factor, contrib in decomp["factor_contributions"].items():
        print(f"   {factor}: {contrib:.1%}")


def example_regime_detection():
    """Example: Market regime detection."""
    print("\n" + "=" * 80)
    print("REGIME DETECTION")
    print("=" * 80)

    # Generate returns with regime changes
    np.random.seed(42)

    # Low volatility regime
    returns1 = np.random.normal(0.001, 0.008, 125)

    # High volatility regime
    returns2 = np.random.normal(-0.002, 0.025, 125)

    returns = pd.Series(np.concatenate([returns1, returns2]))

    # Hidden Markov Model
    print("\n1. Hidden Markov Model (2-State)")
    hmm = HiddenMarkovModel(n_states=2)
    results = hmm.fit(returns, max_iter=50, tolerance=1e-4)

    print("\n   Regime Characteristics:")
    for regime, mean in results["means"].items():
        std = results["std_devs"][regime]
        print(f"   {regime}:")
        print(f"      Mean: {mean:.6f}")
        print(f"      Std Dev: {std:.6f}")
        print(f"      Annualized Return: {mean * 252:.2%}")
        print(f"      Annualized Vol: {std * np.sqrt(252):.2%}")

    print("\n   Transition Matrix:")
    print(results["transition_matrix"])

    # Current state
    states = hmm.predict_state(returns)
    current_state = states.iloc[-1]
    print(f"\n   Current Market Regime: State {current_state}")

    # Market State Classification
    print("\n2. Market State Classification")

    # Generate price series
    pd.Series(100 * (1 + returns).cumprod())

    vol_regime = MarketStateClassifier.classify_volatility_regime(returns, window=60)
    current_vol_regime = vol_regime.iloc[-1]

    print(f"   Current Volatility Regime: {current_vol_regime}")

    # Count regime occurrences
    regime_counts = vol_regime.value_counts()
    print("\n   Regime Distribution:")
    for regime, count in regime_counts.items():
        print(f"   {regime:20s}: {count:3d} periods ({count/len(vol_regime):.1%})")


def main():
    """Run all examples."""
    print("\n")
    print("" + "=" * 78 + "")
    print("" + " " * 78 + "")
    print(
        "" + "MeridianAlgo v5.0.0 - Professional Quant Module Examples".center(78) + ""
    )
    print("" + " " * 78 + "")
    print("" + "=" * 78 + "")

    try:
        example_market_microstructure()
    except Exception as e:
        print(f"\nError in market microstructure example: {e}")

    try:
        example_statistical_arbitrage()
    except Exception as e:
        print(f"\nError in statistical arbitrage example: {e}")

    try:
        example_optimal_execution()
    except Exception as e:
        print(f"\nError in optimal execution example: {e}")

    try:
        example_high_frequency_trading()
    except Exception as e:
        print(f"\nError in high-frequency trading example: {e}")

    try:
        example_factor_models()
    except Exception as e:
        print(f"\nError in factor models example: {e}")

    try:
        example_regime_detection()
    except Exception as e:
        print(f"\nError in regime detection example: {e}")

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 80)
    print("\nFor more information:")
    print("  - Documentation: docs/")
    print("  - Tests: tests/test_quant.py")
    print("  - Source: meridianalgo/quant/")
    print("\n")


if __name__ == "__main__":
    main()
