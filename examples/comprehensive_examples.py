"""
MeridianAlgo v6.2.1 - Comprehensive Examples

Demonstrates all major features of the package including:
- Portfolio Analytics (Pyfolio-style)
- Liquidity Analysis
- Technical Signals
- Risk Management
- Derivatives Pricing
- Factor Models
"""

import numpy as np
import pandas as pd
import meridianalgo as ma
from meridianalgo.analytics import PerformanceAnalyzer, RiskAnalyzer, DrawdownAnalyzer
from meridianalgo.liquidity import VPIN, MarketImpact, OrderBook
from meridianalgo.signals import (
    ATR,
    MACD,
    RSI,
    BollingerBands,
    SignalGenerator,
    TechnicalAnalyzer,
)

print("=" * 70)
print("MeridianAlgo v6.2.1 - Comprehensive Examples")
print("=" * 70)


# =============================================================================
# 1. QUICK START: Get data and analyze
# =============================================================================

print("\n" + "=" * 70)
print("1. QUICK START")
print("=" * 70)

# Get system info
info = ma.get_system_info()
print(f"\nPackage Version: {info['package_version']}")
print(f"Python Version: {info['python_version'].split()[0]}")

# Quick analysis of simulated returns
np.random.seed(42)
n_days = 500
returns = pd.Series(
    np.random.normal(0.0005, 0.015, n_days),
    index=pd.date_range(start="2023-01-01", periods=n_days, freq="B"),
    name="Strategy",
)

analysis = ma.quick_analysis(returns)
print("\nQuick Analysis Results:")
print(f"  Total Return: {analysis['total_return']:.2%}")
print(f"  Annualized Return: {analysis['annualized_return']:.2%}")
print(f"  Annualized Volatility: {analysis['annualized_volatility']:.2%}")
print(f"  Sharpe Ratio: {analysis['sharpe_ratio']:.2f}")
print(f"  Max Drawdown: {analysis['max_drawdown']:.2%}")
print(f"  Win Rate: {analysis['win_rate']:.1%}")


# =============================================================================
# 2. PORTFOLIO ANALYTICS
# =============================================================================

print("\n" + "=" * 70)
print("2. PORTFOLIO ANALYTICS (Pyfolio-Style)")
print("=" * 70)

# Create benchmark returns
benchmark = pd.Series(
    np.random.normal(0.0003, 0.012, n_days), index=returns.index, name="Benchmark"
)

# Performance analysis
perf = PerformanceAnalyzer(returns, benchmark=benchmark, risk_free_rate=0.05)

print("\nPerformance Metrics:")
print(f"  Annualized Return: {perf.annualized_return():.2%}")
print(f"  Annualized Volatility: {perf.annualized_volatility():.2%}")
print(f"  Sharpe Ratio: {perf.sharpe_ratio():.2f}")
print(f"  Sortino Ratio: {perf.sortino_ratio():.2f}")
print(f"  Calmar Ratio: {perf.calmar_ratio():.2f}")

print("\nBenchmark-Relative Metrics:")
print(f"  Alpha: {perf.alpha():.2%}")
print(f"  Beta: {perf.beta():.2f}")
print(f"  Information Ratio: {perf.information_ratio():.2f}")
print(f"  Tracking Error: {perf.tracking_error():.2%}")

# Risk analysis
risk = RiskAnalyzer(returns)

print("\nRisk Metrics:")
print(f"  VaR (95%, Historical): {risk.value_at_risk(0.95):.2%}")
print(f"  VaR (95%, Parametric): {risk.value_at_risk(0.95, 'parametric'):.2%}")
print(f"  CVaR (95%): {risk.conditional_var(0.95):.2%}")
print(f"  Max Drawdown: {risk.max_drawdown():.2%}")
print(f"  Ulcer Index: {risk.ulcer_index():.4f}")


# =============================================================================
# 3. DRAWDOWN ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("3. DRAWDOWN ANALYSIS")
print("=" * 70)

dd = DrawdownAnalyzer(returns)

print("\nDrawdown Metrics:")
print(f"  Maximum Drawdown: {dd.max_drawdown():.2%}")
print(f"  Current Drawdown: {dd.current_drawdown():.2%}")
print(f"  Average Drawdown: {dd.average_drawdown():.2%}")
print(f"  Time Underwater: {dd.time_underwater():.1%}")
print(f"  Max Duration: {dd.max_drawdown_duration()} days")
print(f"  Calmar Ratio: {dd.calmar_ratio():.2f}")

print("\nTop 3 Drawdowns:")
top_dd = dd.top_drawdowns(3)
for _, row in top_dd.iterrows():
    print(
        f"  #{int(row['Rank'])}: {row['Depth']:.2%} depth, {row['Duration (days)']} days"
    )


# =============================================================================
# 4. LIQUIDITY ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("4. LIQUIDITY ANALYSIS")
print("=" * 70)

# Create sample order book
ob = OrderBook()
ob.bids = [(100.00, 500), (99.95, 1000), (99.90, 1500), (99.85, 2000)]
ob.asks = [(100.05, 600), (100.10, 900), (100.15, 1200), (100.20, 1800)]

print("\nOrder Book Analysis:")
print(f"  Best Bid: ${ob.best_bid[0]:.2f} x {ob.best_bid[1]}")
print(f"  Best Ask: ${ob.best_ask[0]:.2f} x {ob.best_ask[1]}")
print(f"  Mid Price: ${ob.mid_price:.2f}")
print(f"  Microprice: ${ob.microprice:.4f}")
print(f"  Spread: ${ob.spread:.2f} ({ob.spread_bps:.1f} bps)")

depth = ob.depth(5)
print(f"  Depth Imbalance: {depth['depth_imbalance']:.2f}")

# Estimate impact
impact_5k = ob.price_impact(5000, "buy")
print(f"  Price Impact (5,000 shares): {impact_5k:.1f} bps")

# Market impact modeling
mi = MarketImpact(daily_volume=1000000, volatility=0.02, spread_bps=5.0)

costs = mi.estimate_total_cost(order_size=10000, price=100)
print("\nMarket Impact Estimation (10,000 shares):")
print(f"  Impact Cost: {costs['impact_cost_bps']:.2f} bps")
print(f"  Spread Cost: {costs['spread_cost_bps']:.2f} bps")
print(f"  Total Cost: ${costs['total_cost_dollars']:.2f}")

# VPIN calculation with simulated trades
trades = pd.DataFrame(
    {
        "price": 100 + np.cumsum(np.random.randn(1000) * 0.01),
        "size": np.random.randint(10, 100, 1000),
        "side": np.random.choice(["buy", "sell"], 1000),
    }
)

vpin_calc = VPIN(trades)
print("\nVPIN Analysis:")
print(f"  Current VPIN: {vpin_calc.current_vpin():.4f}")
print(f"  Average VPIN: {vpin_calc.average_vpin():.4f}")
print(f"  VPIN Percentile: {vpin_calc.vpin_percentile():.1f}%")
print(f"  Toxicity Regime: {vpin_calc.toxicity_regime()}")


# =============================================================================
# 5. TECHNICAL SIGNALS
# =============================================================================

print("\n" + "=" * 70)
print("5. TECHNICAL SIGNALS")
print("=" * 70)

# Generate sample OHLCV data
n = 200
dates = pd.date_range(start="2024-01-01", periods=n, freq="B")
close = pd.Series(100 + np.cumsum(np.random.randn(n) * 1), index=dates)
high = close + np.random.uniform(0.5, 2, n)
low = close - np.random.uniform(0.5, 2, n)
volume = pd.Series(np.random.randint(10000, 100000, n), index=dates)

# Calculate individual indicators
rsi = RSI(close, 14)
macd_line, signal_line, histogram = MACD(close)
bb_upper, bb_middle, bb_lower = BollingerBands(close)
atr = ATR(high, low, close, 14)

print("\nCurrent Indicator Values:")
print(f"  RSI(14): {rsi.iloc[-1]:.2f}")
print(f"  MACD: {macd_line.iloc[-1]:.4f}")
print(f"  MACD Signal: {signal_line.iloc[-1]:.4f}")
print(f"  MACD Histogram: {histogram.iloc[-1]:.4f}")
print(f"  ATR(14): ${atr.iloc[-1]:.2f}")

# Full technical analysis
tech = TechnicalAnalyzer(high, low, close, volume)
indicators = tech.calculate_all()
summary = tech.summary()

print("\nTechnical Analysis Summary:")
print(f"  Trend: {summary['trend'].upper()}")
print(f"  Momentum: {summary['momentum'].upper()}")
print(f"  Volatility: {summary['volatility'].upper()}")
print(
    f"  RSI: {summary['rsi']:.2f} ({'OVERSOLD' if summary['oversold'] else 'OVERBOUGHT' if summary['overbought'] else 'NEUTRAL'})"
)
print(
    f"  Combined Signal: {int(summary['combined_signal'])} ({'+' if summary['combined_signal'] > 0 else '-' if summary['combined_signal'] < 0 else '='} bias)"
)


# =============================================================================
# 6. SIGNAL GENERATION & BACKTESTING
# =============================================================================

print("\n" + "=" * 70)
print("6. SIGNAL GENERATION & BACKTESTING")
print("=" * 70)

# Create signal generator
data = pd.DataFrame(
    {
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
        "rsi": rsi,
        "macd": macd_line,
        "signal": signal_line,
    }
)

gen = SignalGenerator(data)

# Add trading rules
gen.add_rule("rsi_oversold", lambda d: d["rsi"] < 30, weight=1.5, signal_type="long")
gen.add_rule("rsi_overbought", lambda d: d["rsi"] > 70, weight=1.5, signal_type="short")
gen.add_rule(
    "macd_bullish", lambda d: d["macd"] > d["signal"], weight=1.0, signal_type="long"
)
gen.add_rule(
    "macd_bearish", lambda d: d["macd"] < d["signal"], weight=1.0, signal_type="short"
)

# Generate signals
signals = gen.generate(threshold=0.4)

print("\nSignal Distribution:")
print(f"  Long Signals: {(signals['signal'] == 1).sum()}")
print(f"  Short Signals: {(signals['signal'] == -1).sum()}")
print(f"  Neutral: {(signals['signal'] == 0).sum()}")

# Backtest
returns_bt = close.pct_change().dropna()
signals_aligned = signals.reindex(returns_bt.index)

backtest_results = gen.backtest_signals(
    signals_aligned, returns_bt, transaction_cost=0.001
)

print("\nBacktest Results:")
print(f"  Total Return: {backtest_results['total_return']:.2%}")
print(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
print(f"  Max Drawdown: {backtest_results['max_drawdown']:.2%}")
print(f"  Win Rate: {backtest_results['win_rate']:.1%}")
print(f"  Number of Trades: {backtest_results['num_trades']:.0f}")


# =============================================================================
# 7. DERIVATIVES PRICING (from existing module)
# =============================================================================

print("\n" + "=" * 70)
print("7. DERIVATIVES PRICING")
print("=" * 70)

try:
    from meridianalgo.derivatives import OptionsPricer

    pricer = OptionsPricer()

    # Option parameters
    S = 100  # Stock price
    K = 105  # Strike
    T = 0.5  # Time to expiration (6 months)
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility

    # Black-Scholes pricing
    call_price = pricer.black_scholes_merton(S, K, T, r, sigma, "call")
    put_price = pricer.black_scholes_merton(S, K, T, r, sigma, "put")

    print("\nOptions Pricing (Black-Scholes):")
    print(f"  Stock Price: ${S}")
    print(f"  Strike: ${K}")
    print(f"  Time to Expiry: {T} years")
    print(f"  Volatility: {sigma:.0%}")
    print(f"  Call Price: ${call_price:.2f}")
    print(f"  Put Price: ${put_price:.2f}")

    # Greeks
    greeks = pricer.calculate_greeks(S, K, T, r, sigma, "call")

    print("\nCall Option Greeks:")
    print(f"  Delta: {greeks['delta']:.4f}")
    print(f"  Gamma: {greeks['gamma']:.4f}")
    print(f"  Theta: {greeks['theta']:.4f}")
    print(f"  Vega: {greeks['vega']:.4f}")
    print(f"  Rho: {greeks['rho']:.4f}")

except ImportError as e:
    print(f"\nDerivatives module import note: {e}")


# =============================================================================
# 8. QUANTITATIVE STRATEGIES (from existing module)
# =============================================================================

print("\n" + "=" * 70)
print("8. QUANTITATIVE STRATEGIES")
print("=" * 70)

try:
    from meridianalgo.quant import (
        CointegrationAnalyzer,
        OrnsteinUhlenbeck,
        PairsTrading,
    )

    # Create cointegrated pairs
    np.random.seed(123)
    n = 300
    common_trend = np.cumsum(np.random.randn(n) * 0.5)

    stock1 = pd.Series(100 + common_trend + np.random.randn(n) * 2, name="STOCK1")
    stock2 = pd.Series(
        50 + common_trend * 0.6 + np.random.randn(n) * 1.5, name="STOCK2"
    )

    # Cointegration test
    coint = CointegrationAnalyzer()
    result = coint.engle_granger_test(stock1, stock2)

    print("\nCointegration Analysis:")
    print(f"  Test Statistic: {result['test_statistic']:.4f}")
    print(f"  P-value: {result['pvalue']:.4f}")
    print(f"  Cointegrated: {'Yes' if result['is_cointegrated'] else 'No'}")

    # Pairs trading
    pt = PairsTrading(entry_threshold=2.0, exit_threshold=0.5)
    hedge_ratio = pt.calculate_hedge_ratio(stock1, stock2)
    signals = pt.generate_signals(stock1, stock2, window=20)

    print("\nPairs Trading Strategy:")
    print(f"  Hedge Ratio: {hedge_ratio:.4f}")
    print(f"  Current Z-Score: {signals['zscore'].iloc[-1]:.2f}")
    print(f"  Current Position: {signals['signal'].iloc[-1]:.0f}")

    # Ornstein-Uhlenbeck process
    spread = stock1 - hedge_ratio * stock2
    ou = OrnsteinUhlenbeck()
    params = ou.fit(spread)

    print("\nMean Reversion Analysis (OU Process):")
    print(f"  Speed (): {params['theta']:.4f}")
    print(f"  Long-term Mean (): {params['mu']:.2f}")
    print(f"  Volatility (): {params['sigma']:.2f}")
    print(f"  Half-life: {params['half_life']:.1f} periods")

except ImportError as e:
    print(f"\nQuant module import note: {e}")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("EXAMPLES COMPLETED!")
print("=" * 70)

print(
    """
MeridianAlgo v6.2.1 provides:

 Portfolio Analytics
   - 50+ performance metrics (Sharpe, Sortino, Calmar, etc.)
   - Benchmark-relative analysis (alpha, beta, IR)
   - Pyfolio-style tear sheets

 Liquidity Analysis  
   - Order book analysis (microprice, depth, imbalance)
   - VPIN (Volume-Synchronized PIN)
   - Market impact models (Almgren-Chriss)
   - Spread decomposition

 Risk Management
   - VaR/CVaR (Historical, Parametric, Cornish-Fisher)
   - GARCH volatility
   - Stress testing
   - Drawdown analysis

 Technical Analysis
   - 50+ technical indicators
   - Signal generation framework
   - Built-in backtesting

 Derivatives
   - Black-Scholes pricing
   - Binomial & Monte Carlo
   - Greeks calculation
   - Implied volatility

 Quantitative Strategies
   - Pairs trading
   - Cointegration testing
   - Mean reversion (OU process)
   - Factor models
   - Regime detection

For more information, visit: https://github.com/MeridianAlgo/Python-Packages
"""
)
