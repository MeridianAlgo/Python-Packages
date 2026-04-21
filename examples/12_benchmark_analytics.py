"""
Example 12: Benchmark Analytics and Performance Attribution

Demonstrates MeridianAlgo's benchmark analytics module:
- Tracking error, information ratio, up/down capture
- Active share from portfolio holdings
- Rolling information ratio and beta
- Brinson-Hood-Beebower sector attribution
- Manager quality assessment framework
"""

import numpy as np
import pandas as pd

from meridianalgo.analytics.benchmark import (
    ActiveShare,
    BenchmarkAnalytics,
    BrinsonAttribution,
)


def generate_manager(
    n: int,
    benchmark_vol: float = 0.01,
    benchmark_drift: float = 0.0003,
    alpha_daily: float = 0.0002,
    te_daily: float = 0.003,
    seed: int = 42,
) -> tuple:
    rng = np.random.default_rng(seed)
    bench = rng.standard_normal(n) * benchmark_vol + benchmark_drift
    active = rng.standard_normal(n) * te_daily + alpha_daily
    port = bench + active
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    return (
        pd.Series(port, index=dates, name="portfolio"),
        pd.Series(bench, index=dates, name="benchmark"),
    )


# ---------------------------------------------------------------------------
# 1. Core Benchmark-Relative Metrics
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. BENCHMARK-RELATIVE PERFORMANCE METRICS")
print("=" * 60)

port_ret, bench_ret = generate_manager(n=750, alpha_daily=0.0003, te_daily=0.004, seed=42)

analytics = BenchmarkAnalytics(
    portfolio_returns=port_ret,
    benchmark_returns=bench_ret,
    risk_free_rate=0.04,
    periods_per_year=252,
)

m = analytics.active_metrics()

print(f"Sample: {len(port_ret)} daily observations (~{len(port_ret)//252:.0f} years)")
print()
print(f"Active Return (ann.):    {m.active_return:.2%}")
print(f"Tracking Error (ann.):   {m.tracking_error:.2%}")
print(f"Information Ratio:       {m.information_ratio:.4f}")
print()
print(f"Up Capture Ratio:        {m.up_capture:.2%}")
print(f"Down Capture Ratio:      {m.down_capture:.2%}")
print(f"Capture Ratio (U/D):     {m.capture_ratio:.4f}")
print()
print(f"Batting Average:         {m.batting_average:.2%}")
print(f"Max Active Drawdown:     {m.max_active_drawdown:.2%}")
print()
print(f"Beta:                    {m.beta:.4f}")
print(f"Alpha (ann.):            {m.alpha_annualized:.4%}")
print(f"R-Squared:               {m.r_squared:.4f}")
print(f"Treynor Ratio:           {m.treynor_ratio:.4f}")

# IR interpretation
def interpret_ir(ir):
    if ir > 0.75: return "exceptional"
    if ir > 0.50: return "very good"
    if ir > 0.25: return "above average"
    if ir > 0.00: return "positive alpha, marginal skill"
    return "negative alpha"

print(f"\nIR Assessment: {interpret_ir(m.information_ratio)}")


# ---------------------------------------------------------------------------
# 2. Manager Quality Comparison
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("2. MANAGER COMPARISON")
print("=" * 60)

scenarios = [
    ("High IR (skilled)",    0.0004,  0.003, 42),
    ("Mid IR (average)",     0.0001,  0.003, 43),
    ("Negative alpha",      -0.0002,  0.002, 44),
    ("High TE, low alpha",   0.0001,  0.008, 45),
    ("Low TE, high alpha",   0.0003,  0.001, 46),
]

print(f"{'Manager':25}  {'Active Ret':>12}  {'TE':>8}  {'IR':>8}  {'Up Cap':>8}  {'Dn Cap':>8}  {'Batting':>8}")
for name, alpha, te, seed in scenarios:
    p, b = generate_manager(n=750, alpha_daily=alpha, te_daily=te, seed=seed)
    a = BenchmarkAnalytics(p, b, risk_free_rate=0.04)
    m = a.active_metrics()
    print(f"  {name:23}  {m.active_return:>12.2%}  {m.tracking_error:>8.2%}  "
          f"{m.information_ratio:>8.4f}  {m.up_capture:>8.2%}  {m.down_capture:>8.2%}  "
          f"{m.batting_average:>8.2%}")


# ---------------------------------------------------------------------------
# 3. Rolling Analytics
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("3. ROLLING ANALYTICS")
print("=" * 60)

port_long, bench_long = generate_manager(n=1260, alpha_daily=0.0002, te_daily=0.004, seed=42)
analytics_long = BenchmarkAnalytics(port_long, bench_long, risk_free_rate=0.04)

rir_252 = analytics_long.rolling_information_ratio(window=252)
rbeta_252 = analytics_long.rolling_beta(window=252)

rir_clean = rir_252.dropna()
rbeta_clean = rbeta_252.dropna()

print(f"Rolling 252-day Information Ratio (5yr history):")
print(f"  Mean:   {rir_clean.mean():.4f}")
print(f"  Std:    {rir_clean.std():.4f}")
print(f"  Min:    {rir_clean.min():.4f}")
print(f"  Max:    {rir_clean.max():.4f}")
print(f"  % > 0:  {(rir_clean > 0).mean():.1%}")

print(f"\nRolling 252-day Beta:")
print(f"  Mean:   {rbeta_clean.mean():.4f}")
print(f"  Std:    {rbeta_clean.std():.4f}")
print(f"  Min:    {rbeta_clean.min():.4f}")
print(f"  Max:    {rbeta_clean.max():.4f}")


# ---------------------------------------------------------------------------
# 4. Active Share from Holdings
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("4. ACTIVE SHARE FROM PORTFOLIO HOLDINGS")
print("=" * 60)

# S&P 500-like benchmark weights (top 10 positions)
benchmark_weights = pd.Series({
    "AAPL":  0.0750,
    "MSFT":  0.0650,
    "NVDA":  0.0580,
    "AMZN":  0.0420,
    "META":  0.0260,
    "GOOGL": 0.0250,
    "GOOG":  0.0240,
    "BRK_B": 0.0180,
    "LLY":   0.0170,
    "JPM":   0.0160,
    "OTHER": 0.6340,
})

# Three portfolio manager styles
managers = {
    "Closet Indexer": pd.Series({
        "AAPL": 0.0700, "MSFT": 0.0680, "NVDA": 0.0560, "AMZN": 0.0400,
        "META": 0.0280, "GOOGL": 0.0240, "GOOG": 0.0250, "BRK_B": 0.0200,
        "LLY": 0.0160, "JPM": 0.0180, "OTHER": 0.6350,
    }),
    "Moderate Active": pd.Series({
        "AAPL": 0.0950, "MSFT": 0.0800, "NVDA": 0.0400, "AMZN": 0.0200,
        "META": 0.0100, "GOOGL": 0.0100, "GOOG": 0.0100, "BRK_B": 0.0300,
        "LLY": 0.0400, "JPM": 0.0300, "TSLA": 0.0200, "OTHER": 0.5150,
    }),
    "Concentrated Active": pd.Series({
        "AAPL": 0.1200, "MSFT": 0.1100, "BRK_B": 0.1000, "JPM": 0.0900,
        "LLY": 0.0800, "WMT": 0.0700, "XOM": 0.0600, "JNJ": 0.0500,
        "AMZN": 0.0400, "PG": 0.0300, "OTHER": 0.2500,
    }),
}

print(f"{'Manager':25}  {'Active Share':>14}  {'Category':>22}")
for name, weights in managers.items():
    as_val = ActiveShare.compute(weights, benchmark_weights)
    category = ActiveShare.categorize(as_val)
    print(f"  {name:23}  {as_val:>13.2%}  {category:>22}")

print(f"\nActive Share thresholds:")
print(f"  >= 90%: concentrated_active (high conviction, likely high fee)")
print(f"  60-90%: moderately_active")
print(f"  20-60%: closet_indexer (index-like returns, active fees)")
print(f"  < 20%:  index_fund")


# ---------------------------------------------------------------------------
# 5. Brinson-Hood-Beebower Attribution
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("5. BRINSON-HOOD-BEEBOWER ATTRIBUTION")
print("=" * 60)

sectors = ["Technology", "Healthcare", "Financials", "Consumer", "Energy",
           "Industrials", "Materials", "Utilities", "Real Estate", "Communication"]

# Portfolio weights (manager has tech overweight, energy underweight)
portfolio_sector_weights = pd.Series({
    "Technology":    0.35,
    "Healthcare":    0.12,
    "Financials":    0.14,
    "Consumer":      0.10,
    "Energy":        0.02,
    "Industrials":   0.10,
    "Materials":     0.05,
    "Utilities":     0.04,
    "Real Estate":   0.04,
    "Communication": 0.04,
})

benchmark_sector_weights = pd.Series({
    "Technology":    0.28,
    "Healthcare":    0.13,
    "Financials":    0.12,
    "Consumer":      0.10,
    "Energy":        0.05,
    "Industrials":   0.09,
    "Materials":     0.03,
    "Utilities":     0.03,
    "Real Estate":   0.03,
    "Communication": 0.14,
})

# Quarterly returns
portfolio_sector_returns = pd.Series({
    "Technology":    0.0850,
    "Healthcare":    0.0320,
    "Financials":    0.0280,
    "Consumer":      0.0190,
    "Energy":       -0.0120,
    "Industrials":   0.0410,
    "Materials":     0.0230,
    "Utilities":    -0.0050,
    "Real Estate":   0.0080,
    "Communication": 0.0380,
})

benchmark_sector_returns = pd.Series({
    "Technology":    0.0720,
    "Healthcare":    0.0310,
    "Financials":    0.0260,
    "Consumer":      0.0200,
    "Energy":       -0.0150,
    "Industrials":   0.0390,
    "Materials":     0.0250,
    "Utilities":    -0.0040,
    "Real Estate":   0.0090,
    "Communication": 0.0360,
})

attribution = BrinsonAttribution(
    portfolio_weights=portfolio_sector_weights,
    benchmark_weights=benchmark_sector_weights,
    portfolio_returns=portfolio_sector_returns,
    benchmark_returns=benchmark_sector_returns,
)
result = attribution.compute()

port_total = (portfolio_sector_weights * portfolio_sector_returns).sum()
bench_total = (benchmark_sector_weights * benchmark_sector_returns).sum()

print(f"Quarterly portfolio return: {port_total:.4%}")
print(f"Quarterly benchmark return: {bench_total:.4%}")
print(f"Total active return:        {result.total_active_return:.4%}")
print()
print(f"Attribution summary:")
print(f"  Allocation effect: {result.total_allocation:.4%}")
print(f"  Selection effect:  {result.total_selection:.4%}")
print(f"  Interaction effect:{result.total_interaction:.4%}")
print(f"  Total:             {result.total_allocation + result.total_selection + result.total_interaction:.4%}  "
      f"(check: {abs((result.total_allocation + result.total_selection + result.total_interaction) - result.total_active_return) < 1e-10})")

detail = pd.DataFrame({
    "Port Wt": portfolio_sector_weights,
    "Bench Wt": benchmark_sector_weights,
    "Port Ret": portfolio_sector_returns,
    "Bench Ret": benchmark_sector_returns,
    "Allocation": result.allocation_effect,
    "Selection": result.selection_effect,
    "Interaction": result.interaction_effect,
})
detail["Total"] = detail["Allocation"] + detail["Selection"] + detail["Interaction"]
detail = detail.sort_values("Total", ascending=False)

print(f"\nDetailed attribution by sector:")
print(f"  {'Sector':15}  {'PW':>6}  {'BW':>6}  {'PR':>7}  {'BR':>7}  {'Alloc':>8}  {'Select':>8}  {'Total':>8}")
for sector, row in detail.iterrows():
    print(f"  {sector:15}  {row['Port Wt']:>6.2%}  {row['Bench Wt']:>6.2%}  "
          f"{row['Port Ret']:>7.2%}  {row['Bench Ret']:>7.2%}  "
          f"{row['Allocation']:>8.4%}  {row['Selection']:>8.4%}  {row['Total']:>8.4%}")

print(f"\nTop contributors to active return:")
top = detail.nlargest(3, "Total")
for sector, row in top.iterrows():
    print(f"  {sector}: {row['Total']:+.4%}")
print(f"Bottom detractors:")
bot = detail.nsmallest(3, "Total")
for sector, row in bot.iterrows():
    print(f"  {sector}: {row['Total']:+.4%}")
