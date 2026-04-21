"""
Example 09: Volatility Modeling

Demonstrates MeridianAlgo's volatility module:
- Five OHLCV realized volatility estimators
- GARCH(1,1), EGARCH, GJR-GARCH conditional volatility
- Multi-step volatility forecasting
- Volatility term structure and VIX-style index
- Regime detection (low / medium / high vol)
- HAR-RV forecasting model
"""

import numpy as np
import pandas as pd

from meridianalgo.volatility import (
    GARCHModel,
    RealizedVolatility,
    VolatilityForecaster,
    VolatilityRegimeDetector,
    VolatilityTermStructure,
)


def make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Simulate realistic OHLCV data with volatility clustering."""
    rng = np.random.default_rng(seed)
    prices = [100.0]
    vols = [0.015]
    for _ in range(n - 1):
        new_vol = max(0.005, vols[-1] * 0.95 + rng.standard_normal() * 0.003)
        vols.append(new_vol)
        prices.append(prices[-1] * np.exp(rng.standard_normal() * new_vol + 0.0003))

    prices = np.array(prices)
    vols = np.array(vols)
    intraday = np.abs(rng.standard_normal(n)) * vols * 0.5

    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open":   prices * (1 + rng.standard_normal(n) * vols * 0.3),
            "High":   prices * (1 + np.abs(rng.standard_normal(n)) * vols + intraday),
            "Low":    prices * (1 - np.abs(rng.standard_normal(n)) * vols - intraday),
            "Close":  prices,
            "Volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        },
        index=dates,
    )


def make_returns(ohlcv: pd.DataFrame) -> pd.Series:
    return np.log(ohlcv["Close"] / ohlcv["Close"].shift(1)).dropna()


ohlcv = make_ohlcv()
returns = make_returns(ohlcv)


# ---------------------------------------------------------------------------
# 1. Realized Volatility Estimators
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. REALIZED VOLATILITY ESTIMATORS")
print("=" * 60)

rv = RealizedVolatility(ohlcv)
estimators = rv.all_estimators(window=21)
latest = estimators.iloc[-1].dropna()

print(f"21-day annualized volatility (latest observation):")
print(f"  Close-to-Close:  {latest.get('close_to_close_vol', float('nan')):.2%}")
print(f"  Parkinson:       {latest.get('parkinson_vol', float('nan')):.2%}")
print(f"  Garman-Klass:    {latest.get('garman_klass_vol', float('nan')):.2%}")
print(f"  Rogers-Satchell: {latest.get('rogers_satchell_vol', float('nan')):.2%}")
print(f"  Yang-Zhang:      {latest.get('yang_zhang_vol', float('nan')):.2%}")

# Efficiency: Parkinson/Garman-Klass are tighter estimates than close-to-close
cc = estimators["close_to_close_vol"].dropna()
gk = estimators["garman_klass_vol"].dropna()
common = cc.index.intersection(gk.index)
efficiency_ratio = gk.loc[common].std() / cc.loc[common].std()
print(f"\nGarman-Klass / Close-to-Close std ratio: {efficiency_ratio:.4f}")
print(f"(lower = more efficient estimator)")

# Rolling summary
print(f"\n21-day rolling annualized vol summary (Garman-Klass):")
gk_vol = estimators["garman_klass_vol"].dropna()
print(f"  Mean:    {gk_vol.mean():.2%}")
print(f"  Std:     {gk_vol.std():.2%}")
print(f"  Min:     {gk_vol.min():.2%}")
print(f"  Max:     {gk_vol.max():.2%}")
print(f"  Current: {gk_vol.iloc[-1]:.2%}")


# ---------------------------------------------------------------------------
# 2. GARCH Conditional Volatility
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("2. GARCH CONDITIONAL VOLATILITY")
print("=" * 60)

garch = GARCHModel(returns, model_type="garch", p=1, q=1)
result_g = garch.fit()

print(f"GARCH(1,1) Results:")
print(f"  omega (long-run var):  {result_g.omega:.8f}")
print(f"  alpha (ARCH term):     {result_g.alpha[0]:.6f}")
print(f"  beta (GARCH term):     {result_g.beta[0]:.6f}")
print(f"  Persistence (a+b):     {result_g.persistence:.6f}")
print(f"  Half-life:             {result_g.half_life:.1f} days")
print(f"  Log-likelihood:        {result_g.log_likelihood:.2f}")
print(f"  AIC:                   {result_g.aic:.2f}")
print(f"  BIC:                   {result_g.bic:.2f}")

cond_vol = result_g.conditional_volatility.dropna()
print(f"\nConditional volatility summary (annualized):")
ann_cond_vol = cond_vol * np.sqrt(252)
print(f"  Mean:    {ann_cond_vol.mean():.2%}")
print(f"  Min:     {ann_cond_vol.min():.2%}")
print(f"  Max:     {ann_cond_vol.max():.2%}")
print(f"  Current: {ann_cond_vol.iloc[-1]:.2%}")

# Forecast
print("\n10-day volatility forecast:")
forecast = garch.forecast(horizon=10, annualize=True)
print(f"  {'Day':>4}  {'Vol Forecast':>14}  {'Lower 95%':>12}  {'Upper 95%':>12}")
for i, (pt, lo, hi) in enumerate(zip(
    forecast.point_forecast,
    forecast.lower_bound,
    forecast.upper_bound,
), 1):
    print(f"  {i:>4}  {pt:>13.2%}  {lo:>11.2%}  {hi:>11.2%}")

# GJR-GARCH (asymmetric — captures leverage effect)
print("\nGJR-GARCH(1,1,1) — asymmetric volatility:")
gjr = GARCHModel(returns, model_type="gjr", p=1, q=1)
result_gjr = gjr.fit()
print(f"  alpha:       {result_gjr.alpha[0]:.6f}")
print(f"  beta:        {result_gjr.beta[0]:.6f}")
if result_gjr.gamma is not None:
    print(f"  gamma (asym):{result_gjr.gamma:.6f}")
    leverage = result_gjr.gamma > 0.01
    print(f"  Leverage effect present: {leverage}")


# ---------------------------------------------------------------------------
# 3. Volatility Term Structure
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("3. VOLATILITY TERM STRUCTURE")
print("=" * 60)

vts = VolatilityTermStructure(returns)
term_struct = vts.build(horizons=[5, 10, 21, 42, 63, 126, 252])

print(f"Realized volatility term structure (annualized):")
print(f"  {'Horizon':>10}  {'Vol':>10}  {'Label':>12}")
labels = {5: "1-week", 10: "2-week", 21: "1-month", 42: "2-month",
          63: "3-month", 126: "6-month", 252: "1-year"}
for h, vol in term_struct.items():
    print(f"  {h:>10}d  {vol:>10.2%}  {labels.get(h, ''):>12}")

slope = vts.slope(short_window=21, long_window=252)
vix_style = vts.vix_style_index()
vov = vts.vol_of_vol()

print(f"\nTerm structure slope (1yr - 1mo): {slope:.4f}")
print(f"VIX-style index (21d realized):   {vix_style:.2f}")
print(f"Vol-of-vol (63d rolling):         {vov:.4f}")
print(f"Interpretation: {'contango (normal)' if slope > 0 else 'backwardation (stress)'}")


# ---------------------------------------------------------------------------
# 4. Volatility Regime Detection
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("4. VOLATILITY REGIME DETECTION")
print("=" * 60)

detector = VolatilityRegimeDetector(returns, window=21)
regimes = detector.classify()

counts = regimes.value_counts()
total = len(regimes)
print(f"Regime distribution (21-day rolling vol):")
for regime, count in counts.items():
    print(f"  {regime:15s}: {count:5d} days ({count/total:.1%})")

stats = detector.regime_statistics()
print(f"\nReturn statistics by regime:")
print(f"  {'Regime':15s}  {'Ann Return':>12}  {'Ann Vol':>10}  {'Sharpe':>8}  {'Skew':>8}")
for regime, row in stats.iterrows():
    print(f"  {regime:15s}  {row['annualized_return']:>12.2%}  {row['annualized_vol']:>10.2%}  "
          f"{row['sharpe']:>8.3f}  {row['skewness']:>8.3f}")

current_regime = regimes.iloc[-1]
print(f"\nCurrent regime: {current_regime}")


# ---------------------------------------------------------------------------
# 5. HAR-RV Forecasting Model
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("5. HAR-RV FORECASTING MODEL")
print("=" * 60)

realized_variance = returns**2
har = VolatilityForecaster(realized_variance)
params = har.fit()

print(f"HAR-RV model fit (OLS):")
print(f"  Intercept (omega): {params['intercept']:.8f}")
print(f"  beta_d (daily):    {params['beta_d']:.6f}")
print(f"  beta_w (weekly):   {params['beta_w']:.6f}")
print(f"  beta_m (monthly):  {params['beta_m']:.6f}")
print(f"  R-squared:         {params['r_squared']:.4f}")

forecasts_5 = har.forecast(horizon=5)
print(f"\n5-day HAR-RV volatility forecast (annualized):")
for i, vol in enumerate(forecasts_5, 1):
    print(f"  Day {i}: {vol:.2%}")

forecasts_21 = har.forecast(horizon=21)
print(f"\n21-day average forecast: {forecasts_21.mean():.2%}")
print(f"Current 21-day realized: {returns.iloc[-21:].std() * np.sqrt(252):.2%}")
