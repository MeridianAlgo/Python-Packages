# MeridianAlgo Examples

Comprehensive runnable examples covering every major module. Each file is
self-contained and executable with `pip install meridianalgo`.

---

## Index

| File | Topic | Key Concepts |
|------|-------|-------------|
| `01_getting_started.py` | First steps | Market data, returns, correlation, basic risk |
| `02_basic_usage.py` | Core functionality | Portfolio optimization, VaR, stat arb, ML |
| `03_advanced_trading_strategy.py` | Strategy building | Mean reversion, backtesting, performance eval |
| `04_comprehensive_examples.py` | Feature showcase | Analytics, liquidity, microstructure, factors |
| `05_quant_examples.py` | Professional quant | HFT, execution algorithms, Fama-French, HMM |
| `06_transaction_cost_optimization.py` | Transaction costs | VWAP/TWAP/POV, market impact, rebalancing |
| `07_derivatives_and_options.py` | Derivatives | Black-Scholes, Greeks, implied vol, exotic options |
| `08_credit_risk.py` | Credit risk | Merton model, CDS, Z-spread, portfolio EL/UL |
| `09_volatility_models.py` | Volatility | GARCH, realized vol, HAR-RV, regimes |
| `10_monte_carlo.py` | Monte Carlo | GBM, Heston, jump-diffusion, CIR, option pricing |
| `11_portfolio_insurance.py` | CPPI / TIPP | Floor protection, multiplier sensitivity |
| `12_benchmark_analytics.py` | Attribution | Active share, information ratio, BHB attribution |
| `13_scenario_analysis.py` | Stress testing | Historical scenarios, reverse stress, correlation stress |

---

## Running

```bash
python examples/01_getting_started.py
python examples/08_credit_risk.py
python examples/09_volatility_models.py
python examples/10_monte_carlo.py
python examples/11_portfolio_insurance.py
python examples/12_benchmark_analytics.py
python examples/13_scenario_analysis.py
```

Base installation covers all examples except those requiring optional extras:

```bash
pip install meridianalgo            # covers examples 01-08, 11-13
pip install meridianalgo[ml]        # examples 02, 04, 05 (LSTM, HMM)
pip install meridianalgo[volatility]# example 09 (GARCH MLE via arch)
pip install meridianalgo[all]       # everything
```

---

## Learning Path

**Beginner** — start here:
1. `01_getting_started.py` — load data, calculate returns, basic stats
2. `02_basic_usage.py` — portfolio optimization, VaR

**Intermediate** — build intuition:
3. `03_advanced_trading_strategy.py` — strategy construction and backtesting
4. `07_derivatives_and_options.py` — Black-Scholes, Greeks, implied vol
5. `08_credit_risk.py` — Merton model, default probability, CDS

**Advanced** — institutional techniques:
6. `09_volatility_models.py` — GARCH family, realized vol estimators, HAR-RV
7. `10_monte_carlo.py` — GBM, Heston, jump-diffusion, CIR, option pricing
8. `11_portfolio_insurance.py` — CPPI and TIPP floor protection
9. `12_benchmark_analytics.py` — active share, information ratio, BHB attribution
10. `13_scenario_analysis.py` — historical scenarios, reverse stress testing

**Execution and microstructure:**
11. `05_quant_examples.py` — optimal execution, pairs trading, regime detection
12. `06_transaction_cost_optimization.py` — market impact, tax-loss harvesting
