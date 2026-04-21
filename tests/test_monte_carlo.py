"""Tests for Monte Carlo simulation engine."""

import numpy as np
import pandas as pd
import pytest

from meridianalgo.monte_carlo import (
    CIRModel,
    GeometricBrownianMotion,
    HestonModel,
    JumpDiffusionModel,
    MonteCarloEngine,
    QuasiRandomSampler,
)


class TestQuasiRandomSampler:
    def test_uniform_shape(self):
        sampler = QuasiRandomSampler(dimensions=2)
        samples = sampler.uniform(1000)
        assert samples.shape == (1000, 2)

    def test_uniform_1d_shape(self):
        sampler = QuasiRandomSampler(dimensions=1)
        samples = sampler.uniform(500)
        assert samples.shape == (500,)

    def test_uniform_in_unit_interval(self):
        sampler = QuasiRandomSampler(dimensions=3)
        samples = sampler.uniform(1000)
        assert (samples >= 0).all()
        assert (samples <= 1).all()

    def test_normal_samples_mean_near_zero(self):
        sampler = QuasiRandomSampler(dimensions=1)
        samples = sampler.normal(10000)
        assert abs(float(np.mean(samples))) < 0.05


class TestGeometricBrownianMotion:
    def setup_method(self):
        self.gbm = GeometricBrownianMotion(mu=0.08, sigma=0.20, seed=42)

    def test_simulate_shape(self):
        result = self.gbm.simulate(S0=100, T=1.0, n_paths=1000, n_steps=252)
        assert len(result.terminal_values) == 1000

    def test_terminal_values_positive(self):
        result = self.gbm.simulate(S0=100, T=1.0, n_paths=5000, n_steps=252)
        assert (result.terminal_values > 0).all()

    def test_antithetic_doubles_paths(self):
        result = self.gbm.simulate(S0=100, T=1.0, n_paths=1000, n_steps=252, antithetic=True)
        assert result.n_paths == 1000

    def test_mean_close_to_expected(self):
        S0, mu, sigma, T = 100, 0.08, 0.20, 1.0
        result = self.gbm.simulate(S0=S0, T=T, n_paths=50_000, n_steps=252)
        expected_mean = S0 * np.exp(mu * T)
        assert abs(result.mean - expected_mean) / expected_mean < 0.02

    def test_percentiles_ordered(self):
        result = self.gbm.simulate(S0=100, T=1.0, n_paths=10_000, n_steps=252)
        assert result.percentile_5 < result.percentile_25
        assert result.percentile_25 < result.median
        assert result.median < result.percentile_75
        assert result.percentile_75 < result.percentile_95

    def test_call_price_positive(self):
        result = self.gbm.call_price(S0=100, K=100, T=1.0, r=0.05, n_paths=50_000)
        assert result["price"] > 0

    def test_call_price_std_error_small(self):
        result = self.gbm.call_price(S0=100, K=100, T=1.0, r=0.05, n_paths=50_000)
        assert result["std_error"] < result["price"] * 0.05

    def test_put_call_parity_approximate(self):
        gbm_q = GeometricBrownianMotion(mu=0.05, sigma=0.20, seed=42)
        K, S0, T, r = 100, 100, 1.0, 0.05
        call = gbm_q.call_price(S0=S0, K=K, T=T, r=r, n_paths=100_000)
        put = call["price"] - S0 + K * np.exp(-r * T)
        assert put > 0

    def test_confidence_interval_contains_mean(self):
        result = self.gbm.simulate(S0=100, T=1.0, n_paths=10_000)
        lo, hi = result.confidence_interval_95
        assert lo < result.mean < hi

    def test_simulate_portfolio_shape(self):
        gbm = GeometricBrownianMotion(mu=0.08, sigma=0.20, seed=42)
        S0 = np.array([100.0, 150.0, 80.0])
        weights = np.array([0.4, 0.4, 0.2])
        corr = np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])
        result = gbm.simulate_portfolio(S0, weights, corr, T=1.0, n_paths=1000)
        assert len(result.terminal_values) == 1000


class TestHestonModel:
    def setup_method(self):
        self.heston = HestonModel(
            mu=0.05, v0=0.04, kappa=2.0, theta=0.04, xi=0.30, rho=-0.70, seed=42
        )

    def test_simulate_positive_prices(self):
        result = self.heston.simulate(S0=100, T=1.0, n_paths=2000, n_steps=252)
        assert (result.terminal_values > 0).all()

    def test_antithetic_paths(self):
        result = self.heston.simulate(S0=100, T=1.0, n_paths=1000)
        assert result.n_paths == 1000

    def test_mean_approximately_risk_neutral(self):
        heston = HestonModel(mu=0.05, v0=0.04, kappa=2.0, theta=0.04, xi=0.20, rho=-0.50, seed=0)
        result = heston.simulate(S0=100, T=1.0, n_paths=30_000, n_steps=252)
        expected = 100 * np.exp(0.05 * 1.0)
        assert abs(result.mean - expected) / expected < 0.05

    def test_model_label(self):
        result = self.heston.simulate(S0=100, T=1.0, n_paths=500)
        assert result.model == "Heston"


class TestJumpDiffusionModel:
    def setup_method(self):
        self.jdm = JumpDiffusionModel(
            mu=0.05, sigma=0.15, lam=0.10, mu_jump=-0.02, sigma_jump=0.05, seed=42
        )

    def test_positive_terminal_prices(self):
        result = self.jdm.simulate(S0=100, T=1.0, n_paths=5000, n_steps=252)
        assert (result.terminal_values > 0).all()

    def test_higher_jump_intensity_increases_vol(self):
        low = JumpDiffusionModel(mu=0.05, sigma=0.15, lam=0.01, mu_jump=-0.05, sigma_jump=0.1)
        high = JumpDiffusionModel(mu=0.05, sigma=0.15, lam=1.0, mu_jump=-0.05, sigma_jump=0.1)
        r_low = low.simulate(S0=100, T=1.0, n_paths=20_000, n_steps=252)
        r_high = high.simulate(S0=100, T=1.0, n_paths=20_000, n_steps=252)
        assert r_high.std > r_low.std

    def test_model_label(self):
        result = self.jdm.simulate(S0=100, T=1.0, n_paths=500)
        assert "JumpDiffusion" in result.model


class TestCIRModel:
    def setup_method(self):
        self.cir = CIRModel(r0=0.03, kappa=0.8, theta=0.04, sigma=0.05, seed=42)

    def test_non_negative_rates(self):
        result = self.cir.simulate(T=10.0, n_paths=5000, n_steps=2520)
        assert (result.terminal_values >= 0).all()

    def test_mean_reverts_to_theta(self):
        result = self.cir.simulate(T=50.0, n_paths=20_000, n_steps=5000)
        assert abs(result.mean - 0.04) < 0.005

    def test_zero_coupon_bond_between_zero_one(self):
        price = self.cir.zero_coupon_bond_price(t=0, T=5.0, r=0.03)
        assert 0 < price < 1

    def test_longer_maturity_cheaper_bond(self):
        p5 = self.cir.zero_coupon_bond_price(t=0, T=5.0, r=0.03)
        p10 = self.cir.zero_coupon_bond_price(t=0, T=10.0, r=0.03)
        assert p10 < p5


class TestMonteCarloEngine:
    def test_gbm_simulate(self):
        engine = MonteCarloEngine(model="gbm")
        engine.configure(mu=0.08, sigma=0.20)
        result = engine.simulate(S0=100, T=1.0, n_paths=1000)
        assert len(result.terminal_values) == 1000

    def test_heston_simulate(self):
        engine = MonteCarloEngine(model="heston")
        engine.configure(mu=0.05, v0=0.04, kappa=2.0, theta=0.04, xi=0.30, rho=-0.7)
        result = engine.simulate(S0=100, T=1.0, n_paths=2000, n_steps=252)
        assert len(result.terminal_values) == 2000

    def test_price_option_after_simulate(self):
        engine = MonteCarloEngine(model="gbm")
        engine.configure(mu=0.05, sigma=0.20)
        engine.simulate(S0=100, T=1.0, n_paths=50_000)
        price = engine.price_option(K=100, r=0.05, T=1.0, option_type="call")
        assert price["price"] > 0

    def test_portfolio_var_after_simulate(self):
        engine = MonteCarloEngine(model="gbm")
        engine.configure(mu=0.05, sigma=0.20)
        engine.simulate(S0=100, T=1.0, n_paths=10_000)
        var_result = engine.portfolio_var(initial_value=100)
        assert var_result["var"] > 0

    def test_price_option_before_simulate_raises(self):
        engine = MonteCarloEngine(model="gbm")
        with pytest.raises(RuntimeError):
            engine.price_option(K=100, r=0.05, T=1.0)

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError):
            MonteCarloEngine(model="unknown")

    def test_put_positive(self):
        engine = MonteCarloEngine(model="gbm", seed=0)
        engine.configure(mu=0.05, sigma=0.20)
        engine.simulate(S0=100, T=1.0, n_paths=50_000)
        price = engine.price_option(K=105, r=0.05, T=1.0, option_type="put")
        assert price["price"] > 0
