"""Tests for credit risk module."""

import numpy as np
import pandas as pd
import pytest

from meridianalgo.credit import (
    CreditDefaultSwap,
    CreditRiskAnalyzer,
    MertonModel,
    ZSpreadCalculator,
)


class TestMertonModel:
    def test_calibrate_returns_dict(self):
        model = MertonModel(
            equity_value=50.0,
            equity_volatility=0.40,
            debt_face_value=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
        )
        result = model.calibrate()
        assert "asset_value" in result
        assert "default_probability" in result
        assert "distance_to_default" in result
        assert "asset_volatility" in result
        assert "leverage_ratio" in result

    def test_asset_value_exceeds_equity(self):
        model = MertonModel(
            equity_value=50.0,
            equity_volatility=0.40,
            debt_face_value=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
        )
        result = model.calibrate()
        assert result["asset_value"] >= 50.0

    def test_default_probability_in_unit_interval(self):
        model = MertonModel(50.0, 0.40, 100.0, 1.0, 0.05)
        result = model.calibrate()
        assert 0.0 <= result["default_probability"] <= 1.0

    def test_high_leverage_increases_pd(self):
        low_leverage = MertonModel(80.0, 0.30, 100.0, 1.0, 0.05)
        high_leverage = MertonModel(20.0, 0.60, 100.0, 1.0, 0.05)
        r_low = low_leverage.calibrate()
        r_high = high_leverage.calibrate()
        assert r_high["default_probability"] > r_low["default_probability"]

    def test_invalid_equity_value_raises(self):
        with pytest.raises(ValueError):
            MertonModel(-10.0, 0.30, 100.0, 1.0, 0.05)

    def test_invalid_volatility_raises(self):
        with pytest.raises(ValueError):
            MertonModel(50.0, 6.0, 100.0, 1.0, 0.05)

    def test_default_probability_term_structure(self):
        model = MertonModel(50.0, 0.40, 100.0, 1.0, 0.05)
        ts = model.default_probability_term_structure([0.5, 1.0, 2.0, 5.0])
        assert len(ts) == 4
        assert all(0 <= v <= 1 for v in ts.values)

    def test_term_structure_increasing_with_horizon(self):
        model = MertonModel(50.0, 0.40, 100.0, 1.0, 0.05)
        ts = model.default_probability_term_structure([1.0, 2.0, 5.0])
        assert ts[2.0] >= ts[1.0]

    def test_asset_value_property(self):
        model = MertonModel(50.0, 0.40, 100.0, 1.0, 0.05)
        v = model.asset_value
        assert v > 0

    def test_asset_volatility_property(self):
        model = MertonModel(50.0, 0.40, 100.0, 1.0, 0.05)
        sv = model.asset_volatility
        assert sv > 0

    def test_leverage_ratio(self):
        model = MertonModel(50.0, 0.40, 100.0, 1.0, 0.05)
        result = model.calibrate()
        assert 0 < result["leverage_ratio"] <= 1.0


class TestCreditDefaultSwap:
    def test_price_returns_cds_result(self):
        cds = CreditDefaultSwap(
            hazard_rate=0.02, recovery_rate=0.40, risk_free_rate=0.05, maturity=5.0
        )
        result = cds.price()
        assert result.fair_spread > 0
        assert result.risky_annuity > 0
        assert 0 < result.survival_probability < 1

    def test_spread_increases_with_hazard_rate(self):
        low = CreditDefaultSwap(hazard_rate=0.01, recovery_rate=0.40, maturity=5.0)
        high = CreditDefaultSwap(hazard_rate=0.05, recovery_rate=0.40, maturity=5.0)
        assert high.price().fair_spread > low.price().fair_spread

    def test_spread_decreases_with_recovery_rate(self):
        low_rr = CreditDefaultSwap(hazard_rate=0.02, recovery_rate=0.20, maturity=5.0)
        high_rr = CreditDefaultSwap(hazard_rate=0.02, recovery_rate=0.60, maturity=5.0)
        assert low_rr.price().fair_spread > high_rr.price().fair_spread

    def test_from_spread_roundtrip(self):
        spread = 0.0150
        cds = CreditDefaultSwap.from_spread(
            spread, recovery_rate=0.40, risk_free_rate=0.05, maturity=5.0
        )
        recovered_spread = cds.price().fair_spread
        assert abs(recovered_spread - spread) < 1e-4

    def test_from_spread_creates_cds(self):
        cds = CreditDefaultSwap.from_spread(0.02)
        assert cds.hazard_rate > 0

    def test_survival_probability_decreasing(self):
        cds = CreditDefaultSwap(hazard_rate=0.02)
        assert cds.survival_probability(1.0) > cds.survival_probability(5.0)

    def test_bootstrap_hazard_curve(self):
        maturities = [1.0, 3.0, 5.0]
        spreads = [0.01, 0.015, 0.02]
        curve = CreditDefaultSwap.bootstrap_hazard_curve(maturities, spreads)
        assert len(curve) == 3
        assert all(h > 0 for h in curve.values)

    def test_invalid_recovery_raises(self):
        with pytest.raises(ValueError):
            CreditDefaultSwap(hazard_rate=0.02, recovery_rate=1.5)

    def test_invalid_hazard_raises(self):
        with pytest.raises(ValueError):
            CreditDefaultSwap(hazard_rate=-0.01)


class TestCreditRiskAnalyzer:
    def setup_method(self):
        self.analyzer = CreditRiskAnalyzer()

    def test_expected_loss(self):
        el = self.analyzer.expected_loss(pd=0.02, lgd=0.45, ead=1_000_000)
        assert el == pytest.approx(0.02 * 0.45 * 1_000_000)

    def test_unexpected_loss_positive(self):
        ul = self.analyzer.unexpected_loss(pd=0.02, lgd=0.45, ead=1_000_000)
        assert ul > 0

    def test_credit_var_positive(self):
        cvar = self.analyzer.credit_var(pd=0.02, lgd=0.45, ead=1_000_000)
        assert cvar > 0

    def test_credit_var_higher_at_higher_confidence(self):
        cvar_99 = self.analyzer.credit_var(pd=0.02, lgd=0.45, ead=1e6, confidence=0.99)
        cvar_999 = self.analyzer.credit_var(pd=0.02, lgd=0.45, ead=1e6, confidence=0.999)
        assert cvar_999 > cvar_99

    def test_portfolio_expected_loss(self):
        exposures = pd.DataFrame(
            {"pd": [0.01, 0.02, 0.05], "lgd": [0.45, 0.40, 0.60], "ead": [1e6, 2e6, 0.5e6]}
        )
        result = self.analyzer.portfolio_expected_loss(exposures)
        assert result["total_el"] > 0
        assert "herfindahl_index" in result
        assert "top10_concentration" in result

    def test_portfolio_expected_loss_missing_column_raises(self):
        with pytest.raises(ValueError):
            self.analyzer.portfolio_expected_loss(pd.DataFrame({"pd": [0.01]}))


class TestZSpreadCalculator:
    def setup_method(self):
        self.cash_flows = [5.0, 5.0, 5.0, 5.0, 105.0]
        self.times = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.rates = [0.03, 0.035, 0.038, 0.040, 0.042]
        self.calc = ZSpreadCalculator(self.cash_flows, self.times, self.rates)

    def test_z_spread_at_par(self):
        par_price = self.calc.theoretical_price(0.0)
        z = self.calc.z_spread(par_price)
        assert abs(z) < 1e-6

    def test_z_spread_positive_for_discount_bond(self):
        z = self.calc.z_spread(98.0)
        assert z > 0

    def test_z_spread_negative_for_premium_bond(self):
        par_price = self.calc.theoretical_price(0.0)
        z = self.calc.z_spread(par_price + 2.0)
        assert z < 0

    def test_dv01_positive(self):
        dv01 = self.calc.dv01()
        assert dv01 > 0

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            ZSpreadCalculator([5, 5], [1, 2, 3], [0.03, 0.035, 0.038])

    def test_theoretical_price_decreases_with_spread(self):
        p0 = self.calc.theoretical_price(0.0)
        p1 = self.calc.theoretical_price(0.01)
        assert p0 > p1
