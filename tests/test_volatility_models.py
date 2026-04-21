"""Tests for volatility models module."""

import numpy as np
import pandas as pd
import pytest

from meridianalgo.volatility import (
    GARCHModel,
    RealizedVolatility,
    VolatilityForecaster,
    VolatilityRegimeDetector,
    VolatilityTermStructure,
)


@pytest.fixture
def sample_returns():
    rng = np.random.default_rng(42)
    n = 500
    returns = rng.standard_normal(n) * 0.01
    returns[100:120] *= 3
    returns[300:310] *= 5
    return pd.Series(returns, name="returns")


@pytest.fixture
def sample_ohlcv():
    rng = np.random.default_rng(42)
    n = 300
    prices = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    high = prices * (1 + np.abs(rng.standard_normal(n) * 0.005))
    low = prices * (1 - np.abs(rng.standard_normal(n) * 0.005))
    open_p = prices * (1 + rng.standard_normal(n) * 0.003)
    volume = rng.integers(1000000, 5000000, n).astype(float)

    df = pd.DataFrame(
        {"Open": open_p, "High": high, "Low": low, "Close": prices, "Volume": volume}
    )
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)
    return df


class TestRealizedVolatility:
    def test_close_to_close(self, sample_ohlcv):
        rv = RealizedVolatility(sample_ohlcv)
        vol = rv.close_to_close(window=21)
        assert isinstance(vol, pd.Series)
        assert vol.dropna().gt(0).all()

    def test_parkinson(self, sample_ohlcv):
        rv = RealizedVolatility(sample_ohlcv)
        vol = rv.parkinson(window=21)
        assert vol.dropna().gt(0).all()

    def test_garman_klass(self, sample_ohlcv):
        rv = RealizedVolatility(sample_ohlcv)
        vol = rv.garman_klass(window=21)
        assert vol.dropna().gt(0).all()

    def test_rogers_satchell(self, sample_ohlcv):
        rv = RealizedVolatility(sample_ohlcv)
        vol = rv.rogers_satchell(window=21)
        assert vol.dropna().gt(0).all()

    def test_yang_zhang(self, sample_ohlcv):
        rv = RealizedVolatility(sample_ohlcv)
        vol = rv.yang_zhang(window=21)
        assert vol.dropna().gt(0).all()

    def test_all_estimators_returns_dataframe(self, sample_ohlcv):
        rv = RealizedVolatility(sample_ohlcv)
        df = rv.all_estimators(window=21)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[1] == 5

    def test_annualization_scales_correctly(self, sample_ohlcv):
        rv = RealizedVolatility(sample_ohlcv, trading_days=252)
        daily = rv.close_to_close(window=21, annualize=False)
        annual = rv.close_to_close(window=21, annualize=True)
        ratio = (annual.dropna() / daily.dropna()).dropna()
        assert ratio.mean() == pytest.approx(np.sqrt(252), rel=0.01)

    def test_missing_column_raises(self):
        df = pd.DataFrame({"Close": [100, 101, 102]})
        with pytest.raises(ValueError, match="Missing OHLCV columns"):
            RealizedVolatility(df)

    def test_realized_variance_shape(self, sample_ohlcv):
        rv = RealizedVolatility(sample_ohlcv)
        returns = np.log(sample_ohlcv["Close"] / sample_ohlcv["Close"].shift(1)).dropna()
        rvar = rv.realized_variance(returns, window=21)
        assert len(rvar) == len(returns)


class TestGARCHModel:
    def test_fit_garch_returns_result(self, sample_returns):
        model = GARCHModel(sample_returns, model_type="garch", p=1, q=1)
        result = model.fit()
        assert result.omega > 0
        assert len(result.alpha) == 1
        assert len(result.beta) == 1
        assert 0 < result.persistence < 1

    def test_conditional_vol_positive(self, sample_returns):
        model = GARCHModel(sample_returns)
        result = model.fit()
        assert result.conditional_volatility.dropna().gt(0).all()

    def test_half_life_finite(self, sample_returns):
        model = GARCHModel(sample_returns)
        result = model.fit()
        assert result.half_life > 0

    def test_invalid_model_type_raises(self, sample_returns):
        with pytest.raises(ValueError):
            GARCHModel(sample_returns, model_type="invalid")

    def test_forecast_shape(self, sample_returns):
        model = GARCHModel(sample_returns)
        model.fit()
        fc = model.forecast(horizon=10)
        assert len(fc.point_forecast) == 10

    def test_forecast_positive(self, sample_returns):
        model = GARCHModel(sample_returns)
        model.fit()
        fc = model.forecast(horizon=5)
        assert fc.point_forecast.gt(0).all()

    def test_aic_bic_computed(self, sample_returns):
        model = GARCHModel(sample_returns)
        result = model.fit()
        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)


class TestVolatilityTermStructure:
    def test_build_returns_series(self, sample_returns):
        vts = VolatilityTermStructure(sample_returns)
        ts = vts.build()
        assert isinstance(ts, pd.Series)
        assert len(ts) > 0

    def test_all_positive(self, sample_returns):
        vts = VolatilityTermStructure(sample_returns)
        ts = vts.build()
        assert ts.gt(0).all()

    def test_slope_is_float(self, sample_returns):
        vts = VolatilityTermStructure(sample_returns)
        slope = vts.slope()
        assert isinstance(slope, float)

    def test_vix_style_positive(self, sample_returns):
        vts = VolatilityTermStructure(sample_returns)
        vix = vts.vix_style_index()
        assert vix > 0

    def test_vol_of_vol_positive(self, sample_returns):
        vts = VolatilityTermStructure(sample_returns)
        vov = vts.vol_of_vol()
        assert vov > 0

    def test_custom_horizons(self, sample_returns):
        vts = VolatilityTermStructure(sample_returns)
        ts = vts.build(horizons=[10, 21, 63])
        assert list(ts.index) == [10, 21, 63]


class TestVolatilityRegimeDetector:
    def test_classify_returns_series(self, sample_returns):
        detector = VolatilityRegimeDetector(sample_returns)
        regimes = detector.classify()
        assert isinstance(regimes, pd.Series)

    def test_three_regime_labels(self, sample_returns):
        detector = VolatilityRegimeDetector(sample_returns)
        regimes = detector.classify()
        labels = set(regimes.unique())
        assert labels.issubset({"low_vol", "medium_vol", "high_vol"})

    def test_regime_statistics(self, sample_returns):
        detector = VolatilityRegimeDetector(sample_returns)
        stats = detector.regime_statistics()
        assert isinstance(stats, pd.DataFrame)
        assert "annualized_return" in stats.columns


class TestVolatilityForecaster:
    def test_fit_returns_params(self, sample_returns):
        rv = sample_returns**2
        forecaster = VolatilityForecaster(rv)
        params = forecaster.fit()
        assert "beta_d" in params
        assert "beta_w" in params
        assert "beta_m" in params
        assert "r_squared" in params

    def test_forecast_positive(self, sample_returns):
        rv = sample_returns**2
        forecaster = VolatilityForecaster(rv)
        fc = forecaster.forecast(horizon=5)
        assert fc.gt(0).all()

    def test_forecast_horizon_length(self, sample_returns):
        rv = sample_returns**2
        forecaster = VolatilityForecaster(rv)
        fc = forecaster.forecast(horizon=10)
        assert len(fc) == 10

    def test_r_squared_in_unit_interval(self, sample_returns):
        rv = sample_returns**2
        forecaster = VolatilityForecaster(rv)
        params = forecaster.fit()
        assert 0 <= params["r_squared"] <= 1
