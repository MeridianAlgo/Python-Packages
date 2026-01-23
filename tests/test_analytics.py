"""
Tests for meridianalgo.analytics module
"""

import numpy as np
import pandas as pd
import pytest


class TestPerformanceAnalyzer:
    """Tests for PerformanceAnalyzer class."""

    @pytest.fixture
    def returns(self):
        """Generate sample returns."""
        np.random.seed(42)
        return pd.Series(np.random.normal(0.0005, 0.015, 252))

    @pytest.fixture
    def benchmark(self):
        """Generate benchmark returns."""
        np.random.seed(123)
        return pd.Series(np.random.normal(0.0003, 0.012, 252))

    @pytest.fixture
    def analyzer(self, returns, benchmark):
        """Create PerformanceAnalyzer instance."""
        from meridianalgo.analytics import PerformanceAnalyzer

        return PerformanceAnalyzer(returns, benchmark=benchmark)

    def test_total_return(self, analyzer):
        """Test total return calculation."""
        result = analyzer.total_return()
        assert isinstance(result, float)
        assert -1 <= result <= 10  # Reasonable range

    def test_annualized_return(self, analyzer):
        """Test annualized return calculation."""
        result = analyzer.annualized_return()
        assert isinstance(result, float)

    def test_sharpe_ratio(self, analyzer):
        """Test Sharpe ratio calculation."""
        result = analyzer.sharpe_ratio()
        assert isinstance(result, float)
        assert -10 <= result <= 10  # Reasonable range

    def test_sortino_ratio(self, analyzer):
        """Test Sortino ratio calculation."""
        result = analyzer.sortino_ratio()
        assert isinstance(result, float)

    def test_calmar_ratio(self, analyzer):
        """Test Calmar ratio calculation."""
        result = analyzer.calmar_ratio()
        assert isinstance(result, float)

    def test_max_drawdown(self, analyzer):
        """Test max drawdown is negative or zero."""
        result = analyzer.max_drawdown()
        assert result <= 0

    def test_alpha(self, analyzer):
        """Test alpha calculation."""
        result = analyzer.alpha()
        assert isinstance(result, float)

    def test_beta(self, analyzer):
        """Test beta calculation."""
        result = analyzer.beta()
        assert isinstance(result, float)

    def test_summary(self, analyzer):
        """Test summary dictionary."""
        result = analyzer.summary()
        assert isinstance(result, dict)
        assert "sharpe_ratio" in result
        assert "total_return" in result


class TestRiskAnalyzer:
    """Tests for RiskAnalyzer class."""

    @pytest.fixture
    def returns(self):
        """Generate sample returns."""
        np.random.seed(42)
        return pd.Series(np.random.normal(0.0005, 0.015, 252))

    @pytest.fixture
    def analyzer(self, returns):
        """Create RiskAnalyzer instance."""
        from meridianalgo.analytics import RiskAnalyzer

        return RiskAnalyzer(returns)

    def test_value_at_risk_historical(self, analyzer):
        """Test historical VaR."""
        result = analyzer.value_at_risk(0.95, method="historical")
        assert result < 0

    def test_value_at_risk_parametric(self, analyzer):
        """Test parametric VaR."""
        result = analyzer.value_at_risk(0.95, method="parametric")
        assert result < 0

    def test_conditional_var(self, analyzer):
        """Test CVaR is worse than VaR."""
        var = analyzer.value_at_risk(0.95)
        cvar = analyzer.conditional_var(0.95)
        assert cvar <= var

    def test_max_drawdown(self, analyzer):
        """Test max drawdown."""
        result = analyzer.max_drawdown()
        assert result <= 0

    def test_summary(self, analyzer):
        """Test summary dictionary."""
        result = analyzer.summary()
        assert isinstance(result, dict)
        assert "var_95_historical" in result or "var_95" in result


class TestDrawdownAnalyzer:
    """Tests for DrawdownAnalyzer class."""

    @pytest.fixture
    def returns(self):
        """Generate sample returns."""
        np.random.seed(42)
        return pd.Series(np.random.normal(0.0005, 0.015, 252))

    @pytest.fixture
    def analyzer(self, returns):
        """Create DrawdownAnalyzer instance."""
        from meridianalgo.analytics import DrawdownAnalyzer

        return DrawdownAnalyzer(returns)

    def test_max_drawdown(self, analyzer):
        """Test max drawdown is negative."""
        result = analyzer.max_drawdown()
        assert result <= 0

    def test_time_underwater(self, analyzer):
        """Test time underwater is between 0 and 1."""
        result = analyzer.time_underwater()
        assert 0 <= result <= 1

    def test_ulcer_index(self, analyzer):
        """Test ulcer index is non-negative."""
        result = analyzer.ulcer_index()
        assert result >= 0

    def test_calmar_ratio(self, analyzer):
        """Test calmar ratio."""
        result = analyzer.calmar_ratio()
        assert isinstance(result, float)

    def test_drawdown_series(self, analyzer):
        """Test drawdown series."""
        result = analyzer.drawdown_series()
        assert isinstance(result, pd.Series)
        assert (result <= 0).all()
