"""
Comprehensive performance analytics for backtesting results.
Implements 50+ performance metrics, risk-adjusted returns, and advanced analytics.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics container."""

    # Basic Return Metrics
    total_return: float
    annualized_return: float
    cumulative_return: float

    # Risk Metrics
    volatility: float
    downside_deviation: float
    max_drawdown: float
    max_drawdown_duration: int

    # Risk-Adjusted Returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float

    # Drawdown Metrics
    avg_drawdown: float
    avg_drawdown_duration: int
    recovery_factor: float
    ulcer_index: float

    # Value at Risk
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float

    # Tail Risk
    skewness: float
    kurtosis: float
    tail_ratio: float

    # Consistency Metrics
    win_rate: float
    profit_factor: float
    payoff_ratio: float
    expectancy: float

    # Advanced Metrics
    information_ratio: float
    treynor_ratio: float
    jensen_alpha: float
    beta: float

    # Stability Metrics
    stability_of_timeseries: float
    r_squared: float

    # Additional Metrics
    gain_to_pain_ratio: float
    sterling_ratio: float
    burke_ratio: float
    kappa_three: float

    # Trade Statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

    # Time-based Metrics
    best_month: float
    worst_month: float
    best_year: float
    worst_year: float
    positive_months: float

    # Benchmark Comparison (if benchmark provided)
    tracking_error: Optional[float] = None
    up_capture: Optional[float] = None
    down_capture: Optional[float] = None

    # Metadata
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    total_days: Optional[int] = None
    trading_days: Optional[int] = None


class PerformanceAnalyzer:
    """Comprehensive performance analytics engine."""

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = 252

    def analyze_returns(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        trade_data: Optional[pd.DataFrame] = None,
    ) -> PerformanceMetrics:
        """
        Perform comprehensive performance analysis.

        Args:
            returns: Time series of returns
            benchmark_returns: Optional benchmark returns for comparison
            trade_data: Optional trade-level data for trade statistics

        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        if len(returns) == 0:
            raise ValueError("Returns series cannot be empty")

        # Clean returns data
        returns = returns.dropna()
        if len(returns) == 0:
            raise ValueError("No valid returns data after cleaning")

        # Basic calculations
        cumulative_returns = (1 + returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1

        # Annualized return
        periods_per_year = self._infer_frequency(returns.index)
        annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1

        # Risk metrics
        volatility = returns.std() * np.sqrt(periods_per_year)
        downside_returns = returns[returns < 0]
        downside_deviation = (
            downside_returns.std() * np.sqrt(periods_per_year)
            if len(downside_returns) > 0
            else 0.0
        )

        # Drawdown analysis
        drawdown_metrics = self._calculate_drawdown_metrics(cumulative_returns)

        # Risk-adjusted returns
        sharpe_ratio = (
            (annualized_return - self.risk_free_rate) / volatility
            if volatility > 0
            else 0.0
        )
        sortino_ratio = (
            (annualized_return - self.risk_free_rate) / downside_deviation
            if downside_deviation > 0
            else 0.0
        )
        calmar_ratio = (
            annualized_return / abs(drawdown_metrics["max_drawdown"])
            if drawdown_metrics["max_drawdown"] < 0
            else 0.0
        )

        # Omega ratio
        omega_ratio = self._calculate_omega_ratio(
            returns, self.risk_free_rate / periods_per_year
        )

        # VaR and CVaR
        var_metrics = self._calculate_var_metrics(returns)

        # Tail risk metrics
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        tail_ratio = self._calculate_tail_ratio(returns)

        # Trade statistics
        if trade_data is not None:
            trade_stats = self._calculate_trade_statistics(trade_data)
        else:
            trade_stats = self._estimate_trade_statistics_from_returns(returns)

        # Advanced metrics
        ulcer_index = self._calculate_ulcer_index(cumulative_returns)
        gain_to_pain_ratio = self._calculate_gain_to_pain_ratio(returns)
        sterling_ratio = self._calculate_sterling_ratio(
            returns, drawdown_metrics["avg_drawdown"]
        )
        burke_ratio = self._calculate_burke_ratio(returns, cumulative_returns)
        kappa_three = self._calculate_kappa_three(returns)

        # Stability metrics
        stability_metrics = self._calculate_stability_metrics(cumulative_returns)

        # Time-based metrics
        time_metrics = self._calculate_time_based_metrics(returns)

        # Benchmark comparison
        benchmark_metrics = {}
        if benchmark_returns is not None:
            benchmark_metrics = self._calculate_benchmark_metrics(
                returns, benchmark_returns
            )

        # Create comprehensive metrics object
        metrics = PerformanceMetrics(
            # Basic returns
            total_return=total_return,
            annualized_return=annualized_return,
            cumulative_return=total_return,
            # Risk metrics
            volatility=volatility,
            downside_deviation=downside_deviation,
            max_drawdown=drawdown_metrics["max_drawdown"],
            max_drawdown_duration=drawdown_metrics["max_drawdown_duration"],
            # Risk-adjusted returns
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio,
            # Drawdown metrics
            avg_drawdown=drawdown_metrics["avg_drawdown"],
            avg_drawdown_duration=drawdown_metrics["avg_drawdown_duration"],
            recovery_factor=drawdown_metrics["recovery_factor"],
            ulcer_index=ulcer_index,
            # VaR metrics
            var_95=var_metrics["var_95"],
            var_99=var_metrics["var_99"],
            cvar_95=var_metrics["cvar_95"],
            cvar_99=var_metrics["cvar_99"],
            # Tail risk
            skewness=skewness,
            kurtosis=kurtosis,
            tail_ratio=tail_ratio,
            # Trade statistics
            win_rate=trade_stats["win_rate"],
            profit_factor=trade_stats["profit_factor"],
            payoff_ratio=trade_stats["payoff_ratio"],
            expectancy=trade_stats["expectancy"],
            total_trades=trade_stats["total_trades"],
            winning_trades=trade_stats["winning_trades"],
            losing_trades=trade_stats["losing_trades"],
            avg_win=trade_stats["avg_win"],
            avg_loss=trade_stats["avg_loss"],
            largest_win=trade_stats["largest_win"],
            largest_loss=trade_stats["largest_loss"],
            # Advanced metrics
            information_ratio=benchmark_metrics.get("information_ratio", 0.0),
            treynor_ratio=benchmark_metrics.get("treynor_ratio", 0.0),
            jensen_alpha=benchmark_metrics.get("jensen_alpha", 0.0),
            beta=benchmark_metrics.get("beta", 0.0),
            # Stability
            stability_of_timeseries=stability_metrics["stability"],
            r_squared=stability_metrics["r_squared"],
            # Additional metrics
            gain_to_pain_ratio=gain_to_pain_ratio,
            sterling_ratio=sterling_ratio,
            burke_ratio=burke_ratio,
            kappa_three=kappa_three,
            # Time-based
            best_month=time_metrics["best_month"],
            worst_month=time_metrics["worst_month"],
            best_year=time_metrics["best_year"],
            worst_year=time_metrics["worst_year"],
            positive_months=time_metrics["positive_months"],
            # Benchmark comparison
            tracking_error=benchmark_metrics.get("tracking_error"),
            up_capture=benchmark_metrics.get("up_capture"),
            down_capture=benchmark_metrics.get("down_capture"),
            # Metadata
            start_date=returns.index[0] if len(returns) > 0 else None,
            end_date=returns.index[-1] if len(returns) > 0 else None,
            total_days=(
                (returns.index[-1] - returns.index[0]).days if len(returns) > 0 else 0
            ),
            trading_days=len(returns),
        )

        return metrics

    def _infer_frequency(self, index: pd.DatetimeIndex) -> int:
        """Infer frequency and return periods per year."""
        if len(index) < 2:
            return self.trading_days_per_year

        median_diff = (index[1:] - index[:-1]).median()

        if median_diff <= timedelta(days=1):
            return self.trading_days_per_year  # Daily
        elif median_diff <= timedelta(days=7):
            return 52  # Weekly
        elif median_diff <= timedelta(days=31):
            return 12  # Monthly
        elif median_diff <= timedelta(days=92):
            return 4  # Quarterly
        else:
            return 1  # Annual

    def _calculate_drawdown_metrics(
        self, cumulative_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate comprehensive drawdown metrics."""
        # Calculate drawdowns
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak

        # Max drawdown
        max_drawdown = drawdown.min()

        # Drawdown duration analysis
        is_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0

        for is_dd in is_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0

        if current_period > 0:
            drawdown_periods.append(current_period)

        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0

        # Average drawdown
        drawdown_values = drawdown[drawdown < 0]
        avg_drawdown = drawdown_values.mean() if len(drawdown_values) > 0 else 0.0

        # Recovery factor
        total_return = cumulative_returns.iloc[-1] - 1
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_duration": max_drawdown_duration,
            "avg_drawdown": avg_drawdown,
            "avg_drawdown_duration": avg_drawdown_duration,
            "recovery_factor": recovery_factor,
        }

    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float) -> float:
        """Calculate Omega ratio."""
        excess_returns = returns - threshold
        gains = excess_returns[excess_returns > 0].sum()
        losses = abs(excess_returns[excess_returns < 0].sum())

        return gains / losses if losses > 0 else 0.0

    def _calculate_var_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate VaR and CVaR metrics."""
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)

        # Conditional VaR (Expected Shortfall)
        cvar_95_returns = returns[returns <= var_95]
        cvar_99_returns = returns[returns <= var_99]

        cvar_95 = cvar_95_returns.mean() if len(cvar_95_returns) > 0 else var_95
        cvar_99 = cvar_99_returns.mean() if len(cvar_99_returns) > 0 else var_99

        return {
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
        }

    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)."""
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)

        return abs(p95 / p5) if p5 != 0 else 0.0

    def _calculate_trade_statistics(self, trade_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trade-level statistics."""
        if "pnl" not in trade_data.columns:
            # Try to calculate P&L from available columns
            if all(
                col in trade_data.columns
                for col in ["quantity", "entry_price", "exit_price"]
            ):
                trade_data["pnl"] = trade_data["quantity"] * (
                    trade_data["exit_price"] - trade_data["entry_price"]
                )
            else:
                return self._get_default_trade_stats()

        total_trades = len(trade_data)
        winning_trades = len(trade_data[trade_data["pnl"] > 0])
        losing_trades = len(trade_data[trade_data["pnl"] < 0])

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        wins = trade_data[trade_data["pnl"] > 0]["pnl"]
        losses = trade_data[trade_data["pnl"] < 0]["pnl"]

        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0

        largest_win = wins.max() if len(wins) > 0 else 0.0
        largest_loss = losses.min() if len(losses) > 0 else 0.0

        gross_profit = wins.sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss < 0 else 0.0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "profit_factor": profit_factor,
            "payoff_ratio": payoff_ratio,
            "expectancy": expectancy,
        }

    def _estimate_trade_statistics_from_returns(
        self, returns: pd.Series
    ) -> Dict[str, Any]:
        """Estimate trade statistics from returns when trade data unavailable."""
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        total_periods = len(returns)
        winning_periods = len(positive_returns)
        losing_periods = len(negative_returns)

        win_rate = winning_periods / total_periods if total_periods > 0 else 0.0

        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0.0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0.0

        largest_win = positive_returns.max() if len(positive_returns) > 0 else 0.0
        largest_loss = negative_returns.min() if len(negative_returns) > 0 else 0.0

        gross_profit = positive_returns.sum() if len(positive_returns) > 0 else 0.0
        gross_loss = abs(negative_returns.sum()) if len(negative_returns) > 0 else 0.0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss < 0 else 0.0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        return {
            "total_trades": total_periods,  # Approximation
            "winning_trades": winning_periods,
            "losing_trades": losing_periods,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "profit_factor": profit_factor,
            "payoff_ratio": payoff_ratio,
            "expectancy": expectancy,
        }

    def _get_default_trade_stats(self) -> Dict[str, Any]:
        """Get default trade statistics when no data available."""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "profit_factor": 0.0,
            "payoff_ratio": 0.0,
            "expectancy": 0.0,
        }

    def _calculate_ulcer_index(self, cumulative_returns: pd.Series) -> float:
        """Calculate Ulcer Index (measure of downside risk)."""
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak * 100

        squared_drawdowns = drawdown**2
        ulcer_index = np.sqrt(squared_drawdowns.mean())

        return ulcer_index

    def _calculate_gain_to_pain_ratio(self, returns: pd.Series) -> float:
        """Calculate Gain-to-Pain ratio."""
        total_return = (1 + returns).prod() - 1

        negative_returns = returns[returns < 0]
        pain = abs(negative_returns.sum()) if len(negative_returns) > 0 else 0.0

        return total_return / pain if pain > 0 else 0.0

    def _calculate_sterling_ratio(
        self, returns: pd.Series, avg_drawdown: float
    ) -> float:
        """Calculate Sterling ratio."""
        periods_per_year = self._infer_frequency(returns.index)
        annualized_return = (1 + returns.mean()) ** periods_per_year - 1

        return annualized_return / abs(avg_drawdown) if avg_drawdown < 0 else 0.0

    def _calculate_burke_ratio(
        self, returns: pd.Series, cumulative_returns: pd.Series
    ) -> float:
        """Calculate Burke ratio."""
        periods_per_year = self._infer_frequency(returns.index)
        annualized_return = (1 + returns.mean()) ** periods_per_year - 1

        # Calculate drawdowns
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak

        # Square root of sum of squared drawdowns
        drawdown_risk = np.sqrt((drawdown**2).sum())

        return annualized_return / drawdown_risk if drawdown_risk > 0 else 0.0

    def _calculate_kappa_three(self, returns: pd.Series) -> float:
        """Calculate Kappa Three ratio."""
        periods_per_year = self._infer_frequency(returns.index)
        annualized_return = (1 + returns.mean()) ** periods_per_year - 1

        # Third lower partial moment
        threshold = 0.0
        downside_returns = returns[returns < threshold] - threshold
        lpm3 = (downside_returns**3).mean() if len(downside_returns) > 0 else 0.0

        return annualized_return / (lpm3 ** (1 / 3)) if lpm3 > 0 else 0.0

    def _calculate_stability_metrics(
        self, cumulative_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate stability metrics."""
        if len(cumulative_returns) < 2:
            return {"stability": 0.0, "r_squared": 0.0}

        # Linear regression of cumulative returns vs time
        x = np.arange(len(cumulative_returns))
        y = np.log(cumulative_returns.values)

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_squared = r_value**2

            # Stability of timeseries (R-squared of log returns vs time)
            stability = r_squared

        except (ValueError, RuntimeWarning):
            stability = 0.0
            r_squared = 0.0

        return {"stability": stability, "r_squared": r_squared}

    def _calculate_time_based_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate time-based performance metrics."""
        if len(returns) == 0:
            return {
                "best_month": 0.0,
                "worst_month": 0.0,
                "best_year": 0.0,
                "worst_year": 0.0,
                "positive_months": 0.0,
            }

        # Monthly returns
        try:
            monthly_returns = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
            best_month = monthly_returns.max() if len(monthly_returns) > 0 else 0.0
            worst_month = monthly_returns.min() if len(monthly_returns) > 0 else 0.0
            positive_months = (
                (monthly_returns > 0).sum() / len(monthly_returns) * 100
                if len(monthly_returns) > 0
                else 0.0
            )
        except Exception:
            best_month = worst_month = positive_months = 0.0

        # Yearly returns
        try:
            yearly_returns = returns.resample("Y").apply(lambda x: (1 + x).prod() - 1)
            best_year = yearly_returns.max() if len(yearly_returns) > 0 else 0.0
            worst_year = yearly_returns.min() if len(yearly_returns) > 0 else 0.0
        except Exception:
            best_year = worst_year = 0.0

        return {
            "best_month": best_month,
            "worst_month": worst_month,
            "best_year": best_year,
            "worst_year": worst_year,
            "positive_months": positive_months,
        }

    def _calculate_benchmark_metrics(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate benchmark comparison metrics."""
        # Align returns
        common_dates = returns.index.intersection(benchmark_returns.index)
        if len(common_dates) == 0:
            return {}

        port_ret = returns.loc[common_dates]
        bench_ret = benchmark_returns.loc[common_dates]

        # Tracking error
        active_returns = port_ret - bench_ret
        periods_per_year = self._infer_frequency(returns.index)
        tracking_error = active_returns.std() * np.sqrt(periods_per_year)

        # Information ratio
        information_ratio = (
            active_returns.mean() / active_returns.std() * np.sqrt(periods_per_year)
            if active_returns.std() > 0
            else 0.0
        )

        # Beta and alpha
        if len(port_ret) > 1 and bench_ret.var() > 0:
            beta = np.cov(port_ret, bench_ret)[0, 1] / bench_ret.var()

            # Jensen's alpha
            port_ann_ret = (1 + port_ret.mean()) ** periods_per_year - 1
            bench_ann_ret = (1 + bench_ret.mean()) ** periods_per_year - 1
            jensen_alpha = port_ann_ret - (
                self.risk_free_rate + beta * (bench_ann_ret - self.risk_free_rate)
            )

            # Treynor ratio
            treynor_ratio = (
                (port_ann_ret - self.risk_free_rate) / beta if beta != 0 else 0.0
            )
        else:
            beta = 0.0
            jensen_alpha = 0.0
            treynor_ratio = 0.0

        # Up/Down capture ratios
        up_periods = bench_ret > 0
        down_periods = bench_ret < 0

        if up_periods.sum() > 0 and bench_ret[up_periods].mean() != 0:
            up_capture = port_ret[up_periods].mean() / bench_ret[up_periods].mean()
        else:
            up_capture = 0.0

        if down_periods.sum() > 0 and bench_ret[down_periods].mean() != 0:
            down_capture = (
                port_ret[down_periods].mean() / bench_ret[down_periods].mean()
            )
        else:
            down_capture = 0.0

        return {
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "beta": beta,
            "jensen_alpha": jensen_alpha,
            "treynor_ratio": treynor_ratio,
            "up_capture": up_capture,
            "down_capture": down_capture,
        }

    def create_performance_report(
        self, metrics: PerformanceMetrics, strategy_name: str = "Strategy"
    ) -> str:
        """Create a formatted performance report."""
        report = []
        report.append(f"Performance Analysis Report: {strategy_name}")
        report.append("=" * 80)

        # Basic Performance
        report.append("\n BASIC PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Total Return:              {metrics.total_return:8.2%}")
        report.append(f"Annualized Return:         {metrics.annualized_return:8.2%}")
        report.append(f"Volatility:                {metrics.volatility:8.2%}")
        report.append(f"Sharpe Ratio:              {metrics.sharpe_ratio:8.2f}")

        # Risk Metrics
        report.append("\n  RISK METRICS")
        report.append("-" * 40)
        report.append(f"Max Drawdown:              {metrics.max_drawdown:8.2%}")
        report.append(
            f"Max DD Duration:           {metrics.max_drawdown_duration:8d} periods"
        )
        report.append(f"VaR (95%):                 {metrics.var_95:8.2%}")
        report.append(f"CVaR (95%):                {metrics.cvar_95:8.2%}")
        report.append(f"Downside Deviation:        {metrics.downside_deviation:8.2%}")

        # Risk-Adjusted Returns
        report.append("\n RISK-ADJUSTED RETURNS")
        report.append("-" * 40)
        report.append(f"Sortino Ratio:             {metrics.sortino_ratio:8.2f}")
        report.append(f"Calmar Ratio:              {metrics.calmar_ratio:8.2f}")
        report.append(f"Omega Ratio:               {metrics.omega_ratio:8.2f}")
        report.append(f"Gain-to-Pain Ratio:        {metrics.gain_to_pain_ratio:8.2f}")

        # Trade Statistics
        report.append("\n TRADE STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Trades:              {metrics.total_trades:8d}")
        report.append(f"Win Rate:                  {metrics.win_rate:8.2%}")
        report.append(f"Profit Factor:             {metrics.profit_factor:8.2f}")
        report.append(f"Payoff Ratio:              {metrics.payoff_ratio:8.2f}")
        report.append(f"Expectancy:                {metrics.expectancy:8.4f}")

        # Advanced Metrics
        report.append("\n ADVANCED METRICS")
        report.append("-" * 40)
        report.append(f"Ulcer Index:               {metrics.ulcer_index:8.2f}")
        report.append(f"Sterling Ratio:            {metrics.sterling_ratio:8.2f}")
        report.append(f"Burke Ratio:               {metrics.burke_ratio:8.2f}")
        report.append(
            f"Stability:                 {metrics.stability_of_timeseries:8.2f}"
        )

        # Tail Risk
        report.append("\n TAIL RISK")
        report.append("-" * 40)
        report.append(f"Skewness:                  {metrics.skewness:8.2f}")
        report.append(f"Kurtosis:                  {metrics.kurtosis:8.2f}")
        report.append(f"Tail Ratio:                {metrics.tail_ratio:8.2f}")

        # Benchmark Comparison (if available)
        if metrics.tracking_error is not None:
            report.append("\n BENCHMARK COMPARISON")
            report.append("-" * 40)
            report.append(f"Tracking Error:            {metrics.tracking_error:8.2%}")
            report.append(
                f"Information Ratio:         {metrics.information_ratio:8.2f}"
            )
            report.append(f"Beta:                      {metrics.beta:8.2f}")
            report.append(f"Jensen's Alpha:            {metrics.jensen_alpha:8.2%}")
            if metrics.up_capture is not None:
                report.append(f"Up Capture:                {metrics.up_capture:8.2%}")
                report.append(f"Down Capture:              {metrics.down_capture:8.2%}")

        # Time Period
        if metrics.start_date and metrics.end_date:
            report.append("\n TIME PERIOD")
            report.append("-" * 40)
            report.append(
                f"Start Date:                {metrics.start_date.strftime('%Y-%m-%d')}"
            )
            report.append(
                f"End Date:                  {metrics.end_date.strftime('%Y-%m-%d')}"
            )
            report.append(f"Total Days:                {metrics.total_days:8d}")
            report.append(f"Trading Days:              {metrics.trading_days:8d}")

        return "\n".join(report)


# Utility functions for rolling analysis
class RollingPerformanceAnalyzer:
    """Analyzer for rolling performance metrics."""

    def __init__(self, window: int = 252):
        self.window = window
        self.analyzer = PerformanceAnalyzer()

    def calculate_rolling_metrics(self, returns: pd.Series) -> pd.DataFrame:
        """Calculate rolling performance metrics."""
        if len(returns) < self.window:
            raise ValueError(
                f"Need at least {self.window} periods for rolling analysis"
            )

        rolling_metrics = []

        for i in range(self.window, len(returns) + 1):
            window_returns = returns.iloc[i - self.window : i]

            # Calculate basic metrics for this window
            cumulative = (1 + window_returns).prod() - 1
            volatility = window_returns.std() * np.sqrt(252)
            sharpe = (
                (window_returns.mean() * 252) / (window_returns.std() * np.sqrt(252))
                if window_returns.std() > 0
                else 0
            )

            # Drawdown
            cum_rets = (1 + window_returns).cumprod()
            peak = cum_rets.cummax()
            drawdown = ((cum_rets - peak) / peak).min()

            rolling_metrics.append(
                {
                    "date": returns.index[i - 1],
                    "return": cumulative,
                    "volatility": volatility,
                    "sharpe_ratio": sharpe,
                    "max_drawdown": drawdown,
                }
            )

        return pd.DataFrame(rolling_metrics).set_index("date")


if __name__ == "__main__":
    # Example usage
    print("Performance Analytics Example")
    print("=" * 50)

    # Generate sample returns
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    returns = pd.Series(np.random.normal(0.0008, 0.015, len(dates)), index=dates)

    # Create analyzer
    analyzer = PerformanceAnalyzer()

    # Analyze performance
    metrics = analyzer.analyze_returns(returns)

    # Generate report
    report = analyzer.create_performance_report(metrics, "Sample Strategy")
    print(report)

    print("\n Key Metrics Summary:")
    print(f"Total Return: {metrics.total_return:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Win Rate: {metrics.win_rate:.2%}")
