"""
Real-time risk monitoring system with customizable alerts and dashboards.
"""

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskLimit:
    """Risk limit definition."""

    name: str
    limit_type: str  # 'absolute', 'percentage', 'notional'
    threshold: float
    warning_threshold: float
    currency: str = "USD"
    scope: str = "portfolio"  # 'portfolio', 'strategy', 'asset_class', 'individual'
    scope_filter: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class RiskAlert:
    """Risk alert notification."""

    alert_id: str
    timestamp: datetime
    alert_type: str
    severity: str  # 'info', 'warning', 'critical'
    message: str
    current_value: float
    threshold: float
    limit_name: str
    scope: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioPosition:
    """Portfolio position for risk monitoring."""

    symbol: str
    quantity: float
    market_value: float
    unrealized_pnl: float
    asset_class: str
    sector: str = ""
    country: str = ""
    currency: str = "USD"
    last_updated: datetime = field(default_factory=datetime.now)


class RiskMetricCalculator:
    """Calculate various risk metrics for real-time monitoring."""

    @staticmethod
    def calculate_var(
        returns: pd.Series, confidence_level: float = 0.95, method: str = "historical"
    ) -> float:
        """Calculate Value at Risk."""

        if len(returns) == 0:
            return 0.0

        if method == "historical":
            return np.percentile(returns, (1 - confidence_level) * 100)

        elif method == "parametric":
            mean = returns.mean()
            std = returns.std()
            from scipy.stats import norm

            return mean + norm.ppf(1 - confidence_level) * std

        elif method == "monte_carlo":
            # Simplified Monte Carlo
            n_simulations = 10000
            simulated_returns = np.random.normal(
                returns.mean(), returns.std(), n_simulations
            )
            return np.percentile(simulated_returns, (1 - confidence_level) * 100)

        else:
            raise ValueError(f"Unknown VaR method: {method}")

    @staticmethod
    def calculate_expected_shortfall(
        returns: pd.Series, confidence_level: float = 0.95
    ) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""

        if len(returns) == 0:
            return 0.0

        var = RiskMetricCalculator.calculate_var(returns, confidence_level)
        tail_returns = returns[returns <= var]

        if len(tail_returns) == 0:
            return var

        return tail_returns.mean()

    @staticmethod
    def calculate_maximum_drawdown(returns: pd.Series) -> Dict[str, float]:
        """Calculate maximum drawdown and related metrics."""

        if len(returns) == 0:
            return {"max_drawdown": 0.0, "drawdown_duration": 0, "recovery_time": 0}

        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()

        # Calculate running maximum
        running_max = cumulative.expanding().max()

        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max

        # Maximum drawdown
        max_drawdown = drawdown.min()

        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_periods = []
        start_idx = None

        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_idx is None:
                start_idx = i
            elif not is_dd and start_idx is not None:
                drawdown_periods.append((start_idx, i - 1))
                start_idx = None

        # Handle case where drawdown continues to end
        if start_idx is not None:
            drawdown_periods.append((start_idx, len(drawdown) - 1))

        # Calculate maximum drawdown duration
        max_duration = 0
        if drawdown_periods:
            max_duration = max(end - start + 1 for start, end in drawdown_periods)

        return {
            "max_drawdown": max_drawdown,
            "drawdown_duration": max_duration,
            "current_drawdown": drawdown.iloc[-1] if len(drawdown) > 0 else 0.0,
        }

    @staticmethod
    def calculate_portfolio_beta(
        portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """Calculate portfolio beta relative to benchmark."""

        if (
            len(portfolio_returns) != len(benchmark_returns)
            or len(portfolio_returns) < 2
        ):
            return 1.0

        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)

        if benchmark_variance == 0:
            return 1.0

        return covariance / benchmark_variance

    @staticmethod
    def calculate_tracking_error(
        portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """Calculate tracking error."""

        if (
            len(portfolio_returns) != len(benchmark_returns)
            or len(portfolio_returns) < 2
        ):
            return 0.0

        excess_returns = portfolio_returns - benchmark_returns
        return excess_returns.std() * np.sqrt(252)  # Annualized


class RealTimeRiskMonitor:
    """Real-time risk monitoring system."""

    def __init__(self, update_frequency: int = 60):
        """
        Initialize risk monitor.

        Args:
            update_frequency: Update frequency in seconds
        """
        self.update_frequency = update_frequency
        self.positions: Dict[str, PortfolioPosition] = {}
        self.risk_limits: Dict[str, RiskLimit] = {}
        self.alerts: deque = deque(maxlen=1000)
        self.risk_metrics: Dict[str, Any] = {}
        self.returns_history: deque = deque(maxlen=252)  # 1 year of daily returns

        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.last_update = datetime.now()

        # Callbacks
        self.alert_callbacks: List[Callable[[RiskAlert], None]] = []

        # Risk calculation settings
        self.var_confidence_levels = [0.95, 0.99]
        self.var_methods = ["historical", "parametric"]

    def add_risk_limit(self, limit: RiskLimit) -> None:
        """Add risk limit to monitoring."""
        self.risk_limits[limit.name] = limit
        logger.info(f"Added risk limit: {limit.name}")

    def remove_risk_limit(self, limit_name: str) -> None:
        """Remove risk limit."""
        if limit_name in self.risk_limits:
            del self.risk_limits[limit_name]
            logger.info(f"Removed risk limit: {limit_name}")

    def update_position(self, position: PortfolioPosition) -> None:
        """Update portfolio position."""
        self.positions[position.symbol] = position
        position.last_updated = datetime.now()

    def remove_position(self, symbol: str) -> None:
        """Remove position from monitoring."""
        if symbol in self.positions:
            del self.positions[symbol]

    def add_alert_callback(self, callback: Callable[[RiskAlert], None]) -> None:
        """Add callback function for alerts."""
        self.alert_callbacks.append(callback)

    def start_monitoring(self) -> None:
        """Start real-time monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitor_thread.start()
        logger.info("Started real-time risk monitoring")

    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped real-time risk monitoring")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                self._update_risk_metrics()
                self._check_risk_limits()
                time.sleep(self.update_frequency)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_frequency)

    def _update_risk_metrics(self) -> None:
        """Update all risk metrics."""
        current_time = datetime.now()

        # Calculate portfolio-level metrics
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        total_unrealized_pnl = sum(
            pos.unrealized_pnl for pos in self.positions.values()
        )

        # Calculate portfolio return if we have previous value
        if (
            hasattr(self, "_previous_portfolio_value")
            and self._previous_portfolio_value != 0
        ):
            portfolio_return = (
                total_market_value - self._previous_portfolio_value
            ) / self._previous_portfolio_value
            self.returns_history.append(portfolio_return)

        self._previous_portfolio_value = total_market_value

        # Calculate risk metrics
        self.risk_metrics = {
            "timestamp": current_time,
            "total_market_value": total_market_value,
            "total_unrealized_pnl": total_unrealized_pnl,
            "portfolio_return_pct": (
                (total_unrealized_pnl / (total_market_value - total_unrealized_pnl))
                * 100
                if total_market_value != total_unrealized_pnl
                else 0
            ),
            "position_count": len(self.positions),
            "asset_class_exposure": self._calculate_asset_class_exposure(),
            "sector_exposure": self._calculate_sector_exposure(),
            "currency_exposure": self._calculate_currency_exposure(),
            "concentration_risk": self._calculate_concentration_risk(),
        }

        # Calculate VaR and ES if we have enough return history
        if len(self.returns_history) >= 30:
            returns_series = pd.Series(list(self.returns_history))

            for confidence_level in self.var_confidence_levels:
                for method in self.var_methods:
                    var_key = f"var_{int(confidence_level * 100)}_{method}"
                    es_key = f"es_{int(confidence_level * 100)}_{method}"

                    try:
                        var_value = RiskMetricCalculator.calculate_var(
                            returns_series, confidence_level, method
                        )
                        es_value = RiskMetricCalculator.calculate_expected_shortfall(
                            returns_series, confidence_level
                        )

                        self.risk_metrics[var_key] = var_value
                        self.risk_metrics[es_key] = es_value

                        # Convert to dollar terms
                        self.risk_metrics[f"{var_key}_dollar"] = (
                            var_value * total_market_value
                        )
                        self.risk_metrics[f"{es_key}_dollar"] = (
                            es_value * total_market_value
                        )

                    except Exception as e:
                        logger.warning(f"Failed to calculate {var_key}: {e}")

            # Calculate drawdown metrics
            drawdown_metrics = RiskMetricCalculator.calculate_maximum_drawdown(
                returns_series
            )
            self.risk_metrics.update(drawdown_metrics)

        self.last_update = current_time

    def _calculate_asset_class_exposure(self) -> Dict[str, float]:
        """Calculate exposure by asset class."""
        exposure = {}
        total_value = sum(pos.market_value for pos in self.positions.values())

        if total_value == 0:
            return exposure

        for position in self.positions.values():
            asset_class = position.asset_class
            if asset_class not in exposure:
                exposure[asset_class] = 0
            exposure[asset_class] += position.market_value / total_value

        return exposure

    def _calculate_sector_exposure(self) -> Dict[str, float]:
        """Calculate exposure by sector."""
        exposure = {}
        total_value = sum(pos.market_value for pos in self.positions.values())

        if total_value == 0:
            return exposure

        for position in self.positions.values():
            sector = position.sector or "Unknown"
            if sector not in exposure:
                exposure[sector] = 0
            exposure[sector] += position.market_value / total_value

        return exposure

    def _calculate_currency_exposure(self) -> Dict[str, float]:
        """Calculate exposure by currency."""
        exposure = {}
        total_value = sum(pos.market_value for pos in self.positions.values())

        if total_value == 0:
            return exposure

        for position in self.positions.values():
            currency = position.currency
            if currency not in exposure:
                exposure[currency] = 0
            exposure[currency] += position.market_value / total_value

        return exposure

    def _calculate_concentration_risk(self) -> Dict[str, float]:
        """Calculate concentration risk metrics."""
        if not self.positions:
            return {}

        total_value = sum(pos.market_value for pos in self.positions.values())
        if total_value == 0:
            return {}

        # Calculate position weights
        weights = [pos.market_value / total_value for pos in self.positions.values()]
        weights = np.array(weights)

        # Herfindahl-Hirschman Index
        hhi = np.sum(weights**2)

        # Largest position weight
        max_weight = np.max(weights)

        # Top 5 positions weight
        top_5_weight = (
            np.sum(np.sort(weights)[-5:]) if len(weights) >= 5 else np.sum(weights)
        )

        return {
            "herfindahl_index": hhi,
            "max_position_weight": max_weight,
            "top_5_weight": top_5_weight,
            "effective_positions": 1 / hhi if hhi > 0 else 0,
        }

    def _check_risk_limits(self) -> None:
        """Check all risk limits and generate alerts."""
        for limit_name, limit in self.risk_limits.items():
            if not limit.enabled:
                continue

            try:
                current_value = self._get_limit_value(limit)

                # Check if limit is breached
                if self._is_limit_breached(current_value, limit):
                    severity = "critical"
                    message = f"Risk limit '{limit_name}' breached: {current_value:.4f} > {limit.threshold:.4f}"
                elif self._is_warning_breached(current_value, limit):
                    severity = "warning"
                    message = f"Risk limit '{limit_name}' warning: {current_value:.4f} > {limit.warning_threshold:.4f}"
                else:
                    continue  # No breach

                # Create alert
                alert = RiskAlert(
                    alert_id=f"{limit_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    alert_type="risk_limit_breach",
                    severity=severity,
                    message=message,
                    current_value=current_value,
                    threshold=limit.threshold,
                    limit_name=limit_name,
                    scope=limit.scope,
                    metadata={"limit_type": limit.limit_type},
                )

                self._trigger_alert(alert)

            except Exception as e:
                logger.error(f"Error checking limit {limit_name}: {e}")

    def _get_limit_value(self, limit: RiskLimit) -> float:
        """Get current value for a risk limit."""

        if limit.limit_type == "var_95_historical":
            return abs(self.risk_metrics.get("var_95_historical_dollar", 0))

        elif limit.limit_type == "var_99_historical":
            return abs(self.risk_metrics.get("var_99_historical_dollar", 0))

        elif limit.limit_type == "expected_shortfall_95":
            return abs(self.risk_metrics.get("es_95_historical_dollar", 0))

        elif limit.limit_type == "max_drawdown":
            return abs(self.risk_metrics.get("current_drawdown", 0))

        elif limit.limit_type == "portfolio_value":
            return self.risk_metrics.get("total_market_value", 0)

        elif limit.limit_type == "unrealized_pnl":
            return self.risk_metrics.get("total_unrealized_pnl", 0)

        elif limit.limit_type == "concentration_max_position":
            return self.risk_metrics.get("concentration_risk", {}).get(
                "max_position_weight", 0
            )

        elif limit.limit_type == "asset_class_exposure":
            asset_class = limit.scope_filter.get("asset_class")
            exposures = self.risk_metrics.get("asset_class_exposure", {})
            return exposures.get(asset_class, 0)

        elif limit.limit_type == "sector_exposure":
            sector = limit.scope_filter.get("sector")
            exposures = self.risk_metrics.get("sector_exposure", {})
            return exposures.get(sector, 0)

        else:
            logger.warning(f"Unknown limit type: {limit.limit_type}")
            return 0.0

    def _is_limit_breached(self, current_value: float, limit: RiskLimit) -> bool:
        """Check if limit is breached."""
        if limit.limit_type in ["unrealized_pnl"] and current_value < 0:
            return abs(current_value) > limit.threshold
        else:
            return current_value > limit.threshold

    def _is_warning_breached(self, current_value: float, limit: RiskLimit) -> bool:
        """Check if warning threshold is breached."""
        if limit.limit_type in ["unrealized_pnl"] and current_value < 0:
            return abs(current_value) > limit.warning_threshold
        else:
            return current_value > limit.warning_threshold

    def _trigger_alert(self, alert: RiskAlert) -> None:
        """Trigger alert and notify callbacks."""
        self.alerts.append(alert)
        logger.warning(f"Risk Alert: {alert.message}")

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def get_current_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics."""
        return self.risk_metrics.copy()

    def get_recent_alerts(self, hours: int = 24) -> List[RiskAlert]:
        """Get recent alerts."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp > cutoff_time]

    def get_risk_dashboard_data(self) -> Dict[str, Any]:
        """Get data for risk dashboard."""

        recent_alerts = self.get_recent_alerts(24)
        critical_alerts = [a for a in recent_alerts if a.severity == "critical"]
        warning_alerts = [a for a in recent_alerts if a.severity == "warning"]

        return {
            "last_update": self.last_update,
            "monitoring_status": "active" if self.is_monitoring else "stopped",
            "risk_metrics": self.risk_metrics,
            "alert_summary": {
                "total_alerts_24h": len(recent_alerts),
                "critical_alerts_24h": len(critical_alerts),
                "warning_alerts_24h": len(warning_alerts),
            },
            "recent_alerts": recent_alerts[-10:],  # Last 10 alerts
            "risk_limits_status": {
                name: {
                    "enabled": limit.enabled,
                    "current_value": self._get_limit_value(limit),
                    "threshold": limit.threshold,
                    "warning_threshold": limit.warning_threshold,
                    "utilization": (
                        self._get_limit_value(limit) / limit.threshold
                        if limit.threshold > 0
                        else 0
                    ),
                }
                for name, limit in self.risk_limits.items()
            },
        }


class RiskDashboard:
    """Risk monitoring dashboard interface."""

    def __init__(self, monitor: RealTimeRiskMonitor):
        self.monitor = monitor

    def generate_html_dashboard(self) -> str:
        """Generate HTML dashboard."""

        dashboard_data = self.monitor.get_risk_dashboard_data()

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Risk Monitoring Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .metric-box {{ display: inline-block; margin: 10px; padding: 15px; 
                             border: 1px solid #ccc; border-radius: 5px; min-width: 200px; }}
                .critical {{ background-color: #ffebee; border-color: #f44336; }}
                .warning {{ background-color: #fff3e0; border-color: #ff9800; }}
                .normal {{ background-color: #e8f5e8; border-color: #4caf50; }}
                .alert {{ margin: 5px 0; padding: 10px; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Risk Monitoring Dashboard</h1>
                <p>Last Update: {dashboard_data["last_update"]}</p>
                <p>Status: {dashboard_data["monitoring_status"]}</p>
            </div>
            
            <h2>Risk Metrics</h2>
            <div class="metric-box">
                <h3>Portfolio Value</h3>
                <p>${dashboard_data["risk_metrics"].get("total_market_value", 0):,.2f}</p>
            </div>
            
            <div class="metric-box">
                <h3>Unrealized P&L</h3>
                <p>${dashboard_data["risk_metrics"].get("total_unrealized_pnl", 0):,.2f}</p>
            </div>
            
            <div class="metric-box">
                <h3>VaR (95%)</h3>
                <p>${dashboard_data["risk_metrics"].get("var_95_historical_dollar", 0):,.2f}</p>
            </div>
            
            <div class="metric-box">
                <h3>Max Drawdown</h3>
                <p>{dashboard_data["risk_metrics"].get("current_drawdown", 0):.2%}</p>
            </div>
            
            <h2>Recent Alerts</h2>
        """

        for alert in dashboard_data["recent_alerts"]:
            alert_class = alert.severity
            html += f"""
            <div class="alert {alert_class}">
                <strong>{alert.severity.upper()}</strong> - {alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")}<br>
                {alert.message}
            </div>
            """

        html += """
            </body>
        </html>
        """

        return html

    def export_risk_report(self, filename: str = None) -> str:
        """Export comprehensive risk report."""

        if filename is None:
            filename = f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        dashboard_data = self.monitor.get_risk_dashboard_data()

        # Convert datetime objects to strings for JSON serialization
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(filename, "w") as f:
            json.dump(dashboard_data, f, indent=2, default=serialize_datetime)

        return filename


# Predefined risk limit templates
class RiskLimitTemplates:
    """Predefined risk limit templates for common use cases."""

    @staticmethod
    def conservative_portfolio_limits(portfolio_value: float) -> List[RiskLimit]:
        """Conservative risk limits for institutional portfolios."""

        return [
            RiskLimit(
                name="VaR_95_Daily",
                limit_type="var_95_historical",
                threshold=portfolio_value * 0.02,  # 2% of portfolio
                warning_threshold=portfolio_value * 0.015,
                scope="portfolio",
            ),
            RiskLimit(
                name="Max_Drawdown",
                limit_type="max_drawdown",
                threshold=0.10,  # 10% maximum drawdown
                warning_threshold=0.07,
                scope="portfolio",
            ),
            RiskLimit(
                name="Max_Position_Concentration",
                limit_type="concentration_max_position",
                threshold=0.10,  # 10% maximum single position
                warning_threshold=0.08,
                scope="portfolio",
            ),
            RiskLimit(
                name="Equity_Exposure",
                limit_type="asset_class_exposure",
                threshold=0.80,  # 80% maximum equity exposure
                warning_threshold=0.75,
                scope="asset_class",
                scope_filter={"asset_class": "equity"},
            ),
        ]

    @staticmethod
    def aggressive_trading_limits(portfolio_value: float) -> List[RiskLimit]:
        """Aggressive risk limits for active trading strategies."""

        return [
            RiskLimit(
                name="VaR_95_Daily",
                limit_type="var_95_historical",
                threshold=portfolio_value * 0.05,  # 5% of portfolio
                warning_threshold=portfolio_value * 0.04,
                scope="portfolio",
            ),
            RiskLimit(
                name="Max_Drawdown",
                limit_type="max_drawdown",
                threshold=0.20,  # 20% maximum drawdown
                warning_threshold=0.15,
                scope="portfolio",
            ),
            RiskLimit(
                name="Daily_Loss_Limit",
                limit_type="unrealized_pnl",
                threshold=portfolio_value * 0.03,  # 3% daily loss limit
                warning_threshold=portfolio_value * 0.02,
                scope="portfolio",
            ),
        ]
