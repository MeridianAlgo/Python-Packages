"""
Transaction cost optimization and execution algorithms.
Implements market impact models, optimal execution algorithms, and tax-loss harvesting.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of optimal execution algorithm."""

    execution_schedule: pd.DataFrame  # Time, quantity, price, cost
    total_cost: float
    market_impact: float
    timing_risk: float
    total_shares: float
    execution_method: str
    success: bool
    message: str
    metadata: Dict[str, Any] = None


@dataclass
class TaxLossHarvestingResult:
    """Result of tax-loss harvesting optimization."""

    trades: pd.DataFrame  # Asset, action (buy/sell), quantity, tax_impact
    total_tax_savings: float
    realized_losses: float
    realized_gains: float
    wash_sale_violations: List[Dict[str, Any]]
    success: bool
    message: str


class MarketImpactModel(ABC):
    """Abstract base class for market impact models."""

    @abstractmethod
    def calculate_impact(
        self, quantity: float, volume: float, volatility: float, **kwargs
    ) -> float:
        """Calculate market impact for a given trade."""
        pass


class LinearImpactModel(MarketImpactModel):
    """Linear market impact model."""

    def __init__(
        self, temporary_impact_coeff: float = 0.1, permanent_impact_coeff: float = 0.01
    ):
        self.temporary_impact_coeff = temporary_impact_coeff
        self.permanent_impact_coeff = permanent_impact_coeff

    def calculate_impact(
        self, quantity: float, volume: float, volatility: float, **kwargs
    ) -> float:
        """
        Calculate linear market impact.

        Args:
            quantity: Trade size (shares)
            volume: Average daily volume
            volatility: Asset volatility

        Returns:
            Market impact as fraction of price
        """
        participation_rate = abs(quantity) / volume if volume > 0 else 0

        # Temporary impact (recovers after trade)
        temporary_impact = (
            self.temporary_impact_coeff * volatility * np.sqrt(participation_rate)
        )

        # Permanent impact (persists)
        permanent_impact = self.permanent_impact_coeff * volatility * participation_rate

        return temporary_impact + permanent_impact


class SquareRootImpactModel(MarketImpactModel):
    """Square-root market impact model (Almgren-Chriss)."""

    def __init__(self, eta: float = 2.5e-6, gamma: float = 2.5e-7):
        self.eta = eta  # Temporary impact coefficient
        self.gamma = gamma  # Permanent impact coefficient

    def calculate_impact(
        self,
        quantity: float,
        volume: float,
        volatility: float,
        price: float = 100.0,
        **kwargs,
    ) -> float:
        """
        Calculate square-root market impact (Almgren-Chriss model).

        Args:
            quantity: Trade size (shares)
            volume: Average daily volume
            volatility: Asset volatility
            price: Current price

        Returns:
            Market impact in dollars
        """
        # Temporary impact
        temporary_impact = self.eta * volatility * price * np.sqrt(abs(quantity))

        # Permanent impact
        permanent_impact = self.gamma * volatility * price * abs(quantity)

        return temporary_impact + permanent_impact


class OptimalExecutionAlgorithm(ABC):
    """Abstract base class for optimal execution algorithms."""

    def __init__(self, impact_model: MarketImpactModel):
        self.impact_model = impact_model

    @abstractmethod
    def optimize_execution(
        self,
        target_quantity: float,
        market_data: Dict[str, Any],
        execution_horizon: int,
        **kwargs,
    ) -> ExecutionResult:
        """Optimize execution schedule."""
        pass


class TWAPAlgorithm(OptimalExecutionAlgorithm):
    """Time-Weighted Average Price algorithm."""

    def optimize_execution(
        self,
        target_quantity: float,
        market_data: Dict[str, Any],
        execution_horizon: int,
        **kwargs,
    ) -> ExecutionResult:
        """
        Optimize TWAP execution.

        Args:
            target_quantity: Total shares to trade
            market_data: Dict with 'volume', 'volatility', 'price'
            execution_horizon: Number of time periods

        Returns:
            ExecutionResult with TWAP schedule
        """
        # Equal distribution across time periods
        quantity_per_period = target_quantity / execution_horizon

        schedule_data = []
        total_cost = 0.0
        total_impact = 0.0

        for t in range(execution_horizon):
            # Calculate market impact for this slice
            impact = self.impact_model.calculate_impact(
                quantity_per_period,
                market_data["volume"],
                market_data["volatility"],
                price=market_data.get("price", 100.0),
            )

            # Cost includes market impact
            slice_cost = (
                abs(quantity_per_period)
                * market_data.get("price", 100.0)
                * (1 + impact)
            )

            schedule_data.append(
                {
                    "time": t,
                    "quantity": quantity_per_period,
                    "price": market_data.get("price", 100.0),
                    "impact": impact,
                    "cost": slice_cost,
                }
            )

            total_cost += slice_cost
            total_impact += impact

        schedule_df = pd.DataFrame(schedule_data)

        return ExecutionResult(
            execution_schedule=schedule_df,
            total_cost=total_cost,
            market_impact=total_impact,
            timing_risk=0.0,  # TWAP has minimal timing risk
            total_shares=target_quantity,
            execution_method="TWAP",
            success=True,
            message="TWAP execution optimized successfully",
        )


class VWAPAlgorithm(OptimalExecutionAlgorithm):
    """Volume-Weighted Average Price algorithm."""

    def optimize_execution(
        self,
        target_quantity: float,
        market_data: Dict[str, Any],
        execution_horizon: int,
        volume_profile: Optional[np.ndarray] = None,
        **kwargs,
    ) -> ExecutionResult:
        """
        Optimize VWAP execution.

        Args:
            target_quantity: Total shares to trade
            market_data: Dict with 'volume', 'volatility', 'price'
            execution_horizon: Number of time periods
            volume_profile: Expected volume distribution (if None, uses U-shaped)

        Returns:
            ExecutionResult with VWAP schedule
        """
        # Default U-shaped volume profile if not provided
        if volume_profile is None:
            volume_profile = self._generate_u_shaped_profile(execution_horizon)

        # Normalize volume profile
        volume_profile = volume_profile / np.sum(volume_profile)

        schedule_data = []
        total_cost = 0.0
        total_impact = 0.0

        for t in range(execution_horizon):
            # Quantity proportional to expected volume
            quantity_slice = target_quantity * volume_profile[t]

            # Calculate market impact
            impact = self.impact_model.calculate_impact(
                quantity_slice,
                market_data["volume"] * volume_profile[t],
                market_data["volatility"],
                price=market_data.get("price", 100.0),
            )

            slice_cost = (
                abs(quantity_slice) * market_data.get("price", 100.0) * (1 + impact)
            )

            schedule_data.append(
                {
                    "time": t,
                    "quantity": quantity_slice,
                    "price": market_data.get("price", 100.0),
                    "impact": impact,
                    "cost": slice_cost,
                    "volume_weight": volume_profile[t],
                }
            )

            total_cost += slice_cost
            total_impact += impact

        schedule_df = pd.DataFrame(schedule_data)

        return ExecutionResult(
            execution_schedule=schedule_df,
            total_cost=total_cost,
            market_impact=total_impact,
            timing_risk=np.std(volume_profile),  # Higher variance = higher timing risk
            total_shares=target_quantity,
            execution_method="VWAP",
            success=True,
            message="VWAP execution optimized successfully",
            metadata={"volume_profile": volume_profile.tolist()},
        )

    def _generate_u_shaped_profile(self, n_periods: int) -> np.ndarray:
        """Generate U-shaped intraday volume profile."""
        x = np.linspace(0, 1, n_periods)
        # U-shape: high at open/close, low at midday
        profile = 2 * (x**2 - x + 0.5)
        return profile / np.sum(profile)


class ImplementationShortfallAlgorithm(OptimalExecutionAlgorithm):
    """Implementation Shortfall (Almgren-Chriss) algorithm."""

    def __init__(self, impact_model: MarketImpactModel, risk_aversion: float = 1e-6):
        super().__init__(impact_model)
        self.risk_aversion = risk_aversion

    def optimize_execution(
        self,
        target_quantity: float,
        market_data: Dict[str, Any],
        execution_horizon: int,
        **kwargs,
    ) -> ExecutionResult:
        """
        Optimize Implementation Shortfall execution.

        Args:
            target_quantity: Total shares to trade
            market_data: Dict with 'volume', 'volatility', 'price'
            execution_horizon: Number of time periods

        Returns:
            ExecutionResult with optimal schedule
        """
        volatility = market_data["volatility"]
        volume = market_data["volume"]
        price = market_data.get("price", 100.0)

        # Almgren-Chriss parameters
        if isinstance(self.impact_model, SquareRootImpactModel):
            eta = self.impact_model.eta
            gamma = self.impact_model.gamma
        else:
            # Default parameters for other models
            eta = 2.5e-6
            gamma = 2.5e-7

        # Calculate optimal trajectory
        kappa = np.sqrt(self.risk_aversion * volatility**2 / (eta * price))
        tau = execution_horizon

        # Optimal trading trajectory
        schedule_data = []
        total_cost = 0.0
        total_impact = 0.0
        remaining_quantity = target_quantity

        for t in range(execution_horizon):
            # Time remaining
            time_remaining = tau - t

            # Optimal trade size (Almgren-Chriss formula)
            if time_remaining > 0:
                sinh_kt = np.sinh(kappa * time_remaining)
                sinh_k = np.sinh(kappa)

                if sinh_k > 0:
                    optimal_holdings = remaining_quantity * sinh_kt / sinh_k
                    trade_size = remaining_quantity - optimal_holdings
                else:
                    trade_size = remaining_quantity / time_remaining
            else:
                trade_size = remaining_quantity

            # Ensure we don't over-trade
            trade_size = min(trade_size, remaining_quantity)

            # Calculate costs
            impact = self.impact_model.calculate_impact(
                trade_size, volume, volatility, price=price
            )

            slice_cost = abs(trade_size) * price * (1 + impact)

            schedule_data.append(
                {
                    "time": t,
                    "quantity": trade_size,
                    "price": price,
                    "impact": impact,
                    "cost": slice_cost,
                    "remaining": remaining_quantity - trade_size,
                }
            )

            total_cost += slice_cost
            total_impact += impact
            remaining_quantity -= trade_size

            if remaining_quantity <= 0:
                break

        schedule_df = pd.DataFrame(schedule_data)

        # Calculate timing risk (variance of execution)
        timing_risk = np.var(schedule_df["quantity"]) if len(schedule_df) > 1 else 0.0

        return ExecutionResult(
            execution_schedule=schedule_df,
            total_cost=total_cost,
            market_impact=total_impact,
            timing_risk=timing_risk,
            total_shares=target_quantity,
            execution_method="ImplementationShortfall",
            success=True,
            message="Implementation Shortfall execution optimized successfully",
            metadata={
                "risk_aversion": self.risk_aversion,
                "kappa": kappa,
                "eta": eta,
                "gamma": gamma,
            },
        )


class TaxLossHarvester:
    """Tax-loss harvesting optimization with wash sale rules."""

    def __init__(self, wash_sale_days: int = 30):
        self.wash_sale_days = wash_sale_days

    def optimize_tax_harvesting(
        self,
        portfolio: pd.DataFrame,
        prices: pd.DataFrame,
        tax_rate: float = 0.25,
        min_loss_threshold: float = 100.0,
        **kwargs,
    ) -> TaxLossHarvestingResult:
        """
        Optimize tax-loss harvesting opportunities.

        Args:
            portfolio: DataFrame with columns ['asset', 'quantity', 'cost_basis', 'purchase_date']
            prices: DataFrame with current prices for each asset
            tax_rate: Tax rate for capital gains
            min_loss_threshold: Minimum loss to consider harvesting

        Returns:
            TaxLossHarvestingResult with optimal trades
        """
        trades = []
        total_tax_savings = 0.0
        realized_losses = 0.0
        realized_gains = 0.0
        wash_sale_violations = []

        current_date = datetime.now()

        for _, position in portfolio.iterrows():
            asset = position["asset"]
            quantity = position["quantity"]
            cost_basis = position["cost_basis"]
            purchase_date = pd.to_datetime(position["purchase_date"])

            if asset not in prices.columns:
                continue

            current_price = prices[asset].iloc[-1]
            unrealized_pnl = (current_price - cost_basis) * quantity

            # Check if this is a loss position worth harvesting
            if unrealized_pnl < -min_loss_threshold:
                # Check for wash sale rule violations
                wash_sale_violation = self._check_wash_sale_violation(
                    asset, purchase_date, current_date, portfolio
                )

                if not wash_sale_violation["violation"]:
                    # Calculate tax savings
                    tax_savings = abs(unrealized_pnl) * tax_rate

                    trades.append(
                        {
                            "asset": asset,
                            "action": "sell",
                            "quantity": quantity,
                            "price": current_price,
                            "cost_basis": cost_basis,
                            "pnl": unrealized_pnl,
                            "tax_savings": tax_savings,
                            "wash_sale_safe": True,
                        }
                    )

                    total_tax_savings += tax_savings
                    realized_losses += abs(unrealized_pnl)
                else:
                    wash_sale_violations.append(wash_sale_violation)

            elif unrealized_pnl > 0:
                # Track gains for offset opportunities
                realized_gains += unrealized_pnl

        # Look for gain/loss pairing opportunities
        trades = self._optimize_gain_loss_pairing(trades, portfolio, prices, tax_rate)

        trades_df = pd.DataFrame(trades)

        return TaxLossHarvestingResult(
            trades=trades_df,
            total_tax_savings=total_tax_savings,
            realized_losses=realized_losses,
            realized_gains=realized_gains,
            wash_sale_violations=wash_sale_violations,
            success=True,
            message=f"Tax-loss harvesting optimized: {len(trades)} trades identified",
        )

    def _check_wash_sale_violation(
        self,
        asset: str,
        purchase_date: datetime,
        current_date: datetime,
        portfolio: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Check if selling would violate wash sale rules."""
        # Check if we bought the same asset within wash sale period
        wash_sale_start = current_date - timedelta(days=self.wash_sale_days)
        wash_sale_end = current_date + timedelta(days=self.wash_sale_days)

        # Look for purchases of same asset in wash sale window
        asset_positions = portfolio[portfolio["asset"] == asset]

        for _, pos in asset_positions.iterrows():
            pos_date = pd.to_datetime(pos["purchase_date"])
            if (
                wash_sale_start <= pos_date <= wash_sale_end
                and pos_date != purchase_date
            ):
                return {
                    "violation": True,
                    "asset": asset,
                    "conflicting_date": pos_date,
                    "days_difference": abs((pos_date - current_date).days),
                }

        return {"violation": False}

    def _optimize_gain_loss_pairing(
        self,
        trades: List[Dict],
        portfolio: pd.DataFrame,
        prices: pd.DataFrame,
        tax_rate: float,
    ) -> List[Dict]:
        """Optimize pairing of gains and losses for tax efficiency."""
        # This is a simplified version - in practice, this would be more sophisticated
        loss_trades = [t for t in trades if t["pnl"] < 0]

        # Sort by tax savings (highest first)
        loss_trades.sort(key=lambda x: x["tax_savings"], reverse=True)

        return loss_trades


class TransactionCostOptimizer:
    """Main class for transaction cost optimization."""

    def __init__(self):
        self.impact_models = {
            "linear": LinearImpactModel(),
            "square_root": SquareRootImpactModel(),
        }

        self.execution_algorithms = {}
        self._initialize_algorithms()

    def _initialize_algorithms(self):
        """Initialize execution algorithms with different impact models."""
        for model_name, model in self.impact_models.items():
            self.execution_algorithms[f"twap_{model_name}"] = TWAPAlgorithm(model)
            self.execution_algorithms[f"vwap_{model_name}"] = VWAPAlgorithm(model)
            self.execution_algorithms[f"is_{model_name}"] = (
                ImplementationShortfallAlgorithm(model)
            )

    def optimize_execution(
        self,
        target_quantity: float,
        market_data: Dict[str, Any],
        execution_horizon: int,
        algorithm: str = "is_square_root",
        **kwargs,
    ) -> ExecutionResult:
        """
        Optimize trade execution using specified algorithm.

        Args:
            target_quantity: Total shares to trade
            market_data: Market data dictionary
            execution_horizon: Number of time periods
            algorithm: Algorithm to use ('twap_*', 'vwap_*', 'is_*')

        Returns:
            ExecutionResult with optimal execution schedule
        """
        if algorithm not in self.execution_algorithms:
            available = list(self.execution_algorithms.keys())
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {available}")

        algo = self.execution_algorithms[algorithm]
        return algo.optimize_execution(
            target_quantity, market_data, execution_horizon, **kwargs
        )

    def calculate_rebalancing_costs(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        portfolio_value: float,
        market_data: Dict[str, Dict[str, Any]],
        execution_algorithm: str = "is_square_root",
    ) -> Dict[str, Any]:
        """
        Calculate costs for portfolio rebalancing.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value
            market_data: Market data for each asset
            execution_algorithm: Algorithm for execution cost calculation

        Returns:
            Dictionary with rebalancing cost analysis
        """
        # Calculate required trades
        weight_changes = target_weights - current_weights
        trades = weight_changes * portfolio_value

        total_cost = 0.0
        execution_results = {}

        for asset in trades.index:
            if abs(trades[asset]) > 0 and asset in market_data:
                # Convert dollar amount to shares
                price = market_data[asset].get("price", 100.0)
                shares = trades[asset] / price

                # Calculate execution cost
                result = self.optimize_execution(
                    shares,
                    market_data[asset],
                    execution_horizon=10,  # Default 10 periods
                    algorithm=execution_algorithm,
                )

                execution_results[asset] = result
                total_cost += result.total_cost

        return {
            "total_cost": total_cost,
            "cost_percentage": total_cost / portfolio_value * 100,
            "execution_results": execution_results,
            "trades": trades.to_dict(),
        }

    def optimize_tax_harvesting(
        self, portfolio: pd.DataFrame, prices: pd.DataFrame, **kwargs
    ) -> TaxLossHarvestingResult:
        """Optimize tax-loss harvesting."""
        harvester = TaxLossHarvester()
        return harvester.optimize_tax_harvesting(portfolio, prices, **kwargs)


# Example usage and testing functions
def create_sample_market_data() -> Dict[str, Any]:
    """Create sample market data for testing."""
    return {
        "volume": 1000000,  # Average daily volume
        "volatility": 0.25,  # Annual volatility
        "price": 100.0,  # Current price
        "bid_ask_spread": 0.01,  # Bid-ask spread
    }


def create_sample_portfolio() -> pd.DataFrame:
    """Create sample portfolio for tax harvesting testing."""
    return pd.DataFrame(
        {
            "asset": ["AAPL", "GOOGL", "MSFT", "TSLA"],
            "quantity": [100, 50, 200, 75],
            "cost_basis": [150.0, 2800.0, 300.0, 800.0],
            "purchase_date": ["2023-01-15", "2023-03-20", "2023-02-10", "2023-04-05"],
        }
    )


if __name__ == "__main__":
    # Example usage
    optimizer = TransactionCostOptimizer()

    # Test execution optimization
    market_data = create_sample_market_data()
    result = optimizer.optimize_execution(
        target_quantity=10000,
        market_data=market_data,
        execution_horizon=20,
        algorithm="is_square_root",
    )

    print(f"Execution Result: {result.execution_method}")
    print(f"Total Cost: ${result.total_cost:,.2f}")
    print(f"Market Impact: {result.market_impact:.4f}")

    # Test tax harvesting
    portfolio = create_sample_portfolio()
    prices = pd.DataFrame(
        {"AAPL": [120.0], "GOOGL": [2600.0], "MSFT": [280.0], "TSLA": [600.0]}
    )

    tax_result = optimizer.optimize_tax_harvesting(portfolio, prices)
    print("\nTax Harvesting Result:")
    print(f"Total Tax Savings: ${tax_result.total_tax_savings:,.2f}")
    print(f"Realized Losses: ${tax_result.realized_losses:,.2f}")
