"""
MeridianAlgo Alternative Assets Module

Comprehensive alternative assets analysis including real estate, private equity,
hedge funds, commodities, collectibles, and other alternative investments.
"""

import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")


class AlternativeAssetsAnalyzer:
    """
    Comprehensive alternative assets analysis and portfolio integration.

    Features:
    - Real estate investment analysis
    - Private equity valuation
    - Hedge fund performance analysis
    - Commodity futures analysis
    - Collectibles valuation
    - Infrastructure investment analysis
    - Alternative asset correlation analysis
    """

    def __init__(self):
        """Initialize alternative assets analyzer."""
        self.asset_classes = self._get_alternative_asset_classes()
        self.real_etfs = self._get_real_estate_etfs()
        self.commodity_futures = self._get_commodity_futures()
        self.hedge_fund_strategies = self._get_hedge_fund_strategies()
        self.private_equity_metrics = self._get_private_equity_metrics()

    def _get_alternative_asset_classes(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive alternative asset classes."""
        return {
            "real_estate": {
                "name": "Real Estate",
                "subcategories": [
                    "commercial",
                    "residential",
                    "industrial",
                    "retail",
                    "healthcare",
                ],
                "characteristics": [
                    "illiquid",
                    "income_generating",
                    "inflation_hedge",
                    "capital_appreciation",
                ],
                "typical_irr": 0.08,
                "volatility": 0.15,
                "correlation_with_equities": 0.6,
                "liquidity_premium": 0.03,
            },
            "private_equity": {
                "name": "Private Equity",
                "subcategories": [
                    "venture_capital",
                    "buyout",
                    "growth_equity",
                    "distressed",
                ],
                "characteristics": [
                    "illiquid",
                    "long_term",
                    "high_potential",
                    "manager_skill",
                ],
                "typical_irr": 0.15,
                "volatility": 0.25,
                "correlation_with_equities": 0.7,
                "liquidity_premium": 0.04,
            },
            "hedge_funds": {
                "name": "Hedge Funds",
                "subcategories": [
                    "long_short",
                    "global_macro",
                    "event_driven",
                    "relative_value",
                    "cta",
                ],
                "characteristics": [
                    "liquid",
                    "absolute_return",
                    "skill_based",
                    "low_correlation",
                ],
                "typical_irr": 0.10,
                "volatility": 0.12,
                "correlation_with_equities": 0.3,
                "liquidity_premium": 0.02,
            },
            "commodities": {
                "name": "Commodities",
                "subcategories": ["energy", "metals", "agriculture", "livestock"],
                "characteristics": [
                    "inflation_hedge",
                    "cyclical",
                    "supply_demand",
                    "geopolitical",
                ],
                "typical_irr": 0.06,
                "volatility": 0.20,
                "correlation_with_equities": 0.2,
                "liquidity_premium": 0.01,
            },
            "infrastructure": {
                "name": "Infrastructure",
                "subcategories": [
                    "transportation",
                    "utilities",
                    "telecom",
                    "renewable_energy",
                ],
                "characteristics": [
                    "stable_cash_flow",
                    "inflation_linked",
                    "long_term",
                    "essential",
                ],
                "typical_irr": 0.09,
                "volatility": 0.10,
                "correlation_with_equities": 0.4,
                "liquidity_premium": 0.025,
            },
            "collectibles": {
                "name": "Collectibles",
                "subcategories": [
                    "art",
                    "wine",
                    "classic_cars",
                    "watches",
                    "rare_coins",
                ],
                "characteristics": [
                    "passion_investment",
                    "illiquid",
                    "unique",
                    "emotional_value",
                ],
                "typical_irr": 0.05,
                "volatility": 0.30,
                "correlation_with_equities": 0.1,
                "liquidity_premium": 0.05,
            },
            "farmland": {
                "name": "Farmland",
                "subcategories": [
                    "crop_farming",
                    "livestock",
                    "timberland",
                    "permanent_crops",
                ],
                "characteristics": [
                    "inflation_hedge",
                    "stable_returns",
                    "essential",
                    "scarcity",
                ],
                "typical_irr": 0.08,
                "volatility": 0.12,
                "correlation_with_equities": 0.3,
                "liquidity_premium": 0.035,
            },
        }

    def _get_real_estate_etfs(self) -> Dict[str, Dict[str, Any]]:
        """Get major real estate ETFs and REITs."""
        return {
            "VNQ": {
                "name": "Vanguard Real Estate ETF",
                "expense_ratio": 0.0012,
                "dividend_yield": 0.038,
                "holdings_count": 165,
                "sectors": {
                    "retail": 0.25,
                    "residential": 0.20,
                    "office": 0.15,
                    "industrial": 0.15,
                    "healthcare": 0.10,
                    "specialty": 0.15,
                },
                "geographic_exposure": {"us": 0.95, "international": 0.05},
            },
            "IYR": {
                "name": "iShares U.S. Real Estate ETF",
                "expense_ratio": 0.0044,
                "dividend_yield": 0.041,
                "holdings_count": 72,
                "sectors": {
                    "retail": 0.30,
                    "residential": 0.18,
                    "office": 0.17,
                    "industrial": 0.12,
                    "healthcare": 0.08,
                    "specialty": 0.15,
                },
                "geographic_exposure": {"us": 1.0},
            },
            "XLRE": {
                "name": "Real Estate Select Sector SPDR Fund",
                "expense_ratio": 0.0010,
                "dividend_yield": 0.036,
                "holdings_count": 31,
                "sectors": {
                    "retail": 0.28,
                    "residential": 0.22,
                    "office": 0.16,
                    "industrial": 0.14,
                    "healthcare": 0.10,
                    "specialty": 0.10,
                },
                "geographic_exposure": {"us": 1.0},
            },
        }

    def _get_commodity_futures(self) -> Dict[str, Dict[str, Any]]:
        """Get major commodity futures."""
        return {
            "energy": {
                "crude_oil": {
                    "symbol": "CL",
                    "exchange": "NYMEX",
                    "contract_size": 1000,
                    "tick_value": 10,
                    "margin_requirement": 0.10,
                },
                "natural_gas": {
                    "symbol": "NG",
                    "exchange": "NYMEX",
                    "contract_size": 10000,
                    "tick_value": 10,
                    "margin_requirement": 0.12,
                },
                "gasoline": {
                    "symbol": "RB",
                    "exchange": "NYMEX",
                    "contract_size": 42000,
                    "tick_value": 4.2,
                    "margin_requirement": 0.08,
                },
            },
            "metals": {
                "gold": {
                    "symbol": "GC",
                    "exchange": "COMEX",
                    "contract_size": 100,
                    "tick_value": 10,
                    "margin_requirement": 0.05,
                },
                "silver": {
                    "symbol": "SI",
                    "exchange": "COMEX",
                    "contract_size": 5000,
                    "tick_value": 50,
                    "margin_requirement": 0.08,
                },
                "copper": {
                    "symbol": "HG",
                    "exchange": "COMEX",
                    "contract_size": 25000,
                    "tick_value": 12.5,
                    "margin_requirement": 0.10,
                },
            },
            "agriculture": {
                "corn": {
                    "symbol": "ZC",
                    "exchange": "CBOT",
                    "contract_size": 5000,
                    "tick_value": 12.5,
                    "margin_requirement": 0.15,
                },
                "wheat": {
                    "symbol": "ZW",
                    "exchange": "CBOT",
                    "contract_size": 5000,
                    "tick_value": 12.5,
                    "margin_requirement": 0.15,
                },
                "soybeans": {
                    "symbol": "ZS",
                    "exchange": "CBOT",
                    "contract_size": 5000,
                    "tick_value": 12.5,
                    "margin_requirement": 0.12,
                },
            },
        }

    def _get_hedge_fund_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Get hedge fund strategy characteristics."""
        return {
            "long_short_equity": {
                "name": "Long/Short Equity",
                "description": "Long and short equity positions with market exposure",
                "typical_leverage": 1.5,
                "target_return": 0.08,
                "volatility": 0.12,
                "correlation_to_s&p500": 0.6,
                "capacity": "Large",
                "liquidity": "Monthly",
            },
            "global_macro": {
                "name": "Global Macro",
                "description": "Betting on macroeconomic trends across asset classes",
                "typical_leverage": 2.0,
                "target_return": 0.10,
                "volatility": 0.15,
                "correlation_to_s&p500": 0.2,
                "capacity": "Medium",
                "liquidity": "Monthly",
            },
            "event_driven": {
                "name": "Event Driven",
                "description": "Profiting from corporate events like M&A, bankruptcies",
                "typical_leverage": 1.2,
                "target_return": 0.09,
                "volatility": 0.10,
                "correlation_to_s&p500": 0.3,
                "capacity": "Medium",
                "liquidity": "Quarterly",
            },
            "cta": {
                "name": "CTA/Managed Futures",
                "description": "Systematic trend-following across futures markets",
                "typical_leverage": 2.5,
                "target_return": 0.08,
                "volatility": 0.14,
                "correlation_to_s&p500": -0.1,
                "capacity": "Large",
                "liquidity": "Monthly",
            },
        }

    def _get_private_equity_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get private equity performance metrics."""
        return {
            "venture_capital": {
                "name": "Venture Capital",
                "investment_horizon": "7-10 years",
                "target_irr": 0.25,
                "target_multiple": 3.0,
                "success_rate": 0.20,
                "capital_calls_schedule": [0.3, 0.3, 0.2, 0.1, 0.1],
                "distributions_schedule": [0.0, 0.1, 0.2, 0.3, 0.4],
                "j_curve_duration": 4,
            },
            "buyout": {
                "name": "Buyout",
                "investment_horizon": "4-7 years",
                "target_irr": 0.20,
                "target_multiple": 2.5,
                "success_rate": 0.75,
                "capital_calls_schedule": [0.5, 0.3, 0.2],
                "distributions_schedule": [0.0, 0.2, 0.4, 0.4],
                "j_curve_duration": 2,
            },
            "growth_equity": {
                "name": "Growth Equity",
                "investment_horizon": "3-5 years",
                "target_irr": 0.18,
                "target_multiple": 2.0,
                "success_rate": 0.60,
                "capital_calls_schedule": [0.6, 0.4],
                "distributions_schedule": [0.0, 0.3, 0.4, 0.3],
                "j_curve_duration": 1,
            },
        }

    def analyze_real_etf_performance(
        self, etf_symbol: str, market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze real estate ETF performance."""
        if etf_symbol not in self.real_etfs:
            return {"error": f"ETF {etf_symbol} not found"}

        etf_info = self.real_etfs[etf_symbol]

        # Calculate returns
        if "price" in market_data.columns:
            prices = market_data["price"]
            total_returns = prices.pct_change()

            if "dividend" in market_data.columns:
                dividend_returns = market_data["dividend"] / prices.shift(1)
                total_returns = total_returns + dividend_returns.fillna(0)
        else:
            total_returns = market_data.pct_change()

        # Performance metrics
        annual_return = total_returns.mean() * 252
        annual_volatility = total_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

        # Dividend analysis
        dividend_yield = etf_info["dividend_yield"]

        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(
            prices if "price" in market_data.columns else market_data
        )

        # Correlation with broader market
        if "SPY" in market_data.columns:
            correlation = total_returns.corr(market_data["SPY"].pct_change())
        else:
            correlation = 0.6  # Default correlation

        return {
            "etf_info": etf_info,
            "performance_metrics": {
                "annual_return": annual_return,
                "annual_volatility": annual_volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "dividend_yield": dividend_yield,
            },
            "risk_analysis": {
                "correlation_to_market": correlation,
                "beta": correlation * (annual_volatility / 0.15),
                "sector_exposure": etf_info["sectors"],
                "geographic_exposure": etf_info["geographic_exposure"],
            },
        }

    def analyze_commodity_futures_strategy(
        self, commodity_category: str, strategy_type: str = "trend_following"
    ) -> Dict[str, Any]:
        """Analyze commodity futures trading strategy."""
        if commodity_category not in self.commodity_futures:
            return {"error": f"Commodity category {commodity_category} not found"}

        commodities = self.commodity_futures[commodity_category]

        # Strategy parameters
        strategy_params = {
            "trend_following": {
                "lookback_period": 60,
                "volatility_target": 0.15,
                "rebalance_frequency": "weekly",
                "position_sizing": "risk_parity",
            },
            "mean_reversion": {
                "lookback_period": 20,
                "entry_threshold": 2.0,
                "exit_threshold": 0.5,
                "stop_loss": 0.05,
            },
            "carry": {
                "roll_yield_threshold": 0.02,
                "term_structure_weight": 0.7,
                "seasonality_weight": 0.3,
            },
        }

        params = strategy_params.get(strategy_type, strategy_params["trend_following"])

        # Calculate expected performance (simplified)
        base_return = 0.06  # 6% base commodity return
        strategy_alpha = {
            "trend_following": 0.04,
            "mean_reversion": 0.03,
            "carry": 0.02,
        }.get(strategy_type, 0.03)

        expected_return = base_return + strategy_alpha
        expected_volatility = 0.20  # 20% commodity volatility

        # Risk metrics
        max_drawdown = 0.25  # Typical commodity max drawdown
        var_95 = expected_return - 1.65 * expected_volatility

        return {
            "commodity_category": commodity_category,
            "commodities": commodities,
            "strategy_type": strategy_type,
            "strategy_parameters": params,
            "expected_performance": {
                "annual_return": expected_return,
                "annual_volatility": expected_volatility,
                "sharpe_ratio": expected_return / expected_volatility,
                "max_drawdown": max_drawdown,
                "var_95": var_95,
            },
            "risk_analysis": {
                "correlation_to_equities": 0.2,
                "inflation_sensitivity": 0.7,
                "geopolitical_risk": 0.6,
                "seasonality_factor": (
                    0.3 if commodity_category == "agriculture" else 0.1
                ),
            },
        }

    def analyze_hedge_fund_allocation(
        self, capital: float, risk_tolerance: str = "medium"
    ) -> Dict[str, Any]:
        """Analyze hedge fund allocation strategy."""
        # Select strategies based on risk tolerance
        risk_adjusted_strategies = {
            "low": ["relative_value", "multi_strategy"],
            "medium": ["long_short_equity", "relative_value", "multi_strategy"],
            "high": ["global_macro", "cta", "event_driven", "long_short_equity"],
        }

        selected_strategies = risk_adjusted_strategies.get(
            risk_tolerance, risk_adjusted_strategies["medium"]
        )

        # Equal weight allocation
        allocation = {}
        n_strategies = len(selected_strategies)

        for strategy in selected_strategies:
            if strategy in self.hedge_fund_strategies:
                strategy_info = self.hedge_fund_strategies[strategy]
                allocation_amount = capital / n_strategies

                allocation[strategy] = {
                    "allocation": allocation_amount,
                    "strategy_info": strategy_info,
                    "expected_return": allocation_amount
                    * strategy_info["target_return"],
                    "expected_volatility": allocation_amount
                    * strategy_info["volatility"],
                    "leverage": strategy_info["typical_leverage"],
                    "correlation_to_s&p500": strategy_info["correlation_to_s&p500"],
                }

        # Portfolio metrics
        total_expected_return = sum(
            alloc["expected_return"] for alloc in allocation.values()
        )
        portfolio_volatility = np.sqrt(
            sum((alloc["expected_volatility"] ** 2) for alloc in allocation.values())
            + 2
            * sum(
                alloc1["expected_volatility"]
                * alloc2["expected_volatility"]
                * 0.3  # Assuming 0.3 correlation
                for i, alloc1 in enumerate(allocation.values())
                for alloc2 in list(allocation.values())[i + 1 :]
            )
        )

        portfolio_sharpe = (
            total_expected_return / portfolio_volatility
            if portfolio_volatility > 0
            else 0
        )

        return {
            "allocation": allocation,
            "portfolio_metrics": {
                "total_expected_return": total_expected_return,
                "portfolio_volatility": portfolio_volatility,
                "sharpe_ratio": portfolio_sharpe,
                "total_capital": capital,
            },
            "risk_analysis": {
                "diversification_benefit": len(selected_strategies)
                / len(self.hedge_fund_strategies),
                "average_correlation_to_market": sum(
                    alloc["correlation_to_s&p500"] for alloc in allocation.values()
                )
                / len(allocation),
                "liquidity_profile": "Monthly",
            },
        }

    def analyze_private_equity_cash_flows(
        self, strategy: str, committed_capital: float, investment_period: int = 5
    ) -> Dict[str, Any]:
        """Analyze private equity cash flow patterns."""
        if strategy not in self.private_equity_metrics:
            return {"error": f"Strategy {strategy} not found"}

        strategy_info = self.private_equity_metrics[strategy]

        # Generate cash flows
        years = list(
            range(1, investment_period + 5)
        )  # 5 additional years for distributions

        capital_calls = []
        distributions = []
        net_cash_flows = []

        for year in years:
            if year <= len(strategy_info["capital_calls_schedule"]):
                call_rate = strategy_info["capital_calls_schedule"][year - 1]
                capital_call = committed_capital * call_rate
            else:
                capital_call = 0

            if year <= len(strategy_info["distributions_schedule"]):
                dist_rate = strategy_info["distributions_schedule"][year - 1]
                distribution = (
                    committed_capital * strategy_info["target_multiple"] * dist_rate
                )
            else:
                distribution = 0

            capital_calls.append(capital_call)
            distributions.append(distribution)
            net_cash_flows.append(distribution - capital_call)

        # Calculate IRR and MOIC
        cash_flows = [-committed_capital] + net_cash_flows
        irr = self._calculate_irr(cash_flows)
        moic = sum(distributions) / sum(capital_calls) if sum(capital_calls) > 0 else 0

        # DPI, TVPI, RVPI
        dpi = sum(distributions) / committed_capital
        tvpi = (
            sum(distributions) + (committed_capital - sum(capital_calls))
        ) / committed_capital
        rvpi = tvpi - dpi

        return {
            "strategy": strategy,
            "strategy_info": strategy_info,
            "cash_flow_analysis": {
                "years": years,
                "capital_calls": capital_calls,
                "distributions": distributions,
                "net_cash_flows": net_cash_flows,
            },
            "performance_metrics": {
                "irr": irr,
                "moic": moic,
                "dpi": dpi,
                "tvpi": tvpi,
                "rvpi": rvpi,
            },
            "j_curve_analysis": {
                "j_curve_years": strategy_info["j_curve_duration"],
                "peak_negative_year": strategy_info["j_curve_duration"],
                "break_even_year": strategy_info["j_curve_duration"] + 1,
            },
        }

    def optimize_alternative_portfolio(
        self,
        traditional_weights: Dict[str, float],
        alternative_allocation: float = 0.20,
        risk_tolerance: str = "medium",
    ) -> Dict[str, Any]:
        """Optimize portfolio with alternative assets."""
        # Alternative asset selection based on risk tolerance
        alternative_selection = {
            "low": ["real_estate", "infrastructure", "farmland"],
            "medium": [
                "real_estate",
                "private_equity",
                "hedge_funds",
                "infrastructure",
            ],
            "high": ["private_equity", "hedge_funds", "commodities"],
        }

        selected_alternatives = alternative_selection.get(
            risk_tolerance, alternative_selection["medium"]
        )

        # Equal weight within alternatives
        alternative_weights = {}
        for alt_asset in selected_alternatives:
            if alt_asset in self.asset_classes:
                alternative_weights[alt_asset] = alternative_allocation / len(
                    selected_alternatives
                )

        # Adjust traditional weights
        traditional_adjustment = 1 - alternative_allocation
        adjusted_traditional_weights = {
            asset: weight * traditional_adjustment
            for asset, weight in traditional_weights.items()
        }

        # Combined portfolio
        combined_weights = {**adjusted_traditional_weights, **alternative_weights}

        # Calculate portfolio metrics
        expected_return = 0
        portfolio_volatility = 0

        # Traditional assets (assuming 7% return, 12% volatility)
        for weight in adjusted_traditional_weights.values():
            expected_return += weight * 0.07
            portfolio_volatility += weight * 0.12

        # Alternative assets
        for alt_asset, weight in alternative_weights.items():
            asset_info = self.asset_classes[alt_asset]
            expected_return += weight * asset_info["typical_irr"]
            portfolio_volatility += weight * asset_info["volatility"]

        # Simplified portfolio volatility (assuming 0.3 correlation between alternatives and traditional)
        portfolio_volatility = np.sqrt(
            portfolio_volatility**2
            + 2 * 0.3 * alternative_allocation * traditional_adjustment * 0.12 * 0.15
        )

        sharpe_ratio = (
            expected_return / portfolio_volatility if portfolio_volatility > 0 else 0
        )

        return {
            "traditional_weights": adjusted_traditional_weights,
            "alternative_weights": alternative_weights,
            "combined_weights": combined_weights,
            "portfolio_metrics": {
                "expected_return": expected_return,
                "portfolio_volatility": portfolio_volatility,
                "sharpe_ratio": sharpe_ratio,
                "alternative_allocation": alternative_allocation,
            },
            "diversification_benefits": {
                "reduced_correlation": True,
                "inflation_hedge": any(
                    self.asset_classes[alt]["characteristics"].count("inflation_hedge")
                    for alt in selected_alternatives
                ),
                "income_generation": any(
                    self.asset_classes[alt]["characteristics"].count(
                        "income_generating"
                    )
                    for alt in selected_alternatives
                ),
                "illiquidity_premium": sum(
                    self.asset_classes[alt]["liquidity_premium"] * weight
                    for alt, weight in alternative_weights.items()
                ),
            },
        }

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative_returns = (1 + prices.pct_change()).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()

    def _calculate_irr(self, cash_flows: List[float]) -> float:
        """Calculate Internal Rate of Return."""
        try:
            # Simplified IRR calculation
            for rate in np.linspace(-0.5, 1.0, 1000):
                npv = sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))
                if abs(npv) < 0.001:
                    return rate
            return 0.10  # Default if not found
        except Exception:
            return 0.10  # Default return


class CollectiblesValuation:
    """
    Collectibles valuation and investment analysis.

    Features:
    - Art market analysis
    - Wine valuation
    - Classic cars valuation
    - Watch market analysis
    - Rare coins valuation
    """

    def __init__(self):
        """Initialize collectibles valuation."""
        self.art_categories = self._get_art_categories()
        self.wine_regions = self._get_wine_regions()
        self.car_categories = self._get_car_categories()
        self.watch_brands = self._get_watch_brands()

    def _get_art_categories(self) -> Dict[str, Dict[str, Any]]:
        """Get art market categories."""
        return {
            "contemporary": {
                "average_appreciation": 0.08,
                "volatility": 0.25,
                "liquidity_score": 3,  # 1-10 scale
                "market_size": 65000000000,  # $65B
                "top_artists": ["Banksy", "Kaws", "Yayoi Kusama", "Jeff Koons"],
            },
            "modern": {
                "average_appreciation": 0.06,
                "volatility": 0.20,
                "liquidity_score": 4,
                "market_size": 45000000000,  # $45B
                "top_artists": [
                    "Pablo Picasso",
                    "Andy Warhol",
                    "Mark Rothko",
                    "Jackson Pollock",
                ],
            },
            "impressionist": {
                "average_appreciation": 0.05,
                "volatility": 0.15,
                "liquidity_score": 5,
                "market_size": 35000000000,  # $35B
                "top_artists": ["Claude Monet", "Pierre-Auguste Renoir", "Edgar Degas"],
            },
            "old_masters": {
                "average_appreciation": 0.04,
                "volatility": 0.12,
                "liquidity_score": 6,
                "market_size": 25000000000,  # $25B
                "top_artists": ["Leonardo da Vinci", "Rembrandt", "Caravaggio"],
            },
        }

    def _get_wine_regions(self) -> Dict[str, Dict[str, Any]]:
        """Get wine regions and characteristics."""
        return {
            "bordeaux": {
                "average_appreciation": 0.07,
                "volatility": 0.18,
                "holding_period": 10,  # years
                "top_chateaux": [
                    "Lafite Rothschild",
                    "Margaux",
                    "Latour",
                    "Haut-Brion",
                ],
                "investment_grade_vintages": [2000, 2005, 2009, 2010, 2015, 2016, 2018],
            },
            "burgundy": {
                "average_appreciation": 0.08,
                "volatility": 0.22,
                "holding_period": 8,
                "top_producers": [
                    "Domaine de la Romane-Conti",
                    "Leroy",
                    "Armand Rousseau",
                ],
                "investment_grade_vintages": [2005, 2009, 2010, 2015, 2016, 2018],
            },
            "champagne": {
                "average_appreciation": 0.06,
                "volatility": 0.15,
                "holding_period": 12,
                "top_houses": ["Dom Prignon", "Krug", "Salon", "Cristal"],
                "investment_grade_vintages": [2000, 2002, 2004, 2008, 2012, 2016],
            },
        }

    def _get_car_categories(self) -> Dict[str, Dict[str, Any]]:
        """Get classic car categories."""
        return {
            "ferrari": {
                "average_appreciation": 0.09,
                "volatility": 0.30,
                "maintenance_cost": 0.02,  # % of value annually
                "top_models": ["250 GTO", "Testarossa", "F40", "Enzo", "LaFerrari"],
                "auction_records": {"250 GTO": 48000000, "Testarossa": 2000000},
            },
            "porsche": {
                "average_appreciation": 0.07,
                "volatility": 0.25,
                "maintenance_cost": 0.015,
                "top_models": ["911 Carrera RS", "959", "Carrera GT", "918 Spyder"],
                "auction_records": {"911 Carrera RS": 3000000, "959": 1500000},
            },
            "lamborghini": {
                "average_appreciation": 0.08,
                "volatility": 0.28,
                "maintenance_cost": 0.025,
                "top_models": ["Miura", "Countach", "Diablo", "Aventador SVJ"],
                "auction_records": {"Miura": 2500000, "Countach": 800000},
            },
        }

    def _get_watch_brands(self) -> Dict[str, Dict[str, Any]]:
        """Get luxury watch brands."""
        return {
            "rolex": {
                "average_appreciation": 0.05,
                "volatility": 0.15,
                "service_cost": 0.005,  # % of value annually
                "top_models": ["Submariner", "Daytona", "GMT-Master", "Explorer"],
                "investment_models": ["Submariner", "Daytona", "GMT-Master II"],
            },
            "patek_philippe": {
                "average_appreciation": 0.08,
                "volatility": 0.20,
                "service_cost": 0.008,
                "top_models": ["Nautilus", "Aquanaut", "Calatrava", "World Timer"],
                "investment_models": [
                    "Nautilus 5711",
                    "Aquanaut 5167",
                    "World Timer 5930",
                ],
            },
            "audemars_piguet": {
                "average_appreciation": 0.07,
                "volatility": 0.18,
                "service_cost": 0.006,
                "top_models": ["Royal Oak", "Royal Oak Offshore", "Millenary"],
                "investment_models": ["Royal Oak 15202", "Royal Oak Offshore 26400"],
            },
        }

    def analyze_art_investment(
        self, category: str, purchase_price: float, holding_period: int = 5
    ) -> Dict[str, Any]:
        """Analyze art investment potential."""
        if category not in self.art_categories:
            return {"error": f"Art category {category} not found"}

        category_info = self.art_categories[category]

        # Calculate expected returns
        expected_appreciation = category_info["average_appreciation"]
        volatility = category_info["volatility"]

        # Monte Carlo simulation
        n_simulations = 1000
        simulated_returns = np.random.normal(
            expected_appreciation, volatility, (holding_period, n_simulations)
        )

        # Calculate final values
        final_values = purchase_price * np.prod(1 + simulated_returns, axis=0)

        # Statistics
        mean_final_value = np.mean(final_values)
        median_final_value = np.median(final_values)
        percentiles = np.percentile(final_values, [5, 25, 75, 95])

        # Calculate IRR
        irr = (mean_final_value / purchase_price) ** (1 / holding_period) - 1

        # Transaction costs
        buyer_premium = 0.10  # 10% buyer premium
        seller_commission = 0.15  # 15% seller commission
        insurance_cost = 0.01 * holding_period  # 1% per year

        net_proceeds = mean_final_value * (1 - seller_commission)
        total_cost = (
            purchase_price * (1 + buyer_premium) + purchase_price * insurance_cost
        )
        net_irr = (net_proceeds / total_cost) ** (1 / holding_period) - 1

        return {
            "category": category,
            "category_info": category_info,
            "investment_analysis": {
                "purchase_price": purchase_price,
                "holding_period": holding_period,
                "expected_final_value": mean_final_value,
                "median_final_value": median_final_value,
                "value_range": (percentiles[0], percentiles[3]),
                "irr": irr,
                "net_irr": net_irr,
                "volatility": volatility,
            },
            "cost_analysis": {
                "buyer_premium": purchase_price * buyer_premium,
                "seller_commission": mean_final_value * seller_commission,
                "insurance_cost": purchase_price * insurance_cost,
                "total_transaction_costs": purchase_price * buyer_premium
                + mean_final_value * seller_commission
                + purchase_price * insurance_cost,
            },
            "risk_analysis": {
                "liquidity_score": category_info["liquidity_score"],
                "market_size": category_info["market_size"],
                "probability_of_loss": np.mean(final_values < total_cost),
                "downside_protection": 0.3,  # Art has some downside protection
            },
        }


# Export main classes and functions
__all__ = ["AlternativeAssetsAnalyzer", "CollectiblesValuation"]
