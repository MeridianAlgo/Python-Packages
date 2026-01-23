"""
MeridianAlgo Forex Module

Comprehensive forex trading and analysis functionality including currency pair analysis,
carry trade strategies, triangular arbitrage, and forex-specific risk management.
"""

import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# Check for optional dependencies
try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import requests  # noqa: F401

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class ForexAnalyzer:
    """
    Comprehensive forex market analysis and trading strategies.

    Features:
    - Currency pair correlation analysis
    - Carry trade strategies
    - Triangular arbitrage detection
    - Currency strength indicators
    - Forex volatility modeling
    - Interest rate parity analysis
    """

    def __init__(self, currency_pairs: List[str]):
        """
        Initialize forex analyzer with currency pairs.

        Args:
            currency_pairs: List of currency pairs (e.g., ['EURUSD', 'GBPUSD', 'USDJPY'])
        """
        self.currency_pairs = currency_pairs
        self.data = None
        self.returns = None
        self.correlations = None

    def fetch_forex_data(
        self,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        source: str = "yfinance",
    ) -> pd.DataFrame:
        """
        Fetch forex data for currency pairs.

        Args:
            start_date: Start date for data
            end_date: End date for data
            source: Data source ('yfinance', 'manual')

        Returns:
            DataFrame with forex price data
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if source == "yfinance" and YFINANCE_AVAILABLE:
            data = {}
            for pair in self.currency_pairs:
                try:
                    # Convert pair format for yfinance (e.g., EURUSD -> EURUSD=X)
                    ticker = f"{pair}=X"
                    fx_data = yf.download(ticker, start=start_date, end=end_date)
                    data[pair] = fx_data["Adj Close"]

                except Exception as e:
                    print(f"Warning: Could not fetch data for {pair}: {e}")
                    continue

            self.data = pd.DataFrame(data)

        else:
            # Generate synthetic data for demonstration
            np.random.seed(42)
            dates = pd.date_range(start=start_date, end=end_date, freq="D")
            n_days = len(dates)

            data = {}
            for pair in self.currency_pairs:
                # Generate realistic forex price series
                if "USD" in pair:
                    base_price = 1.0 + np.random.uniform(-0.5, 0.5)
                else:
                    base_price = np.random.uniform(0.5, 2.0)

                returns = np.random.normal(0, 0.01, n_days)
                data[pair] = base_price * np.exp(np.cumsum(returns))

            self.data = pd.DataFrame(data, index=dates)

        self.returns = self.data.pct_change().dropna()
        return self.data

    def calculate_currency_correlations(
        self, window: int = 252, method: str = "pearson"
    ) -> pd.DataFrame:
        """
        Calculate correlations between currency pairs.

        Args:
            window: Rolling window for correlation calculation
            method: Correlation method

        Returns:
            Correlation matrix
        """
        if self.returns is None:
            raise ValueError("Must fetch data first using fetch_forex_data()")

        if window is None:
            # Static correlation
            self.correlations = self.returns.corr(method=method)
        else:
            # Rolling correlation
            self.correlations = self.returns.rolling(window).corr(method=method)

        return self.correlations

    def identify_arbitrage_opportunities(
        self, threshold: float = 0.001
    ) -> List[Dict[str, Any]]:
        """
        Identify triangular arbitrage opportunities.

        Args:
            threshold: Minimum profit threshold for arbitrage

        Returns:
            List of arbitrage opportunities
        """
        if self.data is None:
            raise ValueError("Must fetch data first using fetch_forex_data()")

        opportunities = []

        # Extract unique currencies from pairs
        currencies = set()
        for pair in self.currency_pairs:
            if len(pair) == 6:
                currencies.add(pair[:3])
                currencies.add(pair[3:])

        currencies = list(currencies)

        # Check triangular arbitrage
        for i, curr1 in enumerate(currencies):
            for j, curr2 in enumerate(currencies[i + 1 :], i + 1):
                for k, curr3 in enumerate(currencies[j + 1 :], j + 1):
                    # Check if we have all three pairs
                    pair12 = f"{curr1}{curr2}"
                    pair21 = f"{curr2}{curr1}"
                    pair23 = f"{curr2}{curr3}"
                    pair32 = f"{curr3}{curr2}"
                    pair31 = f"{curr3}{curr1}"
                    pair13 = f"{curr1}{curr3}"

                    # Get available pairs
                    available_pairs = []
                    for pair in [pair12, pair21, pair23, pair32, pair31, pair13]:
                        if pair in self.data.columns:
                            available_pairs.append(pair)

                    if len(available_pairs) >= 3:
                        # Calculate arbitrage profit
                        latest_prices = self.data[available_pairs].iloc[-1]

                        # Simple triangular arbitrage check
                        if (
                            pair12 in available_pairs
                            and pair23 in available_pairs
                            and pair31 in available_pairs
                        ):
                            rate12 = latest_prices[pair12]
                            rate23 = latest_prices[pair23]
                            rate31 = latest_prices[pair31]

                            # Triangular product
                            triangular_product = rate12 * rate23 * rate31
                            profit = abs(triangular_product - 1.0)

                            if profit > threshold:
                                opportunities.append(
                                    {
                                        "type": "triangular_arbitrage",
                                        "currencies": [curr1, curr2, curr3],
                                        "pairs": [pair12, pair23, pair31],
                                        "rates": [rate12, rate23, rate31],
                                        "profit": profit,
                                        "triangular_product": triangular_product,
                                    }
                                )

        return opportunities

    def calculate_currency_strength(self, window: int = 252) -> pd.DataFrame:
        """
        Calculate currency strength indicators.

        Args:
            window: Rolling window for strength calculation

        Returns:
            DataFrame with currency strength values
        """
        if self.returns is None:
            raise ValueError("Must fetch data first using fetch_forex_data()")

        # Extract unique currencies
        currencies = set()
        for pair in self.currency_pairs:
            if len(pair) == 6:
                currencies.add(pair[:3])
                currencies.add(pair[3:])

        currencies = list(currencies)

        # Calculate strength for each currency
        strength_data = pd.DataFrame(index=self.returns.index, columns=currencies)

        for currency in currencies:
            # Find all pairs involving this currency
            currency_pairs = []
            for pair in self.currency_pairs:
                if currency in pair:
                    currency_pairs.append(pair)

            if currency_pairs:
                # Calculate average performance
                currency_returns = []
                for pair in currency_pairs:
                    if pair.startswith(currency):
                        # Currency is base currency
                        currency_returns.append(self.returns[pair])
                    else:
                        # Currency is quote currency - invert returns
                        currency_returns.append(-self.returns[pair])

                if currency_returns:
                    avg_returns = pd.concat(currency_returns, axis=1).mean(axis=1)
                    strength_data[currency] = avg_returns.rolling(window).sum()

        return strength_data

    def identify_carry_trade_opportunities(
        self, interest_rates: Dict[str, float], funding_currency: str = "USD"
    ) -> List[Dict[str, Any]]:
        """
        Identify carry trade opportunities based on interest rate differentials.

        Args:
            interest_rates: Dictionary of currency interest rates
            funding_currency: Currency to fund the trade

        Returns:
            List of carry trade opportunities
        """
        opportunities = []

        for pair in self.currency_pairs:
            if len(pair) == 6:
                base_currency = pair[:3]
                quote_currency = pair[3:]

                # Get interest rates
                base_rate = interest_rates.get(base_currency, 0.0)
                quote_rate = interest_rates.get(quote_currency, 0.0)

                # Calculate carry
                if quote_currency == funding_currency:
                    # Long base currency, short funding currency
                    carry = base_rate - quote_rate
                elif base_currency == funding_currency:
                    # Short base currency, long funding currency
                    carry = quote_rate - base_rate
                else:
                    continue

                # Calculate forward premium/discount (simplified)
                if self.data is not None and pair in self.data.columns:
                    current_price = self.data[pair].iloc[-1]

                    # Estimate forward rate using interest rate parity
                    days_to_forward = 90  # 3-month forward
                    forward_points = (base_rate - quote_rate) * days_to_forward / 360
                    forward_price = current_price * (1 + forward_points)

                    opportunities.append(
                        {
                            "pair": pair,
                            "base_currency": base_currency,
                            "quote_currency": quote_currency,
                            "base_rate": base_rate,
                            "quote_rate": quote_rate,
                            "carry": carry,
                            "spot_price": current_price,
                            "forward_price": forward_price,
                            "forward_points": forward_points,
                            "recommendation": "long" if carry > 0 else "short",
                        }
                    )

        # Sort by carry
        opportunities.sort(key=lambda x: x["carry"], reverse=True)

        return opportunities


class ForexRiskManager:
    """
    Forex-specific risk management tools.

    Features:
    - Position sizing based on volatility
    - Correlation-adjusted position limits
    - Currency exposure management
    - Forex-specific stress testing
    """

    def __init__(self, forex_analyzer: ForexAnalyzer):
        """
        Initialize forex risk manager.

        Args:
            forex_analyzer: ForexAnalyzer instance with data
        """
        self.analyzer = forex_analyzer
        self.positions = {}

    def calculate_position_size(
        self,
        pair: str,
        account_balance: float,
        risk_per_trade: float = 0.02,
        stop_loss_pips: float = 50,
        volatility_adjustment: bool = True,
    ) -> float:
        """
        Calculate optimal position size for forex trade.

        Args:
            pair: Currency pair
            account_balance: Account balance
            risk_per_trade: Risk percentage per trade
            stop_loss_pips: Stop loss in pips
            volatility_adjustment: Adjust for volatility

        Returns:
            Position size in units
        """
        if self.analyzer.data is None:
            raise ValueError("Must fetch data first")

        # Base position size
        risk_amount = account_balance * risk_per_trade
        pip_value = 0.0001  # Standard pip value

        # Adjust for JPY pairs
        if "JPY" in pair:
            pip_value = 0.01

        base_position_size = risk_amount / (stop_loss_pips * pip_value)

        # Volatility adjustment
        if volatility_adjustment and pair in self.analyzer.returns.columns:
            volatility = self.analyzer.returns[pair].rolling(21).std().iloc[-1]
            avg_volatility = self.analyzer.returns[pair].rolling(252).std().mean()

            vol_adjustment = avg_volatility / volatility
            base_position_size *= vol_adjustment

        return base_position_size

    def calculate_portfolio_exposure(
        self, positions: Dict[str, float], base_currency: str = "USD"
    ) -> Dict[str, float]:
        """
        Calculate portfolio exposure by currency.

        Args:
            positions: Dictionary of positions by pair
            base_currency: Base currency for exposure calculation

        Returns:
            Exposure by currency
        """
        exposure = {}

        for pair, position in positions.items():
            if len(pair) == 6:
                base_curr = pair[:3]
                quote_curr = pair[3:]

                # Get current price
                if pair in self.analyzer.data.columns:
                    price = self.analyzer.data[pair].iloc[-1]

                    # Calculate exposure in base currency
                    if quote_curr == base_currency:
                        # Position is already in base currency
                        exposure[base_curr] = exposure.get(base_curr, 0) + position
                    elif base_curr == base_currency:
                        # Position needs conversion
                        exposure[quote_curr] = (
                            exposure.get(quote_curr, 0) - position * price
                        )
                    else:
                        # Need cross rate conversion (simplified)
                        exposure[base_curr] = (
                            exposure.get(base_curr, 0) + position * price * 0.5
                        )
                        exposure[quote_curr] = (
                            exposure.get(quote_curr, 0) - position * 0.5
                        )

        return exposure


# Utility functions
def get_major_currency_pairs() -> List[str]:
    """Get list of major currency pairs."""
    return [
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "USDCHF",
        "AUDUSD",
        "NZDUSD",
        "USDCAD",
        "EURJPY",
        "GBPJPY",
        "EURGBP",
        "AUDJPY",
        "EURAUD",
    ]


def get_exotic_currency_pairs() -> List[str]:
    """Get list of exotic currency pairs."""
    return [
        "USDTRY",
        "USDZAR",
        "USDMXN",
        "USDBRL",
        "USDSEK",
        "USDNOK",
        "EURTRY",
        "EURZAR",
        "GBPSEK",
        "EURNOK",
        "AUDCHF",
        "CADCHF",
    ]


# Export main classes and functions
__all__ = [
    "ForexAnalyzer",
    "ForexRiskManager",
    "get_major_currency_pairs",
    "get_exotic_currency_pairs",
]
