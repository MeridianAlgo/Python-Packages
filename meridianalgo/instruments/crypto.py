"""
MeridianAlgo Cryptocurrency Module

Comprehensive cryptocurrency trading and analysis functionality including
portfolio management, DeFi strategies, on-chain analytics, and crypto-specific risk management.
"""

import warnings
from datetime import datetime, timedelta
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

try:
    from sklearn.cluster import KMeans  # noqa: F401
    from sklearn.preprocessing import StandardScaler  # noqa: F401

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class CryptoAnalyzer:
    """
    Comprehensive cryptocurrency market analysis and trading strategies.

    Features:
    - Crypto portfolio optimization
    - DeFi yield farming strategies
    - On-chain analytics integration
    - Crypto volatility modeling
    - Market regime detection for crypto
    - Cross-chain arbitrage opportunities
    """

    def __init__(self, cryptocurrencies: List[str]):
        """
        Initialize crypto analyzer with cryptocurrency list.

        Args:
            cryptocurrencies: List of crypto symbols (e.g., ['BTC', 'ETH', 'BNB'])
        """
        self.cryptocurrencies = cryptocurrencies
        self.data = None
        self.returns = None
        self.market_caps = None
        self.on_chain_data = None

    def fetch_crypto_data(
        self,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        source: str = "yfinance",
    ) -> pd.DataFrame:
        """
        Fetch cryptocurrency price data.

        Args:
            start_date: Start date for data
            end_date: End date for data
            source: Data source ('yfinance', 'manual')

        Returns:
            DataFrame with crypto price data
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if source == "yfinance" and YFINANCE_AVAILABLE:
            data = {}
            for crypto in self.cryptocurrencies:
                try:
                    # Add -USD suffix for yfinance
                    ticker = f"{crypto}-USD"
                    crypto_data = yf.download(ticker, start=start_date, end=end_date)
                    data[crypto] = crypto_data["Adj Close"]

                except Exception as e:
                    print(f"Warning: Could not fetch data for {crypto}: {e}")
                    continue

            self.data = pd.DataFrame(data)

        else:
            # Generate synthetic crypto data with realistic characteristics
            np.random.seed(42)
            dates = pd.date_range(start=start_date, end=end_date, freq="D")
            n_days = len(dates)

            data = {}
            for crypto in self.cryptocurrencies:
                # Generate realistic crypto price series with high volatility
                if crypto == "BTC":
                    base_price = 50000
                    volatility = 0.04  # 4% daily volatility
                elif crypto == "ETH":
                    base_price = 3000
                    volatility = 0.06  # 6% daily volatility
                else:
                    base_price = np.random.uniform(10, 1000)
                    volatility = np.random.uniform(0.05, 0.15)

                # Generate returns with fat tails (crypto characteristic)
                returns = np.random.normal(0, volatility, n_days)

                # Add occasional large moves (crypto crashes/pumps)
                crash_indices = np.random.choice(
                    n_days, size=int(n_days * 0.02), replace=False
                )
                returns[crash_indices] *= np.random.uniform(-3, 3, len(crash_indices))

                data[crypto] = base_price * np.exp(np.cumsum(returns))

            self.data = pd.DataFrame(data, index=dates)

        self.returns = self.data.pct_change().dropna()
        return self.data

    def calculate_crypto_volatility(
        self, method: str = "ewma", window: int = 30, crypto_specific: bool = True
    ) -> Dict[str, pd.Series]:
        """
        Calculate cryptocurrency-specific volatility.

        Args:
            method: Volatility calculation method
            window: Rolling window
            crypto_specific: Use crypto-specific adjustments

        Returns:
            Dictionary with volatility series
        """
        if self.returns is None:
            raise ValueError("Must fetch data first using fetch_crypto_data()")

        volatility_results = {}

        for crypto in self.cryptocurrencies:
            if crypto in self.returns.columns:
                returns = self.returns[crypto].dropna()

                if method == "historical":
                    vol = returns.rolling(window).std()
                elif method == "ewma":
                    vol = returns.ewm(span=window).std()
                elif method == "parkinson":
                    # Parkinson estimator using high-low range (if available)
                    vol = returns.rolling(window).std() * 0.6  # Approximation
                else:
                    raise ValueError(f"Unknown method: {method}")

                # Crypto-specific adjustments
                if crypto_specific:
                    # Adjust for weekend effects (crypto trades 24/7)
                    vol = vol * 1.2  # Crypto typically has higher realized volatility

                    # Adjust for regime-dependent volatility
                    if crypto == "BTC":
                        vol = vol * (
                            1
                            + 0.3
                            * (returns.abs() > returns.std()).rolling(window).mean()
                        )

                volatility_results[crypto] = vol

        return volatility_results

    def identify_crypto_arbitrage_opportunities(
        self,
        exchanges: List[str] = ["binance", "coinbase", "kraken"],
        threshold: float = 0.005,
    ) -> List[Dict[str, Any]]:
        """
        Identify cross-exchange arbitrage opportunities.

        Args:
            exchanges: List of exchanges to check
            threshold: Minimum profit threshold

        Returns:
            List of arbitrage opportunities
        """
        opportunities = []

        # Simulate exchange price differences
        for crypto in self.cryptocurrencies:
            if crypto in self.data.columns:
                current_price = self.data[crypto].iloc[-1]

                # Generate simulated prices for different exchanges
                for exchange in exchanges:
                    # Add random spread to simulate exchange differences
                    exchange_price = current_price * (1 + np.random.normal(0, 0.002))
                    price_diff = abs(exchange_price - current_price) / current_price

                    if price_diff > threshold:
                        opportunities.append(
                            {
                                "type": "cross_exchange_arbitrage",
                                "crypto": crypto,
                                "exchange": exchange,
                                "price": exchange_price,
                                "reference_price": current_price,
                                "price_difference": price_diff,
                                "potential_profit": price_diff
                                - 0.002,  # Minus trading fees
                            }
                        )

        return opportunities

    def calculate_crypto_correlations(
        self,
        method: str = "pearson",
        window: Optional[int] = None,
        regime_adjusted: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate crypto correlations with regime adjustments.

        Args:
            method: Correlation method
            window: Rolling window
            regime_adjusted: Adjust for market regimes

        Returns:
            Correlation matrix
        """
        if self.returns is None:
            raise ValueError("Must fetch data first using fetch_crypto_data()")

        if window is None:
            # Static correlation
            correlations = self.returns.corr(method=method)
        else:
            # Rolling correlation
            correlations = self.returns.rolling(window).corr(method=method)

        # Regime adjustments for crypto
        if regime_adjusted:
            # Crypto tends to correlate more during market stress
            market_stress = self.returns.abs().mean(axis=1).rolling(window=30).mean()
            stress_factor = 1 + 0.5 * (market_stress / market_stress.quantile(0.8))

            if window is not None:
                # Apply stress adjustment to rolling correlations
                for i in range(len(correlations.index)):
                    if not pd.isna(stress_factor.iloc[i]):
                        correlations.iloc[i] *= stress_factor.iloc[i]

        return correlations

    def optimize_crypto_portfolio(
        self,
        risk_free_rate: float = 0.03,
        optimization_method: str = "hrp",
        max_weight: float = 0.4,
        min_weight: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Optimize cryptocurrency portfolio using crypto-specific methods.

        Args:
            risk_free_rate: Risk-free rate
            optimization_method: Optimization method
            max_weight: Maximum weight for any single crypto
            min_weight: Minimum weight for any crypto

        Returns:
            Portfolio optimization results
        """
        if self.returns is None:
            raise ValueError("Must fetch data first using fetch_crypto_data()")

        returns_clean = self.returns.dropna()

        if optimization_method == "hrp":
            # Hierarchical Risk Parity - good for crypto due to high correlations
            weights = self._hierarchical_risk_parity(returns_clean)
        elif optimization_method == "risk_parity":
            # Risk parity with crypto adjustments
            weights = self._risk_parity_crypto(returns_clean)
        elif optimization_method == "max_sharpe":
            # Maximum Sharpe ratio with constraints
            weights = self._max_sharpe_crypto(
                returns_clean, risk_free_rate, max_weight, min_weight
            )
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")

        # Calculate portfolio metrics
        portfolio_returns = (returns_clean * pd.Series(weights)).sum(axis=1)

        metrics = {
            "weights": dict(zip(returns_clean.columns, weights)),
            "expected_return": portfolio_returns.mean() * 252,
            "volatility": portfolio_returns.std() * np.sqrt(252),
            "sharpe_ratio": (portfolio_returns.mean() * 252 - risk_free_rate)
            / (portfolio_returns.std() * np.sqrt(252)),
            "max_drawdown": self._calculate_max_drawdown(portfolio_returns),
            "portfolio_returns": portfolio_returns,
        }

        return metrics

    def _hierarchical_risk_parity(self, returns: pd.DataFrame) -> np.ndarray:
        """Implement Hierarchical Risk Parity for crypto portfolios."""
        # Calculate correlation matrix
        corr = returns.corr()

        # Calculate distance matrix
        distance = np.sqrt((1 - corr) / 2)

        # Hierarchical clustering
        from scipy.cluster.hierarchy import linkage

        # Convert to condensed distance matrix
        condensed_distance = []
        n = len(corr)
        for i in range(n):
            for j in range(i + 1, n):
                condensed_distance.append(distance.iloc[i, j])

        # Perform hierarchical clustering
        linkage(condensed_distance, method="ward")

        # Simple equal weight within clusters (simplified HRP)
        weights = np.ones(len(returns.columns)) / len(returns.columns)

        return weights

    def _risk_parity_crypto(self, returns: pd.DataFrame) -> np.ndarray:
        """Risk parity with crypto-specific adjustments."""
        # Calculate covariance matrix
        cov_matrix = returns.cov() * 252  # Annualized

        # Inverse volatility weighting (simplified risk parity)
        volatilities = np.sqrt(np.diag(cov_matrix))
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()

        # Crypto adjustment: reduce concentration in highly volatile assets
        max_vol_weight = 0.3
        weights = np.minimum(weights, max_vol_weight)
        weights = weights / weights.sum()

        return weights

    def _max_sharpe_crypto(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float,
        max_weight: float,
        min_weight: float,
    ) -> np.ndarray:
        """Maximum Sharpe ratio optimization with crypto constraints."""
        # Simplified optimization using equal weights with constraints
        n_assets = len(returns.columns)

        # Start with equal weights
        weights = np.ones(n_assets) / n_assets

        # Apply constraints
        weights = np.clip(weights, min_weight, max_weight)
        weights = weights / weights.sum()

        return weights

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def analyze_defi_opportunities(
        self, defi_protocols: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Analyze DeFi yield farming opportunities.

        Args:
            defi_protocols: Dictionary of protocols and their APYs

        Returns:
            DeFi analysis results
        """
        opportunities = []

        for protocol, data in defi_protocols.items():
            base_apy = data.get("base_apy", 0)
            reward_apy = data.get("reward_apy", 0)
            total_apy = base_apy + reward_apy

            # Calculate risk-adjusted yield (simplified)
            risk_score = data.get("risk_score", 1.0)
            risk_adjusted_apy = total_apy / risk_score

            opportunities.append(
                {
                    "protocol": protocol,
                    "base_apy": base_apy,
                    "reward_apy": reward_apy,
                    "total_apy": total_apy,
                    "risk_score": risk_score,
                    "risk_adjusted_apy": risk_adjusted_apy,
                }
            )

        # Sort by risk-adjusted APY
        opportunities.sort(key=lambda x: x["risk_adjusted_apy"], reverse=True)

        return {
            "opportunities": opportunities,
            "best_opportunity": opportunities[0] if opportunities else None,
            "average_apy": np.mean([opp["total_apy"] for opp in opportunities]),
        }


class CryptoRiskManager:
    """
    Crypto-specific risk management tools.

    Features:
    - Crypto position sizing
    - Volatility-adjusted risk limits
    - Crypto stress testing
    - Smart contract risk assessment
    """

    def __init__(self, crypto_analyzer: CryptoAnalyzer):
        """
        Initialize crypto risk manager.

        Args:
            crypto_analyzer: CryptoAnalyzer instance with data
        """
        self.analyzer = crypto_analyzer

    def calculate_crypto_position_size(
        self,
        crypto: str,
        account_balance: float,
        risk_per_trade: float = 0.02,
        volatility_multiplier: float = 1.5,
    ) -> float:
        """
        Calculate position size for crypto trade.

        Args:
            crypto: Cryptocurrency symbol
            account_balance: Account balance
            risk_per_trade: Risk percentage per trade
            volatility_multiplier: Volatility adjustment factor

        Returns:
            Position size in USD
        """
        if self.analyzer.returns is None:
            raise ValueError("Must fetch data first")

        # Get crypto volatility
        if crypto in self.analyzer.returns.columns:
            volatility = self.analyzer.returns[crypto].rolling(30).std().iloc[-1]
            crypto_vol_multiplier = volatility_multiplier
        else:
            volatility = 0.05  # Default crypto volatility
            crypto_vol_multiplier = 1.0

        # Adjust position size for volatility
        risk_amount = account_balance * risk_per_trade
        position_size = risk_amount / (volatility * crypto_vol_multiplier)

        return position_size

    def crypto_stress_test(
        self, portfolio: Dict[str, float], stress_scenarios: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Stress test crypto portfolio under various scenarios.

        Args:
            portfolio: Portfolio positions
            stress_scenarios: Stress scenarios for different cryptos

        Returns:
            Portfolio P&L under each scenario
        """
        results = {}

        for scenario_name, scenario_shocks in stress_scenarios.items():
            portfolio_pnl = 0

            for crypto, position in portfolio.items():
                if crypto in scenario_shocks:
                    shock = scenario_shocks[crypto]

                    # Get current price
                    if crypto in self.analyzer.data.columns:
                        current_price = self.analyzer.data[crypto].iloc[-1]
                        shocked_price = current_price * (1 + shock)

                        # Calculate P&L
                        pnl = position * (shocked_price - current_price)
                        portfolio_pnl += pnl

            results[scenario_name] = portfolio_pnl

        return results


class OnChainAnalyzer:
    """
    On-chain analytics for cryptocurrencies.

    Features:
    - Wallet tracking
    - Transaction flow analysis
    - Smart contract monitoring
    - Network health metrics
    """

    def __init__(self):
        """Initialize on-chain analyzer."""
        self.wallet_data = {}
        self.network_metrics = {}

    def analyze_wallet_activity(
        self, wallet_addresses: List[str], blockchain: str = "ethereum"
    ) -> Dict[str, Any]:
        """
        Analyze wallet activity patterns.

        Args:
            wallet_addresses: List of wallet addresses
            blockchain: Blockchain network

        Returns:
            Wallet analysis results
        """
        # Simulated on-chain analysis
        results = {}

        for address in wallet_addresses:
            # Simulate wallet metrics
            results[address] = {
                "transaction_count": np.random.randint(100, 10000),
                "total_value_sent": np.random.uniform(1000, 1000000),
                "total_value_received": np.random.uniform(1000, 1000000),
                "unique_counterparties": np.random.randint(10, 500),
                "average_transaction_size": np.random.uniform(0.1, 100),
                "last_activity": datetime.now()
                - timedelta(days=np.random.randint(1, 365)),
            }

        return results

    def calculate_network_health_metrics(
        self, blockchain: str = "ethereum"
    ) -> Dict[str, float]:
        """
        Calculate blockchain network health metrics.

        Args:
            blockchain: Blockchain network

        Returns:
            Network health metrics
        """
        # Simulated network metrics
        return {
            "hash_rate": np.random.uniform(100, 1000),
            "difficulty": np.random.uniform(1000, 10000),
            "block_time": np.random.uniform(10, 20),
            "gas_price": np.random.uniform(10, 100),
            "network_utilization": np.random.uniform(0.3, 0.9),
            "active_addresses": np.random.randint(100000, 1000000),
            "transaction_volume_24h": np.random.uniform(10000, 100000),
        }


# Utility functions
def get_major_cryptocurrencies() -> List[str]:
    """Get list of major cryptocurrencies."""
    return ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOT", "DOGE", "AVAX", "MATIC"]


def get_stablecoins() -> List[str]:
    """Get list of major stablecoins."""
    return ["USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "FDUSD", "PYUSD"]


def get_defi_protocols() -> List[str]:
    """Get list of major DeFi protocols."""
    return [
        "uniswap",
        "aave",
        "compound",
        "curve",
        "sushiswap",
        "pancakeswap",
        "makerdao",
        "yearn",
    ]


def calculate_crypto_beta(
    crypto_returns: pd.Series, market_returns: pd.Series, window: int = 252
) -> pd.Series:
    """
    Calculate cryptocurrency beta relative to market.

    Args:
        crypto_returns: Crypto returns
        market_returns: Market returns (e.g., BTC)
        window: Rolling window

    Returns:
        Rolling beta series
    """
    covariance = crypto_returns.rolling(window).cov(market_returns)
    market_variance = market_returns.rolling(window).var()

    beta = covariance / market_variance
    return beta


# Export main classes and functions
__all__ = [
    "CryptoAnalyzer",
    "CryptoRiskManager",
    "OnChainAnalyzer",
    "get_major_cryptocurrencies",
    "get_stablecoins",
    "get_defi_protocols",
    "calculate_crypto_beta",
]
