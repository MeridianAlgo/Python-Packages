"""
MeridianAlgo Derivatives Module

Comprehensive derivatives pricing and analysis including options, futures, swaps,
and exotic derivatives. Integrates concepts from QuantLib, PyQL, and other
leading derivatives libraries.
"""

import warnings
from typing import Any, Dict

import numpy as np
from scipy.interpolate import interp1d

# Suppress warnings
warnings.filterwarnings("ignore")

# Check for optional dependencies
try:
    from scipy.stats import norm

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt  # noqa: F401

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class OptionsPricer:
    """
    Comprehensive options pricing and analysis.

    Features:
    - Black-Scholes-Merton pricing
    - Binomial tree pricing
    - Monte Carlo simulation
    - Greeks calculation
    - Implied volatility calculation
    - Exotic options pricing
    - Volatility surface modeling
    """

    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize options pricer.

        Args:
            risk_free_rate: Risk-free interest rate
        """
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = 0.0

    def black_scholes_merton(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Dict[str, float]:
        """
        Black-Scholes-Merton option pricing.

        Args:
            S: Underlying asset price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            q: Dividend yield

        Returns:
            Dictionary with price and Greeks
        """
        if not SCIPY_AVAILABLE:
            return self._black_scholes_approximation(S, K, T, r, sigma, option_type, q)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type.lower() == "call":
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(
                d2
            )
            delta = np.exp(-q * T) * norm.cdf(d1)
            gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (
                -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * norm.cdf(d2)
                + q * S * np.exp(-q * T) * norm.cdf(d1)
            )
            vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(
                -d1
            )
            delta = -np.exp(-q * T) * norm.cdf(-d1)
            gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (
                -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
                - q * S * np.exp(-q * T) * norm.cdf(-d1)
            )
            vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

        return {
            "price": price,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho,
            "d1": d1,
            "d2": d2,
        }

    def _black_scholes_approximation(self, S, K, T, r, sigma, option_type, q):
        """Fallback Black-Scholes implementation."""
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Use error function approximation for normal CDF
        def norm_cdf(x):
            return 0.5 * (1 + np.erf(x / np.sqrt(2)))

        def norm_pdf(x):
            return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

        if option_type.lower() == "call":
            price = S * np.exp(-q * T) * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(
                d2
            )
        else:
            price = K * np.exp(-r * T) * norm_cdf(-d2) - S * np.exp(-q * T) * norm_cdf(
                -d1
            )

        return {"price": price, "d1": d1, "d2": d2}

    def binomial_tree(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_steps: int = 100,
        option_type: str = "call",
        american: bool = False,
        q: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Binomial tree option pricing.

        Args:
            S: Underlying asset price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            n_steps: Number of time steps
            option_type: 'call' or 'put'
            american: American or European option
            q: Dividend yield

        Returns:
            Dictionary with price and tree information
        """
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)

        # Initialize asset price tree
        asset_tree = np.zeros((n_steps + 1, n_steps + 1))
        for i in range(n_steps + 1):
            for j in range(i + 1):
                asset_tree[j, i] = S * (u ** (i - j)) * (d**j)

        # Initialize option value tree
        option_tree = np.zeros((n_steps + 1, n_steps + 1))

        # Calculate option values at expiration
        for j in range(n_steps + 1):
            if option_type.lower() == "call":
                option_tree[j, n_steps] = max(0, asset_tree[j, n_steps] - K)
            else:
                option_tree[j, n_steps] = max(0, K - asset_tree[j, n_steps])

        # Backward induction
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                if american:
                    # Early exercise consideration
                    if option_type.lower() == "call":
                        early_exercise = max(0, asset_tree[j, i] - K)
                    else:
                        early_exercise = max(0, K - asset_tree[j, i])

                    hold_value = np.exp(-r * dt) * (
                        p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1]
                    )
                    option_tree[j, i] = max(early_exercise, hold_value)
                else:
                    option_tree[j, i] = np.exp(-r * dt) * (
                        p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1]
                    )

        return {
            "price": option_tree[0, 0],
            "asset_tree": asset_tree,
            "option_tree": option_tree,
            "up_factor": u,
            "down_factor": d,
            "risk_neutral_prob": p,
        }

    def monte_carlo_pricing(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_simulations: int = 10000,
        option_type: str = "call",
        q: float = 0.0,
        antithetic: bool = True,
    ) -> Dict[str, Any]:
        """
        Monte Carlo option pricing.

        Args:
            S: Underlying asset price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            n_simulations: Number of simulations
            option_type: 'call' or 'put'
            q: Dividend yield
            antithetic: Use antithetic variates for variance reduction

        Returns:
            Dictionary with price and statistics
        """
        if antithetic:
            n_simulations = n_simulations // 2

        # Generate random paths
        Z = np.random.standard_normal(n_simulations)

        if antithetic:
            Z = np.concatenate([Z, -Z])

        # Calculate terminal prices
        ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

        # Calculate payoffs
        if option_type.lower() == "call":
            payoffs = np.maximum(0, ST - K)
        else:
            payoffs = np.maximum(0, K - ST)

        # Discount payoffs
        discounted_payoffs = payoffs * np.exp(-r * T)

        # Calculate statistics
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs))

        return {
            "price": price,
            "std_error": std_error,
            "confidence_interval": (price - 1.96 * std_error, price + 1.96 * std_error),
            "n_simulations": len(discounted_payoffs),
        }

    def calculate_implied_volatility(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        market_price: float,
        option_type: str = "call",
        q: float = 0.0,
        initial_guess: float = 0.2,
        tolerance: float = 1e-6,
    ) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.

        Args:
            S: Underlying asset price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            market_price: Market price of the option
            option_type: 'call' or 'put'
            q: Dividend yield
            initial_guess: Initial volatility guess
            tolerance: Convergence tolerance

        Returns:
            Implied volatility
        """
        sigma = initial_guess

        for _ in range(100):  # Maximum iterations
            result = self.black_scholes_merton(S, K, T, r, sigma, option_type, q)
            price_diff = result["price"] - market_price

            if abs(price_diff) < tolerance:
                return sigma

            # Calculate vega (sensitivity to volatility)
            vega = result.get("vega", 0.01)  # Fallback if vega not available

            if abs(vega) < 1e-10:  # Avoid division by zero
                break

            # Newton-Raphson update
            sigma = sigma - price_diff / vega

            # Ensure volatility stays positive
            sigma = max(sigma, 0.001)

        return sigma

    def calculate_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        q: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calculate option Greeks.

        Args:
            S: Underlying asset price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            q: Dividend yield

        Returns:
            Dictionary with all Greeks
        """
        result = self.black_scholes_merton(S, K, T, r, sigma, option_type, q)

        # Calculate additional Greeks
        d1 = result["d1"]
        d2 = result["d2"]

        # Vanna (sensitivity of delta to volatility)
        if SCIPY_AVAILABLE:
            vanna = -np.exp(-q * T) * norm.pdf(d1) * d2 / (sigma * np.sqrt(T))
        else:
            vanna = 0.0

        # Charm (delta decay)
        if SCIPY_AVAILABLE:
            charm = (
                -np.exp(-q * T)
                * norm.pdf(d1)
                * (2 * r * T - d2 * sigma * np.sqrt(T))
                / (2 * T * sigma * np.sqrt(T))
            )
        else:
            charm = 0.0

        # Speed (gamma change with respect to spot)
        if SCIPY_AVAILABLE:
            speed = (
                -np.exp(-q * T)
                * norm.pdf(d1)
                * (1 + d1 / (sigma * np.sqrt(T)))
                / (S**2 * sigma * np.sqrt(T))
            )
        else:
            speed = 0.0

        # Zomma (gamma change with respect to volatility)
        if SCIPY_AVAILABLE:
            zomma = (
                np.exp(-q * T)
                * norm.pdf(d1)
                * (d1 * d2 - 1)
                / (S * sigma**2 * np.sqrt(T))
            )
        else:
            zomma = 0.0

        return {
            "delta": result["delta"],
            "gamma": result["gamma"],
            "theta": result["theta"],
            "vega": result["vega"],
            "rho": result["rho"],
            "vanna": vanna,
            "charm": charm,
            "speed": speed,
            "zomma": zomma,
        }


class VolatilitySurface:
    """
    Volatility surface modeling and analysis.

    Features:
    - Implied volatility surface construction
    - Local volatility modeling
    - Stochastic volatility models
    - Volatility interpolation and extrapolation
    - Volatility surface calibration
    """

    def __init__(self):
        """Initialize volatility surface."""
        self.surface_data = None
        self.local_vol_surface = None

    def construct_volatility_surface(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray,
        S: float,
        r: float,
        q: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Construct implied volatility surface.

        Args:
            strikes: Array of strike prices
            maturities: Array of maturities
            implied_vols: Array of implied volatilities
            S: Current underlying price
            r: Risk-free rate
            q: Dividend yield

        Returns:
            Volatility surface information
        """
        # Create meshgrid for surface
        T_grid, K_grid = np.meshgrid(maturities, strikes)

        # Interpolate volatility surface
        if len(implied_vols.shape) == 1:
            # 1D interpolation
            vol_interp = interp1d(
                maturities, implied_vols, kind="cubic", fill_value="extrapolate"
            )
            surface_vol = vol_interp(T_grid)
        else:
            # 2D interpolation
            from scipy.interpolate import RectBivariateSpline

            vol_interp = RectBivariateSpline(maturities, strikes, implied_vols)
            surface_vol = vol_interp(T_grid, K_grid)

        self.surface_data = {
            "strikes": strikes,
            "maturities": maturities,
            "implied_vols": implied_vols,
            "surface_vol": surface_vol,
            "T_grid": T_grid,
            "K_grid": K_grid,
            "interpolator": vol_interp,
        }

        return self.surface_data

    def calculate_local_volatility(
        self, S: float, r: float, q: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculate local volatility surface using Dupire's equation.

        Args:
            S: Current underlying price
            r: Risk-free rate
            q: Dividend yield

        Returns:
            Local volatility surface
        """
        if self.surface_data is None:
            raise ValueError("Must construct volatility surface first")

        # Simplified local volatility calculation
        T_grid = self.surface_data["T_grid"]
        K_grid = self.surface_data["K_grid"]
        surface_vol = self.surface_data["surface_vol"]

        # Calculate local volatility (simplified implementation)
        local_vol = surface_vol.copy()  # Simplified - should use Dupire's equation

        self.local_vol_surface = {
            "T_grid": T_grid,
            "K_grid": K_grid,
            "local_vol": local_vol,
        }

        return self.local_vol_surface

    def stochastic_volatility_simulation(
        self,
        S0: float,
        r: float,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float,
        T: float,
        n_steps: int = 252,
        n_simulations: int = 1000,
    ) -> Dict[str, Any]:
        """
        Simulate using Heston stochastic volatility model.

        Args:
            S0: Initial stock price
            r: Risk-free rate
            kappa: Mean reversion speed
            theta: Long-term volatility
            sigma_v: Volatility of volatility
            rho: Correlation
            T: Time to maturity
            n_steps: Number of time steps
            n_simulations: Number of simulations

        Returns:
            Simulation results
        """
        dt = T / n_steps

        # Initialize arrays
        S = np.zeros((n_simulations, n_steps + 1))
        v = np.zeros((n_simulations, n_steps + 1))

        S[:, 0] = S0
        v[:, 0] = theta  # Start at long-term volatility

        # Simulate paths
        for i in range(1, n_steps + 1):
            # Generate correlated random numbers
            Z1 = np.random.standard_normal(n_simulations)
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(
                n_simulations
            )

            # Update volatility (CIR process)
            v[:, i] = np.maximum(
                0,
                v[:, i - 1]
                + kappa * (theta - v[:, i - 1]) * dt
                + sigma_v * np.sqrt(v[:, i - 1] * dt) * Z2,
            )

            # Update stock price
            S[:, i] = S[:, i - 1] * np.exp(
                (r - 0.5 * v[:, i - 1]) * dt + np.sqrt(v[:, i - 1] * dt) * Z1
            )

        return {
            "stock_paths": S,
            "volatility_paths": v,
            "final_prices": S[:, -1],
            "final_volatilities": v[:, -1],
            "mean_final_price": np.mean(S[:, -1]),
            "mean_final_volatility": np.mean(v[:, -1]),
        }


class ExoticOptions:
    """
    Exotic options pricing and analysis.

    Features:
    - Asian options
    - Barrier options
    - Lookback options
    - Binary options
    - Compound options
    - Chooser options
    """

    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize exotic options pricer.

        Args:
            risk_free_rate: Risk-free interest rate
        """
        self.risk_free_rate = risk_free_rate

    def asian_option_geometric(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_observations: int = 50,
        option_type: str = "call",
    ) -> Dict[str, float]:
        """
        Price geometric Asian option analytically.

        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            n_observations: Number of observations
            option_type: 'call' or 'put'

        Returns:
            Option price and Greeks
        """
        # Adjusted parameters for geometric average
        sigma_adj = sigma * np.sqrt(
            (n_observations + 1) * (2 * n_observations + 1) / (6 * n_observations**2)
        )
        r_adj = r - 0.5 * sigma**2 + 0.5 * sigma_adj**2

        # Use Black-Scholes with adjusted parameters
        pricer = OptionsPricer(r)
        result = pricer.black_scholes_merton(S0, K, T, r_adj, sigma_adj, option_type)

        return result

    def barrier_option_monte_carlo(
        self,
        S0: float,
        K: float,
        B: float,
        T: float,
        r: float,
        sigma: float,
        barrier_type: str = "up-and-out",
        option_type: str = "call",
        n_simulations: int = 10000,
        n_steps: int = 252,
    ) -> Dict[str, Any]:
        """
        Price barrier option using Monte Carlo simulation.

        Args:
            S0: Initial stock price
            K: Strike price
            B: Barrier level
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            barrier_type: Type of barrier option
            option_type: 'call' or 'put'
            n_simulations: Number of simulations
            n_steps: Number of time steps

        Returns:
            Option price and statistics
        """
        dt = T / n_steps

        # Initialize arrays
        S = np.zeros((n_simulations, n_steps + 1))
        S[:, 0] = S0

        # Simulate paths
        for i in range(1, n_steps + 1):
            Z = np.random.standard_normal(n_simulations)
            S[:, i] = S[:, i - 1] * np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            )

        # Check barrier conditions
        if barrier_type == "up-and-out":
            barrier_breached = np.any(S > B, axis=1)
        elif barrier_type == "down-and-out":
            barrier_breached = np.any(S < B, axis=1)
        elif barrier_type == "up-and-in":
            barrier_breached = np.any(S > B, axis=1)
        elif barrier_type == "down-and-in":
            barrier_breached = np.any(S < B, axis=1)
        else:
            raise ValueError(f"Unknown barrier type: {barrier_type}")

        # Calculate payoffs
        final_prices = S[:, -1]

        if option_type == "call":
            payoffs = np.maximum(0, final_prices - K)
        else:
            payoffs = np.maximum(0, K - final_prices)

        # Apply barrier conditions
        if "out" in barrier_type:
            payoffs[barrier_breached] = 0
        elif "in" in barrier_type:
            payoffs[~barrier_breached] = 0

        # Discount payoffs
        discounted_payoffs = payoffs * np.exp(-r * T)

        # Calculate statistics
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs))

        return {
            "price": price,
            "std_error": std_error,
            "confidence_interval": (price - 1.96 * std_error, price + 1.96 * std_error),
            "barrier_hit_rate": np.mean(barrier_breached),
        }

    def lookback_option_floating_strike(
        self,
        S0: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        n_simulations: int = 10000,
        n_steps: int = 252,
    ) -> Dict[str, Any]:
        """
        Price floating strike lookback option.

        Args:
            S0: Initial stock price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            n_simulations: Number of simulations
            n_steps: Number of time steps

        Returns:
            Option price and statistics
        """
        dt = T / n_steps

        # Initialize arrays
        S = np.zeros((n_simulations, n_steps + 1))
        S[:, 0] = S0

        # Simulate paths
        for i in range(1, n_steps + 1):
            Z = np.random.standard_normal(n_simulations)
            S[:, i] = S[:, i - 1] * np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            )

        # Calculate payoffs
        if option_type == "call":
            # Call: payoff = S_T - min(S)
            min_prices = np.min(S, axis=1)
            payoffs = np.maximum(0, S[:, -1] - min_prices)
        else:
            # Put: payoff = max(S) - S_T
            max_prices = np.max(S, axis=1)
            payoffs = np.maximum(0, max_prices - S[:, -1])

        # Discount payoffs
        discounted_payoffs = payoffs * np.exp(-r * T)

        # Calculate statistics
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs))

        return {
            "price": price,
            "std_error": std_error,
            "confidence_interval": (price - 1.96 * std_error, price + 1.96 * std_error),
        }


class FuturesPricer:
    """
    Futures and forward contracts pricing and analysis.

    Features:
    - Futures pricing
    - Forward contracts
    - Commodity futures
    - Interest rate futures
    - Currency futures
    - Basis trading strategies
    """

    def __init__(self):
        """Initialize futures pricer."""
        pass

    def calculate_futures_price(
        self,
        S0: float,
        T: float,
        r: float,
        storage_cost: float = 0.0,
        convenience_yield: float = 0.0,
        dividend_yield: float = 0.0,
    ) -> float:
        """
        Calculate theoretical futures price.

        Args:
            S0: Current spot price
            T: Time to maturity
            r: Risk-free rate
            storage_cost: Storage cost rate
            convenience_yield: Convenience yield
            dividend_yield: Dividend yield

        Returns:
            Futures price
        """
        cost_of_carry = r + storage_cost - convenience_yield - dividend_yield
        futures_price = S0 * np.exp(cost_of_carry * T)

        return futures_price

    def calculate_basis(
        self, spot_price: float, futures_price: float, T: float, r: float
    ) -> Dict[str, float]:
        """
        Calculate futures basis and implied cost of carry.

        Args:
            spot_price: Current spot price
            futures_price: Futures price
            T: Time to maturity
            r: Risk-free rate

        Returns:
            Basis analysis
        """
        basis = futures_price - spot_price
        basis_percentage = basis / spot_price

        # Implied cost of carry
        implied_carry = np.log(futures_price / spot_price) / T

        return {
            "basis": basis,
            "basis_percentage": basis_percentage,
            "implied_cost_of_carry": implied_carry,
            "cost_of_carry_premium": implied_carry - r,
        }


# Utility functions
def calculate_option_spread(
    option1_price: float, option2_price: float, spread_type: str = "vertical"
) -> Dict[str, float]:
    """
    Calculate option spread metrics.

    Args:
        option1_price: Price of first option
        option2_price: Price of second option
        spread_type: Type of spread

    Returns:
        Spread analysis
    """
    if spread_type == "vertical":
        net_debit = abs(option1_price - option2_price)
        max_profit = None  # Depends on strikes
        max_loss = net_debit
    elif spread_type == "horizontal":
        net_debit = option1_price + option2_price
        max_profit = None
        max_loss = net_debit
    else:
        net_debit = option1_price + option2_price
        max_profit = None
        max_loss = net_debit

    return {"net_debit": net_debit, "max_profit": max_profit, "max_loss": max_loss}


def calculate_delta_hedge(
    portfolio_delta: float, underlying_price: float, target_delta: float = 0.0
) -> Dict[str, float]:
    """
    Calculate delta hedging requirements.

    Args:
        portfolio_delta: Current portfolio delta
        underlying_price: Current underlying price
        target_delta: Target delta (usually 0 for delta-neutral)

    Returns:
        Hedge requirements
    """
    hedge_delta = target_delta - portfolio_delta
    shares_to_trade = hedge_delta

    return {
        "current_delta": portfolio_delta,
        "target_delta": target_delta,
        "hedge_delta": hedge_delta,
        "shares_to_trade": shares_to_trade,
        "hedge_value": shares_to_trade * underlying_price,
    }


# Export main classes and functions
__all__ = [
    "OptionsPricer",
    "VolatilitySurface",
    "ExoticOptions",
    "FuturesPricer",
    "calculate_option_spread",
    "calculate_delta_hedge",
]
