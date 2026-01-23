"""
Comprehensive options pricing models with Greeks calculation.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class Option:
    """Option contract representation."""

    underlying_price: float
    strike_price: float
    time_to_expiry: float  # Years
    risk_free_rate: float
    volatility: float
    option_type: str = "call"  # "call" or "put"
    style: str = "european"  # "european" or "american"
    dividend_yield: float = 0.0

    def __post_init__(self):
        if self.option_type not in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")
        if self.style not in ["european", "american"]:
            raise ValueError("style must be 'european' or 'american'")


@dataclass
class Greeks:
    """Option Greeks container."""

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
        }


class BlackScholesModel:
    """Black-Scholes option pricing model."""

    @staticmethod
    def price_option(option: Option) -> Dict[str, float]:
        """Price European option using Black-Scholes formula."""

        S = option.underlying_price
        K = option.strike_price
        T = option.time_to_expiry
        r = option.risk_free_rate
        sigma = option.volatility
        q = option.dividend_yield

        if T <= 0:
            # Option has expired
            if option.option_type == "call":
                return {"price": max(S - K, 0), "intrinsic_value": max(S - K, 0)}
            else:
                return {"price": max(K - S, 0), "intrinsic_value": max(K - S, 0)}

        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Calculate option price
        if option.option_type == "call":
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(
                d2
            )
            intrinsic_value = max(S - K, 0)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(
                -d1
            )
            intrinsic_value = max(K - S, 0)

        time_value = price - intrinsic_value

        return {
            "price": price,
            "intrinsic_value": intrinsic_value,
            "time_value": time_value,
            "d1": d1,
            "d2": d2,
        }

    @staticmethod
    def calculate_greeks(option: Option) -> Greeks:
        """Calculate all Greeks for the option."""

        S = option.underlying_price
        K = option.strike_price
        T = option.time_to_expiry
        r = option.risk_free_rate
        sigma = option.volatility
        q = option.dividend_yield

        if T <= 0:
            return Greeks(0, 0, 0, 0, 0)

        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Delta
        if option.option_type == "call":
            delta = np.exp(-q * T) * norm.cdf(d1)
        else:
            delta = -np.exp(-q * T) * norm.cdf(-d1)

        # Gamma (same for calls and puts)
        gamma = (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))

        # Theta
        term1 = -(S * norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))
        if option.option_type == "call":
            term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
            term3 = q * S * np.exp(-q * T) * norm.cdf(d1)
            theta = (term1 - term2 + term3) / 365  # Per day
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            term3 = q * S * np.exp(-q * T) * norm.cdf(-d1)
            theta = (term1 + term2 - term3) / 365  # Per day

        # Vega (same for calls and puts)
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% vol change

        # Rho
        if option.option_type == "call":
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Per 1% rate change
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100  # Per 1% rate change

        return Greeks(delta, gamma, theta, vega, rho)

    @staticmethod
    def implied_volatility(option: Option, market_price: float) -> float:
        """Calculate implied volatility from market price."""

        def objective(vol):
            option_copy = Option(
                underlying_price=option.underlying_price,
                strike_price=option.strike_price,
                time_to_expiry=option.time_to_expiry,
                risk_free_rate=option.risk_free_rate,
                volatility=vol,
                option_type=option.option_type,
                dividend_yield=option.dividend_yield,
            )
            theoretical_price = BlackScholesModel.price_option(option_copy)["price"]
            return (theoretical_price - market_price) ** 2

        try:
            result = minimize_scalar(objective, bounds=(0.001, 5.0), method="bounded")
            return result.x
        except Exception:
            logger.warning("Implied volatility calculation failed")
            return 0.2  # Default 20% volatility


class BinomialTreeModel:
    """Binomial tree model for American options."""

    def __init__(self, steps: int = 100):
        self.steps = steps

    def price_option(self, option: Option) -> Dict[str, float]:
        """Price option using binomial tree."""

        S = option.underlying_price
        K = option.strike_price
        T = option.time_to_expiry
        r = option.risk_free_rate
        sigma = option.volatility
        q = option.dividend_yield
        n = self.steps

        if T <= 0:
            if option.option_type == "call":
                return {"price": max(S - K, 0)}
            else:
                return {"price": max(K - S, 0)}

        # Calculate parameters
        dt = T / n
        u = np.exp(sigma * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp((r - q) * dt) - d) / (u - d)  # Risk-neutral probability

        # Initialize asset prices at maturity
        asset_prices = np.zeros(n + 1)
        for i in range(n + 1):
            asset_prices[i] = S * (u ** (n - i)) * (d**i)

        # Initialize option values at maturity
        option_values = np.zeros(n + 1)
        for i in range(n + 1):
            if option.option_type == "call":
                option_values[i] = max(asset_prices[i] - K, 0)
            else:
                option_values[i] = max(K - asset_prices[i], 0)

        # Backward induction
        for j in range(n - 1, -1, -1):
            for i in range(j + 1):
                # Calculate continuation value
                continuation_value = np.exp(-r * dt) * (
                    p * option_values[i] + (1 - p) * option_values[i + 1]
                )

                # Calculate intrinsic value
                current_price = S * (u ** (j - i)) * (d**i)
                if option.option_type == "call":
                    intrinsic_value = max(current_price - K, 0)
                else:
                    intrinsic_value = max(K - current_price, 0)

                # For American options, take maximum of continuation and intrinsic
                if option.style == "american":
                    option_values[i] = max(continuation_value, intrinsic_value)
                else:
                    option_values[i] = continuation_value

        return {"price": option_values[0]}

    def calculate_greeks(self, option: Option) -> Greeks:
        """Calculate Greeks using finite differences."""

        # Base price
        base_price = self.price_option(option)["price"]

        # Delta (finite difference)
        h = 0.01 * option.underlying_price
        option_up = Option(
            underlying_price=option.underlying_price + h,
            strike_price=option.strike_price,
            time_to_expiry=option.time_to_expiry,
            risk_free_rate=option.risk_free_rate,
            volatility=option.volatility,
            option_type=option.option_type,
            style=option.style,
            dividend_yield=option.dividend_yield,
        )
        option_down = Option(
            underlying_price=option.underlying_price - h,
            strike_price=option.strike_price,
            time_to_expiry=option.time_to_expiry,
            risk_free_rate=option.risk_free_rate,
            volatility=option.volatility,
            option_type=option.option_type,
            style=option.style,
            dividend_yield=option.dividend_yield,
        )

        price_up = self.price_option(option_up)["price"]
        price_down = self.price_option(option_down)["price"]
        delta = (price_up - price_down) / (2 * h)

        # Gamma
        gamma = (price_up - 2 * base_price + price_down) / (h**2)

        # Theta
        h_time = 1 / 365  # One day
        if option.time_to_expiry > h_time:
            option_theta = Option(
                underlying_price=option.underlying_price,
                strike_price=option.strike_price,
                time_to_expiry=option.time_to_expiry - h_time,
                risk_free_rate=option.risk_free_rate,
                volatility=option.volatility,
                option_type=option.option_type,
                style=option.style,
                dividend_yield=option.dividend_yield,
            )
            price_theta = self.price_option(option_theta)["price"]
            theta = price_theta - base_price  # Already per day
        else:
            theta = 0

        # Vega
        h_vol = 0.01  # 1% volatility
        option_vega = Option(
            underlying_price=option.underlying_price,
            strike_price=option.strike_price,
            time_to_expiry=option.time_to_expiry,
            risk_free_rate=option.risk_free_rate,
            volatility=option.volatility + h_vol,
            option_type=option.option_type,
            style=option.style,
            dividend_yield=option.dividend_yield,
        )
        price_vega = self.price_option(option_vega)["price"]
        vega = price_vega - base_price

        # Rho
        h_rate = 0.01  # 1% rate
        option_rho = Option(
            underlying_price=option.underlying_price,
            strike_price=option.strike_price,
            time_to_expiry=option.time_to_expiry,
            risk_free_rate=option.risk_free_rate + h_rate,
            volatility=option.volatility,
            option_type=option.option_type,
            style=option.style,
            dividend_yield=option.dividend_yield,
        )
        price_rho = self.price_option(option_rho)["price"]
        rho = price_rho - base_price

        return Greeks(delta, gamma, theta, vega, rho)


class MonteCarloModel:
    """Monte Carlo simulation for path-dependent options."""

    def __init__(self, n_simulations: int = 100000, n_steps: int = 252):
        self.n_simulations = n_simulations
        self.n_steps = n_steps

    def price_european_option(self, option: Option) -> Dict[str, float]:
        """Price European option using Monte Carlo."""

        S = option.underlying_price
        K = option.strike_price
        T = option.time_to_expiry
        r = option.risk_free_rate
        sigma = option.volatility
        q = option.dividend_yield

        if T <= 0:
            if option.option_type == "call":
                return {"price": max(S - K, 0)}
            else:
                return {"price": max(K - S, 0)}

        # Generate random paths
        dt = T / self.n_steps
        payoffs = []

        for _ in range(self.n_simulations):
            # Generate path
            path = [S]
            for _ in range(self.n_steps):
                dW = np.random.normal(0, np.sqrt(dt))
                S_next = path[-1] * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * dW)
                path.append(S_next)

            # Calculate payoff
            final_price = path[-1]
            if option.option_type == "call":
                payoff = max(final_price - K, 0)
            else:
                payoff = max(K - final_price, 0)

            payoffs.append(payoff)

        # Discount expected payoff
        option_price = np.exp(-r * T) * np.mean(payoffs)
        standard_error = np.std(payoffs) / np.sqrt(self.n_simulations)

        return {
            "price": option_price,
            "standard_error": standard_error,
            "confidence_interval": (
                option_price - 1.96 * standard_error,
                option_price + 1.96 * standard_error,
            ),
        }

    def price_asian_option(
        self, option: Option, average_type: str = "arithmetic"
    ) -> Dict[str, float]:
        """Price Asian option (average price option)."""

        S = option.underlying_price
        K = option.strike_price
        T = option.time_to_expiry
        r = option.risk_free_rate
        sigma = option.volatility
        q = option.dividend_yield

        dt = T / self.n_steps
        payoffs = []

        for _ in range(self.n_simulations):
            # Generate path
            path = [S]
            for _ in range(self.n_steps):
                dW = np.random.normal(0, np.sqrt(dt))
                S_next = path[-1] * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * dW)
                path.append(S_next)

            # Calculate average
            if average_type == "arithmetic":
                average_price = np.mean(path)
            else:  # geometric
                average_price = np.exp(np.mean(np.log(path)))

            # Calculate payoff
            if option.option_type == "call":
                payoff = max(average_price - K, 0)
            else:
                payoff = max(K - average_price, 0)

            payoffs.append(payoff)

        option_price = np.exp(-r * T) * np.mean(payoffs)
        standard_error = np.std(payoffs) / np.sqrt(self.n_simulations)

        return {
            "price": option_price,
            "standard_error": standard_error,
            "average_type": average_type,
        }

    def price_barrier_option(
        self, option: Option, barrier: float, barrier_type: str = "up_and_out"
    ) -> Dict[str, float]:
        """Price barrier option."""

        S = option.underlying_price
        K = option.strike_price
        T = option.time_to_expiry
        r = option.risk_free_rate
        sigma = option.volatility
        q = option.dividend_yield

        dt = T / self.n_steps
        payoffs = []

        for _ in range(self.n_simulations):
            # Generate path
            path = [S]
            barrier_hit = False

            for _ in range(self.n_steps):
                dW = np.random.normal(0, np.sqrt(dt))
                S_next = path[-1] * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * dW)
                path.append(S_next)

                # Check barrier condition
                if barrier_type == "up_and_out" and S_next >= barrier:
                    barrier_hit = True
                elif barrier_type == "down_and_out" and S_next <= barrier:
                    barrier_hit = True
                elif barrier_type == "up_and_in" and S_next >= barrier:
                    barrier_hit = True
                elif barrier_type == "down_and_in" and S_next <= barrier:
                    barrier_hit = True

            # Calculate payoff based on barrier type
            final_price = path[-1]
            if option.option_type == "call":
                vanilla_payoff = max(final_price - K, 0)
            else:
                vanilla_payoff = max(K - final_price, 0)

            if barrier_type in ["up_and_out", "down_and_out"]:
                payoff = 0 if barrier_hit else vanilla_payoff
            else:  # knock-in
                payoff = vanilla_payoff if barrier_hit else 0

            payoffs.append(payoff)

        option_price = np.exp(-r * T) * np.mean(payoffs)
        standard_error = np.std(payoffs) / np.sqrt(self.n_simulations)

        return {
            "price": option_price,
            "standard_error": standard_error,
            "barrier": barrier,
            "barrier_type": barrier_type,
        }


class OptionPortfolio:
    """Option portfolio analytics."""

    def __init__(self):
        self.positions: List[
            Tuple[Option, float, str]
        ] = []  # (option, quantity, model)

    def add_position(
        self, option: Option, quantity: float, model: str = "black_scholes"
    ):
        """Add option position to portfolio."""
        self.positions.append((option, quantity, model))

    def calculate_portfolio_greeks(self) -> Greeks:
        """Calculate portfolio-level Greeks."""

        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        total_rho = 0

        for option, quantity, model in self.positions:
            if model == "black_scholes":
                greeks = BlackScholesModel.calculate_greeks(option)
            elif model == "binomial":
                tree_model = BinomialTreeModel()
                greeks = tree_model.calculate_greeks(option)
            else:
                continue

            total_delta += quantity * greeks.delta
            total_gamma += quantity * greeks.gamma
            total_theta += quantity * greeks.theta
            total_vega += quantity * greeks.vega
            total_rho += quantity * greeks.rho

        return Greeks(total_delta, total_gamma, total_theta, total_vega, total_rho)

    def calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value."""

        total_value = 0

        for option, quantity, model in self.positions:
            if model == "black_scholes":
                price = BlackScholesModel.price_option(option)["price"]
            elif model == "binomial":
                tree_model = BinomialTreeModel()
                price = tree_model.price_option(option)["price"]
            elif model == "monte_carlo":
                mc_model = MonteCarloModel()
                price = mc_model.price_european_option(option)["price"]
            else:
                continue

            total_value += quantity * price

        return total_value

    def scenario_analysis(
        self, spot_changes: List[float], vol_changes: List[float] = None
    ) -> pd.DataFrame:
        """Perform scenario analysis on portfolio."""

        if vol_changes is None:
            vol_changes = [0]

        results = []
        base_value = self.calculate_portfolio_value()

        for spot_change in spot_changes:
            for vol_change in vol_changes:
                # Create scenario portfolio
                scenario_portfolio = OptionPortfolio()

                for option, quantity, model in self.positions:
                    scenario_option = Option(
                        underlying_price=option.underlying_price * (1 + spot_change),
                        strike_price=option.strike_price,
                        time_to_expiry=option.time_to_expiry,
                        risk_free_rate=option.risk_free_rate,
                        volatility=option.volatility + vol_change,
                        option_type=option.option_type,
                        style=option.style,
                        dividend_yield=option.dividend_yield,
                    )
                    scenario_portfolio.add_position(scenario_option, quantity, model)

                scenario_value = scenario_portfolio.calculate_portfolio_value()
                pnl = scenario_value - base_value

                results.append(
                    {
                        "spot_change": spot_change,
                        "vol_change": vol_change,
                        "portfolio_value": scenario_value,
                        "pnl": pnl,
                        "pnl_pct": pnl / base_value if base_value != 0 else 0,
                    }
                )

        return pd.DataFrame(results)


class FiniteDifferenceModel:
    """Finite difference methods for option pricing."""

    def __init__(self, n_space: int = 100, n_time: int = 100):
        self.n_space = n_space
        self.n_time = n_time

    def price_option_explicit(
        self, option: Option, S_max: float = None
    ) -> Dict[str, float]:
        """Price option using explicit finite difference method."""

        if S_max is None:
            S_max = 3 * option.strike_price

        S = option.underlying_price
        K = option.strike_price
        T = option.time_to_expiry
        r = option.risk_free_rate
        sigma = option.volatility
        q = option.dividend_yield

        # Grid setup
        dS = S_max / self.n_space
        dt = T / self.n_time

        # Check stability condition
        alpha = 0.5 * sigma**2 * dt / dS**2
        if alpha > 0.5:
            warnings.warn(
                "Explicit scheme may be unstable. Consider using implicit method."
            )

        # Initialize grids
        S_grid = np.linspace(0, S_max, self.n_space + 1)
        V = np.zeros((self.n_time + 1, self.n_space + 1))

        # Boundary conditions at maturity
        for i in range(self.n_space + 1):
            if option.option_type == "call":
                V[self.n_time, i] = max(S_grid[i] - K, 0)
            else:
                V[self.n_time, i] = max(K - S_grid[i], 0)

        # Backward time stepping
        for j in range(self.n_time - 1, -1, -1):
            for i in range(1, self.n_space):
                # Finite difference coefficients
                a = 0.5 * dt * (sigma**2 * i**2 - (r - q) * i)
                b = 1 - dt * (sigma**2 * i**2 + r)
                c = 0.5 * dt * (sigma**2 * i**2 + (r - q) * i)

                V[j, i] = a * V[j + 1, i - 1] + b * V[j + 1, i] + c * V[j + 1, i + 1]

                # American option early exercise
                if option.style == "american":
                    if option.option_type == "call":
                        V[j, i] = max(V[j, i], S_grid[i] - K)
                    else:
                        V[j, i] = max(V[j, i], K - S_grid[i])

            # Boundary conditions
            if option.option_type == "call":
                V[j, 0] = 0  # S = 0
                V[j, self.n_space] = S_max - K * np.exp(-r * (T - j * dt))  # S = S_max
            else:
                V[j, 0] = K * np.exp(-r * (T - j * dt))  # S = 0
                V[j, self.n_space] = 0  # S = S_max

        # Interpolate to get option price at current spot
        price = np.interp(S, S_grid, V[0, :])

        return {"price": price, "grid": V, "S_grid": S_grid}

    def price_option_implicit(
        self, option: Option, S_max: float = None
    ) -> Dict[str, float]:
        """Price option using implicit finite difference method (Crank-Nicolson)."""

        if S_max is None:
            S_max = 3 * option.strike_price

        S = option.underlying_price
        K = option.strike_price
        T = option.time_to_expiry
        r = option.risk_free_rate
        sigma = option.volatility
        q = option.dividend_yield

        # Grid setup
        S_max / self.n_space
        dt = T / self.n_time

        # Initialize grids
        S_grid = np.linspace(0, S_max, self.n_space + 1)
        V = np.zeros((self.n_time + 1, self.n_space + 1))

        # Boundary conditions at maturity
        for i in range(self.n_space + 1):
            if option.option_type == "call":
                V[self.n_time, i] = max(S_grid[i] - K, 0)
            else:
                V[self.n_time, i] = max(K - S_grid[i], 0)

        # Set up tridiagonal matrix for implicit scheme
        alpha = np.zeros(self.n_space - 1)
        beta = np.zeros(self.n_space - 1)
        gamma = np.zeros(self.n_space - 1)

        for i in range(1, self.n_space):
            alpha[i - 1] = -0.25 * dt * (sigma**2 * i**2 - (r - q) * i)
            beta[i - 1] = 1 + 0.5 * dt * (sigma**2 * i**2 + r)
            gamma[i - 1] = -0.25 * dt * (sigma**2 * i**2 + (r - q) * i)

        # Backward time stepping
        for j in range(self.n_time - 1, -1, -1):
            # Right-hand side
            rhs = np.zeros(self.n_space - 1)
            for i in range(1, self.n_space):
                rhs[i - 1] = V[j + 1, i] + 0.25 * dt * (
                    (sigma**2 * i**2 - (r - q) * i) * V[j + 1, i - 1]
                    + (-(sigma**2) * i**2 - r) * V[j + 1, i]
                    + (sigma**2 * i**2 + (r - q) * i) * V[j + 1, i + 1]
                )

            # Boundary conditions
            if option.option_type == "call":
                rhs[0] -= alpha[0] * 0  # V[j, 0] = 0
                rhs[-1] -= gamma[-1] * (S_max - K * np.exp(-r * (T - j * dt)))
            else:
                rhs[0] -= alpha[0] * (K * np.exp(-r * (T - j * dt)))
                rhs[-1] -= gamma[-1] * 0  # V[j, n_space] = 0

            # Solve tridiagonal system
            V[j, 1 : self.n_space] = self._solve_tridiagonal(alpha, beta, gamma, rhs)

            # Set boundary conditions
            if option.option_type == "call":
                V[j, 0] = 0
                V[j, self.n_space] = S_max - K * np.exp(-r * (T - j * dt))
            else:
                V[j, 0] = K * np.exp(-r * (T - j * dt))
                V[j, self.n_space] = 0

            # American option early exercise
            if option.style == "american":
                for i in range(1, self.n_space):
                    if option.option_type == "call":
                        V[j, i] = max(V[j, i], S_grid[i] - K)
                    else:
                        V[j, i] = max(V[j, i], K - S_grid[i])

        # Interpolate to get option price at current spot
        price = np.interp(S, S_grid, V[0, :])

        return {"price": price, "grid": V, "S_grid": S_grid}

    def _solve_tridiagonal(
        self, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
    ) -> np.ndarray:
        """Solve tridiagonal system using Thomas algorithm."""
        n = len(d)
        c_prime = np.zeros(n - 1)
        d_prime = np.zeros(n)

        # Forward sweep
        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]

        for i in range(1, n - 1):
            c_prime[i] = c[i] / (b[i] - a[i] * c_prime[i - 1])

        for i in range(1, n):
            d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / (b[i] - a[i] * c_prime[i - 1])

        # Back substitution
        x = np.zeros(n)
        x[n - 1] = d_prime[n - 1]

        for i in range(n - 2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i + 1]

        return x


class ExoticOptions:
    """Pricing models for exotic options."""

    @staticmethod
    def price_lookback_option(
        option: Option, lookback_type: str = "floating_strike"
    ) -> Dict[str, float]:
        """Price lookback option using Monte Carlo."""

        S = option.underlying_price
        K = option.strike_price
        T = option.time_to_expiry
        r = option.risk_free_rate
        sigma = option.volatility
        q = option.dividend_yield

        n_simulations = 100000
        n_steps = 252
        dt = T / n_steps

        payoffs = []

        for _ in range(n_simulations):
            # Generate path
            path = [S]
            for _ in range(n_steps):
                dW = np.random.normal(0, np.sqrt(dt))
                S_next = path[-1] * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * dW)
                path.append(S_next)

            path = np.array(path)

            if lookback_type == "floating_strike":
                if option.option_type == "call":
                    payoff = max(path[-1] - path.min(), 0)
                else:
                    payoff = max(path.max() - path[-1], 0)
            else:  # fixed_strike
                if option.option_type == "call":
                    payoff = max(path.max() - K, 0)
                else:
                    payoff = max(K - path.min(), 0)

            payoffs.append(payoff)

        option_price = np.exp(-r * T) * np.mean(payoffs)
        standard_error = np.std(payoffs) / np.sqrt(n_simulations)

        return {
            "price": option_price,
            "standard_error": standard_error,
            "lookback_type": lookback_type,
        }

    @staticmethod
    def price_rainbow_option(
        options: List[Option], correlation: float = 0.5, option_type: str = "max"
    ) -> Dict[str, float]:
        """Price rainbow option on multiple underlyings."""

        if len(options) != 2:
            raise ValueError("Currently supports only 2-asset rainbow options")

        option1, option2 = options
        S1, S2 = option1.underlying_price, option2.underlying_price
        K = option1.strike_price  # Assume same strike
        T = option1.time_to_expiry
        r = option1.risk_free_rate
        sigma1, sigma2 = option1.volatility, option2.volatility

        n_simulations = 100000
        n_steps = 252
        dt = T / n_steps

        payoffs = []

        for _ in range(n_simulations):
            # Generate correlated random walks
            path1, path2 = [S1], [S2]

            for _ in range(n_steps):
                # Generate correlated random numbers
                z1 = np.random.normal(0, 1)
                z2 = correlation * z1 + np.sqrt(1 - correlation**2) * np.random.normal(
                    0, 1
                )

                dW1 = z1 * np.sqrt(dt)
                dW2 = z2 * np.sqrt(dt)

                S1_next = path1[-1] * np.exp((r - 0.5 * sigma1**2) * dt + sigma1 * dW1)
                S2_next = path2[-1] * np.exp((r - 0.5 * sigma2**2) * dt + sigma2 * dW2)

                path1.append(S1_next)
                path2.append(S2_next)

            # Calculate payoff
            final_S1, final_S2 = path1[-1], path2[-1]

            if option_type == "max":
                payoff = max(max(final_S1, final_S2) - K, 0)
            elif option_type == "min":
                payoff = max(min(final_S1, final_S2) - K, 0)
            elif option_type == "spread":
                payoff = max(final_S1 - final_S2 - K, 0)
            else:
                raise ValueError(f"Unknown rainbow option type: {option_type}")

            payoffs.append(payoff)

        option_price = np.exp(-r * T) * np.mean(payoffs)
        standard_error = np.std(payoffs) / np.sqrt(n_simulations)

        return {
            "price": option_price,
            "standard_error": standard_error,
            "option_type": option_type,
            "correlation": correlation,
        }

    @staticmethod
    def price_compound_option(
        option: Option, underlying_option: Option
    ) -> Dict[str, float]:
        """Price compound option (option on option)."""

        # This is a simplified implementation
        # In practice, would use more sophisticated methods

        S = option.underlying_price
        K1 = option.strike_price  # Strike of compound option
        K2 = underlying_option.strike_price  # Strike of underlying option
        T1 = option.time_to_expiry  # Expiry of compound option
        T2 = underlying_option.time_to_expiry  # Expiry of underlying option
        r = option.risk_free_rate
        sigma = option.volatility

        if T1 >= T2:
            raise ValueError("Compound option must expire before underlying option")

        # At T1, the compound option holder decides whether to exercise
        # This gives them the underlying option worth V(S(T1), T2-T1)

        n_simulations = 100000
        payoffs = []

        for _ in range(n_simulations):
            # Simulate price at T1
            S_T1 = S * np.exp(
                (r - 0.5 * sigma**2) * T1 + sigma * np.sqrt(T1) * np.random.normal()
            )

            # Value of underlying option at T1
            remaining_option = Option(
                underlying_price=S_T1,
                strike_price=K2,
                time_to_expiry=T2 - T1,
                risk_free_rate=r,
                volatility=sigma,
                option_type=underlying_option.option_type,
            )

            underlying_value = BlackScholesModel.price_option(remaining_option)["price"]

            # Payoff of compound option
            if option.option_type == "call":
                payoff = max(underlying_value - K1, 0)
            else:
                payoff = max(K1 - underlying_value, 0)

            payoffs.append(payoff)

        option_price = np.exp(-r * T1) * np.mean(payoffs)
        standard_error = np.std(payoffs) / np.sqrt(n_simulations)

        return {"price": option_price, "standard_error": standard_error}


class VolatilitySurface:
    """Implied volatility surface construction and analysis."""

    def __init__(self):
        self.data_points: List[
            Tuple[float, float, float, float]
        ] = []  # (strike, expiry, price, vol)
        self.surface = None

    def add_market_data(
        self,
        strike: float,
        expiry: float,
        market_price: float,
        underlying_price: float,
        risk_free_rate: float,
        option_type: str = "call",
    ):
        """Add market data point and calculate implied volatility."""

        option = Option(
            underlying_price=underlying_price,
            strike_price=strike,
            time_to_expiry=expiry,
            risk_free_rate=risk_free_rate,
            volatility=0.2,  # Initial guess
            option_type=option_type,
        )

        implied_vol = BlackScholesModel.implied_volatility(option, market_price)
        self.data_points.append((strike, expiry, market_price, implied_vol))

    def build_surface(self, method: str = "rbf") -> None:
        """Build volatility surface using interpolation."""

        if len(self.data_points) < 4:
            raise ValueError("Need at least 4 data points to build surface")

        strikes = [point[0] for point in self.data_points]
        expiries = [point[1] for point in self.data_points]
        vols = [point[3] for point in self.data_points]

        if method == "rbf":
            try:
                from scipy.interpolate import Rbf

                self.surface = Rbf(strikes, expiries, vols, function="multiquadric")
            except ImportError:
                # Fallback to linear interpolation
                from scipy.interpolate import LinearNDInterpolator

                points = list(zip(strikes, expiries))
                self.surface = LinearNDInterpolator(points, vols)

        elif method == "linear":
            from scipy.interpolate import LinearNDInterpolator

            points = list(zip(strikes, expiries))
            self.surface = LinearNDInterpolator(points, vols)

    def get_volatility(self, strike: float, expiry: float) -> float:
        """Get interpolated volatility for given strike and expiry."""

        if self.surface is None:
            self.build_surface()

        vol = self.surface(strike, expiry)

        # Handle extrapolation
        if np.isnan(vol):
            # Use nearest neighbor as fallback
            distances = [
                (abs(s - strike) + abs(e - expiry), v)
                for s, e, _, v in self.data_points
            ]
            distances.sort()
            vol = distances[0][1]

        return float(vol)

    def calculate_skew(self, expiry: float, atm_strike: float) -> Dict[str, float]:
        """Calculate volatility skew metrics."""

        # Calculate volatilities at different moneyness levels
        strikes = [
            0.8 * atm_strike,
            0.9 * atm_strike,
            atm_strike,
            1.1 * atm_strike,
            1.2 * atm_strike,
        ]

        vols = [self.get_volatility(strike, expiry) for strike in strikes]

        # Skew metrics
        atm_vol = vols[2]
        put_skew = vols[0] - atm_vol  # 20% OTM put vs ATM
        call_skew = vols[4] - atm_vol  # 20% OTM call vs ATM

        return {
            "atm_volatility": atm_vol,
            "put_skew": put_skew,
            "call_skew": call_skew,
            "skew_slope": (vols[4] - vols[0]) / (strikes[4] - strikes[0]),
        }


class OptionStrategy:
    """Common option trading strategies."""

    def __init__(self):
        self.legs: List[Tuple[Option, float, str]] = []  # (option, quantity, action)

    def add_leg(self, option: Option, quantity: float, action: str = "buy"):
        """Add leg to strategy."""
        self.legs.append((option, quantity, action))

    def calculate_payoff_diagram(self, spot_range: np.ndarray) -> pd.DataFrame:
        """Calculate payoff diagram for the strategy."""

        payoffs = []

        for spot in spot_range:
            total_payoff = 0

            for option, quantity, action in self.legs:
                # Calculate option value at expiry
                if option.option_type == "call":
                    option_payoff = max(spot - option.strike_price, 0)
                else:
                    option_payoff = max(option.strike_price - spot, 0)

                # Account for premium paid/received
                premium = BlackScholesModel.price_option(option)["price"]

                if action == "buy":
                    total_payoff += quantity * (option_payoff - premium)
                else:  # sell
                    total_payoff += quantity * (premium - option_payoff)

            payoffs.append(total_payoff)

        return pd.DataFrame({"spot_price": spot_range, "payoff": payoffs})

    def calculate_breakeven_points(self) -> List[float]:
        """Calculate breakeven points for the strategy."""

        # This is a simplified calculation
        # In practice, would solve for payoff = 0 numerically

        spot_range = np.linspace(50, 200, 1000)  # Adjust range as needed
        payoff_df = self.calculate_payoff_diagram(spot_range)

        breakevens = []
        payoffs = payoff_df["payoff"].values

        # Find zero crossings
        for i in range(len(payoffs) - 1):
            if payoffs[i] * payoffs[i + 1] < 0:  # Sign change
                # Linear interpolation to find exact breakeven
                x1, x2 = spot_range[i], spot_range[i + 1]
                y1, y2 = payoffs[i], payoffs[i + 1]
                breakeven = x1 - y1 * (x2 - x1) / (y2 - y1)
                breakevens.append(breakeven)

        return breakevens

    @classmethod
    def create_straddle(
        cls,
        underlying_price: float,
        strike: float,
        expiry: float,
        risk_free_rate: float,
        volatility: float,
    ) -> "OptionStrategy":
        """Create long straddle strategy."""

        strategy = cls()

        call = Option(
            underlying_price, strike, expiry, risk_free_rate, volatility, "call"
        )
        put = Option(
            underlying_price, strike, expiry, risk_free_rate, volatility, "put"
        )

        strategy.add_leg(call, 1, "buy")
        strategy.add_leg(put, 1, "buy")

        return strategy

    @classmethod
    def create_iron_condor(
        cls,
        underlying_price: float,
        strikes: List[float],
        expiry: float,
        risk_free_rate: float,
        volatility: float,
    ) -> "OptionStrategy":
        """Create iron condor strategy."""

        if len(strikes) != 4:
            raise ValueError("Iron condor requires 4 strikes")

        strikes.sort()
        strategy = cls()

        # Buy OTM put
        put1 = Option(
            underlying_price, strikes[0], expiry, risk_free_rate, volatility, "put"
        )
        strategy.add_leg(put1, 1, "buy")

        # Sell ITM put
        put2 = Option(
            underlying_price, strikes[1], expiry, risk_free_rate, volatility, "put"
        )
        strategy.add_leg(put2, 1, "sell")

        # Sell ITM call
        call1 = Option(
            underlying_price, strikes[2], expiry, risk_free_rate, volatility, "call"
        )
        strategy.add_leg(call1, 1, "sell")

        # Buy OTM call
        call2 = Option(
            underlying_price, strikes[3], expiry, risk_free_rate, volatility, "call"
        )
        strategy.add_leg(call2, 1, "buy")

        return strategy


# Utility functions
def calculate_option_parity(
    call_price: float,
    put_price: float,
    spot: float,
    strike: float,
    expiry: float,
    risk_free_rate: float,
    dividend_yield: float = 0,
) -> Dict[str, float]:
    """Check put-call parity and calculate synthetic prices."""

    # Put-call parity: C - P = S*e^(-q*T) - K*e^(-r*T)
    parity_value = spot * np.exp(-dividend_yield * expiry) - strike * np.exp(
        -risk_free_rate * expiry
    )
    actual_difference = call_price - put_price

    # Synthetic prices
    synthetic_call = put_price + parity_value
    synthetic_put = call_price - parity_value

    return {
        "parity_value": parity_value,
        "actual_difference": actual_difference,
        "parity_violation": actual_difference - parity_value,
        "synthetic_call": synthetic_call,
        "synthetic_put": synthetic_put,
    }


def calculate_option_elasticity(option: Option) -> float:
    """Calculate option elasticity (percentage change in option price per percentage change in underlying)."""

    option_price = BlackScholesModel.price_option(option)["price"]
    greeks = BlackScholesModel.calculate_greeks(option)

    if option_price == 0:
        return 0

    elasticity = greeks.delta * option.underlying_price / option_price
    return elasticity


def calculate_probability_of_expiring_itm(option: Option) -> float:
    """Calculate probability of option expiring in-the-money."""

    S = option.underlying_price
    K = option.strike_price
    T = option.time_to_expiry
    r = option.risk_free_rate
    sigma = option.volatility
    q = option.dividend_yield

    if T <= 0:
        if option.option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return 1.0 if S < K else 0.0

    # Calculate d2 from Black-Scholes
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if option.option_type == "call":
        return norm.cdf(d2)
    else:
        return norm.cdf(-d2)
