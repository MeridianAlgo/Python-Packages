"""
MeridianAlgo Fixed Income Module

Comprehensive fixed income analysis including bond pricing, yield curve modeling,
credit risk analysis, and structured products. Integrates concepts from
QuantLib, PyQL, and other leading fixed income libraries.
"""

import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import interpolate, optimize

# Suppress warnings
warnings.filterwarnings("ignore")

# Check for optional dependencies
try:
    from scipy.stats import norm

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class BondPricer:
    """
    Comprehensive bond pricing and analysis.

    Features:
    - Government bond pricing
    - Corporate bond pricing
    - Municipal bond pricing
    - Zero-coupon bond pricing
    - Bond duration and convexity
    - Yield to maturity calculation
    - Credit spread analysis
    """

    def __init__(self):
        """Initialize bond pricer."""
        self.day_count_conventions = {
            "30/360": self._day_count_30_360,
            "actual/360": self._day_count_actual_360,
            "actual/365": self._day_count_actual_365,
            "actual/actual": self._day_count_actual_actual,
        }

    def price_bond(
        self,
        face_value: float,
        coupon_rate: float,
        yield_to_maturity: float,
        years_to_maturity: float,
        frequency: int = 2,
        day_count: str = "30/360",
    ) -> Dict[str, float]:
        """
        Price a bond using standard bond pricing formula.

        Args:
            face_value: Face value of the bond
            coupon_rate: Annual coupon rate
            yield_to_maturity: Yield to maturity
            years_to_maturity: Years to maturity
            frequency: Coupon frequency per year
            day_count: Day count convention

        Returns:
            Bond pricing results
        """
        n_periods = int(years_to_maturity * frequency)
        period_yield = yield_to_maturity / frequency
        period_coupon = coupon_rate * face_value / frequency

        # Calculate present value of coupons
        if period_yield == 0:
            pv_coupons = period_coupon * n_periods
        else:
            pv_coupons = (
                period_coupon * (1 - (1 + period_yield) ** -n_periods) / period_yield
            )

        # Calculate present value of face value
        pv_face = face_value / (1 + period_yield) ** n_periods

        # Bond price
        price = pv_coupons + pv_face

        # Calculate duration and convexity
        duration = self._calculate_macaulay_duration(
            face_value,
            coupon_rate,
            yield_to_maturity,
            years_to_maturity,
            frequency,
            price,
        )

        modified_duration = duration / (1 + yield_to_maturity / frequency)

        convexity = self._calculate_convexity(
            face_value,
            coupon_rate,
            yield_to_maturity,
            years_to_maturity,
            frequency,
            price,
        )

        return {
            "price": price,
            "duration": duration,
            "modified_duration": modified_duration,
            "convexity": convexity,
            "yield_to_maturity": yield_to_maturity,
            "current_yield": (coupon_rate * face_value) / price,
            "yield_spread": yield_to_maturity
            - self._get_risk_free_rate(years_to_maturity),
        }

    def calculate_yield_to_maturity(
        self,
        price: float,
        face_value: float,
        coupon_rate: float,
        years_to_maturity: float,
        frequency: int = 2,
        initial_guess: float = 0.05,
    ) -> float:
        """
        Calculate yield to maturity using Newton-Raphson method.

        Args:
            price: Current bond price
            face_value: Face value of the bond
            coupon_rate: Annual coupon rate
            years_to_maturity: Years to maturity
            frequency: Coupon frequency
            initial_guess: Initial yield guess

        Returns:
            Yield to maturity
        """

        def price_diff(ytm):
            result = self.price_bond(
                face_value, coupon_rate, ytm, years_to_maturity, frequency
            )
            return result["price"] - price

        def price_derivative(ytm):
            # Numerical derivative
            h = 1e-6
            return (price_diff(ytm + h) - price_diff(ytm - h)) / (2 * h)

        ytm = initial_guess

        for _ in range(100):  # Maximum iterations
            diff = price_diff(ytm)

            if abs(diff) < 1e-8:
                return ytm

            derivative = price_derivative(ytm)

            if abs(derivative) < 1e-10:
                break

            ytm = ytm - diff / derivative

            # Ensure yield stays positive
            ytm = max(ytm, 0.0001)

        return ytm

    def _calculate_macaulay_duration(
        self,
        face_value: float,
        coupon_rate: float,
        yield_to_maturity: float,
        years_to_maturity: float,
        frequency: int,
        bond_price: float,
    ) -> float:
        """Calculate Macaulay duration."""
        n_periods = int(years_to_maturity * frequency)
        period_yield = yield_to_maturity / frequency
        period_coupon = coupon_rate * face_value / frequency

        weighted_pv = 0.0

        for t in range(1, n_periods + 1):
            time_in_years = t / frequency
            cash_flow = period_coupon

            if t == n_periods:
                cash_flow += face_value

            pv_cash_flow = cash_flow / (1 + period_yield) ** t
            weighted_pv += time_in_years * pv_cash_flow

        return weighted_pv / bond_price

    def _calculate_convexity(
        self,
        face_value: float,
        coupon_rate: float,
        yield_to_maturity: float,
        years_to_maturity: float,
        frequency: int,
        bond_price: float,
    ) -> float:
        """Calculate bond convexity."""
        n_periods = int(years_to_maturity * frequency)
        period_yield = yield_to_maturity / frequency
        period_coupon = coupon_rate * face_value / frequency

        convexity_sum = 0.0

        for t in range(1, n_periods + 1):
            t / frequency
            cash_flow = period_coupon

            if t == n_periods:
                cash_flow += face_value

            pv_cash_flow = cash_flow / (1 + period_yield) ** t
            convexity_sum += (t / frequency + 1) * t / frequency * pv_cash_flow

        convexity = convexity_sum / (bond_price * (1 + period_yield) ** 2)

        return convexity

    def _get_risk_free_rate(self, years_to_maturity: float) -> float:
        """Get risk-free rate for given maturity (simplified)."""
        # Simplified risk-free rate curve
        if years_to_maturity <= 2:
            return 0.02
        elif years_to_maturity <= 5:
            return 0.025
        elif years_to_maturity <= 10:
            return 0.03
        else:
            return 0.035

    def _day_count_30_360(self, start_date: datetime, end_date: datetime) -> float:
        """30/360 day count convention."""
        y1, m1, d1 = start_date.year, start_date.month, start_date.day
        y2, m2, d2 = end_date.year, end_date.month, end_date.day

        if d1 == 31:
            d1 = 30
        if d2 == 31 and d1 == 30:
            d2 = 30

        days = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
        return days / 360

    def _day_count_actual_360(self, start_date: datetime, end_date: datetime) -> float:
        """Actual/360 day count convention."""
        return (end_date - start_date).days / 360

    def _day_count_actual_365(self, start_date: datetime, end_date: datetime) -> float:
        """Actual/365 day count convention."""
        return (end_date - start_date).days / 365

    def _day_count_actual_actual(
        self, start_date: datetime, end_date: datetime
    ) -> float:
        """Actual/Actual day count convention."""
        return (end_date - start_date).days / 365.25


class YieldCurveModel:
    """
    Yield curve modeling and analysis.

    Features:
    - Nelson-Siegel model
    - Svensson model
    - Spline interpolation
    - Bootstrap yield curve
    - Forward rate calculation
    - Yield curve dynamics
    """

    def __init__(self):
        """Initialize yield curve model."""
        self.curve_params = None
        self.bootstrap_curve = None

    def nelson_siegel_model(
        self,
        maturities: np.ndarray,
        yields: np.ndarray,
        initial_params: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Fit Nelson-Siegel yield curve model.

        Args:
            maturities: Array of maturities
            yields: Array of yields
            initial_params: Initial parameter guesses

        Returns:
            Fitted model parameters and curve
        """

        def nelson_siegel_func(t, beta0, beta1, beta2, tau):
            """Nelson-Siegel function."""
            exp_term = np.exp(-t / tau)
            factor = (1 - exp_term) / (t / tau)
            return beta0 + beta1 * factor + beta2 * (factor - exp_term)

        if initial_params is None:
            initial_params = [0.05, 0.02, 0.01, 2.0]  # [beta0, beta1, beta2, tau]

        # Optimize parameters
        def objective(params):
            return np.sum((yields - nelson_siegel_func(maturities, *params)) ** 2)

        result = optimize.minimize(objective, initial_params, method="L-BFGS-B")

        if result.success:
            fitted_params = result.x
            fitted_yields = nelson_siegel_func(maturities, *fitted_params)

            self.curve_params = {
                "beta0": fitted_params[0],
                "beta1": fitted_params[1],
                "beta2": fitted_params[2],
                "tau": fitted_params[3],
                "model_type": "Nelson-Siegel",
            }

            return {
                "parameters": self.curve_params,
                "fitted_yields": fitted_yields,
                "residuals": yields - fitted_yields,
                "rmse": np.sqrt(np.mean((yields - fitted_yields) ** 2)),
                "success": True,
            }
        else:
            return {"success": False, "message": result.message}

    def svensson_model(
        self,
        maturities: np.ndarray,
        yields: np.ndarray,
        initial_params: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Fit Svensson yield curve model.

        Args:
            maturities: Array of maturities
            yields: Array of yields
            initial_params: Initial parameter guesses

        Returns:
            Fitted model parameters and curve
        """

        def svensson_func(t, beta0, beta1, beta2, beta3, tau1, tau2):
            """Svensson function."""
            exp_term1 = np.exp(-t / tau1)
            exp_term2 = np.exp(-t / tau2)
            factor1 = (1 - exp_term1) / (t / tau1)
            factor2 = (1 - exp_term2) / (t / tau2)

            return (
                beta0
                + beta1 * factor1
                + beta2 * (factor1 - exp_term1)
                + beta3 * (factor2 - exp_term2)
            )

        if initial_params is None:
            initial_params = [0.05, 0.02, 0.01, 0.005, 2.0, 5.0]

        # Optimize parameters
        def objective(params):
            return np.sum((yields - svensson_func(maturities, *params)) ** 2)

        result = optimize.minimize(objective, initial_params, method="L-BFGS-B")

        if result.success:
            fitted_params = result.x
            fitted_yields = svensson_func(maturities, *fitted_params)

            self.curve_params = {
                "beta0": fitted_params[0],
                "beta1": fitted_params[1],
                "beta2": fitted_params[2],
                "beta3": fitted_params[3],
                "tau1": fitted_params[4],
                "tau2": fitted_params[5],
                "model_type": "Svensson",
            }

            return {
                "parameters": self.curve_params,
                "fitted_yields": fitted_yields,
                "residuals": yields - fitted_yields,
                "rmse": np.sqrt(np.mean((yields - fitted_yields) ** 2)),
                "success": True,
            }
        else:
            return {"success": False, "message": result.message}

    def bootstrap_yield_curve(
        self,
        bond_prices: np.ndarray,
        cash_flows: List[np.ndarray],
        cash_flow_times: List[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Bootstrap yield curve from bond prices.

        Args:
            bond_prices: Array of bond prices
            cash_flows: List of cash flow arrays for each bond
            cash_flow_times: List of cash flow time arrays for each bond

        Returns:
            Bootstrapped yield curve
        """
        # Collect all unique cash flow times
        all_times = sorted(set().union(*cash_flow_times))

        # Initialize zero coupon bond prices
        zcb_prices = np.ones(len(all_times))

        # Bootstrap iteratively
        for i, (price, cf_times, cf_amounts) in enumerate(
            zip(bond_prices, cash_flow_times, cash_flows)
        ):
            for j, time in enumerate(cf_times):
                time_idx = all_times.index(time)

                # Calculate present value of known cash flows
                pv_known = 0.0
                for k, (cf_time, cf_amount) in enumerate(
                    zip(cf_times[:j], cf_amounts[:j])
                ):
                    if cf_time < time:
                        time_idx_known = all_times.index(cf_time)
                        pv_known += cf_amount * zcb_prices[time_idx_known]

                # Solve for discount factor
                if cf_amounts[j] > 0:
                    zcb_prices[time_idx] = (price - pv_known) / cf_amounts[j]

        # Convert to yields
        yields = -np.log(zcb_prices) / np.array(all_times)

        self.bootstrap_curve = {
            "times": np.array(all_times),
            "yields": yields,
            "zcb_prices": zcb_prices,
        }

        return self.bootstrap_curve

    def calculate_forward_rates(
        self, maturities: np.ndarray, yields: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate forward rates from yield curve.

        Args:
            maturities: Array of maturities
            yields: Array of yields

        Returns:
            Forward rates
        """
        # Calculate instantaneous forward rates
        forward_rates = np.zeros_like(maturities)

        for i in range(len(maturities)):
            if i == 0:
                # First forward rate equals spot rate
                forward_rates[i] = yields[i]
            else:
                # Calculate forward rate using formula
                t1 = maturities[i - 1]
                t2 = maturities[i]
                y1 = yields[i - 1]
                y2 = yields[i]

                forward_rates[i] = (y2 * t2 - y1 * t1) / (t2 - t1)

        return {
            "maturities": maturities,
            "spot_rates": yields,
            "forward_rates": forward_rates,
        }

    def spline_interpolation(
        self,
        maturities: np.ndarray,
        yields: np.ndarray,
        knot_points: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Spline interpolation of yield curve.

        Args:
            maturities: Array of maturities
            yields: Array of yields
            knot_points: Optional knot points for spline

        Returns:
            Interpolated yield curve
        """
        if knot_points is None:
            # Use quantiles as knot points
            n_knots = min(5, len(maturities) // 2)
            knot_points = np.quantile(maturities, np.linspace(0, 1, n_knots))

        # Create spline interpolation
        spline = interpolate.UnivariateSpline(maturities, yields, s=0)

        # Generate smooth curve
        smooth_maturities = np.linspace(maturities.min(), maturities.max(), 100)
        smooth_yields = spline(smooth_maturities)

        return {
            "original_maturities": maturities,
            "original_yields": yields,
            "smooth_maturities": smooth_maturities,
            "smooth_yields": smooth_yields,
            "spline_function": spline,
            "knot_points": knot_points,
        }


class CreditRiskModel:
    """
    Credit risk modeling and analysis.

    Features:
    - Merton structural model
    - Reduced form models
    - Credit VaR calculation
    - CDS pricing
    - Default probability estimation
    - Recovery rate modeling
    """

    def __init__(self):
        """Initialize credit risk model."""
        pass

    def merton_model(
        self,
        asset_value: float,
        asset_volatility: float,
        debt_face_value: float,
        risk_free_rate: float,
        time_to_maturity: float,
        dividend_yield: float = 0.0,
    ) -> Dict[str, float]:
        """
        Merton structural model for default probability.

        Args:
            asset_value: Current asset value
            asset_volatility: Asset volatility
            debt_face_value: Face value of debt
            risk_free_rate: Risk-free rate
            time_to_maturity: Time to maturity
            dividend_yield: Dividend yield

        Returns:
            Credit risk metrics
        """
        if not SCIPY_AVAILABLE:
            return self._merton_approximation(
                asset_value,
                asset_volatility,
                debt_face_value,
                risk_free_rate,
                time_to_maturity,
                dividend_yield,
            )

        d1 = (
            np.log(asset_value / debt_face_value)
            + (risk_free_rate - dividend_yield + 0.5 * asset_volatility**2)
            * time_to_maturity
        ) / (asset_volatility * np.sqrt(time_to_maturity))

        d2 = d1 - asset_volatility * np.sqrt(time_to_maturity)

        # Default probability
        default_probability = norm.cdf(-d2)

        # Distance to default
        distance_to_default = d2

        # Expected loss
        recovery_rate = 0.4  # Assumed recovery rate
        expected_loss = default_probability * (1 - recovery_rate)

        # Credit spread
        credit_spread = -np.log(1 - expected_loss) / time_to_maturity

        return {
            "default_probability": default_probability,
            "distance_to_default": distance_to_default,
            "expected_loss": expected_loss,
            "credit_spread": credit_spread,
            "d1": d1,
            "d2": d2,
        }

    def _merton_approximation(
        self,
        asset_value,
        asset_volatility,
        debt_face_value,
        risk_free_rate,
        time_to_maturity,
        dividend_yield,
    ):
        """Fallback Merton model implementation."""
        d1 = (
            np.log(asset_value / debt_face_value)
            + (risk_free_rate - dividend_yield + 0.5 * asset_volatility**2)
            * time_to_maturity
        ) / (asset_volatility * np.sqrt(time_to_maturity))

        d2 = d1 - asset_volatility * np.sqrt(time_to_maturity)

        # Use error function approximation for normal CDF
        def norm_cdf(x):
            return 0.5 * (1 + np.erf(x / np.sqrt(2)))

        default_probability = norm_cdf(-d2)

        return {
            "default_probability": default_probability,
            "distance_to_default": d2,
            "expected_loss": default_probability * 0.6,
            "credit_spread": -np.log(1 - default_probability * 0.6) / time_to_maturity,
        }

    def cds_pricing(
        self,
        hazard_rate: float,
        recovery_rate: float,
        maturity: float,
        risk_free_rate: float,
        premium_frequency: int = 4,
    ) -> Dict[str, float]:
        """
        Price Credit Default Swap.

        Args:
            hazard_rate: Hazard rate (default intensity)
            recovery_rate: Recovery rate
            maturity: Time to maturity
            risk_free_rate: Risk-free rate
            premium_frequency: Premium payment frequency

        Returns:
            CDS pricing results
        """
        # Simplified CDS pricing
        n_periods = int(maturity * premium_frequency)
        dt = maturity / n_periods

        # Calculate survival probabilities
        survival_probabilities = np.exp(-hazard_rate * np.arange(1, n_periods + 1) * dt)

        # Calculate default probabilities
        default_probabilities = np.diff(np.concatenate([[1.0], survival_probabilities]))

        # Calculate protection leg PV
        protection_leg = 0.0
        for i in range(n_periods):
            time = (i + 1) * dt
            default_prob = default_probabilities[i]
            discount_factor = np.exp(-risk_free_rate * time)
            protection_leg += default_prob * (1 - recovery_rate) * discount_factor

        # Calculate premium leg PV
        premium_leg = 0.0
        for i in range(n_periods):
            time = (i + 1) * dt
            survival_prob = survival_probabilities[i]
            discount_factor = np.exp(-risk_free_rate * time)
            premium_leg += survival_prob * dt * discount_factor

        # CDS spread
        cds_spread = protection_leg / premium_leg

        return {
            "cds_spread": cds_spread,
            "protection_leg": protection_leg,
            "premium_leg": premium_leg,
            "hazard_rate": hazard_rate,
            "recovery_rate": recovery_rate,
        }

    def credit_var(
        self,
        exposures: np.ndarray,
        default_probabilities: np.ndarray,
        recovery_rates: np.ndarray,
        correlation_matrix: np.ndarray,
        confidence_level: float = 0.99,
    ) -> Dict[str, float]:
        """
        Calculate Credit Value at Risk.

        Args:
            exposures: Array of exposures
            default_probabilities: Array of default probabilities
            recovery_rates: Array of recovery rates
            correlation_matrix: Correlation matrix
            confidence_level: Confidence level

        Returns:
            Credit VaR metrics
        """
        n_obligors = len(exposures)

        # Generate correlated default scenarios using Gaussian copula
        n_simulations = 10000

        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(correlation_matrix)
        except np.linalg.LinAlgError:
            # If not positive definite, use nearest positive definite matrix
            eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
            eigenvals[eigenvals < 0] = 0
            correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            L = np.linalg.cholesky(correlation_matrix)

        # Generate correlated random variables
        Z = np.random.standard_normal((n_simulations, n_obligors))
        correlated_Z = Z @ L.T

        # Convert to default indicators
        defaults = np.zeros((n_simulations, n_obligors))
        for i in range(n_obligors):
            defaults[:, i] = correlated_Z[:, i] < norm.ppf(default_probabilities[i])

        # Calculate losses
        losses = np.zeros(n_simulations)
        for i in range(n_obligors):
            losses += defaults[:, i] * exposures[i] * (1 - recovery_rates[i])

        # Calculate VaR and Expected Shortfall
        var_99 = np.percentile(losses, confidence_level * 100)
        expected_shortfall = np.mean(losses[losses >= var_99])

        return {
            "credit_var": var_99,
            "expected_shortfall": expected_shortfall,
            "expected_loss": np.mean(losses),
            "loss_std": np.std(losses),
            "n_simulations": n_simulations,
        }


class StructuredProducts:
    """
    Structured products pricing and analysis.

    Features:
    - Mortgage-backed securities
    - Collateralized debt obligations
    - Asset-backed securities
    - Credit-linked notes
    - Structured notes
    """

    def __init__(self):
        """Initialize structured products analyzer."""
        pass

    def mortgage_backed_security(
        self,
        principal: float,
        coupon_rate: float,
        wac: float,  # Weighted average coupon
        wam: float,  # Weighted average maturity
        psa_speed: float = 100,  # PSA prepayment speed
        n_months: int = 360,
    ) -> Dict[str, Any]:
        """
        Price mortgage-backed security.

        Args:
            principal: Principal amount
            coupon_rate: Coupon rate
            wac: Weighted average coupon
            wam: Weighted average maturity in months
            psa_speed: PSA prepayment speed percentage
            n_months: Total number of months

        Returns:
            MBS pricing results
        """
        # Simplified PSA prepayment model
        monthly_psa = psa_speed / 100

        # Initialize arrays
        scheduled_principal = np.zeros(n_months)
        scheduled_interest = np.zeros(n_months)
        prepayments = np.zeros(n_months)
        total_principal = np.zeros(n_months)
        cash_flows = np.zeros(n_months)

        remaining_principal = principal

        for month in range(1, n_months + 1):
            # Scheduled payment
            monthly_rate = wac / 12

            if remaining_principal > 0:
                # Calculate scheduled payment (simplified)
                if month <= wam:
                    payment = (
                        remaining_principal
                        * monthly_rate
                        / (1 - (1 + monthly_rate) ** -(wam - month + 1))
                    )
                else:
                    payment = remaining_principal * (1 + monthly_rate)

                interest_payment = remaining_principal * monthly_rate
                principal_payment = payment - interest_payment

                # Prepayment calculation (simplified PSA)
                if month <= 30:
                    cpr = monthly_psa * 0.002 * month
                else:
                    cpr = monthly_psa * 0.06

                smm = 1 - (1 - cpr) ** (1 / 12)
                prepayment_amount = (remaining_principal - principal_payment) * smm

                # Update cash flows
                scheduled_interest[month - 1] = interest_payment
                scheduled_principal[month - 1] = principal_payment
                prepayments[month - 1] = prepayment_amount
                total_principal[month - 1] = principal_payment + prepayment_amount
                cash_flows[month - 1] = (
                    interest_payment + principal_payment + prepayment_amount
                )

                # Update remaining principal
                remaining_principal -= principal_payment + prepayment_amount
                remaining_principal = max(0, remaining_principal)

        # Calculate price (simplified - assuming 5% discount rate)
        discount_rate = 0.05 / 12
        price = np.sum(cash_flows / (1 + discount_rate) ** np.arange(1, n_months + 1))

        return {
            "price": price,
            "cash_flows": cash_flows,
            "scheduled_principal": scheduled_principal,
            "scheduled_interest": scheduled_interest,
            "prepayments": prepayments,
            "total_cash_flow": np.sum(cash_flows),
            "weighted_average_life": np.sum(
                np.arange(1, n_months + 1) * total_principal
            )
            / np.sum(total_principal),
        }

    def collateralized_debt_obligation(
        self,
        pool_principal: float,
        tranche_sizes: List[float],
        tranche_attachments: List[float],
        tranche_detachments: List[float],
        default_rate: float,
        recovery_rate: float = 0.4,
    ) -> Dict[str, Any]:
        """
        Analyze Collateralized Debt Obligation (CDO).

        Args:
            pool_principal: Total principal in the pool
            tranche_sizes: Sizes of each tranche
            tranche_attachments: Attachment points for each tranche
            tranche_detachments: Detachment points for each tranche
            default_rate: Expected default rate
            recovery_rate: Recovery rate

        Returns:
            CDO analysis results
        """
        n_tranches = len(tranche_sizes)

        # Calculate losses
        total_loss = pool_principal * default_rate * (1 - recovery_rate)

        # Allocate losses to tranches (waterfall structure)
        tranche_losses = np.zeros(n_tranches)
        remaining_loss = total_loss

        for i in range(n_tranches):
            tranche_size = tranche_sizes[i]
            tranche_attachments[i]
            tranche_detachments[i]

            # Calculate loss for this tranche
            if remaining_loss > 0:
                tranche_loss = min(remaining_loss, tranche_size)
                tranche_losses[i] = tranche_loss
                remaining_loss -= tranche_loss

        # Calculate tranche returns
        tranche_returns = []
        for i in range(n_tranches):
            tranche_size = tranche_sizes[i]
            tranche_loss = tranche_losses[i]
            tranche_return = (tranche_size - tranche_loss) / tranche_size
            tranche_returns.append(tranche_return)

        return {
            "total_loss": total_loss,
            "tranche_losses": tranche_losses,
            "tranche_returns": tranche_returns,
            "pool_return": (pool_principal - total_loss) / pool_principal,
            "expected_loss_rate": default_rate * (1 - recovery_rate),
        }


# Utility functions
def calculate_bond_spread(
    bond_yield: float,
    benchmark_yield: float,
    bond_price: float,
    benchmark_price: float = 100.0,
) -> Dict[str, float]:
    """
    Calculate bond spread and spread duration.

    Args:
        bond_yield: Bond yield
        benchmark_yield: Benchmark yield
        bond_price: Bond price
        benchmark_price: Benchmark price

    Returns:
        Spread analysis
    """
    spread = bond_yield - benchmark_yield
    spread_duration = (benchmark_price - bond_price) / (bond_price * spread)

    return {
        "spread": spread,
        "spread_duration": spread_duration,
        "spread_in_bps": spread * 10000,
    }


def calculate_portfolio_duration(
    bond_weights: np.ndarray, bond_durations: np.ndarray
) -> float:
    """
    Calculate portfolio duration.

    Args:
        bond_weights: Array of bond weights
        bond_durations: Array of bond durations

    Returns:
        Portfolio duration
    """
    return np.sum(bond_weights * bond_durations)


# Export main classes and functions
__all__ = [
    "BondPricer",
    "YieldCurveModel",
    "CreditRiskModel",
    "StructuredProducts",
    "calculate_bond_spread",
    "calculate_portfolio_duration",
]
