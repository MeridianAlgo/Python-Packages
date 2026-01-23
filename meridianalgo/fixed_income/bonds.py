"""
Comprehensive bond pricing and yield curve construction system.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


@dataclass
class Bond:
    """Bond instrument representation."""

    face_value: float = 100.0
    coupon_rate: float = 0.0
    maturity_date: datetime = None
    issue_date: datetime = None
    coupon_frequency: int = 2  # Semi-annual
    day_count_convention: str = "30/360"
    credit_rating: str = "AAA"
    issuer: str = "Government"

    def __post_init__(self):
        if self.maturity_date is None:
            self.maturity_date = datetime.now() + timedelta(days=365 * 10)  # 10 years
        if self.issue_date is None:
            self.issue_date = datetime.now()


@dataclass
class YieldCurvePoint:
    """Point on yield curve."""

    maturity: float  # Years
    yield_rate: float
    instrument_type: str = "bond"


class YieldCurve:
    """Yield curve construction and interpolation."""

    def __init__(self, curve_date: datetime = None):
        self.curve_date = curve_date or datetime.now()
        self.points: List[YieldCurvePoint] = []
        self.interpolator = None
        self.method = "linear"

    def add_point(
        self, maturity: float, yield_rate: float, instrument_type: str = "bond"
    ):
        """Add point to yield curve."""
        point = YieldCurvePoint(maturity, yield_rate, instrument_type)
        self.points.append(point)
        self._sort_points()

    def _sort_points(self):
        """Sort points by maturity."""
        self.points.sort(key=lambda x: x.maturity)

    def build_curve(self, method: str = "cubic_spline"):
        """Build interpolated yield curve."""
        if len(self.points) < 2:
            raise ValueError("Need at least 2 points to build curve")

        self.method = method
        maturities = [p.maturity for p in self.points]
        yields = [p.yield_rate for p in self.points]

        if method == "linear":
            self.interpolator = interp1d(
                maturities,
                yields,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
        elif method == "cubic_spline":
            self.interpolator = CubicSpline(maturities, yields, extrapolate=True)
        elif method == "nelson_siegel":
            self.interpolator = self._fit_nelson_siegel(maturities, yields)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

    def get_yield(self, maturity: float) -> float:
        """Get yield for given maturity."""
        if self.interpolator is None:
            self.build_curve()

        return float(self.interpolator(maturity))

    def get_forward_rate(self, t1: float, t2: float) -> float:
        """Calculate forward rate between two maturities."""
        if t1 >= t2:
            raise ValueError("t1 must be less than t2")

        y1 = self.get_yield(t1)
        y2 = self.get_yield(t2)

        # Forward rate formula
        forward_rate = ((1 + y2) ** t2 / (1 + y1) ** t1) ** (1 / (t2 - t1)) - 1
        return forward_rate

    def _fit_nelson_siegel(self, maturities: List[float], yields: List[float]):
        """Fit Nelson-Siegel model to yield curve."""

        def nelson_siegel(tau, beta0, beta1, beta2, lambda_param):
            """Nelson-Siegel yield curve model."""
            tau = np.array(tau)
            term1 = beta1 * (1 - np.exp(-tau / lambda_param)) / (tau / lambda_param)
            term2 = beta2 * (
                (1 - np.exp(-tau / lambda_param)) / (tau / lambda_param)
                - np.exp(-tau / lambda_param)
            )
            return beta0 + term1 + term2

        def objective(params):
            beta0, beta1, beta2, lambda_param = params
            predicted = nelson_siegel(maturities, beta0, beta1, beta2, lambda_param)
            return np.sum((np.array(yields) - predicted) ** 2)

        # Initial guess
        initial_params = [0.05, -0.02, 0.01, 2.0]

        try:
            from scipy.optimize import minimize

            result = minimize(
                objective,
                initial_params,
                method="L-BFGS-B",
                bounds=[(0, 0.2), (-0.1, 0.1), (-0.1, 0.1), (0.1, 10)],
            )

            if result.success:
                beta0, beta1, beta2, lambda_param = result.x
                return lambda tau: nelson_siegel(tau, beta0, beta1, beta2, lambda_param)
            else:
                logger.warning("Nelson-Siegel fitting failed, using cubic spline")
                return CubicSpline(maturities, yields, extrapolate=True)
        except Exception:
            logger.warning("Nelson-Siegel fitting failed, using cubic spline")
            return CubicSpline(maturities, yields, extrapolate=True)

    def bootstrap_curve(self, instruments: List[Dict[str, Any]]):
        """Bootstrap yield curve from market instruments."""

        # Sort instruments by maturity
        instruments.sort(key=lambda x: x["maturity"])

        bootstrapped_points = []

        for instrument in instruments:
            maturity = instrument["maturity"]
            price = instrument["price"]
            coupon_rate = instrument.get("coupon_rate", 0)
            face_value = instrument.get("face_value", 100)
            frequency = instrument.get("frequency", 2)

            if maturity <= 1.0:  # Money market instruments
                # Simple yield calculation
                yield_rate = (face_value - price) / price / maturity
            else:
                # Bootstrap from existing curve
                yield_rate = self._bootstrap_yield(
                    maturity,
                    price,
                    coupon_rate,
                    face_value,
                    frequency,
                    bootstrapped_points,
                )

            bootstrapped_points.append(YieldCurvePoint(maturity, yield_rate))

        self.points = bootstrapped_points
        self.build_curve()

    def _bootstrap_yield(
        self,
        maturity: float,
        price: float,
        coupon_rate: float,
        face_value: float,
        frequency: int,
        existing_points: List[YieldCurvePoint],
    ) -> float:
        """Bootstrap yield for a bond given existing curve points."""

        def bond_price_from_yield(yield_rate):
            return self._calculate_bond_price_from_yield(
                yield_rate,
                maturity,
                coupon_rate,
                face_value,
                frequency,
                existing_points,
            )

        def objective(yield_rate):
            return (bond_price_from_yield(yield_rate) - price) ** 2

        # Find yield that matches market price
        try:
            result = minimize_scalar(objective, bounds=(0.001, 0.5), method="bounded")
            return result.x
        except Exception:
            # Fallback to simple approximation
            return coupon_rate + (face_value - price) / (price * maturity)

    def _calculate_bond_price_from_yield(
        self,
        yield_rate: float,
        maturity: float,
        coupon_rate: float,
        face_value: float,
        frequency: int,
        existing_points: List[YieldCurvePoint],
    ) -> float:
        """Calculate bond price given yield and existing curve points."""

        # Create temporary curve with new point
        temp_points = existing_points + [YieldCurvePoint(maturity, yield_rate)]
        temp_curve = YieldCurve()
        temp_curve.points = temp_points
        temp_curve.build_curve()

        # Calculate bond price using temporary curve
        return BondPricer.price_bond_from_curve(
            Bond(
                face_value=face_value,
                coupon_rate=coupon_rate,
                coupon_frequency=frequency,
            ),
            temp_curve,
            maturity,
        )


class BondPricer:
    """Bond pricing calculations."""

    @staticmethod
    def price_bond(
        bond: Bond, yield_to_maturity: float, settlement_date: datetime = None
    ) -> Dict[str, float]:
        """Price bond given yield to maturity."""

        if settlement_date is None:
            settlement_date = datetime.now()

        # Calculate time to maturity
        time_to_maturity = (bond.maturity_date - settlement_date).days / 365.25

        if time_to_maturity <= 0:
            return {"price": bond.face_value, "accrued_interest": 0}

        # Calculate coupon payments
        coupon_payment = bond.coupon_rate * bond.face_value / bond.coupon_frequency
        periods_per_year = bond.coupon_frequency
        total_periods = int(time_to_maturity * periods_per_year)

        # Present value of coupon payments
        pv_coupons = 0
        for period in range(1, total_periods + 1):
            period / periods_per_year
            discount_factor = (1 + yield_to_maturity / periods_per_year) ** (-period)
            pv_coupons += coupon_payment * discount_factor

        # Present value of principal
        discount_factor_principal = (1 + yield_to_maturity / periods_per_year) ** (
            -total_periods
        )
        pv_principal = bond.face_value * discount_factor_principal

        # Total price
        clean_price = pv_coupons + pv_principal

        # Calculate accrued interest
        accrued_interest = BondPricer._calculate_accrued_interest(bond, settlement_date)

        dirty_price = clean_price + accrued_interest

        return {
            "clean_price": clean_price,
            "dirty_price": dirty_price,
            "accrued_interest": accrued_interest,
            "yield_to_maturity": yield_to_maturity,
        }

    @staticmethod
    def price_bond_from_curve(
        bond: Bond, yield_curve: YieldCurve, time_to_maturity: float = None
    ) -> float:
        """Price bond using yield curve."""

        if time_to_maturity is None:
            time_to_maturity = (bond.maturity_date - datetime.now()).days / 365.25

        if time_to_maturity <= 0:
            return bond.face_value

        coupon_payment = bond.coupon_rate * bond.face_value / bond.coupon_frequency
        periods_per_year = bond.coupon_frequency
        total_periods = int(time_to_maturity * periods_per_year)

        # Present value using spot rates from curve
        pv_coupons = 0
        for period in range(1, total_periods + 1):
            time_to_payment = period / periods_per_year
            spot_rate = yield_curve.get_yield(time_to_payment)
            discount_factor = np.exp(-spot_rate * time_to_payment)
            pv_coupons += coupon_payment * discount_factor

        # Present value of principal
        spot_rate_maturity = yield_curve.get_yield(time_to_maturity)
        discount_factor_principal = np.exp(-spot_rate_maturity * time_to_maturity)
        pv_principal = bond.face_value * discount_factor_principal

        return pv_coupons + pv_principal

    @staticmethod
    def calculate_yield_to_maturity(
        bond: Bond, market_price: float, settlement_date: datetime = None
    ) -> float:
        """Calculate yield to maturity given market price."""

        def objective(ytm):
            pricing_result = BondPricer.price_bond(bond, ytm, settlement_date)
            return (pricing_result["clean_price"] - market_price) ** 2

        try:
            result = minimize_scalar(objective, bounds=(0.001, 0.5), method="bounded")
            return result.x
        except Exception:
            # Fallback approximation
            time_to_maturity = (
                bond.maturity_date - (settlement_date or datetime.now())
            ).days / 365.25
            annual_coupon = bond.coupon_rate * bond.face_value
            return (
                annual_coupon + (bond.face_value - market_price) / time_to_maturity
            ) / market_price

    @staticmethod
    def calculate_duration(bond: Bond, yield_to_maturity: float) -> Dict[str, float]:
        """Calculate Macaulay and Modified duration."""

        time_to_maturity = (bond.maturity_date - datetime.now()).days / 365.25
        coupon_payment = bond.coupon_rate * bond.face_value / bond.coupon_frequency
        periods_per_year = bond.coupon_frequency
        total_periods = int(time_to_maturity * periods_per_year)

        # Calculate bond price
        BondPricer.price_bond(bond, yield_to_maturity)["clean_price"]

        # Calculate weighted average time to cash flows
        weighted_time = 0
        total_pv = 0

        for period in range(1, total_periods + 1):
            time_to_payment = period / periods_per_year
            cash_flow = (
                coupon_payment
                if period < total_periods
                else coupon_payment + bond.face_value
            )
            discount_factor = (1 + yield_to_maturity / periods_per_year) ** (-period)
            pv_cash_flow = cash_flow * discount_factor

            weighted_time += time_to_payment * pv_cash_flow
            total_pv += pv_cash_flow

        macaulay_duration = weighted_time / total_pv
        modified_duration = macaulay_duration / (
            1 + yield_to_maturity / periods_per_year
        )

        return {
            "macaulay_duration": macaulay_duration,
            "modified_duration": modified_duration,
        }

    @staticmethod
    def calculate_convexity(bond: Bond, yield_to_maturity: float) -> float:
        """Calculate bond convexity."""

        time_to_maturity = (bond.maturity_date - datetime.now()).days / 365.25
        coupon_payment = bond.coupon_rate * bond.face_value / bond.coupon_frequency
        periods_per_year = bond.coupon_frequency
        total_periods = int(time_to_maturity * periods_per_year)

        bond_price = BondPricer.price_bond(bond, yield_to_maturity)["clean_price"]

        convexity = 0
        for period in range(1, total_periods + 1):
            cash_flow = (
                coupon_payment
                if period < total_periods
                else coupon_payment + bond.face_value
            )
            discount_factor = (1 + yield_to_maturity / periods_per_year) ** (-period)
            pv_cash_flow = cash_flow * discount_factor

            convexity += pv_cash_flow * period * (period + 1) / (periods_per_year**2)

        convexity = convexity / (
            bond_price * (1 + yield_to_maturity / periods_per_year) ** 2
        )

        return convexity

    @staticmethod
    def _calculate_accrued_interest(bond: Bond, settlement_date: datetime) -> float:
        """Calculate accrued interest."""

        # Find last coupon date
        days_in_year = 365.25
        coupon_frequency = bond.coupon_frequency
        days_between_coupons = days_in_year / coupon_frequency

        # Simplified calculation - assumes regular coupon schedule
        days_since_last_coupon = (
            settlement_date - bond.issue_date
        ).days % days_between_coupons

        annual_coupon = bond.coupon_rate * bond.face_value
        daily_coupon = annual_coupon / days_in_year

        return daily_coupon * days_since_last_coupon


class CreditSpreadAnalyzer:
    """Credit spread analysis for corporate bonds."""

    def __init__(self, risk_free_curve: YieldCurve):
        self.risk_free_curve = risk_free_curve
        self.credit_spreads = {}

    def calculate_credit_spread(
        self, corporate_bond: Bond, corporate_yield: float, maturity: float
    ) -> float:
        """Calculate credit spread over risk-free rate."""

        risk_free_yield = self.risk_free_curve.get_yield(maturity)
        credit_spread = corporate_yield - risk_free_yield

        return credit_spread

    def price_corporate_bond(
        self, bond: Bond, credit_spread: float, maturity: float = None
    ) -> float:
        """Price corporate bond with credit spread."""

        if maturity is None:
            maturity = (bond.maturity_date - datetime.now()).days / 365.25

        risk_free_yield = self.risk_free_curve.get_yield(maturity)
        corporate_yield = risk_free_yield + credit_spread

        return BondPricer.price_bond(bond, corporate_yield)["clean_price"]

    def calculate_default_probability(
        self, credit_spread: float, recovery_rate: float = 0.4
    ) -> float:
        """Estimate default probability from credit spread."""

        # Simplified model: spread = (1 - recovery_rate) * default_probability
        default_probability = credit_spread / (1 - recovery_rate)

        return min(default_probability, 1.0)  # Cap at 100%


class InflationLinkedBond:
    """Inflation-linked bond (TIPS) pricing."""

    def __init__(self, bond: Bond, inflation_curve: YieldCurve):
        self.bond = bond
        self.inflation_curve = inflation_curve

    def calculate_real_yield(self, nominal_yield: float, maturity: float) -> float:
        """Calculate real yield from nominal yield."""

        expected_inflation = self.inflation_curve.get_yield(maturity)

        # Fisher equation: (1 + nominal) = (1 + real) * (1 + inflation)
        real_yield = (1 + nominal_yield) / (1 + expected_inflation) - 1

        return real_yield

    def price_tips(
        self, real_yield: float, current_cpi: float, base_cpi: float
    ) -> Dict[str, float]:
        """Price Treasury Inflation-Protected Securities."""

        # Inflation adjustment factor
        inflation_factor = current_cpi / base_cpi

        # Adjust principal and coupons for inflation
        adjusted_face_value = self.bond.face_value * inflation_factor
        adjusted_coupon_rate = (
            self.bond.coupon_rate
        )  # Rate stays same, but applied to adjusted principal

        # Create adjusted bond
        adjusted_bond = Bond(
            face_value=adjusted_face_value,
            coupon_rate=adjusted_coupon_rate,
            maturity_date=self.bond.maturity_date,
            coupon_frequency=self.bond.coupon_frequency,
        )

        # Price using real yield
        pricing_result = BondPricer.price_bond(adjusted_bond, real_yield)

        return {
            "inflation_adjusted_price": pricing_result["clean_price"],
            "inflation_factor": inflation_factor,
            "adjusted_face_value": adjusted_face_value,
            "real_yield": real_yield,
        }


class BondPortfolio:
    """Bond portfolio analytics."""

    def __init__(self):
        self.bonds: List[Tuple[Bond, float, float]] = []  # (bond, weight, yield)

    def add_bond(self, bond: Bond, weight: float, yield_to_maturity: float):
        """Add bond to portfolio."""
        self.bonds.append((bond, weight, yield_to_maturity))

    def calculate_portfolio_duration(self) -> float:
        """Calculate portfolio duration."""

        weighted_duration = 0
        total_weight = sum(weight for _, weight, _ in self.bonds)

        for bond, weight, ytm in self.bonds:
            duration_result = BondPricer.calculate_duration(bond, ytm)
            modified_duration = duration_result["modified_duration"]
            weighted_duration += (weight / total_weight) * modified_duration

        return weighted_duration

    def calculate_portfolio_convexity(self) -> float:
        """Calculate portfolio convexity."""

        weighted_convexity = 0
        total_weight = sum(weight for _, weight, _ in self.bonds)

        for bond, weight, ytm in self.bonds:
            convexity = BondPricer.calculate_convexity(bond, ytm)
            weighted_convexity += (weight / total_weight) * convexity

        return weighted_convexity

    def calculate_portfolio_yield(self) -> float:
        """Calculate portfolio yield."""

        weighted_yield = 0
        total_weight = sum(weight for _, weight, _ in self.bonds)

        for bond, weight, ytm in self.bonds:
            weighted_yield += (weight / total_weight) * ytm

        return weighted_yield

    def estimate_price_change(self, yield_change: float) -> Dict[str, float]:
        """Estimate portfolio price change for yield change."""

        duration = self.calculate_portfolio_duration()
        convexity = self.calculate_portfolio_convexity()

        # Duration-convexity approximation
        duration_effect = -duration * yield_change
        convexity_effect = 0.5 * convexity * (yield_change**2)

        total_price_change = duration_effect + convexity_effect

        return {
            "duration_effect": duration_effect,
            "convexity_effect": convexity_effect,
            "total_price_change": total_price_change,
            "portfolio_duration": duration,
            "portfolio_convexity": convexity,
        }


class ZeroCouponBond:
    """Zero-coupon bond pricing and analytics."""

    def __init__(self, face_value: float = 100.0, maturity_date: datetime = None):
        self.face_value = face_value
        self.maturity_date = maturity_date or (datetime.now() + timedelta(days=365 * 5))

    def price(self, yield_rate: float, settlement_date: datetime = None) -> float:
        """Price zero-coupon bond."""
        if settlement_date is None:
            settlement_date = datetime.now()

        time_to_maturity = (self.maturity_date - settlement_date).days / 365.25

        if time_to_maturity <= 0:
            return self.face_value

        return self.face_value / ((1 + yield_rate) ** time_to_maturity)

    def yield_from_price(self, price: float, settlement_date: datetime = None) -> float:
        """Calculate yield from price."""
        if settlement_date is None:
            settlement_date = datetime.now()

        time_to_maturity = (self.maturity_date - settlement_date).days / 365.25

        if time_to_maturity <= 0:
            return 0.0

        return (self.face_value / price) ** (1 / time_to_maturity) - 1


class CallableBond:
    """Callable bond pricing with embedded options."""

    def __init__(self, bond: Bond, call_schedule: List[Dict[str, Any]]):
        """
        Initialize callable bond.

        Args:
            bond: Underlying bond
            call_schedule: List of call options with 'date' and 'price' keys
        """
        self.bond = bond
        self.call_schedule = sorted(call_schedule, key=lambda x: x["date"])

    def price_callable_bond(
        self, yield_curve: YieldCurve, volatility: float = 0.15
    ) -> Dict[str, float]:
        """
        Price callable bond using binomial tree.

        Args:
            yield_curve: Interest rate curve
            volatility: Interest rate volatility

        Returns:
            Dictionary with pricing results
        """
        # Simplified binomial tree approach
        time_to_maturity = (self.bond.maturity_date - datetime.now()).days / 365.25

        # Price straight bond
        straight_bond_price = BondPricer.price_bond_from_curve(
            self.bond, yield_curve, time_to_maturity
        )

        # Estimate option value (simplified)
        option_value = self._estimate_call_option_value(yield_curve, volatility)

        callable_bond_price = straight_bond_price - option_value

        return {
            "callable_bond_price": callable_bond_price,
            "straight_bond_price": straight_bond_price,
            "option_value": option_value,
            "option_adjusted_spread": self._calculate_oas(
                yield_curve, callable_bond_price
            ),
        }

    def _estimate_call_option_value(
        self, yield_curve: YieldCurve, volatility: float
    ) -> float:
        """Estimate embedded call option value."""

        # Simplified Black-Scholes approximation for bond options
        time_to_maturity = (self.bond.maturity_date - datetime.now()).days / 365.25

        if not self.call_schedule:
            return 0.0

        # Use first call date and price
        first_call = self.call_schedule[0]
        call_price = first_call["price"]
        time_to_call = (first_call["date"] - datetime.now()).days / 365.25

        if time_to_call <= 0:
            return 0.0

        # Current bond price (without call feature)
        current_price = BondPricer.price_bond_from_curve(
            self.bond, yield_curve, time_to_maturity
        )

        # Risk-free rate
        risk_free_rate = yield_curve.get_yield(time_to_call)

        # Simplified option value using Black-Scholes approximation
        try:
            from scipy.stats import norm

            d1 = (
                np.log(current_price / call_price)
                + (risk_free_rate + 0.5 * volatility**2) * time_to_call
            ) / (volatility * np.sqrt(time_to_call))
            d2 = d1 - volatility * np.sqrt(time_to_call)

            call_value = current_price * norm.cdf(d1) - call_price * np.exp(
                -risk_free_rate * time_to_call
            ) * norm.cdf(d2)

            return max(call_value, 0)

        except ImportError:
            # Fallback: simple intrinsic value
            return max(current_price - call_price, 0) * 0.5  # Rough approximation

    def _calculate_oas(self, yield_curve: YieldCurve, callable_price: float) -> float:
        """Calculate Option-Adjusted Spread."""

        # Find spread that makes theoretical price equal market price
        def objective(spread):
            # Shift yield curve by spread
            shifted_curve = YieldCurve(yield_curve.curve_date)
            for point in yield_curve.points:
                shifted_curve.add_point(point.maturity, point.yield_rate + spread)
            shifted_curve.build_curve(yield_curve.method)

            # Price bond with shifted curve
            theoretical_price = BondPricer.price_bond_from_curve(
                self.bond, shifted_curve
            )

            return (theoretical_price - callable_price) ** 2

        try:
            result = minimize_scalar(objective, bounds=(-0.1, 0.1), method="bounded")
            return result.x
        except Exception:
            return 0.0


class MortgageBackedSecurity:
    """Mortgage-Backed Security pricing and analytics."""

    def __init__(
        self,
        principal_balance: float,
        coupon_rate: float,
        maturity_years: int,
        prepayment_model: str = "PSA",
    ):
        self.principal_balance = principal_balance
        self.coupon_rate = coupon_rate
        self.maturity_years = maturity_years
        self.prepayment_model = prepayment_model

    def calculate_cash_flows(self, psa_speed: float = 100) -> pd.DataFrame:
        """
        Calculate MBS cash flows with prepayment model.

        Args:
            psa_speed: PSA prepayment speed (100 = 100% PSA)

        Returns:
            DataFrame with monthly cash flows
        """
        months = self.maturity_years * 12
        monthly_rate = self.coupon_rate / 12

        # Initialize arrays
        periods = np.arange(1, months + 1)
        remaining_balance = np.zeros(months + 1)
        remaining_balance[0] = self.principal_balance

        scheduled_principal = np.zeros(months)
        prepayments = np.zeros(months)
        interest_payments = np.zeros(months)

        # Calculate scheduled payment
        if monthly_rate > 0:
            scheduled_payment = (
                self.principal_balance * monthly_rate * (1 + monthly_rate) ** months
            ) / ((1 + monthly_rate) ** months - 1)
        else:
            scheduled_payment = self.principal_balance / months

        for month in range(months):
            if remaining_balance[month] <= 0:
                break

            # Interest payment
            interest_payments[month] = remaining_balance[month] * monthly_rate

            # Scheduled principal payment
            scheduled_principal[month] = scheduled_payment - interest_payments[month]

            # Prepayment calculation (PSA model)
            if self.prepayment_model == "PSA":
                # PSA ramp: 0.2% in month 1, increasing by 0.2% each month until 6% at month 30
                if month < 30:
                    cpr = 0.002 * (month + 1) * (psa_speed / 100)
                else:
                    cpr = 0.06 * (psa_speed / 100)

                # Convert CPR to SMM (Single Monthly Mortality)
                smm = 1 - (1 - cpr) ** (1 / 12)

                # Prepayment amount
                prepayments[month] = (
                    remaining_balance[month] - scheduled_principal[month]
                ) * smm

            # Update remaining balance
            remaining_balance[month + 1] = (
                remaining_balance[month]
                - scheduled_principal[month]
                - prepayments[month]
            )

        # Create cash flow DataFrame
        cash_flows = pd.DataFrame(
            {
                "Month": periods,
                "Beginning_Balance": remaining_balance[:-1],
                "Scheduled_Principal": scheduled_principal,
                "Prepayments": prepayments,
                "Interest": interest_payments,
                "Total_Principal": scheduled_principal + prepayments,
                "Total_Cash_Flow": scheduled_principal
                + prepayments
                + interest_payments,
                "Ending_Balance": remaining_balance[1:],
            }
        )

        return cash_flows

    def price_mbs(
        self, discount_rate: float, psa_speed: float = 100
    ) -> Dict[str, float]:
        """Price MBS given discount rate and prepayment speed."""

        cash_flows = self.calculate_cash_flows(psa_speed)

        # Present value of cash flows
        pv = 0
        for month, cf in enumerate(cash_flows["Total_Cash_Flow"], 1):
            pv += cf / (1 + discount_rate / 12) ** month

        # Calculate weighted average life
        principal_flows = cash_flows["Total_Principal"]
        total_principal = principal_flows.sum()

        if total_principal > 0:
            wal = (
                sum(
                    month * principal / total_principal
                    for month, principal in enumerate(principal_flows, 1)
                )
                / 12
            )
        else:
            wal = self.maturity_years

        return {
            "price": pv,
            "weighted_average_life": wal,
            "total_cash_flows": cash_flows["Total_Cash_Flow"].sum(),
            "total_interest": cash_flows["Interest"].sum(),
            "total_principal": total_principal,
        }


class BondAnalytics:
    """Comprehensive bond analytics and risk measures."""

    @staticmethod
    def calculate_key_rate_durations(
        bond: Bond, yield_curve: YieldCurve, key_rates: List[float] = None
    ) -> Dict[str, float]:
        """Calculate key rate durations."""

        if key_rates is None:
            key_rates = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]  # Standard key rates

        base_price = BondPricer.price_bond_from_curve(bond, yield_curve)

        key_rate_durations = {}
        shock_size = 0.0001  # 1 basis point

        for key_rate in key_rates:
            # Create shocked curve
            shocked_curve = YieldCurve(yield_curve.curve_date)
            for point in yield_curve.points:
                if abs(point.maturity - key_rate) < 0.1:  # Shock nearby points
                    shocked_curve.add_point(
                        point.maturity, point.yield_rate + shock_size
                    )
                else:
                    shocked_curve.add_point(point.maturity, point.yield_rate)

            shocked_curve.build_curve(yield_curve.method)

            # Calculate price with shocked curve
            shocked_price = BondPricer.price_bond_from_curve(bond, shocked_curve)

            # Key rate duration
            krd = -(shocked_price - base_price) / (base_price * shock_size)
            key_rate_durations[f"KRD_{key_rate}Y"] = krd

        return key_rate_durations

    @staticmethod
    def calculate_effective_duration(
        bond: Bond, yield_curve: YieldCurve, shock_size: float = 0.01
    ) -> float:
        """Calculate effective duration using yield curve shifts."""

        base_price = BondPricer.price_bond_from_curve(bond, yield_curve)

        # Create up and down shocked curves
        up_curve = YieldCurve(yield_curve.curve_date)
        down_curve = YieldCurve(yield_curve.curve_date)

        for point in yield_curve.points:
            up_curve.add_point(point.maturity, point.yield_rate + shock_size)
            down_curve.add_point(point.maturity, point.yield_rate - shock_size)

        up_curve.build_curve(yield_curve.method)
        down_curve.build_curve(yield_curve.method)

        # Calculate prices
        up_price = BondPricer.price_bond_from_curve(bond, up_curve)
        down_price = BondPricer.price_bond_from_curve(bond, down_curve)

        # Effective duration
        effective_duration = (down_price - up_price) / (2 * base_price * shock_size)

        return effective_duration

    @staticmethod
    def calculate_effective_convexity(
        bond: Bond, yield_curve: YieldCurve, shock_size: float = 0.01
    ) -> float:
        """Calculate effective convexity using yield curve shifts."""

        base_price = BondPricer.price_bond_from_curve(bond, yield_curve)

        # Create up and down shocked curves
        up_curve = YieldCurve(yield_curve.curve_date)
        down_curve = YieldCurve(yield_curve.curve_date)

        for point in yield_curve.points:
            up_curve.add_point(point.maturity, point.yield_rate + shock_size)
            down_curve.add_point(point.maturity, point.yield_rate - shock_size)

        up_curve.build_curve(yield_curve.method)
        down_curve.build_curve(yield_curve.method)

        # Calculate prices
        up_price = BondPricer.price_bond_from_curve(bond, up_curve)
        down_price = BondPricer.price_bond_from_curve(bond, down_curve)

        # Effective convexity
        effective_convexity = (up_price + down_price - 2 * base_price) / (
            base_price * shock_size**2
        )

        return effective_convexity

    @staticmethod
    def calculate_spread_duration(
        bond: Bond,
        yield_curve: YieldCurve,
        credit_spread: float,
        shock_size: float = 0.01,
    ) -> float:
        """Calculate spread duration (sensitivity to credit spread changes)."""

        # Base price with current spread
        base_yield = yield_curve.get_yield(
            (bond.maturity_date - datetime.now()).days / 365.25
        )
        base_corporate_yield = base_yield + credit_spread
        base_price = BondPricer.price_bond(bond, base_corporate_yield)["clean_price"]

        # Price with shocked spread
        shocked_corporate_yield = base_yield + credit_spread + shock_size
        shocked_price = BondPricer.price_bond(bond, shocked_corporate_yield)[
            "clean_price"
        ]

        # Spread duration
        spread_duration = -(shocked_price - base_price) / (base_price * shock_size)

        return spread_duration


# Utility functions for bond market analysis
def create_treasury_curve_from_data(treasury_data: Dict[str, float]) -> YieldCurve:
    """
    Create treasury yield curve from market data.

    Args:
        treasury_data: Dictionary with maturity (in years) as keys and yields as values

    Returns:
        YieldCurve object
    """
    curve = YieldCurve()

    for maturity_str, yield_rate in treasury_data.items():
        # Parse maturity (e.g., "3M", "1Y", "10Y")
        if maturity_str.endswith("M"):
            maturity = float(maturity_str[:-1]) / 12
        elif maturity_str.endswith("Y"):
            maturity = float(maturity_str[:-1])
        else:
            maturity = float(maturity_str)

        curve.add_point(maturity, yield_rate)

    curve.build_curve("cubic_spline")
    return curve


def calculate_bond_equivalent_yield(
    discount_rate: float, days_to_maturity: int
) -> float:
    """Convert discount rate to bond equivalent yield."""

    if days_to_maturity <= 0:
        return 0.0

    # Bond equivalent yield formula
    bey = (365 * discount_rate) / (360 - discount_rate * days_to_maturity)

    return bey


def calculate_current_yield(bond: Bond, market_price: float) -> float:
    """Calculate current yield (annual coupon / market price)."""

    annual_coupon = bond.coupon_rate * bond.face_value
    return annual_coupon / market_price


def calculate_yield_to_call(
    bond: Bond, market_price: float, call_date: datetime, call_price: float
) -> float:
    """Calculate yield to call."""

    time_to_call = (call_date - datetime.now()).days / 365.25

    if time_to_call <= 0:
        return 0.0

    # Create temporary bond with call date as maturity
    temp_bond = Bond(
        face_value=call_price,
        coupon_rate=bond.coupon_rate,
        maturity_date=call_date,
        coupon_frequency=bond.coupon_frequency,
    )

    return BondPricer.calculate_yield_to_maturity(temp_bond, market_price)


def calculate_after_tax_yield(pre_tax_yield: float, tax_rate: float) -> float:
    """Calculate after-tax yield."""
    return pre_tax_yield * (1 - tax_rate)


def calculate_taxable_equivalent_yield(tax_free_yield: float, tax_rate: float) -> float:
    """Calculate taxable equivalent yield for municipal bonds."""
    return tax_free_yield / (1 - tax_rate)
