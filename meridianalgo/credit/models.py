"""
Credit Risk Models

Merton structural model for default probability and asset valuation,
CDS pricing, Z-spread calculation, and expected loss analytics.

References:
    Merton (1974) - On the Pricing of Corporate Debt
    Hull (2012) - Options, Futures, and Other Derivatives
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq, fsolve
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class DefaultProbability:
    """Container for default probability estimates."""

    risk_neutral_pd: float
    physical_pd: Optional[float]
    distance_to_default: float
    expected_recovery: float
    confidence_interval: Tuple[float, float]
    horizon_years: float


@dataclass
class CDSResult:
    """Container for CDS pricing results."""

    fair_spread: float
    upfront_payment: float
    duration: float
    risky_annuity: float
    survival_probability: float
    implied_hazard_rate: float


class MertonModel:
    """
    Merton (1974) structural credit risk model.

    Models a firm's equity as a call option on its assets. Derives asset value,
    asset volatility, distance-to-default, and risk-neutral default probability
    from observable equity price and volatility.

    Example:
        >>> model = MertonModel(
        ...     equity_value=50.0,
        ...     equity_volatility=0.40,
        ...     debt_face_value=100.0,
        ...     time_to_maturity=1.0,
        ...     risk_free_rate=0.05,
        ... )
        >>> result = model.calibrate()
        >>> print(f"Distance to Default: {result['distance_to_default']:.4f}")
        >>> print(f"Default Probability: {result['default_probability']:.4%}")
    """

    def __init__(
        self,
        equity_value: float,
        equity_volatility: float,
        debt_face_value: float,
        time_to_maturity: float = 1.0,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
    ):
        """
        Initialize Merton model.

        Parameters
        ----------
        equity_value : float
            Market value of equity (total market cap)
        equity_volatility : float
            Annualized equity volatility (e.g., 0.40 for 40%)
        debt_face_value : float
            Face value of debt (total liabilities)
        time_to_maturity : float
            Time horizon in years (default 1.0)
        risk_free_rate : float
            Continuously compounded risk-free rate
        dividend_yield : float
            Continuous dividend yield
        """
        if equity_value <= 0:
            raise ValueError("equity_value must be positive")
        if equity_volatility <= 0 or equity_volatility > 5:
            raise ValueError("equity_volatility must be between 0 and 5")
        if debt_face_value <= 0:
            raise ValueError("debt_face_value must be positive")

        self.equity_value = equity_value
        self.equity_volatility = equity_volatility
        self.debt_face_value = debt_face_value
        self.time_to_maturity = time_to_maturity
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield

        self._asset_value: Optional[float] = None
        self._asset_volatility: Optional[float] = None

    def _d1(self, asset_value: float, asset_volatility: float) -> float:
        """Black-Scholes d1 parameter for asset option."""
        return (
            np.log(asset_value / self.debt_face_value)
            + (
                self.risk_free_rate
                - self.dividend_yield
                + 0.5 * asset_volatility**2
            )
            * self.time_to_maturity
        ) / (asset_volatility * np.sqrt(self.time_to_maturity))

    def _d2(self, asset_value: float, asset_volatility: float) -> float:
        return self._d1(asset_value, asset_volatility) - asset_volatility * np.sqrt(
            self.time_to_maturity
        )

    def _equity_from_assets(self, asset_value: float, asset_volatility: float) -> float:
        """Black-Scholes call value = equity value."""
        d1 = self._d1(asset_value, asset_volatility)
        d2 = d1 - asset_volatility * np.sqrt(self.time_to_maturity)
        discount = np.exp(-self.risk_free_rate * self.time_to_maturity)
        return asset_value * np.exp(-self.dividend_yield * self.time_to_maturity) * norm.cdf(
            d1
        ) - self.debt_face_value * discount * norm.cdf(d2)

    def _equity_vol_from_assets(
        self, asset_value: float, asset_volatility: float
    ) -> float:
        """Equity volatility implied by asset parameters via Ito's lemma."""
        d1 = self._d1(asset_value, asset_volatility)
        delta = np.exp(-self.dividend_yield * self.time_to_maturity) * norm.cdf(d1)
        return (delta * asset_value * asset_volatility) / self.equity_value

    def calibrate(self) -> Dict[str, float]:
        """
        Calibrate asset value and asset volatility from observable equity data.

        Solves the two-equation system (Black-Scholes equity value and Ito's
        lemma equity-vol relation) simultaneously via Newton-Raphson iteration.

        Returns
        -------
        dict
            Keys: asset_value, asset_volatility, distance_to_default,
            default_probability, expected_recovery_rate, leverage_ratio
        """
        initial_asset_value = self.equity_value + self.debt_face_value
        initial_asset_vol = self.equity_volatility * self.equity_value / initial_asset_value

        def equations(params: np.ndarray) -> np.ndarray:
            Va, sigma_a = params
            if Va <= 0 or sigma_a <= 0:
                return np.array([1e6, 1e6])
            eq1 = self._equity_from_assets(Va, sigma_a) - self.equity_value
            eq2 = self._equity_vol_from_assets(Va, sigma_a) - self.equity_volatility
            return np.array([eq1, eq2])

        solution, info, ier, msg = fsolve(
            equations,
            [initial_asset_value, initial_asset_vol],
            full_output=True,
        )

        if ier != 1:
            warnings.warn(
                f"Merton calibration did not fully converge: {msg}",
                RuntimeWarning,
                stacklevel=2,
            )

        self._asset_value = max(solution[0], self.equity_value)
        self._asset_volatility = abs(solution[1])

        d1 = self._d1(self._asset_value, self._asset_volatility)
        d2 = self._d2(self._asset_value, self._asset_volatility)

        default_probability = norm.cdf(-d2)
        distance_to_default = d2
        expected_recovery = (
            self._asset_value
            * np.exp(-self.dividend_yield * self.time_to_maturity)
            * norm.cdf(-d1)
            / self.debt_face_value
            if default_probability > 0
            else 1.0
        )

        return {
            "asset_value": self._asset_value,
            "asset_volatility": self._asset_volatility,
            "distance_to_default": distance_to_default,
            "default_probability": default_probability,
            "expected_recovery_rate": expected_recovery,
            "leverage_ratio": self.debt_face_value / self._asset_value,
            "d1": d1,
            "d2": d2,
        }

    def default_probability_term_structure(
        self, horizons: List[float]
    ) -> pd.Series:
        """
        Compute default probability at multiple horizons.

        Parameters
        ----------
        horizons : list of float
            Time horizons in years (e.g., [0.5, 1, 2, 3, 5])

        Returns
        -------
        pd.Series
            Default probability indexed by horizon
        """
        if self._asset_value is None:
            self.calibrate()

        pds = {}
        original_t = self.time_to_maturity
        for t in horizons:
            self.time_to_maturity = t
            d2 = self._d2(self._asset_value, self._asset_volatility)
            pds[t] = norm.cdf(-d2)
        self.time_to_maturity = original_t

        return pd.Series(pds, name="default_probability")

    @property
    def asset_value(self) -> float:
        if self._asset_value is None:
            self.calibrate()
        return self._asset_value

    @property
    def asset_volatility(self) -> float:
        if self._asset_volatility is None:
            self.calibrate()
        return self._asset_volatility


class CreditDefaultSwap:
    """
    CDS (Credit Default Swap) pricing and credit spread analytics.

    Prices a CDS contract given a hazard rate or credit spread, and
    bootstraps a hazard rate curve from market CDS spreads.

    Example:
        >>> cds = CreditDefaultSwap(
        ...     hazard_rate=0.02,
        ...     recovery_rate=0.40,
        ...     risk_free_rate=0.05,
        ...     maturity=5.0,
        ... )
        >>> result = cds.price()
        >>> print(f"Fair CDS Spread: {result.fair_spread * 10000:.1f} bps")
    """

    def __init__(
        self,
        hazard_rate: float,
        recovery_rate: float = 0.40,
        risk_free_rate: float = 0.05,
        maturity: float = 5.0,
        payment_frequency: int = 4,
    ):
        """
        Parameters
        ----------
        hazard_rate : float
            Constant hazard rate (intensity) for default
        recovery_rate : float
            Fraction of face value recovered upon default (0.40 is market convention)
        risk_free_rate : float
            Continuously compounded risk-free rate
        maturity : float
            CDS maturity in years
        payment_frequency : int
            Premium payments per year (4 = quarterly, 2 = semi-annual)
        """
        if not 0 <= recovery_rate <= 1:
            raise ValueError("recovery_rate must be in [0, 1]")
        if hazard_rate < 0:
            raise ValueError("hazard_rate must be non-negative")

        self.hazard_rate = hazard_rate
        self.recovery_rate = recovery_rate
        self.risk_free_rate = risk_free_rate
        self.maturity = maturity
        self.payment_frequency = payment_frequency

    def survival_probability(self, t: float) -> float:
        """Survival probability at time t under constant hazard rate."""
        return np.exp(-self.hazard_rate * t)

    def discount_factor(self, t: float) -> float:
        """Risk-free discount factor at time t."""
        return np.exp(-self.risk_free_rate * t)

    def price(self) -> CDSResult:
        """
        Compute fair CDS spread and upfront payment.

        Uses the standard ISDA-style model: risky annuity (protection leg DV01)
        and protection leg (expected loss).

        Returns
        -------
        CDSResult
        """
        dt = 1.0 / self.payment_frequency
        payment_times = np.arange(dt, self.maturity + dt / 2, dt)

        risky_annuity = 0.0
        for t in payment_times:
            q = self.survival_probability(t)
            df = self.discount_factor(t)
            risky_annuity += dt * q * df

        protection_leg = 0.0
        n_steps = max(int(self.maturity * 52), 100)
        integration_times = np.linspace(0, self.maturity, n_steps + 1)
        for i in range(n_steps):
            t_mid = (integration_times[i] + integration_times[i + 1]) / 2
            dt_int = integration_times[i + 1] - integration_times[i]
            q = self.survival_probability(t_mid)
            df = self.discount_factor(t_mid)
            protection_leg += (1 - self.recovery_rate) * self.hazard_rate * q * df * dt_int

        fair_spread = protection_leg / risky_annuity if risky_annuity > 0 else 0.0

        return CDSResult(
            fair_spread=fair_spread,
            upfront_payment=0.0,
            duration=risky_annuity,
            risky_annuity=risky_annuity,
            survival_probability=self.survival_probability(self.maturity),
            implied_hazard_rate=self.hazard_rate,
        )

    @classmethod
    def from_spread(
        cls,
        spread: float,
        recovery_rate: float = 0.40,
        risk_free_rate: float = 0.05,
        maturity: float = 5.0,
        payment_frequency: int = 4,
    ) -> "CreditDefaultSwap":
        """
        Create CDS instance from market spread (bootstraps constant hazard rate).

        Parameters
        ----------
        spread : float
            Market CDS spread in decimal (0.01 = 100 bps)

        Returns
        -------
        CreditDefaultSwap
        """
        def objective(h: float) -> float:
            cds = cls(h, recovery_rate, risk_free_rate, maturity, payment_frequency)
            return cds.price().fair_spread - spread

        try:
            hazard_rate = brentq(objective, 1e-6, 5.0, xtol=1e-8)
        except ValueError:
            hazard_rate = spread / (1 - recovery_rate)

        return cls(hazard_rate, recovery_rate, risk_free_rate, maturity, payment_frequency)

    @staticmethod
    def bootstrap_hazard_curve(
        maturities: List[float],
        spreads: List[float],
        recovery_rate: float = 0.40,
        risk_free_rate: float = 0.05,
    ) -> pd.Series:
        """
        Bootstrap piecewise-constant hazard rate curve from market CDS spreads.

        Parameters
        ----------
        maturities : list of float
            CDS maturities in years (e.g., [1, 3, 5, 7, 10])
        spreads : list of float
            Market CDS spreads in decimal (e.g., [0.01, 0.015, 0.02])

        Returns
        -------
        pd.Series
            Hazard rates indexed by maturity
        """
        hazard_rates = {}
        for maturity, spread in zip(sorted(maturities), spreads):
            cds = CreditDefaultSwap.from_spread(
                spread, recovery_rate, risk_free_rate, maturity
            )
            hazard_rates[maturity] = cds.hazard_rate
        return pd.Series(hazard_rates, name="hazard_rate")


class CreditRiskAnalyzer:
    """
    Portfolio-level credit risk metrics: PD, LGD, EAD, EL, UL, and CVaR.

    Example:
        >>> analyzer = CreditRiskAnalyzer()
        >>> el = analyzer.expected_loss(pd=0.02, lgd=0.45, ead=1_000_000)
        >>> print(f"Expected Loss: ${el:,.0f}")
    """

    @staticmethod
    def expected_loss(pd: float, lgd: float, ead: float) -> float:
        """
        Expected loss = PD * LGD * EAD.

        Parameters
        ----------
        pd : float
            Probability of default
        lgd : float
            Loss given default (1 - recovery rate)
        ead : float
            Exposure at default (notional or market value)
        """
        return pd * lgd * ead

    @staticmethod
    def unexpected_loss(
        pd: float, lgd: float, ead: float, lgd_volatility: float = 0.0
    ) -> float:
        """
        Unexpected loss at 1-sigma confidence.

        UL = EAD * sqrt(PD * lgd_vol^2 + lgd^2 * PD * (1 - PD))

        Parameters
        ----------
        lgd_volatility : float
            Standard deviation of LGD
        """
        variance = pd * lgd_volatility**2 + lgd**2 * pd * (1 - pd)
        return ead * np.sqrt(variance)

    @staticmethod
    def credit_var(
        pd: float,
        lgd: float,
        ead: float,
        confidence: float = 0.999,
        rho: float = 0.12,
    ) -> float:
        """
        Basel II/III IRB credit VaR using the single-factor Vasicek model.

        Parameters
        ----------
        confidence : float
            Confidence level (e.g., 0.999 for 99.9%)
        rho : float
            Asset correlation parameter (0.12 typical for corporate)

        Returns
        -------
        float
            Credit VaR (unexpected loss at confidence level)
        """
        n_pd = norm.ppf(pd)
        n_conf = norm.ppf(confidence)
        conditional_pd = norm.cdf(
            (n_pd + np.sqrt(rho) * n_conf) / np.sqrt(1 - rho)
        )
        wcdr = conditional_pd
        return ead * lgd * (wcdr - pd)

    def portfolio_expected_loss(
        self,
        exposures: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Compute portfolio-level EL, UL, and concentration metrics.

        Parameters
        ----------
        exposures : pd.DataFrame
            Columns: pd, lgd, ead (one row per obligor)

        Returns
        -------
        dict
            Keys: total_el, total_ul, herfindahl_index, top10_concentration
        """
        required = {"pd", "lgd", "ead"}
        if not required.issubset(exposures.columns):
            raise ValueError(f"exposures must contain columns: {required}")

        els = exposures.apply(
            lambda r: self.expected_loss(r["pd"], r["lgd"], r["ead"]), axis=1
        )
        uls = exposures.apply(
            lambda r: self.unexpected_loss(r["pd"], r["lgd"], r["ead"]), axis=1
        )

        total_ead = exposures["ead"].sum()
        weights = exposures["ead"] / total_ead if total_ead > 0 else exposures["ead"] * 0

        herfindahl = (weights**2).sum()
        top10 = weights.nlargest(10).sum()

        return {
            "total_el": els.sum(),
            "total_ul": float(np.sqrt((uls**2).sum())),
            "el_rate": els.sum() / total_ead if total_ead > 0 else 0.0,
            "herfindahl_index": herfindahl,
            "top10_concentration": top10,
            "n_obligors": len(exposures),
        }


class ZSpreadCalculator:
    """
    Z-spread and OAS (Option-Adjusted Spread) computation.

    Z-spread is the constant spread added to the risk-free spot curve
    that equates a bond's discounted cash flows to its market price.

    Example:
        >>> calc = ZSpreadCalculator(
        ...     cash_flows=[5, 5, 5, 5, 105],
        ...     times=[1, 2, 3, 4, 5],
        ...     risk_free_rates=[0.03, 0.035, 0.038, 0.04, 0.042],
        ... )
        >>> z = calc.z_spread(market_price=98.5)
        >>> print(f"Z-Spread: {z * 10000:.1f} bps")
    """

    def __init__(
        self,
        cash_flows: List[float],
        times: List[float],
        risk_free_rates: List[float],
    ):
        """
        Parameters
        ----------
        cash_flows : list of float
            Bond cash flows (coupons + face value at maturity)
        times : list of float
            Cash flow payment times in years
        risk_free_rates : list of float
            Spot risk-free rates at each cash flow time
        """
        if len(cash_flows) != len(times) or len(times) != len(risk_free_rates):
            raise ValueError("cash_flows, times, and risk_free_rates must have equal length")

        self.cash_flows = np.array(cash_flows)
        self.times = np.array(times)
        self.risk_free_rates = np.array(risk_free_rates)

    def theoretical_price(self, spread: float = 0.0) -> float:
        """Discount cash flows at risk-free rates plus a constant spread."""
        discount_rates = self.risk_free_rates + spread
        discount_factors = np.exp(-discount_rates * self.times)
        return float(np.sum(self.cash_flows * discount_factors))

    def z_spread(self, market_price: float) -> float:
        """
        Solve for Z-spread given market price.

        Parameters
        ----------
        market_price : float
            Observed market price of the bond

        Returns
        -------
        float
            Z-spread in decimal (divide by 0.0001 for basis points)
        """
        def objective(spread: float) -> float:
            return self.theoretical_price(spread) - market_price

        try:
            z = brentq(objective, -0.20, 0.50, xtol=1e-8)
        except ValueError:
            price_at_zero = self.theoretical_price(0.0)
            z = (price_at_zero - market_price) / (
                sum(t * cf * np.exp(-r * t) for cf, t, r in zip(
                    self.cash_flows, self.times, self.risk_free_rates
                ))
            )
        return z

    def dv01(self, spread: float = 0.0) -> float:
        """Dollar value of 1 basis point (0.0001) change in spread."""
        price_up = self.theoretical_price(spread + 0.0001)
        price_down = self.theoretical_price(spread - 0.0001)
        return (price_down - price_up) / 2
