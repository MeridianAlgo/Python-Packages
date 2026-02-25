"""
Derivatives pricing and analysis module for MeridianAlgo.
"""

from .core import (OptionsPricer, VolatilitySurface, ExoticOptions, 
                   FuturesPricer)

# Global instances/aliases for convenience
pricer = OptionsPricer()
surface = VolatilitySurface()

BlackScholes = pricer.black_scholes_merton
MonteCarloPricer = pricer.monte_carlo_pricing
GreeksCalculator = pricer.calculate_greeks
ImpliedVolatility = pricer.calculate_implied_volatility
OptionChain = surface.construct_volatility_surface

__all__ = [
    "OptionsPricer",
    "VolatilitySurface",
    "ExoticOptions",
    "FuturesPricer",
    "BlackScholes",
    "MonteCarloPricer",
    "GreeksCalculator",
    "ImpliedVolatility",
    "OptionChain"
]
