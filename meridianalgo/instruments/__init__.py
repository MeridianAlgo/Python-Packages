"""
Financial Instruments module for MeridianAlgo.
"""

# Import instrument-specific logic
try:
    from .crypto import CryptoPricer, CryptoExchangeAdapter # placeholder or actual
    from .forex import ForexPricer, FXRateProvider
except ImportError:
    pass

__all__ = [
    "CryptoPricer",
    "ForexPricer"
]
