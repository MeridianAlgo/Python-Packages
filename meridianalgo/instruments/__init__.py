"""
Financial Instruments module for MeridianAlgo.
"""

# Import instrument-specific logic
try:
    from .crypto import CryptoExchangeAdapter, CryptoPricer  # placeholder or actual
    from .forex import ForexPricer, FXRateProvider
except ImportError:
    pass

__all__ = ["CryptoPricer", "ForexPricer", "CryptoExchangeAdapter", "FXRateProvider"]
