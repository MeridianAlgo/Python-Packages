"""
Factor modeling module for MeridianAlgo.
"""

from .core import (
                   AlphaCapture,
                   APTModel,
                   CustomFactorModel,
                   FactorRiskDecomposition,
                   FamaFrenchModel,
)

# Aliases
FamaFrench = FamaFrenchModel
AlphaModel = AlphaCapture  # Maintenance alias

__all__ = [
    "APTModel",
    "AlphaCapture",
    "AlphaModel",
    "CustomFactorModel",
    "FactorRiskDecomposition",
    "FamaFrenchModel",
    "FamaFrench",
]
