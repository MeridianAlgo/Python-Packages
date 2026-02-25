"""
Factor modeling module for MeridianAlgo.
"""

from .core import (AlphaModel, FactorModel, FactorRiskDecomposition,
                   FamaFrenchModel, StyleAnalysis)

# Alias
FamaFrench = FamaFrenchModel

__all__ = [
    "FactorModel",
    "FamaFrench",
    "AlphaModel",
    "FactorRiskDecomposition",
    "StyleAnalysis"
]
