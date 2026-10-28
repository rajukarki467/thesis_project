"""
Re-export FastKAN model classes for backwards compatibility.
"""

from src.models.fastkan_model import (
    FastKANClassifier,
    FastKANLayer,
    SUPPORTED_GRID_SIZES,
    SUPPORTED_SPLINE_ORDERS,
)

__all__ = [
    "FastKANClassifier",
    "FastKANLayer",
    "SUPPORTED_GRID_SIZES",
    "SUPPORTED_SPLINE_ORDERS",
]
