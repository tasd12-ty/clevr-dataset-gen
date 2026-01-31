"""
ORDINAL-SPATIAL: A Benchmark for Relational Spatial Intelligence in VLMs

This benchmark evaluates spatial understanding through comparative ordinal
constraints (QRR/TRR) rather than absolute measurements.
"""

__version__ = "0.1.0"
__author__ = "ORDINAL-SPATIAL Team"

from ordinal_spatial.dsl.comparators import Comparator, compare, TolerancePreset
from ordinal_spatial.dsl.predicates import (
    QRRConstraint,
    TRRConstraint,
    MetricType,
    compute_qrr,
    compute_trr,
)
from ordinal_spatial.dsl.schema import OrdinalSceneDescription, ObjectSpec

__all__ = [
    "Comparator",
    "compare",
    "TolerancePreset",
    "QRRConstraint",
    "TRRConstraint",
    "MetricType",
    "compute_qrr",
    "compute_trr",
    "OrdinalSceneDescription",
    "ObjectSpec",
]
