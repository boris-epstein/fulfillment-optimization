"""Backward compatibility shim — use dp and estimation modules directly."""

from .dp import (
    IndependentDynamicProgram,
    MarkovianDynamicProgram,
    DPOutput,
    generate_tuples,
    generate_bounded_tuples,
    generate_bounded_tuples_with_sum,
)
from .estimation import ModelEstimator
