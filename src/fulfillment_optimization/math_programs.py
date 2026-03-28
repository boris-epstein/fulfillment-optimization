"""Backward compatibility shim — use lp and solver modules directly."""

from .lp import LPResult, ConstraintResult, VariableResult, LPBuilder, MathPrograms
from .solver import solve
