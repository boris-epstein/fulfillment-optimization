"""Algorithms for online bipartite matching applied to fulfillment optimization.

This package provides implementations of inventory placement procedures,
fulfillment policies, and demand models for studying the joint placement
and fulfillment problem in e-commerce supply chains.
"""

from .graph import Graph, Node, DemandNode, Edge, RandomGraphGenerator
from .demand import (
    Request,
    Sequence,
    CorrelGenerator,
    TemporalIndependenceGenerator,
    IndepGenerator,
    RWGenerator,
    MarkovianGenerator,
    HiddenMarkovGenerator,
    RandomDistributionGenerator,
)
from .fulfillment import (
    Inventory,
    InventoryOptimizer,
    Fulfillment,
    BalanceFulfillment,
    MultiPriceBalanceFulfillment,
    OffLpReSolvingFulfillment,
    FluLpReSolvingFulfillment,
    ExtrapolationLpReSolvingFulfillment,
    DualMirrorDescentFulfillment,
    PolicyFulfillment,
    FormatConverter,
)
from .math_programs import MathPrograms, LPBuilder, LPResult
from .model_based import (
    IndependentDynamicProgram,
    MarkovianDynamicProgram,
    DPOutput,
    ModelEstimator,
)
from .utils import correl_graph

# model_free requires nevergrad and torch — import directly:
#   from fulfillment_optimization.model_free import ThresholdsFulfillment, ...

__version__ = "0.1.0"
