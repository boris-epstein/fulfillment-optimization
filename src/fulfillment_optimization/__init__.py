"""Algorithms for online bipartite matching applied to fulfillment optimization.

This package provides implementations of inventory placement procedures,
fulfillment policies, and demand models for studying the joint placement
and fulfillment problem in e-commerce supply chains.
"""

# --- Graph data structures ---
from .graph import Graph, Node, DemandNode, Edge, RandomGraphGenerator

# --- Demand models ---
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

# --- Inventory ---
from .inventory import Inventory, InventoryOptimizer, FormatConverter

# --- LP formulations ---
from .lp import MathPrograms, LPBuilder, LPResult

# --- Fulfillment policies ---
from .policies import (
    FulfillmentResult,
    PriorityListPolicy, Fulfillment,
    BalancePolicy, BalanceFulfillment,
    MultiPriceBalancePolicy, MultiPriceBalanceFulfillment,
    OffLpReSolvingPolicy, OffLpReSolvingFulfillment,
    FluLpReSolvingPolicy, FluLpReSolvingFulfillment,
    ExtrapolationLpReSolvingPolicy, ExtrapolationLpReSolvingFulfillment,
    DualMirrorDescentPolicy, DualMirrorDescentFulfillment,
    DPPolicy, PolicyFulfillment,
)

# --- Dynamic programming ---
from .dp import IndependentDynamicProgram, MarkovianDynamicProgram, DPOutput

# --- Model estimation ---
from .estimation import ModelEstimator

# --- Utilities ---
from .utils import correl_graph

# model_free / policies.learned requires nevergrad and torch — import directly:
#   from fulfillment_optimization.model_free import ThresholdsFulfillment, ...
#   from fulfillment_optimization.policies.learned import ThresholdsPolicy, ...

__version__ = "0.1.0"
