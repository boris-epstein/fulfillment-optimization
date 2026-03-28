"""Backward compatibility shim — use inventory and policies modules directly."""

from .inventory import Inventory, InventoryOptimizer, FormatConverter
from .policies import (
    FulfillmentResult, extended_division,
    Fulfillment, PriorityListPolicy,
    BalanceFulfillment, BalancePolicy,
    MultiPriceBalanceFulfillment, MultiPriceBalancePolicy,
    OffLpReSolvingFulfillment, OffLpReSolvingPolicy,
    FluLpReSolvingFulfillment, FluLpReSolvingPolicy,
    ExtrapolationLpReSolvingFulfillment, ExtrapolationLpReSolvingPolicy,
    DualMirrorDescentFulfillment, DualMirrorDescentPolicy,
    PolicyFulfillment, DPPolicy,
    inventory_dict_to_tuple,
)
