"""Base classes for fulfillment policies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class FulfillmentResult:
    """Result of fulfilling a demand sequence.

    Supports tuple unpacking for backward compatibility:
        ``number_fulfillments, reward, lost = policy.fulfill(...)``

    Attributes:
        number_fulfillments: Dict mapping (supply_id, demand_id) to count.
        collected_rewards: Total reward collected.
        lost_sales: Number of unfulfilled requests.
    """
    number_fulfillments: Dict[Tuple[int, int], int]
    collected_rewards: float
    lost_sales: int

    def __iter__(self):
        return iter((self.number_fulfillments, self.collected_rewards, self.lost_sales))


def extended_division(a, b):
    """Safe division returning 1 when the denominator is 0."""
    if b == 0:
        return 1
    else:
        return float(a) / float(b)
