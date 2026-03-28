"""Dynamic-programming policy fulfillment."""

from collections import defaultdict
from typing import Dict, Tuple

from ..graph import Graph
from ..demand import Sequence
from ..inventory import Inventory
from .base import FulfillmentResult


def inventory_dict_to_tuple(inventory_dict: Dict[int, int], graph: Graph) -> Tuple[int]:
    """Convert an inventory dict to a tuple ordered by supply node IDs."""
    inventory_list = [inventory_dict[supply_node_id] for supply_node_id in graph.supply_nodes]
    return tuple(inventory_list)


class DPPolicy:
    """Fulfills demand using a precomputed policy mapping (state, time, demand) -> action.

    Used to execute optimal policies computed by dynamic programming.
    """

    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def fulfill(self, sequence: Sequence, inventory: Inventory, policy, verbose=False):
        """Fulfill a sequence by looking up the optimal action at each state.

        Args:
            sequence: Demand sequence.
            inventory: Initial inventory.
            policy: Dict mapping (inventory_tuple, t, demand_node_id) to supply_node_id
                (-1 means no fulfillment).
            verbose: Print decisions if True.

        Returns:
            FulfillmentResult (supports tuple unpacking).
        """
        collected_rewards = 0
        lost_sales = 0
        total_fulfillments = 0
        number_fulfillments = defaultdict(int)

        current_inventories = inventory.initial_inventory.copy()

        t = 0
        for request in sequence.requests:
            demand_node = self.graph.demand_nodes[request.demand_node]
            inventory_state = inventory_dict_to_tuple(current_inventories, self.graph)
            supply_node_chosen = policy[inventory_state, t, demand_node.id]

            if supply_node_chosen == -1:
                lost_sales += 1
                if verbose:
                    print(f'Demand from {demand_node.id} lost')
            else:
                if verbose:
                    print(f'Demand from {demand_node.id} fulfilled from warehouse {supply_node_chosen}')
                current_inventories[supply_node_chosen] -= 1
                number_fulfillments[supply_node_chosen, demand_node.id] += 1
                collected_rewards += self.graph.edges[supply_node_chosen, demand_node.id].reward
                total_fulfillments += 1

            t += 1

        return FulfillmentResult(number_fulfillments, collected_rewards, lost_sales)


# Backward compatibility alias
PolicyFulfillment = DPPolicy
