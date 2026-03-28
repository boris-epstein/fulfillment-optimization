"""Priority-list-based fulfillment policy."""

from collections import defaultdict
from typing import Dict, List, Tuple

from ..graph import Graph, DemandNode
from ..demand import Sequence
from ..inventory import Inventory
from .base import FulfillmentResult


class PriorityListPolicy:
    """Fulfillment engine using fixed priority lists.

    For each demand request, assigns it to the highest-priority supply node
    that still has inventory, following a precomputed priority list.
    """

    def __init__(self, graph: Graph, **kwargs) -> None:
        self.graph = graph

    def fixed_list_fulfillment(self, sequence: Sequence, inventory: Inventory,
                               priority_list_name: Tuple[str, str],
                               allow_rejections: bool = True, verbose=False):
        """Fulfill a sequence by following a precomputed priority list.

        Args:
            sequence: Demand sequence to fulfill.
            inventory: Initial inventory.
            priority_list_name: Key identifying which priority list to use.
            allow_rejections: Whether to allow rejecting demand.
            verbose: Print fulfillment decisions if True.

        Returns:
            FulfillmentResult (supports tuple unpacking).
        """
        collected_rewards = 0
        lost_sales = 0

        number_fulfillments = defaultdict(int)

        current_inventories = inventory.initial_inventory.copy()
        demand_node_reachable = defaultdict(lambda: True)
        best_supply_node_with_inventory = defaultdict(int)

        for request in sequence.requests:
            demand_node = self.graph.demand_nodes[request.demand_node]
            (best_supply_node_with_inventory, demand_node_reachable) = self.update_feasible_priorities(
                demand_node=demand_node,
                priority_list=demand_node.priority_lists[priority_list_name],
                best_supply_node_with_inventory=best_supply_node_with_inventory,
                current_inventories=current_inventories,
                demand_node_reachable=demand_node_reachable
            )
            if demand_node_reachable[demand_node.id]:
                supply_node_id = demand_node.priority_lists[priority_list_name][best_supply_node_with_inventory[demand_node.id]]
                current_inventories[supply_node_id] -= 1
                number_fulfillments[supply_node_id, demand_node.id] += 1
                collected_rewards += self.graph.edges[supply_node_id, demand_node.id].reward

                if verbose:
                    print(f'Demand from {demand_node.id} fulfilled from {supply_node_id}')
            else:
                lost_sales += 1
                if verbose:
                    print(f'Demand from {demand_node.id} lost')

        return FulfillmentResult(number_fulfillments, collected_rewards, lost_sales)

    def update_feasible_priorities(self, demand_node: DemandNode, priority_list: List[int],
                                   best_supply_node_with_inventory: Dict[int, int],
                                   current_inventories: Dict[int, int],
                                   demand_node_reachable: Dict[int, bool]):
        """Advance the priority list pointer past depleted supply nodes.

        Returns:
            Updated (best_supply_node_with_inventory, demand_node_reachable).
        """
        while (best_supply_node_with_inventory[demand_node.id] < len(priority_list)
               and current_inventories[priority_list[best_supply_node_with_inventory[demand_node.id]]] <= 0):
            best_supply_node_with_inventory[demand_node.id] += 1
        if best_supply_node_with_inventory[demand_node.id] == len(priority_list):
            demand_node_reachable[demand_node.id] = False

        return best_supply_node_with_inventory, demand_node_reachable


# Backward compatibility alias
Fulfillment = PriorityListPolicy
