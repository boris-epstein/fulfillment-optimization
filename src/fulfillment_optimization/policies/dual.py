"""Dual mirror descent fulfillment policy."""

from collections import defaultdict
from typing import List

import numpy as np

from ..graph import Graph, DemandNode
from ..demand import Sequence
from ..inventory import Inventory
from .base import FulfillmentResult


class DualMirrorDescentPolicy:
    """Online fulfillment using dual mirror descent for adaptive opportunity costs.

    Updates dual variables (opportunity costs) after each fulfillment decision
    using either subgradient descent or multiplicative weights.
    """

    def __init__(self, graph: Graph):
        self.graph = graph

    def fulfill(self, sequence: Sequence, inventory: Inventory,
                dual_solution: List[float], rho: List[float],
                step_size: float, update_rule: str = 'subgradient_descent',
                verbose=False):
        """Fulfill a sequence with online dual updates.

        Args:
            sequence: Demand sequence.
            inventory: Initial inventory.
            dual_solution: Initial dual variable values.
            rho: Target consumption rate per supply node.
            step_size: Learning rate for dual updates.
            update_rule: 'subgradient_descent' or 'multiplicative_weights'.
            verbose: Print decisions if True.

        Returns:
            FulfillmentResult (supports tuple unpacking).
        """
        collected_rewards = 0
        lost_sales = 0

        number_fulfillments = defaultdict(int)

        current_dual_solution = dual_solution.copy()
        current_inventories = inventory.initial_inventory.copy()

        for request in sequence.requests:
            demand_node = self.graph.demand_nodes[request.demand_node]

            supply_node_chosen = self.choose_supply_node(demand_node, current_inventories, current_dual_solution)
            if supply_node_chosen == -1:
                lost_sales += 1
            else:
                if verbose:
                    print(f'Demand from {demand_node.id} fulfilled from warehouse {supply_node_chosen}')
                current_inventories[supply_node_chosen] -= 1
                number_fulfillments[supply_node_chosen, demand_node.id] += 1
                collected_rewards += self.graph.edges[supply_node_chosen, demand_node.id].reward


            current_dual_solution = self.dual_update(supply_node_chosen, current_dual_solution, rho, step_size, update_rule)

        return FulfillmentResult(number_fulfillments, collected_rewards, lost_sales)

    def choose_supply_node(self, demand_node: DemandNode, current_inventories, dual_solution):
        """Choose the supply node with the highest reward minus dual variable.

        Returns:
            Supply node ID, or -1 if no non-negative pseudo-reward exists or no inventory.
        """
        pseudo_rewards = [
            [supply_node_id,
             self.graph.edges[supply_node_id, demand_node.id].reward - dual_solution[supply_node_id]]
            for supply_node_id in demand_node.neighbors
        ]

        if not pseudo_rewards:
            return -1

        chosen_supply_node, max_pseudo_reward = max(pseudo_rewards, key=lambda x: x[1])

        if max_pseudo_reward >= 0 and current_inventories[chosen_supply_node] > 0:
            return chosen_supply_node
        else:
            return -1

    def dual_update(self, supply_node_chosen, dual_solution, rho, step_size, update_rule):
        """Update dual variables based on the fulfillment decision."""
        step = rho.copy()
        step[supply_node_chosen] -= 1

        if update_rule == 'subgradient_descent':
            new_dual_solution = self.subgradient_descent_update(dual_solution, step, step_size)
        elif update_rule == 'multiplicative_weights':
            new_dual_solution = self.multiplicative_weights_update(dual_solution, step, step_size)

        return new_dual_solution

    def subgradient_descent_update(self, dual_solution, step, step_size):
        """Projected subgradient descent: dual -= step_size * step, clipped to >= 0."""
        new_dual_solution = dual_solution.copy()
        for supply_node_id in self.graph.supply_nodes:
            new_dual_solution[supply_node_id] = max(new_dual_solution[supply_node_id] - step_size * step[supply_node_id], 0)
        return new_dual_solution

    def multiplicative_weights_update(self, dual_solution, step, step_size):
        """Multiplicative weights: dual *= exp(-step_size * step)."""
        new_dual_solution = dual_solution.copy()
        for supply_node_id in self.graph.supply_nodes:
            new_dual_solution[supply_node_id] = new_dual_solution[supply_node_id] * np.exp(-step_size * step[supply_node_id])
        return new_dual_solution


# Backward compatibility alias
DualMirrorDescentFulfillment = DualMirrorDescentPolicy
