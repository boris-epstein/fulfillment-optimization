"""Balance-based fulfillment policies."""

from collections import defaultdict
from typing import Dict

import numpy as np

from ..graph import Graph, DemandNode
from ..demand import Sequence
from ..inventory import Inventory
from .base import FulfillmentResult, extended_division


class BalancePolicy:
    """Balance algorithm for online bipartite matching.

    Assigns each demand request to the supply node that maximizes
    reward * (1 - exp(used_fraction - 1)), which penalizes heavily-used
    supply nodes to maintain balance.
    """

    def __init__(self, graph: Graph):
        self.graph = graph

    def fulfill(self, sequence: Sequence, inventory: Inventory, verbose: bool = False):
        """Fulfill a demand sequence using the balance algorithm.

        Returns:
            FulfillmentResult (supports tuple unpacking).
        """
        collected_rewards = 0
        lost_sales = 0
        total_fulfillments = 0
        number_fulfillments = defaultdict(int)

        current_inventories = inventory.initial_inventory.copy()

        for request in sequence.requests:
            demand_node = self.graph.demand_nodes[request.demand_node]
            supply_node_chosen = self.choose_supply_node(demand_node, current_inventories, inventory)
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

        return FulfillmentResult(number_fulfillments, collected_rewards, lost_sales)

    def choose_supply_node(self, demand_node: DemandNode, current_inventories, inventory: Inventory):
        """Select the supply node maximizing balance-adjusted pseudo-reward.

        Returns:
            Supply node ID, or -1 if no supply node has inventory.
        """
        fractions = {
            supply_node_id: 1 - extended_division(current_inventories[supply_node_id], inventory.initial_inventory[supply_node_id])
            for supply_node_id in demand_node.neighbors
        }
        pseudo_rewards = [
            [supply_node_id,
             self.graph.edges[supply_node_id, demand_node.id].reward * (1 - np.exp(fractions[supply_node_id] - 1))]
            for supply_node_id in fractions
        ]

        pseudo_rewards.sort(key=lambda x: x[1], reverse=True)

        for supply_node_id, _ in pseudo_rewards:
            if current_inventories[supply_node_id] > 0:
                return supply_node_id

        return -1


class MultiPriceBalancePolicy:
    """Multi-price balance algorithm for heterogeneous edge rewards.

    Generalizes the balance algorithm to handle supply nodes with multiple
    distinct reward levels. Uses a piecewise-exponential opportunity cost
    function phi that adapts to the reward structure.
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self.compute_distinct_prices()
        self.compute_alphas()

    def compute_distinct_prices(self):
        """Identify the set of distinct reward values for each supply node."""
        self.num_distinct_prices = {}
        self.distinct_prices = {}

        for supply_node_id in self.graph.supply_nodes:
            distinct_price_set = set()
            supply_node = self.graph.supply_nodes[supply_node_id]
            for demand_node_id in supply_node.neighbors:
                edge = self.graph.edges[supply_node_id, demand_node_id]
                distinct_price_set.add(edge.reward)

            self.distinct_prices[supply_node_id] = sorted(list(distinct_price_set))
            self.num_distinct_prices[supply_node_id] = len(distinct_price_set)

    def compute_alphas(self):
        """Compute breakpoint parameters for the two-price phi function.

        For supply nodes with exactly two distinct prices, computes the
        optimal breakpoint alpha that balances acceptance rates.
        """
        self.alphas = defaultdict(list)
        for supply_node_id in self.graph.supply_nodes:
            if self.num_distinct_prices[supply_node_id] == 2:
                eps = self.distinct_prices[supply_node_id][1] / self.distinct_prices[supply_node_id][0]
                sqrt = np.sqrt(1 + 4 * eps * (eps - 1) / np.e)
                self.alphas[supply_node_id].append(-1 * np.log((sqrt - 1) / (2 * (eps - 1))))
                self.alphas[supply_node_id].append(1 - self.alphas[supply_node_id][0])

    def phi(self, supply_node_id: float, used_fraction: float):
        """Compute the opportunity cost for a supply node at a given usage level.

        For single-price nodes, uses a standard exponential cost.
        For two-price nodes, uses a piecewise-exponential with a breakpoint.

        Args:
            supply_node_id: The supply node.
            used_fraction: Fraction of initial inventory consumed (0 to 1).

        Returns:
            Opportunity cost value.
        """
        if self.num_distinct_prices[supply_node_id] == 1:
            return self.distinct_prices[supply_node_id][0] * (np.exp(used_fraction) - 1) / (np.e - 1)

        elif self.num_distinct_prices[supply_node_id] == 2:
            if used_fraction >= self.alphas[supply_node_id][0]:
                exponential_factor = (np.exp(used_fraction - self.alphas[supply_node_id][0]) - 1) / (np.exp(self.alphas[supply_node_id][1]) - 1)
                return self.distinct_prices[supply_node_id][0] + (self.distinct_prices[supply_node_id][1] - self.distinct_prices[supply_node_id][0]) * exponential_factor
            else:
                return self.distinct_prices[supply_node_id][0] * (np.exp(used_fraction) - 1) / (np.exp(self.alphas[supply_node_id][0]) - 1)

    def fulfill(self, sequence: Sequence, inventory: Inventory, verbose: bool = False):
        """Fulfill a demand sequence using the multi-price balance algorithm.

        Returns:
            FulfillmentResult (supports tuple unpacking).
        """
        collected_rewards = 0
        lost_sales = 0
        total_fulfillments = 0
        number_fulfillments = defaultdict(int)

        current_inventories = inventory.initial_inventory.copy()

        for request in sequence.requests:
            demand_node = self.graph.demand_nodes[request.demand_node]
            supply_node_chosen = self.choose_supply_node(demand_node, current_inventories, inventory)
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

        return FulfillmentResult(number_fulfillments, collected_rewards, lost_sales)

    def choose_supply_node(self, demand_node: DemandNode, current_inventories, inventory: Inventory):
        """Select the supply node maximizing reward minus phi opportunity cost.

        Returns:
            Supply node ID, or -1 if no supply node has inventory.
        """
        used_fractions = {
            supply_node_id: 1 - extended_division(current_inventories[supply_node_id], inventory.initial_inventory[supply_node_id])
            for supply_node_id in demand_node.neighbors
        }
        pseudo_rewards = [
            [supply_node_id,
             self.graph.edges[supply_node_id, demand_node.id].reward - self.phi(supply_node_id, used_fractions[supply_node_id])]
            for supply_node_id in used_fractions
        ]

        pseudo_rewards.sort(key=lambda x: x[1], reverse=True)

        for supply_node_id, _ in pseudo_rewards:
            if current_inventories[supply_node_id] > 0:
                return supply_node_id

        return -1


# Backward compatibility aliases
BalanceFulfillment = BalancePolicy
MultiPriceBalanceFulfillment = MultiPriceBalancePolicy
