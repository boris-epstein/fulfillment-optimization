"""Inventory data structures and placement optimization."""

from .graph import Graph
from .lp import MathPrograms
from typing import Dict, List
import numpy as np


class Inventory:
    """Represents the initial inventory allocation across supply nodes.

    Attributes:
        name: Descriptive label for this inventory configuration.
        initial_inventory: Dict mapping supply node ID to initial units.
        total_inventory: Sum of all initial inventory units.
    """

    def __init__(self, initial_inventory: Dict[int, int], name: str = 'unnamed') -> None:
        self.name = name
        self.initial_inventory = initial_inventory.copy()
        self.total_inventory = sum(initial_inventory.values())


class FormatConverter:
    """Utility for converting solver variable solutions to Python types."""

    def gurobi_to_int(self, gurobi_variables: dict):
        """Extract solution values from solver variables into a plain dict."""
        output = {}
        for key in gurobi_variables:
            output[key] = float(gurobi_variables[key].X)
        return output


class InventoryOptimizer:
    """Optimizes inventory placement across supply nodes using LP-based methods.

    Provides multiple strategies: LP rounding, greedy shadow-price placement,
    and myopic greedy placement.
    """

    def __init__(self, graph: Graph, solver: str = 'gurobi'):
        self.graph = graph
        self.programs = MathPrograms(self.graph, solver=solver)
        self.converter = FormatConverter()

    def set_inventory_to_n(self, n: int) -> Dict[int, int]:
        """Create an inventory with n units at every supply node.

        Returns:
            Inventory with constant allocation.
        """
        initial_inventories = {}
        for i in self.graph.supply_nodes:
            initial_inventories[i] = n
        return Inventory(initial_inventories, f'constant_{n}')

    def fluid_inventory_placement_rounding(self, average_demand: Dict[int, float],
                                           total_inventory: int,
                                           rescale_inventory: bool = False) -> Dict[int, int]:
        """Solve the fluid LP with variable inventory and round the fractional solution.

        Args:
            average_demand: Expected demand per demand node.
            total_inventory: Total inventory budget.
            rescale_inventory: If True, scale demand so total matches inventory.

        Returns:
            Inventory with integer allocations from rounding.
        """
        scaling_factor = 1.0

        if rescale_inventory:
            scaling_factor = total_inventory / sum(average_demand[demand_node_index] for demand_node_index in self.graph.demand_nodes)

        fluid_lp, fluid_inventory = self.programs.fluid_linear_program_variable_inventory(
            average_demand=average_demand, total_inventory=total_inventory, scaling_factor=scaling_factor
        )
        fluid_lp.optimize()

        fluid_inventory_dict = self.converter.gurobi_to_int(gurobi_variables=fluid_inventory)
        rounded_fluid_inventory = self.greedy_round_inventory(fluid_inventory_dict)

        inventory_name = 'fluid_lp_rounding'
        if rescale_inventory:
            inventory_name += '_withscaling'
        return Inventory(rounded_fluid_inventory, inventory_name)

    def offline_inventory_placement_rounding(self, demand_samples: List,
                                             total_inventory: int) -> Dict[int, int]:
        """Solve the offline LP with variable inventory and round the solution.

        Args:
            demand_samples: Sample demand sequences for the LP.
            total_inventory: Total inventory budget.

        Returns:
            Inventory with integer allocations from rounding.
        """
        offline_lp, offline_inventory = self.programs.offline_linear_program_variable_inventory(
            demand_samples=demand_samples, total_inventory=total_inventory
        )
        offline_lp.optimize()

        offline_inventory_dict = self.converter.gurobi_to_int(gurobi_variables=offline_inventory)
        rounded_offline_inventory = self.greedy_round_inventory(offline_inventory_dict)

        return Inventory(rounded_offline_inventory, 'offline_lp_rounding')

    def fluid_greedy_inventory_placement(self, average_demand: Dict[int, float],
                                         total_inventory: int, verbose: bool = False):
        """Greedily place inventory one unit at a time, maximizing fluid LP value.

        At each step, tries adding one unit to each supply node and keeps the
        placement that yields the highest LP objective.

        Args:
            average_demand: Expected demand per demand node.
            total_inventory: Number of units to place.
            verbose: Print progress if True.

        Returns:
            Inventory with the greedy placement.
        """
        inventory_placement = {}
        for supply_node_id in self.graph.supply_nodes:
            inventory_placement[supply_node_id] = 0

        units_placed = 0
        while units_placed < total_inventory:
            if verbose:
                print(f'Placing unit {units_placed + 1}/{total_inventory}')
            best_value = 0
            best_next_sol = {}
            for supply_node_id in self.graph.supply_nodes:
                candidate_sol = inventory_placement.copy()
                candidate_sol[supply_node_id] += 1

                fluid_lp, inventory_constraints = self.programs.fluid_linear_program_fixed_inventory(
                    average_demand, Inventory(candidate_sol, name='candidate')
                )
                fluid_lp.optimize()

                if fluid_lp.ObjVal >= best_value:
                    best_value = fluid_lp.ObjVal
                    best_next_sol = candidate_sol

            inventory_placement = best_next_sol.copy()
            units_placed += 1

        return Inventory(inventory_placement, name='fluid_greedy')

    def offline_greedy_inventory_placement(self, demand_samples: List,
                                           total_inventory: int, verbose: bool = False):
        """Greedily place inventory using offline LP shadow prices.

        At each step, solves the offline LP and adds one unit to the supply node
        with the highest average shadow price on its inventory constraint.

        Args:
            demand_samples: Sample demand sequences.
            total_inventory: Number of units to place.
            verbose: Print progress if True.

        Returns:
            Inventory with the greedy placement.
        """
        inventory_placement = {}
        for supply_node_id in self.graph.supply_nodes:
            inventory_placement[supply_node_id] = 0

        units_placed = 0
        while units_placed < total_inventory:
            if verbose:
                print(f'Placing unit {units_placed + 1}/{total_inventory}')

            offline_lp, inventory_constraints = self.programs.offline_linear_program_fixed_inventory(
                demand_samples, Inventory(inventory_placement, name='candidate')
            )
            offline_lp.optimize()

            highest_shadow_price = 0
            best_supply_node = -1

            for supply_node_id in self.graph.supply_nodes:
                shadow_price = sum(
                    inventory_constraints[supply_node_id, sample_index].Pi
                    for sample_index in range(len(demand_samples))
                ) / len(demand_samples)
                if shadow_price >= highest_shadow_price:
                    highest_shadow_price = shadow_price
                    best_supply_node = supply_node_id

            inventory_placement[best_supply_node] += 1
            units_placed += 1

        return Inventory(inventory_placement, name='offline_greedy')

    def myopic_greedy_inventory_placement(self, demand_samples: List,
                                          total_inventory: int, verbose: bool = False):
        """Greedily place inventory by maximizing myopic fulfillment reward.

        At each step, simulates myopic fulfillment with each candidate
        placement and picks the one with highest average reward.

        Args:
            demand_samples: Sample demand sequences.
            total_inventory: Number of units to place.
            verbose: Print progress if True.

        Returns:
            Inventory with the greedy placement.
        """
        from .policies.priority_list import PriorityListPolicy
        fulfillment = PriorityListPolicy(self.graph)

        inventory_placement = {}
        for supply_node_id in self.graph.supply_nodes:
            inventory_placement[supply_node_id] = 0

        units_placed = 0
        while units_placed < total_inventory:
            if verbose:
                print(f'Placing unit {units_placed + 1}/{total_inventory}')
            best_value = 0
            best_next_sol = {}
            for supply_node_id in self.graph.supply_nodes:
                candidate_sol = inventory_placement.copy()
                candidate_sol[supply_node_id] += 1

                candidate_reward = 0
                for sequence in demand_samples:
                    _, collected_rewards, _ = fulfillment.fixed_list_fulfillment(
                        sequence, Inventory(candidate_sol, name='candidate'), 'myopic'
                    )
                    candidate_reward += collected_rewards / len(demand_samples)

                if candidate_reward >= best_value:
                    best_value = candidate_reward
                    best_next_sol = candidate_sol

            inventory_placement = best_next_sol.copy()
            units_placed += 1

        return Inventory(inventory_placement, name='myopic_greedy')

    def greedy_round_inventory(self, fractional_placement: dict) -> dict:
        """Round fractional inventory to integers, preserving the total.

        Floors all values, then distributes remaining units to positions
        with the largest fractional remainders.

        Args:
            fractional_placement: Dict mapping supply node ID to fractional inventory.

        Returns:
            Dict with integer inventory values summing to the original total.
        """
        rounded_inventory = fractional_placement.copy()

        amount_to_round = 0
        remainders = []

        for supply_node_id in self.graph.supply_nodes:
            remainders.append((supply_node_id, rounded_inventory[supply_node_id] - np.floor(rounded_inventory[supply_node_id])))
            amount_to_round += rounded_inventory[supply_node_id] - np.floor(rounded_inventory[supply_node_id])
            rounded_inventory[supply_node_id] = int(np.floor(rounded_inventory[supply_node_id]))

        amount_to_round = np.round(amount_to_round)
        remainders.sort(reverse=True, key=lambda x: x[1])

        if amount_to_round > 0:
            for elem in remainders:
                rounded_inventory[elem[0]] += 1
                amount_to_round -= 1
                if amount_to_round <= 0:
                    break

        return rounded_inventory
