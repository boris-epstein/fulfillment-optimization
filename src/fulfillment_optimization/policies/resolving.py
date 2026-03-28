"""LP re-solving fulfillment policies.

Three variants that share the same fulfill/choose loop but differ in how
they compute dual variables at each re-solving epoch:
  - OfflineLpReSolving: solves offline LP on training samples
  - FluidLpReSolving: solves fluid LP on cumulative average demand
  - ExtrapolationLpReSolving: solves fluid LP on extrapolated observed demand
"""

from abc import abstractmethod
from collections import defaultdict
from typing import Dict, List

from ..graph import Graph, DemandNode
from ..demand import Sequence
from ..inventory import Inventory
from ..lp import MathPrograms
from .base import FulfillmentResult


class LpReSolvingPolicy:
    """Base class for LP re-solving fulfillment policies.

    Provides the shared fulfill loop and supply-node selection logic.
    Subclasses implement ``_compute_duals`` to define their specific
    dual-variable computation strategy.
    """

    def __init__(self, graph: Graph, solver: str = 'gurobi'):
        self.graph = graph
        self.programs = MathPrograms(graph, solver=solver)

    def _run_fulfill_loop(self, sequence, inventory, initial_dual_solution,
                          re_solving_epochs, verbose=False):
        """Shared fulfillment loop with epoch-based re-solving.

        Returns:
            FulfillmentResult (supports tuple unpacking).
        """
        collected_rewards = 0
        lost_sales = 0

        number_fulfillments = defaultdict(int)

        current_epoch_index = 0
        next_epoch = re_solving_epochs[current_epoch_index]

        dual_variables = initial_dual_solution

        current_inventories = inventory.initial_inventory.copy()

        T = len(sequence)
        observed_demand = defaultdict(int)

        for t, request in enumerate(sequence.requests):
            demand_node = self.graph.demand_nodes[request.demand_node]

            # Extrapolation needs observed_demand updated before re-solving
            if self._update_demand_before_resolve:
                observed_demand[demand_node.id] += 1

            if t == next_epoch:
                if verbose:
                    print(f'Re-solving at {t}')
                dual_variables = self._compute_duals(
                    t=t,
                    current_inventories=current_inventories,
                    demand_node=demand_node,
                    observed_demand=observed_demand,
                    T=T,
                )
                if current_epoch_index == len(re_solving_epochs) - 1:
                    next_epoch = len(sequence) + 1
                else:
                    current_epoch_index += 1
                    next_epoch = re_solving_epochs[current_epoch_index]

            # Non-extrapolation policies update demand after re-solving
            if not self._update_demand_before_resolve:
                observed_demand[demand_node.id] += 1

            supply_node_chosen = self._choose_supply_node(demand_node, current_inventories, dual_variables)
            if supply_node_chosen == -1:
                if verbose:
                    print(f'Demand from {demand_node.id} lost')
                lost_sales += 1
            else:
                if verbose:
                    print(f'Demand from {demand_node.id} fulfilled from warehouse {supply_node_chosen}')
                current_inventories[supply_node_chosen] -= 1
                number_fulfillments[supply_node_chosen, demand_node.id] += 1
                collected_rewards += self.graph.edges[supply_node_chosen, demand_node.id].reward


        return FulfillmentResult(number_fulfillments, collected_rewards, lost_sales)

    def _choose_supply_node(self, demand_node: DemandNode, current_inventories, dual_variables):
        """Choose the supply node with the highest reward minus dual variable.

        Returns:
            Supply node ID, or -1 if no non-negative pseudo-reward exists.
        """
        supply_node_chosen = -1
        best_reward = 0

        for supply_node_id in demand_node.neighbors:
            if current_inventories[supply_node_id] > 0:
                pseudo_reward = self.graph.edges[supply_node_id, demand_node.id].reward - dual_variables[supply_node_id]
                if pseudo_reward >= best_reward:
                    best_reward = pseudo_reward
                    supply_node_chosen = supply_node_id

        return supply_node_chosen

    @abstractmethod
    def _compute_duals(self, t, current_inventories, demand_node, observed_demand, T):
        """Compute dual variables. Subclasses implement their specific strategy."""
        ...

    # Whether to update observed_demand before or after the re-solve check.
    # Extrapolation needs the count before; others don't.
    _update_demand_before_resolve = False


class OffLpReSolvingPolicy(LpReSolvingPolicy):
    """Fulfillment policy that periodically re-solves an offline LP.

    At specified epochs, solves the offline LP over training samples to obtain
    dual variables (shadow prices), then uses reward - dual as the score for
    choosing supply nodes.
    """

    def fulfill(self, sequence: Sequence, inventory: Inventory,
                initial_dual_solution: Dict[int, float],
                train_sample: List[Sequence],
                re_solving_epochs: List[int] = None,
                filter_samples=False, verbose=False):
        """Fulfill a sequence with periodic offline LP re-solving.

        Args:
            sequence: Demand sequence.
            inventory: Initial inventory.
            initial_dual_solution: Starting dual variable values.
            train_sample: Training sequences for the offline LP.
            re_solving_epochs: Time steps at which to re-solve.
            filter_samples: If True, filter training samples by current demand node.
            verbose: Print decisions if True.

        Returns:
            FulfillmentResult (supports tuple unpacking).
        """
        self._train_sample = train_sample
        self._filter_samples = filter_samples
        return self._run_fulfill_loop(sequence, inventory, initial_dual_solution,
                                      re_solving_epochs, verbose)

    def _compute_duals(self, t, current_inventories, demand_node, observed_demand, T):
        if self._filter_samples:
            filtered_sample = self._filter_sample(self._train_sample, t, demand_node)
            offline_lp, inventory_constraints = self.programs.offline_linear_program_fixed_inventory_partial_demand(
                filtered_sample, current_inventories, t + 1, demand_node.id
            )
            n_samples = len(filtered_sample)
        else:
            offline_lp, inventory_constraints = self.programs.offline_linear_program_fixed_inventory_partial_demand(
                self._train_sample, current_inventories, t + 1, None
            )
            n_samples = len(self._train_sample)

        offline_lp.optimize()

        return {
            supply_node_id: sum(
                inventory_constraints[supply_node_id, sample_index].Pi
                for sample_index in range(n_samples)
            ) / n_samples
            for supply_node_id in self.graph.supply_nodes
        }

    def _filter_sample(self, train_sample: List[Sequence], t, current_demand_node: DemandNode,
                       size_threshold=1):
        """Filter training samples to those matching the current demand node at time t."""
        filtered_sample = []
        for sequence in train_sample:
            if sequence.requests[t].demand_node == current_demand_node.id:
                filtered_sample.append(sequence)

        if len(filtered_sample) >= size_threshold:
            return filtered_sample
        else:
            return train_sample

    # Keep old method name for backward compat
    def compute_dual_variables(self, train_sample, t, current_inventories,
                               current_demand_node, filter_samples):
        self._train_sample = train_sample
        self._filter_samples = filter_samples
        return self._compute_duals(t, current_inventories, current_demand_node, {}, 0)

    def filter_sample(self, train_sample, t, current_demand_node, size_threshold=1):
        return self._filter_sample(train_sample, t, current_demand_node, size_threshold)


class FluLpReSolvingPolicy(LpReSolvingPolicy):
    """Fulfillment policy that periodically re-solves a fluid LP.

    Uses cumulative average demand estimates from training data to formulate
    the fluid LP, then extracts dual variables as opportunity costs.
    """

    def fulfill(self, sequence: Sequence, inventory: Inventory,
                initial_dual_solution: Dict[int, float],
                cumulative_average_demand: Dict[int, Dict[int, float]],
                re_solving_epochs: List[int] = None,
                filter_samples=False, verbose=False):
        """Fulfill a sequence with periodic fluid LP re-solving.

        Args:
            sequence: Demand sequence.
            inventory: Initial inventory.
            initial_dual_solution: Starting dual variable values.
            cumulative_average_demand: Maps time step t to average remaining
                demand per demand node from t onward.
            re_solving_epochs: Time steps at which to re-solve.
            verbose: Print decisions if True.

        Returns:
            FulfillmentResult (supports tuple unpacking).
        """
        self._cumulative_average_demand = cumulative_average_demand
        return self._run_fulfill_loop(sequence, inventory, initial_dual_solution,
                                      re_solving_epochs, verbose)

    def _compute_duals(self, t, current_inventories, demand_node, observed_demand, T):
        fluid_lp, inventory_constraints = self.programs.fluid_linear_program_fixed_inventory(
            self._cumulative_average_demand[t + 1], Inventory(current_inventories, 'aux')
        )
        fluid_lp.optimize()
        return {supply_node_id: inventory_constraints[supply_node_id].Pi
                for supply_node_id in self.graph.supply_nodes}

    # Keep old method name for backward compat
    def compute_dual_variables(self, t, current_inventories, cumulative_demand):
        self._cumulative_average_demand = cumulative_demand
        return self._compute_duals(t, current_inventories, None, {}, 0)

    def filter_sample(self, train_sample, t, current_demand_node, size_threshold=1):
        filtered_sample = []
        for sequence in train_sample:
            if sequence.requests[t].demand_node == current_demand_node.id:
                filtered_sample.append(sequence)
        if len(filtered_sample) >= size_threshold:
            return filtered_sample
        else:
            return train_sample


class ExtrapolationLpReSolvingPolicy(LpReSolvingPolicy):
    """Fulfillment policy that re-solves using extrapolated demand estimates.

    At each re-solving epoch, estimates future demand by scaling the observed
    demand rate so far to the remaining time horizon, then solves the fluid LP.
    Does not require training data at fulfillment time.
    """

    _update_demand_before_resolve = True

    def fulfill(self, sequence: Sequence, inventory: Inventory,
                initial_dual_solution: Dict[int, float],
                re_solving_epochs: List[int] = None,
                filter_samples=False, verbose=False):
        """Fulfill a sequence with periodic extrapolation-based LP re-solving.

        Args:
            sequence: Demand sequence.
            inventory: Initial inventory.
            initial_dual_solution: Starting dual variable values (typically zeros).
            re_solving_epochs: Time steps at which to re-solve.
            verbose: Print decisions if True.

        Returns:
            FulfillmentResult (supports tuple unpacking).
        """
        return self._run_fulfill_loop(sequence, inventory, initial_dual_solution,
                                      re_solving_epochs, verbose)

    def _compute_duals(self, t, current_inventories, demand_node, observed_demand, T):
        interpolated_demand = {}
        for demand_node_id in self.graph.demand_nodes:
            if observed_demand[demand_node_id] == 0:
                interpolated_demand[demand_node_id] = 0
            else:
                interpolated_demand[demand_node_id] = (T - t - 1) * observed_demand[demand_node_id] / (t + 1)

        fluid_lp, inventory_constraints = self.programs.fluid_linear_program_fixed_inventory(
            interpolated_demand, Inventory(current_inventories, 'aux')
        )
        fluid_lp.optimize()
        return {supply_node_id: inventory_constraints[supply_node_id].Pi
                for supply_node_id in self.graph.supply_nodes}

    # Keep old method name for backward compat
    def compute_dual_variables(self, t, current_inventories, observed_demand, T):
        return self._compute_duals(t, current_inventories, None, observed_demand, T)


# Backward compatibility aliases
OffLpReSolvingFulfillment = OffLpReSolvingPolicy
FluLpReSolvingFulfillment = FluLpReSolvingPolicy
ExtrapolationLpReSolvingFulfillment = ExtrapolationLpReSolvingPolicy
