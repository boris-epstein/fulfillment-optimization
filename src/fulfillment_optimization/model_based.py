from collections import defaultdict
from itertools import combinations, product
from typing import Dict, List

from .demand import Sequence, Request
from .graph import Graph
from .fulfillment import Inventory

import numpy as np


def generate_tuples(n, t):
    """Generate all non-negative integer tuples of length n that sum to t.

    Uses a stars-and-bars combinatorial approach.

    Args:
        n: Length of each tuple.
        t: Required sum of elements.

    Yields:
        Tuples of length n whose elements sum to t.
    """
    for comb in combinations(range(t + n - 1), n - 1):
        result = [comb[0]]
        for i in range(1, n - 1):
            result.append(comb[i] - comb[i - 1] - 1)
        result.append(t + n - 2 - comb[-1])
        yield tuple(result)


def generate_bounded_tuples(initial_tuple):
    """Generate all tuples where each element is between 0 and the corresponding bound.

    Args:
        initial_tuple: Upper bounds for each position.

    Returns:
        Iterator over all bounded tuples.
    """
    return product(*(range(a + 1) for a in initial_tuple))


def generate_bounded_tuples_with_sum(initial_tuple, T, current_tuple=(), index=0, current_sum=0):
    """Generate bounded tuples whose elements sum to at least T.

    Used to enumerate feasible inventory states: each component is bounded by
    the initial inventory, and the total remaining inventory must be at least T
    (since at most one unit is consumed per time step).

    Args:
        initial_tuple: Upper bounds for each position (initial inventory levels).
        T: Minimum required sum.
        current_tuple: Partial tuple built so far (internal recursion state).
        index: Current position being filled (internal recursion state).
        current_sum: Running sum of elements so far (internal recursion state).

    Yields:
        Tuples satisfying the element bounds and minimum sum constraint.
    """
    if index == len(initial_tuple):
        if current_sum >= T:
            yield current_tuple
        return

    min_required = max(0, T - current_sum - sum(initial_tuple[index + 1:]))

    for value in range(min_required, initial_tuple[index] + 1):
        yield from generate_bounded_tuples_with_sum(
            initial_tuple, T, current_tuple + (value,), index + 1, current_sum + value
        )


class IndependentDynamicProgram:
    """Exact dynamic programming solver for temporally independent demand.

    Computes the optimal fulfillment policy when the demand node at each time
    step is drawn independently from a known, time-varying distribution.
    """

    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def compute_optimal_policy(self, inventory: Inventory, T: int, p: Dict[int, List[float]]):
        """Solve the DP to find the optimal policy.

        Args:
            inventory: Initial inventory levels.
            T: Time horizon (number of periods).
            p: Dict mapping time step t to probability vector over demand nodes.

        Returns:
            DPOutput containing value_function and optimal_action mappings.
        """
        inventory_states = self.generate_inventory_state_space(inventory, T)

        value_function = defaultdict(float)
        optimal_action = {}

        # Boundary conditions (last period)
        for inventory_state in inventory_states:
            for demand_node_id in self.graph.demand_nodes:
                value_function[inventory_state, T - 1, demand_node_id], optimal_action[inventory_state, T - 1, demand_node_id] = self.find_best_final_supply_node(demand_node_id, inventory_state)

        # Bellman equations (backward induction)
        for t in range(T - 2, -1, -1):
            for inventory_state in inventory_states:
                for demand_node_id in self.graph.demand_nodes:
                    if sum(inventory_state) >= inventory.total_inventory - t:
                        value_function[inventory_state, t, demand_node_id], optimal_action[inventory_state, t, demand_node_id] = self.find_best_continuation(demand_node_id, inventory_state, value_function, p, t)

        return DPOutput(value_function, optimal_action)

    def find_best_final_supply_node(self, demand_node_id, inventory_state):
        """Find the best supply node to fulfill from at the last time step.

        Returns:
            Tuple of (best_reward, best_supply_node_id). Returns (0, -1) if
            no fulfillment is beneficial.
        """
        best_supply_node = -1
        best_reward = 0

        demand_node = self.graph.demand_nodes[demand_node_id]

        for supply_node_id in demand_node.neighbors:
            if inventory_state[supply_node_id] > 0 and self.graph.edges[supply_node_id, demand_node_id].reward > best_reward:
                best_reward = self.graph.edges[supply_node_id, demand_node_id].reward
                best_supply_node = supply_node_id

        return best_reward, best_supply_node

    def find_best_continuation(self, demand_node_id, inventory_state, value_function, p, t):
        """Find the action maximizing immediate reward plus expected future value.

        Considers all feasible supply nodes plus the option of not fulfilling.

        Returns:
            Tuple of (best_expected_value, best_supply_node_id). Returns -1 as
            supply node if no-action is optimal.
        """
        best_reward = 0
        best_supply_node = -1

        demand_node = self.graph.demand_nodes[demand_node_id]

        for supply_node_id in demand_node.neighbors:
            if inventory_state[supply_node_id] > 0:
                next_inventory = [inv for inv in inventory_state]
                next_inventory[supply_node_id] -= 1
                next_inventory = tuple(next_inventory)

                action_reward = 0
                for next_demand_node in self.graph.demand_nodes:
                    action_reward += (self.graph.edges[supply_node_id, demand_node_id].reward + value_function[next_inventory, t + 1, next_demand_node]) * p[t + 1][next_demand_node]

                if action_reward > best_reward:
                    best_reward = action_reward
                    best_supply_node = supply_node_id

        # Consider no-action
        no_action_reward = 0
        for next_demand_node in self.graph.demand_nodes:
            no_action_reward += value_function[inventory_state, t + 1, next_demand_node] * p[t + 1][next_demand_node]
        if no_action_reward > best_reward:
            best_reward = no_action_reward
            best_supply_node = -1

        return best_reward, best_supply_node

    def generate_inventory_state_space(self, inventory: Inventory, T: int):
        """Enumerate all feasible inventory states reachable within T time steps.

        Returns:
            List of inventory state tuples.
        """
        initial_inventory_tuple = tuple([inventory.initial_inventory[supply_node_id] for supply_node_id in self.graph.supply_nodes])
        bounded_tuples = list(generate_bounded_tuples_with_sum(initial_inventory_tuple, inventory.total_inventory - T + 1))
        return bounded_tuples


class MarkovianDynamicProgram:
    """Exact dynamic programming solver for Markovian demand.

    Computes the optimal fulfillment policy when demand nodes follow a
    Markov chain with known transition probabilities.
    """

    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def compute_optimal_policy(self, inventory: Inventory, T: int, transition_matrix):
        """Solve the DP to find the optimal policy under Markovian demand.

        Args:
            inventory: Initial inventory levels.
            T: Time horizon (number of periods).
            transition_matrix: Row-stochastic matrix of demand node transitions.

        Returns:
            DPOutput containing value_function and optimal_action mappings.
        """
        inventory_states = self.generate_inventory_state_space(inventory, T)

        value_function = defaultdict(float)
        optimal_action = {}

        # Boundary conditions
        for inventory_state in inventory_states:
            for demand_node_id in self.graph.demand_nodes:
                value_function[inventory_state, T - 1, demand_node_id], optimal_action[inventory_state, T - 1, demand_node_id] = self.find_best_final_supply_node(demand_node_id, inventory_state)

        # Bellman equations
        for t in range(T - 2, -1, -1):
            for inventory_state in inventory_states:
                for demand_node_id in self.graph.demand_nodes:
                    if sum(inventory_state) >= inventory.total_inventory - t:
                        value_function[inventory_state, t, demand_node_id], optimal_action[inventory_state, t, demand_node_id] = self.find_best_continuation(demand_node_id, inventory_state, value_function, transition_matrix, t)

        return DPOutput(value_function, optimal_action)

    def find_best_final_supply_node(self, demand_node_id, inventory_state):
        """Find the best supply node to fulfill from at the last time step.

        Returns:
            Tuple of (best_reward, best_supply_node_id).
        """
        best_supply_node = -1
        best_reward = 0

        demand_node = self.graph.demand_nodes[demand_node_id]

        for supply_node_id in demand_node.neighbors:
            if inventory_state[supply_node_id] > 0 and self.graph.edges[supply_node_id, demand_node_id].reward > best_reward:
                best_reward = self.graph.edges[supply_node_id, demand_node_id].reward
                best_supply_node = supply_node_id

        return best_reward, best_supply_node

    def find_best_continuation(self, demand_node_id, inventory_state, value_function, transition_matrix, t):
        """Find the action maximizing immediate reward plus expected continuation value.

        Uses the Markov transition matrix to compute expected future value.

        Returns:
            Tuple of (best_expected_value, best_supply_node_id).
        """
        best_reward = 0
        best_supply_node = -1

        demand_node = self.graph.demand_nodes[demand_node_id]

        for supply_node_id in demand_node.neighbors:
            if inventory_state[supply_node_id] > 0:
                next_inventory = [inv for inv in inventory_state]
                next_inventory[supply_node_id] -= 1
                next_inventory = tuple(next_inventory)

                action_reward = 0
                for next_demand_node in self.graph.demand_nodes:
                    action_reward += (self.graph.edges[supply_node_id, demand_node_id].reward + value_function[next_inventory, t + 1, next_demand_node]) * transition_matrix[demand_node.id, next_demand_node]

                if action_reward > best_reward:
                    best_reward = action_reward
                    best_supply_node = supply_node_id

        # Consider no-action
        no_action_reward = 0
        for next_demand_node in self.graph.demand_nodes:
            no_action_reward += value_function[inventory_state, t + 1, next_demand_node] * transition_matrix[demand_node.id, next_demand_node]
        if no_action_reward > best_reward:
            best_reward = no_action_reward
            best_supply_node = -1

        return best_reward, best_supply_node

    def generate_inventory_state_space(self, inventory: Inventory, T: int):
        """Enumerate all feasible inventory states reachable within T time steps.

        Returns:
            List of inventory state tuples.
        """
        initial_inventory_tuple = tuple([inventory.initial_inventory[supply_node_id] for supply_node_id in self.graph.supply_nodes])
        bounded_tuples = list(generate_bounded_tuples_with_sum(initial_inventory_tuple, inventory.total_inventory - T + 1))
        return bounded_tuples


class DPOutput:
    """Container for dynamic programming results.

    Attributes:
        value_function: Maps (inventory_state, t, demand_node_id) to expected value.
        optimal_action: Maps (inventory_state, t, demand_node_id) to best supply node
            (-1 means no fulfillment).
    """

    def __init__(self, value_function, optimal_action):
        self.value_function = value_function
        self.optimal_action = optimal_action


class ModelEstimator:
    """Estimates demand model parameters from observed sequences.

    Supports IID, temporally independent, and Markovian estimation.
    """

    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def estimate_iid(self, sequences: List[Sequence]) -> List[float]:
        """Estimate a single probability vector assuming IID demand across all time steps.

        Args:
            sequences: Training sequences.

        Returns:
            List of probabilities, one per demand node.
        """
        p = defaultdict(float)

        T = sequences[0].length
        denominator = len(sequences) * T

        for sequence in sequences:
            for request in sequence.requests:
                p[request.demand_node] += 1

        p_list = [None for demand_node_id in self.graph.demand_nodes]
        for demand_node_id in self.graph.demand_nodes:
            p_list[demand_node_id] = p[demand_node_id] / denominator

        return p_list

    def estimate_independent(self, sequences: List[Sequence]) -> Dict[int, List[float]]:
        """Estimate per-period probability vectors assuming temporal independence.

        Args:
            sequences: Training sequences (all must have the same length).

        Returns:
            Dict mapping time step t to estimated probability vector.
        """
        p = {}
        T = sequences[0].length
        n_sequences = len(sequences)
        for t in range(T):
            p[t] = [0 for _ in self.graph.demand_nodes]

        for sequence in sequences:
            t = 0
            for request in sequence.requests:
                p[t][request.demand_node] += 1 / n_sequences
                t += 1

        return p

    def estimate_markovian(self, sequences: List[Sequence]):
        """Estimate a Markov chain transition matrix and initial distribution.

        Counts transitions between consecutive demand nodes and normalizes.
        If a state has no outgoing transitions in the data, assigns uniform.

        Args:
            sequences: Training sequences.

        Returns:
            Tuple of (transition_matrix, initial_distribution).
        """
        n_demand_nodes = len(self.graph.demand_nodes)
        initial_distribution = [0 for _ in self.graph.demand_nodes]
        n_sequences = len(sequences)
        transition_matrix = np.zeros((n_demand_nodes, n_demand_nodes))

        zeros_vector = np.zeros(n_demand_nodes)

        for sequence in sequences:
            T = sequence.length
            initial_distribution[sequence.requests[0].demand_node] += 1 / n_sequences
            last_demand_node = sequence.requests[0].demand_node
            for t in range(1, T):
                request = sequence.requests[t]
                demand_node = request.demand_node
                transition_matrix[last_demand_node, demand_node] += 1
                last_demand_node = demand_node

        # Normalize; use uniform for states with no observed transitions
        for demand_node_id in self.graph.demand_nodes:
            if all(transition_matrix[demand_node_id, :] == zeros_vector):
                transition_matrix[demand_node_id, :] = np.ones(n_demand_nodes) / n_demand_nodes
            else:
                transition_matrix[demand_node_id, :] = transition_matrix[demand_node_id, :] / transition_matrix[demand_node_id, :].sum()

        return transition_matrix, initial_distribution
