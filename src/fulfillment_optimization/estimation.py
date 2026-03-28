"""Demand model estimation from observed sequences."""

from collections import defaultdict
from typing import Dict, List

from .demand import Sequence
from .graph import Graph

import numpy as np


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
