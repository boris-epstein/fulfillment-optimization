"""Learned fulfillment policies optimized via derivative-free optimization.

Requires optional dependencies: nevergrad and torch.
Install with: pip install fulfillment-optimization[ml]
"""

from collections import defaultdict
from typing import List

import nevergrad as ng
import numpy as np
import torch
import torch.nn as nn

from ..graph import Graph
from ..demand import Sequence
from ..inventory import Inventory
from .balance import MultiPriceBalancePolicy


class ThresholdsPolicy:
    """Fulfillment policy based on per-edge usage thresholds.

    Each edge (i, j) has a threshold limiting how many times it can be used.
    Among eligible edges, the one with the highest reward is chosen.
    Parameters are optimized via derivative-free optimization.
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self.edge_map = {}
        e = 0
        for supply_node_id, demand_node_id in graph.edges:
            self.edge_map[supply_node_id, demand_node_id] = e
            e += 1

    def fulfill(self, sequence: Sequence, inventory: Inventory, Theta):
        """Fulfill a demand sequence using threshold-based edge limits.

        Args:
            sequence: Demand sequence to fulfill.
            inventory: Initial inventory levels.
            Theta: Vector of thresholds, one per edge.

        Returns:
            Total reward collected.
        """
        uses = defaultdict(int)
        current_inventory = inventory.initial_inventory.copy()
        total_reward = 0
        for t, request in enumerate(sequence.requests):
            j = request.demand_node
            demand_neighbors = self.graph.demand_nodes[j].neighbors
            chosen_supply = -1
            best_reward = -float('inf')
            for i in demand_neighbors:
                edge_id = self.edge_map[i, j]
                if current_inventory[i] > 0 and uses[(i, j)] < Theta[edge_id]:
                    r_ij = self.graph.edges[(i, j)].reward
                    if r_ij > best_reward:
                        best_reward = r_ij
                        chosen_supply = i
            if chosen_supply != -1:
                current_inventory[chosen_supply] -= 1
                uses[(chosen_supply, j)] += 1
                total_reward += self.graph.edges[(chosen_supply, j)].reward
        return total_reward

    def _evaluate_theta_vector(self, theta_vector: List[float], inventory: Inventory, sequences: List[Sequence]) -> float:
        """Evaluate average reward of a threshold vector over multiple sequences."""
        total_reward = 0.0
        for sequence in sequences:
            reward = self.fulfill(sequence, inventory, theta_vector)
            total_reward += reward
        return total_reward / len(sequences) if sequences else 0.0

    def train(self, inventory: Inventory, train_samples: List[Sequence], optimizer_name: str = 'DE', budget: int = 1000):
        """Optimize threshold parameters using derivative-free optimization.

        Returns:
            Best threshold parameter vector found.
        """
        edge_list = list(self.graph.edges.keys())
        param_dim = len(edge_list)

        init_theta = [inventory.initial_inventory[i] for i in self.graph.supply_nodes for _ in range(len(self.graph.demand_nodes))]
        max_inventory = max(init_theta)

        param = ng.p.Array(init=init_theta, lower=0.0, upper=max_inventory)
        optimizer = ng.optimizers.registry[optimizer_name](parametrization=param, budget=budget)

        best_candidate = optimizer.minimize(lambda x: -self._evaluate_theta_vector(x, inventory, train_samples))
        return best_candidate


class TimeSupplyEnhancedMPB:
    """Multi-price balance with learnable supply-node offsets and time decay.

    Uses composition: delegates phi computation to a MultiPriceBalancePolicy instance.

    Parameters: len(supply_nodes) + 1 (thetas + gamma).
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self._balance = MultiPriceBalancePolicy(graph)
        self.num_parameters = len(graph.supply_nodes) + 1

    def fulfill(self, sequence: Sequence, inventory: Inventory, theta_and_gamma: List[float]) -> float:
        """Fulfill using time- and supply-enhanced balance scoring.

        Args:
            sequence: Demand sequence.
            inventory: Initial inventory.
            theta_and_gamma: Parameter vector [theta_0, ..., theta_m-1, gamma].

        Returns:
            Total reward collected.
        """
        m = len(self.graph.supply_nodes)
        theta_list = theta_and_gamma[:m]
        gamma = theta_and_gamma[m]

        current_inventory = inventory.initial_inventory.copy()
        total_reward = 0
        T = len(sequence)
        for t, request in enumerate(sequence.requests):
            j = request.demand_node
            best_score = -float('inf')
            chosen_supply = -1

            for i in self.graph.demand_nodes[j].neighbors:
                if current_inventory[i] <= 0:
                    continue

                used_fraction = 1.0 - current_inventory[i] / inventory.initial_inventory[i]
                base_cost = self._balance.phi(i, used_fraction)
                cost = (base_cost + theta_list[i]) * (1 - np.exp(gamma * (t / T) - 1))
                score = self.graph.edges[(i, j)].reward - cost

                if score > best_score:
                    best_score = score
                    chosen_supply = i

            if chosen_supply != -1 and best_score >= 0:
                current_inventory[chosen_supply] -= 1
                total_reward += self.graph.edges[(chosen_supply, j)].reward

        return total_reward

    def _evaluate(self, theta_and_gamma: List[float], inventory: Inventory, sequences: List[Sequence]) -> float:
        """Evaluate average reward over multiple sequences."""
        total_reward = 0.0
        for sequence in sequences:
            reward = self.fulfill(sequence, inventory, theta_and_gamma)
            total_reward += reward
        return total_reward / len(sequences) if sequences else 0.0

    def train(self, inventory: Inventory, sequences: List[Sequence], optimizer_name="DE", budget=500):
        """Optimize theta and gamma parameters via derivative-free optimization.

        Returns:
            Best parameter vector [theta_0, ..., theta_m-1, gamma].
        """
        m = len(self.graph.supply_nodes)
        init = [0.0] * m + [1.0]

        param = ng.p.Array(init=init)
        param.set_bounds(lower=[-3.0] * m + [-1.0], upper=[3.0] * m + [8.0])

        optimizer_cls = ng.optimizers.registry[optimizer_name]
        optimizer = optimizer_cls(parametrization=param, budget=budget)

        best = optimizer.minimize(lambda x: -self._evaluate(x, inventory, sequences))
        best_params = best.value
        self.best_params = best_params
        self.best_thetas = best_params[:m]
        self.best_gamma = best_params[m]

        return best_params


class SupplyEnhancedMPB:
    """Multi-price balance with learnable per-supply-node bias terms.

    Uses composition: delegates phi computation to a MultiPriceBalancePolicy instance.

    Parameters: len(supply_nodes).
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self._balance = MultiPriceBalancePolicy(graph)
        self.num_parameters = len(graph.supply_nodes)

    def fulfill(self, sequence: Sequence, inventory: Inventory, theta_list: List[float]) -> float:
        """Fulfill using supply-enhanced balance scoring.

        Args:
            sequence: Demand sequence.
            inventory: Initial inventory.
            theta_list: Per-supply-node bias parameters.

        Returns:
            Total reward collected.
        """
        m = len(self.graph.supply_nodes)

        current_inventory = inventory.initial_inventory.copy()
        total_reward = 0
        T = len(sequence)
        for t, request in enumerate(sequence.requests):
            j = request.demand_node
            best_score = -float('inf')
            chosen_supply = -1

            for i in self.graph.demand_nodes[j].neighbors:
                if current_inventory[i] <= 0:
                    continue

                used_fraction = 1.0 - current_inventory[i] / inventory.initial_inventory[i]
                base_cost = self._balance.phi(i, used_fraction)
                cost = (base_cost + theta_list[i])
                score = self.graph.edges[(i, j)].reward - cost

                if score > best_score:
                    best_score = score
                    chosen_supply = i

            if chosen_supply != -1 and best_score >= 0:
                current_inventory[chosen_supply] -= 1
                total_reward += self.graph.edges[(chosen_supply, j)].reward

        return total_reward

    def _evaluate(self, theta_list: List[float], inventory: Inventory, sequences: List[Sequence]) -> float:
        """Evaluate average reward over multiple sequences."""
        total_reward = 0.0
        for sequence in sequences:
            reward = self.fulfill(sequence, inventory, theta_list)
            total_reward += reward
        return total_reward / len(sequences) if sequences else 0.0

    def train(self, inventory: Inventory, sequences: List[Sequence], optimizer_name="DE", budget=500):
        """Optimize theta parameters via derivative-free optimization.

        Returns:
            Best theta parameter vector.
        """
        m = len(self.graph.supply_nodes)
        init = [0.0] * m

        param = ng.p.Array(init=init)
        param.set_bounds(lower=[-3.0] * m, upper=[3.0] * m)

        optimizer_cls = ng.optimizers.registry[optimizer_name]
        optimizer = optimizer_cls(parametrization=param, budget=budget)

        best = optimizer.minimize(lambda x: -self._evaluate(x, inventory, sequences))
        best_params = best.value
        self.best_thetas = best_params

        return best_params


class TimeEnhancedMPB:
    """Multi-price balance with a learnable time-decay parameter.

    Uses composition: delegates phi computation to a MultiPriceBalancePolicy instance.

    Scales the opportunity cost by (1 - exp(gamma * t/T - 1)), so the cost
    evolves over the time horizon. Only one parameter (gamma) is trained.
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self._balance = MultiPriceBalancePolicy(graph)
        self.num_parameters = 1

    def fulfill(self, sequence: Sequence, inventory: Inventory, gamma: float) -> float:
        """Fulfill using time-enhanced balance scoring.

        Args:
            sequence: Demand sequence.
            inventory: Initial inventory.
            gamma: Time-decay parameter.

        Returns:
            Total reward collected.
        """
        current_inventory = inventory.initial_inventory.copy()
        total_reward = 0
        T = len(sequence)
        for t, request in enumerate(sequence.requests):
            j = request.demand_node
            best_score = -float('inf')
            chosen_supply = -1

            for i in self.graph.demand_nodes[j].neighbors:
                if current_inventory[i] <= 0:
                    continue

                used_fraction = 1.0 - current_inventory[i] / inventory.initial_inventory[i]
                base_cost = self._balance.phi(i, used_fraction)
                cost = base_cost * (1 - np.exp(gamma * (t / T) - 1))
                score = self.graph.edges[(i, j)].reward - cost

                if score > best_score:
                    best_score = score
                    chosen_supply = i

            if chosen_supply != -1 and best_score >= 0:
                current_inventory[chosen_supply] -= 1
                total_reward += self.graph.edges[(chosen_supply, j)].reward

        return total_reward

    def _evaluate(self, gamma: float, inventory: Inventory, sequences: List[Sequence]) -> float:
        """Evaluate average reward over multiple sequences."""
        total_reward = 0.0
        for sequence in sequences:
            reward = self.fulfill(sequence, inventory, gamma)
            total_reward += reward
        return total_reward / len(sequences) if sequences else 0.0

    def train(self, inventory: Inventory, sequences: List[Sequence], optimizer_name="DE", budget=500):
        """Optimize gamma via derivative-free optimization.

        Returns:
            Best gamma value.
        """
        init = [0.0]

        param = ng.p.Array(init=init)
        param.set_bounds(lower=[-1.0], upper=[10.0])

        optimizer_cls = ng.optimizers.registry[optimizer_name]
        optimizer = optimizer_cls(parametrization=param, budget=budget)

        best = optimizer.minimize(lambda x: -self._evaluate(x, inventory, sequences))
        best_params = best.value

        self.best_gamma = best_params[0]

        return self.best_gamma


class NeuralOpportunityCostPolicy:
    """Neural network policy that learns opportunity costs from features.

    Uses a small neural network (3 inputs -> 8 hidden -> 1 output) to predict
    the opportunity cost of using a supply node. Inputs: used fraction, time
    fraction, and opportunity-weighted future demand value.

    Weights are optimized via Nevergrad derivative-free optimization.
    """

    def __init__(self, graph: Graph, seed: int = 42):
        self.graph = graph
        self.supply_ids = sorted(graph.supply_nodes.keys())
        self.demand_ids = sorted(graph.demand_nodes.keys())
        self.num_supply = len(self.supply_ids)
        self.num_demand = len(self.demand_ids)
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        self.model = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        self.num_parameters = sum(p.numel() for p in self.model.parameters())

    def fulfill(self, sequence: Sequence, inventory: Inventory, weight_vector: np.ndarray) -> float:
        """Fulfill a sequence using the neural opportunity cost policy.

        Args:
            sequence: Demand sequence.
            inventory: Initial inventory.
            weight_vector: Flattened neural network weight vector.

        Returns:
            Total reward collected.
        """
        self._set_weights_from_vector(weight_vector)
        current_inventory = inventory.initial_inventory.copy()
        total_reward = 0.0
        demand_counts = {j: 0 for j in self.demand_ids}
        T = len(sequence)

        with torch.no_grad():
            for t, request in enumerate(sequence.requests):
                j = request.demand_node
                demand_counts[j] += 1
                time_fraction = t / T if T > 0 else 0.0

                djt = {jp: demand_counts[jp] / (t + 1) for jp in self.demand_ids}

                best_score = 0.0
                chosen_i = None

                for i in self.graph.demand_nodes[j].neighbors:
                    if current_inventory[i] <= 0:
                        continue

                    used_frac = (inventory.initial_inventory[i] - current_inventory[i]) / inventory.initial_inventory[i]

                    r_ij = self.graph.edges[(i, j)].reward
                    future_value = sum(
                        self.graph.edges[(i, jp)].reward * djt[jp]
                        for jp in self.demand_ids
                        if (i, jp) in self.graph.edges
                    )

                    x = [used_frac, time_fraction, future_value]
                    x_tensor = torch.tensor(x, dtype=torch.float32)
                    opportunity_cost = self.model(x_tensor).item()

                    net_score = r_ij - opportunity_cost
                    if net_score > best_score:
                        best_score = net_score
                        chosen_i = i

                if chosen_i is not None and best_score > 0:
                    current_inventory[chosen_i] -= 1
                    total_reward += self.graph.edges[(chosen_i, j)].reward

        return total_reward

    def _set_weights_from_vector(self, weight_vector: np.ndarray):
        """Load neural network parameters from a flat weight vector."""
        vector = np.array(weight_vector, dtype=np.float32)
        idx = 0
        for param in self.model.parameters():
            shape = param.data.shape
            size = param.data.numel()
            param.data.copy_(torch.from_numpy(vector[idx: idx + size].reshape(shape)))
            idx += size

    def _evaluate(self, weight_vector: np.ndarray, inventory: Inventory, sequences: List[Sequence]) -> float:
        """Evaluate average reward of a weight vector over multiple sequences."""
        self._set_weights_from_vector(weight_vector)
        total_reward = 0.0
        for seq in sequences:
            reward = self.fulfill(seq, inventory, weight_vector)
            total_reward += reward
        return total_reward / len(sequences)

    def train(self, inventory: Inventory, train_samples: List[Sequence], optimizer_name: str = "DE", budget: int = 1001):
        """Optimize neural network weights via derivative-free optimization.

        Returns:
            Best weight vector found.
        """
        init_params = np.concatenate([p.detach().cpu().numpy().ravel() for p in self.model.parameters()]).astype(np.float32)
        param = ng.p.Array(init=init_params).set_bounds(lower=-5.0, upper=5.0)
        optimizer = ng.optimizers.registry[optimizer_name](parametrization=param, budget=budget)

        best_candidate = optimizer.minimize(
            lambda w: -self._evaluate(w.value if hasattr(w, "value") else w, inventory, train_samples)
        )
        return best_candidate.value if hasattr(best_candidate, "value") else best_candidate


class NeuralOpportunityCostWithIDPolicy:
    """Neural opportunity cost policy with supply node identity features.

    Extends NeuralOpportunityCostPolicy by adding a one-hot encoding of the
    supply node ID as additional input features. Input dimension: num_supply + 3.
    """

    def __init__(self, graph: Graph, seed: int = 42):
        self.graph = graph
        self.supply_ids = sorted(graph.supply_nodes.keys())
        self.demand_ids = sorted(graph.demand_nodes.keys())
        self.num_supply = len(self.supply_ids)
        self.num_demand = len(self.demand_ids)
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        input_dim = self.num_supply + 3

        self.model = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        self.num_parameters = sum(p.numel() for p in self.model.parameters())

    def _set_weights_from_vector(self, weight_vector: np.ndarray):
        """Load neural network parameters from a flat weight vector."""
        vector = np.array(weight_vector, dtype=np.float32)
        idx = 0
        for param in self.model.parameters():
            shape = param.data.shape
            size = param.data.numel()
            param.data.copy_(torch.from_numpy(vector[idx: idx + size].reshape(shape)))
            idx += size

    def fulfill(self, sequence: Sequence, inventory: Inventory, weight_vector: np.ndarray) -> float:
        """Fulfill a sequence using the neural policy with supply node identity.

        Args:
            sequence: Demand sequence.
            inventory: Initial inventory.
            weight_vector: Flattened neural network weight vector.

        Returns:
            Total reward collected.
        """
        self._set_weights_from_vector(weight_vector)
        current_inventory = inventory.initial_inventory.copy()
        total_reward = 0.0
        demand_counts = {j: 0 for j in self.demand_ids}
        T = len(sequence)

        with torch.no_grad():
            for t, request in enumerate(sequence.requests):
                j = request.demand_node
                demand_counts[j] += 1
                time_fraction = t / T if T > 0 else 0.0

                djt = {jp: demand_counts[jp] / (t + 1) for jp in self.demand_ids}

                best_score = 0.0
                chosen_i = None

                for i in self.graph.demand_nodes[j].neighbors:
                    if current_inventory[i] <= 0:
                        continue

                    used_frac = (inventory.initial_inventory[i] - current_inventory[i]) / inventory.initial_inventory[i]
                    r_ij = self.graph.edges[(i, j)].reward

                    future_value = sum(
                        self.graph.edges[(i, jp)].reward * djt[jp]
                        for jp in self.demand_ids
                        if (i, jp) in self.graph.edges
                    )

                    one_hot = [0.0] * self.num_supply
                    idx = self.supply_ids.index(i)
                    one_hot[idx] = 1.0

                    x = one_hot + [used_frac, time_fraction, future_value]
                    x_tensor = torch.tensor(x, dtype=torch.float32)
                    opportunity_cost = self.model(x_tensor).item()

                    net_score = r_ij - opportunity_cost
                    if net_score > best_score:
                        best_score = net_score
                        chosen_i = i

                if chosen_i is not None and best_score > 0:
                    current_inventory[chosen_i] -= 1
                    total_reward += self.graph.edges[(chosen_i, j)].reward

        return total_reward

    def _evaluate(self, weight_vector: np.ndarray, inventory: Inventory, sequences: List[Sequence]) -> float:
        """Evaluate average reward of a weight vector over multiple sequences."""
        self._set_weights_from_vector(weight_vector)
        total_reward = 0.0
        for seq in sequences:
            total_reward += self.fulfill(seq, inventory, weight_vector)
        return total_reward / len(sequences)

    def train(self, inventory: Inventory, train_samples: List[Sequence], optimizer_name: str = "DE", budget: int = 1000):
        """Optimize neural network weights via derivative-free optimization.

        Returns:
            Best weight vector found.
        """
        init_params = np.concatenate([p.detach().cpu().numpy().ravel() for p in self.model.parameters()]).astype(np.float32)
        param = ng.p.Array(init=init_params).set_bounds(lower=-5.0, upper=5.0)

        optimizer = ng.optimizers.registry[optimizer_name](parametrization=param, budget=budget)

        best_candidate = optimizer.minimize(
            lambda w: -self._evaluate(w.value if hasattr(w, "value") else w, inventory, train_samples)
        )
        return best_candidate.value if hasattr(best_candidate, "value") else best_candidate


# Backward compatibility aliases
ThresholdsFulfillment = ThresholdsPolicy
