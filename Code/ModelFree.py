
from Graph import Graph
from Demand import Sequence
from collections import defaultdict
from FulfillmentOptimization import Inventory, MultiPriceBalanceFulfillment
from typing import List
import nevergrad as ng
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class ThresholdsFulfillment:
    
    def __init__(self, graph: Graph):
        self.graph = graph
        self.edge_map = {}
        e = 0
        for supply_node_id, demand_node_id in graph.edges:
            self.edge_map[supply_node_id,demand_node_id] = e
            e+=1
    
    
    def fulfill(self, sequence: Sequence, inventory: Inventory, Theta):
    # Theta is a dict or matrix of thresholds theta_{ij}
        uses = defaultdict(int)  # track usage count for each edge (i,j)
        current_inventory = inventory.initial_inventory.copy()
        total_reward = 0
        for t, request in enumerate(sequence.requests):
            j = request.demand_node  # arriving demand node ID
            demand_neighbors = self.graph.demand_nodes[j].neighbors  # supply nodes that can serve j
            chosen_supply = -1
            best_reward = -float('inf')
            # Examine each neighbor i for eligibility
            for i in demand_neighbors:
                edge_id = self.edge_map[i,j]
                if current_inventory[i] > 0 and uses[(i, j)] < Theta[edge_id]:
                    r_ij = self.graph.edges[(i, j)].reward
                    if r_ij > best_reward:
                        best_reward = r_ij
                        chosen_supply = i
            # Fulfill if a valid supply was found
            if chosen_supply != -1:
                current_inventory[chosen_supply] -= 1          # consume one unit of inventory
                uses[(chosen_supply, j)] += 1                  # record this edge usage
                total_reward += self.graph.edges[(chosen_supply, j)].reward
            # if chosen_supply == -1, the demand is lost (no fulfillment)
        return total_reward
    
    def _evaluate_theta_vector(self, theta_vector: List[float], inventory: Inventory, sequences: List[Sequence]) -> float:
        

        total_reward = 0.0
        for sequence in sequences:
            reward = self.fulfill(sequence, inventory, theta_vector)
            total_reward += reward

        return total_reward / len(sequences) if sequences else 0.0
    
    def train(self, inventory: Inventory, train_samples: List[Sequence], optimizer_name: str = 'DE', budget: int = 1000, seed: int = 42):
        edge_list = list(self.graph.edges.keys())
        param_dim = len(edge_list)

        init_theta = [inventory.initial_inventory[i] for i in self.graph.supply_nodes for _ in range(len(self.graph.demand_nodes))]  # crude init
        max_inventory = max(init_theta)

        param = ng.p.Array(init=init_theta, lower=0.0, upper=max_inventory)
        optimizer = ng.optimizers.registry[optimizer_name](parametrization=param, budget=budget)
        
        # Use a bound method in the lambda
        best_candidate = optimizer.minimize(lambda x: -self._evaluate_theta_vector(x, inventory, train_samples))
        return best_candidate

        
class TimeSupplyEnhancedMPB(MultiPriceBalanceFulfillment):
    
    def __init__(self, graph: Graph):
        super().__init__(graph)
        self.graph = graph
        self.num_parameters = len(graph.supply_nodes) + 1
        

    def fulfill(self, sequence: Sequence, inventory: Inventory, theta_and_gamma: List[float]) -> float:
        
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
                base_cost = self.phi(i, used_fraction)
                cost = (base_cost + theta_list[i]) * (1 - np.exp(gamma * (t/T)-1))
                score = self.graph.edges[(i, j)].reward - cost

                if score > best_score:
                    best_score = score
                    chosen_supply = i

            if chosen_supply != -1 and best_score >= 0:
                current_inventory[chosen_supply] -= 1
                total_reward += self.graph.edges[(chosen_supply, j)].reward

        return total_reward

    def _evaluate(self, theta_and_gamma: List[float], inventory: Inventory, sequences: List[Sequence]) -> float:
        # m = len(self.graph.supply_nodes)
        # theta_list = theta_and_gamma[:m]
        # gamma = theta_and_gamma[m]

        total_reward = 0.0
        for sequence in sequences:
            reward = self.fulfill(sequence, inventory, theta_and_gamma)
            total_reward += reward

        return total_reward / len(sequences) if sequences else 0.0

    def train(self, inventory: Inventory, sequences: List[Sequence], optimizer_name="DE", budget=500, seed: int = 42):
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

class SupplyEnhancedMPB(MultiPriceBalanceFulfillment):
    
    def __init__(self, graph: Graph):
        super().__init__(graph)
        self.graph = graph
        self.num_parameters = len(graph.supply_nodes)
        

    def fulfill(self, sequence: Sequence, inventory: Inventory, theta_list: List[float]) -> float:
        
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
                base_cost = self.phi(i, used_fraction)
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
        # m = len(self.graph.supply_nodes)
        # theta_list = theta_and_gamma[:m]
        # gamma = theta_and_gamma[m]

        total_reward = 0.0
        for sequence in sequences:
            reward = self.fulfill(sequence, inventory, theta_list)
            total_reward += reward

        return total_reward / len(sequences) if sequences else 0.0

    def train(self, inventory: Inventory, sequences: List[Sequence], optimizer_name="DE", budget=500, seed: int = 42):
        m = len(self.graph.supply_nodes)
        init = [0.0] * m 

        param = ng.p.Array(init=init)
        param.set_bounds(lower=[-3.0] * m , upper=[3.0] * m )



        optimizer_cls = ng.optimizers.registry[optimizer_name]
        optimizer = optimizer_cls(parametrization=param, budget=budget )

        best = optimizer.minimize(lambda x: -self._evaluate(x, inventory, sequences))
        best_params = best.value
        self.best_thetas = best_params
        

        return best_params

    
class TimeEnhancedMPB(MultiPriceBalanceFulfillment):
    
    def __init__(self, graph: Graph):
        super().__init__(graph)
        self.graph = graph
        self.num_parameters = 1
        

    def fulfill(self, sequence: Sequence, inventory: Inventory, gamma: float) -> float:
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
                base_cost = self.phi(i, used_fraction)
                cost = base_cost * (1 - np.exp(gamma * (t/T)-1))
                score = self.graph.edges[(i, j)].reward - cost

                if score > best_score:
                    best_score = score
                    chosen_supply = i

            if chosen_supply != -1 and best_score >= 0:
                current_inventory[chosen_supply] -= 1
                total_reward += self.graph.edges[(chosen_supply, j)].reward

        return total_reward

    def _evaluate(self,gamma: float, inventory: Inventory, sequences: List[Sequence]) -> float:
        

        total_reward = 0.0
        for sequence in sequences:
            reward = self.fulfill(sequence, inventory,  gamma)
            total_reward += reward

        return total_reward / len(sequences) if sequences else 0.0

    def train(self, inventory: Inventory, sequences: List[Sequence], optimizer_name="DE", budget=500, seed: int = 42):
        
        init = [0.0]

        param = ng.p.Array(init=init)
        param.set_bounds(lower = [-1.0], upper= [10.0])


        optimizer_cls = ng.optimizers.registry[optimizer_name]
        optimizer = optimizer_cls(parametrization=param, budget=budget)

        best = optimizer.minimize(lambda x: -self._evaluate(x, inventory, sequences))
        best_params = best.value
        
        self.best_gamma = best_params[0]

        return self.best_gamma

    
    

class NeuralOpportunityCostPolicy:
    def __init__(self, graph: Graph, seed: int = 42):
        self.graph = graph
        self.supply_ids = sorted(graph.supply_nodes.keys())
        self.demand_ids = sorted(graph.demand_nodes.keys())
        self.num_supply = len(self.supply_ids)
        self.num_demand = len(self.demand_ids)
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        # Only 3 inputs now
        self.model = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        self.num_parameters = sum(p.numel() for p in self.model.parameters())


    def fulfill(self, sequence: Sequence, inventory: Inventory, weight_vector: np.ndarray) -> float:
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

                # Compute empirical demand frequencies d_{j't}
                djt = {jp: demand_counts[jp] / (t + 1) for jp in self.demand_ids}

                best_score = 0.0
                chosen_i = None

                for i in self.graph.demand_nodes[j].neighbors:
                    if current_inventory[i] <= 0:
                        continue

                    # x₁: used fraction
                    used_frac = (inventory.initial_inventory[i] - current_inventory[i]) / inventory.initial_inventory[i]

                    # x₃: opportunity-weighted demand for more valuable connections
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
        vector = np.array(weight_vector, dtype=np.float32)
        idx = 0
        for param in self.model.parameters():
            shape = param.data.shape
            size = param.data.numel()
            param.data.copy_(torch.from_numpy(vector[idx: idx + size].reshape(shape)))
            idx += size

    def _evaluate(self, weight_vector: np.ndarray, inventory: Inventory, sequences: List[Sequence]) -> float:
        self._set_weights_from_vector(weight_vector)

        total_reward = 0.0
        
        for seq in sequences:
            reward = self.fulfill(seq, inventory, weight_vector)
            total_reward += reward

        return total_reward/len(sequences)


    def train(self, inventory: Inventory, train_samples: List[Sequence], optimizer_name: str = "DE", budget: int = 1001, seed: int = 42):
        init_params = np.concatenate([p.detach().cpu().numpy().ravel() for p in self.model.parameters()]).astype(np.float32)
        param = ng.p.Array(init=init_params).set_bounds(lower=-5.0, upper=5.0)
        optimizer = ng.optimizers.registry[optimizer_name](parametrization=param, budget=budget)

        best_candidate = optimizer.minimize(
            lambda w: -self._evaluate(w.value if hasattr(w, "value") else w, inventory, train_samples)
        )
        return best_candidate.value if hasattr(best_candidate, "value") else best_candidate


class NeuralOpportunityCostWithIDPolicy:
    def __init__(self, graph: Graph, seed: int = 42):
        self.graph = graph
        self.supply_ids = sorted(graph.supply_nodes.keys())
        self.demand_ids = sorted(graph.demand_nodes.keys())
        self.num_supply = len(self.supply_ids)
        self.num_demand = len(self.demand_ids)
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        input_dim = self.num_supply + 3  # one-hot + 3 features

        self.model = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        self.num_parameters = sum(p.numel() for p in self.model.parameters())
        

    def _set_weights_from_vector(self, weight_vector: np.ndarray):
        vector = np.array(weight_vector, dtype=np.float32)
        idx = 0
        for param in self.model.parameters():
            shape = param.data.shape
            size = param.data.numel()
            param.data.copy_(torch.from_numpy(vector[idx: idx + size].reshape(shape)))
            idx += size

    def fulfill(self, sequence: Sequence, inventory: Inventory, weight_vector: np.ndarray) -> float:
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

                # Compute empirical demand frequencies d_{j't}
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

                    # One-hot encode the supply node i
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
        self._set_weights_from_vector(weight_vector)

        total_reward = 0.0
        for seq in sequences:
            total_reward += self.fulfill(seq, inventory, weight_vector)
            
        return total_reward/len(sequences)

    def train(self, inventory: Inventory, train_samples: List[Sequence], optimizer_name: str = "DE", budget: int = 1000, seed: int = 42):
        init_params = np.concatenate([p.detach().cpu().numpy().ravel() for p in self.model.parameters()]).astype(np.float32)
        param = ng.p.Array(init=init_params).set_bounds(lower=-5.0, upper=5.0)

        optimizer = ng.optimizers.registry[optimizer_name](parametrization=param, budget=budget)

        best_candidate = optimizer.minimize(
            lambda w: -self._evaluate(w.value if hasattr(w, "value") else w, inventory, train_samples)
        )
        return best_candidate.value if hasattr(best_candidate, "value") else best_candidate