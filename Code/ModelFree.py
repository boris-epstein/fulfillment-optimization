
from Graph import Graph
from Demand import Sequence
from collections import defaultdict
from FulfillmentOptimization import Inventory, MultiPriceBalanceFulfillment
from typing import List
import nevergrad as ng
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
    
    def train(self, inventory: Inventory, train_samples: List[Sequence], optimizer_name: str = 'DE', budget: int = 1000):
        edge_list = list(self.graph.edges.keys())
        param_dim = len(edge_list)

        init_theta = [inventory.initial_inventory[i] for i in self.graph.supply_nodes for _ in range(len(self.graph.demand_nodes))]  # crude init
        max_inventory = max(init_theta)

        param = ng.p.Array(init=init_theta, lower=0.0, upper=max_inventory)
        optimizer = ng.optimizers.registry[optimizer_name](parametrization=param, budget=budget)
        
        # Use a bound method in the lambda
        best_candidate = optimizer.minimize(lambda x: -self._evaluate_theta_vector(x, inventory, train_samples))
        return best_candidate

class AdaptiveThresholdsFulfillment:
    
    def __init__(self, graph: Graph):
        self.graph = graph
        self.edge_map = {}
        self.edge_list = list(graph.edges.keys())

        for e, (i, j) in enumerate(self.edge_list):
            self.edge_map[(i, j)] = e

    def fulfill(self, sequence: Sequence, inventory: Inventory, theta_vector: List[float], gamma: float) -> float:
        uses = defaultdict(int)
        current_inventory = inventory.initial_inventory.copy()
        total_reward = 0.0
        T = len(sequence)

        for t, request in enumerate(sequence.requests):
            j = request.demand_node
            demand_neighbors = self.graph.demand_nodes[j].neighbors
            chosen_supply = -1
            best_reward = -float('inf')

            for i in demand_neighbors:
                edge_id = self.edge_map[(i, j)]
                soft_threshold = theta_vector[edge_id] * (1 - np.exp(-gamma * t / T))

                if current_inventory[i] > 0 and uses[(i, j)] < soft_threshold:
                    r_ij = self.graph.edges[(i, j)].reward
                    if r_ij > best_reward:
                        best_reward = r_ij
                        chosen_supply = i

            if chosen_supply != -1:
                current_inventory[chosen_supply] -= 1
                uses[(chosen_supply, j)] += 1
                total_reward += self.graph.edges[(chosen_supply, j)].reward

        return total_reward

    def _evaluate(self, param_vector: List[float], inventory: Inventory, sequences: List[Sequence]) -> float:
        theta_vector = param_vector[:-1]
        gamma = param_vector[-1]

        total_reward = 0.0
        for sequence in sequences:
            reward = self.fulfill(sequence, inventory, theta_vector, gamma)
            total_reward += reward

        return total_reward / len(sequences) if sequences else 0.0

    def train(self, inventory: Inventory, train_samples: List[Sequence], optimizer_name: str = 'DE', budget: int = 1000):
        edge_list = self.edge_list
        param_dim = len(edge_list)

        init_theta = [inventory.initial_inventory[i] for (i, _) in edge_list]
        init_gamma = [0.5]  # start with moderate smoothing

        param = ng.p.Array(init=init_theta + init_gamma)
        param.set_bounds(
            lower=[0.0] * param_dim + [0.0],   # theta >= 0, gamma >= 0
            upper=[4 * inventory.initial_inventory[i] for (i, _) in edge_list] + [5.0]  # loose upper bounds
        )

        optimizer = ng.optimizers.registry[optimizer_name](parametrization=param, budget=budget)
        best_candidate = optimizer.minimize(lambda x: -self._evaluate(x, inventory, train_samples))

        best_vector = best_candidate.value
        best_theta = best_vector[:-1]
        best_gamma = best_vector[-1]

        return best_theta, best_gamma
        
class TimeSupplyEnhancedMPB(MultiPriceBalanceFulfillment):
    
    def __init__(self, graph: Graph):
        super().__init__(graph)
        self.graph = graph
        self.num_parameters = len(graph.supply_nodes) + 1
        

    def fulfill(self, sequence: Sequence, inventory: Inventory, theta_and_gamma: list[float]) -> float:
        
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

    def _evaluate(self, theta_and_gamma: list[float], inventory: Inventory, sequences: List[Sequence]) -> float:
        # m = len(self.graph.supply_nodes)
        # theta_list = theta_and_gamma[:m]
        # gamma = theta_and_gamma[m]

        total_reward = 0.0
        for sequence in sequences:
            reward = self.fulfill(sequence, inventory, theta_and_gamma)
            total_reward += reward

        return total_reward / len(sequences) if sequences else 0.0

    def train(self, inventory: Inventory, sequences: List[Sequence], optimizer_name="DE", budget=500):
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
        

    def fulfill(self, sequence: Sequence, inventory: Inventory, theta_list: list[float]) -> float:
        
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

    def _evaluate(self, theta_list: list[float], inventory: Inventory, sequences: List[Sequence]) -> float:
        # m = len(self.graph.supply_nodes)
        # theta_list = theta_and_gamma[:m]
        # gamma = theta_and_gamma[m]

        total_reward = 0.0
        for sequence in sequences:
            reward = self.fulfill(sequence, inventory, theta_list)
            total_reward += reward

        return total_reward / len(sequences) if sequences else 0.0

    def train(self, inventory: Inventory, sequences: List[Sequence], optimizer_name="DE", budget=500):
        m = len(self.graph.supply_nodes)
        init = [0.0] * m 

        param = ng.p.Array(init=init)
        param.set_bounds(lower=[-3.0] * m , upper=[3.0] * m )


        optimizer_cls = ng.optimizers.registry[optimizer_name]
        optimizer = optimizer_cls(parametrization=param, budget=budget)

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

    def train(self, inventory: Inventory, sequences: List[Sequence], optimizer_name="DE", budget=500):
        
        init = [0.0]

        param = ng.p.Array(init=init)
        param.set_bounds(lower = [-1.0], upper= [10.0])


        optimizer_cls = ng.optimizers.registry[optimizer_name]
        optimizer = optimizer_cls(parametrization=param, budget=budget)

        best = optimizer.minimize(lambda x: -self._evaluate(x, inventory, sequences))
        best_params = best.value
        
        self.best_gamma = best_params[0]

        return self.best_gamma
    
class DemandTrackingMPB(TimeSupplyEnhancedMPB):
    def fulfill(self, sequence: Sequence, inventory: Inventory, theta_list: List[float], gamma: float, beta_list: List[float]) -> float:
        current_inventory = inventory.initial_inventory.copy()
        total_reward = 0
        T = len(sequence)
        arrival_counts = defaultdict(int)  # Count of how often each demand node has arrived

        for t, request in enumerate(sequence.requests):
            j = request.demand_node
            arrival_counts[j] += 1
            d_jt = arrival_counts[j] / (t + 1)

            best_score = -float('inf')
            chosen_supply = -1

            for i in self.graph.demand_nodes[j].neighbors:
                if current_inventory[i] <= 0:
                    continue

                used_fraction = 1.0 - current_inventory[i] / inventory.initial_inventory[i]
                base_cost = self.phi(i, used_fraction)
                modifier = base_cost + theta_list[i] 
                cost = modifier * np.exp(gamma * t / T) * np.exp(beta_list[j] * d_jt)
                score = self.graph.edges[(i, j)].reward - cost

                if score > best_score:
                    best_score = score
                    chosen_supply = i

            if chosen_supply != -1 and best_score >= 0:
                current_inventory[chosen_supply] -= 1
                total_reward += self.graph.edges[(chosen_supply, j)].reward

        return total_reward

    def _evaluate(self, param_vector: List[float], inventory: Inventory, sequences: List[Sequence]) -> float:
        m = len(self.graph.supply_nodes)
        n = len(self.graph.demand_nodes)

        theta_list = param_vector[:m]
        gamma = param_vector[m]
        beta_list = param_vector[m + 1:]

        total_reward = 0.0
        for sequence in sequences:
            reward = self.fulfill(sequence, inventory, theta_list, gamma, beta_list)
            total_reward += reward

        return total_reward / len(sequences) if sequences else 0.0

    def train(self, inventory: Inventory, sequences: List[Sequence], optimizer_name="DE", budget=500):
        m = len(self.graph.supply_nodes)
        n = len(self.graph.demand_nodes)

        init = [0.0] * (m + 1 + n)  # [theta_1, ..., theta_m, gamma, beta_1, ..., beta_n]
        param = ng.p.Array(init=init)
        param.set_bounds(
            lower=[-5.0] * m + [-3.0] + [-3.0] * n,
            upper=[5.0] * m + [3.0] + [3.0] * n,
        )

        optimizer_cls = ng.optimizers.registry[optimizer_name]
        optimizer = optimizer_cls(parametrization=param, budget=budget)

        best = optimizer.minimize(lambda x: -self._evaluate(x, inventory, sequences))
        best_params = best.value

        self.best_thetas = best_params[:m]
        self.best_gamma = best_params[m]
        self.best_betas = best_params[m + 1:]

        return self.best_thetas, self.best_gamma, self.best_betas