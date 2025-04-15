
from collections import defaultdict
from itertools import combinations, product
from typing import Dict, List

from Demand import Sequence, Request, MarkovianGenerator
from Graph import Graph
from FulfillmentOptimization import Inventory

import numpy as np



def generate_tuples(n, t):
    # Generate all combinations of (t + n - 1) choose (n - 1)
    for comb in combinations(range(t + n - 1), n - 1):
        # Convert positions of "bars" into tuple of counts
        result = [comb[0]]  # First element
        for i in range(1, n - 1):
            result.append(comb[i] - comb[i - 1] - 1)
        result.append(t + n - 2 - comb[-1])  # Last element
        yield tuple(result)

def generate_bounded_tuples(initial_tuple):
    return product(*(range(a + 1) for a in initial_tuple))


def generate_bounded_tuples_with_sum(initial_tuple, T, current_tuple=(), index=0, current_sum=0):
    # Base case: If we have filled the tuple
    if index == len(initial_tuple):
        if current_sum >= T:
            yield current_tuple  # Only yield if sum condition is met
        return
    
    # Determine the remaining sum required to reach T
    min_required = max(0, T - current_sum - sum(initial_tuple[index+1:]))  # Pruning

    # Iterate only through feasible values
    for value in range(min_required, initial_tuple[index] + 1):
        yield from generate_bounded_tuples_with_sum(
            initial_tuple, T, current_tuple + (value,), index + 1, current_sum + value
        )

class IndependentDynamicProgram:
    def __init__(self, graph:Graph) -> None:
        self.graph = graph
    
    def compute_optimal_policy(self, inventory: Inventory, T: int, p: Dict[int, List[float]]):
        
        # Define state space
        inventory_states= self.generate_inventory_state_space(inventory, T)
        
        # Define value functions and optimal actions
        
        value_function = defaultdict(float)
        optimal_action = {}
        
        # Boundary Conditions

        
        for inventory_state in inventory_states:
            for demand_node_id in self.graph.demand_nodes:
                value_function[inventory_state, T-1, demand_node_id], optimal_action[inventory_state, T-1,demand_node_id] = self.find_best_final_supply_node(demand_node_id, inventory_state)
                
        
        # Bellman equations
        
        t = T-2
        
        for t in range(T-2, -1,-1):
  
            for inventory_state in inventory_states:
                for demand_node_id in self.graph.demand_nodes:
                    if sum(inventory_state)>= inventory.total_inventory-t:
                        value_function[inventory_state, t, demand_node_id], optimal_action[inventory_state,t, demand_node_id] = self.find_best_continuation(demand_node_id,inventory_state, value_function, p ,t)

        
        return DPOutput(value_function, optimal_action)
        
    def find_best_final_supply_node(self, demand_node_id, inventory_state):
        best_supply_node = -1
        best_reward = 0
        
        demand_node = self.graph.demand_nodes[demand_node_id]
        
        for supply_node_id in demand_node.neighbors:
            if inventory_state[supply_node_id]>0 and self.graph.edges[supply_node_id, demand_node_id].reward > best_reward:
                best_reward = self.graph.edges[supply_node_id, demand_node_id].reward
                best_supply_node = supply_node_id
                
        
        return best_reward, best_supply_node
    
    
    
    def find_best_continuation(self, demand_node_id, inventory_state, value_function, p, t):
        best_reward = 0
        best_supply_node = -1
        
        demand_node = self.graph.demand_nodes[demand_node_id]
        
        for supply_node_id in demand_node.neighbors:
            if inventory_state[supply_node_id]>0:
                next_inventory = [inv for inv in inventory_state]
                next_inventory[supply_node_id] -=1
                next_inventory = tuple(next_inventory)
                
                action_reward = 0
                
                for next_demand_node in self.graph.demand_nodes:
                    action_reward += (self.graph.edges[supply_node_id,demand_node_id].reward + value_function[next_inventory, t+1, next_demand_node])*p[t+1][next_demand_node]

                if action_reward > best_reward:
                    best_reward = action_reward
                    best_supply_node = supply_node_id
        
        
        no_action_reward = 0
        
        for next_demand_node in self.graph.demand_nodes:
            no_action_reward += value_function[inventory_state,t+1, next_demand_node]*p[t+1][next_demand_node]
        if no_action_reward > best_reward:
            best_reward = no_action_reward
            best_supply_node = -1
        
        return best_reward, best_supply_node
    
        
    def generate_inventory_state_space(self, inventory: Inventory, T: int):
        
        initial_inventory_tuple = tuple([inventory.initial_inventory[supply_node_id] for supply_node_id in self.graph.supply_nodes])
        
        bounded_tuples = list(generate_bounded_tuples_with_sum(initial_inventory_tuple, inventory.total_inventory - T+1))
        
        return bounded_tuples
        

class MarkovianDynamicProgram:
    def __init__(self, graph:Graph) -> None:
        self.graph = graph
    
    def compute_optimal_policy(self, inventory: Inventory, T: int, transition_matrix):
        
        # Define state space
        inventory_states = self.generate_inventory_state_space(inventory, T)
        
        # Define value functions and optimal actions
        
        value_function = defaultdict(float)
        optimal_action = {}
        
        # Boundary Conditions

        
        for inventory_state in inventory_states:
            for demand_node_id in self.graph.demand_nodes:
                value_function[inventory_state, T-1, demand_node_id], optimal_action[inventory_state, T-1,demand_node_id] = self.find_best_final_supply_node(demand_node_id, inventory_state)
                
        
        # Bellman equations
        
        
        for t in range(T-2, -1,-1):
  
            for inventory_state in inventory_states:
                for demand_node_id in self.graph.demand_nodes:
                    if sum(inventory_state)>= inventory.total_inventory-t:
                        value_function[inventory_state, t, demand_node_id], optimal_action[inventory_state,t, demand_node_id] = self.find_best_continuation(demand_node_id,inventory_state, value_function, transition_matrix ,t)

        
        return DPOutput(value_function, optimal_action)

    def find_best_final_supply_node(self, demand_node_id, inventory_state):
        best_supply_node = -1
        best_reward = 0
        
        demand_node = self.graph.demand_nodes[demand_node_id]
        
        for supply_node_id in demand_node.neighbors:
            if inventory_state[supply_node_id]>0 and self.graph.edges[supply_node_id, demand_node_id].reward > best_reward:
                best_reward = self.graph.edges[supply_node_id, demand_node_id].reward
                best_supply_node = supply_node_id
                
        
        return best_reward, best_supply_node
    
    
    
    def find_best_continuation(self, demand_node_id, inventory_state, value_function, transition_matrix, t):
        
        best_reward = 0
        best_supply_node = -1
        
        demand_node = self.graph.demand_nodes[demand_node_id]
        
        for supply_node_id in demand_node.neighbors:
            if inventory_state[supply_node_id]>0:
                next_inventory = [inv for inv in inventory_state]
                next_inventory[supply_node_id] -=1
                next_inventory = tuple(next_inventory)
                
                action_reward = 0
                
                for next_demand_node in self.graph.demand_nodes:
                    action_reward += (self.graph.edges[supply_node_id,demand_node_id].reward + value_function[next_inventory, t+1, next_demand_node])*transition_matrix[demand_node.id, next_demand_node]

                if action_reward > best_reward:
                    best_reward = action_reward
                    best_supply_node = supply_node_id
        
        
        no_action_reward = 0
        
        for next_demand_node in self.graph.demand_nodes:
            no_action_reward += value_function[inventory_state,t+1, next_demand_node]*transition_matrix[demand_node.id, next_demand_node]
        if no_action_reward > best_reward:
            best_reward = no_action_reward
            best_supply_node = -1
        
        return best_reward, best_supply_node
    
        
    def generate_inventory_state_space(self, inventory: Inventory, T: int):
        
        
        initial_inventory_tuple = tuple([inventory.initial_inventory[supply_node_id] for supply_node_id in self.graph.supply_nodes])
        
        bounded_tuples = list(generate_bounded_tuples_with_sum(initial_inventory_tuple, inventory.total_inventory - T+1))
        
        return bounded_tuples

class DPOutput:
    def __init__(self, value_function, optimal_action):
        self.value_function = value_function
        self.optimal_action = optimal_action        
    
class ModelEstimator:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        
    
    def estimate_iid(self, sequences: List[Sequence]) -> List[float]:
        p = defaultdict(float)
        
        T = sequences[0].length
        denominator = len(sequences) * T
        
        for sequence in sequences:
            for request in sequence.requests:
                p[request.demand_node]+= 1
        
        p_list = [None for demand_node_id in self.graph.demand_nodes]
        for demand_node_id in self.graph.demand_nodes:
            p_list[demand_node_id] = p[demand_node_id]/denominator
        
        return p_list
    
    def estimate_independent(self, sequences: List[Sequence]) -> Dict[int, List[float]]:
        p = {}
        T = sequences[0].length
        n_sequences = len(sequences)
        for t in range(T):
            p[t] = [0 for _ in self.graph.demand_nodes]

        for sequence in sequences:
            t=0
            for request in sequence.requests:
                p[t][request.demand_node] += 1/n_sequences
                t+=1

        return p
    
    def estimate_markovian(self, sequences: List[Sequence]):
        
        
        n_demand_nodes = len(self.graph.demand_nodes)
        initial_distribution = [0 for _ in self.graph.demand_nodes]
        n_sequences = len(sequences)
        transition_matrix = np.zeros((n_demand_nodes, n_demand_nodes))
        
        zeros_vector = np.zeros(n_demand_nodes)
        
        for sequence in sequences:
            T = sequence.length
            initial_distribution[sequence.requests[0].demand_node] += 1/n_sequences
            last_demand_node = sequence.requests[0].demand_node
            for t in range(1,T):
                request = sequence.requests[t]
                demand_node = request.demand_node
                transition_matrix[last_demand_node, demand_node] += 1
                last_demand_node = demand_node
                
        # Normalize distributions/ check if there are no samples
        
        for demand_node_id in self.graph.demand_nodes:
            
            if all(transition_matrix[demand_node_id,:] == zeros_vector):
                transition_matrix[demand_node_id,:] = np.ones(n_demand_nodes)/n_demand_nodes
            
            else:
                transition_matrix[demand_node_id,:] = transition_matrix[demand_node_id,:] / transition_matrix[demand_node_id,:].sum()
        
        return transition_matrix, initial_distribution
        
        
    
if __name__ == '__main__':
    # Example usage:
    # initial_tuple = (3, 1, 2)  # Example tuple
    # tuples = list(generate_bounded_tuples_with_sum(initial_tuple,3))
    # print(tuples)
    
    supply_nodes, demand_nodes, rewards = ('simplest_graph_109_supplynodes.csv', 'simplest_graph_109_demandnodes.csv', 'simplest_graph_109_rewards.csv')

    ### Graph Reading
    print('Reading graph')
    graph = Graph(supply_nodes, demand_nodes, rewards)
    
    
    T = 3
    
    p = {}
    p[0] = [0, 1 ,0]
    p[1] = [1,0,0]
    p[2] = [0,0,1]
    
    initial_inventory = Inventory({0:1, 1:2})
    
   
    
    indep_dp = IndependentDynamicProgram(graph)
    
    output = indep_dp.compute_optimal_policy(initial_inventory, T, p)
    for state in output.optimal_action:
        print(state)
        print(output.optimal_action[state])
        print(output.value_function[state])
        print('')
    
    n_samples = 100
    intial_distribution = [1/3, 1/3, 1/3]
    
    transition_matrix = np.array(
        [[0.8,0.05,0.15],
         [0.1,0.7,0.2],
         [0.1,0.1,0.8]
         ]
    )
    demand_nodes_list = [demand_node_id for demand_node_id in graph.demand_nodes]
    
    generator = MarkovianGenerator(demand_nodes_list, transition_matrix, intial_distribution, seed =2)
    
    sequences = [generator.generate_sequence(T) for _ in range(n_samples)]

    estimator = ModelEstimator(graph)
    
    estimated_tm, estimated_id = estimator.estimate_markovian(sequences)
    print(estimated_id)
    print(estimated_tm)

    print(intial_distribution)
    print(transition_matrix)
    
    markov_dp = MarkovianDynamicProgram(graph)
    
    dp_output = markov_dp.compute_optimal_policy(initial_inventory,T, transition_matrix)
    print(dp_output.optimal_action)