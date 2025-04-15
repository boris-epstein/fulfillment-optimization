
from Demand import Sequence, MarkovianGenerator
from Graph import Graph, DemandNode
from FulfillmentOptimization import Inventory
from collections import defaultdict
from typing import List
import numpy as np


class SimplestGraphThresholdFulfillment:
    def __init__(self, graph: Graph):
        
        self.graph = graph
        
    def fulfill(self, sequence: Sequence, inventory : Inventory, thresholds: List[List[int]],verbose : bool = False):
        
        collected_rewards = 0
        lost_sales = 0
        total_fulfillments = 0
        number_fulfillments = defaultdict(int) # maps edges to number of times the edge was used
        
        max_inventory_at_1 = inventory.initial_inventory[1]
        
        current_inventories = inventory.initial_inventory.copy()
        t=0
        for request in sequence.requests:    
            demand_node = self.graph.demand_nodes[request.demand_node]
            # supply_node_chosen = self.choose_supply_node(demand_node, current_inventories,inventory)
            
            
            #logic for choosing supply node
            
            
            supply_node_chosen = self.choose_supply_node(demand_node, current_inventories, thresholds, t, max_inventory_at_1)
            
            
            if supply_node_chosen == -1:
                lost_sales+=1
                if verbose:
                    print(f'Demand from {demand_node.id} lost')
            else:
                if verbose:
                    print(f'Demand from {demand_node.id} fulfilled from warehouse {supply_node_chosen}')
                current_inventories[supply_node_chosen] -=1
                number_fulfillments[ supply_node_chosen, demand_node.id] += 1
                
                collected_rewards += self.graph.edges[supply_node_chosen, demand_node.id].reward

                total_fulfillments += 1          
            t+=1
        return number_fulfillments, collected_rewards, lost_sales
        
    def choose_supply_node(self, demand_node: DemandNode, current_inventories, thresholds,t, max_inventory_at_1):
        if demand_node.id == 0:
            if current_inventories[0] >0:
                supply_node_chosen = 0
            else:
                supply_node_chosen = -1
        
        elif demand_node.id == 2:
            if current_inventories[1] >0:
                supply_node_chosen = 1
            else:
                supply_node_chosen = -1

        else:
            if current_inventories[0] ==0 and current_inventories[1] ==0:
                supply_node_chosen = -1
                
            elif current_inventories[0] ==0 and current_inventories[1] >0:
                supply_node_chosen = 1
                
            elif current_inventories[0] >0 and current_inventories[1] ==0:
                supply_node_chosen = 0
                
            else:
                threshold = thresholds[max_inventory_at_1 - current_inventories[1]][t]
                if current_inventories[0]>= threshold:
                    supply_node_chosen = 0
                else:
                    supply_node_chosen = 1
        
        return supply_node_chosen            
def generate_grids(M, T, K):
    def backtrack(grid, t, x):
        if t == T:
            yield [row[:] for row in grid]
            return
        if x < 0:
            yield from backtrack(grid, t + 1, M - 1)
            return

        # Bounds for this cell
        below = grid[x + 1][t] if x + 1 < M else 1      # ≥ this
        left = grid[x][t - 1] if t > 0 else K           # ≤ this
        for val in range(max(below, 1), min(left, K) + 1):
            grid[x][t] = val
            yield from backtrack(grid, t, x - 1)

    grid = [[0] * T for _ in range(M)]
    yield from backtrack(grid, 0, M - 1)

    
def threshold_policy_brute_force_search(graph: Graph, sequences: List[Sequence], inventory: Inventory, T: int):
    fulfiller = SimplestGraphThresholdFulfillment(graph)
    M = inventory.initial_inventory[1]
    K = inventory.initial_inventory[0]+1
    threshold_grids = generate_grids(M, T, K)
    
    best_grid = -1
    best_reward = -1
    
    for thresholds in threshold_grids:
        current_reward = 0
        for sequence in sequences:
            _, reward, _ = fulfiller.fulfill(sequence, inventory, thresholds)
            current_reward += reward
        
        if current_reward > best_reward:
            best_reward = current_reward
            best_grid = thresholds
    # print(best_reward, best_grid)
    return best_grid
    
    
if __name__ == '__main__':
     
     
    supply_nodes, demand_nodes, rewards = ('simplest_graph_109_supplynodes.csv', 'simplest_graph_109_demandnodes.csv', 'simplest_graph_109_rewards.csv')

    ### Graph Reading
    print('Reading graph')
    graph = Graph(supply_nodes, demand_nodes, rewards)
    
    for g in generate_grids(M=2, T=4, K=3):
        for row in g:
            print(row)
        print("---")
        
    T = 6
    inventory = Inventory({0:3, 1:3})
    n_samples = 100
    
    intial_distribution = [1/3, 1/3, 1/3]
    
    transition_matrix = np.array(
        [[0,1,0],
         [1/3,1/3,1/3],
         [1/2,1/2,0]
         ]
    )
    demand_nodes_list = [demand_node_id for demand_node_id in graph.demand_nodes]
    
    generator = MarkovianGenerator(T,demand_nodes_list, transition_matrix, intial_distribution, seed =2)
    
    sequences = [generator.generate_sequence() for _ in range(n_samples)]
    
    best_grid = threshold_policy_brute_force_search(graph, sequences, inventory, T)
    