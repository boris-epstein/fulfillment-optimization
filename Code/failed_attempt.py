from Graph import Graph
from Demand import CorrelGenerator, IndepGenerator, RWGenerator, Sequence
from FulfillmentOptimization import Fulfillment, Inventory, InventoryOptimizer, BalanceFulfillment, DualMirrorDescentFulfillment
import time
import numpy as np
from collections import defaultdict
# import csv
# from statistics import mean, stdev
from MathPrograms import MathPrograms
import csv
from typing import Dict, List
import matplotlib.pyplot as plt

def compute_rho(inventory: Inventory, expected_T: float):
    
    rho = {}
    for supply_node_id in inventory.initial_inventory:
        rho[supply_node_id] = inventory.initial_inventory[supply_node_id]/expected_T

    return rho

def compute_ET(sequences: List[Sequence]):
    
    n_sequences = len(sequences)
    
    ET = 0
    
    for sequence in sequences:
        ET += sequence.length/n_sequences
    
    return ET
    


def experiment(
        mean_demand: float,
        total_inventory: int,
        graph_name: str,
        weights_mode: str,
        demand_model: str,
        train_seed: int,
        test_seed: int,
        n_train_sequences: int,
        n_test_sequences: int,
    ):
    
    if graph_name=='complete':
        supply_nodes, demand_nodes, rewards = ('complete_supplynodes_seed=20.csv', 'complete_demandnodes_seed=20.csv', 'complete_rewards_seed=20.csv')

    if graph_name=='longchain':
        supply_nodes, demand_nodes, rewards = ('longchain_supplynodes_seed=14.csv', 'longchain_demandnodes_seed=14.csv', 'longchain_rewards_seed=14.csv')

    if graph_name=='starlike':
        supply_nodes, demand_nodes, rewards = ('starlike_supplynodes_seed=8.csv', 'starlike_demandnodes_seed=8.csv', 'starlike_rewards_seed=8.csv')

    ### Graph Reading
    print('Reading graph')
    graph = Graph(supply_nodes, demand_nodes, rewards)
    
    # create myopic priority list
    myopic_scores = {(edge.supply_node_id, edge.demand_node_id): edge.reward for edge in graph.edges.values()}
    graph.construct_priority_list( 'myopic', myopic_scores, allow_rejections=False)

    ### Demand Node Weigheting
    demand_nodes_list = list(graph.demand_nodes.keys())
    
    train_generator = RWGenerator(mean = mean_demand, demand_nodes=demand_nodes_list, seed = train_seed )
    test_generator = RWGenerator(mean = mean_demand, demand_nodes=demand_nodes_list, seed=test_seed)
    
    
    
    print('Generating training sequence')
    train_sequences = [train_generator.generate_sequence() for _ in range(n_train_sequences)]
    print('Generating testing sequence')
    test_sequences = [test_generator.generate_sequence() for _ in range(n_test_sequences)]


    expected_T = compute_ET(train_sequences)

    print(expected_T)
    
    inventory_optimizer = InventoryOptimizer(graph)
    
    inventory_list: List[Inventory] = []
    print('Obtaining offline inventory with LP rounding')
    offline_inventory = inventory_optimizer.offline_inventory_placement_rounding(train_sequences, total_inventory=total_inventory)
    inventory_list.append(offline_inventory)
    
    rhos = {}
    
    inventories: Dict[str,Inventory] = {}
    
    for inventory in inventory_list:
        inventories[inventory.name] = inventory
        rhos[inventory.name] = compute_rho(inventory, expected_T)
        print(inventory.name, inventory.initial_inventory, rhos[inventory.name])
    
    
    programs = MathPrograms(graph)

    duals = {}
    # Generate scores
    print('Generating scores for fixed priority list fulfillment')
    scores = {} # dictionary with (penalizer, inventory name) as keys
    for inventory in inventories.values():
        
        # fluid, fluid_inventory_constraints = programs.fluid_linear_program_fixed_inventory(train_generator.average_demands, inventory=inventory)
        # fluid.optimize()
        # scores['fluid', inventory.name] =  {(edge.supply_node_id, edge.demand_node_id): edge.reward - fluid_inventory_constraints[edge.supply_node_id].Pi for edge in graph.edges.values()}
        
        offline, offline_inventory_constraints = programs.offline_linear_program_fixed_inventory(train_sequences, inventory=inventory)
        offline.optimize()
        duals['offline', inventory.name] = [sum(offline_inventory_constraints[supply_node_id, sample_index].Pi for sample_index in range(n_train_sequences))/n_train_sequences for supply_node_id in graph.supply_nodes]

        scores['offline', inventory.name] =  {(edge.supply_node_id, edge.demand_node_id): edge.reward - sum(offline_inventory_constraints[edge.supply_node_id, sample_index].Pi for sample_index in range(n_train_sequences))/n_train_sequences for edge in graph.edges.values()}


    print('Generating priority lists')
    for inventory in inventories.values():

        # graph.construct_priority_list(('fluid_with_rejections', inventory.name), scores['fluid',inventory.name], allow_rejections=True)
        # graph.construct_priority_list(('fluid_without_rejections', inventory.name), scores['fluid',inventory.name], allow_rejections=False)
        graph.construct_priority_list(('offline_with_rejections', inventory.name), scores['offline',inventory.name], allow_rejections=True)
        # graph.construct_priority_list(('offline_without_rejections', inventory.name), scores['offline',inventory.name], allow_rejections=False)



    fulfillment = Fulfillment(graph)
    balance_fulfillment = BalanceFulfillment(graph)
    mirror_descent_fulfillment = DualMirrorDescentFulfillment(graph)
    
    fixed_list_fulfillment_algorithms = ['myopic', 'offline_with_rejections']
    
    rewards = {}

    print('Computing offline relaxation for test data')
    offline_relaxation, _ = programs.offline_linear_program_variable_inventory(test_sequences, total_inventory)
    offline_relaxation.optimize()
    rewards['offline_relaxation'] = offline_relaxation.ObjVal/n_test_sequences

    print('Computing rewards for fulfillment algorithms')
    for inventory in inventories.values():
        offline, _ = programs.offline_linear_program_fixed_inventory(test_sequences, inventory)
        offline.optimize()
        rewards[inventory.name, 'offline'] = offline.ObjVal/n_test_sequences
        print(inventory.name, 'offline',rewards[inventory.name, 'offline'])
        
        for algorithm in fixed_list_fulfillment_algorithms:
            print(f'Simulating {inventory.name} inventory with {algorithm} fulfillment')
            inventory = inventories[inventory.name]
            rewards[inventory.name, algorithm] = 0
            for sequence in test_sequences:
                if algorithm=='myopic':
                    list_name = 'myopic'
                else:
                    list_name = (algorithm, inventory.name)
                _, collected_rewards, _ = fulfillment.fixed_list_fulfillment(sequence, inventory, list_name, allow_rejections=False, verbose=False)
                rewards[inventory.name, algorithm]+= collected_rewards/n_test_sequences
    
        print(f'Simulating {inventory.name} balance fulfillment')
        rewards[inventory.name, 'balance'] = 0
        for sequence in test_sequences:
            _, collected_rewards, _ = balance_fulfillment.fulfill(sequence, inventory, verbose=False)
            rewards[inventory.name, 'balance'] += collected_rewards/n_test_sequences

        print(f'Simulating {inventory.name} mirror dual descent fulfillment')
        step_sizes = [0.0001, 1/np.sqrt(expected_T)]
        update_rules = ['subgradient_descent', 'multiplicative_weights']
        for update_rule in update_rules:
            for step_size in step_sizes:
                name = f'mirror_{update_rule}_sz{step_size:.3f}'
                rewards[inventory.name, name] = 0
                for sequence in test_sequences:
                    _, collected_rewards, _ = mirror_descent_fulfillment.fulfill(sequence, inventory, duals['offline', inventory.name], rhos[inventory.name], step_size, update_rule, verbose=False)
                    rewards[inventory.name, name] += collected_rewards/n_test_sequences

            
    return rewards
    
    
    
def main():
    results = {}
    
    
    graph_names = ['starlike']
    # graph_names=['longchain']
   
    
    # demand_models = ['deter','indep', 'correl']
    # demand_models = ['indep', 'deter']
    # demand_models = ['correl']
    demand_models = ['randomwalk']
    
    weight_modes = ['rewardbased']
    # weight_modes = ['uniform']
    
    mean_demand = 120
    # mean_demand = 10
    
    
    total_inventories = [30, 45, 60, 75, 90]
    # total_inventories = [100, 200, 300]
    # total_inventories = [30, 60, 90]

    
    n_test_samples = 3000
    n_train_samples = 3000
    
    train_seed = 10
    test_seed = 20



    
    for graph_name in graph_names:
        for demand_model in demand_models:
            for weight_mode in weight_modes:
                for total_inventory in total_inventories:
                    print(graph_name, demand_model, weight_mode, total_inventory)
                    results[graph_name, demand_model, weight_mode, total_inventory] = experiment(
                        mean_demand,
                        total_inventory,
                        graph_name,
                        weight_mode,
                        demand_model,
                        train_seed,
                        test_seed,
                        n_train_samples,
                        n_test_samples
                        )
                    
                    print(results[graph_name, demand_model, weight_mode, total_inventory])
                    print('')
    
    output_name = 'whole_experiment_n_samples_1000_resolve_samples_80'
    
    

    
    


if __name__ == '__main__':
    main()