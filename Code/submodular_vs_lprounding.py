from Graph import Graph
from Demand import CorrelGenerator, IndepGenerator
from FulfillmentOptimization import InventoryOptimizer
from MathPrograms import MathPrograms
from dataclasses import dataclass

import csv
import numpy as np
import time

@dataclass
class ExperimentResult:
    train_greedy_performance: float
    train_lp_performance: float
    train_lp_benchmark: float
    test_greedy_performance: float
    test_lp_performance: float
    test_lp_benchmark: float
    l1_norm: float
    off_lp_time: float
    off_greedy_time: float


def experiment(mean_demand: float,
               total_inventory: int,
               graph_name: str,
               weights_mode: str,
               demand_model: str,
               train_seed: int,
               test_seed: int,
               n_train_sequences: int,
               n_test_sequences: int,
            ):
    """
    Experiment goal: compare the value of the offline surrogate with solutions obtained using the lp rounding method and the greedy method
    
    Experiment logic:
    - Have training and testing sets. Both drawn from the same distribution.
    - With train set: run the inventory placement procedures
    - With test set: evaluate offline surrogate with fixed inventories
    
    Variants:
    - Demand model: Correl (deterministic and random) and Indep
    - Graphs: Three of them, with both uniform weights and weights proportional to sum of rewards
    - Initial inventory: 1.5x, 1.25x, 0.75x and 0.5x average demand
    
    What to look at:
    - % of times that offline lp solution is not integer
    - Objective value obtained. Can be measured as ratio between greedy/lp, or ratio between each benchmark and the offline relaxation on the test set.
        
    """
    
    # mean_demand = 200
    # total_inventory = 300
    
    # GRAPH_NAME = 'complete' # complete, longchain or startlike
    # WEIGHTS_MODE = 'rewardbased' # uniform or rewardbased
    # DEMAND_MODEL = 'correl' # deter, indep or correl. Correl will be with exponential mode
    
    # TRAINING_SEED = 10
    # TESTING_SEED = 20

    # N_TRAINING_SEQUENCES = 500
    # N_TESTING_SEQUENCES = 500
    
    if graph_name=='complete':
        supply_nodes, demand_nodes, rewards = ('complete_supplynodes_seed=20.csv', 'complete_demandnodes_seed=20.csv', 'complete_rewards_seed=20.csv')

    if graph_name=='longchain':
        supply_nodes, demand_nodes, rewards = ('longchain_supplynodes_seed=14.csv', 'longchain_demandnodes_seed=14.csv', 'longchain_rewards_seed=14.csv')

    if graph_name=='starlike':
        supply_nodes, demand_nodes, rewards = ('starlike_supplynodes_seed=8.csv', 'starlike_demandnodes_seed=8.csv', 'starlike_rewards_seed=8.csv')

    ### Graph Reading
    print('Reading graph')
    graph = Graph(supply_nodes, demand_nodes, rewards)

    ### Demand Node Weigheting
    demand_nodes_list = list(graph.demand_nodes.keys())
    if weights_mode == 'uniform':
        weights = np.array([ 1.0 for _ in demand_nodes_list])
    if weights_mode == 'rewardbased':
        weights = np.array([ sum(graph.edges[supply_node_id, demand_node_id].reward for supply_node_id in graph.demand_nodes[demand_node_id].neighbors) for demand_node_id in demand_nodes_list])
        
    weights /= weights.sum()

    ### Random Generator Initialization

    if demand_model == 'deter':
        train_generator = CorrelGenerator(mean = mean_demand, demand_nodes=demand_nodes_list, weights=weights, seed = train_seed, distribution='deterministic')
        test_generator = CorrelGenerator(mean = mean_demand, demand_nodes=demand_nodes_list, weights=weights, seed = test_seed, distribution='deterministic')

    if demand_model == 'indep':
        means = [weights[i] * mean_demand for i in range(len(weights))]
        train_generator = IndepGenerator(means = means, demand_nodes=demand_nodes_list, seed = train_seed,distribution='geometric')
        test_generator = IndepGenerator(means = means, demand_nodes=demand_nodes_list, seed = test_seed,distribution='geometric')
    
    if demand_model == 'correl':
        train_generator = CorrelGenerator(mean = mean_demand, demand_nodes=demand_nodes_list, weights=weights, seed = train_seed, distribution='geometric')
        test_generator = CorrelGenerator(mean = mean_demand, demand_nodes=demand_nodes_list, weights=weights, seed = test_seed, distribution='geometric')


    ### Sampling of train and test data

    print('Generating training sequence')
    train_sequences = [train_generator.generate_sequence() for _ in range(n_train_sequences)]
    print('Generating testing sequence')
    test_sequences = [test_generator.generate_sequence() for _ in range(n_test_sequences)]

    # for i in range(10):
    #     seq1 = [request.demand_node for request in train_sequences[i].requests]
    #     seq2 = [request.demand_node for request in train_sequences[i].requests]
    #     print(
    #         seq1
    #     )
    #     print(seq2)

    ### Obtain Offline inventory placements
    
    inventory_optimizer = InventoryOptimizer(graph)
    
    print('Obtaining offline inventory with LP rounding')
    start = time.time()
    offline_lp_rounding_inventory = inventory_optimizer.offline_inventory_placement_rounding(train_sequences, total_inventory=total_inventory)
    off_lp_time = time.time() - start
    
    print('Obtaining offline inventory with Greedy')
    start2 = time.time()
    offline_greedy_inventory = inventory_optimizer.offline_greedy_inventory_placement(train_sequences, total_inventory=total_inventory)
    off_greedy_time = time.time() - start2
    
    print(offline_lp_rounding_inventory.initial_inventory, off_lp_time)
    print(offline_greedy_inventory.initial_inventory, off_greedy_time)

    l1_norm = sum(
        np.abs( offline_lp_rounding_inventory.initial_inventory[supply_node_id] - offline_greedy_inventory.initial_inventory[supply_node_id] )
        for supply_node_id in graph.supply_nodes
        )

    ### Evaluate Placements on Test Data
    
    math_programs = MathPrograms(graph)
    
    print('Evaluating LP rounding inventory on train sample')
    offline_rounding_lp_train, _ = math_programs.offline_linear_program_fixed_inventory(train_sequences, offline_lp_rounding_inventory)
    offline_rounding_lp_train.optimize()
    print('Evaluating LP rounding inventory on train sample')
    offline_greedy_train, _ = math_programs.offline_linear_program_fixed_inventory(train_sequences, offline_greedy_inventory)
    offline_greedy_train.optimize()
    print('Evaluating LP benchmark on train sample')
    offline_benchmark_train, _ = math_programs.offline_linear_program_variable_inventory(train_sequences, total_inventory=total_inventory)
    offline_benchmark_train.optimize()
    
    print('Evaluating LP rounding inventory on test sample')
    offline_rounding_lp_test, _ = math_programs.offline_linear_program_fixed_inventory(test_sequences, offline_lp_rounding_inventory)
    offline_rounding_lp_test.optimize()
    print('Evaluating LP rounding inventory on test sample')
    offline_greedy_test, _ = math_programs.offline_linear_program_fixed_inventory(test_sequences, offline_greedy_inventory)
    offline_greedy_test.optimize()
    print('Evaluating LP benchmark on test sample')
    offline_benchmark_test, _ = math_programs.offline_linear_program_variable_inventory(test_sequences, total_inventory=total_inventory)
    offline_benchmark_test.optimize()
    
    # print(f'Performance using lp rounding: {offline_rounding_lp.ObjVal:.3f}')
    # print(f'Performance using greedy: {offline_greedy.ObjVal:.3f}')
    # print(f'Ratio between LP rounding and Greedy: {offline_rounding_lp.ObjVal/offline_greedy.ObjVal:.3f}')
    
    return ExperimentResult(train_greedy_performance=offline_greedy_train.ObjVal/n_train_sequences,
                            train_lp_performance=offline_rounding_lp_train.ObjVal/n_train_sequences,
                            train_lp_benchmark=offline_benchmark_train.ObjVal/n_train_sequences,
                            test_greedy_performance=offline_greedy_test.ObjVal/n_test_sequences,
                            test_lp_performance=offline_rounding_lp_test.ObjVal/n_test_sequences,
                            test_lp_benchmark=offline_benchmark_test.ObjVal/n_test_sequences,
                            l1_norm = l1_norm,
                            off_lp_time = off_lp_time,
                            off_greedy_time = off_greedy_time
                        )

def main():
    
    results = {}
    graph_names = ['complete', 'longchain', 'starlike']
    demand_models = ['deter', 'indep', 'correl']
    # demand_models = ['deter','indep']
    # demand_models = ['indep']
    
    weight_modes = ['uniform', 'rewardbased']
    # weight_modes = ['uniform']
    
    mean_demand = 60
    # mean_demand = 10
    
    
    total_inventories = [30,45,60,75,90]
    # total_inventories = [150, 200, 250]
    # total_inventories = [250]
    # total_inventories = [5,10,15,20]
    # total_inventories = range(6,13)
    # total_inventories = [7]
    
    n_test_samples = 500
    n_train_samples = 500
    
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
    
    output_name = 'sm_lp_experiment_nsamples_500'
    save_output(results, output_name)

def save_output(results, output_name):
    with open(output_name + '.csv','w',newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter =',')
        spamwriter.writerow(['graph_name','demand_model',
                             'weight_mode',
                             'total_inventory',
                             'train_greedy_performance',
                             'train_lp_performance',
                             'train_lp_benchmark',
                             'test_greedy_performance',
                             'test_lp_performance',
                             'test_lp_benchmark',
                             'l1_norm',
                             'off_lp_time',
                             'off_greedy_time'
                        ])
        for graph_name, demand_model, weight_mode, total_inventory in results:
            result = results[graph_name, demand_model, weight_mode, total_inventory]
            spamwriter.writerow([
                graph_name,
                demand_model,
                weight_mode,
                total_inventory
            ]+[
                result.train_greedy_performance,
                result.train_lp_performance,
                result.train_lp_benchmark,
                result.test_greedy_performance,
                result.test_lp_performance,
                result.test_lp_benchmark,
                result.l1_norm,
                result.off_lp_time,
                result.off_greedy_time
            ])
            


if __name__=='__main__':
    # experiment(300,
    #            300,
    #            'complete',
    #            'rewardbased',
    #            'indep',
    #            0,
    #            1,
    #            100,
    #            100,
    #         )
    main()