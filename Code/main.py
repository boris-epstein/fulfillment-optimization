from Graph import Graph
from Demand import CorrelGenerator, IndepGenerator
from FulfillmentOptimization import Fulfillment, Inventory, InventoryOptimizer
import time
import numpy as np
from collections import defaultdict
# import csv
# from statistics import mean, stdev
from MathPrograms import MathPrograms
import csv

def experiment(
        mean_demand: float,
        total_inventory: int,
        graph_name: str,
        weights_mode: str,
        demand_model: str,
        train_seed: int,
        test_seed: int,
        resolve_seed: int,
        n_train_sequences: int,
        n_test_sequences: int,
    ):
    """
    Experiment goal: compare how different combinations of inventory placement procedures and fulfillment algorithms perform in total reward collected
    compare the value of the offline surrogate with solutions obtained using the lp rounding method and the greedy method
    
    Experiment logic:
    - Have training and testing sets. Both drawn from the same distribution.
    - With train set: run the inventory placement procedures AND compute fixed priority lists based on the shadow prices of FLU and OFF
    - With test set: simluate how different fulfillment policies perform with the different inventory placements 
    
    Variants:
    - Demand model: Correl (deterministic and random) and Indep
    - Graphs: Three of them, with both uniform weights and weights proportional to sum of rewards
    - Initial inventory: 1.5x, 1.25x, 0.75x and 0.5x average demand
    
    What to look at:
    - % of times that offline lp solution is not integer
    - Reward collected on different combinations of load_factor/inventory/fulfillment (normalized by LP benchmark)
    """
    
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

    inventory_optimizer = InventoryOptimizer(graph)
    
    inventory_list = []
    print('Obtaining offline inventory with LP rounding')
    offline_inventory = inventory_optimizer.offline_inventory_placement_rounding(train_sequences, total_inventory=total_inventory)
    inventory_list.append(offline_inventory)
    print('Obtaining fluid inventory with LP rounding')
    fluid_inventory = inventory_optimizer.fluid_inventory_placement_rounding(average_demand=train_generator.average_demands, total_inventory=total_inventory, rescale_inventory=False)
    inventory_list.append(fluid_inventory)
    print('Obtaining fluid inventory with LP rounding and scaling inventory')
    fluid_inventory_scaled = inventory_optimizer.fluid_inventory_placement_rounding(average_demand=train_generator.average_demands, total_inventory=total_inventory, rescale_inventory=True)
    inventory_list.append(fluid_inventory_scaled)
    print('Obtaining myopic inventory with Greedy')
    myopic_inventory = inventory_optimizer.myopic_greedy_inventory_placement( demand_samples=train_sequences,total_inventory=total_inventory, verbose=False)
    inventory_list.append(myopic_inventory)
    
    inventories = {}
    
    for inventory in inventory_list:
        inventories[inventory.name] = inventory
        print(inventory.name, inventory.initial_inventory)


    # print(offline_inventory.name, offline_inventory.initial_inventory)
    # print(fluid_inventory.name, fluid_inventory.initial_inventory)
    # print(fluid_inventory_scaled.name, fluid_inventory_scaled.initial_inventory)


    ### Construct Priority Lists ###

    programs = MathPrograms(graph)

    # Generate scores
    print('Generating scores for fixed priority list fulfillment')
    scores = {} # dictionary with (penalizer, inventory name) as keys
    for inventory in inventories.values():
        
        fluid, fluid_inventory_constraints = programs.fluid_linear_program_fixed_inventory(train_generator.average_demands, inventory=inventory)
        fluid.optimize()
        scores['fluid', inventory.name] =  {(edge.supply_node_id, edge.demand_node_id): edge.reward - fluid_inventory_constraints[edge.supply_node_id].Pi for edge in graph.edges.values()}
        
        offline, offline_inventory_constraints = programs.offline_linear_program_fixed_inventory(train_sequences, inventory=inventory)
        offline.optimize()

        scores['offline', inventory.name] =  {(edge.supply_node_id, edge.demand_node_id): edge.reward - sum(offline_inventory_constraints[edge.supply_node_id, sample_index].Pi for sample_index in range(n_train_sequences))/n_train_sequences for edge in graph.edges.values()}


    print('Generating priority lists')
    for inventory in inventories.values():

        graph.construct_priority_list(('fluid_with_rejections', inventory.name), scores['fluid',inventory.name], allow_rejections=True)
        # graph.construct_priority_list(('fluid_without_rejections', inventory.name), scores['fluid',inventory.name], allow_rejections=False)
        graph.construct_priority_list(('offline_with_rejections', inventory.name), scores['offline',inventory.name], allow_rejections=True)
        # graph.construct_priority_list(('offline_without_rejections', inventory.name), scores['offline',inventory.name], allow_rejections=False)

    ### Simulate Fulfillment ###
    resolve_times = [1/3, 2/3]
    if demand_model == 'indep':
        fulfillment = Fulfillment(graph, resolve_seed= resolve_seed)
    else:
        fulfillment = Fulfillment(graph, train_generator = train_generator, resolve_seed= resolve_seed)
        
    fixed_list_fulfillment_algorithms = ['myopic', 'fluid_with_rejections',  'offline_with_rejections']
    resolving_fulfillment_algorithms = ['fluid_with_rejections', 'offline_with_rejections']
    
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
                _, collected_rewards, _ = fulfillment.fixed_list_fulfillment(sequence, inventory, list_name, allow_rejections=False)
                rewards[inventory.name, algorithm]+= collected_rewards/n_test_sequences
        
        for algorithm in resolving_fulfillment_algorithms:
            print(f'Simulating {inventory.name} inventory with {algorithm} fulfillment with re-solving')
            inventory = inventories[inventory.name]
            rewards[inventory.name,  algorithm+'_resolving'] = 0
            for sequence in test_sequences:
                list_name = (algorithm, inventory.name)
                if demand_model == 'indep':
                     _, collected_rewards, _ = fulfillment.fulfillment_with_resolving(sequence, inventory, list_name, resolve_times, demand_model, train_generator.means, n_resolve_samples=50, verbose = False)
                else:
                    _, collected_rewards, _ = fulfillment.fulfillment_with_resolving(sequence, inventory, list_name, resolve_times, demand_model, mean_demand, n_resolve_samples=80, verbose = False)
                rewards[inventory.name, algorithm+'_resolving']+= collected_rewards/n_test_sequences

    return rewards

def save_output(results, output_name):
    with open(output_name + '.csv','w',newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter =',')
        spamwriter.writerow(['graph_name','demand_model',
                             'weight_mode',
                             'total_inventory',
                             'inventory_placement',
                             'fulfillment_algorithm',
                             'reward'
                        ])
        for graph_name, demand_model, weight_mode, total_inventory in results:
            result = results[graph_name, demand_model, weight_mode, total_inventory]
            for key in result:
                row = [
                    graph_name,
                    demand_model,
                    weight_mode,
                    total_inventory
                ]
                if key == 'offline_relaxation':
                    row += [key,'',result[key]]
                else:
                    row += [key[0],key[1],result[key]]
                spamwriter.writerow(row)


def main():
    results = {}
    graph_names = ['complete', 'longchain', 'starlike']
    # graph_names=['longchain']
   
    
    demand_models = ['deter','indep', 'correl']
    # demand_models = ['indep', 'deter']
    # demand_models = ['correl']
    
    weight_modes = ['uniform', 'rewardbased']
    # weight_modes = ['uniform']
    
    mean_demand = 60
    # mean_demand = 10
    
    
    total_inventories = [30, 45, 60, 75, 90]
    # total_inventories = [100, 200, 300]
    # total_inventories = [30, 60, 90]

    
    n_test_samples = 1000
    n_train_samples = 1000
    
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
    save_output(results, output_name)

if __name__ == '__main__':
    main()

# start_time = time.time()


# TRAINING_SEED = 10
# TESTING_SEED = 20

# N_TRAINING_SEQUENCES = 500
# N_TESTING_SEQUENCES = 1000

# #complete graph
# supply_nodes, demand_nodes, rewards = ('complete_supplynodes_seed=20.csv', 'complete_demandnodes_seed=20.csv', 'complete_rewards_seed=20.csv')

# # #long chain graph
# # supply_nodes, demand_nodes, rewards = ('longchain_supplynodes_seed=10.csv', 'longchain_demandnodes_seed=10.csv', 'longchain_rewards_seed=10.csv')


# # graph = Graph(mode = 'test_graph_1')
# graph = Graph(supply_nodes, demand_nodes, rewards)
# print(len(graph.supply_nodes))
# print(len(graph.demand_nodes))
# print(len(graph.edges))
# # print(len(graph.destinations['DAB4'].closest_origins))




# mu = 10
# std_dev = 120
# seed = 1

# z = 0.5

# units_per_wh = 10
# total_inventory = units_per_wh * len(graph.supply_nodes)


# # demand_nodes_list
# demand_nodes_list = list(graph.demand_nodes.keys())
# # weights = np.array([ 1.0 for _ in demand_nodes_list])
# weights = np.array([ sum(graph.edges[supply_node_id, demand_node_id].reward for supply_node_id in graph.supply_nodes) for demand_node_id in demand_nodes_list])
# # weights = np.array([ sum(graph.edges[supply_node_id, demand_node_id].reward for supply_node_id in graph.supply_nodes)**2 for demand_node_id in demand_nodes_list])
# # weights = np.array([ sum(graph.edges[supply_node_id, demand_node_id].reward for supply_node_id in graph.supply_nodes)**2 for demand_node_id in demand_nodes_list])

# weights /= weights.sum()




# ### Generate train and test demand ###
# correl_generator_train = CorrelGenerator(mean = mu, demand_nodes=demand_nodes_list, weights= weights, seed = TRAINING_SEED, distribution='exponential', std_dev=std_dev)
# correl_generator_test = CorrelGenerator(mean = mu, demand_nodes=demand_nodes_list, weights= weights, seed = TESTING_SEED, distribution='exponential', std_dev=std_dev)

# print('Generating training sequence')
# training_sequences = [correl_generator_train.generate_sequence() for _ in range(N_TRAINING_SEQUENCES)]
# print('Generating testing sequence')
# testing_sequences = [correl_generator_test.generate_sequence() for _ in range(N_TESTING_SEQUENCES)]





# ### Optimize starting inventories ###


# myopic_scores = {(edge.supply_node_id, edge.demand_node_id): edge.reward for edge in graph.edges.values()}
# graph.construct_priority_list( 'myopic', myopic_scores, allow_rejections=False)


# inventory_optimizer = InventoryOptimizer(graph)
# print('Creating constant inventory')
# constant_inventory = inventory_optimizer.set_inventory_to_n(units_per_wh)
# print('Creating fluid inventory from LP')
# fluid_inventory = inventory_optimizer.fluid_inventory_placement_rounding(average_demand=correl_generator_train.average_demands, total_inventory=total_inventory, rescale_inventory=False)
# print('Creating fluid inventory with scaling from LP')
# fluid_inventory_scaled = inventory_optimizer.fluid_inventory_placement_rounding(average_demand=correl_generator_train.average_demands, total_inventory=total_inventory, rescale_inventory=True)
# print('Creating offline inventory from LP')
# offline_inventory = inventory_optimizer.offline_inventory_placement_rounding(demand_samples=training_sequences, total_inventory=total_inventory)
# print('Creating myopic inventory using Greedy')
# myopic_inventory_greedy = inventory_optimizer.myopic_greedy_inventory_placement( demand_samples=training_sequences,total_inventory=total_inventory, verbose=False)
# # print('Creating fluid inventory using Greedy')
# # fluid_inventory_greedy = inventory_optimizer.fluid_greedy_inventory_placement( correl_generator_train.average_demands,total_inventory=total_inventory, verbose=False)
# # print('Creating offline inventory using Greedy')
# # offline_inventory_greedy = inventory_optimizer.offline_greedy_inventory_placement(training_sequences, total_inventory=total_inventory, verbose = False)
# print(fluid_inventory.name ,fluid_inventory.initial_inventory)
# print(fluid_inventory_scaled.name ,fluid_inventory_scaled.initial_inventory)
# print(offline_inventory.name, offline_inventory.initial_inventory)
# print(myopic_inventory_greedy.name, myopic_inventory_greedy.initial_inventory)




# fulfillment = Fulfillment(graph)

# inventories = {}
# inventories[constant_inventory.name] = constant_inventory
# inventories[fluid_inventory.name] = fluid_inventory
# # inventories[fluid_inventory_greedy.name] = fluid_inventory_greedy
# inventories[fluid_inventory_scaled.name] = fluid_inventory_scaled
# inventories[offline_inventory.name] = offline_inventory
# inventories[myopic_inventory_greedy.name] =myopic_inventory_greedy

# # inventories[offline_inventory_greedy.name] = offline_inventory_greedy


# fulfillment_algorithms = ['myopic', 'fluid_with_rejections', 'fluid_without_rejections', 'offline_with_rejections', 'offline_without_rejections']
# # fulfillment_algorithms = ['myopic',  'fluid_without_rejections', 'offline_without_rejections']
# fulfillment_algorithms = ['myopic', 'fluid_with_rejections',  'offline_with_rejections']

# ### Construct Priority Lists ###

# programs = MathPrograms(graph)

# # Generate scores
# print('Generating scores for fixed priority list fulfillment')
# scores = {} # dictionary with (penalizer, inventory name) as keys
# for inventory in inventories.values():
    
    
#     fluid, fluid_inventory_constraints = programs.fluid_linear_program_fixed_inventory(correl_generator_train.average_demands, inventory=inventory)
#     fluid.optimize()
#     scores['fluid', inventory.name] =  {(edge.supply_node_id, edge.demand_node_id): edge.reward - fluid_inventory_constraints[edge.supply_node_id].Pi for edge in graph.edges.values()}
    
#     offline, offline_inventory_constraints = programs.offline_linear_program_fixed_inventory(training_sequences, inventory=inventory)
#     offline.optimize()

#     scores['offline', inventory.name] =  {(edge.supply_node_id, edge.demand_node_id): edge.reward - sum(offline_inventory_constraints[edge.supply_node_id, sample_index].Pi for sample_index in range(N_TRAINING_SEQUENCES))/N_TRAINING_SEQUENCES for edge in graph.edges.values()}


# print('Generating priority lists')
# for inventory in inventories.values():

#     graph.construct_priority_list(('fluid_with_rejections', inventory.name), scores['fluid',inventory.name], allow_rejections=True)
#     graph.construct_priority_list(('fluid_without_rejections', inventory.name), scores['fluid',inventory.name], allow_rejections=False)
#     graph.construct_priority_list(('offline_with_rejections', inventory.name), scores['offline',inventory.name], allow_rejections=True)
#     graph.construct_priority_list(('offline_without_rejections', inventory.name), scores['offline',inventory.name], allow_rejections=False)





# rewards = {}

# print('Computing offline relaxation for test data')
# offline_relaxation, _ = programs.offline_linear_program_variable_inventory(testing_sequences, total_inventory)
# offline_relaxation.optimize()

# print('Computing rewards for fulfillment algorithms')
# for inventory in inventories.values():
#     offline, _ = programs.offline_linear_program_fixed_inventory(testing_sequences, inventory)
#     offline.optimize()
#     rewards[inventory.name, 'offline'] = offline.ObjVal/N_TESTING_SEQUENCES
#     print(inventory.name, 'offline',rewards[inventory.name, 'offline'])
    
#     for algorithm in fulfillment_algorithms:
#         inventory = inventories[inventory.name]
#         rewards[inventory.name, algorithm] = []
#         for sequence in testing_sequences:
#             if algorithm=='myopic':
#                 list_name = 'myopic'
#             else:
#                 list_name = (algorithm, inventory.name)
#             number_fulfillments, collected_rewards, lost_sales = fulfillment.fixed_list_fulfillment(sequence, inventory, list_name, allow_rejections=False)
#             rewards[inventory.name, algorithm].append(collected_rewards)
    
        


# for inventory in inventories.values():
#     print('')
#     print(f'Inventory: {inventory.name, inventories[inventory.name].initial_inventory}')
#     for algorithm in fulfillment_algorithms:
#         print(f'Algorithm: {algorithm}, reward = {np.mean(rewards[inventory.name, algorithm])/offline_relaxation.ObjVal*100*N_TESTING_SEQUENCES:.3f}')
#     print(f'Algorithm: offline, reward = {rewards[inventory.name, "offline"]/offline_relaxation.ObjVal*100*N_TESTING_SEQUENCES:.3f}')
    
    

# # print('')
# # print('fluid inventory')
# # total_flu = 0
# # for i in graph.supply_nodes:
# #     print(i, fluid_inventory.initial_inventory[i])
# #     total_flu += fluid_inventory.initial_inventory[i]
# # total_off = 0
# # print('')
# # print('offline inventory')
# # for i in graph.supply_nodes:
# #     print(i, offline_inventory.initial_inventory[i])
# #     total_off += offline_inventory.initial_inventory[i]
    
# # print(total_off, total_flu, total_inventory)




# # print('fixed lists method')
# # number_fulfillments_myo, collected_rewards_myo, lost_sales_myo = fulfillment.fixed_list_fulfillment(seq, fluid_inventory, 'myopic', allow_rejections=True)
# # number_fulfillments_flu, collected_rewards_flu, lost_sales_flu = fulfillment.fixed_list_fulfillment(seq, fluid_inventory, 'fluid_without_rejections', allow_rejections=True)
# # number_fulfillments_flurej, collected_rewards_flurej, lost_sales_flurej = fulfillment.fixed_list_fulfillment(seq, fluid_inventory, 'fluid_with_rejections', allow_rejections=True)
# # number_fulfillments_off, collected_rewards_off, lost_sales_off = fulfillment.fixed_list_fulfillment(seq, fluid_inventory, 'offline_without_rejections', allow_rejections=True)
# # number_fulfillments_offrej, collected_rewards_offrej, lost_sales_offrej = fulfillment.fixed_list_fulfillment(seq, fluid_inventory, 'offline_with_rejections', allow_rejections=True)


# # # print(number_fulfillments)
# # print(f'collected reward myopic: {collected_rewards_myo}')
# # print(f'collected reward fluid: {collected_rewards_flu}')
# # print(f'collected reward fluid with rejections: {collected_rewards_flurej}')

# # print(f'collected reward true offline: {offline.ObjVal}')
# # print('')











