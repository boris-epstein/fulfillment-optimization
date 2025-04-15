
from Graph import Graph
from FulfillmentOptimization import Fulfillment, Inventory, PolicyFulfillment, BalanceFulfillment, LpReSolvingFulfillment
from ModelBased import IndependentDynamicProgram, ModelEstimator, MarkovianDynamicProgram
from Demand import TemporalIndependenceGenerator, RWGenerator, MarkovianGenerator
# from LearningPolicy import DepletionAwarePolicy, train_depletion_policy, extract_reward_matrix, SubscriptableDepletionPolicyWrapper, train_depletion_policy_black_box, train_depletion_nn_policy, NNPolicyWrapper
# from ModelFree import threshold_policy_brute_force_search, SimplestGraphThresholdFulfillment
# from NNPolicy import OnlineMatchingPolicy, evaluate_policy_with_params,create_and_train_policy_ng


from collections import defaultdict
import numpy as np
import csv

from MathPrograms import MathPrograms
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
    
    return



def main(demand_model):
    
    low_reward = '05'
    # demand_model = 'indep' # Can be indep, markov, rw
    print(f'demand_model = {demand_model}')
    print(f'low_reward = {low_reward}')
    train_sample_sizes = [10,50,100, 500, 1000, 5000]#, 500, 1000, 5000]
    n_samples_per_size = 20
    print(f'Sample sizes tried: {train_sample_sizes}')
    print(f'Number of samples per sample size: {n_samples_per_size}')
    
    
    
    # supply_nodes, demand_nodes, rewards = (f'simplest_graph_1{low_reward}_supplynodes.csv', f'simplest_graph_1{low_reward}_demandnodes.csv', f'simplest_graph_1{low_reward}_rewards.csv')


    supply_nodes, demand_nodes, rewards = (f'three_node_graph_supplynodes.csv', f'three_node_graph_demandnodes.csv', f'three_node_graph_rewards.csv')
    ### Graph Reading
    print('Reading graph')
    graph = Graph()
    graph.read_from_files(supply_nodes, demand_nodes, rewards)
    myopic_scores = {(edge.supply_node_id, edge.demand_node_id): edge.reward for edge in graph.edges.values()}
    graph.construct_priority_list( 'myopic', myopic_scores, allow_rejections=False)
    
    # reward_matrix = extract_reward_matrix(graph)
    
    # print(reward_matrix)
    # print(graph.edges)
    
    T = 15
    
    
    
    
    
    
    #### THIS IS FOR SIMPLEST GRAPH ####
    initial_inventory = Inventory({0:5, 1:5})
    
    initial_inventory_tuple = (5,5)
    
    iid_p = [1/2, 1/4, 1/4]
    p = {t: iid_p for t in range(T)}
    
    p = {}
    # p[0] = [0, 1, 0]
    # p[1] = [0, 1, 0]
    # p[2] = [0, 1, 0]
    # p[3] = [1, 0, 0]
    # p[4] = [1, 0, 0]
    # p[5] = [1,0,0]
    # p[6] = [1,0,0]
    # p[7] = [1,0,0]
    # p[8] = [0,0,1]
    # p[9] = [0,0,1]
    
    p[0] = [1/12, 5/6, 1/12]
    p[1] = [1/6, 2/3, 1/6]
    p[2] = [1/12, 5/6, 1/12]
    p[3] = [1/6, 2/3, 1/6]
    p[4] = [3/4, 1/8, 1/8]
    p[5] = [2/3,1/6,1/6]
    p[6] = [3/4,1/8,1/8]
    p[7] = [1,0,0]
    p[8] = [0,0,1]
    p[9] = [0,0,1]
    
    # For T=5
    # p[0] = [1/12, 5/6, 1/12]
    # p[1] = [1/6, 2/3, 1/6]
    # p[2] = [1/6, 2/3, 1/6]
    # p[3] = [3/4, 1/8, 1/8]
    # p[4] = [2/3,1/6,1/6]
    
    # p[0] = [0, 1, 0]
    # p[1] = [0, 1, 0]
    # p[2] = [0, 1, 0]
    # p[3] = [0, 1, 0]
    # p[4] = [0, 1, 0]
    # p[5] = [1/4, 0, 3/4]
    # p[6] = [1/4, 0, 3/4]
    # p[7] = [1/4, 0, 3/4]
    # p[8] = [1/4, 0, 3/4]
    # p[9] = [3/4, 0, 1/4]
    
    initial_distribution = [1/4, 1/2, 1/4]
    
    transition_matrix = np.array(
        [[0.8,0.05,0.15],
         [0.1,0.7,0.2],
         [0.1,0.1,0.8]
         ]
    )
    
    # initial_distribution = [0.9, 0.05, 0.05]
    
    # transition_matrix = np.array(
    #     [[0.01,0.98,0.01],
    #      [0.2,0.6,0.2],
    #      [0.1,0.2,0.7]
    #      ]
    # )
    
    
    
     #### ADD ONE MORE NODE ####
     
     
    initial_inventory = Inventory({0:5, 1:5, 2:5})
    
    initial_inventory_tuple = (5,5,5)
    
     
    p = {}
    
    p[0] = [1/3, 1/3, 1/6, 1/6]
    p[1] = [1/3, 1/3, 1/6, 1/6]
    p[2] = [1/3, 1/3, 1/6, 1/6]
    p[3] = [1/3, 1/3, 1/6, 1/6]
    p[4] = [1/3, 1/3, 1/6, 1/6]
    p[5] = [5/12, 5/12, 1/12, 1/12]
    p[6] = [5/12, 5/12, 1/12, 1/12]
    p[7] = [5/12, 5/12, 1/12, 1/12]
    p[8] = [5/12, 5/12, 1/12, 1/12]
    p[9] = [5/12, 5/12, 1/12, 1/12]
    p[10] = [0, 0, 1/2, 1/2]
    p[11] = [0, 0, 1/2, 1/2]
    p[12] = [0, 0, 1/2, 1/2]
    p[13] = [0, 0, 1/2, 1/2]
    p[14] = [0, 0, 1/2, 1/2]
   
    
    initial_distribution = [1/4, 1/4, 1/4, 1/4]
    
    transition_matrix = np.array(
        [[0.85,0.05,0.05,0.05],
         [0.05,0.85,0.05,0.05],
         [0.05,0.05,0.85,0.05],
         [0.05,0.05,0.05,0.85],
         ]
    )
    
    indep_dp = IndependentDynamicProgram(graph)
    markov_dp = MarkovianDynamicProgram(graph)
    
    estimator = ModelEstimator(graph)
    
    if demand_model == 'indep':
        optimal_policy = indep_dp.compute_optimal_policy(initial_inventory, T, p)
        expected_value = 0
        for demand_node_id in graph.demand_nodes:
            expected_value+= p[0][demand_node_id] * optimal_policy.value_function[initial_inventory_tuple, 0, demand_node_id]
        
    if demand_model == 'markov':
        optimal_policy = markov_dp.compute_optimal_policy(initial_inventory,T, transition_matrix)
        expected_value = 0
        for demand_node_id in graph.demand_nodes:
            expected_value+= initial_distribution[demand_node_id] * optimal_policy.value_function[initial_inventory_tuple, 0, demand_node_id]
        
    
    
    sample_based_iid_probabilities = defaultdict(list)
    sample_based_iid_policies = defaultdict(list)
    
    sample_based_indep_probabilities = defaultdict(list)
    sample_based_indep_policies = defaultdict(list)
    
    sample_based_markov_transiton_matrix = defaultdict(list)
    sample_based_markov_policies = defaultdict(list)
    
    sample_based_threshold_grids = defaultdict(list)
    learned_policies = defaultdict(list)
    
    
    # programs = MathPrograms(graph)
    # scores = {} # dictionary with (penalizer, inventory name) as keys
    
    demand_node_list = sorted([demand_node_id for demand_node_id in graph.demand_nodes])
    
    if demand_model == 'indep':
        train_generator = TemporalIndependenceGenerator(demand_node_list,p,seed=12)
    if demand_model == 'rw':
        train_generator = RWGenerator(mean = T, demand_nodes=demand_node_list,seed = 12,distribution='deterministic',step_size = 3)
    if demand_model == 'markov':
        train_generator = MarkovianGenerator(T, demand_node_list, transition_matrix, initial_distribution, seed = 12)
    
    
    train_samples = defaultdict(list)
    for n_train_samples in train_sample_sizes:
        print(f'Processing sample size = {n_train_samples}')
        for sample_id in range(n_samples_per_size):
            
            
            train_sample = [train_generator.generate_sequence() for _ in range(n_train_samples)]
            
            train_samples[n_train_samples].append(train_sample)
            
            iid_p_estimate = estimator.estimate_iid(train_sample)
            iid_p_dict = {t:iid_p_estimate for t in range(T)}
            # print(p_estimate)
            sample_based_iid_probabilities[n_train_samples].append(iid_p_dict)
            sample_based_iid_policies[n_train_samples].append(indep_dp.compute_optimal_policy(initial_inventory,T, iid_p_dict))

            indep_p_estimate = estimator.estimate_independent(train_sample)
            sample_based_indep_probabilities[n_train_samples].append(indep_p_estimate)
            sample_based_indep_policies[n_train_samples].append( indep_dp.compute_optimal_policy(initial_inventory, T, indep_p_estimate))

            transition_matrix_estimate, _ = estimator.estimate_markovian(train_sample)
            sample_based_markov_transiton_matrix[n_train_samples].append(transition_matrix_estimate)
            sample_based_markov_policies[n_train_samples].append(markov_dp.compute_optimal_policy(initial_inventory, T, transition_matrix_estimate))
            
            # sample_based_threshold_grids[n_train_samples].append(threshold_policy_brute_force_search(graph, train_sample, initial_inventory, T))
            
            
            
            # learned_policy = create_and_train_policy_ng(graph, initial_inventory, train_sample)
            # learned_policies[n_train_samples].append(learned_policy)
            
            
            # learned_policy = train_depletion_policy_black_box(graph, train_sample, reward_matrix, T=T)
            # learned_policy =trained_policy = train_depletion_nn_policy(graph, train_sample, reward_matrix, num_epochs=151, lr=0.001, hidden_dim=16, T=10)
            # learned_policies[n_train_samples].append(SubscriptableDepletionPolicyWrapper(learned_policy, graph, reward_matrix, initial_inventory_tuple))
            # learned_policies[n_train_samples].append(NNPolicyWrapper(learned_policy, graph, initial_inventory_tuple,reward_matrix, T))
            # offline, offline_inventory_constraints = programs.offline_linear_program_fixed_inventory(train_sample, inventory=initial_inventory)
            # offline.optimize()

            # scores['offline', n_train_samples, sample_id] =  {(edge.supply_node_id, edge.demand_node_id): edge.reward - sum(offline_inventory_constraints[edge.supply_node_id, sample_index].Pi for sample_index in range(n_train_samples))/n_train_samples for edge in graph.edges.values()}
            # print(n_train_samples, scores['offline', n_train_samples, sample_id])
            # graph.construct_priority_list(('offline_with_rejections', n_train_samples, sample_id), scores['offline',n_train_samples, sample_id], allow_rejections=True)
            # graph.construct_priority_list(('offline_without_rejections', n_train_samples, sample_id), scores['offline',n_train_samples, sample_id], allow_rejections=False)
            
            
    
    
        
   
    # print(expected_value)
    if demand_model == 'indep':
        test_generator = TemporalIndependenceGenerator(demand_node_list,p,seed=111)
    if demand_model == 'rw':
        test_generator = RWGenerator(mean = T, demand_nodes=demand_node_list,seed = 111,distribution='deterministic',step_size = 3)
    if demand_model == 'markov':
        test_generator = MarkovianGenerator(T, demand_node_list, transition_matrix, initial_distribution, seed= 111)
    
    n_test_samples = 201
    dp_fulfiller = PolicyFulfillment(graph)
    balance_fulfiller = BalanceFulfillment(graph)
    greedy_fulfiller = Fulfillment(graph)
    # threshold_fulfiller = SimplestGraphThresholdFulfillment(graph)
    
    re_solving_fulfiller = LpReSolvingFulfillment(graph)
    re_solving_epochs = list(range(14))
    
    test_samples = [test_generator.generate_sequence() for _ in range(n_test_samples)]
    
    balance_reward =0
    opt_reward = 0
    myopic_reward = 0
    iid_data_driven_reward = {}
    indep_data_driven_reward = {}
    markov_data_driven_reward = {}
    off_rejections_reward= {}
    off_no_rejections_reward = {}
    threshold_rewards = {}
    re_solving_rewards = {}
    filtered_re_solving_rewards = {}
    depletion_reward = {}
    
    nn_rewards = {}
    
    for n_train_samples in train_sample_sizes:
        iid_data_driven_reward[n_train_samples] = [0 for _ in range(n_samples_per_size)]
        indep_data_driven_reward[n_train_samples] = [0 for _ in range(n_samples_per_size)]
        # off_rejections_reward[n_train_samples] = [0 for _ in range(n_samples_per_size)]
        # off_no_rejections_reward[n_train_samples] = [0 for _ in range(n_samples_per_size)]
        markov_data_driven_reward[n_train_samples] = [0 for _ in range(n_samples_per_size)]
        # threshold_rewards[n_train_samples] = [0 for _ in range(n_samples_per_size)]
        # nn_rewards[n_train_samples] = [0 for _ in range(n_samples_per_size)]
        re_solving_rewards[n_train_samples] = [0 for _ in range(n_samples_per_size)]
        filtered_re_solving_rewards[n_train_samples] = [0 for _ in range(n_samples_per_size)]
        
        
    print('Simulating policies out of sample')
    
    j = 0
    for sequence in test_samples:
        if True:
            print(f'Test sample {j}')
        j+=1
        if demand_model == 'indep':
            _, collected_reward, _ = dp_fulfiller.fulfill(sequence, initial_inventory, optimal_policy.optimal_action)
            opt_reward += collected_reward/n_test_samples
        
        if demand_model =='markov':
            _, collected_reward, _ = dp_fulfiller.fulfill(sequence, initial_inventory, optimal_policy.optimal_action)
            opt_reward += collected_reward/n_test_samples

        _, balance_sample, _ = balance_fulfiller.fulfill(sequence, initial_inventory,verbose=False)
        balance_reward+= balance_sample/n_test_samples
        
        _, myo_sample, _ = greedy_fulfiller.fixed_list_fulfillment(sequence, initial_inventory, 'myopic')
        myopic_reward += myo_sample/n_test_samples
        
        for n_train_samples in train_sample_sizes:
            for i in range(n_samples_per_size):
                _, iid_reward, _ = dp_fulfiller.fulfill(sequence, initial_inventory, sample_based_iid_policies[n_train_samples][i].optimal_action)
                iid_data_driven_reward[n_train_samples][i] += iid_reward/n_test_samples
                
                _, indep_reward, _ = dp_fulfiller.fulfill(sequence, initial_inventory, sample_based_indep_policies[n_train_samples][i].optimal_action)
                indep_data_driven_reward[n_train_samples][i] += indep_reward/n_test_samples
        
        
                _, markov_reward, _ = dp_fulfiller.fulfill(sequence, initial_inventory, sample_based_markov_policies[n_train_samples][i].optimal_action)
                markov_data_driven_reward[n_train_samples][i] += markov_reward/n_test_samples
                # _, off_rejections_sample, _ = greedy_fulfiller.fixed_list_fulfillment(sequence, initial_inventory, ('offline_with_rejections', n_train_samples, i),allow_rejections=True)
                # off_rejections_reward[n_train_samples][i] += off_rejections_sample/n_test_samples
                
                # _, off_no_rejection_sample, _ = greedy_fulfiller.fixed_list_fulfillment(sequence, initial_inventory, ('offline_without_rejections', n_train_samples, i),allow_rejections=False)
                # off_no_rejections_reward[n_train_samples][i] += off_no_rejection_sample/n_test_samples

                # _ , dep_reward, _ = dp_fulfiller.fulfill(sequence, initial_inventory, learned_policies[n_train_samples][i], verbose = False)
                
                # depletion_reward[n_train_samples][i] += dep_reward/n_test_samples
                
                # _, threshold_reward, _ = threshold_fulfiller.fulfill(sequence, initial_inventory, sample_based_threshold_grids[n_train_samples][i])
                # threshold_rewards[n_train_samples][i] += threshold_reward/n_test_samples
                
                # _ , nn_reward, _ = dp_fulfiller.fulfill(sequence, initial_inventory, learned_policies[n_train_samples][i], verbose = False)
                # nn_rewards[n_train_samples][i] += nn_reward/n_test_samples
                train_sample = train_samples[n_train_samples][i]
                _, re_solving_reward, _ = re_solving_fulfiller.fulfill(sequence, initial_inventory, train_sample, re_solving_epochs, verbose=False)
                re_solving_rewards[n_train_samples][i] += re_solving_reward/n_test_samples
                
                _, filtered_re_solving_reward, _ = re_solving_fulfiller.fulfill(sequence, initial_inventory, train_sample, re_solving_epochs, filter_samples = True, verbose=False)
                filtered_re_solving_rewards[n_train_samples][i] += filtered_re_solving_reward/n_test_samples
                
                
    # print(f'Reward with optimal policy: {opt_reward}')
    # print(f'Reward with balance: {balance_reward}')
    # print(f'Reward with myopic policy: {myopic_reward}')
    # print('Data-driven DP:')
    # for n_train_samples in train_sample_sizes:
    #     print(f'IID train_samples: {n_train_samples}, average reward = {np.mean(iid_data_driven_reward[n_train_samples])}')
    #     print(f'Indep train_samples: {n_train_samples}, average reward = {np.mean(indep_data_driven_reward[n_train_samples])}')
    #     print('')
    
    
    ## Save output
    output_name = f'three_node_{demand_model}_T{T}_filter'
    with open(output_name + '.csv','w',newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter =',')
        spamwriter.writerow(['policy_name', 'number_samples', 'average', 'standard_deviation'
                        ])
        
       
            
        for n_train_samples in train_sample_sizes:
            spamwriter.writerow(['myopic', n_train_samples, myopic_reward, 0
                        ])
        for n_train_samples in train_sample_sizes:
            spamwriter.writerow(['balance', n_train_samples, balance_reward, 0
                        ])
            
        for n_train_samples in train_sample_sizes:
            spamwriter.writerow(['iid_dp', n_train_samples, np.mean(iid_data_driven_reward[n_train_samples]), np.std(iid_data_driven_reward[n_train_samples])
                        ])
        
        for n_train_samples in train_sample_sizes:
            spamwriter.writerow(['indep_dp', n_train_samples, np.mean(indep_data_driven_reward[n_train_samples]), np.std(indep_data_driven_reward[n_train_samples])
                        ])
        
        # for n_train_samples in train_sample_sizes:
        #     spamwriter.writerow(['off_sp_rejections', n_train_samples, np.mean(off_rejections_reward[n_train_samples]), np.std(off_rejections_reward[n_train_samples])
        #                 ])
            
        # for n_train_samples in train_sample_sizes:
        #     spamwriter.writerow(['off_sp_no_rejections', n_train_samples, np.mean(off_no_rejections_reward[n_train_samples]), np.std(off_no_rejections_reward[n_train_samples])
        #                 ])
        
        # for n_train_samples in train_sample_sizes:
        #     spamwriter.writerow(['depletion_learned', n_train_samples, np.mean(depletion_reward[n_train_samples]), np.std(depletion_reward[n_train_samples])
        #                 ])
        
        for n_train_samples in train_sample_sizes:
            spamwriter.writerow(['markov_dp', n_train_samples, np.mean(markov_data_driven_reward[n_train_samples]), np.std(markov_data_driven_reward[n_train_samples])
                        ])
            
        # for n_train_samples in train_sample_sizes:
        #     spamwriter.writerow(['thresholds', n_train_samples, np.mean(threshold_rewards[n_train_samples]), np.std(threshold_rewards[n_train_samples])
        #                 ])


        # for n_train_samples in train_sample_sizes:
        #     spamwriter.writerow(['neural_net', n_train_samples, np.mean(nn_rewards[n_train_samples]), np.std(nn_rewards[n_train_samples])
        #                 ])
        
        for n_train_samples in train_sample_sizes:
            spamwriter.writerow(['lp_re_solving', n_train_samples, np.mean(re_solving_rewards[n_train_samples]), np.std(re_solving_rewards[n_train_samples])
                        ])
            
        for n_train_samples in train_sample_sizes:
            spamwriter.writerow(['filtered_lp_re_solving', n_train_samples, np.mean(filtered_re_solving_rewards[n_train_samples]), np.std(filtered_re_solving_rewards[n_train_samples])
                        ])
            
            
        if demand_model == 'indep' or demand_model =='markov':
            for n_train_samples in train_sample_sizes:
                spamwriter.writerow(['optimal_dp', n_train_samples, opt_reward, 0
                        ])
    
if __name__ == '__main__':

    # main('markov')
    main('indep')
    main('rw')