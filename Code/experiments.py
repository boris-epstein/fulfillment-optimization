
from Graph import Graph, RandomGraphGenerator
from FulfillmentOptimization import Fulfillment, Inventory, PolicyFulfillment, MultiPriceBalanceFulfillment, LpReSolvingFulfillment, BalanceFulfillment
from ModelBased import IndependentDynamicProgram, ModelEstimator, MarkovianDynamicProgram
from Demand import TemporalIndependenceGenerator, RWGenerator, MarkovianGenerator, Sequence, RandomDistributionGenerator
# from LearningPolicy import DepletionAwarePolicy, train_depletion_policy, extract_reward_matrix, SubscriptableDepletionPolicyWrapper, train_depletion_policy_black_box, train_depletion_nn_policy, NNPolicyWrapper
from ModelFree import ThresholdsFulfillment, EnhancedMPB, DemandTrackingMPB, AdaptiveThresholdsFulfillment
# from NNPolicy import OnlineMatchingPolicy, evaluate_policy_with_params,create_and_train_policy_ng
from typing import Dict, List

from collections import defaultdict
import numpy as np
import csv
from copy import deepcopy
from MathPrograms import MathPrograms

class ExperimentInstance:
    """
    Should include a graph and a distribution
    Distribution can be either indep or Markov
    """
    def __init__(self, graph: Graph, distribution, distribution_type: str):
        self.distribution_type = distribution_type
        self.graph = graph
        self.distribution = distribution
        
 
class PolicyOutput:
    """
    class dedicated to store the performance obtained by a specific policy in all train sets for a specific instance
    """
    
    def __init__(self, results: List[float], train_set_size : int, policy_name: str = ''):
    
        self.results = results
        
        self.train_set_size = train_set_size
        self.policy_name = policy_name


def experiment(
        graph: Graph,
        train_sample_sizes: List[int],
        train_samples: Dict[int, List[List[Sequence]]],
        test_samples: List[Sequence],
        inventory: Inventory,
        optimal_policy = None,
    ):
    
    initial_inventory_tuple = tuple([inventory.initial_inventory[supply_node_id] for supply_node_id in inventory.initial_inventory])
    
    myopic_scores = {(edge.supply_node_id, edge.demand_node_id): edge.reward for edge in graph.edges.values()}
    graph.construct_priority_list( 'myopic', myopic_scores, allow_rejections=False)
    
    T = len(test_samples[0])
    n_samples_per_size = len(train_samples[train_sample_sizes[0]])
    


    indep_dp = IndependentDynamicProgram(graph)
    markov_dp = MarkovianDynamicProgram(graph)
    
    estimator = ModelEstimator(graph)

    sample_based_iid_probabilities = defaultdict(list)
    sample_based_iid_policies = defaultdict(list)
    
    sample_based_indep_probabilities = defaultdict(list)
    sample_based_indep_policies = defaultdict(list)
    
    sample_based_markov_transiton_matrix = defaultdict(list)
    sample_based_markov_policies = defaultdict(list)
    
    thresholds = defaultdict(list)
    
    threhsold_fulfiller = ThresholdsFulfillment(graph)
    enhanced_MPB = EnhancedMPB(graph)
    # dt_MPB = DemandTrackingMPB(graph)
    
    best_thetas = defaultdict(list)
    best_gammas = defaultdict(list)
    
    # best_thetas_dt = defaultdict(list)
    # best_gammas_dt = defaultdict(list)
    # best_betas_dt = defaultdict(list)
    
    adaptive_threshold_fulfiller = AdaptiveThresholdsFulfillment(graph)
    best_adaptive_thresholds = defaultdict(list)
    best_threshold_gammas = defaultdict(list)
    
    for n_train_samples in train_sample_sizes:
        print(f'Processing sample size = {n_train_samples}')
        
        for sample_id in range(n_samples_per_size):
            print(f'sample {sample_id}')
            train_sample = train_samples[n_train_samples][sample_id]
            
            iid_p_estimate = estimator.estimate_iid(train_sample)
            iid_p_dict = {t:iid_p_estimate for t in range(T)}
            
            # print(p_estimate)
            sample_based_iid_probabilities[n_train_samples].append(iid_p_dict)
            sample_based_iid_policies[n_train_samples].append(indep_dp.compute_optimal_policy(inventory, T, iid_p_dict))

            indep_p_estimate = estimator.estimate_independent(train_sample)
            # print(indep_p_estimate[0])
            sample_based_indep_probabilities[n_train_samples].append(indep_p_estimate)
            sample_based_indep_policies[n_train_samples].append( indep_dp.compute_optimal_policy(inventory, T, indep_p_estimate))

            transition_matrix_estimate, _ = estimator.estimate_markovian(train_sample)
            sample_based_markov_transiton_matrix[n_train_samples].append(transition_matrix_estimate)
            sample_based_markov_policies[n_train_samples].append(markov_dp.compute_optimal_policy(inventory, T, transition_matrix_estimate))
    
            print('Training E MPB')
            best_theta, best_gamma = enhanced_MPB.train(inventory, train_sample, budget = 2000)
            best_thetas[n_train_samples].append(best_theta)
            best_gammas[n_train_samples].append(best_gamma)
            
            # print(best_theta)
            # print(best_gamma)
            # print('Training DT MPB')
            # best_theta_dt, best_gamma_dt, best_beta_dt = dt_MPB.train(inventory, train_sample, budget = 1500)
            # best_betas_dt[n_train_samples].append(best_beta_dt)
            # best_thetas_dt[n_train_samples].append(best_theta_dt)
            # best_gammas_dt[n_train_samples].append(best_gamma_dt)
            
            print('Training Static Thresholds')
            best_thresholds = threhsold_fulfiller.train(inventory, train_sample, budget = 2000)
            thresholds[n_train_samples].append(best_thresholds)
            # print(best_thresholds)
            
            print('Training Adaptive Thresholds')
            adaptive_threshold, threshold_gamma = adaptive_threshold_fulfiller.train(inventory, train_sample, budget = 2000)
            best_adaptive_thresholds[n_train_samples].append(adaptive_threshold)
            best_threshold_gammas[n_train_samples].append(threshold_gamma)
            
    
    
    # print(sample_based_iid_probabilities)
    
    dp_fulfiller = PolicyFulfillment(graph)
    balance_fulfiller = MultiPriceBalanceFulfillment(graph)
    greedy_fulfiller = Fulfillment(graph)
    
    balance_reward = 0
    opt_reward = 0
    myopic_reward = 0
    iid_data_driven_reward = {}
    indep_data_driven_reward = {}
    markov_data_driven_reward = {}
   
    threshold_rewards = {}
    adaptive_threshold_rewards = {}
    
    re_solving_rewards = {}
    filtered_re_solving_rewards = {}
    depletion_reward = {}
    enhanced_mpb_rewards = {}
    nn_rewards = {}
    
    
    
    
    for n_train_samples in train_sample_sizes:
        iid_data_driven_reward[n_train_samples] = [0 for _ in range(n_samples_per_size)]
        indep_data_driven_reward[n_train_samples] = [0 for _ in range(n_samples_per_size)]
        # off_rejections_reward[n_train_samples] = [0 for _ in range(n_samples_per_size)]
        # off_no_rejections_reward[n_train_samples] = [0 for _ in range(n_samples_per_size)]
        markov_data_driven_reward[n_train_samples] = [0 for _ in range(n_samples_per_size)]
        threshold_rewards[n_train_samples] = [0 for _ in range(n_samples_per_size)]
        adaptive_threshold_rewards[n_train_samples] = [0 for _ in range(n_samples_per_size)]
        # nn_rewards[n_train_samples] = [0 for _ in range(n_samples_per_size)]
        re_solving_rewards[n_train_samples] = [0 for _ in range(n_samples_per_size)]
        filtered_re_solving_rewards[n_train_samples] = [0 for _ in range(n_samples_per_size)]
        enhanced_mpb_rewards[n_train_samples] = [0 for _ in range(n_samples_per_size)]
        # dt_mpb_rewards[n_train_samples] = [0 for _ in range(n_samples_per_size)]
        
   
        
        
    print('Simulating policies out of sample')
    
    n_test_samples = len(test_samples)
    j = 0
    for sequence in test_samples:
        if j%100==0:
            print(f'Test sample {j}')
        j+=1
        if optimal_policy is not None:
            _, collected_reward, _ = dp_fulfiller.fulfill(sequence, inventory, optimal_policy.optimal_action, verbose = False)
            opt_reward += collected_reward/n_test_samples
        

        _, balance_sample, _ = balance_fulfiller.fulfill(sequence, inventory,verbose=False)
        balance_reward+= balance_sample/n_test_samples
        
        _, myo_sample, _ = greedy_fulfiller.fixed_list_fulfillment(sequence, inventory, 'myopic')
        myopic_reward += myo_sample/n_test_samples
        
        for n_train_samples in train_sample_sizes:
            for i in range(n_samples_per_size):
                _, iid_reward, _ = dp_fulfiller.fulfill(sequence, inventory, sample_based_iid_policies[n_train_samples][i].optimal_action)
                iid_data_driven_reward[n_train_samples][i] += iid_reward/n_test_samples
                
                _, indep_reward, _ = dp_fulfiller.fulfill(sequence, inventory, sample_based_indep_policies[n_train_samples][i].optimal_action)
                indep_data_driven_reward[n_train_samples][i] += indep_reward/n_test_samples
        
        
                _, markov_reward, _ = dp_fulfiller.fulfill(sequence, inventory, sample_based_markov_policies[n_train_samples][i].optimal_action)
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
                # train_sample = train_samples[n_train_samples][i]
                # _, re_solving_reward, _ = re_solving_fulfiller.fulfill(sequence, initial_inventory, train_sample, re_solving_epochs, verbose=False)
                # re_solving_rewards[n_train_samples][i] += re_solving_reward/n_test_samples
                
                # _, filtered_re_solving_reward, _ = re_solving_fulfiller.fulfill(sequence, initial_inventory, train_sample, re_solving_epochs, filter_samples = True, verbose=False)
                # filtered_re_solving_rewards[n_train_samples][i] += filtered_re_solving_reward/n_test_samples
                
                threshold_reward = threhsold_fulfiller.fulfill(sequence, inventory,thresholds[n_train_samples][i].value)
                threshold_rewards[n_train_samples][i] += threshold_reward/n_test_samples
                
                adaptive_threshold_reward = adaptive_threshold_fulfiller.fulfill(sequence, inventory, best_adaptive_thresholds[n_train_samples][i], best_threshold_gammas[n_train_samples][i])
                adaptive_threshold_rewards[n_train_samples][i] += adaptive_threshold_reward/n_test_samples
                
                enhanced_mpb_reward = enhanced_MPB.fulfill(sequence, inventory, best_thetas[n_train_samples][i], best_gammas[n_train_samples][i])
                enhanced_mpb_rewards[n_train_samples][i] += enhanced_mpb_reward/n_test_samples
                
                
                # dt_mpb_reward = dt_MPB.fulfill(sequence, inventory, best_thetas_dt[n_train_samples][i], best_gammas_dt[n_train_samples][i],best_betas_dt[n_train_samples][i])
                # dt_mpb_rewards[n_train_samples][i] += dt_mpb_reward/n_test_samples
                
    # print(f'Reward with optimal policy: {opt_reward}')
    # print(f'Reward with balance: {balance_reward}')
    # print(f'Reward with myopic policy: {myopic_reward}')
    # print('Data-driven DP:')
    # for n_train_samples in train_sample_sizes:
    #     print(f'IID train_samples: {n_train_samples}, average reward = {np.mean(iid_data_driven_reward[n_train_samples])}')
    #     print(f'Indep train_samples: {n_train_samples}, average reward = {np.mean(indep_data_driven_reward[n_train_samples])}')
    #     print('')
    
    results = {}
    if optimal_policy is not None:
        results['optimal'] = PolicyOutput([opt_reward],0, 'optimal')
    results['myopic'] = PolicyOutput([myopic_reward], 0, 'myopic')
    results['balance'] = PolicyOutput([balance_reward], 0, 'balance')
    
    for n_train_samples in train_sample_sizes:
        results['iid_dp', n_train_samples] = PolicyOutput(iid_data_driven_reward[n_train_samples],n_train_samples, 'iid_dp')
        results['indep_dp', n_train_samples] = PolicyOutput(indep_data_driven_reward[n_train_samples],n_train_samples, 'indep_dp')
        results['markov_dp', n_train_samples] = PolicyOutput(markov_data_driven_reward[n_train_samples],n_train_samples, 'markov_dp')
        
    if optimal_policy is not None:  
        print(f'Optimal reward: {opt_reward}')
    print(f'Myopic reward: {myopic_reward}')
    print(f'Balance reward: {balance_reward}')
    print('')  
    print('Data-driven rewards:')
    for n_train_samples in train_sample_sizes:
        print(f'num samples = {n_train_samples}')
        print(f'iid dp: {np.mean(iid_data_driven_reward[n_train_samples])}')
        print(f'indep_dp: {np.mean(indep_data_driven_reward[n_train_samples])}')
        print(f'Markov_dp: {np.mean(markov_data_driven_reward[n_train_samples])}')
        print(f'Static Thresholds: {np.mean(threshold_rewards[n_train_samples])}')
        print(f'Enhanced MPB: {np.mean(enhanced_mpb_rewards[n_train_samples])}')
        # print(f'Demand-Tracking MPB: {np.mean(dt_mpb_rewards[n_train_samples])}')
        print(f'Adaptive Thresholds: {np.mean(adaptive_threshold_rewards[n_train_samples])}')

        print('')

    return results

def main(demand_model):
    n_supply_nodes = 3
    n_demand_nodes = 15
    
    train_sample_sizes = [1, 10, 100, 1000]#, 100, 500]#, 500, 1000, 5000]
    n_samples_per_size = 10
    
    inventory = Inventory({0:2, 1:2, 2:2}, name = 'test')
    
    n_test_samples = 1000
    
    T = 12
    
    vertex_values = {}
    vertex_values[0] = [0.1, 0.9]
    vertex_values[1] = [0.4, 0.6]
    vertex_values[2] = [0.7,0.91]
    
    vertex_values[3] = [0.59,0.89]
    
    graph_generator_seed = 0
    distribution_generator_seed = 1
    train_generator_seed = 2
    test_generator_seed = 3
    
    num_instances = 3
    
    
    graph_generator = RandomGraphGenerator(seed = graph_generator_seed)
    dist_generator = RandomDistributionGenerator(seed = distribution_generator_seed)
    
    instances: Dict[int, ExperimentInstance] = {}
    optimal_policies = {}
    
    for i in range(num_instances):
    
        graph = graph_generator.two_valued_vertex_graph(n_supply_nodes, n_demand_nodes, vertex_values)
        myopic_scores = {(edge.supply_node_id, edge.demand_node_id): edge.reward for edge in graph.edges.values()}
        
        graph.construct_priority_list( 'myopic', myopic_scores, allow_rejections=False)
        
        if demand_model == 'indep':
            p = dist_generator.generate_indep(n_demand_nodes, T)
            distribution = p
            indep_dp = IndependentDynamicProgram(graph)
            optimal_policy = indep_dp.compute_optimal_policy(inventory, T, p)
            optimal_policies[i] = optimal_policy
        
        elif demand_model == 'markov':
            initial_distribution, transition_matrix = dist_generator.generate_markov(n_demand_nodes, 2)
            distribution = (initial_distribution, transition_matrix)
            markov_dp = MarkovianDynamicProgram(graph)
            optimal_policy = markov_dp.compute_optimal_policy(inventory, T, transition_matrix)
            optimal_policies[i] = optimal_policy
        
        
        elif demand_model =='rw':
            step_size = 3
            distribution = step_size
            # train_generator = RWGenerator(T, demand_node_list, seed = train_generator_seed, step_size = step_size)
            # test_generator = RWGenerator(T, demand_node_list, seed = test_generator_seed, step_size = step_size)
        
        instances[i] = ExperimentInstance(graph, deepcopy(distribution), demand_model)
        
    
    
    
    
    
    
    
    demand_node_list = list(graph.demand_nodes.keys())
    

    train_samples = {}
    test_samples = {}
    
    results = {}
    
    for i in range(num_instances):
        print(f'Instance {i}')
        
        instance = instances[i]
        
        graph = instance.graph
        print(graph.edges)
        demand_node_list = [demand_node_id for demand_node_id in graph.demand_nodes]
        
        if demand_model =='markov':
            
            initial_distribution, transition_matrix = instance.distribution
            train_generator = MarkovianGenerator(T, demand_node_list,transition_matrix,initial_distribution, seed = train_generator_seed)
            test_generator = MarkovianGenerator(T, demand_node_list,transition_matrix,initial_distribution, seed = test_generator_seed)
        
        if demand_model == 'indep':
            p = instance.distribution
            train_generator = TemporalIndependenceGenerator(demand_node_list, p, seed = train_generator_seed)
            test_generator = TemporalIndependenceGenerator(demand_node_list, p, seed = test_generator_seed)
            
        if demand_model =='rw':
            train_generator = RWGenerator(T, demand_node_list, seed = train_generator_seed, step_size= step_size)
            test_generator = RWGenerator(T, demand_node_list, seed = test_generator_seed, step_size= step_size)
        
        train_samples[i] = defaultdict(list)
        for n_train_samples in train_sample_sizes:
            for _ in range(n_samples_per_size):
                train_samples[i][n_train_samples].append( [train_generator.generate_sequence() for _ in range(n_train_samples)])
                
        test_samples[i] = [test_generator.generate_sequence() for _ in range(n_test_samples)]

        if demand_model == 'indep' or demand_model == 'markov':
            results[i] = experiment(graph,train_sample_sizes, train_samples[i], test_samples[i], inventory, optimal_policies[i])
        else:
            results[i] = experiment(graph,train_sample_sizes, train_samples[i], test_samples[i], inventory)

    print(results)


if __name__ == '__main__':

    # main('markov')
    main('markov')