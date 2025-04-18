import time

from Graph import Graph, RandomGraphGenerator
from FulfillmentOptimization import Fulfillment, Inventory, PolicyFulfillment, MultiPriceBalanceFulfillment, LpReSolvingFulfillment, BalanceFulfillment
from ModelBased import IndependentDynamicProgram, ModelEstimator, MarkovianDynamicProgram
from Demand import TemporalIndependenceGenerator, RWGenerator, MarkovianGenerator, Sequence, RandomDistributionGenerator
# from LearningPolicy import DepletionAwarePolicy, train_depletion_policy, extract_reward_matrix, SubscriptableDepletionPolicyWrapper, train_depletion_policy_black_box, train_depletion_nn_policy, NNPolicyWrapper
from ModelFree import ThresholdsFulfillment, TimeSupplyEnhancedMPB, DemandTrackingMPB, AdaptiveThresholdsFulfillment
# from NNPolicy import OnlineMatchingPolicy, evaluate_policy_with_params,create_and_train_policy_ng
from typing import Any, Dict, List

from collections import defaultdict
import numpy as np
import csv
from copy import deepcopy
from MathPrograms import MathPrograms

class Instance:
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
    
    def __init__(self, rewards: List[float], train_times, test_times, train_set_size : int, policy_name: str = ''):
    
        self.rewards = rewards
        self.train_set_size = train_set_size
        self.policy_name = policy_name
        self.train_times = train_times
        self.test_times = test_times
        self.policy_name = policy_name

class OutputWriter:
    
    def __init__(self,
                data_agnostic_policies:List[str],
                model_based_dynamic_programs: List[str],
                model_free_parametrized_policies: List[str],
                num_instances: int,
                train_sample_sizes: List[int],
                num_samples_per_size : int,
                include_optimal: bool,
                results: Dict[int, Dict[Any, PolicyOutput]]
                ):
        
        self.data_agnostic_policies = data_agnostic_policies
        self.model_based_dynamic_programs = model_based_dynamic_programs
        self.num_instances = num_instances
        self.model_free_parametrized_policies = model_free_parametrized_policies
        self.train_sample_sizes = train_sample_sizes
        self.results = results
        
        self.num_samples_per_size =num_samples_per_size
        
        self.include_optimal = include_optimal
        
    def write_output(self, path):
        
        with open(path, 'w', newline='') as csv_file:
            spamwriter = csv.writer(csv_file, delimiter =',')
            spamwriter.writerow(['instance_id', 'policy_name', 'sample_size', 'sample_id', 'average_test_reward', 'training_time', 'average_testing_time'])
        
        
            
        
            for instance_id in range(self.num_instances):
                
                if self.include_optimal:
                    spamwriter.writerow([instance_id, 'optimal', 0, 0, self.results[instance_id]['optimal'].rewards, self.results[instance_id]['optimal'].train_times, self.results[instance_id]['optimal'].test_times])
                
                for policy in self.data_agnostic_policies:
                    spamwriter.writerow([instance_id, policy, 0, 0, self.results[instance_id][policy].rewards, self.results[instance_id][policy].train_times, self.results[instance_id][policy].test_times])
                
                for policy in self.model_based_dynamic_programs + self.model_free_parametrized_policies:
                
                    for num_samples in self.train_sample_sizes:
                        for sample_id in range(self.num_samples_per_size):
                            
                                spamwriter.writerow([instance_id, policy, num_samples, sample_id, self.results[instance_id][policy][num_samples].rewards[sample_id], self.results[instance_id][policy][num_samples].train_times[sample_id], self.results[instance_id][policy][num_samples].test_times[sample_id]])
                



  
class Experiment:
    
    def __init__(self,
        graph: Graph,
        demand_model: str,
        instance_id: int,
        data_agnostic_policies:List[str],
        model_based_dynamic_programs: List[str],
        model_free_parametrized_policies: List[str],
        train_sample_sizes: List[int],
        train_samples: Dict[int, List[List[Sequence]]],
        test_samples: List[Sequence],
        inventory: Inventory,
        training_budget,
        optimal_policy = None,
    ):
        
        self.demand_model = demand_model
        self.instance_id = instance_id
        self.graph = graph
        self.data_agnostic_policies = data_agnostic_policies
        self.model_based_dynamic_programs = model_based_dynamic_programs
        self.model_free_parametrized_policies = model_free_parametrized_policies
        self.train_sample_sizes = train_sample_sizes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.inventory = inventory
        self.optimal_policy = optimal_policy
        
        self.n_samples_per_size = len(train_samples[train_sample_sizes[0]])
        
        self.n_test_samples = len(test_samples)
        self.training_budget = training_budget
        self.T = len(test_samples[0])

    def conduct_experiment(self) -> Dict[any, PolicyOutput]:
        
        instance_results = {}
        
        myopic_scores = {(edge.supply_node_id, edge.demand_node_id): edge.reward for edge in self.graph.edges.values()}
        self.graph.construct_priority_list( 'myopic', myopic_scores, allow_rejections=False)
        
        T = len(self.test_samples[0])
        n_samples_per_size = len(self.train_samples[self.train_sample_sizes[0]])
        
        
        ### OPTIMAL POLICY
        
        if self.optimal_policy is not None:
            instance_results['optimal'] = self.process_optimal()
        
        
        ### DATA AGNOSTIC ALGORITHMS
        
        # Myopic policy
        
        if 'myopic' in self.data_agnostic_policies:
            instance_results['myopic'] = self.process_myopic()
        
        if 'balance' in self.data_agnostic_policies:
            instance_results['balance'] = self.process_balance()
        
        
        ### DATA DRIVEN ALGORITHMS
        
        
        # DP Based algorithms
        self.estimator = ModelEstimator(self.graph)
        
        for policy in self.model_based_dynamic_programs:
            instance_results[policy] = self.process_dp(policy)
        
        # Model free policies

        for policy in self.model_free_parametrized_policies:
            instance_results[policy] = self.process_model_free(policy)
        

        
        
        return instance_results
        
    
    
    
    ## DATA AGNOSTIC
    
    def process_optimal(self):
        
        dp_fulfiller = PolicyFulfillment(self.graph)
        
        rewards = 0
        start = time.time()
        for sequence in self.test_samples:
            _, reward, _ = dp_fulfiller.fulfill(sequence, self.inventory, self.optimal_policy.optimal_action)
            rewards += reward/self.n_test_samples
        
        test_time = time.time() - start
        
        policy_output = PolicyOutput(rewards, 0, test_time, 0, 'optimal')
        
        return policy_output
        
    
    def process_myopic(self):
        start = time.time()
        myopic_scores = {(edge.supply_node_id, edge.demand_node_id): edge.reward for edge in self.graph.edges.values()}
        self.graph.construct_priority_list( 'myopic', myopic_scores, allow_rejections=False)
        train_time = time.time() - start
        
        
        # TEST SET
        
        greedy_fulfiller = Fulfillment(self.graph)

            
        rewards = 0
        
        start = time.time()
        
        for sequence in self.test_samples:
            _, reward, _ = greedy_fulfiller.fixed_list_fulfillment(sequence, self.inventory, 'myopic')
            rewards += reward/self.n_test_samples
            
        test_time = (time.time()-start)/self.n_test_samples
                
        
        policy_output = PolicyOutput(rewards, train_time, test_time, 0, 'myopic')
        
        return policy_output
    
    def process_balance(self):
        
        start = time.time()
        balance_fulfiller = MultiPriceBalanceFulfillment(self.graph)
        train_time = time.time() - start
        
        rewards = 0
        start = time.time()
        
        for sequence in self.test_samples:
            _, reward, _ = balance_fulfiller.fulfill(sequence, self.inventory,verbose=False)
            rewards+= reward/self.n_test_samples
            
        test_time = (time.time()-start)/self.n_test_samples
                
        
        policy_output = PolicyOutput(rewards, train_time, test_time, 0, 'balance')
        
        return policy_output
    
    
    ## MODEL FREE
    
    def process_model_free(self, policy: str):
        
        train_times = defaultdict(list)
        
        best_parameters = defaultdict(list)
        
        
        start = time.time()
        
        if policy =='time-supply_enhanced_balance':
            fulfiller = TimeSupplyEnhancedMPB(self.graph)
        
        setup_time = time.time() - start
        
        for n_train_samples in self.train_sample_sizes:
            for sample_id in range(self.n_samples_per_size):
                
                train_sample = self.train_samples[n_train_samples][sample_id]
                
                start = time.time()
                best_param = fulfiller.train(self.inventory, train_sample, budget = self.training_budget)
                best_parameters[n_train_samples].append(best_param)
                
                
                train_times[n_train_samples].append( time.time()-start + setup_time)
        
        
        
        rewards = {}
        for n_train_samples in self.train_sample_sizes:
            rewards[n_train_samples] = [0 for _ in range(self.n_samples_per_size)]
        
        
        test_times = defaultdict(list)
        
        
        for n_train_samples in self.train_sample_sizes:
            
            for i in range(self.n_samples_per_size):
                start = time.time()
                for sequence in self.test_samples:
                    reward = fulfiller.fulfill(sequence, self.inventory, best_parameters[n_train_samples][i])
                    rewards[n_train_samples][i] += reward/self.n_test_samples
                test_times[n_train_samples].append( (time.time()-start)/self.n_test_samples )
                
                
        
        policy_output = {}
            
        for n_train_samples in self.train_sample_sizes:
            policy_output[n_train_samples] = PolicyOutput(rewards[n_train_samples], train_times[n_train_samples], test_times[n_train_samples],n_train_samples, policy)
        
        return policy_output
    
    ## MODEL BASED
    
    def process_dp(self,dp_name):
        
        if dp_name == 'iid_dp':
            dp_solver = IndependentDynamicProgram(self.graph)
            


        if dp_name == 'indep_dp':
            dp_solver = IndependentDynamicProgram(self.graph)
            
            
        if dp_name == 'markov_dp':
            dp_solver = MarkovianDynamicProgram(self.graph)
            
        
        
        estimated_distribution = defaultdict(list)
        estimated_policy = defaultdict(list)
        
        train_times = defaultdict(list)
        
        for n_train_samples in self.train_sample_sizes:
            for sample_id in range(self.n_samples_per_size):
                
                train_sample = self.train_samples[n_train_samples][sample_id]
                
                start = time.time()
                if dp_name =='iid_dp':
                    iid_p_estimate = self.estimator.estimate_iid(train_sample)
                    iid_p_dict = {t:iid_p_estimate for t in range(self.T)}
                    estimated_distribution[n_train_samples].append(iid_p_dict)
                    estimated_policy[n_train_samples].append(dp_solver.compute_optimal_policy(self.inventory, self.T, iid_p_dict))
                    
                if dp_name == 'indep_dp':
                    indep_p_estimate = self.estimator.estimate_independent(train_sample)
                    estimated_distribution[n_train_samples].append(indep_p_estimate)
                    estimated_policy[n_train_samples].append( dp_solver.compute_optimal_policy(self.inventory, self.T, indep_p_estimate))

                if dp_name =='markov_dp':
                    transition_matrix_estimate, _ = self.estimator.estimate_markovian(train_sample)
                    estimated_distribution[n_train_samples].append(transition_matrix_estimate)
                    estimated_policy[n_train_samples].append(dp_solver.compute_optimal_policy(self.inventory, self.T, transition_matrix_estimate))
                
                train_times[n_train_samples].append(time.time()-start)
        
        
        
        rewards = {}
        for n_train_samples in self.train_sample_sizes:
            rewards[n_train_samples] = [0 for _ in range(self.n_samples_per_size)]
        
        dp_fulfiller = PolicyFulfillment(self.graph)
        
        test_times = defaultdict(list)
        
        
        for n_train_samples in self.train_sample_sizes:
            
           
            for i in range(self.n_samples_per_size):
                start = time.time()
                for sequence in self.test_samples:
                    _, reward, _ = dp_fulfiller.fulfill(sequence, self.inventory, estimated_policy[n_train_samples][i].optimal_action)
                    rewards[n_train_samples][i] += reward/self.n_test_samples
                test_times[n_train_samples].append( (time.time()-start)/self.n_test_samples ) 
                
        
        policy_output = {}
            
        for n_train_samples in self.train_sample_sizes:
            policy_output[n_train_samples] = PolicyOutput(rewards[n_train_samples], train_times[n_train_samples], test_times[n_train_samples],n_train_samples,dp_name)
        
        return policy_output
                
        
        



def main(demand_model):
    n_supply_nodes = 3
    n_demand_nodes = 15
    
    num_instances = 5
    
    train_sample_sizes = [ 5, 10]#, 100, 500]#, 500, 1000, 5000]
    n_samples_per_size = 5
    
    inventory = Inventory({0:2, 1:2, 2:2}, name = 'test')
    
    data_agnostic_policies = ['myopic', 'balance']
    model_based_dynamic_programs = ['iid_dp', 'indep_dp', 'markov_dp']#, 'time_enhanced_balance', 'supply_enhanced_balance']
    
    model_free_parametrized_policies = ['time-supply_enhanced_balance']
    
    n_test_samples = 500
    
    training_budget = 1000
    
    T = 12
    
    # vertex_values = {}
    # vertex_values[0] = [0.1, 0.9]
    # vertex_values[1] = [0.4, 0.6]
    # vertex_values[2] = [0.7,0.91]
    
    # vertex_values[3] = [0.59,0.89]
    
    graph_generator_seed = 0
    distribution_generator_seed = 1
    train_generator_seed = 2
    test_generator_seed = 3
    
    
    include_optimal = False
    
    if demand_model =='indep' or demand_model =='markov':
        include_optimal = True
    
    graph_generator = RandomGraphGenerator(seed = graph_generator_seed)
    dist_generator = RandomDistributionGenerator(seed = distribution_generator_seed)
    
    instances: Dict[int, Instance] = {}
    optimal_policies = {}
    
    
    for instance_id in range(num_instances):
    
        graph = graph_generator.two_valued_vertex_graph(n_supply_nodes, n_demand_nodes)
        myopic_scores = {(edge.supply_node_id, edge.demand_node_id): edge.reward for edge in graph.edges.values()}
        
        graph.construct_priority_list( 'myopic', myopic_scores, allow_rejections=False)
        
        if demand_model == 'indep':
            p = dist_generator.generate_indep(n_demand_nodes, T)
            distribution = p
            indep_dp = IndependentDynamicProgram(graph)
            optimal_policy = indep_dp.compute_optimal_policy(inventory, T, p)
            optimal_policies[instance_id] = optimal_policy
        
        elif demand_model == 'markov':
            initial_distribution, transition_matrix = dist_generator.generate_markov(n_demand_nodes, 2)
            distribution = (initial_distribution, transition_matrix)
            markov_dp = MarkovianDynamicProgram(graph)
            optimal_policy = markov_dp.compute_optimal_policy(inventory, T, transition_matrix)
            optimal_policies[instance_id] = optimal_policy
        
        
        elif demand_model =='rw':
            step_size = 3
            distribution = step_size
            # train_generator = RWGenerator(T, demand_node_list, seed = train_generator_seed, step_size = step_size)
            # test_generator = RWGenerator(T, demand_node_list, seed = test_generator_seed, step_size = step_size)
        
        instances[instance_id] = Instance(graph, deepcopy(distribution), demand_model)
        
    
    
    
    
    
    
    
    demand_node_list = list(graph.demand_nodes.keys())
    

    train_samples = {}
    test_samples = {}
    
    results = {}
    
    for instance_id in range(num_instances):
        print(f'Instance {instance_id}')
        
        instance = instances[instance_id]
        
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
        
        train_samples[instance_id] = defaultdict(list)
        for n_train_samples in train_sample_sizes:
            for _ in range(n_samples_per_size):
                train_samples[instance_id][n_train_samples].append( [train_generator.generate_sequence() for _ in range(n_train_samples)])
                
        test_samples[instance_id] = [test_generator.generate_sequence() for _ in range(n_test_samples)]

        if demand_model == 'indep' or demand_model == 'markov':
            experiment = Experiment(
                graph,
                demand_model,
                instance_id,
                data_agnostic_policies,
                model_based_dynamic_programs,
                model_free_parametrized_policies,
                train_sample_sizes,
                train_samples[instance_id],
                test_samples[instance_id],
                inventory,
                training_budget,
                optimal_policies[instance_id]
            )
            # results[i] = experiment(graph,train_sample_sizes, train_samples[i], test_samples[i], inventory, optimal_policies[i])
        else:
            experiment = Experiment(
                graph,
                demand_model,
                instance_id,
                data_agnostic_policies,
                model_based_dynamic_programs,
                model_free_parametrized_policies,
                train_sample_sizes,
                train_samples[instance_id],
                test_samples[instance_id],
                inventory,
                training_budget,
            )
            # results[i] = experiment(graph,train_sample_sizes, train_samples[i], test_samples[i], inventory)

        results[instance_id] = experiment.conduct_experiment()
        
        
    writer = OutputWriter(data_agnostic_policies, model_based_dynamic_programs, model_free_parametrized_policies, num_instances, train_sample_sizes,n_samples_per_size, include_optimal, results)
    writer.write_output(f'{demand_model}_test2.csv')

    for instance in range(num_instances):
        instance_results = results[instance]
        
        for policy in data_agnostic_policies:
            print(policy)
            print(f'Average reward: {instance_results[policy].rewards}')
            print(f'Train time:  {instance_results[policy].train_times}')
            print(f'Test time:  {instance_results[policy].test_times}')
            print('')
            print('')
            
           
            
        for policy in model_based_dynamic_programs:     
            print(policy)
            print('')
            for n_train_samples in train_sample_sizes:
                print(f'Number of train samples: {n_train_samples}')
                print(f'Average reward: {instance_results[policy][n_train_samples].rewards}')
                print(f'Train time:  {instance_results[policy][n_train_samples].train_times}')
                print(f'Test time:  {instance_results[policy][n_train_samples].test_times}')
                print('')
        
        
        
        for policy in  model_free_parametrized_policies:
            print(policy)
            print('')
            for n_train_samples in train_sample_sizes:
                print(f'Number of train samples: {n_train_samples}')
                print(f'Average reward: {instance_results[policy][n_train_samples].rewards}')
                print(f'Train time:  {instance_results[policy][n_train_samples].train_times}')
                print(f'Test time:  {instance_results[policy][n_train_samples].test_times}')
                print('')
            

if __name__ == '__main__':

    # main('markov')
    main('markov')