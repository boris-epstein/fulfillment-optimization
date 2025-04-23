import time
import logging
import os
import hashlib
from datetime import datetime

import multiprocessing as mp



from Graph import Graph, RandomGraphGenerator
from FulfillmentOptimization import Fulfillment, Inventory, PolicyFulfillment, MultiPriceBalanceFulfillment, BalanceFulfillment, FluLpReSolvingFulfillment
from ModelBased import IndependentDynamicProgram, ModelEstimator, MarkovianDynamicProgram
from Demand import TemporalIndependenceGenerator, RWGenerator, MarkovianGenerator, Sequence, RandomDistributionGenerator
# from LearningPolicy import DepletionAwarePolicy, train_depletion_policy, extract_reward_matrix, SubscriptableDepletionPolicyWrapper, train_depletion_policy_black_box, train_depletion_nn_policy, NNPolicyWrapper
from ModelFree import ThresholdsFulfillment, TimeSupplyEnhancedMPB, DemandTrackingMPB, AdaptiveThresholdsFulfillment, TimeEnhancedMPB, SupplyEnhancedMPB, NeuralOpportunityCostPolicy, NeuralOpportunityCostWithIDPolicy

# from NNPolicy import OnlineMatchingPolicy, evaluate_policy_with_params,create_and_train_policy_ng
from typing import Any, Dict, List

from collections import defaultdict
import numpy as np
import csv
from copy import deepcopy
from MathPrograms import MathPrograms


def generate_experiment_logging_setup(demand_model: str, base_log_dir: str = "logs"):
    os.makedirs(base_log_dir, exist_ok=True)
    now = datetime.now().isoformat()
    hash_id = hashlib.md5(now.encode()).hexdigest()[:8]
    experiment_id = f"{demand_model}_{hash_id}"
    master_log_path = os.path.join(base_log_dir, f"{experiment_id}.log")
    return experiment_id, master_log_path




def run_single_instance(args):
    (
        instance_id,
        instance,
        demand_model,
        train_sample_sizes,
        train_samples,
        test_samples,
        inventory,
        data_agnostic_policies,
        model_based_dynamic_programs,
        model_free_parametrized_policies,
        lp_resolving_policies,
        training_budget_per_parameter,
        optimal_policy,
        experiment_id
    ) = args
    
    def setup_instance_logger(experiment_id: str, instance_id: int, log_dir: str = "logs"):
        log_path = os.path.join(log_dir, f"{experiment_id}_instance_{instance_id}.log")
        logger = logging.getLogger(f"instance_{instance_id}")
        logger.setLevel(logging.INFO)

        # Avoid duplicate handlers if rerun
        if not logger.handlers:
            handler = logging.FileHandler(log_path, mode='w')
            formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False

        return logger

    logger = setup_instance_logger(experiment_id, instance_id)
    # Remove root handlers and use this logger instead
    logging.getLogger().handlers.clear()
    logging.getLogger().addHandler(logger.handlers[0])
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().propagate = False
    logger.info(f"Starting instance {instance_id}")

    graph = instance.graph


    experiment = Experiment(
        graph,
        demand_model,
        instance_id,
        data_agnostic_policies,
        model_based_dynamic_programs,
        model_free_parametrized_policies,
        lp_resolving_policies,
        train_sample_sizes,
        train_samples,
        test_samples,
        inventory,
        training_budget_per_parameter,
        optimal_policy,
    ) if demand_model in ['indep', 'markov'] else Experiment(
        graph,
        demand_model,
        instance_id,
        data_agnostic_policies,
        model_based_dynamic_programs,
        model_free_parametrized_policies,
        lp_resolving_policies,
        train_sample_sizes,
        train_samples,
        test_samples,
        inventory,
        training_budget_per_parameter,
    )

    result = experiment.conduct_experiment()
    return (instance_id, result)

def generate_experiment_id(demand_model: str) -> str:
    now = datetime.now().isoformat()
    hash_id = hashlib.md5(now.encode()).hexdigest()[:8]  # short hash
    return f"{demand_model}_{hash_id}"


def setup_logging(experiment_id):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)


    log_path = os.path.join(log_dir, f"{experiment_id}.log")

    logging.basicConfig(
        filename=log_path,
        filemode='a',  # append mode
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)



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




def log_data_agnostic_result(instance_id, policy_output: PolicyOutput):
    logging.info(f'{instance_id},{policy_output.policy_name},{policy_output.train_set_size},0,{policy_output.rewards},{policy_output.train_times},{policy_output.test_times}')
    
    

def log_data_driven_result(instance_id, policy_outputs: Dict[int,PolicyOutput], train_sample_sizes):
    
    for n_train_samples in train_sample_sizes:
        policy_output = policy_outputs[n_train_samples]
        for sample_id in range(len(policy_output.rewards)):
            
            logging.info(f'{instance_id},{policy_output.policy_name},{policy_output.train_set_size},{sample_id},{policy_output.rewards[sample_id]},{policy_output.train_times[sample_id]},{policy_output.test_times[sample_id]}')
        
            # policy_output[n_train_samples] = PolicyOutput(rewards[n_train_samples], train_times[n_train_samples], test_times[n_train_samples],n_train_samples, policy)


class OutputWriter:
    
    def __init__(self,
                data_agnostic_policies:List[str],
                model_based_dynamic_programs: List[str],
                model_free_parametrized_policies: List[str],
                lp_resolving_policies: List[str],
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
        self.lp_resolving_policies = lp_resolving_policies
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
                
                for policy in self.model_based_dynamic_programs + self.model_free_parametrized_policies + self.lp_resolving_policies:
                
                    for num_samples in self.train_sample_sizes:
                        for sample_id in range(self.num_samples_per_size):
                            
                                spamwriter.writerow([instance_id, policy, num_samples, sample_id, self.results[instance_id][policy][num_samples].rewards[sample_id], self.results[instance_id][policy][num_samples].train_times[sample_id], self.results[instance_id][policy][num_samples].test_times[sample_id]])
                



def compute_cumulative_average_demand(train_sample: List[Sequence], graph: Graph):
    
    cumulative_average_demand = {}
    n_train_samples = len(train_sample)
    T = len(train_sample[0])
    
    for t in range(T+1):
        cumulative_average_demand[t] = defaultdict(int)
        
    
    for t in range(T-1, -1, -1):
        
        current_t_prob = defaultdict(int)
        
        for sequence in train_sample:
            current_t_prob[sequence.requests[t].demand_node] += 1/n_train_samples   
        
        for demand_node_id in graph.demand_nodes:
            cumulative_average_demand[t][demand_node_id] = current_t_prob[demand_node_id] + cumulative_average_demand[t+1][demand_node_id]
        
    
    return cumulative_average_demand

  
class Experiment:
    
    def __init__(self,
        graph: Graph,
        demand_model: str,
        instance_id: int,
        data_agnostic_policies:List[str],
        model_based_dynamic_programs: List[str],
        model_free_parametrized_policies: List[str],
        lp_resolving_policies: List[str],
        train_sample_sizes: List[int],
        train_samples: Dict[int, List[List[Sequence]]],
        test_samples: List[Sequence],
        inventory: Inventory,
        training_budget_per_parameter: int,
        optimal_policy = None,
    ):
        
        self.demand_model = demand_model
        self.instance_id = instance_id
        self.graph = graph
        self.data_agnostic_policies = data_agnostic_policies
        self.model_based_dynamic_programs = model_based_dynamic_programs
        self.model_free_parametrized_policies = model_free_parametrized_policies
        self.lp_resolving_policies = lp_resolving_policies
        self.train_sample_sizes = train_sample_sizes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.inventory = inventory
        self.optimal_policy = optimal_policy
        
        self.n_samples_per_size = len(train_samples[train_sample_sizes[0]])
        
        self.n_test_samples = len(test_samples)
        self.training_budget_per_parameter = training_budget_per_parameter
        self.T = len(test_samples[0])
        
        self.math_progs = MathPrograms(self.graph)


    def conduct_experiment(self) -> Dict[any, PolicyOutput]:
        
        instance_results = {}
        
        
        myopic_scores = {(edge.supply_node_id, edge.demand_node_id): edge.reward for edge in self.graph.edges.values()}
        self.graph.construct_priority_list( 'myopic', myopic_scores, allow_rejections=False)
        
        T = len(self.test_samples[0])
        n_samples_per_size = len(self.train_samples[self.train_sample_sizes[0]])
        
        
        ### OPTIMAL POLICY
        
        
        if self.optimal_policy is not None:
            instance_results['optimal'] = self.process_optimal()
        
        
        ### OFFLINE POLICY
        
        if 'offline' in self.data_agnostic_policies:
            
            instance_results['offline'] = self.process_offline()
        
        
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
        
        for policy in self.lp_resolving_policies:
            instance_results[policy] = self.process_lp(policy)
        
        # Model free policies

        for policy in self.model_free_parametrized_policies:
            instance_results[policy] = self.process_model_free(policy)
        

        
        
        return instance_results
        
    
    
    
    ## DATA AGNOSTIC
    
    
    def process_offline(self):
        
        start = time.time()
        offline_lp, _ = self.math_progs.offline_linear_program_fixed_inventory(self.test_samples, self.inventory)
        offline_lp.optimize()
        off_time = time.time() - start
    
        
        policy_output = PolicyOutput(offline_lp.ObjVal/self.n_test_samples, 0, off_time, 0, 'offline')
        
        return policy_output

    
    def process_optimal(self):
        
        dp_fulfiller = PolicyFulfillment(self.graph)
        
        rewards = 0
        start = time.time()
        for sequence in self.test_samples:
            _, reward, _ = dp_fulfiller.fulfill(sequence, self.inventory, self.optimal_policy.optimal_action)
            rewards += reward/self.n_test_samples
        
        test_time = (time.time() - start)/self.n_test_samples
        
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
        
        log_data_agnostic_result(self.instance_id, policy_output)
        
        return policy_output
    
    
    ## MODEL FREE
    
    def process_model_free(self, policy: str):
        
        train_times = defaultdict(list)
        
        best_parameters = defaultdict(list)
        
        
        start = time.time()
        
        if policy =='time-supply_enhanced_balance':
            fulfiller = TimeSupplyEnhancedMPB(self.graph)
            
        if policy == 'time_enhanced_balance':
            fulfiller = TimeEnhancedMPB(self.graph)
        if policy == 'supply_enhanced_balance':
            fulfiller = SupplyEnhancedMPB(self.graph)
            
        if policy == 'neural_opportunity_cost':
            fulfiller = NeuralOpportunityCostPolicy(self.graph)
            
        if policy == 'neural_opportunity_cost_with_id':
            fulfiller = NeuralOpportunityCostWithIDPolicy(self.graph)
        
        setup_time = time.time() - start
        
        for n_train_samples in self.train_sample_sizes:
            for sample_id in range(self.n_samples_per_size):
                
                train_sample = self.train_samples[n_train_samples][sample_id]
                
                train_budget = min(self.training_budget_per_parameter * fulfiller.num_parameters, 2000)
                
                start = time.time()
                best_param = fulfiller.train(self.inventory, train_sample, budget = train_budget)
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
        
        log_data_driven_result(self.instance_id, policy_output, self.train_sample_sizes)
        
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
        
        log_data_driven_result(self.instance_id, policy_output, self.train_sample_sizes)
        
        return policy_output
                
    
    def process_lp(self,lp_name):
        if lp_name == 'fluid_lp_resolving':
            
            return self.process_fluid_lp_resolving()
    
    def process_fluid_lp_resolving(self):
        
        re_solving_epochs = [2,5,9]
        
        initial_dual_variables = defaultdict(list)
        backwards_cumulative_average_demand = defaultdict(list)
        train_times = defaultdict(list)
        
        fulfiller = FluLpReSolvingFulfillment(self.graph)
        
        for n_train_samples in self.train_sample_sizes:
            for sample_id in range(self.n_samples_per_size):
                
                train_sample = self.train_samples[n_train_samples][sample_id]
                
                start = time.time()
                
                
                backwards_cumulative_average_demand[n_train_samples].append( compute_cumulative_average_demand(train_sample, self.graph) )
                
                
                initial_dual_variables[n_train_samples].append(fulfiller.compute_dual_variables(0,self.inventory.initial_inventory,backwards_cumulative_average_demand[n_train_samples][sample_id]))
                
                train_times[n_train_samples].append(time.time()-start)
        
        
        
        rewards = {}
        for n_train_samples in self.train_sample_sizes:
            rewards[n_train_samples] = [0 for _ in range(self.n_samples_per_size)]
        
        
        test_times = defaultdict(list)
        
        
        for n_train_samples in self.train_sample_sizes:
            for i in range(self.n_samples_per_size):
                start = time.time()
                for sequence in self.test_samples:
                    _, reward, _ = fulfiller.fulfill(sequence, self.inventory, initial_dual_variables[n_train_samples][i], backwards_cumulative_average_demand[n_train_samples][i],re_solving_epochs)
                    rewards[n_train_samples][i] += reward/self.n_test_samples
                test_times[n_train_samples].append( (time.time()-start)/self.n_test_samples ) 
                
        
        policy_output = {}
            
        for n_train_samples in self.train_sample_sizes:
            policy_output[n_train_samples] = PolicyOutput(rewards[n_train_samples], train_times[n_train_samples], test_times[n_train_samples],n_train_samples,'fluid_lp_resolving')
        
        log_data_driven_result(self.instance_id, policy_output, self.train_sample_sizes)
        return policy_output


def main(demand_model):
    
    
    
    experiment_id, master_log_path = generate_experiment_logging_setup(demand_model)

    logging.basicConfig(
        filename=master_log_path,
        filemode='w',
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    

    
    parallel = True
    
    n_supply_nodes = 3
    n_demand_nodes = 15
    
    num_instances = 8
    logging.info(f"Starting experiment {experiment_id} with {num_instances} instances")
    
    train_sample_sizes = [ 10, 50, 100, 500]#, 100, 500]#, 500, 1000, 5000]
    n_samples_per_size = 5
    
    inventory = Inventory({0:2, 1:2, 2:2}, name = 'test')
    
    data_agnostic_policies = ['myopic', 'balance', 'offline']
    model_based_dynamic_programs = ['iid_dp', 'indep_dp', 'markov_dp']#, 'time_enhanced_balance', 'supply_enhanced_balance']
    
    model_free_parametrized_policies = ['time-supply_enhanced_balance', 'neural_opportunity_cost', 'neural_opportunity_cost_with_id']# ['neural_opportunity_cost']#,'time_enhanced_balance','supply_enhanced_balance','time-supply_enhanced_balance']
    
    lp_resolving_policies = ['fluid_lp_resolving']
    
    n_test_samples = 5000
    
    training_budget_per_parameter = 100
    
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
    
    logging.info('experiment metadata')
    logging.info(f'parllel = {parallel}')
    logging.info(f'graph_generator_seed={graph_generator_seed}')
    logging.info(f'distribution_generator_seed={distribution_generator_seed}')
    logging.info(f'train_generator_seed={train_generator_seed}')
    logging.info(f'test_generator_seed={test_generator_seed}')
    
    
    
    logging.info(f'T = {T}')
    logging.info(f'demand_model = {demand_model}')
    logging.info(f'n_supply_nodes = {n_supply_nodes}')
    logging.info(f'n_demand_nodes = {n_demand_nodes}')
    
    logging.info(f'n_instances = {num_instances}')
    logging.info(f'train_sample_sizes = {train_sample_sizes}')
    logging.info(f'n_test_samples = {n_test_samples}')
    logging.info(f'training_budget_per_parameter = {training_budget_per_parameter}')
    
    logging.info(f'data_agnostic_policies = {data_agnostic_policies}')
    logging.info(f'model_based_dynamic_programs = {model_based_dynamic_programs}')
    logging.info(f'model_free_parametrized_policies = {model_free_parametrized_policies}')
    logging.info(f'lp_resolving_policies = {lp_resolving_policies}')
    
    
    
    
    include_optimal = False
    
    if demand_model =='indep' or demand_model =='markov':
        include_optimal = True
    
    graph_generator = RandomGraphGenerator(seed = graph_generator_seed)
    dist_generator = RandomDistributionGenerator(seed = distribution_generator_seed)
    
    instances: Dict[int, Instance] = {}
    optimal_policies = {}
    
    
    # demand_node_list = list(graph.demand_nodes.keys())
    

    train_samples = {}
    test_samples = {}
    
    
    for instance_id in range(num_instances):
    
        graph = graph_generator.two_valued_vertex_graph(n_supply_nodes, n_demand_nodes)
        demand_node_list = [demand_node_id for demand_node_id in graph.demand_nodes]
        myopic_scores = {(edge.supply_node_id, edge.demand_node_id): edge.reward for edge in graph.edges.values()}
        
        graph.construct_priority_list( 'myopic', myopic_scores, allow_rejections=False)
        
        if demand_model == 'indep':
            p = dist_generator.generate_indep(n_demand_nodes, T)
            distribution = p
            indep_dp = IndependentDynamicProgram(graph)
            optimal_policy = indep_dp.compute_optimal_policy(inventory, T, p)
            optimal_policies[instance_id] = optimal_policy
        
        elif demand_model == 'markov':
            initial_distribution, transition_matrix = dist_generator.generate_markov(n_demand_nodes, 4)
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
        
    
        
        
        instance = instances[instance_id]
        
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

    
    
    
    
    
    
    results = {}
    
    
    if parallel:
        
        args_list = []
        for instance_id in range(num_instances):
            instance = instances[instance_id]
            args = (
                instance_id,
                instance,
                demand_model,
                train_sample_sizes,
                train_samples[instance_id],
                test_samples[instance_id],
                inventory,
                data_agnostic_policies,
                model_based_dynamic_programs,
                model_free_parametrized_policies,
                lp_resolving_policies,
                training_budget_per_parameter,
                optimal_policies.get(instance_id),  # will be None if not available
                experiment_id
            )
            args_list.append(args)
        
        # Run in parallel
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results_list = pool.map(run_single_instance, args_list)

        # Collect results
        results = {instance_id: result for instance_id, result in results_list}
        
        
    else:
    
        logging.info('instance_id,policy_name,sample_size,sample_id,average_test_reward,training_time,average_testing_time')
        
        for instance_id in range(num_instances):
            
            
            instance = instances[instance_id]
            
            graph = instance.graph

            
            if demand_model == 'indep' or demand_model == 'markov':
                experiment = Experiment(
                    graph,
                    demand_model,
                    instance_id,
                    data_agnostic_policies,
                    model_based_dynamic_programs,
                    model_free_parametrized_policies,
                    lp_resolving_policies,
                    train_sample_sizes,
                    train_samples[instance_id],
                    test_samples[instance_id],
                    inventory,
                    training_budget_per_parameter,
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
                    lp_resolving_policies,
                    train_sample_sizes,
                    train_samples[instance_id],
                    test_samples[instance_id],
                    inventory,
                    training_budget_per_parameter,
                )
                # results[i] = experiment(graph,train_sample_sizes, train_samples[i], test_samples[i], inventory)

            results[instance_id] = experiment.conduct_experiment()
            
            
    writer = OutputWriter(data_agnostic_policies, model_based_dynamic_programs, model_free_parametrized_policies, lp_resolving_policies, num_instances, train_sample_sizes,n_samples_per_size, include_optimal, results)
    writer.write_output(f'{experiment_id}.csv')

            

if __name__ == '__main__':

    # main('markov')
    main('markov')