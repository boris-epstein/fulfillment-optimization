
import numpy as np
from collections import defaultdict
from typing import List
from Graph import Node
from typing import Dict

class Request:
    def __init__(self, demand_node: int, arrival_time: float = 0.0) -> None:
        self.demand_node = demand_node
        self.arrival_time = arrival_time


class Sequence:
    def __init__(self, requests: List[Request], compute_aggregates = True) -> None:
        self.requests = requests
        self.length = len(requests)

        if compute_aggregates:
            self.compute_aggregates()
            self.compute_leftover_aggregates()

    def compute_aggregates(self):
        self.aggregate_demand = defaultdict(int)

        for request in self.requests:
            self.aggregate_demand[request.demand_node] += 1

    def compute_leftover_aggregates(self):
        self.leftover_aggregate_demand = {}
        T = self.length
        self.leftover_aggregate_demand[T] = defaultdict(int)
        self.leftover_aggregate_demand[T-1] = defaultdict(int)
        self.leftover_aggregate_demand[T-1][self.requests[T-1].demand_node] +=1
        
        for t in range(T-2, -1, -1):
            self.leftover_aggregate_demand[t] = self.leftover_aggregate_demand[t+1].copy()
            self.leftover_aggregate_demand[t][self.requests[t].demand_node] +=1
    
    def __len__(self):
        return len(self.requests)
    
    def __str__(self) -> str:

        return str([req.demand_node for req in self.requests])

class CorrelGenerator:
    def __init__(self, mean: float,  demand_nodes: List[int], weights: List[float] , seed: int = 0, distribution: str = 'geometric', std_dev: float = 1,) -> None:

        self.rng = np.random.default_rng(seed = seed)
        self.mean = mean
        self.std_dev = std_dev
        self.weights = weights
        self.demand_nodes = demand_nodes
        self.distribution = distribution
        
        if distribution == 'geometric':
            self.p = 1/(1+mean)
        
        self.average_demands = {}
        
        for demand_node in demand_nodes:
            self.average_demands[demand_node] = self.mean * self.weights[demand_node]

    def generate_sequence(self) -> Sequence:

        if self.distribution =='normal':
            T = max(0,int(round( self.rng.normal( loc = self.mean, scale = self.std_dev) )))
            
        if self.distribution == 'exponential':
            T = int(round( self.rng.exponential( scale = self.mean) ))

        if self.distribution == 'deterministic':
            T = int(round(self.mean))
        
        if self.distribution == 'geometric':
            T = self.rng.geometric(self.p) - 1
        
        reqs = self.rng.choice( a = self.demand_nodes, size = T, p = self.weights )
        
        if self.distribution == 'deterministic':
            arrival_times = np.arange(1/(T+1),1, 1/(T+1))
        
        else:
            arrival_times =  np.sort( self.rng.uniform(size = T) )
        
        
        requests = [ Request(reqs[t],arrival_times[t]) for t in range(T) ]

        return Sequence(requests)

    def set_mean(self, new_mean: float):
        self.mean = new_mean

class TemporalIndependenceGenerator:
    def __init__(self,demand_nodes: List[int], p: Dict[int, List[float]],seed: int):
        self.p = p
        self.demand_nodes = demand_nodes
        self.T = len(p)
        self.rng =np.random.default_rng(seed)


    def generate_sequence(self) -> Sequence:
        
        reqs = [None for t in range(self.T)]
        
        for t in range(self.T):
            choice = self.rng.choice(a = self.demand_nodes, p = self.p[t])
            reqs[t] = choice
        requests = [Request(reqs[t]) for t in range(self.T)]
        
        return Sequence(requests)

class IndepGenerator:
    def __init__(self,
                 means: List[float],
                 demand_nodes: List[int],
                 seed: int = 0,
                 distribution: str = 'geometric',
                 std_dev: float = 1,
            ):
        """

        Args:
            means (List[float]): List of means, one for each demand node in the demand nodes list above (they must correspond)
            demand_nodes (List[int]): _description_
            seed (int, optional): _description_. Defaults to 0.
            distribution (str, optional): _description_. Defaults to 'exponential'.
            std_dev (float, optional): _description_. Defaults to 1.
        """
        self.rng = np.random.default_rng(seed = seed)
        self.means = means
        self.std_dev = std_dev
        self.demand_nodes = demand_nodes
        self.distribution = distribution
        self.average_demands = {}
        for demand_node_id in self.demand_nodes:
            self.average_demands[demand_node_id] = self.means[demand_node_id]

        if distribution == 'geometric':
            self.p = {}
            for demand_node_id in demand_nodes:
                self.p[demand_node_id] = 1/(1 + means[demand_node_id])

    def generate_sequence(self) -> Sequence:

        total_demand = {} # dict mapping demand node to total demand there
        reqs = []
        
        for demand_node_id in self.demand_nodes:
            if self.distribution == 'exponential':
                total_demand[demand_node_id] = int(round( self.rng.exponential( scale = self.means[demand_node_id]) ))
            if self.distribution == 'geometric':
                total_demand[demand_node_id] = self.rng.geometric(self.p[demand_node_id]) - 1
                
            reqs += [demand_node_id for _ in range(total_demand[demand_node_id])]
        self.rng.shuffle(reqs)
        
        T = len(reqs)

        arrival_times =  np.sort( self.rng.uniform(size = T) )        
        requests = [ Request(reqs[t],arrival_times[t]) for t in range(T) ]

        return Sequence(requests)



class RWGenerator:
    
    def __init__(self, mean: float,  demand_nodes: List[int] , seed: int = 0, distribution: str = 'deterministic', step_size: int = 1) -> None:
        
        self.rng = np.random.default_rng(seed = seed)
        self.mean = mean
        
        if distribution == 'geometric':
            self.p = 1/(1+mean)
            
        
        self.step_size = step_size
        self.distribution = distribution
        self.demand_nodes = demand_nodes
        
            
    def generate_sequence(self) -> Sequence:
        
        if self.distribution == 'geometric':
            T = self.rng.geometric(self.p) - 1
            
        if self.distribution == 'deterministic':
            T = int(self.mean)
            
        walk = np.array([1.0 for demand_node in self.demand_nodes])
        

        
        reqs = []
        for t in range(T):
            
            #update weights
            for demand_node in self.demand_nodes:
                bit = self.rng.binomial(1,1/2)
                step = 2*(bit-1/2)*self.step_size
                walk[demand_node] =walk[demand_node] * np.exp(step)
            
            
            req = self.rng.choice( a = self.demand_nodes, p = walk/walk.sum() )
            
            reqs.append(req)
            
            
        if self.distribution == 'deterministic':
            arrival_times = np.arange(1/(T+1),1, 1/(T+1))
        
        else:
            arrival_times =  np.sort( self.rng.uniform(size = T) )
        
        
        requests = [ Request(reqs[t],arrival_times[t]) for t in range(T) ]

        return Sequence(requests)            
            
            
class MarkovianGenerator:
    def __init__(self,T , demand_nodes: List[int],transition_matrix, initial_distribution, seed: int = 0):
        self.demand_nodes = demand_nodes
        self.transition_matrix = transition_matrix
        self.initial_distribution = initial_distribution
        self.rng = np.random.default_rng(seed=seed)
        self.seed = seed
        self.T= T

    def generate_sequence(self):
        arrival_times = np.arange(1/(self.T+1),1, 1/(self.T+1))
        reqs = []
        latest_arrival = self.rng.choice(a = self.demand_nodes, p = self.initial_distribution)
        t=0
        reqs.append(latest_arrival)
        for t in range(1,self.T):
            latest_arrival = self.rng.choice(a = self.demand_nodes, p = self.transition_matrix[latest_arrival,:])
            reqs.append(latest_arrival)
        
        requests = [ Request(reqs[t],arrival_times[t]) for t in range(self.T) ]

        return Sequence(requests)          
            
        
class HiddenMarkovGenerator:
    
    def __init__(self,
                 T: int,
                 demand_nodes,
                 states: List[int],
                 state_distribution: Dict[int, List[float]],
                 initial_distribution: List[float],
                 transition_matrix : np.array,
                 seed: int = 0,
                 ):
        
        self.states = states
        self.demand_nodes = demand_nodes
        self.state_distribution = state_distribution
        self.initial_distribution = initial_distribution
        self.transition_matrix = transition_matrix
        self.seed = seed
        self.rng = np.random.default_rng(seed = seed)
        self.T = T
        
    def generate_sequence(self):
        arrival_times = np.arange(1/(self.T+1),1, 1/(self.T+1))
        reqs = []
        
        current_state = self.rng.choice(a = self.states,p = self.initial_distribution)
        
        # latest_arrival = self.rng.choice(a = self.demand_nodes, p = self.initial_distribution)
        
        for t in range(self.T):
            
            latest_arrival = self.rng.choice(a = self.demand_nodes, p = self.state_distribution[current_state])
            reqs.append(latest_arrival)
            current_state = self.rng.choice(a = self.states, p = self.transition_matrix[current_state,:])
        
        requests = [ Request(reqs[t],arrival_times[t]) for t in range(self.T) ]

        return Sequence(requests)

class RandomDistributionGenerator:
    def __init__(self, seed = 0):
        
        self.rng = np.random.default_rng(seed=seed)
    
    
    def generate_indep(self, num_demand_nodes, num_periods, bias = None):
        
        if bias is None:
            bias = np.array([0 for demand_node_id in range(num_demand_nodes)])
        else:
            bias = np.array(bias)
            
        
            
            
        p = {}
        
        for t in range(num_periods):
            exp_vector = self.rng.exponential(1,size = num_demand_nodes) + bias
            p[t] = exp_vector/np.sum(exp_vector)
        
        return p
            
    def generate_markov(self, num_demand_nodes, diagonal_bias: int=0, initial_dist_bias = None):
        
        if initial_dist_bias is None:
            initial_dist_bias = np.zeros(num_demand_nodes)
        
        else:
            initial_dist_bias = np.array(initial_dist_bias)
            
        exp_vector = self.rng.exponential(1,size = num_demand_nodes) + initial_dist_bias
        initial_distribution = np.ones(num_demand_nodes) /num_demand_nodes
        
        transition_matrix = np.zeros((num_demand_nodes,num_demand_nodes))
        
        for demand_node_id in range(num_demand_nodes):
            bias_term = np.zeros(num_demand_nodes)
            bias_term[demand_node_id] = diagonal_bias
            exp_vector = self.rng.exponential(1,size = num_demand_nodes) + bias_term
            transition_matrix[demand_node_id] = exp_vector/np.sum(exp_vector)
        
        return initial_distribution, transition_matrix
        
if __name__=='__main__':

    # mean = 300
    # std_dev = 1
    # demand_nodes = [0,1,2,3]
    # weights = [0.25,0.25,0.25,0.25]
    # means = [weights[i]*mean for i in range(len(weights))]
    # seed = 3

    # generator = RWGenerator(mean = mean, std_dev = std_dev, demand_nodes = demand_nodes, weights = weights, seed=seed)
    
    # # generator = IndepGenerator(means, demand_nodes, seed = seed)
    # for i in range(10):
    #     seq = generator.generate_sequence()
    #     print(seq)
    
    demand_nodes = [0,1,2]
    
    # p = {}
    # p[0] = [1/2,0, 1/2]
    # p[1] = [1, 0, 0]
    # p[2] = [1/3,1/3,1/3]
    
    # generator = TemporalIndependenceGenerator(demand_nodes,p, seed=0)
    # for i in range(20):
    #     seq = generator.generate_sequence()
    #     print(seq)
        
    # initial_distribution = [1/3, 1/3, 1/3]
    
    # transition_matrix = np.array(
    #     [[0.9,0.1,0],
    #      [0.05,0.9,0.05],
    #      [0.1,0.1,0.8]
    #      ]
    # )
    # T = 4
    
    # generator = MarkovianGenerator(6,demand_nodes, transition_matrix, initial_distribution, seed =0 )
    
    # for i in range(2):
    #     seq = generator.generate_sequence()
    #     print(seq)
    #     print(seq.leftover_aggregate_demand)


    states = [0,1,2]
    initial_distribution = [1, 0, 0]
    
    transition_matrix = np.array(
        [[0, 1/2, 1/2],
        [0, 1, 0],
        [0, 0, 1]]
    )
    state_distribution = {}
    
    state_distribution[0] = [1/2, 1/2, 0]
    state_distribution[1] = [1/2, 1/2, 0]
    state_distribution[2] = [0 , 0, 1]
    
    T = 5
    
    generator = HiddenMarkovGenerator(T,demand_nodes, states, state_distribution,initial_distribution,transition_matrix, seed = 0)
    for _ in range(16):
        sequence = generator.generate_sequence()
        print(sequence)    

    # dist_generator = RandomDistributionGenerator(seed = 0)
    # print(dist_generator.generate_indep(len(demand_nodes), 5, bias = [2,0,0]))
    # print(dist_generator.generate_markov(len(demand_nodes), 3))
    