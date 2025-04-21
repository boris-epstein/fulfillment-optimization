from Graph import Graph
from Demand import HiddenMarkovGenerator

import numpy as np

def correl_graph(low_reward, high_reward_probability, seed = 0):
    
    
    graph = Graph()
    
    graph.add_supply_node(0)
    
    graph.add_demand_node(0)
    graph.add_demand_node(1)
    
    graph.add_edge(0,0,low_reward)
    graph.add_edge(0,1,1)
    
    
    states = list(range(13))
    
    initial_distribution = np.zeros(13)
    initial_distribution[0] = 1
    
    state_distributions = {}
    
    state_distributions[0] = [1 - high_reward_probability, high_reward_probability, 0]
    state_distributions[1] = [0,0,1]
    state_distributions[2] = [0,0,1]
    state_distributions[3] = [1 - high_reward_probability, high_reward_probability, 0]
    state_distributions[4] = [0,0,1]
    state_distributions[5] = [0,0,1]
    state_distributions[6] = [0,0,1]
    state_distributions[7] = [1 - high_reward_probability, high_reward_probability, 0]
    state_distributions[8] = [0,0,1]
    state_distributions[9] = [0,0,1]
    state_distributions[10] = [0,0,1]
    state_distributions[11] = [1 - high_reward_probability, high_reward_probability, 0]
    state_distributions[12] = [1 - high_reward_probability, high_reward_probability, 0]
    
    transition_matrix = np.zeros((13,13))
    
    transition_matrix[0,1] = 1/2
    transition_matrix[0,12] = 1/2
    for i in range(1,12):
        transition_matrix[i,i+1]=1
        
    
    transition_matrix[12,12] = 1
        
    print(transition_matrix)
    distribution = HiddenMarkovGenerator(12, [0,1,2], states, state_distributions, initial_distribution, transition_matrix, seed=  seed)    
    
    return graph, distribution
    
if __name__ == '__main__':
    graph, distribution = correl_graph(0.5,0.2)
    
    for i in range(10):
        print(distribution.generate_sequence())