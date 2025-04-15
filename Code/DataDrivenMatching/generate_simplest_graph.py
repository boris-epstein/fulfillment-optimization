import numpy as np
import matplotlib.pyplot as plt
import csv

def dist(x1, y1, x2, y2):
    return np.sqrt( (x1-x2)**2 + (y1-y2)**2 )

class Node:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        
    def __repr__(self):
        return f'(id: {self.id}, x: {self.x}, y: {self.y})'
        


graph_name = 'three_node'



RANDOM_SEED = 14
# "Nicest" result so far with RANDOM_SEED = 10 for 10 nodes
# for 5 nodes: seed 14
np.random.seed(RANDOM_SEED)
N_SUPPLY_NODES = 3
N_DEMAND_NODES = 4

node_rewards = {}
node_rewards[0] = 1.0
node_rewards[1] = 0.75
node_rewards[2] = 0.5


def generate_supply_nodes(n_supply_nodes):
    
    supply_nodes = {}
    for i in range(n_supply_nodes):
        x = np.random.uniform()
        y = np.random.uniform()
        node = Node(i, x, y)
        supply_nodes[i] = node
    return supply_nodes


def generate_demand_nodes(n_demand_nodes):
    demand_nodes = {}

    for i in range(n_demand_nodes):
        x = np.random.uniform()
        y = np.random.uniform()
        node = Node(i, x, y)
        demand_nodes[i] = node
    return demand_nodes

def generate_rewards(supply_nodes, demand_nodes, node_rewards):

    rewards = {}
    
    rewards[0,0] = node_rewards[0]
    rewards[0,1] = node_rewards[0]
    
    rewards[1,0] = node_rewards[1]
    rewards[1,2] = node_rewards[1]
    
    rewards[2,1] = node_rewards[2]
    rewards[2,3] = node_rewards[2]
        
    return rewards


def save_nodes(nodes, type = 'supply', dir = ''):
    path = dir + f'{graph_name}_graph_{type}nodes.csv'
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(['index', 'x', 'y'])
        for i in nodes:
            node = nodes[i]
            writer.writerow([node.id, node.x, node.y])

def save_rewards(rewards, dir = ''):
    path = dir + f'{graph_name}_graph_rewards.csv'
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(['supply_node','demand_node','reward'])
        for i,j in rewards:
            writer.writerow([i,j,rewards[i,j]])


supply_nodes = generate_supply_nodes(N_SUPPLY_NODES)
demand_nodes = generate_demand_nodes(N_DEMAND_NODES)
rewards = generate_rewards(supply_nodes, demand_nodes, node_rewards)
        
print("Supply nodes")
print(supply_nodes)
print('')
print('Demand nodes')
print(demand_nodes)
print('')
print('Rewards')
print(rewards)

print('Saving output')

save_nodes(supply_nodes, 'supply')
save_nodes(demand_nodes, 'demand')
save_rewards(rewards)