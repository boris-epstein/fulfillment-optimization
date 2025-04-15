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
        




RANDOM_SEED = 20
# "Nicest" result so far with RANDOM_SEED = 2 for 10,30 = supply,demand
# For 5,15: seed = 20
np.random.seed(RANDOM_SEED)
N_SUPPLY_NODES = 5
N_DEMAND_NODES = 15


def generate_supply_nodes(n_supply_nodes):
    
    supply_nodes = {}
    for i in range(n_supply_nodes):
        x = np.random.uniform(0.2,0.8)
        y = np.random.uniform(0.2,0.8)
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

def compute_costs(supply_nodes, demand_nodes):

    costs = {}
    sqrt2 = np.sqrt(2)
    for i in supply_nodes:
        supply_node = supply_nodes[i]
        x1 = supply_node.x
        y1 = supply_node.y
        for j in demand_nodes:
            demand_node = demand_nodes[j]
            x2 = demand_node.x
            y2 = demand_node.y
            costs[i,j] = 1 - dist(x1, y1, x2, y2  )/sqrt2
    
    return costs


def save_nodes(nodes, type = 'supply', dir = ''):
    path = dir + f'complete_{type}nodes_seed={RANDOM_SEED}.csv'
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(['index', 'x', 'y'])
        for i in nodes:
            node = nodes[i]
            writer.writerow([node.id, node.x, node.y])

def save_rewards(rewards, dir = ''):
    path = dir + f'complete_rewards_seed={RANDOM_SEED}.csv'
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(['supply_node','demand_node','reward'])
        for i,j in rewards:
            writer.writerow([i,j,rewards[i,j]])


supply_nodes = generate_supply_nodes(N_SUPPLY_NODES)
demand_nodes = generate_demand_nodes(N_DEMAND_NODES)
costs = compute_costs(supply_nodes, demand_nodes)
        
print("Supply nodes")
print(supply_nodes)
print('')
print('Demand nodes')
print(demand_nodes)
print('')
print('Costs')
print(costs)

print('Saving output')

save_nodes(supply_nodes, 'supply')
save_nodes(demand_nodes, 'demand')
save_rewards(costs)


plt.scatter( [ supply_nodes[n].x for n in supply_nodes],  [supply_nodes[n].y for n in supply_nodes])
plt.scatter( [ demand_nodes[n].x for n in demand_nodes],  [demand_nodes[n].y for n in demand_nodes])
plt.show()