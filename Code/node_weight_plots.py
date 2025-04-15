from Graph import Graph
import numpy as np
import matplotlib.pyplot as plt


#complete graph
supply_nodes, demand_nodes, rewards = ('complete_supplynodes_seed=20.csv', 'complete_demandnodes_seed=20.csv', 'complete_rewards_seed=20.csv')

# #long chain graph
# supply_nodes, demand_nodes, rewards = ('longchain_supplynodes_seed=10.csv', 'longchain_demandnodes_seed=10.csv', 'longchain_rewards_seed=10.csv')


# graph = Graph(mode = 'test_graph_1')
graph = Graph(supply_nodes, demand_nodes, rewards)
print(len(graph.supply_nodes))
print(len(graph.demand_nodes))
print(len(graph.edges))
# print(len(graph.destinations['DAB4'].closest_origins))



# demand_nodes_list
demand_nodes_list = list(graph.demand_nodes.keys())
weights = np.array([ 1.0 for _ in demand_nodes_list])
# weights = np.array([ sum(graph.edges[supply_node_id, demand_node_id].reward for supply_node_id in graph.supply_nodes) for demand_node_id in demand_nodes_list])
# weights = np.array([ sum(graph.edges[supply_node_id, demand_node_id].reward**2 for supply_node_id in graph.supply_nodes) for demand_node_id in demand_nodes_list])
# weights = np.array([ sum(graph.edges[supply_node_id, demand_node_id].reward for supply_node_id in graph.supply_nodes)**2 for demand_node_id in demand_nodes_list])

weights /= weights.sum()

# Normalize weights to a 0-1 range

if np.max(weights) - np.min(weights)>0:
    normalized_weights =(weights - np.min(weights)) / (np.max(weights) - np.min(weights))
else:
    normalized_weights = weights


# Scale to desired size range (e.g., 10 to 200 points)
dot_sizes = normalized_weights * (200 - 10) + 10


dn_x = [dn.x for dn in graph.demand_nodes.values()]
dn_y = [dn.y for dn in graph.demand_nodes.values()]

sn_x = [sn.x for sn in graph.supply_nodes.values()]
sn_y = [sn.y for sn in graph.supply_nodes.values()]


# Scatter plot
plt.scatter(sn_x, sn_y, color = 'orange')

plt.scatter(dn_x, dn_y, s=dot_sizes, c=weights, cmap='viridis', alpha=0.7)
plt.colorbar(label='Weight')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot with Scaled Dot Sizes')
plt.show()