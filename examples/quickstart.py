"""Quickstart: build a small graph, generate demand, and compare two policies.

This example creates a small bipartite graph (3 warehouses, 4 demand regions),
generates demand sequences, places inventory, and compares Balance vs Myopic
fulfillment policies.
"""

from fulfillment_optimization import (
    Graph, Inventory, Sequence, Request,
    RandomGraphGenerator, TemporalIndependenceGenerator,
    RandomDistributionGenerator,
    BalancePolicy, PriorityListPolicy,
)

# --- 1. Build a random bipartite graph ---
graph = RandomGraphGenerator(seed=42).two_valued_vertex_graph(
    num_supply_nodes=3, num_demand_nodes=4,
)
print("Graph edges:")
for (s, d), edge in graph.edges.items():
    print(f"  warehouse {s} -> region {d}: reward = {edge.reward:.3f}")

# --- 2. Generate demand ---
demand_nodes = list(graph.demand_nodes.keys())
T = 12  # time horizon
dist_gen = RandomDistributionGenerator(seed=10)
p = dist_gen.generate_indep(num_demand_nodes=len(demand_nodes), num_periods=T)
gen = TemporalIndependenceGenerator(demand_nodes=demand_nodes, p=p, seed=99)

n_sequences = 20
sequences = [gen.generate_sequence() for _ in range(n_sequences)]
print(f"\nGenerated {n_sequences} demand sequences of length {T}")

# --- 3. Set up inventory ---
inventory = Inventory({0: 4, 1: 4, 2: 4}, name='uniform')
print(f"Inventory: {inventory.initial_inventory} (total={inventory.total_inventory})")

# --- 4. Run Balance policy ---
balance = BalancePolicy(graph)
balance_rewards = []
for seq in sequences:
    result = balance.fulfill(seq, inventory)
    balance_rewards.append(result.collected_rewards)

print(f"\nBalance policy:  avg reward = {sum(balance_rewards)/len(balance_rewards):.2f}")

# --- 5. Run Myopic (priority list) policy ---
# Build a "myopic" priority list: for each demand node, order supply nodes
# by descending reward.
graph.populate_best_supply_nodes()
for dn in graph.demand_nodes.values():
    dn.priority_lists['myopic'] = list(dn.best_supply_nodes)

myopic = PriorityListPolicy(graph)
myopic_rewards = []
for seq in sequences:
    result = myopic.fixed_list_fulfillment(seq, inventory, 'myopic')
    myopic_rewards.append(result.collected_rewards)

print(f"Myopic policy:   avg reward = {sum(myopic_rewards)/len(myopic_rewards):.2f}")
