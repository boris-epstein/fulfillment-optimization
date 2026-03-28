"""Compare inventory placement strategies.

Evaluates uniform, fluid LP rounding, and offline LP rounding inventory
placements by running the Balance policy on each.
"""

from collections import defaultdict

from fulfillment_optimization import (
    RandomGraphGenerator, RandomDistributionGenerator,
    TemporalIndependenceGenerator,
    Inventory, InventoryOptimizer,
    BalancePolicy,
)

# --- Instance setup ---
N_SUPPLY, N_DEMAND, T = 4, 5, 15
TOTAL_INVENTORY = 12
N_TRAIN, N_TEST = 40, 30
SOLVER = 'highs'

graph = RandomGraphGenerator(seed=7).two_valued_vertex_graph(N_SUPPLY, N_DEMAND)
demand_nodes = list(graph.demand_nodes.keys())

dist_gen = RandomDistributionGenerator(seed=20)
p = dist_gen.generate_indep(N_DEMAND, T)
train_gen = TemporalIndependenceGenerator(demand_nodes, p, seed=300)
test_gen = TemporalIndependenceGenerator(demand_nodes, p, seed=400)

train_seqs = [train_gen.generate_sequence() for _ in range(N_TRAIN)]
test_seqs = [test_gen.generate_sequence() for _ in range(N_TEST)]

# Compute average demand for the fluid LP
avg_demand = defaultdict(float)
for seq in train_seqs:
    for req in seq.requests:
        avg_demand[req.demand_node] += 1.0 / N_TRAIN

optimizer = InventoryOptimizer(graph, solver=SOLVER)
balance = BalancePolicy(graph)


def evaluate_placement(inv, name):
    rewards = [balance.fulfill(seq, inv).collected_rewards for seq in test_seqs]
    avg = sum(rewards) / len(rewards)
    return avg


# --- 1. Uniform placement ---
units_per_node = TOTAL_INVENTORY // N_SUPPLY
remainder = TOTAL_INVENTORY % N_SUPPLY
uniform_inv = {i: units_per_node + (1 if i < remainder else 0) for i in range(N_SUPPLY)}
inv_uniform = Inventory(uniform_inv, 'uniform')

# --- 2. Fluid LP rounding ---
inv_fluid = optimizer.fluid_inventory_placement_rounding(avg_demand, TOTAL_INVENTORY)

# --- 3. Offline LP rounding ---
inv_offline = optimizer.offline_inventory_placement_rounding(train_seqs, TOTAL_INVENTORY)

# --- Results ---
placements = [
    ("Uniform", inv_uniform),
    ("Fluid LP Rounding", inv_fluid),
    ("Offline LP Rounding", inv_offline),
]

print(f"Instance: {N_SUPPLY} warehouses, {N_DEMAND} regions, T={T}")
print(f"Total inventory budget: {TOTAL_INVENTORY}")
print(f"Evaluated with Balance policy on {N_TEST} test sequences\n")

print(f"{'Strategy':<25} {'Allocation':<30} {'Avg Reward':>10}")
print("-" * 67)
for name, inv in placements:
    alloc = {k: v for k, v in inv.initial_inventory.items()}
    avg = evaluate_placement(inv, name)
    print(f"{name:<25} {str(alloc):<30} {avg:>10.2f}")
