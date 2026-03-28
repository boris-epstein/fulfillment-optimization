"""Compare all major fulfillment policy families on the same instance.

Runs Balance, MultiPriceBalance, LP re-solving (Fluid, Offline, Extrapolation),
and DP-optimal on a small graph with temporally independent demand.
"""

from collections import defaultdict

from fulfillment_optimization import (
    RandomGraphGenerator, RandomDistributionGenerator,
    TemporalIndependenceGenerator, ModelEstimator,
    Inventory, MathPrograms,
    BalancePolicy, MultiPriceBalancePolicy,
    PriorityListPolicy,
    FluLpReSolvingPolicy, OffLpReSolvingPolicy, ExtrapolationLpReSolvingPolicy,
    IndependentDynamicProgram, DPPolicy,
)

# --- Instance setup ---
N_SUPPLY, N_DEMAND, T = 3, 4, 12
N_TRAIN, N_TEST = 50, 30
SOLVER = 'highs'

graph = RandomGraphGenerator(seed=42).two_valued_vertex_graph(N_SUPPLY, N_DEMAND)
demand_nodes = list(graph.demand_nodes.keys())

dist_gen = RandomDistributionGenerator(seed=10)
p = dist_gen.generate_indep(N_DEMAND, T)
train_gen = TemporalIndependenceGenerator(demand_nodes, p, seed=100)
test_gen = TemporalIndependenceGenerator(demand_nodes, p, seed=200)

train_seqs = [train_gen.generate_sequence() for _ in range(N_TRAIN)]
test_seqs = [test_gen.generate_sequence() for _ in range(N_TEST)]

inventory = Inventory({i: 4 for i in range(N_SUPPLY)}, name='uniform')

# Precompute average demand for fluid LP
estimator = ModelEstimator(graph)
p_hat = estimator.estimate_independent(train_seqs)

# Cumulative average demand from time t onward
cumulative_avg = {}
for t in range(T + 1):
    cumulative_avg[t] = defaultdict(float)
    for seq in train_seqs:
        for tau in range(t, T):
            cumulative_avg[t][seq.requests[tau].demand_node] += 1.0 / N_TRAIN

# Zero dual solution for initialization
zero_duals = {i: 0.0 for i in graph.supply_nodes}

# Re-solving epochs: re-solve at the start and midpoint
re_solving_epochs = [0, T // 2]

# --- Myopic ---
graph.populate_best_supply_nodes()
for dn in graph.demand_nodes.values():
    dn.priority_lists['myopic'] = list(dn.best_supply_nodes)
myopic = PriorityListPolicy(graph)

# --- Policies ---
results = {}


def evaluate(name, run_fn):
    rewards = [run_fn(seq) for seq in test_seqs]
    avg = sum(rewards) / len(rewards)
    results[name] = avg


# Myopic
evaluate("Myopic", lambda seq: myopic.fixed_list_fulfillment(seq, inventory, 'myopic').collected_rewards)

# Balance
balance = BalancePolicy(graph)
evaluate("Balance", lambda seq: balance.fulfill(seq, inventory).collected_rewards)

# Multi-Price Balance
mpb = MultiPriceBalancePolicy(graph)
evaluate("MultiPriceBalance", lambda seq: mpb.fulfill(seq, inventory).collected_rewards)

# Fluid LP Re-solving
flu = FluLpReSolvingPolicy(graph, solver=SOLVER)
evaluate("Fluid LP Re-solving", lambda seq: flu.fulfill(
    seq, inventory, zero_duals, cumulative_avg, re_solving_epochs
).collected_rewards)

# Offline LP Re-solving
off = OffLpReSolvingPolicy(graph, solver=SOLVER)
evaluate("Offline LP Re-solving", lambda seq: off.fulfill(
    seq, inventory, zero_duals, train_seqs, re_solving_epochs
).collected_rewards)

# Extrapolation LP Re-solving
ext = ExtrapolationLpReSolvingPolicy(graph, solver=SOLVER)
evaluate("Extrapolation LP Re-solving", lambda seq: ext.fulfill(
    seq, inventory, zero_duals, re_solving_epochs
).collected_rewards)

# DP Optimal (feasible for small instances)
dp_solver = IndependentDynamicProgram(graph)
dp_output = dp_solver.compute_optimal_policy(inventory, T, p)
dp_policy = DPPolicy(graph)
evaluate("DP Optimal", lambda seq: dp_policy.fulfill(seq, inventory, dp_output.optimal_action).collected_rewards)

# --- Print results ---
print(f"\nInstance: {N_SUPPLY} warehouses, {N_DEMAND} regions, T={T}")
print(f"Inventory: {inventory.total_inventory} total units")
print(f"Evaluated on {N_TEST} test sequences\n")
print(f"{'Policy':<30} {'Avg Reward':>10}")
print("-" * 42)
for name, avg in results.items():
    print(f"{name:<30} {avg:>10.2f}")
