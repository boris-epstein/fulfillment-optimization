"""Showcase the different demand generation models.

Generates sequences from each demand model (IID / correlated, temporally
independent, Markov chain, random walk, hidden Markov model) and prints
summary statistics.
"""

import numpy as np

from fulfillment_optimization import (
    RandomGraphGenerator, RandomDistributionGenerator,
    CorrelGenerator, TemporalIndependenceGenerator,
    IndepGenerator, RWGenerator,
    MarkovianGenerator, HiddenMarkovGenerator,
    BalancePolicy, Inventory,
)

# --- Shared setup ---
N_SUPPLY, N_DEMAND = 3, 4
N_SEQS = 30

graph = RandomGraphGenerator(seed=5).two_valued_vertex_graph(N_SUPPLY, N_DEMAND)
demand_nodes = list(graph.demand_nodes.keys())
balance = BalancePolicy(graph)
inventory = Inventory({i: 5 for i in range(N_SUPPLY)}, name='uniform')


def summarize(name, sequences):
    lengths = [len(seq) for seq in sequences]
    rewards = [balance.fulfill(seq, inventory).collected_rewards for seq in sequences]
    print(f"  {name}")
    print(f"    Avg length: {np.mean(lengths):.1f}  (min={min(lengths)}, max={max(lengths)})")
    print(f"    Avg reward: {np.mean(rewards):.2f}")


print(f"Graph: {N_SUPPLY} warehouses, {N_DEMAND} regions")
print(f"Inventory: 5 units per warehouse")
print(f"Each model generates {N_SEQS} sequences\n")

# --- 1. Correlated (IID with random total length) ---
weights = [1.0 / N_DEMAND] * N_DEMAND
correl_gen = CorrelGenerator(
    mean=12, demand_nodes=demand_nodes, weights=weights,
    seed=10, distribution='geometric',
)
seqs_correl = [correl_gen.generate_sequence() for _ in range(N_SEQS)]
summarize("Correlated (geometric length)", seqs_correl)

# Deterministic length variant
correl_det = CorrelGenerator(
    mean=12, demand_nodes=demand_nodes, weights=weights,
    seed=11, distribution='deterministic',
)
seqs_det = [correl_det.generate_sequence() for _ in range(N_SEQS)]
summarize("Correlated (deterministic length)", seqs_det)

# --- 2. Temporally independent ---
dist_gen = RandomDistributionGenerator(seed=20)
T = 12
p = dist_gen.generate_indep(N_DEMAND, T)
temp_gen = TemporalIndependenceGenerator(demand_nodes, p, seed=30)
seqs_temp = [temp_gen.generate_sequence() for _ in range(N_SEQS)]
summarize("Temporally independent", seqs_temp)

# --- 3. Independent per-node counts ---
means = [3.0] * N_DEMAND  # mean demand per node
indep_gen = IndepGenerator(means, demand_nodes, seed=40, distribution='geometric')
seqs_indep = [indep_gen.generate_sequence() for _ in range(N_SEQS)]
summarize("Independent per-node (geometric)", seqs_indep)

# --- 4. Random walk ---
rw_gen = RWGenerator(mean=T, demand_nodes=demand_nodes, seed=50,
                     distribution='deterministic', step_size=1)
seqs_rw = [rw_gen.generate_sequence() for _ in range(N_SEQS)]
summarize("Random walk", seqs_rw)

# --- 5. Markov chain ---
dist_gen2 = RandomDistributionGenerator(seed=60)
init_dist, trans_matrix = dist_gen2.generate_markov(N_DEMAND, diagonal_bias=2)
markov_gen = MarkovianGenerator(T, demand_nodes, trans_matrix, init_dist, seed=70)
seqs_markov = [markov_gen.generate_sequence() for _ in range(N_SEQS)]
summarize("Markov chain", seqs_markov)

# --- 6. Hidden Markov model ---
N_STATES = 2
states = list(range(N_STATES))
state_dist = {
    0: [0.6, 0.2, 0.1, 0.1],  # state 0 favors demand node 0
    1: [0.1, 0.1, 0.2, 0.6],  # state 1 favors demand node 3
}
hmm_init = [0.5, 0.5]
hmm_trans = np.array([[0.8, 0.2],
                      [0.3, 0.7]])
hmm_gen = HiddenMarkovGenerator(T, demand_nodes, states, state_dist,
                                 hmm_init, hmm_trans, seed=80)
seqs_hmm = [hmm_gen.generate_sequence() for _ in range(N_SEQS)]
summarize("Hidden Markov model", seqs_hmm)
