# Fulfillment Optimization

Algorithms for **online bipartite matching** applied to fulfillment optimization. Models the problem of assigning demand (customer orders) to supply nodes (warehouses) in real-time, where demand arrives sequentially and fulfillment decisions are irrevocable.

## Installation

```bash
pip install -e .
```

The base install requires only NumPy. For LP-based policies, install a solver backend:

```bash
pip install -e ".[highs]"    # free/open-source HiGHS solver
pip install -e ".[gurobi]"   # commercial Gurobi solver (requires license)
```

For model-free learned policies (Nevergrad + PyTorch):

```bash
pip install -e ".[ml]"
```

## Quick start

```python
from fulfillment_optimization import (
    RandomGraphGenerator, RandomDistributionGenerator,
    TemporalIndependenceGenerator,
    Inventory, BalancePolicy,
)

# Build a random bipartite graph (3 warehouses, 4 demand regions)
graph = RandomGraphGenerator(seed=42).two_valued_vertex_graph(3, 4)

# Generate demand sequences
demand_nodes = list(graph.demand_nodes.keys())
p = RandomDistributionGenerator(seed=10).generate_indep(4, 12)
gen = TemporalIndependenceGenerator(demand_nodes, p, seed=99)
sequences = [gen.generate_sequence() for _ in range(20)]

# Place inventory and run the Balance policy
inventory = Inventory({0: 4, 1: 4, 2: 4})
balance = BalancePolicy(graph)
for seq in sequences:
    result = balance.fulfill(seq, inventory)
    print(f"Reward: {result.collected_rewards:.2f}, Lost sales: {result.lost_sales}")
```

See the [`examples/`](examples/) directory for more detailed tutorials.

## Fulfillment policies

| Policy | Description |
|--------|-------------|
| **Myopic** (priority list) | Greedy: assign to the highest-reward available supply node |
| **Balance** | Penalizes heavily-used supply nodes to maintain balance |
| **Multi-Price Balance** | Generalizes Balance for heterogeneous edge rewards |
| **Fluid LP Re-solving** | Periodically re-solves a fluid LP for dual-based opportunity costs |
| **Offline LP Re-solving** | Re-solves an offline LP over training samples |
| **Extrapolation LP Re-solving** | Re-solves using extrapolated observed demand (no training data needed at runtime) |
| **Dual Mirror Descent** | Updates opportunity costs online via subgradient or multiplicative weights |
| **DP Optimal** | Exact dynamic programming (feasible for small instances) |
| **Learned policies** | Threshold, enhanced balance, and neural opportunity cost variants trained via Nevergrad |

## Demand models

- **Correlated** -- IID node sampling with random total length (geometric, normal, exponential, or deterministic)
- **Temporally independent** -- independent distribution over demand nodes at each time step
- **Independent per-node** -- each demand node has an independent random count
- **Random walk** -- demand probabilities evolve via multiplicative random walk
- **Markov chain** -- demand nodes follow a Markov chain
- **Hidden Markov model** -- demand driven by latent Markov states with emission distributions

## Inventory placement

`InventoryOptimizer` provides several placement strategies given a total inventory budget:

- Fluid LP rounding
- Offline LP rounding
- Fluid greedy (shadow-price based)
- Offline greedy
- Myopic greedy

## Project structure

```
src/fulfillment_optimization/
    graph.py          -- bipartite graph data structures
    demand.py         -- demand sequence generators
    inventory.py      -- inventory placement optimization
    lp.py             -- LP formulations (fluid, offline)
    solver.py         -- solver backend dispatch (Gurobi, HiGHS)
    dp.py             -- dynamic programming solvers
    estimation.py     -- demand model estimation from samples
    policies/
        base.py           -- FulfillmentResult, utilities
        priority_list.py  -- myopic / priority-list policy
        balance.py        -- Balance and Multi-Price Balance
        resolving.py      -- LP re-solving policies (fluid, offline, extrapolation)
        dual.py           -- dual mirror descent
        dp.py             -- DP policy executor
        learned.py        -- model-free learned policies (requires ml extras)
examples/             -- runnable tutorial scripts
```

## Authors

Boris Epstein
