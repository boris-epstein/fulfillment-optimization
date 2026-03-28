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

| Policy | Description | Reference |
|--------|-------------|-----------|
| **Myopic** (priority list) | Greedy: assign to the highest-reward available supply node | |
| **Multi-Price Balance** | Penalizes heavily-used supply nodes with a piecewise-exponential cost adapted to the reward structure | [Ma and Simchi-Levi (2020)](https://doi.org/10.1287/opre.2020.1994) |
| **Fluid LP Re-solving** | Periodically re-solves a fluid LP for dual-based opportunity costs | [Gallego and Van Ryzin (1997)](https://doi.org/10.1287/opre.45.1.24) |
| **Offline LP Re-solving** | Re-solves an offline LP over training samples | [DeValve et al. (2023)](https://doi.org/10.1287/msom.2022.1169) |
| **Extrapolation LP Re-solving** | Re-solves using extrapolated observed demand (no training data needed at runtime) | |
| **Dual Mirror Descent** | Updates opportunity costs online via subgradient or multiplicative weights | [Balseiro et al. (2023)](https://doi.org/10.1287/opre.2021.2242) |
| **DP Optimal** | Exact dynamic programming (feasible for small instances) | |
| **Enhanced Multi-Price Balance** | Multi-Price Balance with learnable time-decay and per-resource offset parameters | [Epstein and Ma (2025)](#references) |
| **Neural Opportunity Cost** | Neural network that outputs per-resource opportunity costs from inventory, time, and demand features | [Epstein and Ma (2025)](#references) |

## Demand models

- **Correlated** -- IID node sampling with random total length (geometric, normal, exponential, or deterministic); based on [Aouad and Ma (2022)](https://arxiv.org/abs/2208.02229)
- **Temporally independent** -- independent distribution over demand nodes at each time step; see [Gallego and Van Ryzin (1994)](https://doi.org/10.1287/mnsc.40.8.999)
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

## References

- B. Epstein and W. Ma. *Data-driven Online Matching Simulations*. Working paper, 2025.
- W. Ma and D. Simchi-Levi. [Algorithms for Online Matching, Assortment, and Pricing with Tight Weight-Dependent Competitive Ratios](https://doi.org/10.1287/opre.2020.1994). *Operations Research*, 68(6):1787--1803, 2020.
- L. DeValve, Y. Wei, D. Wu, and R. Yuan. [Understanding the Value of Fulfillment Flexibility in an Online Retailing Environment](https://doi.org/10.1287/msom.2022.1169). *Manufacturing & Service Operations Management*, 25(2):391--408, 2023.
- S. Balseiro, H. Lu, and V. Mirrokni. [The Best of Many Worlds: Dual Mirror Descent for Online Allocation Problems](https://doi.org/10.1287/opre.2021.2242). *Operations Research*, 71(1):101--119, 2023.
- G. Gallego and G. Van Ryzin. [A Multiproduct Dynamic Pricing Problem and Its Applications to Network Yield Management](https://doi.org/10.1287/opre.45.1.24). *Operations Research*, 45(1):24--41, 1997.
- G. Gallego and G. Van Ryzin. [Optimal Dynamic Pricing of Inventories with Stochastic Demand over Finite Horizons](https://doi.org/10.1287/mnsc.40.8.999). *Management Science*, 40(8):999--1020, 1994.
- A. Aouad and W. Ma. [A Nonparametric Framework for Online Stochastic Matching with Correlated Arrivals](https://arxiv.org/abs/2208.02229). arXiv preprint, 2022.

## Authors

Boris Epstein
