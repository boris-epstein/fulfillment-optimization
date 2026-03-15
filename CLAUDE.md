# Fulfillment Optimization

## Project Overview

This project implements and benchmarks algorithms for **online bipartite matching** applied to fulfillment optimization. It models the problem of assigning demand (customer orders) to supply nodes (warehouses) in real-time, where demand arrives sequentially and fulfillment decisions are irrevocable.

The core research question: how do different fulfillment policies perform under various demand models, and how does data availability (training sample size) affect data-driven approaches?

## Architecture

All source code lives in `Code/`. There is no package structure — modules import each other directly.

### Key Modules

- **Graph.py** — Bipartite graph data structures (`Graph`, `Node`, `DemandNode`, `Edge`) and random graph generation
- **Demand.py** — Demand sequence generation under various stochastic models (independent, Markov, random walk, HMM, correlated)
- **FulfillmentOptimization.py** — Core fulfillment policies: myopic, balance, multi-price balance, LP re-solving (fluid, offline, extrapolation), dual mirror descent, and policy-based fulfillment
- **MathPrograms.py** — Gurobi LP formulations (fluid LP, offline LP) used by re-solving policies
- **ModelBased.py** — Dynamic programming solutions (independent DP, Markovian DP) and distribution estimation from samples
- **ModelFree.py** — Parametrized policies optimized via Nevergrad: threshold-based, enhanced balance variants, neural opportunity cost
- **experiments.py** — Experiment orchestration: instance generation, parallel execution, result collection, CSV output
- **utils.py** — Helper for constructing the "correlated" demand graph (HMM-based)

## Dependencies

- `numpy` — numerical computation
- `gurobipy` — LP/optimization solver (requires Gurobi license)
- `sortedcontainers` — sorted data structures
- `nevergrad` — derivative-free optimization (model-free policy training)
- `torch` — neural network policies
- `matplotlib` — plotting (used only in `__main__` blocks)

## Running

```bash
cd Code
python experiments.py
```

The `main()` function in `experiments.py` configures and runs the full experiment. Key parameters are set at the top of `main()`:
- `demand_model`: one of `'indep'`, `'markov'`, `'rw'`, `'correl'`
- `n_supply_nodes`, `n_demand_nodes`: graph size
- `train_sample_sizes`: list of training set sizes to evaluate
- `parallel`: whether to use multiprocessing

Results are written to a CSV file and logs go to `logs/`.

## Conventions

- Supply nodes are indexed by `int`, demand nodes by `int`
- Edges are keyed by `(supply_node_id, demand_node_id)` tuples
- Inventory is tracked as `Dict[int, int]` mapping supply node ID to units available
- Sequences are lists of `Request` objects wrapped in a `Sequence`
- Policy output convention: `(number_fulfillments, collected_rewards, lost_sales)`
- Seeds are used throughout for reproducibility: separate seeds for graph generation, distribution generation, training samples, and test samples
