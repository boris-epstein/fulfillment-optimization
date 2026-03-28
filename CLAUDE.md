# Fulfillment Optimization

## Project Overview

This project implements and benchmarks algorithms for **online bipartite matching** applied to fulfillment optimization. It models the problem of assigning demand (customer orders) to supply nodes (warehouses) in real-time, where demand arrives sequentially and fulfillment decisions are irrevocable.

The core research question: how do different fulfillment policies perform under various demand models, and how does data availability (training sample size) affect data-driven approaches?

## Architecture

The library is packaged as `fulfillment_optimization` under `src/`. Paper-specific experiments, plots, and documents live in `paper/` (gitignored, never committed).

### Package Modules (`src/fulfillment_optimization/`)

- **graph.py** — Bipartite graph data structures (`Graph`, `Node`, `DemandNode`, `Edge`) and random graph generation
- **demand.py** — Demand sequence generation under various stochastic models (independent, Markov, random walk, HMM, correlated)
- **fulfillment.py** — Core fulfillment policies: myopic, balance, multi-price balance, LP re-solving (fluid, offline, extrapolation), dual mirror descent, and policy-based fulfillment
- **math_programs.py** — LP formulations (fluid LP, offline LP) with a solver abstraction layer supporting Gurobi and HiGHS backends
- **model_based.py** — Dynamic programming solutions (independent DP, Markovian DP) and distribution estimation from samples
- **model_free.py** — Parametrized policies optimized via Nevergrad: threshold-based, enhanced balance variants, neural opportunity cost (requires `[ml]` extras)
- **utils.py** — Helper for constructing the "correlated" demand graph (HMM-based)

### Paper Files (`paper/`, gitignored)

- `paper_experiments.py`, `experiments.py` — Experiment scripts
- `plotter.py`, `patch_fluid.py` — Plotting and patching utilities
- `*.tex`, `*.md` — Paper documents
- `figures/`, `results/` — Generated outputs

## Dependencies

Core: `numpy`

Optional (install via extras):
- `pip install fulfillment-optimization[gurobi]` — Gurobi LP solver
- `pip install fulfillment-optimization[highs]` — HiGHS LP solver (free/open-source)
- `pip install fulfillment-optimization[ml]` — Nevergrad + PyTorch for model-free policies
- `pip install fulfillment-optimization[all]` — everything

## Installation

```bash
pip install -e .           # core only
pip install -e ".[all]"    # with all optional dependencies
```

## Usage

```python
from fulfillment_optimization import Graph, Inventory, Fulfillment, MathPrograms
```

## Conventions

- Supply nodes are indexed by `int`, demand nodes by `int`
- Edges are keyed by `(supply_node_id, demand_node_id)` tuples
- Inventory is tracked as `Dict[int, int]` mapping supply node ID to units available
- Sequences are lists of `Request` objects wrapped in a `Sequence`
- Policy output convention: `(number_fulfillments, collected_rewards, lost_sales)`
- Seeds are used throughout for reproducibility: separate seeds for graph generation, distribution generation, training samples, and test samples
