# Fulfillment Optimization

## Project Overview

This project implements and benchmarks algorithms for **online bipartite matching** applied to fulfillment optimization. It models the problem of assigning demand (customer orders) to supply nodes (warehouses) in real-time, where demand arrives sequentially and fulfillment decisions are irrevocable.

The core research question: how do different fulfillment policies perform under various demand models, and how does data availability (training sample size) affect data-driven approaches?

## Architecture

The library is packaged as `fulfillment_optimization` under `src/`. Paper-specific experiments, plots, and documents live in `paper/` (gitignored, never committed).

### Package Modules (`src/fulfillment_optimization/`)

**Core data structures:**
- **graph.py** — Bipartite graph data structures (`Graph`, `Node`, `DemandNode`, `Edge`) and random graph generation
- **demand.py** — Demand sequence generation under various stochastic models (independent, Markov, random walk, HMM, correlated)

**Optimization:**
- **solver.py** — LP solver dispatch layer supporting Gurobi and HiGHS backends
- **lp.py** — LP formulations (`MathPrograms`, `LPBuilder`, `LPResult`) for fluid and offline linear programs
- **inventory.py** — `Inventory` data class and `InventoryOptimizer` for LP-based placement strategies

**Fulfillment policies (`policies/` subpackage):**
- **policies/base.py** — `FulfillmentResult` dataclass (supports tuple unpacking) and `extended_division` helper
- **policies/priority_list.py** — `PriorityListPolicy` (alias: `Fulfillment`) — follow precomputed priority lists
- **policies/balance.py** — `BalancePolicy`, `MultiPriceBalancePolicy` — online balance algorithms
- **policies/resolving.py** — `LpReSolvingPolicy` base class with `OffLpReSolvingPolicy`, `FluLpReSolvingPolicy`, `ExtrapolationLpReSolvingPolicy` — template-method pattern for LP re-solving
- **policies/dual.py** — `DualMirrorDescentPolicy` — online dual variable updates
- **policies/dp.py** — `DPPolicy` (alias: `PolicyFulfillment`) — execute precomputed DP policies
- **policies/learned.py** — `ThresholdsPolicy`, `TimeSupplyEnhancedMPB`, etc. — Nevergrad/PyTorch-based learned policies (requires `[ml]` extras)

**Model-based methods:**
- **dp.py** — `IndependentDynamicProgram`, `MarkovianDynamicProgram` — unified DP with pluggable demand transition model
- **estimation.py** — `ModelEstimator` — estimate IID, independent, or Markovian demand parameters from data

**Utilities:**
- **utils.py** — `correl_graph` helper for constructing correlated demand experiments

**Backward compatibility shims** (re-export from new locations):
- `fulfillment.py`, `math_programs.py`, `model_based.py`, `model_free.py`

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
from fulfillment_optimization import Graph, Inventory, BalancePolicy, MathPrograms
```

## Conventions

- Supply nodes are indexed by `int`, demand nodes by `int`
- Edges are keyed by `(supply_node_id, demand_node_id)` tuples
- Inventory is tracked as `Dict[int, int]` mapping supply node ID to units available
- Sequences are lists of `Request` objects wrapped in a `Sequence`
- Policy output: `FulfillmentResult` dataclass (supports tuple unpacking as `(number_fulfillments, collected_rewards, lost_sales)`)
- Seeds are used throughout for reproducibility: separate seeds for graph generation, distribution generation, training samples, and test samples
