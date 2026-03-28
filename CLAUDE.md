# Fulfillment Optimization

## Architecture

The library is packaged as `fulfillment_optimization` under `src/`. Paper-specific experiments, plots, and documents live in `paper/` (gitignored, never committed).

### Package Modules (`src/fulfillment_optimization/`)

**Core data structures:**
- **graph.py** — Bipartite graph (`Graph`, `Node`, `DemandNode`, `Edge`) and `RandomGraphGenerator`
- **demand.py** — Demand sequence generators (correlated, temporally independent, independent, random walk, Markov, HMM)

**Optimization:**
- **solver.py** — LP solver dispatch layer (Gurobi, HiGHS)
- **lp.py** — LP formulations (`MathPrograms`, `LPBuilder`, `LPResult`)
- **inventory.py** — `Inventory` data class and `InventoryOptimizer`

**Fulfillment policies (`policies/` subpackage):**
- **base.py** — `FulfillmentResult` dataclass (supports tuple unpacking), `extended_division`
- **priority_list.py** — `PriorityListPolicy` (alias: `Fulfillment`)
- **balance.py** — `BalancePolicy`, `MultiPriceBalancePolicy`
- **resolving.py** — `LpReSolvingPolicy` base with `OffLpReSolvingPolicy`, `FluLpReSolvingPolicy`, `ExtrapolationLpReSolvingPolicy` (template-method pattern)
- **dual.py** — `DualMirrorDescentPolicy`
- **dp.py** — `DPPolicy` (alias: `PolicyFulfillment`)
- **learned.py** — Nevergrad/PyTorch-based learned policies (requires `[ml]` extras)

**Model-based methods:**
- **dp.py** (top-level) — `IndependentDynamicProgram`, `MarkovianDynamicProgram` — unified DP with pluggable demand transition
- **estimation.py** — `ModelEstimator` — estimate IID, independent, or Markovian demand from data

**Utilities:**
- **utils.py** — `correl_graph` helper for correlated demand experiments

**Backward compatibility shims** (thin re-exports):
- `fulfillment.py`, `math_programs.py`, `model_based.py`, `model_free.py`

### Paper Files (`paper/`, gitignored)

- `paper_experiments.py`, `experiments.py` — Experiment scripts
- `plotter.py`, `patch_fluid.py` — Plotting and patching utilities

## Conventions

- Supply/demand nodes indexed by `int`; edges keyed by `(supply_node_id, demand_node_id)` tuples
- Inventory: `Dict[int, int]` mapping supply node ID to units
- Sequences: lists of `Request` objects wrapped in `Sequence`
- Policy output: `FulfillmentResult` (tuple-unpackable as `(number_fulfillments, collected_rewards, lost_sales)`)
- Seeds used throughout for reproducibility: separate seeds for graph, distribution, training, and test
