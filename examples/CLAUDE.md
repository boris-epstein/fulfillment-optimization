# Examples

Runnable tutorial scripts demonstrating the `fulfillment_optimization` package.

## Scripts

- **quickstart.py** — Minimal example: build a graph, generate demand, compare Balance vs Myopic policies
- **comparing_policies.py** — Run all major policy families (Myopic, Balance, MultiPriceBalance, LP re-solving variants, DP optimal) on the same instance
- **inventory_placement.py** — Compare inventory placement strategies (uniform, fluid LP rounding, offline LP rounding)
- **demand_models.py** — Showcase all demand generators (correlated, temporally independent, independent per-node, random walk, Markov, HMM)

## Running

All examples use the HiGHS solver (no Gurobi license required):

```bash
pip install -e ".[highs]"
python examples/quickstart.py
```

## Conventions

- Each script is self-contained — no shared state or data files
- Scripts print human-readable output to stdout
- Small instance sizes so they run in seconds
- Seeds are fixed for reproducibility
