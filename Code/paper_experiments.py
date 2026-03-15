"""Experiments for the inventory placement and fulfillment paper revision.

Three experiments:
  1. d-sensitivity: How does the degree parameter d affect placement quality
     across different load factors?
  2. SAA convergence: How many samples K does the Offline placement need as
     instance size grows?
  3. Placement x fulfillment interaction: Does Offline Placement's advantage
     depend on fulfillment policy quality?

Usage:
    python paper_experiments.py --experiment {1,2,3} [--seed SEED] [--output_dir DIR]
"""

import argparse
import csv
import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from Graph import Graph
from Demand import TemporalIndependenceGenerator, Sequence, Request
from MathPrograms import MathPrograms
from FulfillmentOptimization import (
    Inventory, InventoryOptimizer, Fulfillment,
    OffLpReSolvingFulfillment, FluLpReSolvingFulfillment,
)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_d_regular_graph(n: int, m: int, d: int, rng: np.random.Generator) -> Graph:
    """Build a random d-regular bipartite graph.

    Each demand node connects to exactly d distinct warehouses chosen uniformly
    at random, with edge rewards drawn from Uniform(0, 1).

    Args:
        n: Number of supply nodes (warehouses).
        m: Number of demand nodes.
        d: Degree — each demand node connects to exactly d supply nodes.
        rng: Numpy random generator for reproducibility.

    Returns:
        A Graph instance with the constructed edges.
    """
    graph = Graph()
    for i in range(n):
        graph.add_supply_node(i)
    for j in range(m):
        graph.add_demand_node(j)

    supply_ids = list(range(n))
    for j in range(m):
        neighbors = rng.choice(supply_ids, size=d, replace=False)
        for i in neighbors:
            reward = rng.uniform(0, 1)
            graph.add_edge(i, j, reward)

    graph.populate_neighbors()
    return graph


# ---------------------------------------------------------------------------
# Demand generation (DH-TI: Deterministic-Horizon Temporal Independence)
# ---------------------------------------------------------------------------

def make_dh_ti_generator(demand_nodes: List[int], T: int, weights: np.ndarray,
                         seed: int) -> TemporalIndependenceGenerator:
    """Create a DH-TI demand generator with uniform per-period probabilities.

    Args:
        demand_nodes: List of demand node IDs.
        T: Fixed time horizon (number of arrivals).
        weights: Probability vector over demand nodes (same every period).
        seed: Random seed.

    Returns:
        A TemporalIndependenceGenerator instance.
    """
    p = {t: list(weights) for t in range(T)}
    return TemporalIndependenceGenerator(demand_nodes, p, seed)


def generate_samples(generator: TemporalIndependenceGenerator, n_samples: int) -> List[Sequence]:
    """Generate a list of demand sequences from a generator."""
    return [generator.generate_sequence() for _ in range(n_samples)]


# ---------------------------------------------------------------------------
# Placement procedures
# ---------------------------------------------------------------------------

def gandhi_round(fractional_placement: Dict[int, float], total_inventory: int,
                 rng: np.random.Generator) -> Dict[int, int]:
    """Round a fractional inventory placement using dependent rounding (Gandhi et al., 2006).

    Implements the star-graph special case of the dependent rounding algorithm.
    The fractional parts of the placement are rounded to 0 or 1 such that:
      (P1) E[R_i] = x_i for each warehouse i,
      (P2) sum of rounded values = total_inventory with probability 1,
      (P3) negative correlation across warehouses.

    The algorithm works on the fractional layer only. It pairs up floating
    (non-integer) entries two at a time and randomly pushes one up and the
    other down, preserving marginals and the sum, until all entries are integral.

    Args:
        fractional_placement: Dict mapping supply node ID to fractional inventory.
        total_inventory: Total units that must be placed (sum constraint).
        rng: Random generator for the randomized rounding.

    Returns:
        Dict mapping supply node ID to integer inventory.
    """
    # Separate integer floors and fractional parts
    keys = list(fractional_placement.keys())
    floors = {s: int(np.floor(fractional_placement[s])) for s in keys}
    fracs = {s: fractional_placement[s] - floors[s] for s in keys}

    # Collect indices with non-zero fractional parts (the "floating" edges)
    floating = [s for s in keys if 0 < fracs[s] < 1]

    # Dependent rounding on the fractional layer (star graph case):
    # In each iteration, pick two floating entries, and randomly push one
    # up and one down so that at least one becomes 0 or 1.
    while len(floating) >= 2:
        # Pick a pair (in a star graph, any pair works)
        s1, s2 = floating[0], floating[1]
        f1, f2 = fracs[s1], fracs[s2]

        # alpha: amount to increase s1 / decrease s2 until one hits a boundary
        # beta:  amount to decrease s1 / increase s2 until one hits a boundary
        alpha = min(1 - f1, f2)
        beta = min(f1, 1 - f2)

        # Randomized step preserving marginals
        if rng.random() < beta / (alpha + beta):
            fracs[s1] = f1 + alpha
            fracs[s2] = f2 - alpha
        else:
            fracs[s1] = f1 - beta
            fracs[s2] = f2 + beta

        # Snap to 0/1 within tolerance and remove rounded entries
        floating_next = []
        for s in floating:
            if fracs[s] < 1e-12:
                fracs[s] = 0.0
            elif fracs[s] > 1 - 1e-12:
                fracs[s] = 1.0
            if 0 < fracs[s] < 1:
                floating_next.append(s)
        floating = floating_next

    # At most one floating entry can remain; its fractional part must be 0 or 1
    # due to the degree-preservation property (sum of fracs is integer).
    for s in floating:
        fracs[s] = round(fracs[s])

    # Combine floors and rounded fractional parts
    rounded = {s: floors[s] + int(round(fracs[s])) for s in keys}

    # Sanity check
    assert sum(rounded.values()) == total_inventory, \
        f"Gandhi rounding sum mismatch: {sum(rounded.values())} != {total_inventory}"

    return rounded


def offline_placement(graph: Graph, demand_samples: List[Sequence],
                      total_inventory: int, rng: np.random.Generator,
                      solver: str = 'highs') -> Inventory:
    """Compute Offline Placement: optimize the Offline surrogate via LP + rounding.

    Args:
        graph: Bipartite graph.
        demand_samples: SAA samples of demand sequences.
        total_inventory: Total inventory budget Q.
        rng: Random generator for rounding.
        solver: LP solver backend.

    Returns:
        An Inventory object with the rounded placement.
    """
    programs = MathPrograms(graph, solver=solver)
    result, x_vars = programs.offline_linear_program_variable_inventory(
        demand_samples, total_inventory
    )

    fractional = {s: x_vars[s].X for s in graph.supply_nodes}
    rounded = gandhi_round(fractional, total_inventory, rng)
    return Inventory(rounded, 'offline')


def fluid_placement(graph: Graph, average_demand: Dict[int, float],
                    total_inventory: int, rng: np.random.Generator,
                    solver: str = 'highs') -> Inventory:
    """Compute Fluid Placement: optimize the Fluid surrogate via LP + rounding.

    Args:
        graph: Bipartite graph.
        average_demand: Expected demand per demand node.
        total_inventory: Total inventory budget Q.
        rng: Random generator for rounding.
        solver: LP solver backend.

    Returns:
        An Inventory object with the rounded placement.
    """
    programs = MathPrograms(graph, solver=solver)
    result, x_vars = programs.fluid_linear_program_variable_inventory(
        average_demand, total_inventory
    )

    fractional = {s: x_vars[s].X for s in graph.supply_nodes}
    rounded = gandhi_round(fractional, total_inventory, rng)
    return Inventory(rounded, 'fluid')


def scaled_fluid_placement(graph: Graph, average_demand: Dict[int, float],
                           total_inventory: int, rng: np.random.Generator,
                           solver: str = 'highs') -> Inventory:
    """Compute Scaled-Demand Fluid Placement.

    Scales expected demand so that total expected demand equals total inventory,
    then solves the Fluid LP and rounds.

    Args:
        graph: Bipartite graph.
        average_demand: Expected demand per demand node.
        total_inventory: Total inventory budget Q.
        rng: Random generator for rounding.
        solver: LP solver backend.

    Returns:
        An Inventory object with the rounded placement.
    """
    total_demand = sum(average_demand.values())
    scale = total_inventory / total_demand if total_demand > 0 else 1.0
    scaled_demand = {j: avg * scale for j, avg in average_demand.items()}

    programs = MathPrograms(graph, solver=solver)
    result, x_vars = programs.fluid_linear_program_variable_inventory(
        scaled_demand, total_inventory
    )

    fractional = {s: x_vars[s].X for s in graph.supply_nodes}
    rounded = gandhi_round(fractional, total_inventory, rng)
    return Inventory(rounded, 'scaled_fluid')


def myopic_placement(graph: Graph, demand_samples: List[Sequence],
                     total_inventory: int,
                     solver: str = 'highs') -> Inventory:
    """Compute Myopic Placement: greedy simulation-based placement.

    Places units one at a time, each time picking the warehouse that maximizes
    the average myopic fulfillment reward across the SAA samples.

    Args:
        graph: Bipartite graph.
        demand_samples: SAA samples for evaluating myopic reward.
        total_inventory: Total inventory budget Q.
        solver: LP solver backend.

    Returns:
        An Inventory object with the greedy placement.
    """
    optimizer = InventoryOptimizer(graph, solver=solver)
    return optimizer.myopic_greedy_inventory_placement(demand_samples, total_inventory)


# ---------------------------------------------------------------------------
# Fulfillment policies
# ---------------------------------------------------------------------------

def compute_initial_shadow_prices(graph: Graph, inventory: Inventory,
                                  average_demand: Dict[int, float],
                                  solver: str = 'highs') -> Dict[int, float]:
    """Compute initial shadow prices from the Fluid LP (used by F-SP and F-SP-R)."""
    programs = MathPrograms(graph, solver=solver)
    _, inv_constrs = programs.fluid_linear_program_fixed_inventory(average_demand, inventory)
    return {s: inv_constrs[s].Pi for s in graph.supply_nodes}


def compute_initial_offline_shadow_prices(graph: Graph, inventory: Inventory,
                                          demand_samples: List[Sequence],
                                          solver: str = 'highs') -> Dict[int, float]:
    """Compute initial shadow prices from the Offline LP (used by O-SP and O-SP-R)."""
    programs = MathPrograms(graph, solver=solver)
    _, inv_constrs = programs.offline_linear_program_fixed_inventory(demand_samples, inventory)
    n_samples = len(demand_samples)
    return {
        s: sum(inv_constrs[s, k].Pi for k in range(n_samples)) / n_samples
        for s in graph.supply_nodes
    }


def run_myopic_fulfillment(graph: Graph, inventory: Inventory,
                           test_samples: List[Sequence],
                           solver: str = 'highs') -> float:
    """Evaluate the myopic (greedy by reward) fulfillment policy."""
    scores = {(e.supply_node_id, e.demand_node_id): e.reward for e in graph.edges.values()}
    graph.construct_priority_list('myopic', scores, allow_rejections=False)

    fulfiller = Fulfillment(graph, solver=solver)
    total_reward = 0
    for seq in test_samples:
        _, reward, _ = fulfiller.fixed_list_fulfillment(seq, inventory, 'myopic')
        total_reward += reward
    return total_reward / len(test_samples)


def run_shadow_price_fulfillment(graph: Graph, inventory: Inventory,
                                 test_samples: List[Sequence],
                                 dual_variables: Dict[int, float],
                                 policy_name: str,
                                 solver: str = 'highs') -> float:
    """Evaluate a static shadow price fulfillment policy (F-SP or O-SP, no re-solving)."""
    scores = {
        (e.supply_node_id, e.demand_node_id): e.reward - dual_variables[e.supply_node_id]
        for e in graph.edges.values()
    }
    graph.construct_priority_list(policy_name, scores, allow_rejections=True)

    fulfiller = Fulfillment(graph, solver=solver)
    total_reward = 0
    for seq in test_samples:
        _, reward, _ = fulfiller.fixed_list_fulfillment(seq, inventory, policy_name)
        total_reward += reward
    return total_reward / len(test_samples)


def run_offline_resolving_fulfillment(graph: Graph, inventory: Inventory,
                                      test_samples: List[Sequence],
                                      train_samples: List[Sequence],
                                      re_solving_epochs: List[int],
                                      solver: str = 'highs') -> float:
    """Evaluate the O-SP-R (offline shadow prices with re-solving) policy."""
    fulfiller = OffLpReSolvingFulfillment(graph, solver=solver)

    initial_duals = {s: 0.0 for s in graph.supply_nodes}
    # Compute initial duals from training data
    initial_duals = fulfiller.compute_dual_variables(
        train_samples, 0, inventory.initial_inventory, None, False
    )

    total_reward = 0
    for seq in test_samples:
        _, reward, _ = fulfiller.fulfill(
            seq, inventory, initial_duals, train_samples,
            re_solving_epochs=re_solving_epochs, filter_samples=False
        )
        total_reward += reward
    return total_reward / len(test_samples)


def run_fluid_resolving_fulfillment(graph: Graph, inventory: Inventory,
                                    test_samples: List[Sequence],
                                    cumulative_avg_demand: Dict[int, Dict[int, float]],
                                    re_solving_epochs: List[int],
                                    solver: str = 'highs') -> float:
    """Evaluate the F-SP-R (fluid shadow prices with re-solving) policy."""
    fulfiller = FluLpReSolvingFulfillment(graph, solver=solver)

    programs = MathPrograms(graph, solver=solver)
    _, inv_constrs = programs.fluid_linear_program_fixed_inventory(
        cumulative_avg_demand[0], inventory
    )
    initial_duals = {s: inv_constrs[s].Pi for s in graph.supply_nodes}

    total_reward = 0
    for seq in test_samples:
        _, reward, _ = fulfiller.fulfill(
            seq, inventory, initial_duals, cumulative_avg_demand,
            re_solving_epochs=re_solving_epochs
        )
        total_reward += reward
    return total_reward / len(test_samples)


# ---------------------------------------------------------------------------
# Prophet's reward (upper bound for normalization)
# ---------------------------------------------------------------------------

def compute_prophet_reward(graph: Graph, test_samples: List[Sequence],
                           total_inventory: int,
                           solver: str = 'highs') -> float:
    """Compute the prophet's reward: optimal offline LP value with optimized placement.

    This is max_{x in CH(X)} OFF_hat(x) over the test samples, serving as an
    upper bound on any policy's expected reward.
    """
    programs = MathPrograms(graph, solver=solver)
    result, _ = programs.offline_linear_program_variable_inventory(
        test_samples, total_inventory
    )
    return result.ObjVal / len(test_samples)


# ---------------------------------------------------------------------------
# Cumulative average demand from training samples
# ---------------------------------------------------------------------------

def compute_cumulative_average_demand(graph: Graph, train_samples: List[Sequence],
                                      T: int) -> Dict[int, Dict[int, float]]:
    """Compute backwards cumulative average demand from training samples.

    For each time step t, computes the average remaining demand for each demand
    node from t to T-1, averaged over all training samples.

    Returns:
        Dict mapping t -> {demand_node_id: average remaining demand}.
    """
    cumulative = {}
    for t in range(T + 1):
        cumulative[t] = {j: 0.0 for j in graph.demand_nodes}

    for seq in train_samples:
        for t in range(T):
            if t < len(seq):
                demand_node = seq.requests[t].demand_node
                for t2 in range(t + 1):
                    cumulative[t2][demand_node] += 1.0 / len(train_samples)

    return cumulative


# ---------------------------------------------------------------------------
# Re-solving epochs
# ---------------------------------------------------------------------------

def get_resolving_epochs(T: int, n_epochs: int = 2) -> List[int]:
    """Compute equi-spaced re-solving epochs within the time horizon.

    For the paper, we use 2 epochs (at roughly 1/3 and 2/3 of T).
    """
    return [int(T * (k + 1) / (n_epochs + 1)) for k in range(n_epochs)]


# ---------------------------------------------------------------------------
# Experiment 1: d-sensitivity across load factors
# ---------------------------------------------------------------------------

def experiment_1(seed: int = 0, solver: str = 'highs', output_dir: str = 'results'):
    """Experiment 1: How does d affect placement quality across load factors?

    Varies d in {2, 3, 4, 5, 8} and load factor Q/E[demand] in {0.5, 0.75, 1.0, 1.25, 1.5}.
    Uses n=8 warehouses, m=15 demand nodes, DH-TI demand with T=60.
    Evaluates 4 placement procedures under O-SP-R fulfillment.
    """
    n, m, T = 8, 15, 60
    d_values = [2, 3, 4, 5, 8]
    load_factors = [0.50, 0.75, 1.00, 1.25, 1.50]
    n_instances = 10
    K_train = 500
    K_test = 500
    n_epochs = 2

    demand_nodes = list(range(m))
    weights = np.ones(m) / m  # uniform weights

    rng_master = np.random.default_rng(seed)

    results = []

    for d in d_values:
        for load_factor in load_factors:
            Q = int(round(T * load_factor))
            re_solving_epochs = get_resolving_epochs(T, n_epochs)

            logging.info(f'd={d}, load_factor={load_factor}, Q={Q}')

            for instance_id in range(n_instances):
                graph_seed = rng_master.integers(0, 2**31)
                train_seed = rng_master.integers(0, 2**31)
                test_seed = rng_master.integers(0, 2**31)
                round_seed = rng_master.integers(0, 2**31)

                graph_rng = np.random.default_rng(graph_seed)
                round_rng = np.random.default_rng(round_seed)

                graph = build_d_regular_graph(n, m, d, graph_rng)

                train_gen = make_dh_ti_generator(demand_nodes, T, weights, train_seed)
                test_gen = make_dh_ti_generator(demand_nodes, T, weights, test_seed)
                train_samples = generate_samples(train_gen, K_train)
                test_samples = generate_samples(test_gen, K_test)

                average_demand = {j: T * weights[j] for j in demand_nodes}

                prophet = compute_prophet_reward(graph, test_samples, Q, solver)

                # --- Placement procedures ---
                placements = {}
                placement_times = {}

                start = time.time()
                placements['offline'] = offline_placement(
                    graph, train_samples, Q, round_rng, solver
                )
                placement_times['offline'] = time.time() - start

                start = time.time()
                placements['fluid'] = fluid_placement(
                    graph, average_demand, Q, round_rng, solver
                )
                placement_times['fluid'] = time.time() - start

                start = time.time()
                placements['scaled_fluid'] = scaled_fluid_placement(
                    graph, average_demand, Q, round_rng, solver
                )
                placement_times['scaled_fluid'] = time.time() - start

                start = time.time()
                placements['myopic'] = myopic_placement(
                    graph, train_samples, Q, solver
                )
                placement_times['myopic'] = time.time() - start

                # --- Fulfillment: O-SP-R for all placements ---
                for placement_name, inventory in placements.items():
                    reward = run_offline_resolving_fulfillment(
                        graph, inventory, test_samples, train_samples,
                        re_solving_epochs, solver
                    )
                    comp_ratio = reward / prophet if prophet > 0 else 0

                    results.append({
                        'd': d,
                        'load_factor': load_factor,
                        'Q': Q,
                        'instance_id': instance_id,
                        'placement': placement_name,
                        'fulfillment': 'O-SP-R',
                        'reward': reward,
                        'prophet': prophet,
                        'competitive_ratio': comp_ratio,
                        'placement_time': placement_times[placement_name],
                    })

                logging.info(
                    f'  instance {instance_id}: prophet={prophet:.2f}, '
                    + ', '.join(
                        f'{p}={r["competitive_ratio"]:.4f}'
                        for p in placements
                        for r in [results[-len(placements) + list(placements).index(p)]]
                    )
                )

    write_results(results, os.path.join(output_dir, 'experiment_1.csv'))
    return results


# ---------------------------------------------------------------------------
# Experiment 2: SAA convergence
# ---------------------------------------------------------------------------

def experiment_2(seed: int = 0, solver: str = 'highs', output_dir: str = 'results'):
    """Experiment 2: How many SAA samples does Offline Placement need?

    Varies instance size (n, m) in {(3,9), (5,15), (8,15)} and K in
    {50, 100, 250, 500}. Fixed d=3, load factor=0.75.
    Reports competitive ratio and solve time.
    """
    instance_sizes = [(3, 9), (5, 15), (8, 15)]
    K_values = [50, 100, 250, 500]
    d, T = 3, 60
    load_factor = 0.75
    n_instances = 10
    K_test = 500
    n_epochs = 2

    rng_master = np.random.default_rng(seed)

    results = []

    for n, m in instance_sizes:
        Q = int(round(T * load_factor))
        demand_nodes = list(range(m))
        weights = np.ones(m) / m
        re_solving_epochs = get_resolving_epochs(T, n_epochs)

        for instance_id in range(n_instances):
            graph_seed = rng_master.integers(0, 2**31)
            test_seed = rng_master.integers(0, 2**31)
            train_seed_base = rng_master.integers(0, 2**31)
            round_seed = rng_master.integers(0, 2**31)

            graph_rng = np.random.default_rng(graph_seed)
            actual_d = min(d, n)
            graph = build_d_regular_graph(n, m, actual_d, graph_rng)

            test_gen = make_dh_ti_generator(demand_nodes, T, weights, test_seed)
            test_samples = generate_samples(test_gen, K_test)

            prophet = compute_prophet_reward(graph, test_samples, Q, solver)

            # Generate the largest training set; subsets for smaller K
            max_K = max(K_values)
            train_gen = make_dh_ti_generator(demand_nodes, T, weights, train_seed_base)
            all_train_samples = generate_samples(train_gen, max_K)

            for K in K_values:
                round_rng = np.random.default_rng(round_seed)
                train_samples = all_train_samples[:K]

                start = time.time()
                inventory = offline_placement(graph, train_samples, Q, round_rng, solver)
                solve_time = time.time() - start

                reward = run_offline_resolving_fulfillment(
                    graph, inventory, test_samples, train_samples,
                    re_solving_epochs, solver
                )
                comp_ratio = reward / prophet if prophet > 0 else 0

                results.append({
                    'n': n,
                    'm': m,
                    'K': K,
                    'instance_id': instance_id,
                    'reward': reward,
                    'prophet': prophet,
                    'competitive_ratio': comp_ratio,
                    'solve_time': solve_time,
                })

                logging.info(
                    f'  n={n}, m={m}, K={K}, instance={instance_id}: '
                    f'ratio={comp_ratio:.4f}, time={solve_time:.2f}s'
                )

    write_results(results, os.path.join(output_dir, 'experiment_2.csv'))
    return results


# ---------------------------------------------------------------------------
# Experiment 3: Placement x Fulfillment interaction
# ---------------------------------------------------------------------------

def experiment_3(seed: int = 0, solver: str = 'highs', output_dir: str = 'results'):
    """Experiment 3: How does fulfillment quality interact with placement?

    Full cross of 4 placements x 5 fulfillment policies.
    Fixed n=5, m=15, d=3, load factor=0.75, DH-TI demand.
    """
    n, m, d, T = 5, 15, 3, 60
    load_factor = 0.75
    Q = int(round(T * load_factor))
    n_instances = 10
    K_train = 500
    K_test = 500
    n_epochs = 2

    demand_nodes = list(range(m))
    weights = np.ones(m) / m
    average_demand = {j: T * weights[j] for j in demand_nodes}
    re_solving_epochs = get_resolving_epochs(T, n_epochs)

    rng_master = np.random.default_rng(seed)

    results = []

    for instance_id in range(n_instances):
        graph_seed = rng_master.integers(0, 2**31)
        train_seed = rng_master.integers(0, 2**31)
        test_seed = rng_master.integers(0, 2**31)
        round_seed = rng_master.integers(0, 2**31)

        graph_rng = np.random.default_rng(graph_seed)
        round_rng = np.random.default_rng(round_seed)

        graph = build_d_regular_graph(n, m, d, graph_rng)

        train_gen = make_dh_ti_generator(demand_nodes, T, weights, train_seed)
        test_gen = make_dh_ti_generator(demand_nodes, T, weights, test_seed)
        train_samples = generate_samples(train_gen, K_train)
        test_samples = generate_samples(test_gen, K_test)

        prophet = compute_prophet_reward(graph, test_samples, Q, solver)

        # --- Compute placements ---
        placements = {}

        placements['offline'] = offline_placement(
            graph, train_samples, Q, np.random.default_rng(round_seed), solver
        )
        placements['fluid'] = fluid_placement(
            graph, average_demand, Q, np.random.default_rng(round_seed), solver
        )
        placements['scaled_fluid'] = scaled_fluid_placement(
            graph, average_demand, Q, np.random.default_rng(round_seed), solver
        )
        placements['myopic'] = myopic_placement(
            graph, train_samples, Q, solver
        )

        # --- Precompute cumulative average demand for F-SP-R ---
        cumulative_avg_demand = compute_cumulative_average_demand(
            graph, train_samples, T
        )

        # --- Evaluate all (placement, fulfillment) pairs ---
        for placement_name, inventory in placements.items():

            # 1. Myopic fulfillment
            reward = run_myopic_fulfillment(graph, inventory, test_samples, solver)
            results.append(_make_result(
                instance_id, placement_name, 'Myopic', reward, prophet
            ))

            # 2. F-SP (fluid shadow prices, no re-solving)
            fluid_duals = compute_initial_shadow_prices(
                graph, inventory, average_demand, solver
            )
            reward = run_shadow_price_fulfillment(
                graph, inventory, test_samples, fluid_duals, 'F-SP', solver
            )
            results.append(_make_result(
                instance_id, placement_name, 'F-SP', reward, prophet
            ))

            # 3. O-SP (offline shadow prices, no re-solving)
            offline_duals = compute_initial_offline_shadow_prices(
                graph, inventory, train_samples, solver
            )
            reward = run_shadow_price_fulfillment(
                graph, inventory, test_samples, offline_duals, 'O-SP', solver
            )
            results.append(_make_result(
                instance_id, placement_name, 'O-SP', reward, prophet
            ))

            # 4. F-SP-R (fluid shadow prices with re-solving)
            reward = run_fluid_resolving_fulfillment(
                graph, inventory, test_samples, cumulative_avg_demand,
                re_solving_epochs, solver
            )
            results.append(_make_result(
                instance_id, placement_name, 'F-SP-R', reward, prophet
            ))

            # 5. O-SP-R (offline shadow prices with re-solving)
            reward = run_offline_resolving_fulfillment(
                graph, inventory, test_samples, train_samples,
                re_solving_epochs, solver
            )
            results.append(_make_result(
                instance_id, placement_name, 'O-SP-R', reward, prophet
            ))

        logging.info(f'  instance {instance_id} done (prophet={prophet:.2f})')

    write_results(results, os.path.join(output_dir, 'experiment_3.csv'))
    return results


def _make_result(instance_id, placement, fulfillment, reward, prophet):
    return {
        'instance_id': instance_id,
        'placement': placement,
        'fulfillment': fulfillment,
        'reward': reward,
        'prophet': prophet,
        'competitive_ratio': reward / prophet if prophet > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_results(results: List[dict], filepath: str):
    """Write results to a CSV file."""
    if not results:
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    logging.info(f'Results written to {filepath}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Paper experiments')
    parser.add_argument('--experiment', type=int, required=True, choices=[1, 2, 3],
                        help='Which experiment to run (1, 2, or 3)')
    parser.add_argument('--seed', type=int, default=42, help='Master random seed')
    parser.add_argument('--solver', type=str, default='highs', choices=['highs', 'gurobi'],
                        help='LP solver backend')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory for output CSV files')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )

    if args.experiment == 1:
        logging.info('Running Experiment 1: d-sensitivity across load factors')
        experiment_1(seed=args.seed, solver=args.solver, output_dir=args.output_dir)
    elif args.experiment == 2:
        logging.info('Running Experiment 2: SAA convergence')
        experiment_2(seed=args.seed, solver=args.solver, output_dir=args.output_dir)
    elif args.experiment == 3:
        logging.info('Running Experiment 3: Placement x Fulfillment interaction')
        experiment_3(seed=args.seed, solver=args.solver, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
