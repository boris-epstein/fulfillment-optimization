"""Experiments for the inventory placement and fulfillment paper revision.

Three experiments:
  1. d-sensitivity: How does the degree parameter d affect placement quality
     across different load factors?
  2. Sample convergence: How do ALL placement procedures behave as a function
     of K (SAA samples), using empirical averages for Fluid/Scaled Fluid?
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
from typing import Dict, List

import numpy as np

from Graph import Graph
from Demand import TemporalIndependenceGenerator, CorrelGenerator, Sequence
from MathPrograms import MathPrograms
from FulfillmentOptimization import (
    Inventory, InventoryOptimizer, Fulfillment,
)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_d_regular_graph(n: int, m: int, d: int, rng: np.random.Generator,
                          max_attempts: int = 100) -> Graph:
    """Build a random d-regular bipartite graph with no isolated supply nodes.

    Each demand node connects to exactly d distinct warehouses chosen uniformly
    at random, with edge rewards drawn from Uniform(0, 1). Resamples if any
    supply node ends up disconnected.

    Args:
        n: Number of supply nodes (warehouses).
        m: Number of demand nodes.
        d: Degree — each demand node connects to exactly d supply nodes.
        rng: Numpy random generator for reproducibility.
        max_attempts: Maximum resampling attempts before raising an error.

    Returns:
        A Graph instance with the constructed edges.
    """
    supply_ids = list(range(n))

    for _ in range(max_attempts):
        graph = Graph()
        for i in range(n):
            graph.add_supply_node(i)
        for j in range(m):
            graph.add_demand_node(j)

        connected_supply = set()
        edges = []
        for j in range(m):
            neighbors = rng.choice(supply_ids, size=d, replace=False)
            for i in neighbors:
                reward = rng.uniform(0, 1)
                edges.append((i, j, reward))
                connected_supply.add(i)

        if len(connected_supply) == n:
            for i, j, reward in edges:
                graph.add_edge(i, j, reward)
            graph.populate_neighbors()
            return graph

    raise RuntimeError(
        f'Could not build a connected d-regular graph after {max_attempts} attempts '
        f'(n={n}, m={m}, d={d})'
    )


# ---------------------------------------------------------------------------
# Demand generation
# ---------------------------------------------------------------------------

def make_dh_ti_generator(demand_nodes: List[int], T: int, weights: np.ndarray,
                         seed: int) -> TemporalIndependenceGenerator:
    """Create a DH-TI demand generator (deterministic horizon, IID arrivals)."""
    p = {t: list(weights) for t in range(T)}
    return TemporalIndependenceGenerator(demand_nodes, p, seed)


def make_rh_ti_generator(demand_nodes: List[int], expected_T: int,
                         weights: np.ndarray, seed: int) -> CorrelGenerator:
    """Create a RH-TI demand generator (geometric horizon, IID arrivals).

    T ~ Geometric(1/(1+expected_T)), so E[T] = expected_T and
    Std[T] ≈ expected_T (high variance).
    """
    return CorrelGenerator(
        mean=expected_T,
        demand_nodes=demand_nodes,
        weights=list(weights),
        seed=seed,
        distribution='geometric',
    )


def make_generator(demand_model: str, demand_nodes: List[int], T: int,
                   weights: np.ndarray, seed: int):
    """Dispatch to the appropriate demand generator factory."""
    if demand_model == 'DH-TI':
        return make_dh_ti_generator(demand_nodes, T, weights, seed)
    elif demand_model == 'RH-TI':
        return make_rh_ti_generator(demand_nodes, T, weights, seed)
    else:
        raise ValueError(f'Unknown demand model: {demand_model}')


def generate_samples(generator, n_samples: int) -> List[Sequence]:
    """Generate a list of demand sequences from a generator."""
    return [generator.generate_sequence() for _ in range(n_samples)]


def empirical_average_demand(demand_nodes: List[int],
                             samples: List[Sequence]) -> Dict[int, float]:
    """Compute empirical average demand per demand node from K sample sequences.

    For each sample, counts how many times each demand node appears across all
    time steps, then averages over all K samples.

    Returns:
        Dict mapping demand_node_id to average total arrivals across the horizon.
    """
    counts = {j: 0.0 for j in demand_nodes}
    for seq in samples:
        for req in seq.requests:
            counts[req.demand_node] += 1.0
    K = len(samples)
    return {j: counts[j] / K for j in demand_nodes}


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
    # Myopic fulfillment requires a 'myopic' priority list on the graph
    scores = {(e.supply_node_id, e.demand_node_id): e.reward for e in graph.edges.values()}
    graph.construct_priority_list('myopic', scores, allow_rejections=False)

    optimizer = InventoryOptimizer(graph, solver=solver)
    return optimizer.myopic_greedy_inventory_placement(demand_samples, total_inventory)


# ---------------------------------------------------------------------------
# Fulfillment policies
# ---------------------------------------------------------------------------

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
# Experiment 1: d-sensitivity across load factors
# ---------------------------------------------------------------------------

def experiment_1(seed: int = 42, solver: str = 'highs', output_dir: str = 'results'):
    """Experiment 1: How does d affect placement quality across load factors?

    Varies d in {2, 3, 4, 5, 8} and load factor Q/E[demand] in {0.5, 0.75, 1.0, 1.25, 1.5}.
    Uses n=8 warehouses, m=15 demand nodes, E[T]=60.
    Two demand models: DH-TI and RH-TI.
    Evaluates 4 placement procedures under Myopic and O-SP fulfillment.
    """
    n, m, T = 8, 15, 60
    d_values = [2, 3, 4, 5, 8]
    load_factors = [0.50, 0.75, 1.00, 1.25, 1.50]
    demand_models = ['DH-TI', 'RH-TI']
    n_instances = 20
    K_train = 500
    K_test = 500

    demand_nodes = list(range(m))
    weights = np.ones(m) / m  # uniform weights

    rng_master = np.random.default_rng(seed)

    results = []

    for demand_model in demand_models:
        for d in d_values:
            for load_factor in load_factors:
                Q = int(round(T * load_factor))

                logging.info(f'{demand_model}, d={d}, load_factor={load_factor}, Q={Q}')

                for instance_id in range(n_instances):
                    graph_seed = rng_master.integers(0, 2**31)
                    train_seed = rng_master.integers(0, 2**31)
                    test_seed = rng_master.integers(0, 2**31)
                    round_seed = rng_master.integers(0, 2**31)

                    graph_rng = np.random.default_rng(graph_seed)

                    graph = build_d_regular_graph(n, m, d, graph_rng)

                    train_gen = make_generator(demand_model, demand_nodes, T, weights, train_seed)
                    test_gen = make_generator(demand_model, demand_nodes, T, weights, test_seed)
                    train_samples = generate_samples(train_gen, K_train)
                    test_samples = generate_samples(test_gen, K_test)

                    emp_avg = empirical_average_demand(demand_nodes, train_samples)

                    prophet = compute_prophet_reward(graph, test_samples, Q, solver)

                    # --- Placement procedures ---
                    placements = {}
                    placement_times = {}

                    start = time.time()
                    placements['offline'] = offline_placement(
                        graph, train_samples, Q,
                        np.random.default_rng(round_seed), solver
                    )
                    placement_times['offline'] = time.time() - start

                    start = time.time()
                    placements['fluid'] = fluid_placement(
                        graph, emp_avg, Q,
                        np.random.default_rng(round_seed), solver
                    )
                    placement_times['fluid'] = time.time() - start

                    start = time.time()
                    placements['scaled_fluid'] = scaled_fluid_placement(
                        graph, emp_avg, Q,
                        np.random.default_rng(round_seed), solver
                    )
                    placement_times['scaled_fluid'] = time.time() - start

                    start = time.time()
                    placements['myopic'] = myopic_placement(
                        graph, train_samples, Q, solver
                    )
                    placement_times['myopic'] = time.time() - start

                    # --- Fulfillment: Myopic and O-SP for all placements ---
                    for placement_name, inventory in placements.items():
                        myopic_reward = run_myopic_fulfillment(
                            graph, inventory, test_samples, solver
                        )

                        offline_duals = compute_initial_offline_shadow_prices(
                            graph, inventory, train_samples, solver
                        )
                        osp_reward = run_shadow_price_fulfillment(
                            graph, inventory, test_samples, offline_duals,
                            f'O-SP-{demand_model}-{placement_name}-d{d}', solver
                        )

                        for ful_name, reward in [('Myopic', myopic_reward), ('O-SP', osp_reward)]:
                            comp_ratio = reward / prophet if prophet > 0 else 0
                            results.append({
                                'demand_model': demand_model,
                                'd': d,
                                'load_factor': load_factor,
                                'Q': Q,
                                'instance_id': instance_id,
                                'placement': placement_name,
                                'fulfillment': ful_name,
                                'reward': reward,
                                'prophet': prophet,
                                'competitive_ratio': comp_ratio,
                                'placement_time': placement_times[placement_name],
                            })

                    logging.info(
                        f'  instance {instance_id}: prophet={prophet:.2f}'
                    )

    write_results(results, os.path.join(output_dir, 'experiment_1.csv'))
    return results


# ---------------------------------------------------------------------------
# Experiment 2: SAA convergence
# ---------------------------------------------------------------------------

def experiment_2(seed: int = 42, solver: str = 'highs', output_dir: str = 'results'):
    """Experiment 2: Sample convergence — all placements as a function of K.

    All placements use the same K training samples:
      - Offline: LP over K samples + Gandhi rounding.
      - Fluid: Fluid LP using *empirical* average demand from K samples + rounding.
      - Scaled Fluid: same as Fluid but with scaled empirical demand.
      - Myopic: greedy simulation over K samples.

    Configuration:
      - Two demand models: DH-TI and RH-TI.
      - Two weight settings: uniform (1/m each) and skewed (proportional to
        total edge reward per demand node).
      - Two load factors: 0.75 (scarce) and 1.25 (excess).
      - K in {25, 50, 100, 250, 500}.
      - n=5, m=15, d=3, E[T]=60, 20 instances per configuration.
      - Fulfillment: Myopic and O-SP.
    """
    n, m, d, T = 5, 15, 3, 60
    K_values = [25, 50, 100, 250, 500]
    load_factors = [0.75, 1.25]
    weight_settings = ['uniform', 'skewed']
    demand_models = ['DH-TI', 'RH-TI']
    n_instances = 20
    K_test = 500

    demand_nodes = list(range(m))

    rng_master = np.random.default_rng(seed)

    results = []

    for demand_model in demand_models:
        for weight_setting in weight_settings:
            for load_factor in load_factors:
                Q = int(round(T * load_factor))

                logging.info(f'{demand_model}, weights={weight_setting}, '
                             f'load_factor={load_factor}, Q={Q}')

                for instance_id in range(n_instances):
                    graph_seed = rng_master.integers(0, 2**31)
                    train_seed = rng_master.integers(0, 2**31)
                    test_seed = rng_master.integers(0, 2**31)
                    round_seed = rng_master.integers(0, 2**31)

                    graph_rng = np.random.default_rng(graph_seed)
                    graph = build_d_regular_graph(n, m, d, graph_rng)

                    # Compute weights based on setting
                    if weight_setting == 'uniform':
                        weights = np.ones(m) / m
                    else:
                        # Skewed: proportional to total edge reward into each demand node
                        raw = np.zeros(m)
                        for e in graph.edges.values():
                            raw[e.demand_node_id] += e.reward
                        raw = np.maximum(raw, 1e-8)
                        weights = raw / raw.sum()

                    # Generate test samples and the full training pool
                    test_gen = make_generator(demand_model, demand_nodes, T, weights, test_seed)
                    test_samples = generate_samples(test_gen, K_test)

                    prophet = compute_prophet_reward(graph, test_samples, Q, solver)

                    max_K = max(K_values)
                    train_gen = make_generator(demand_model, demand_nodes, T, weights, train_seed)
                    all_train_samples = generate_samples(train_gen, max_K)

                    for K in K_values:
                        train_samples = all_train_samples[:K]

                        # Empirical average demand from K samples
                        emp_avg = empirical_average_demand(demand_nodes, train_samples)

                        # --- All four placements ---
                        placements = {}
                        placement_times = {}

                        start = time.time()
                        placements['offline'] = offline_placement(
                            graph, train_samples, Q,
                            np.random.default_rng(round_seed), solver
                        )
                        placement_times['offline'] = time.time() - start

                        start = time.time()
                        placements['fluid'] = fluid_placement(
                            graph, emp_avg, Q,
                            np.random.default_rng(round_seed), solver
                        )
                        placement_times['fluid'] = time.time() - start

                        start = time.time()
                        placements['scaled_fluid'] = scaled_fluid_placement(
                            graph, emp_avg, Q,
                            np.random.default_rng(round_seed), solver
                        )
                        placement_times['scaled_fluid'] = time.time() - start

                        start = time.time()
                        placements['myopic'] = myopic_placement(
                            graph, train_samples, Q, solver
                        )
                        placement_times['myopic'] = time.time() - start

                        # --- Evaluate each placement with Myopic and O-SP ---
                        for placement_name, inventory in placements.items():
                            myopic_reward = run_myopic_fulfillment(
                                graph, inventory, test_samples, solver
                            )

                            offline_duals = compute_initial_offline_shadow_prices(
                                graph, inventory, train_samples, solver
                            )
                            osp_reward = run_shadow_price_fulfillment(
                                graph, inventory, test_samples, offline_duals,
                                f'O-SP-{demand_model}-{placement_name}-K{K}', solver
                            )

                            for ful_name, reward in [('Myopic', myopic_reward), ('O-SP', osp_reward)]:
                                comp_ratio = reward / prophet if prophet > 0 else 0
                                results.append({
                                    'demand_model': demand_model,
                                    'weight_setting': weight_setting,
                                    'load_factor': load_factor,
                                    'Q': Q,
                                    'K': K,
                                    'instance_id': instance_id,
                                    'placement': placement_name,
                                    'fulfillment': ful_name,
                                    'reward': reward,
                                    'prophet': prophet,
                                    'competitive_ratio': comp_ratio,
                                    'placement_time': placement_times[placement_name],
                                })

                        logging.info(
                            f'  {demand_model}, weights={weight_setting}, load={load_factor}, '
                            f'K={K}, instance={instance_id}: done'
                        )

    write_results(results, os.path.join(output_dir, 'experiment_2.csv'))
    return results


# ---------------------------------------------------------------------------
# Experiment 3: Placement x Fulfillment interaction
# ---------------------------------------------------------------------------

def experiment_3(seed: int = 42, solver: str = 'highs', output_dir: str = 'results'):
    """Experiment 3: How does fulfillment quality interact with placement?

    Full cross of 4 placements x 2 fulfillment policies (Myopic, O-SP).
    Two demand models: DH-TI and RH-TI.
    Fixed n=5, m=15, d=3, load factor=0.75.
    All placements use empirical average demand from training samples.
    """
    n, m, d, T = 5, 15, 3, 60
    load_factor = 0.75
    Q = int(round(T * load_factor))
    demand_models = ['DH-TI', 'RH-TI']
    n_instances = 20
    K_train = 500
    K_test = 500

    demand_nodes = list(range(m))
    weights = np.ones(m) / m

    rng_master = np.random.default_rng(seed)

    results = []

    for demand_model in demand_models:
        for instance_id in range(n_instances):
            graph_seed = rng_master.integers(0, 2**31)
            train_seed = rng_master.integers(0, 2**31)
            test_seed = rng_master.integers(0, 2**31)
            round_seed = rng_master.integers(0, 2**31)

            graph_rng = np.random.default_rng(graph_seed)

            graph = build_d_regular_graph(n, m, d, graph_rng)

            train_gen = make_generator(demand_model, demand_nodes, T, weights, train_seed)
            test_gen = make_generator(demand_model, demand_nodes, T, weights, test_seed)
            train_samples = generate_samples(train_gen, K_train)
            test_samples = generate_samples(test_gen, K_test)

            emp_avg = empirical_average_demand(demand_nodes, train_samples)

            prophet = compute_prophet_reward(graph, test_samples, Q, solver)

            # --- Compute placements (Fluid/Scaled Fluid use empirical averages) ---
            placements = {}

            placements['offline'] = offline_placement(
                graph, train_samples, Q, np.random.default_rng(round_seed), solver
            )
            placements['fluid'] = fluid_placement(
                graph, emp_avg, Q, np.random.default_rng(round_seed), solver
            )
            placements['scaled_fluid'] = scaled_fluid_placement(
                graph, emp_avg, Q, np.random.default_rng(round_seed), solver
            )
            placements['myopic'] = myopic_placement(
                graph, train_samples, Q, solver
            )

            # --- Evaluate all (placement, fulfillment) pairs ---
            for placement_name, inventory in placements.items():

                # 1. Myopic fulfillment
                reward = run_myopic_fulfillment(graph, inventory, test_samples, solver)
                results.append(_make_result(
                    demand_model, instance_id, placement_name, 'Myopic', reward, prophet
                ))

                # 2. O-SP (offline shadow prices, no re-solving)
                offline_duals = compute_initial_offline_shadow_prices(
                    graph, inventory, train_samples, solver
                )
                reward = run_shadow_price_fulfillment(
                    graph, inventory, test_samples, offline_duals,
                    f'O-SP-{demand_model}-{placement_name}', solver
                )
                results.append(_make_result(
                    demand_model, instance_id, placement_name, 'O-SP', reward, prophet
                ))

            logging.info(f'  {demand_model}, instance {instance_id} done (prophet={prophet:.2f})')

    write_results(results, os.path.join(output_dir, 'experiment_3.csv'))
    return results


def _make_result(demand_model, instance_id, placement, fulfillment, reward, prophet):
    return {
        'demand_model': demand_model,
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
