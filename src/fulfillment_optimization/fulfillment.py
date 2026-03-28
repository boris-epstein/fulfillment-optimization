from .graph import Graph, DemandNode
from .demand import CorrelGenerator, Sequence, Request
from .math_programs import MathPrograms
from collections import defaultdict
import numpy as np
from typing import List, Dict, Union, Tuple
from copy import deepcopy


class Inventory:
    """Represents the initial inventory allocation across supply nodes.

    Attributes:
        name: Descriptive label for this inventory configuration.
        initial_inventory: Dict mapping supply node ID to initial units.
        total_inventory: Sum of all initial inventory units.
    """

    def __init__(self, initial_inventory: Dict[int, int], name='str') -> None:
        self.name = name
        self.initial_inventory = initial_inventory.copy()
        self.total_inventory = sum(initial_inventory.values())


class InventoryOptimizer:
    """Optimizes inventory placement across supply nodes using LP-based methods.

    Provides multiple strategies: LP rounding, greedy shadow-price placement,
    and myopic greedy placement.
    """

    def __init__(self, graph: Graph, solver: str = 'gurobi'):
        self.graph = graph
        self.programs = MathPrograms(self.graph, solver=solver)
        self.converter = FormatConverter()

    def set_inventory_to_n(self, n: int) -> Dict[int, int]:
        """Create an inventory with n units at every supply node.

        Returns:
            Inventory with constant allocation.
        """
        initial_inventories = {}
        for i in self.graph.supply_nodes:
            initial_inventories[i] = n
        return Inventory(initial_inventories, f'constant_{n}')

    def fluid_inventory_placement_rounding(self, average_demand: Dict[int, float],
                                           total_inventory: int,
                                           rescale_inventory: bool = False) -> Dict[int, int]:
        """Solve the fluid LP with variable inventory and round the fractional solution.

        Args:
            average_demand: Expected demand per demand node.
            total_inventory: Total inventory budget.
            rescale_inventory: If True, scale demand so total matches inventory.

        Returns:
            Inventory with integer allocations from rounding.
        """
        scaling_factor = 1.0

        if rescale_inventory:
            scaling_factor = total_inventory / sum(average_demand[demand_node_index] for demand_node_index in self.graph.demand_nodes)

        fluid_lp, fluid_inventory = self.programs.fluid_linear_program_variable_inventory(
            average_demand=average_demand, total_inventory=total_inventory, scaling_factor=scaling_factor
        )
        fluid_lp.optimize()

        fluid_inventory_dict = self.converter.gurobi_to_int(gurobi_variables=fluid_inventory)
        rounded_fluid_inventory = self.greedy_round_inventory(fluid_inventory_dict)

        inventory_name = 'fluid_lp_rounding'
        if rescale_inventory:
            inventory_name += '_withscaling'
        return Inventory(rounded_fluid_inventory, inventory_name)

    def offline_inventory_placement_rounding(self, demand_samples: List[Sequence],
                                             total_inventory: int) -> Dict[int, int]:
        """Solve the offline LP with variable inventory and round the solution.

        Args:
            demand_samples: Sample demand sequences for the LP.
            total_inventory: Total inventory budget.

        Returns:
            Inventory with integer allocations from rounding.
        """
        offline_lp, offline_inventory = self.programs.offline_linear_program_variable_inventory(
            demand_samples=demand_samples, total_inventory=total_inventory
        )
        offline_lp.optimize()

        offline_inventory_dict = self.converter.gurobi_to_int(gurobi_variables=offline_inventory)
        rounded_offline_inventory = self.greedy_round_inventory(offline_inventory_dict)

        return Inventory(rounded_offline_inventory, 'offline_lp_rounding')

    def fluid_greedy_inventory_placement(self, average_demand: Dict[int, float],
                                         total_inventory: int, verbose: bool = False):
        """Greedily place inventory one unit at a time, maximizing fluid LP value.

        At each step, tries adding one unit to each supply node and keeps the
        placement that yields the highest LP objective.

        Args:
            average_demand: Expected demand per demand node.
            total_inventory: Number of units to place.
            verbose: Print progress if True.

        Returns:
            Inventory with the greedy placement.
        """
        inventory_placement = {}
        for supply_node_id in self.graph.supply_nodes:
            inventory_placement[supply_node_id] = 0

        units_placed = 0
        while units_placed < total_inventory:
            if verbose:
                print(f'Placing unit {units_placed + 1}/{total_inventory}')
            best_value = 0
            best_next_sol = {}
            for supply_node_id in self.graph.supply_nodes:
                candidate_sol = inventory_placement.copy()
                candidate_sol[supply_node_id] += 1

                fluid_lp, inventory_constraints = self.programs.fluid_linear_program_fixed_inventory(
                    average_demand, Inventory(candidate_sol, name='candidate')
                )
                fluid_lp.optimize()

                if fluid_lp.ObjVal >= best_value:
                    best_value = fluid_lp.ObjVal
                    best_next_sol = candidate_sol

            inventory_placement = best_next_sol.copy()
            units_placed += 1

        return Inventory(inventory_placement, name='fluid_greedy')

    def offline_greedy_inventory_placement(self, demand_samples: List[Sequence],
                                           total_inventory: int, verbose: bool = False):
        """Greedily place inventory using offline LP shadow prices.

        At each step, solves the offline LP and adds one unit to the supply node
        with the highest average shadow price on its inventory constraint.

        Args:
            demand_samples: Sample demand sequences.
            total_inventory: Number of units to place.
            verbose: Print progress if True.

        Returns:
            Inventory with the greedy placement.
        """
        inventory_placement = {}
        for supply_node_id in self.graph.supply_nodes:
            inventory_placement[supply_node_id] = 0

        units_placed = 0
        while units_placed < total_inventory:
            if verbose:
                print(f'Placing unit {units_placed + 1}/{total_inventory}')

            offline_lp, inventory_constraints = self.programs.offline_linear_program_fixed_inventory(
                demand_samples, Inventory(inventory_placement, name='candidate')
            )
            offline_lp.optimize()

            highest_shadow_price = 0
            best_supply_node = -1

            for supply_node_id in self.graph.supply_nodes:
                shadow_price = sum(
                    inventory_constraints[supply_node_id, sample_index].Pi
                    for sample_index in range(len(demand_samples))
                ) / len(demand_samples)
                if shadow_price >= highest_shadow_price:
                    highest_shadow_price = shadow_price
                    best_supply_node = supply_node_id

            inventory_placement[best_supply_node] += 1
            units_placed += 1

        return Inventory(inventory_placement, name='offline_greedy')

    def myopic_greedy_inventory_placement(self, demand_samples: List[Sequence],
                                          total_inventory: int, verbose: bool = False):
        """Greedily place inventory by maximizing myopic fulfillment reward.

        At each step, simulates myopic fulfillment with each candidate
        placement and picks the one with highest average reward.

        Args:
            demand_samples: Sample demand sequences.
            total_inventory: Number of units to place.
            verbose: Print progress if True.

        Returns:
            Inventory with the greedy placement.
        """
        fulfillment = Fulfillment(self.graph)

        inventory_placement = {}
        for supply_node_id in self.graph.supply_nodes:
            inventory_placement[supply_node_id] = 0

        units_placed = 0
        while units_placed < total_inventory:
            if verbose:
                print(f'Placing unit {units_placed + 1}/{total_inventory}')
            best_value = 0
            best_next_sol = {}
            for supply_node_id in self.graph.supply_nodes:
                candidate_sol = inventory_placement.copy()
                candidate_sol[supply_node_id] += 1

                candidate_reward = 0
                for sequence in demand_samples:
                    _, collected_rewards, _ = fulfillment.fixed_list_fulfillment(
                        sequence, Inventory(candidate_sol, name='candidate'), 'myopic'
                    )
                    candidate_reward += collected_rewards / len(demand_samples)

                if candidate_reward >= best_value:
                    best_value = candidate_reward
                    best_next_sol = candidate_sol

            inventory_placement = best_next_sol.copy()
            units_placed += 1

        return Inventory(inventory_placement, name='myopic_greedy')

    def greedy_round_inventory(self, fractional_placement: dict) -> dict:
        """Round fractional inventory to integers, preserving the total.

        Floors all values, then distributes remaining units to positions
        with the largest fractional remainders.

        Args:
            fractional_placement: Dict mapping supply node ID to fractional inventory.

        Returns:
            Dict with integer inventory values summing to the original total.
        """
        rounded_inventory = fractional_placement.copy()

        amount_to_round = 0
        remainders = []

        for supply_node_id in self.graph.supply_nodes:
            remainders.append((supply_node_id, rounded_inventory[supply_node_id] - np.floor(rounded_inventory[supply_node_id])))
            amount_to_round += rounded_inventory[supply_node_id] - np.floor(rounded_inventory[supply_node_id])
            rounded_inventory[supply_node_id] = int(np.floor(rounded_inventory[supply_node_id]))

        amount_to_round = np.round(amount_to_round)
        if amount_to_round > 0:
            print('Rounding something')
        remainders.sort(reverse=True, key=lambda x: x[1])

        if amount_to_round > 0:
            for elem in remainders:
                rounded_inventory[elem[0]] += 1
                amount_to_round -= 1
                if amount_to_round <= 0:
                    break

        return rounded_inventory


class Fulfillment:
    """Fulfillment engine using fixed priority lists and optional re-solving.

    Supports myopic fulfillment (follow a static priority list) and re-solving
    fulfillment (periodically recompute priority lists using LP shadow prices).
    """

    def __init__(self, graph: Graph, resolve_seed: int = 0, train_generator: CorrelGenerator = None, solver: str = 'gurobi') -> None:
        self.graph = graph
        self.programs = MathPrograms(graph, solver=solver)
        self.setup_resolve_generator(train_generator, resolve_seed)

    def setup_resolve_generator(self, train_generator: CorrelGenerator, resolve_seed: int):
        """Initialize generators used for re-solving (sampling future demand)."""
        if train_generator is not None:
            self.resolve_generator = CorrelGenerator(
                mean=train_generator.mean,
                demand_nodes=train_generator.demand_nodes,
                weights=train_generator.weights,
                seed=resolve_seed,
                distribution='deterministic'
            )
            self.train_generator = train_generator

        self.time_horizon_generator = np.random.default_rng(seed=resolve_seed)

    def fixed_list_fulfillment(self, sequence: Sequence, inventory: Inventory,
                               priority_list_name: Tuple[str, str],
                               allow_rejections: bool = True, verbose=False):
        """Fulfill a sequence by following a precomputed priority list.

        For each demand request, assigns it to the highest-priority supply node
        that still has inventory.

        Args:
            sequence: Demand sequence to fulfill.
            inventory: Initial inventory.
            priority_list_name: Key identifying which priority list to use.
            allow_rejections: Whether to allow rejecting demand.
            verbose: Print fulfillment decisions if True.

        Returns:
            Tuple of (number_fulfillments, collected_rewards, lost_sales).
        """
        collected_rewards = 0
        lost_sales = 0
        total_fulfillments = 0
        number_fulfillments = defaultdict(int)

        current_inventories = inventory.initial_inventory.copy()
        demand_node_reachable = defaultdict(lambda: True)
        best_supply_node_with_inventory = defaultdict(int)

        for request in sequence.requests:
            demand_node = self.graph.demand_nodes[request.demand_node]
            (best_supply_node_with_inventory, demand_node_reachable) = self.update_feasible_priorities(
                demand_node=demand_node,
                priority_list=demand_node.priority_lists[priority_list_name],
                best_supply_node_with_inventory=best_supply_node_with_inventory,
                current_inventories=current_inventories,
                demand_node_reachable=demand_node_reachable
            )
            if demand_node_reachable[demand_node.id]:
                supply_node_id = demand_node.priority_lists[priority_list_name][best_supply_node_with_inventory[demand_node.id]]
                current_inventories[supply_node_id] -= 1
                number_fulfillments[supply_node_id, demand_node.id] += 1
                collected_rewards += self.graph.edges[supply_node_id, demand_node.id].reward
                total_fulfillments += 1
                if verbose:
                    print(f'Demand from {demand_node.id} fulfilled from {supply_node_id}')
            else:
                lost_sales += 1
                if verbose:
                    print(f'Demand from {demand_node.id} lost')

        return number_fulfillments, collected_rewards, lost_sales

    def update_feasible_priorities(self, demand_node: DemandNode, priority_list: List[int],
                                   best_supply_node_with_inventory: Dict[int, int],
                                   current_inventories: Dict[int, int],
                                   demand_node_reachable: Dict[int, bool]):
        """Advance the priority list pointer past depleted supply nodes.

        Returns:
            Updated (best_supply_node_with_inventory, demand_node_reachable).
        """
        while (best_supply_node_with_inventory[demand_node.id] < len(priority_list)
               and current_inventories[priority_list[best_supply_node_with_inventory[demand_node.id]]] <= 0):
            best_supply_node_with_inventory[demand_node.id] += 1
        if best_supply_node_with_inventory[demand_node.id] == len(priority_list):
            demand_node_reachable[demand_node.id] = False

        return best_supply_node_with_inventory, demand_node_reachable

    def fulfillment_with_resolving(self, sequence: Sequence, inventory,
                                   priority_list_name: Tuple[str, str],
                                   resolve_times: List[float], demand_model: str,
                                   prior_mean: Union[float, List[float]],
                                   verbose=False, n_resolve_samples=40):
        """Fulfill a sequence with periodic re-solving of priority lists.

        At specified resolve times, recomputes priority lists using LP shadow
        prices based on updated demand estimates and remaining inventory.

        Args:
            sequence: Demand sequence to fulfill.
            inventory: Initial inventory.
            priority_list_name: Identifies the re-solving strategy.
            resolve_times: List of times (in [0,1]) at which to re-solve.
            demand_model: Demand model type ('correl', 'deter', 'indep').
            prior_mean: Prior mean for demand estimation.
            verbose: Print decisions if True.
            n_resolve_samples: Number of samples to generate for re-solving.

        Returns:
            Tuple of (number_fulfillments, collected_rewards, lost_sales).
        """
        collected_rewards = 0
        lost_sales = 0
        total_fulfillments = 0
        number_fulfillments = defaultdict(int)

        current_inventories = inventory.initial_inventory.copy()
        priority_lists = {demand_node.id: demand_node.priority_lists[priority_list_name] for demand_node in self.graph.demand_nodes.values()}

        demand_node_reachable = defaultdict(lambda: True)
        best_supply_node_with_inventory = defaultdict(int)

        resolve_point = resolve_times[0]
        next_resolve_index = 1

        num_arrivals = 0
        num_arrivals_per_node = defaultdict(int)

        for request in sequence.requests:
            if request.arrival_time >= resolve_point:
                if verbose:
                    print(f'Resolving at time {resolve_point}')

                priority_lists = self.re_compute_priority_lists(
                    Inventory(current_inventories, name='current'),
                    resolve_point, num_arrivals, priority_list_name,
                    demand_model, prior_mean, n_resolve_samples, num_arrivals_per_node
                )

                best_supply_node_with_inventory = defaultdict(int)
                demand_node_reachable = defaultdict(lambda: True)

                if next_resolve_index < len(resolve_times):
                    resolve_point = resolve_times[next_resolve_index]
                    next_resolve_index += 1
                else:
                    resolve_point = 1.1

            num_arrivals += 1

            demand_node = self.graph.demand_nodes[request.demand_node]
            num_arrivals_per_node[demand_node.id] += 1

            (best_supply_node_with_inventory, demand_node_reachable) = self.update_feasible_priorities(
                demand_node=demand_node,
                priority_list=priority_lists[demand_node.id],
                best_supply_node_with_inventory=best_supply_node_with_inventory,
                current_inventories=current_inventories,
                demand_node_reachable=demand_node_reachable
            )
            if demand_node_reachable[demand_node.id]:
                supply_node_id = priority_lists[demand_node.id][best_supply_node_with_inventory[demand_node.id]]
                current_inventories[supply_node_id] -= 1
                number_fulfillments[supply_node_id, demand_node.id] += 1
                collected_rewards += self.graph.edges[supply_node_id, demand_node.id].reward
                total_fulfillments += 1
                if verbose:
                    print(f'Demand from {demand_node.id} fulfilled from {supply_node_id}')
            else:
                lost_sales += 1

        return number_fulfillments, collected_rewards, lost_sales

    def re_compute_priority_lists(self, current_inventory, resolve_point, num_arrivals,
                                  priority_list_name, demand_model, prior_mean,
                                  n_resolve_samples, num_arrivals_per_node):
        """Recompute priority lists using LP shadow prices at a resolve point."""
        if priority_list_name[0] == 'offline_with_rejections':
            scores = self.compute_offline_shadow_price_scores(
                current_inventory, resolve_point, num_arrivals, demand_model,
                prior_mean, n_resolve_samples, num_arrivals_per_node
            )

        if priority_list_name[0] == 'fluid_with_rejections':
            scores = self.compute_fluid_shadow_price_scores(
                current_inventory, resolve_point, num_arrivals, demand_model,
                prior_mean, num_arrivals_per_node
            )

        priority_list = self.construct_priority_list(scores)
        return priority_list

    def compute_offline_shadow_price_scores(self, current_inventory, resolve_point,
                                            num_arrivals, demand_model, prior_mean,
                                            n_resolve_samples, num_arrivals_per_node):
        """Compute edge scores using offline LP shadow prices on inventory constraints.

        Score = edge reward - average shadow price of the supply node's constraint.
        """
        if demand_model == 'correl':
            resolve_samples = [None for _ in range(n_resolve_samples)]
            p = 1 / (1 + prior_mean)
            q = 1 - (1 - resolve_point) * (1 - p)
            for sample_index in range(n_resolve_samples):
                new_mean = self.time_horizon_generator.negative_binomial(num_arrivals + 1, q)
                self.resolve_generator.set_mean(new_mean)
                resolve_samples[sample_index] = self.resolve_generator.generate_sequence()

        if demand_model == 'deter':
            new_mean = int(prior_mean - num_arrivals)
            self.resolve_generator.set_mean(new_mean)
            resolve_samples = [self.resolve_generator.generate_sequence() for _ in range(n_resolve_samples)]

        if demand_model == 'indep':
            resolve_samples = self.sample_indep_resolve_samples(
                resolve_point, num_arrivals_per_node, prior_mean, n_resolve_samples
            )

        offline_lp, inventory_constraints = self.programs.offline_linear_program_fixed_inventory(
            resolve_samples, current_inventory
        )
        offline_lp.optimize()

        scores = {
            (edge.supply_node_id, edge.demand_node_id):
                edge.reward - sum(
                    inventory_constraints[edge.supply_node_id, sample_index].Pi
                    for sample_index in range(len(resolve_samples))
                ) / len(resolve_samples)
            for edge in self.graph.edges.values()
        }

        return scores

    def construct_priority_list(self, scores: Dict[Tuple[int, int], float], allow_rejections=True):
        """Build per-demand-node priority lists from edge scores.

        Args:
            scores: Map from (supply_node_id, demand_node_id) to score.
            allow_rejections: If True, exclude edges with negative scores.

        Returns:
            Dict mapping demand_node_id to ordered list of supply node IDs.
        """
        priority_list = defaultdict(list)

        score_ordered_edges = [(supply_node_id, demand_node_id, scores[supply_node_id, demand_node_id])
                               for (supply_node_id, demand_node_id) in self.graph.edges]
        score_ordered_edges.sort(key=lambda x: x[2], reverse=True)

        for supply_node_id, demand_node_id, score in score_ordered_edges:
            if (not allow_rejections) or (score >= 0):
                priority_list[demand_node_id].append(supply_node_id)

        return priority_list

    def sample_indep_resolve_samples(self, resolve_point, num_arrivals_per_node,
                                     prior_mean, n_resolve_samples):
        """Generate resolve samples under independent demand model."""
        resolve_samples = [None for sample_index in range(n_resolve_samples)]
        for sample_index in range(n_resolve_samples):
            resolve_samples[sample_index] = self.generate_indep_sample(
                resolve_point, num_arrivals_per_node, prior_mean
            )
        return resolve_samples

    def generate_indep_sample(self, resolve_point, num_arrivals_per_node, prior_mean):
        """Generate a single resolve sample for independent demand.

        Uses negative binomial posterior to estimate remaining demand per node.
        """
        reqs = []

        for demand_node_id in self.graph.demand_nodes:
            p = 1 / (1 + prior_mean[demand_node_id])
            q = 1 - (1 - resolve_point) * (1 - p)
            time_horizon = self.time_horizon_generator.negative_binomial(num_arrivals_per_node[demand_node_id] + 1, q)
            reqs += [demand_node_id for _ in range(time_horizon)]

        T = len(reqs)
        arrival_times = [0 for t in range(T)]
        requests = [Request(reqs[t], arrival_times[t]) for t in range(T)]

        return Sequence(requests)

    def compute_fluid_shadow_price_scores(self, current_inventory, resolve_point,
                                          num_arrivals, demand_model, prior_mean,
                                          num_arrivals_per_node):
        """Compute edge scores using fluid LP shadow prices.

        Updates expected remaining demand based on the demand model and observed
        arrivals, then solves the fluid LP to get shadow prices.
        """
        updated_average_demands = {}

        if demand_model == 'deter':
            remaining_arrivals = prior_mean - num_arrivals
            for demand_node_id in self.graph.demand_nodes:
                updated_average_demands[demand_node_id] = remaining_arrivals * self.train_generator.weights[demand_node_id]

        if demand_model == 'correl':
            p = 1 / (1 + prior_mean)
            q = 1 - (1 - resolve_point) * (1 - p)
            remaining_arrivals = (num_arrivals + 1) * (1 - q) / q
            for demand_node_id in self.graph.demand_nodes:
                updated_average_demands[demand_node_id] = remaining_arrivals * self.train_generator.weights[demand_node_id]

        if demand_model == 'indep':
            for demand_node_id in self.graph.demand_nodes:
                p = 1 / (1 + prior_mean[demand_node_id])
                q = 1 - (1 - resolve_point) * (1 - p)
                updated_average_demands[demand_node_id] = (num_arrivals_per_node[demand_node_id] + 1) * (1 - q) / q

        fluid_lp, inventory_constraints = self.programs.fluid_linear_program_fixed_inventory(
            updated_average_demands, current_inventory
        )
        fluid_lp.optimize()

        scores = {
            (edge.supply_node_id, edge.demand_node_id): edge.reward - inventory_constraints[edge.supply_node_id].Pi
            for edge in self.graph.edges.values()
        }

        return scores


class BalanceFulfillment:
    """Balance algorithm for online bipartite matching.

    Assigns each demand request to the supply node that maximizes
    reward * (1 - exp(used_fraction - 1)), which penalizes heavily-used
    supply nodes to maintain balance.
    """

    def __init__(self, graph: Graph):
        self.graph = graph

    def fulfill(self, sequence: Sequence, inventory: Inventory, verbose: bool = False):
        """Fulfill a demand sequence using the balance algorithm.

        Returns:
            Tuple of (number_fulfillments, collected_rewards, lost_sales).
        """
        collected_rewards = 0
        lost_sales = 0
        total_fulfillments = 0
        number_fulfillments = defaultdict(int)

        current_inventories = inventory.initial_inventory.copy()

        for request in sequence.requests:
            demand_node = self.graph.demand_nodes[request.demand_node]
            supply_node_chosen = self.choose_supply_node(demand_node, current_inventories, inventory)
            if supply_node_chosen == -1:
                lost_sales += 1
                if verbose:
                    print(f'Demand from {demand_node.id} lost')
            else:
                if verbose:
                    print(f'Demand from {demand_node.id} fulfilled from warehouse {supply_node_chosen}')
                current_inventories[supply_node_chosen] -= 1
                number_fulfillments[supply_node_chosen, demand_node.id] += 1
                collected_rewards += self.graph.edges[supply_node_chosen, demand_node.id].reward
                total_fulfillments += 1

        return number_fulfillments, collected_rewards, lost_sales

    def choose_supply_node(self, demand_node: DemandNode, current_inventories, inventory: Inventory):
        """Select the supply node maximizing balance-adjusted pseudo-reward.

        Returns:
            Supply node ID, or -1 if no supply node has inventory.
        """
        fractions = {
            supply_node_id: 1 - extended_division(current_inventories[supply_node_id], inventory.initial_inventory[supply_node_id])
            for supply_node_id in demand_node.neighbors
        }
        pseudo_rewards = [
            [supply_node_id,
             self.graph.edges[supply_node_id, demand_node.id].reward * (1 - np.exp(fractions[supply_node_id] - 1))]
            for supply_node_id in fractions
        ]

        pseudo_rewards.sort(key=lambda x: x[1], reverse=True)

        for supply_node_id, _ in pseudo_rewards:
            if current_inventories[supply_node_id] > 0:
                return supply_node_id

        return -1


def extended_division(a, b):
    """Safe division returning 1 when the denominator is 0."""
    if b == 0:
        return 1
    else:
        return float(a) / float(b)


class MultiPriceBalanceFulfillment:
    """Multi-price balance algorithm for heterogeneous edge rewards.

    Generalizes the balance algorithm to handle supply nodes with multiple
    distinct reward levels. Uses a piecewise-exponential opportunity cost
    function phi that adapts to the reward structure.
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self.compute_distinct_prices()
        self.compute_alphas()

    def compute_distinct_prices(self):
        """Identify the set of distinct reward values for each supply node."""
        self.num_distinct_prices = {}
        self.distinct_prices = {}

        for supply_node_id in self.graph.supply_nodes:
            distinct_price_set = set()
            supply_node = self.graph.supply_nodes[supply_node_id]
            for demand_node_id in supply_node.neighbors:
                edge = self.graph.edges[supply_node_id, demand_node_id]
                distinct_price_set.add(edge.reward)

            self.distinct_prices[supply_node_id] = sorted(list(distinct_price_set))
            self.num_distinct_prices[supply_node_id] = len(distinct_price_set)

    def compute_alphas(self):
        """Compute breakpoint parameters for the two-price phi function.

        For supply nodes with exactly two distinct prices, computes the
        optimal breakpoint alpha that balances acceptance rates.
        """
        self.alphas = defaultdict(list)
        for supply_node_id in self.graph.supply_nodes:
            if self.num_distinct_prices[supply_node_id] == 2:
                eps = self.distinct_prices[supply_node_id][1] / self.distinct_prices[supply_node_id][0]
                sqrt = np.sqrt(1 + 4 * eps * (eps - 1) / np.e)
                self.alphas[supply_node_id].append(-1 * np.log((sqrt - 1) / (2 * (eps - 1))))
                self.alphas[supply_node_id].append(1 - self.alphas[supply_node_id][0])

    def phi(self, supply_node_id: float, used_fraction: float):
        """Compute the opportunity cost for a supply node at a given usage level.

        For single-price nodes, uses a standard exponential cost.
        For two-price nodes, uses a piecewise-exponential with a breakpoint.

        Args:
            supply_node_id: The supply node.
            used_fraction: Fraction of initial inventory consumed (0 to 1).

        Returns:
            Opportunity cost value.
        """
        if self.num_distinct_prices[supply_node_id] == 1:
            return self.distinct_prices[supply_node_id][0] * (np.exp(used_fraction) - 1) / (np.e - 1)

        elif self.num_distinct_prices[supply_node_id] == 2:
            if used_fraction >= self.alphas[supply_node_id][0]:
                exponential_factor = (np.exp(used_fraction - self.alphas[supply_node_id][0]) - 1) / (np.exp(self.alphas[supply_node_id][1]) - 1)
                return self.distinct_prices[supply_node_id][0] + (self.distinct_prices[supply_node_id][1] - self.distinct_prices[supply_node_id][0]) * exponential_factor
            else:
                return self.distinct_prices[supply_node_id][0] * (np.exp(used_fraction) - 1) / (np.exp(self.alphas[supply_node_id][0]) - 1)

    def fulfill(self, sequence: Sequence, inventory: Inventory, verbose: bool = False):
        """Fulfill a demand sequence using the multi-price balance algorithm.

        Returns:
            Tuple of (number_fulfillments, collected_rewards, lost_sales).
        """
        collected_rewards = 0
        lost_sales = 0
        total_fulfillments = 0
        number_fulfillments = defaultdict(int)

        current_inventories = inventory.initial_inventory.copy()

        for request in sequence.requests:
            demand_node = self.graph.demand_nodes[request.demand_node]
            supply_node_chosen = self.choose_supply_node(demand_node, current_inventories, inventory)
            if supply_node_chosen == -1:
                lost_sales += 1
                if verbose:
                    print(f'Demand from {demand_node.id} lost')
            else:
                if verbose:
                    print(f'Demand from {demand_node.id} fulfilled from warehouse {supply_node_chosen}')
                current_inventories[supply_node_chosen] -= 1
                number_fulfillments[supply_node_chosen, demand_node.id] += 1
                collected_rewards += self.graph.edges[supply_node_chosen, demand_node.id].reward
                total_fulfillments += 1

        return number_fulfillments, collected_rewards, lost_sales

    def choose_supply_node(self, demand_node: DemandNode, current_inventories, inventory: Inventory):
        """Select the supply node maximizing reward minus phi opportunity cost.

        Returns:
            Supply node ID, or -1 if no supply node has inventory.
        """
        used_fractions = {
            supply_node_id: 1 - extended_division(current_inventories[supply_node_id], inventory.initial_inventory[supply_node_id])
            for supply_node_id in demand_node.neighbors
        }
        pseudo_rewards = [
            [supply_node_id,
             self.graph.edges[supply_node_id, demand_node.id].reward - self.phi(supply_node_id, used_fractions[supply_node_id])]
            for supply_node_id in used_fractions
        ]

        pseudo_rewards.sort(key=lambda x: x[1], reverse=True)

        for supply_node_id, _ in pseudo_rewards:
            if current_inventories[supply_node_id] > 0:
                return supply_node_id

        return -1


class OffLpReSolvingFulfillment:
    """Fulfillment policy that periodically re-solves an offline LP.

    At specified epochs, solves the offline LP over training samples to obtain
    dual variables (shadow prices), then uses reward - dual as the score for
    choosing supply nodes.
    """

    def __init__(self, graph: Graph, solver: str = 'gurobi'):
        self.graph = graph
        self.programs = MathPrograms(graph, solver=solver)

    def fulfill(self, sequence: Sequence, inventory: Inventory,
                initial_dual_solution: Dict[int, float],
                train_sample: List[Sequence],
                re_solving_epochs: List[int] = None,
                filter_samples=False, verbose=False):
        """Fulfill a sequence with periodic offline LP re-solving.

        Args:
            sequence: Demand sequence.
            inventory: Initial inventory.
            initial_dual_solution: Starting dual variable values.
            train_sample: Training sequences for the offline LP.
            re_solving_epochs: Time steps at which to re-solve.
            filter_samples: If True, filter training samples by current demand node.
            verbose: Print decisions if True.

        Returns:
            Tuple of (number_fulfillments, collected_rewards, lost_sales).
        """
        collected_rewards = 0
        lost_sales = 0
        total_fulfillments = 0
        number_fulfillments = defaultdict(int)

        n_epochs = len(re_solving_epochs)
        current_epoch_index = 0
        next_epoch = re_solving_epochs[current_epoch_index]

        dual_variables = initial_dual_solution

        current_inventories = inventory.initial_inventory.copy()

        for t, request in enumerate(sequence.requests):
            demand_node = self.graph.demand_nodes[request.demand_node]

            if t == next_epoch:
                if verbose:
                    print(f'Re-solving at {t}')
                dual_variables = self.compute_dual_variables(
                    train_sample, t, current_inventories, demand_node, filter_samples
                )
                if current_epoch_index == len(re_solving_epochs) - 1:
                    next_epoch = len(sequence) + 1
                else:
                    current_epoch_index += 1
                    next_epoch = re_solving_epochs[current_epoch_index]

            supply_node_chosen = self.choose_supply_node(demand_node, current_inventories, dual_variables)
            if supply_node_chosen == -1:
                if verbose:
                    print(f'Demand from {demand_node.id} lost')
                lost_sales += 1
            else:
                if verbose:
                    print(f'Demand from {demand_node.id} fulfilled from warehouse {supply_node_chosen}')
                current_inventories[supply_node_chosen] -= 1
                number_fulfillments[supply_node_chosen, demand_node.id] += 1
                collected_rewards += self.graph.edges[supply_node_chosen, demand_node.id].reward
                total_fulfillments += 1

        return number_fulfillments, collected_rewards, lost_sales

    def choose_supply_node(self, demand_node: DemandNode, current_inventories, dual_variables):
        """Choose the supply node with the highest reward minus dual variable.

        Returns:
            Supply node ID, or -1 if no non-negative pseudo-reward exists.
        """
        supply_node_chosen = -1
        best_reward = 0

        for supply_node_id in demand_node.neighbors:
            if current_inventories[supply_node_id] > 0:
                pseudo_reward = self.graph.edges[supply_node_id, demand_node.id].reward - dual_variables[supply_node_id]
                if pseudo_reward >= best_reward:
                    best_reward = pseudo_reward
                    supply_node_chosen = supply_node_id

        return supply_node_chosen

    def compute_dual_variables(self, train_sample: List[Sequence], t, current_inventories,
                               current_demand_node, filter_samples):
        """Compute dual variables by solving the offline LP on remaining demand.

        Returns:
            Dict mapping supply_node_id to its average shadow price.
        """
        if filter_samples:
            filtered_sample = self.filter_sample(train_sample, t, current_demand_node)
            offline_lp, inventory_constraints = self.programs.offline_linear_program_fixed_inventory_partial_demand(
                filtered_sample, current_inventories, t + 1, current_demand_node.id
            )
            n_samples = len(filtered_sample)
        else:
            offline_lp, inventory_constraints = self.programs.offline_linear_program_fixed_inventory_partial_demand(
                train_sample, current_inventories, t + 1, None
            )
            n_samples = len(train_sample)

        offline_lp.optimize()

        duals = {
            supply_node_id: sum(
                inventory_constraints[supply_node_id, sample_index].Pi
                for sample_index in range(n_samples)
            ) / n_samples
            for supply_node_id in self.graph.supply_nodes
        }

        return duals

    def filter_sample(self, train_sample: List[Sequence], t, current_demand_node: DemandNode,
                      size_threshold=1):
        """Filter training samples to those matching the current demand node at time t.

        Falls back to the full sample if too few matches are found.
        """
        filtered_sample = []
        for sequence in train_sample:
            if sequence.requests[t].demand_node == current_demand_node.id:
                filtered_sample.append(sequence)

        if len(filtered_sample) >= size_threshold:
            return filtered_sample
        else:
            return train_sample


class FluLpReSolvingFulfillment:
    """Fulfillment policy that periodically re-solves a fluid LP.

    Uses cumulative average demand estimates from training data to formulate
    the fluid LP, then extracts dual variables as opportunity costs.
    """

    def __init__(self, graph: Graph, solver: str = 'gurobi'):
        self.graph = graph
        self.programs = MathPrograms(graph, solver=solver)

    def fulfill(self, sequence: Sequence, inventory: Inventory,
                initial_dual_solution: Dict[int, float],
                cumulative_average_demand: Dict[int, Dict[int, float]],
                re_solving_epochs: List[int] = None,
                filter_samples=False, verbose=False):
        """Fulfill a sequence with periodic fluid LP re-solving.

        Args:
            sequence: Demand sequence.
            inventory: Initial inventory.
            initial_dual_solution: Starting dual variable values.
            cumulative_average_demand: Maps time step t to average remaining
                demand per demand node from t onward.
            re_solving_epochs: Time steps at which to re-solve.
            verbose: Print decisions if True.

        Returns:
            Tuple of (number_fulfillments, collected_rewards, lost_sales).
        """
        collected_rewards = 0
        lost_sales = 0
        total_fulfillments = 0
        number_fulfillments = defaultdict(int)

        n_epochs = len(re_solving_epochs)
        current_epoch_index = 0
        next_epoch = re_solving_epochs[current_epoch_index]

        dual_variables = initial_dual_solution

        current_inventories = inventory.initial_inventory.copy()

        for t, request in enumerate(sequence.requests):
            demand_node = self.graph.demand_nodes[request.demand_node]

            if t == next_epoch:
                if verbose:
                    print(f'Re-solving at {t}')
                dual_variables = self.compute_dual_variables(t, current_inventories, cumulative_average_demand)
                if current_epoch_index == len(re_solving_epochs) - 1:
                    next_epoch = len(sequence) + 1
                else:
                    current_epoch_index += 1
                    next_epoch = re_solving_epochs[current_epoch_index]

            supply_node_chosen = self.choose_supply_node(demand_node, current_inventories, dual_variables)
            if supply_node_chosen == -1:
                if verbose:
                    print(f'Demand from {demand_node.id} lost')
                lost_sales += 1
            else:
                if verbose:
                    print(f'Demand from {demand_node.id} fulfilled from warehouse {supply_node_chosen}')
                current_inventories[supply_node_chosen] -= 1
                number_fulfillments[supply_node_chosen, demand_node.id] += 1
                collected_rewards += self.graph.edges[supply_node_chosen, demand_node.id].reward
                total_fulfillments += 1

        return number_fulfillments, collected_rewards, lost_sales

    def choose_supply_node(self, demand_node: DemandNode, current_inventories, dual_variables):
        """Choose the supply node with the highest reward minus dual variable.

        Returns:
            Supply node ID, or -1 if no non-negative pseudo-reward exists.
        """
        supply_node_chosen = -1
        best_reward = 0

        for supply_node_id in demand_node.neighbors:
            if current_inventories[supply_node_id] > 0:
                pseudo_reward = self.graph.edges[supply_node_id, demand_node.id].reward - dual_variables[supply_node_id]
                if pseudo_reward >= best_reward:
                    best_reward = pseudo_reward
                    supply_node_chosen = supply_node_id

        return supply_node_chosen

    def compute_dual_variables(self, t, current_inventories, cumulative_demand):
        """Compute dual variables by solving the fluid LP on remaining average demand.

        Returns:
            Dict mapping supply_node_id to its shadow price.
        """
        fluid_lp, inventory_constraints = self.programs.fluid_linear_program_fixed_inventory(
            cumulative_demand[t + 1], Inventory(current_inventories, 'aux')
        )
        fluid_lp.optimize()

        duals = {supply_node_id: inventory_constraints[supply_node_id].Pi for supply_node_id in self.graph.supply_nodes}

        return duals

    def filter_sample(self, train_sample: List[Sequence], t, current_demand_node: DemandNode,
                      size_threshold=1):
        """Filter training samples to those matching the current demand node at time t."""
        filtered_sample = []
        for sequence in train_sample:
            if sequence.requests[t].demand_node == current_demand_node.id:
                filtered_sample.append(sequence)

        if len(filtered_sample) >= size_threshold:
            return filtered_sample
        else:
            return train_sample


class ExtrapolationLpReSolvingFulfillment:
    """Fulfillment policy that re-solves using extrapolated demand estimates.

    At each re-solving epoch, estimates future demand by scaling the observed
    demand rate so far to the remaining time horizon, then solves the fluid LP.
    Does not require training data at fulfillment time.
    """

    def __init__(self, graph: Graph, solver: str = 'gurobi'):
        self.graph = graph
        self.programs = MathPrograms(graph, solver=solver)

    def fulfill(self, sequence: Sequence, inventory: Inventory,
                initial_dual_solution: Dict[int, float],
                re_solving_epochs: List[int] = None,
                filter_samples=False, verbose=False):
        """Fulfill a sequence with periodic extrapolation-based LP re-solving.

        Args:
            sequence: Demand sequence.
            inventory: Initial inventory.
            initial_dual_solution: Starting dual variable values (typically zeros).
            re_solving_epochs: Time steps at which to re-solve.
            verbose: Print decisions if True.

        Returns:
            Tuple of (number_fulfillments, collected_rewards, lost_sales).
        """
        collected_rewards = 0
        lost_sales = 0
        total_fulfillments = 0
        number_fulfillments = defaultdict(int)

        T = len(sequence)

        n_epochs = len(re_solving_epochs)
        current_epoch_index = 0
        next_epoch = re_solving_epochs[current_epoch_index]

        dual_variables = initial_dual_solution

        current_inventories = inventory.initial_inventory.copy()

        observed_demand = defaultdict(int)

        for t, request in enumerate(sequence.requests):
            demand_node = self.graph.demand_nodes[request.demand_node]
            observed_demand[demand_node.id] += 1

            if t == next_epoch:
                if verbose:
                    print(f'Re-solving at {t}')
                dual_variables = self.compute_dual_variables(t, current_inventories, observed_demand, T)
                if current_epoch_index == len(re_solving_epochs) - 1:
                    next_epoch = len(sequence) + 1
                else:
                    current_epoch_index += 1
                    next_epoch = re_solving_epochs[current_epoch_index]

            supply_node_chosen = self.choose_supply_node(demand_node, current_inventories, dual_variables)
            if supply_node_chosen == -1:
                if verbose:
                    print(f'Demand from {demand_node.id} lost')
                lost_sales += 1
            else:
                if verbose:
                    print(f'Demand from {demand_node.id} fulfilled from warehouse {supply_node_chosen}')
                current_inventories[supply_node_chosen] -= 1
                number_fulfillments[supply_node_chosen, demand_node.id] += 1
                collected_rewards += self.graph.edges[supply_node_chosen, demand_node.id].reward
                total_fulfillments += 1

        return number_fulfillments, collected_rewards, lost_sales

    def choose_supply_node(self, demand_node: DemandNode, current_inventories, dual_variables):
        """Choose the supply node with the highest reward minus dual variable.

        Returns:
            Supply node ID, or -1 if no non-negative pseudo-reward exists.
        """
        supply_node_chosen = -1
        best_reward = 0

        for supply_node_id in demand_node.neighbors:
            if current_inventories[supply_node_id] > 0:
                pseudo_reward = self.graph.edges[supply_node_id, demand_node.id].reward - dual_variables[supply_node_id]
                if pseudo_reward >= best_reward:
                    best_reward = pseudo_reward
                    supply_node_chosen = supply_node_id

        return supply_node_chosen

    def compute_dual_variables(self, t, current_inventories, observed_demand, T):
        """Compute dual variables using extrapolated demand.

        Extrapolates future demand as: (T - t - 1) * observed_count / (t + 1)
        for each demand node, then solves the fluid LP.

        Returns:
            Dict mapping supply_node_id to its shadow price.
        """
        interpolated_demand = {}

        for demand_node_id in self.graph.demand_nodes:
            if observed_demand[demand_node_id] == 0:
                interpolated_demand[demand_node_id] = 0
            else:
                interpolated_demand[demand_node_id] = (T - t - 1) * observed_demand[demand_node_id] / (t + 1)

        fluid_lp, inventory_constraints = self.programs.fluid_linear_program_fixed_inventory(
            interpolated_demand, Inventory(current_inventories, 'aux')
        )
        fluid_lp.optimize()

        duals = {supply_node_id: inventory_constraints[supply_node_id].Pi for supply_node_id in self.graph.supply_nodes}

        return duals


class DualMirrorDescentFulfillment:
    """Online fulfillment using dual mirror descent for adaptive opportunity costs.

    Updates dual variables (opportunity costs) after each fulfillment decision
    using either subgradient descent or multiplicative weights.
    """

    def __init__(self, graph: Graph):
        self.graph = graph

    def fulfill(self, sequence: Sequence, inventory: Inventory,
                dual_solution: List[float], rho: List[float],
                step_size: float, update_rule: str = 'subgradient_descent',
                verbose=False):
        """Fulfill a sequence with online dual updates.

        Args:
            sequence: Demand sequence.
            inventory: Initial inventory.
            dual_solution: Initial dual variable values.
            rho: Target consumption rate per supply node.
            step_size: Learning rate for dual updates.
            update_rule: 'subgradient_descent' or 'multiplicative_weights'.
            verbose: Print decisions if True.

        Returns:
            Tuple of (number_fulfillments, collected_rewards, lost_sales).
        """
        collected_rewards = 0
        lost_sales = 0
        total_fulfillments = 0
        number_fulfillments = defaultdict(int)

        current_dual_solution = dual_solution.copy()
        current_inventories = inventory.initial_inventory.copy()

        for request in sequence.requests:
            demand_node = self.graph.demand_nodes[request.demand_node]

            supply_node_chosen = self.choose_supply_node(demand_node, current_inventories, current_dual_solution)
            if supply_node_chosen == -1:
                lost_sales += 1
            else:
                if verbose:
                    print(f'Demand from {demand_node.id} fulfilled from warehouse {supply_node_chosen}')
                current_inventories[supply_node_chosen] -= 1
                number_fulfillments[supply_node_chosen, demand_node.id] += 1
                collected_rewards += self.graph.edges[supply_node_chosen, demand_node.id].reward
                total_fulfillments += 1

            current_dual_solution = self.dual_update(supply_node_chosen, current_dual_solution, rho, step_size, update_rule)

        return number_fulfillments, collected_rewards, lost_sales

    def choose_supply_node(self, demand_node: DemandNode, current_inventories, dual_solution):
        """Choose the supply node with the highest reward minus dual variable.

        Returns:
            Supply node ID, or -1 if no non-negative pseudo-reward exists or no inventory.
        """
        pseudo_rewards = [
            [supply_node_id,
             self.graph.edges[supply_node_id, demand_node.id].reward - dual_solution[supply_node_id]]
            for supply_node_id in demand_node.neighbors
        ]

        chosen_supply_node, max_pseudo_reward = max(pseudo_rewards, key=lambda x: x[1])

        if max_pseudo_reward >= 0 and current_inventories[chosen_supply_node] > 0:
            return chosen_supply_node
        else:
            return -1

    def dual_update(self, supply_node_chosen, dual_solution, rho, step_size, update_rule):
        """Update dual variables based on the fulfillment decision.

        The step direction is rho - e_{supply_node_chosen}, representing the
        gap between target and actual consumption.
        """
        step = rho.copy()
        step[supply_node_chosen] -= 1

        if update_rule == 'subgradient_descent':
            new_dual_solution = self.subgradient_descent_update(dual_solution, step, step_size)
        elif update_rule == 'multiplicative_weights':
            new_dual_solution = self.multiplicative_weights_update(dual_solution, step, step_size)

        return new_dual_solution

    def subgradient_descent_update(self, dual_solution, step, step_size):
        """Projected subgradient descent: dual -= step_size * step, clipped to >= 0."""
        new_dual_solution = [dual_sol_component for dual_sol_component in dual_solution]
        for supply_node_id in self.graph.supply_nodes:
            new_dual_solution[supply_node_id] = max(new_dual_solution[supply_node_id] - step_size * step[supply_node_id], 0)
        return new_dual_solution

    def multiplicative_weights_update(self, dual_solution, step, step_size):
        """Multiplicative weights: dual *= exp(-step_size * step)."""
        new_dual_solution = [dual_sol_component for dual_sol_component in dual_solution]
        for supply_node_id in self.graph.supply_nodes:
            new_dual_solution[supply_node_id] = new_dual_solution[supply_node_id] * np.exp(-step_size * step[supply_node_id])
        return new_dual_solution


def inventory_dict_to_tuple(inventory_dict: Dict[int, int], graph: Graph) -> Tuple[int]:
    """Convert an inventory dict to a tuple ordered by supply node IDs."""
    inventory_list = [inventory_dict[supply_node_id] for supply_node_id in graph.supply_nodes]
    return tuple(inventory_list)


class PolicyFulfillment:
    """Fulfills demand using a precomputed policy mapping (state, time, demand) -> action.

    Used to execute optimal policies computed by dynamic programming.
    """

    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def fulfill(self, sequence: Sequence, inventory: Inventory, policy, verbose=False):
        """Fulfill a sequence by looking up the optimal action at each state.

        Args:
            sequence: Demand sequence.
            inventory: Initial inventory.
            policy: Dict mapping (inventory_tuple, t, demand_node_id) to supply_node_id
                (-1 means no fulfillment).
            verbose: Print decisions if True.

        Returns:
            Tuple of (number_fulfillments, collected_rewards, lost_sales).
        """
        collected_rewards = 0
        lost_sales = 0
        total_fulfillments = 0
        number_fulfillments = defaultdict(int)

        current_inventories = inventory.initial_inventory.copy()

        t = 0
        for request in sequence.requests:
            demand_node = self.graph.demand_nodes[request.demand_node]
            inventory_state = inventory_dict_to_tuple(current_inventories, self.graph)
            supply_node_chosen = policy[inventory_state, t, demand_node.id]

            if supply_node_chosen == -1:
                lost_sales += 1
                if verbose:
                    print(f'Demand from {demand_node.id} lost')
            else:
                if verbose:
                    print(f'Demand from {demand_node.id} fulfilled from warehouse {supply_node_chosen}')
                current_inventories[supply_node_chosen] -= 1
                number_fulfillments[supply_node_chosen, demand_node.id] += 1
                collected_rewards += self.graph.edges[supply_node_chosen, demand_node.id].reward
                total_fulfillments += 1

            t += 1

        return number_fulfillments, collected_rewards, lost_sales


class FormatConverter:
    """Utility for converting Gurobi variable solutions to Python types."""

    def gurobi_to_int(self, gurobi_variables: dict):
        """Extract solution values from Gurobi variables into a plain dict."""
        output = {}
        for key in gurobi_variables:
            output[key] = float(gurobi_variables[key].X)
        return output
