import numpy as np
from collections import defaultdict
from typing import Dict, List

from Graph import Node


class Request:
    """A single demand request arriving at a specific demand node."""

    def __init__(self, demand_node: int, arrival_time: float = 0.0) -> None:
        self.demand_node = demand_node
        self.arrival_time = arrival_time


class Sequence:
    """An ordered sequence of demand requests representing one sample path.

    Precomputes aggregate and leftover aggregate demand counts for use
    in LP formulations and re-solving policies.
    """

    def __init__(self, requests: List[Request], compute_aggregates=True) -> None:
        self.requests = requests
        self.length = len(requests)

        if compute_aggregates:
            self.compute_aggregates()
            self.compute_leftover_aggregates()

    def compute_aggregates(self):
        """Compute total demand count per demand node across the entire sequence."""
        self.aggregate_demand = defaultdict(int)
        for request in self.requests:
            self.aggregate_demand[request.demand_node] += 1

    def compute_leftover_aggregates(self):
        """Compute remaining demand from time t onward for each demand node.

        Stores a dict mapping time step t to a dict of demand node counts
        for requests at times t, t+1, ..., T-1.
        """
        self.leftover_aggregate_demand = {}
        T = self.length
        self.leftover_aggregate_demand[T] = defaultdict(int)
        self.leftover_aggregate_demand[T - 1] = defaultdict(int)
        self.leftover_aggregate_demand[T - 1][self.requests[T - 1].demand_node] += 1

        for t in range(T - 2, -1, -1):
            self.leftover_aggregate_demand[t] = self.leftover_aggregate_demand[t + 1].copy()
            self.leftover_aggregate_demand[t][self.requests[t].demand_node] += 1

    def __len__(self):
        return len(self.requests)

    def __str__(self) -> str:
        return str([req.demand_node for req in self.requests])


class CorrelGenerator:
    """Generates demand sequences with correlated total length.

    The total number of requests is drawn from a distribution (geometric,
    normal, exponential, or deterministic), then individual requests are
    sampled from the demand nodes according to fixed weights.

    Args:
        mean: Expected number of requests.
        demand_nodes: List of demand node IDs.
        weights: Probability weights for sampling demand nodes.
        seed: Random seed for reproducibility.
        distribution: Distribution for total length ('geometric', 'normal',
            'exponential', or 'deterministic').
        std_dev: Standard deviation (used only for 'normal' distribution).
    """

    def __init__(self, mean: float, demand_nodes: List[int], weights: List[float],
                 seed: int = 0, distribution: str = 'geometric', std_dev: float = 1) -> None:
        self.rng = np.random.default_rng(seed=seed)
        self.mean = mean
        self.std_dev = std_dev
        self.weights = weights
        self.demand_nodes = demand_nodes
        self.distribution = distribution

        if distribution == 'geometric':
            self.p = 1 / (1 + mean)

        self.average_demands = {}
        for demand_node in demand_nodes:
            self.average_demands[demand_node] = self.mean * self.weights[demand_node]

    def generate_sequence(self) -> Sequence:
        """Generate a single demand sequence with random length and node assignments."""
        if self.distribution == 'normal':
            T = max(0, int(round(self.rng.normal(loc=self.mean, scale=self.std_dev))))

        if self.distribution == 'exponential':
            T = int(round(self.rng.exponential(scale=self.mean)))

        if self.distribution == 'deterministic':
            T = int(round(self.mean))

        if self.distribution == 'geometric':
            T = self.rng.geometric(self.p) - 1

        reqs = self.rng.choice(a=self.demand_nodes, size=T, p=self.weights)

        if self.distribution == 'deterministic':
            arrival_times = np.arange(1 / (T + 1), 1, 1 / (T + 1))
        else:
            arrival_times = np.sort(self.rng.uniform(size=T))

        requests = [Request(reqs[t], arrival_times[t]) for t in range(T)]
        return Sequence(requests)

    def set_mean(self, new_mean: float):
        """Update the mean parameter for sequence length generation."""
        self.mean = new_mean


class TemporalIndependenceGenerator:
    """Generates sequences where each time step has an independent demand distribution.

    Args:
        demand_nodes: List of demand node IDs.
        p: Dict mapping time step t to a probability vector over demand nodes.
        seed: Random seed.
    """

    def __init__(self, demand_nodes: List[int], p: Dict[int, List[float]], seed: int):
        self.p = p
        self.demand_nodes = demand_nodes
        self.T = len(p)
        self.rng = np.random.default_rng(seed)

    def generate_sequence(self) -> Sequence:
        """Generate a sequence by independently sampling a demand node at each time step."""
        reqs = [None for t in range(self.T)]

        for t in range(self.T):
            choice = self.rng.choice(a=self.demand_nodes, p=self.p[t])
            reqs[t] = choice
        requests = [Request(reqs[t]) for t in range(self.T)]

        return Sequence(requests)


class IndepGenerator:
    """Generates sequences where each demand node has an independent random count.

    The number of requests per demand node is drawn from a specified distribution,
    then all requests are shuffled together with random arrival times.

    Args:
        means: List of means, one per demand node (must correspond to demand_nodes).
        demand_nodes: List of demand node IDs.
        seed: Random seed.
        distribution: Distribution for per-node counts ('geometric' or 'exponential').
        std_dev: Standard deviation (unused for geometric/exponential).
    """

    def __init__(self, means: List[float], demand_nodes: List[int], seed: int = 0,
                 distribution: str = 'geometric', std_dev: float = 1):
        self.rng = np.random.default_rng(seed=seed)
        self.means = means
        self.std_dev = std_dev
        self.demand_nodes = demand_nodes
        self.distribution = distribution
        self.average_demands = {}
        for demand_node_id in self.demand_nodes:
            self.average_demands[demand_node_id] = self.means[demand_node_id]

        if distribution == 'geometric':
            self.p = {}
            for demand_node_id in demand_nodes:
                self.p[demand_node_id] = 1 / (1 + means[demand_node_id])

    def generate_sequence(self) -> Sequence:
        """Generate a sequence with independent per-node demand counts, shuffled together."""
        total_demand = {}
        reqs = []

        for demand_node_id in self.demand_nodes:
            if self.distribution == 'exponential':
                total_demand[demand_node_id] = int(round(self.rng.exponential(scale=self.means[demand_node_id])))
            if self.distribution == 'geometric':
                total_demand[demand_node_id] = self.rng.geometric(self.p[demand_node_id]) - 1

            reqs += [demand_node_id for _ in range(total_demand[demand_node_id])]
        self.rng.shuffle(reqs)

        T = len(reqs)
        arrival_times = np.sort(self.rng.uniform(size=T))
        requests = [Request(reqs[t], arrival_times[t]) for t in range(T)]

        return Sequence(requests)


class RWGenerator:
    """Generates sequences where demand node probabilities evolve via a random walk.

    At each time step, multiplicative noise is applied to the probability weights,
    and the demand node is sampled from the renormalized distribution.

    Args:
        mean: Expected sequence length (or exact length for 'deterministic').
        demand_nodes: List of demand node IDs.
        seed: Random seed.
        distribution: Length distribution ('geometric' or 'deterministic').
        step_size: Controls the magnitude of the random walk steps.
    """

    def __init__(self, mean: float, demand_nodes: List[int], seed: int = 0,
                 distribution: str = 'deterministic', step_size: int = 1) -> None:
        self.rng = np.random.default_rng(seed=seed)
        self.mean = mean

        if distribution == 'geometric':
            self.p = 1 / (1 + mean)

        self.step_size = step_size
        self.distribution = distribution
        self.demand_nodes = demand_nodes

    def generate_sequence(self) -> Sequence:
        """Generate a sequence with random-walk-driven demand probabilities."""
        if self.distribution == 'geometric':
            T = self.rng.geometric(self.p) - 1

        if self.distribution == 'deterministic':
            T = int(self.mean)

        walk = np.array([1.0 for demand_node in self.demand_nodes])

        reqs = []
        for t in range(T):
            for demand_node in self.demand_nodes:
                bit = self.rng.binomial(1, 1/2)
                step = 2 * (bit - 1/2) * self.step_size
                walk[demand_node] = walk[demand_node] * np.exp(step)

            req = self.rng.choice(a=self.demand_nodes, p=walk / walk.sum())
            reqs.append(req)

        if self.distribution == 'deterministic':
            arrival_times = np.arange(1 / (T + 1), 1, 1 / (T + 1))
        else:
            arrival_times = np.sort(self.rng.uniform(size=T))

        requests = [Request(reqs[t], arrival_times[t]) for t in range(T)]
        return Sequence(requests)


class MarkovianGenerator:
    """Generates sequences where demand nodes follow a Markov chain.

    Args:
        T: Fixed sequence length.
        demand_nodes: List of demand node IDs.
        transition_matrix: Row-stochastic matrix of transition probabilities.
        initial_distribution: Probability vector for the first demand node.
        seed: Random seed.
    """

    def __init__(self, T, demand_nodes: List[int], transition_matrix,
                 initial_distribution, seed: int = 0):
        self.demand_nodes = demand_nodes
        self.transition_matrix = transition_matrix
        self.initial_distribution = initial_distribution
        self.rng = np.random.default_rng(seed=seed)
        self.seed = seed
        self.T = T

    def generate_sequence(self):
        """Generate a Markov chain sequence of demand nodes."""
        arrival_times = np.arange(1 / (self.T + 1), 1, 1 / (self.T + 1))
        reqs = []
        latest_arrival = self.rng.choice(a=self.demand_nodes, p=self.initial_distribution)
        reqs.append(latest_arrival)
        for t in range(1, self.T):
            latest_arrival = self.rng.choice(a=self.demand_nodes, p=self.transition_matrix[latest_arrival, :])
            reqs.append(latest_arrival)

        requests = [Request(reqs[t], arrival_times[t]) for t in range(self.T)]
        return Sequence(requests)


class HiddenMarkovGenerator:
    """Generates sequences where demand is driven by a hidden Markov model.

    A latent state evolves according to a Markov chain, and each state emits
    a demand node according to a state-specific distribution.

    Args:
        T: Fixed sequence length.
        demand_nodes: List of demand node IDs.
        states: List of hidden state IDs.
        state_distribution: Dict mapping state ID to emission probability vector.
        initial_distribution: Probability vector over initial hidden states.
        transition_matrix: Row-stochastic matrix for hidden state transitions.
        seed: Random seed.
    """

    def __init__(self, T: int, demand_nodes, states: List[int],
                 state_distribution: Dict[int, List[float]],
                 initial_distribution: List[float],
                 transition_matrix: np.array, seed: int = 0):
        self.states = states
        self.demand_nodes = demand_nodes
        self.state_distribution = state_distribution
        self.initial_distribution = initial_distribution
        self.transition_matrix = transition_matrix
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.T = T

    def generate_sequence(self):
        """Generate a sequence by sampling from the HMM emission distributions."""
        arrival_times = np.arange(1 / (self.T + 1), 1, 1 / (self.T + 1))
        reqs = []

        current_state = self.rng.choice(a=self.states, p=self.initial_distribution)

        for t in range(self.T):
            latest_arrival = self.rng.choice(a=self.demand_nodes, p=self.state_distribution[current_state])
            reqs.append(latest_arrival)
            current_state = self.rng.choice(a=self.states, p=self.transition_matrix[current_state, :])

        requests = [Request(reqs[t], arrival_times[t]) for t in range(self.T)]
        return Sequence(requests)


class RandomDistributionGenerator:
    """Generates random probability distributions for use in experiments.

    Args:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed=seed)

    def generate_indep(self, num_demand_nodes, num_periods, bias=None):
        """Generate independent per-period probability vectors over demand nodes.

        Args:
            num_demand_nodes: Number of demand nodes.
            num_periods: Number of time periods T.
            bias: Optional array added to the exponential draws before normalization.

        Returns:
            Dict mapping time step t to a probability vector of length num_demand_nodes.
        """
        if bias is None:
            bias = np.array([0 for demand_node_id in range(num_demand_nodes)])
        else:
            bias = np.array(bias)

        p = {}
        for t in range(num_periods):
            exp_vector = self.rng.exponential(1, size=num_demand_nodes) + bias
            p[t] = exp_vector / np.sum(exp_vector)

        return p

    def generate_markov(self, num_demand_nodes, diagonal_bias: int = 0, initial_dist_bias=None):
        """Generate a random Markov chain transition matrix and initial distribution.

        Args:
            num_demand_nodes: Number of states (demand nodes).
            diagonal_bias: Extra weight on the diagonal to encourage self-transitions.
            initial_dist_bias: Optional bias for the initial distribution.

        Returns:
            Tuple of (initial_distribution, transition_matrix).
        """
        if initial_dist_bias is None:
            initial_dist_bias = np.zeros(num_demand_nodes)
        else:
            initial_dist_bias = np.array(initial_dist_bias)

        exp_vector = self.rng.exponential(1, size=num_demand_nodes) + initial_dist_bias
        initial_distribution = np.ones(num_demand_nodes) / num_demand_nodes

        transition_matrix = np.zeros((num_demand_nodes, num_demand_nodes))

        for demand_node_id in range(num_demand_nodes):
            bias_term = np.zeros(num_demand_nodes)
            bias_term[demand_node_id] = diagonal_bias
            exp_vector = self.rng.exponential(1, size=num_demand_nodes) + bias_term
            transition_matrix[demand_node_id] = exp_vector / np.sum(exp_vector)

        return initial_distribution, transition_matrix
