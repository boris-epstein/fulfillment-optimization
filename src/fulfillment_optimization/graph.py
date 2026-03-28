import csv
import numpy as np

from typing import Dict, Tuple, List


class Graph:
    """Bipartite graph connecting supply nodes (warehouses) to demand nodes via edges.

    Each edge has an associated reward representing the value of fulfilling
    a demand node's request from a particular supply node.
    """

    def __init__(self) -> None:
        self.supply_nodes: Dict[int, Node] = {}
        self.demand_nodes: Dict[int, DemandNode] = {}
        self.edges: Dict[Tuple[int, int], Edge] = {}

    def read_from_files(self, supply_node_file: str, demand_node_file: str, edges_file: str):
        """Load graph structure from CSV files for supply nodes, demand nodes, and edges."""
        self.read_supply_nodes(supply_node_file)
        self.read_demand_nodes(demand_node_file)
        self.read_edges(edges_file)
        self.populate_neighbors()

    def read_supply_nodes(self, supply_nodes_file: str):
        """Read supply nodes from a CSV file with columns: index, (optional) x, y."""
        with open(supply_nodes_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                id = int(row['index'])
                if 'x' in row and 'y' in row:
                    self.supply_nodes[id] = Node(id, x=float(row['x']), y=float(row['y']))
                else:
                    self.supply_nodes[id] = Node(id)

    def read_demand_nodes(self, demand_nodes_file: str):
        """Read demand nodes from a CSV file with columns: index, (optional) x, y."""
        with open(demand_nodes_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                id = int(row['index'])
                if 'x' in row and 'y' in row:
                    self.demand_nodes[id] = DemandNode(id, x=float(row['x']), y=float(row['y']))
                else:
                    self.demand_nodes[id] = DemandNode(id)

    def read_edges(self, edges_file: str):
        """Read edges from a CSV file with columns: supply_node, demand_node, reward."""
        with open(edges_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                supply_node_id = int(row['supply_node'])
                demand_node_id = int(row['demand_node'])
                reward = float(row['reward'])
                self.edges[supply_node_id, demand_node_id] = Edge(supply_node_id, demand_node_id, reward)

    def populate_neighbors(self):
        """Set neighbor relationships for all nodes based on existing edges."""
        for supply_node_id, demand_node_id in self.edges:
            self.demand_nodes[demand_node_id].neighbors.add(supply_node_id)
            self.supply_nodes[supply_node_id].neighbors.add(demand_node_id)

    def populate_best_supply_nodes(self):
        """For each demand node, build a list of supply nodes sorted by reward (descending)."""
        reward_ordered_edges = [(edge.supply_node_id, edge.demand_node_id, edge.reward) for edge in self.edges.values()]
        reward_ordered_edges.sort(key=lambda x: x[2], reverse=True)
        for supply_node_id, demand_node_id, reward in reward_ordered_edges:
            self.demand_nodes[demand_node_id].best_supply_nodes.append(supply_node_id)

    def construct_priority_list(self, list_name: Tuple[str, str], scores: Dict[Tuple[int, int], float], allow_rejections=True):
        """Build a priority list for each demand node, ordering supply nodes by score.

        Args:
            list_name: Identifier for this priority list.
            scores: Map from (supply_node_id, demand_node_id) to score values.
            allow_rejections: If True, exclude supply nodes with negative scores
                from the priority list.
        """
        self.create_priority_lists(list_name)

        score_ordered_edges = [(supply_node_id, demand_node_id, scores[supply_node_id, demand_node_id])
                               for (supply_node_id, demand_node_id) in self.edges]
        score_ordered_edges.sort(key=lambda x: x[2], reverse=True)

        for supply_node_id, demand_node_id, score in score_ordered_edges:
            if (not allow_rejections) or (score >= 0):
                self.demand_nodes[demand_node_id].priority_lists[list_name].append(supply_node_id)

    def create_priority_lists(self, list_name: str):
        """Initialize empty priority lists with the given name for all demand nodes."""
        for demand_node in self.demand_nodes.values():
            demand_node.priority_lists[list_name] = []

    def add_supply_node(self, supply_node_id: int):
        """Add a supply node to the graph. Prints a warning if the ID already exists."""
        if supply_node_id in self.supply_nodes:
            print('Supply node id already exists')
        else:
            self.supply_nodes[supply_node_id] = Node(supply_node_id)

    def add_demand_node(self, demand_node_id: int):
        """Add a demand node to the graph. Prints a warning if the ID already exists."""
        if demand_node_id in self.demand_nodes:
            print('Demand node id already exists')
        else:
            self.demand_nodes[demand_node_id] = DemandNode(demand_node_id)

    def add_edge(self, supply_node_id: int, demand_node_id: int, reward: float):
        """Add an edge between existing supply and demand nodes with the given reward."""
        if supply_node_id in self.supply_nodes and demand_node_id in self.demand_nodes:
            self.edges[supply_node_id, demand_node_id] = Edge(supply_node_id, demand_node_id, reward)
        else:
            if supply_node_id not in self.supply_nodes:
                print('Supply node not in graph')
            if demand_node_id not in self.demand_nodes:
                print('Demand node not in graph')


class Node:
    """A node in the bipartite graph (used for supply nodes)."""

    def __init__(self, identifier: str, weight: float = 0, x: float = 0, y: float = 0) -> None:
        self.id = identifier
        self.neighbors = set()
        self.x = x
        self.y = y


class DemandNode(Node):
    """A demand-side node that tracks which supply nodes can serve it and their priority ordering."""

    def __init__(self, identifier: str, weight: float = 0, x: float = 0, y: float = 0):
        super().__init__(identifier, weight, x, y)
        self.best_supply_nodes = []
        self.priority_lists = {}


class Edge:
    """A directed edge from a supply node to a demand node with an associated reward."""

    def __init__(self, supply_node_id: int, demand_node_id: int, reward: float) -> None:
        self.supply_node_id = supply_node_id
        self.demand_node_id = demand_node_id
        self.reward = reward

    def __str__(self):
        return f'supply_node: {self.supply_node_id}, demand_node: {self.demand_node_id}, reward: {self.reward}'

    def __repr__(self):
        return f'supply_node: {self.supply_node_id}, demand_node: {self.demand_node_id}, reward: {self.reward}'


class RandomGraphGenerator:
    """Generates random bipartite graphs with controlled randomness via a seed."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed=seed)

    def two_valued_vertex_graph(self, num_supply_nodes: int, num_demand_nodes: int, given_vertex_values=None) -> Graph:
        """Generate a random bipartite graph where each supply node has two possible reward values.

        Each edge exists independently with probability 2/3, and its reward is
        randomly chosen from the supply node's two values.

        Args:
            num_supply_nodes: Number of supply nodes to create.
            num_demand_nodes: Number of demand nodes to create.
            given_vertex_values: Optional dict mapping supply node ID to a sorted
                pair [low, high] of reward values. If None, values are drawn uniformly.

        Returns:
            A Graph with randomly generated edges and rewards.
        """
        if given_vertex_values is None:
            vertex_values = {}
            for i in range(num_supply_nodes):
                vertex_values[i] = sorted([self.rng.uniform(), self.rng.uniform()])
        else:
            vertex_values = given_vertex_values

        graph = Graph()

        for supply_node_id in range(num_supply_nodes):
            graph.add_supply_node(supply_node_id)

        for demand_node_id in range(num_demand_nodes):
            graph.add_demand_node(demand_node_id)

        for demand_node_id in graph.demand_nodes:
            for supply_node_id in graph.supply_nodes:
                edge_exists = self.rng.binomial(1, 2/3)
                if edge_exists == 1:
                    value_choice = self.rng.binomial(1, 1/2)
                    graph.add_edge(supply_node_id, demand_node_id, vertex_values[supply_node_id][value_choice])

        graph.populate_neighbors()

        return graph
