import csv
import numpy as np

from typing import Dict, Tuple, List

class Graph:

    def __init__(self, ) -> None:
        
        self.supply_nodes: Dict[int, Node] = {} # warehouses
        self.demand_nodes: Dict[int, DemandNode] = {} #demand nodes
        self.edges: Dict[Tuple[int,int], Edge] = {} # fulfillment routes


        
        # self.construct_priority_list('myopic', {(edge.supply_node_id, edge.demand_node_id):edge.reward for edge in self.edges.values()})

    def read_from_files(self, supply_node_file: str, demand_node_file: str, edges_file: str):
        self.read_supply_nodes(supply_node_file)
        self.read_demand_nodes(demand_node_file)
        self.read_edges(edges_file)
        self.populate_neighbors()

    def read_supply_nodes(self, supply_nodes_file: str):
        with open(supply_nodes_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                id = int(row['index'])
                if 'x' in row and 'y' in row:
                    self.supply_nodes[id] = Node(id, x = float(row['x']), y = float(row['y']))
                else:
                    self.supply_nodes[id] = Node(id)
                
    def read_demand_nodes(self, demand_nodes_file: str):
        with open(demand_nodes_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                id = int(row['index'])
                if 'x' in row and 'y' in row:
                    self.demand_nodes[id] = DemandNode(id, x = float(row['x']), y = float(row['y']))
                else:
                    self.demand_nodes[id] = DemandNode(id)

    
    def read_edges(self, edges_file: str):
        with open(edges_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                supply_node_id = int(row['supply_node'])
                demand_node_id = int(row['demand_node'])
                reward = float(row['reward'])
                self.edges[supply_node_id, demand_node_id]= Edge(supply_node_id, demand_node_id, reward)


    def populate_neighbors(self):
        for supply_node_id, demand_node_id in self.edges:
            self.demand_nodes[demand_node_id].neighbors.add(supply_node_id)
            self.supply_nodes[supply_node_id].neighbors.add(demand_node_id)
        
    def populate_best_supply_nodes(self):
        reward_ordered_edges = [ (edge.supply_node_id, edge.demand_node_id, edge.reward) for edge in self.edges.values()]
        reward_ordered_edges.sort(key = lambda x: x[2], reverse=True)
        for supply_node_id, demand_node_id, reward in reward_ordered_edges:
            self.demand_nodes[demand_node_id].best_supply_nodes.append(supply_node_id)
            
        for demand_node_id in self.demand_nodes:
            demand_node = self.demand_nodes[demand_node_id]
            # demand_node.priority_lists['myopic'] = demand_node.best_supply_nodes

    def construct_priority_list(self, list_name: Tuple[str,str], scores: Dict[Tuple[int,int],float], allow_rejections = True):
        """
        Args:
            score (Dict[Tuple[int,int],float]): _description_
            allow_rejections (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
            
        If allow_rejections is set to true, then for any supply-demand node pairs with negative score, the supply node won't be included in the demand node's list
        """
        
        
        self.create_priority_lists(list_name)
        
        score_ordered_edges = [(supply_node_id, demand_node_id, scores[supply_node_id, demand_node_id])for (supply_node_id, demand_node_id) in self.edges]
        score_ordered_edges.sort(key = lambda x: x[2], reverse=True)
        
        for supply_node_id, demand_node_id, score in score_ordered_edges:
            if (not allow_rejections) or (score>=0):
                self.demand_nodes[demand_node_id].priority_lists[list_name].append(supply_node_id)
                
    def create_priority_lists(self, list_name: str):
        for demand_node in self.demand_nodes.values():
            demand_node.priority_lists[list_name] = []

    def add_supply_node(self, supply_node_id: int):
        if supply_node_id in self.supply_nodes:
            print('Supply node id already exists')
        else:
            self.supply_nodes[supply_node_id] = Node(supply_node_id)
            
    def add_demand_node(self, demand_node_id: int):
        if demand_node_id in self.demand_nodes:
            print('Demand node id already exists')
        else:
            self.demand_nodes[demand_node_id] = DemandNode(demand_node_id)

    def add_edge(self, supply_node_id: int, demand_node_id: int, reward: float):
        
        if supply_node_id in self.supply_nodes and demand_node_id in self.demand_nodes:
            self.edges[supply_node_id, demand_node_id] = Edge(supply_node_id, demand_node_id, reward)
        else:
            if supply_node_id not in self.supply_nodes:
                print('Supply node not in graph')
            if demand_node_id not in self.supply_nodes:
                print('Demand node not in graph')
class Node:
    def __init__(self, identifier: str, weight:float = 0, x: float = 0, y: float = 0) -> None:
        self.id = identifier
        self.neighbors = set()
        self.x = x
        self.y = y

class DemandNode(Node):
    def __init__(self, identifier: str, weight: float =0, x: float = 0, y: float = 0):
        super().__init__(identifier, weight, x, y)
        self.best_supply_nodes = [] # list of supply node ids that can fulfill the demand node, ordered by the reward (descending)
        self.priority_lists = {}

class Edge:
    def __init__(self,supply_node_id: int, demand_node_id: int, reward: float) -> None:
        self.supply_node_id = supply_node_id
        self.demand_node_id = demand_node_id
        self.reward = reward
    def __str__(self):
        return f'supply_node: {self.supply_node_id}, demand_node: {self.demand_node_id}, reward: {self.reward}'
    def __repr__(self):
        return f'supply_node: {self.supply_node_id}, demand_node: {self.demand_node_id}, reward: {self.reward}'
        
        
class RandomGraphGenerator:
    
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed=seed)
    
    def two_valued_vertex_graph(self, num_supply_nodes: int, num_demand_nodes:int, given_vertex_values = None) -> Graph:
        
        if given_vertex_values is None:
            vertex_values = {}
            for i in range(num_supply_nodes):
                vertex_values[i] = sorted( [self.rng.uniform(), self.rng.uniform()])
        
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
                    value_choice = self.rng.binomial(1,1/2)
                    graph.add_edge(supply_node_id, demand_node_id, vertex_values[supply_node_id][value_choice])
        
        graph.populate_neighbors()
        
        return graph
        


if __name__=='__main__':
    # graph = Graph()
    # graph.read_from_files(f'three_node_graph_supplynodes.csv', f'three_node_graph_demandnodes.csv', f'three_node_graph_rewards.csv')
    # print('SUPPLY')
    # print(graph.supply_nodes)
    # print('DEMAND')
    # print(graph.demand_nodes)
    # print('EDGES')
    # print(graph.edges)
    # print(len(graph.demand_nodes))
    # for id in graph.demand_nodes:
    #     dn = graph.demand_nodes[id]
    #     print(dn.neighbors)
    
    vertex_values = {}
    vertex_values[0] = [0.1, 0.9]
    vertex_values[1] = [0.2, 0.6]
    vertex_values[2] = [0.4,0.8]
    vertex_values[3] = [0.3,0.8]
    
    graph_generator = RandomGraphGenerator(seed = 12)
    
    
    for i in range(10):
        print(i)
        graph = graph_generator.two_valued_vertex_graph(4, 15, vertex_values)
        for supply_node_id in graph.supply_nodes:
            sn = graph.supply_nodes[supply_node_id]
            
            for demand_node_id in sn.neighbors:
                print(graph.edges[supply_node_id,demand_node_id].reward)
            print('')
        print('')
    