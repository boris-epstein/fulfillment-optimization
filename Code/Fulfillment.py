
from Graph import Graph, DemandNode
from Demand import CorrelGenerator, Sequence, Request
from MathPrograms import MathPrograms
from Inventory import Inventory
from collections import defaultdict
import numpy as np
import gurobipy as gp
from typing import List, Dict, Tuple


"""For resolving

See # of arrivals in time interval. Assume that rate for the rest of the horizon. Scale variance accordingly.


Try:
- Change sigma and z
- Resolving for fluid
- Caps
- Indep demand


What to look at:



"""




class Fulfillment:

    def __init__(self, graph: Graph, train_generator: CorrelGenerator = None, resolve_seed: int = 0) -> None:
        
        self.graph = graph
        self.programs = MathPrograms(graph)

        if train_generator is not None:
            self.setup_resolve_generator(train_generator, resolve_seed)
    def setup_resolve_generator(self, train_generator: CorrelGenerator, resolve_seed: int):
        self.resolve_generator = CorrelGenerator(mean =train_generator.mean,
                                                 demand_nodes=train_generator.demand_nodes,
                                                 weights=train_generator.weights,
                                                 seed = resolve_seed,
                                                 distribution='deterministic'
                                            )
        
                

    ### FIXED LIST FULFILLMENT ###
    
    def fixed_list_fulfillment(self, sequence: Sequence, inventory, priority_list_name: Tuple[str,str], allow_rejections: bool = True, verbose = False):
        
        collected_rewards = 0
        lost_sales = 0
        total_fulfillments = 0
        number_fulfillments = defaultdict(int) # maps edges to number of times the edge was used
        
        current_inventories = inventory.initial_inventory.copy()
        demand_node_reachable = defaultdict(lambda: True) # dictionary that maps demand node ids to booleans saying if there is some supply node with inventory that can deliver to it
        best_supply_node_with_inventory = defaultdict(int) # maps demand node id to index of list that contains supply nodes ordered by edge rewards
        
        
        for request in sequence.requests:
            
            demand_node = self.graph.demand_nodes[request.demand_node]
            (best_supply_node_with_inventory, demand_node_reachable) =self.update_feasible_priorities(
                                                                                                demand_node=demand_node,
                                                                                                priority_list=demand_node.priority_lists[priority_list_name],
                                                                                                best_supply_node_with_inventory=best_supply_node_with_inventory,
                                                                                                current_inventories=current_inventories,
                                                                                                demand_node_reachable=demand_node_reachable
                                                                                            )
            if demand_node_reachable[demand_node.id]:
                supply_node_id = demand_node.priority_lists[priority_list_name][best_supply_node_with_inventory[demand_node.id]]
                current_inventories[supply_node_id] -=1
                number_fulfillments[ supply_node_id, demand_node.id] += 1
                
                collected_rewards += self.graph.edges[supply_node_id, demand_node.id].reward

                total_fulfillments += 1
                if verbose == True:
                    print(f'Demand from {demand_node.id} fulfilled from {supply_node_id}')
                
            else:
                lost_sales+=1
        
        return number_fulfillments, collected_rewards, lost_sales

    def update_feasible_priorities(
            self,
            demand_node: DemandNode,
            priority_list: List[int],
            best_supply_node_with_inventory: Dict[int, int],
            current_inventories: Dict[int,int],
            demand_node_reachable: Dict[int, bool]
        ):
        while best_supply_node_with_inventory[demand_node.id] < len(priority_list)  and current_inventories[ priority_list[best_supply_node_with_inventory[demand_node.id]] ] <=0:
            best_supply_node_with_inventory[demand_node.id] +=1
        if best_supply_node_with_inventory[demand_node.id] == len(priority_list):
            demand_node_reachable[demand_node.id] = False
        
        return best_supply_node_with_inventory, demand_node_reachable


    ### RESOLVING FULFILLMENT POLICIES ###
    
    def fulfillment_with_resolving(self,
                                   sequence: Sequence,
                                   inventory, priority_list_name: Tuple[str,str],
                                   resolve_times: List[float],
                                   demand_model: str,
                                   prior_mean: float,
                                   verbose = False,
                                   n_resolve_samples = 40):
        
        
        
        
        collected_rewards = 0
        lost_sales = 0
        total_fulfillments = 0
        number_fulfillments = defaultdict(int) # maps edges to number of times the edge was used
        
        current_inventories = inventory.initial_inventory.copy()
        priority_lists = {demand_node.id: demand_node.priority_lists[priority_list_name] for demand_node in self.graph.demand_nodes.values()}
        
        demand_node_reachable = defaultdict(lambda: True) # dictionary that maps demand node ids to booleans saying if there is some supply node with inventory that can deliver to it
        best_supply_node_with_inventory = defaultdict(int) # maps demand node id to index of list that contains supply nodes ordered by edge rewards
        
        resolve_point = resolve_times[0]
        next_resolve_index = 1
        
        num_arrivals =0
        for request in sequence.requests:
            
            
            ## if the arrival is after the resolve time, update priority lists
            
            
            
            if request.arrival_time > resolve_point:
                
                # UPDATE PRIORITY LISTS AND RESET THE BEST SUPPLY NODE INDICES
                
                priority_lists = self.re_compute_priority_lists(self, Inventory(current_inventories, name = 'current'), resolve_point, num_arrivals, priority_list_name, demand_model, prior_mean, n_resolve_samples)
               
                # 3. resent indices
                
                if next_resolve_index < len(resolve_times):
                    resolve_point = resolve_times[next_resolve_index]
                    next_resolve_index += 1
            
            
            num_arrivals +=1
            
            demand_node = self.graph.demand_nodes[request.demand_node]
            (best_supply_node_with_inventory, demand_node_reachable) =self.update_feasible_priorities(
                                                                                                demand_node=demand_node,
                                                                                                priority_list=priority_lists[demand_node.id],
                                                                                                best_supply_node_with_inventory=best_supply_node_with_inventory,
                                                                                                current_inventories=current_inventories,
                                                                                                demand_node_reachable=demand_node_reachable
                                                                                            )
            if demand_node_reachable[demand_node.id]:
                supply_node_id = demand_node.priority_lists[priority_list_name][best_supply_node_with_inventory[demand_node.id]]
                current_inventories[supply_node_id] -=1
                number_fulfillments[ supply_node_id, demand_node.id] += 1
                
                collected_rewards += self.graph.edges[supply_node_id, demand_node.id].reward

                total_fulfillments += 1
                if verbose == True:
                    print(f'Demand from {demand_node.id} fulfilled from {supply_node_id}')
                
            else:
                lost_sales+=1
        
        return number_fulfillments, collected_rewards, lost_sales
    

    def re_compute_priority_lists(self, current_inventory, resolve_point, num_arrivals, priority_list_name, demand_model, prior_mean, n_resolve_samples):
        
        # 1. compute shadow prices

        if priority_list_name[0] == 'offline_with_rejections':
            shadow_prices = self.compute_offline_shadow_prices(self, current_inventory, resolve_point, num_arrivals, demand_model, prior_mean,  n_resolve_samples)
            
        # 2. construct priority lists
            
        
        
        
        
        pass
    
    def compute_offline_shadow_prices(self, current_inventory, resolve_point, num_arrivals, demand_model, prior_mean, n_resolve_samples):
        
        # Update mean belief
        
        if demand_model == 'correl':
            new_mean = int(np.round(num_arrivals*(1-resolve_point)/resolve_point))
        
        if demand_model == 'deter':
            new_mean = int(prior_mean - num_arrivals)
        
        self.resolve_generator.set_mean(new_mean)
        
        resolve_samples = [self.resolve_generator.generate_sequence() for _ in range(n_resolve_samples)]
        
        offline_lp, inventory_constraints = self.programs.offline_linear_program_fixed_inventory(resolve_samples, current_inventory)
        
        pass