import gurobipy as gp
from Demand import Sequence
from Graph import Graph
from typing import Dict, List


class MathPrograms:
    
    def __init__(self, graph: Graph):
        self.graph = graph
        
    def fluid_linear_program_fixed_inventory(self, average_demand : Dict[int, float], inventory, scaling_factor: float = 1.0, demand_node_id_to_add_1: int = None):
        model = gp.Model('Fluid_fixed_inventory')

        # VARIABLES
        
        y = {} #fulfillment variables

        for supply_node_id, demand_node_id in self.graph.edges:
            y[supply_node_id, demand_node_id] = model.addVar( lb = 0, obj = self.graph.edges[supply_node_id, demand_node_id].reward, name = f'flow_{supply_node_id}_{demand_node_id}' )

        # CONSTRAINTS
        inventory_constraints = {}

        for demand_node_id in self.graph.demand_nodes:
            demand_node = self.graph.demand_nodes[demand_node_id]
            if demand_node.id == demand_node_id_to_add_1:
                model.addConstr( gp.quicksum( y[supply_node_id,demand_node_id] for supply_node_id in demand_node.neighbors) <= average_demand[demand_node_id]*scaling_factor + 1 )
            else:
                model.addConstr( gp.quicksum( y[supply_node_id,demand_node_id] for supply_node_id in demand_node.neighbors) <= average_demand[demand_node_id]*scaling_factor )

        for supply_node_id in self.graph.supply_nodes:
            supply_node = self.graph.supply_nodes[supply_node_id]
            inventory_constraints[supply_node_id] = model.addConstr( gp.quicksum( y[supply_node_id,demand_node_id] for demand_node_id in supply_node.neighbors ) <= inventory.initial_inventory[supply_node_id] )

            
        model.ModelSense = -1 # maximize
        
        model.Params.LogToConsole = 0

        model.update()

        return model, inventory_constraints
    
    
    
    def fluid_linear_program_variable_inventory(self,
                                                average_demand : Dict[int, float],
                                                total_inventory: int,
                                                scaling_factor: float = 1.0
                                            ):
        model = gp.Model('Fluid_variable_inventory')

        # VARIABLES
        
        y = {} # fulfillment variables
        
        x = {} # inventory variables

        for supply_node_id in self.graph.supply_nodes:
            
            x[supply_node_id] = model.addVar(lb = 0, name = f'inventory_{supply_node_id}')
            
        for supply_node_id, demand_node_id in self.graph.edges:
            
            y[supply_node_id, demand_node_id] = model.addVar(
                lb = 0, obj = self.graph.edges[supply_node_id,demand_node_id].reward,
                name = f'flow_{supply_node_id}_{demand_node_id}'
            )

        # CONSTRAINTS

        model.addConstr( gp.quicksum( x[supply_node_id] for supply_node_id in self.graph.supply_nodes ) == total_inventory )

        for demand_node_id in self.graph.demand_nodes:
            demand_node = self.graph.demand_nodes[demand_node_id]
            model.addConstr( gp.quicksum( y[supply_node_id, demand_node_id] for supply_node_id in demand_node.neighbors) <= average_demand[demand_node_id]*scaling_factor )

        for supply_node_id in self.graph.supply_nodes:
            supply_node = self.graph.supply_nodes[supply_node_id]
            model.addConstr( gp.quicksum( y[supply_node_id,demand_node_id] for demand_node_id in supply_node.neighbors ) <= x[supply_node_id] )

            
        model.ModelSense = -1 # maximize
        
        model.Params.LogToConsole = 0

        model.update()

        return model, x
    
    
    
    
    def offline_linear_program_fixed_inventory(self, demand_samples: List[Sequence], inventory):
        
        model = gp.Model('Offline_fixed_inventory')

        # VARIABLES
        
        y = {} #fulfillment variables

        for supply_node_id, demand_node_id in self.graph.edges:
            for sample_index in range(len(demand_samples)):
                y[supply_node_id, demand_node_id, sample_index] = model.addVar(
                    lb = 0,
                    obj = self.graph.edges[supply_node_id,demand_node_id].reward,
                    name = f'flow_{supply_node_id}_{demand_node_id}_{sample_index}'
                )

        # CONSTRAINTS
        
        inventory_constraints = {}

        for demand_node_id in self.graph.demand_nodes:
            demand_node = self.graph.demand_nodes[demand_node_id]
            for sample_index, demand_sample in enumerate(demand_samples):
                model.addConstr(
                    gp.quicksum( y[supply_node_id,demand_node_id, sample_index] for supply_node_id in demand_node.neighbors) <= demand_sample.aggregate_demand[demand_node_id] 
                )

        for supply_node_id in self.graph.supply_nodes:
            supply_node = self.graph.supply_nodes[supply_node_id]
            for sample_index in range(len(demand_samples)):
                inventory_constraints[supply_node_id, sample_index] = model.addConstr(
                    gp.quicksum( y[supply_node_id,demand_node_id,sample_index] for demand_node_id in supply_node.neighbors ) <= inventory.initial_inventory[supply_node_id]
                )

            
        model.ModelSense = -1 # maximize
        
        model.Params.LogToConsole = 0

        model.update()

        return model, inventory_constraints
        
    def offline_linear_program_fixed_inventory_partial_demand(self, demand_samples: List[Sequence], current_inventory, time_step, current_demand_node = -1):
        
        model = gp.Model('Offline_fixed_inventory')

        # VARIABLES
        
        y = {} #fulfillment variables

        for supply_node_id, demand_node_id in self.graph.edges:
            for sample_index in range(len(demand_samples)):
                y[supply_node_id, demand_node_id, sample_index] = model.addVar(
                    lb = 0,
                    obj = self.graph.edges[supply_node_id,demand_node_id].reward,
                    name = f'flow_{supply_node_id}_{demand_node_id}_{sample_index}'
                )

        # CONSTRAINTS
        
        inventory_constraints = {}

        for demand_node_id in self.graph.demand_nodes:
            demand_node = self.graph.demand_nodes[demand_node_id]
            for sample_index, demand_sample in enumerate(demand_samples):
                if demand_node_id == current_demand_node:
                    model.addConstr(
                        gp.quicksum( y[supply_node_id,demand_node_id, sample_index] for supply_node_id in demand_node.neighbors) <= demand_sample.leftover_aggregate_demand[time_step][demand_node_id] + 1
                    )
                else:
                    model.addConstr(
                        gp.quicksum( y[supply_node_id,demand_node_id, sample_index] for supply_node_id in demand_node.neighbors) <= demand_sample.leftover_aggregate_demand[time_step][demand_node_id] 
                    )
        
        for supply_node_id in self.graph.supply_nodes:
            supply_node = self.graph.supply_nodes[supply_node_id]
            for sample_index in range(len(demand_samples)):
                inventory_constraints[supply_node_id, sample_index] = model.addConstr(
                    gp.quicksum( y[supply_node_id,demand_node_id,sample_index] for demand_node_id in supply_node.neighbors ) <= current_inventory[supply_node_id]
                )

            
        model.ModelSense = -1 # maximize
        
        model.Params.LogToConsole = 0

        model.update()

        return model, inventory_constraints
        
     
        
    def offline_linear_program_variable_inventory(self, demand_samples: List[Sequence], total_inventory: int):
        
        model = gp.Model('Offline_variable_inventory')

        # VARIABLES
        
        y = {} #fulfillment variables
        
        x = {} # inventory variables

        for supply_node_id in self.graph.supply_nodes:
            
            x[supply_node_id] = model.addVar(lb = 0, name = f'inventory_{supply_node_id}')
            
        for supply_node_id, demand_node_id in self.graph.edges:
            for sample_index in range(len(demand_samples)):
                y[supply_node_id, demand_node_id, sample_index] = model.addVar(
                    lb = 0,
                    obj = self.graph.edges[supply_node_id,demand_node_id].reward,
                    name = f'flow_{supply_node_id}_{demand_node_id}_{sample_index}'
                )

        # CONSTRAINTS
        
        
        model.addConstr( gp.quicksum( x[supply_node_id] for supply_node_id in self.graph.supply_nodes ) == total_inventory )
        
        inventory_constraints = {}

        for demand_node_id in self.graph.demand_nodes:
            demand_node = self.graph.demand_nodes[demand_node_id]
            for sample_index, demand_sample in enumerate(demand_samples):
                model.addConstr(
                    gp.quicksum( y[supply_node_id,demand_node_id, sample_index] for supply_node_id in demand_node.neighbors) <= demand_sample.aggregate_demand[demand_node_id] 
                )

        for supply_node_id in self.graph.supply_nodes:
            supply_node = self.graph.supply_nodes[supply_node_id]
            for sample_index in range(len(demand_samples)):
                inventory_constraints[supply_node_id, sample_index] = model.addConstr(
                    gp.quicksum( y[supply_node_id,demand_node_id,sample_index] for demand_node_id in supply_node.neighbors ) <= x[supply_node_id]
                )

            
        model.ModelSense = -1 # maximize
        
        model.Params.LogToConsole = 0

        model.update()

        return model, x
        