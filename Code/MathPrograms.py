import gurobipy as gp
from Demand import Sequence
from Graph import Graph
from typing import Dict, List

gp.setParam('LogToConsole', 0)
gp.setParam('OutputFlag', 0)


class MathPrograms:
    """Collection of Gurobi LP formulations for fulfillment optimization.

    Provides fluid (expected demand) and offline (sample-based) linear programs
    with either fixed or variable inventory placement.
    """

    def __init__(self, graph: Graph):
        self.graph = graph

    def fluid_linear_program_fixed_inventory(self, average_demand: Dict[int, float], inventory,
                                             scaling_factor: float = 1.0,
                                             demand_node_id_to_add_1: int = None):
        """Build a fluid LP that maximizes reward given fixed inventory and expected demand.

        Args:
            average_demand: Expected demand per demand node.
            inventory: Inventory object with fixed initial_inventory per supply node.
            scaling_factor: Multiplier applied to demand (for rescaling).
            demand_node_id_to_add_1: If set, adds 1 to this demand node's capacity
                (used for marginal value computation).

        Returns:
            Tuple of (model, inventory_constraints) where inventory_constraints maps
            supply_node_id to the corresponding Gurobi constraint.
        """
        model = gp.Model('Fluid_fixed_inventory')

        y = {}
        for supply_node_id, demand_node_id in self.graph.edges:
            y[supply_node_id, demand_node_id] = model.addVar(
                lb=0, obj=self.graph.edges[supply_node_id, demand_node_id].reward,
                name=f'flow_{supply_node_id}_{demand_node_id}'
            )

        inventory_constraints = {}

        for demand_node_id in self.graph.demand_nodes:
            demand_node = self.graph.demand_nodes[demand_node_id]
            if demand_node.id == demand_node_id_to_add_1:
                model.addConstr(
                    gp.quicksum(y[supply_node_id, demand_node_id] for supply_node_id in demand_node.neighbors)
                    <= average_demand[demand_node_id] * scaling_factor + 1
                )
            else:
                model.addConstr(
                    gp.quicksum(y[supply_node_id, demand_node_id] for supply_node_id in demand_node.neighbors)
                    <= average_demand[demand_node_id] * scaling_factor
                )

        for supply_node_id in self.graph.supply_nodes:
            supply_node = self.graph.supply_nodes[supply_node_id]
            inventory_constraints[supply_node_id] = model.addConstr(
                gp.quicksum(y[supply_node_id, demand_node_id] for demand_node_id in supply_node.neighbors)
                <= inventory.initial_inventory[supply_node_id]
            )

        model.ModelSense = -1  # maximize
        model.Params.LogToConsole = 0
        model.update()

        return model, inventory_constraints

    def fluid_linear_program_variable_inventory(self, average_demand: Dict[int, float],
                                                total_inventory: int,
                                                scaling_factor: float = 1.0):
        """Build a fluid LP that jointly optimizes inventory placement and fulfillment.

        Args:
            average_demand: Expected demand per demand node.
            total_inventory: Total units of inventory to distribute across supply nodes.
            scaling_factor: Multiplier applied to demand.

        Returns:
            Tuple of (model, x) where x maps supply_node_id to its inventory variable.
        """
        model = gp.Model('Fluid_variable_inventory')

        y = {}
        x = {}

        for supply_node_id in self.graph.supply_nodes:
            x[supply_node_id] = model.addVar(lb=0, name=f'inventory_{supply_node_id}')

        for supply_node_id, demand_node_id in self.graph.edges:
            y[supply_node_id, demand_node_id] = model.addVar(
                lb=0, obj=self.graph.edges[supply_node_id, demand_node_id].reward,
                name=f'flow_{supply_node_id}_{demand_node_id}'
            )

        model.addConstr(
            gp.quicksum(x[supply_node_id] for supply_node_id in self.graph.supply_nodes) == total_inventory
        )

        for demand_node_id in self.graph.demand_nodes:
            demand_node = self.graph.demand_nodes[demand_node_id]
            model.addConstr(
                gp.quicksum(y[supply_node_id, demand_node_id] for supply_node_id in demand_node.neighbors)
                <= average_demand[demand_node_id] * scaling_factor
            )

        for supply_node_id in self.graph.supply_nodes:
            supply_node = self.graph.supply_nodes[supply_node_id]
            model.addConstr(
                gp.quicksum(y[supply_node_id, demand_node_id] for demand_node_id in supply_node.neighbors)
                <= x[supply_node_id]
            )

        model.ModelSense = -1  # maximize
        model.Params.LogToConsole = 0
        model.update()

        return model, x

    def offline_linear_program_fixed_inventory(self, demand_samples: List[Sequence], inventory):
        """Build an offline LP over multiple demand samples with fixed inventory.

        Maximizes total reward across all samples subject to per-sample demand
        constraints and shared inventory constraints.

        Args:
            demand_samples: List of Sequence objects (each with aggregate_demand).
            inventory: Inventory object with fixed initial_inventory.

        Returns:
            Tuple of (model, inventory_constraints) where inventory_constraints maps
            (supply_node_id, sample_index) to the corresponding constraint.
        """
        model = gp.Model('Offline_fixed_inventory')

        y = {}
        for supply_node_id, demand_node_id in self.graph.edges:
            for sample_index in range(len(demand_samples)):
                y[supply_node_id, demand_node_id, sample_index] = model.addVar(
                    lb=0, obj=self.graph.edges[supply_node_id, demand_node_id].reward,
                    name=f'flow_{supply_node_id}_{demand_node_id}_{sample_index}'
                )

        inventory_constraints = {}

        for demand_node_id in self.graph.demand_nodes:
            demand_node = self.graph.demand_nodes[demand_node_id]
            for sample_index, demand_sample in enumerate(demand_samples):
                model.addConstr(
                    gp.quicksum(y[supply_node_id, demand_node_id, sample_index] for supply_node_id in demand_node.neighbors)
                    <= demand_sample.aggregate_demand[demand_node_id]
                )

        for supply_node_id in self.graph.supply_nodes:
            supply_node = self.graph.supply_nodes[supply_node_id]
            for sample_index in range(len(demand_samples)):
                inventory_constraints[supply_node_id, sample_index] = model.addConstr(
                    gp.quicksum(y[supply_node_id, demand_node_id, sample_index] for demand_node_id in supply_node.neighbors)
                    <= inventory.initial_inventory[supply_node_id]
                )

        model.ModelSense = -1  # maximize
        model.Params.LogToConsole = 0
        model.update()

        return model, inventory_constraints

    def offline_linear_program_fixed_inventory_partial_demand(self, demand_samples: List[Sequence],
                                                             current_inventory, time_step,
                                                             current_demand_node=-1):
        """Build an offline LP using only future (leftover) demand from a given time step.

        Used by re-solving policies to compute dual variables mid-sequence.

        Args:
            demand_samples: List of Sequence objects (each with leftover_aggregate_demand).
            current_inventory: Current inventory levels per supply node.
            time_step: Time step from which to count remaining demand.
            current_demand_node: If >= 0, adds 1 to this node's demand (for the
                current arrival being decided upon).

        Returns:
            Tuple of (model, inventory_constraints).
        """
        model = gp.Model('Offline_fixed_inventory')

        y = {}
        for supply_node_id, demand_node_id in self.graph.edges:
            for sample_index in range(len(demand_samples)):
                y[supply_node_id, demand_node_id, sample_index] = model.addVar(
                    lb=0, obj=self.graph.edges[supply_node_id, demand_node_id].reward,
                    name=f'flow_{supply_node_id}_{demand_node_id}_{sample_index}'
                )

        inventory_constraints = {}

        for demand_node_id in self.graph.demand_nodes:
            demand_node = self.graph.demand_nodes[demand_node_id]
            for sample_index, demand_sample in enumerate(demand_samples):
                if demand_node_id == current_demand_node:
                    model.addConstr(
                        gp.quicksum(y[supply_node_id, demand_node_id, sample_index] for supply_node_id in demand_node.neighbors)
                        <= demand_sample.leftover_aggregate_demand[time_step][demand_node_id] + 1
                    )
                else:
                    model.addConstr(
                        gp.quicksum(y[supply_node_id, demand_node_id, sample_index] for supply_node_id in demand_node.neighbors)
                        <= demand_sample.leftover_aggregate_demand[time_step][demand_node_id]
                    )

        for supply_node_id in self.graph.supply_nodes:
            supply_node = self.graph.supply_nodes[supply_node_id]
            for sample_index in range(len(demand_samples)):
                inventory_constraints[supply_node_id, sample_index] = model.addConstr(
                    gp.quicksum(y[supply_node_id, demand_node_id, sample_index] for demand_node_id in supply_node.neighbors)
                    <= current_inventory[supply_node_id]
                )

        model.ModelSense = -1  # maximize
        model.Params.LogToConsole = 0
        model.update()

        return model, inventory_constraints

    def offline_linear_program_variable_inventory(self, demand_samples: List[Sequence], total_inventory: int):
        """Build an offline LP that jointly optimizes inventory placement over samples.

        Args:
            demand_samples: List of Sequence objects.
            total_inventory: Total inventory budget to allocate.

        Returns:
            Tuple of (model, x) where x maps supply_node_id to its inventory variable.
        """
        model = gp.Model('Offline_variable_inventory')

        y = {}
        x = {}

        for supply_node_id in self.graph.supply_nodes:
            x[supply_node_id] = model.addVar(lb=0, name=f'inventory_{supply_node_id}')

        for supply_node_id, demand_node_id in self.graph.edges:
            for sample_index in range(len(demand_samples)):
                y[supply_node_id, demand_node_id, sample_index] = model.addVar(
                    lb=0, obj=self.graph.edges[supply_node_id, demand_node_id].reward,
                    name=f'flow_{supply_node_id}_{demand_node_id}_{sample_index}'
                )

        model.addConstr(
            gp.quicksum(x[supply_node_id] for supply_node_id in self.graph.supply_nodes) == total_inventory
        )

        inventory_constraints = {}

        for demand_node_id in self.graph.demand_nodes:
            demand_node = self.graph.demand_nodes[demand_node_id]
            for sample_index, demand_sample in enumerate(demand_samples):
                model.addConstr(
                    gp.quicksum(y[supply_node_id, demand_node_id, sample_index] for supply_node_id in demand_node.neighbors)
                    <= demand_sample.aggregate_demand[demand_node_id]
                )

        for supply_node_id in self.graph.supply_nodes:
            supply_node = self.graph.supply_nodes[supply_node_id]
            for sample_index in range(len(demand_samples)):
                inventory_constraints[supply_node_id, sample_index] = model.addConstr(
                    gp.quicksum(y[supply_node_id, demand_node_id, sample_index] for demand_node_id in supply_node.neighbors)
                    <= x[supply_node_id]
                )

        model.ModelSense = -1  # maximize
        model.Params.LogToConsole = 0
        model.update()

        return model, x
