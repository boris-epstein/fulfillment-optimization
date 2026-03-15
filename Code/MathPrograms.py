from Demand import Sequence
from Graph import Graph
from typing import Dict, List


class LPResult:
    """Result of a solved LP, providing the same interface as a Gurobi Model.

    Attributes:
        ObjVal: Optimal objective value.
    """

    def __init__(self, obj_val: float):
        self.ObjVal = obj_val

    def optimize(self):
        """No-op for compatibility. The LP is already solved at construction time."""
        pass


class ConstraintResult:
    """Wrapper holding the dual value (shadow price) of a constraint.

    Attributes:
        Pi: Shadow price (dual variable) of this constraint.
    """

    def __init__(self, pi: float):
        self.Pi = pi


class VariableResult:
    """Wrapper holding the solution value of a decision variable.

    Attributes:
        X: Optimal value of this variable.
    """

    def __init__(self, x: float):
        self.X = x


class LPBuilder:
    """Declarative LP builder that collects variables and constraints.

    Variables are referenced by integer indices. Constraints are stored as
    sparse rows. After building, call solve() with a backend to get results.
    """

    def __init__(self):
        self.var_objs = []
        self.var_lbs = []
        self.n_vars = 0
        self.constraints = []  # list of (var_indices, coeffs, sense, rhs)

    def add_var(self, lb=0.0, obj=0.0):
        """Add a variable and return its index."""
        idx = self.n_vars
        self.var_objs.append(obj)
        self.var_lbs.append(lb)
        self.n_vars += 1
        return idx

    def add_le_constraint(self, var_indices, coeffs, rhs):
        """Add a <= constraint. Returns the constraint index."""
        idx = len(self.constraints)
        self.constraints.append((var_indices, coeffs, '<=', rhs))
        return idx

    def add_eq_constraint(self, var_indices, coeffs, rhs):
        """Add an == constraint. Returns the constraint index."""
        idx = len(self.constraints)
        self.constraints.append((var_indices, coeffs, '==', rhs))
        return idx


def _solve_gurobi(builder: LPBuilder):
    """Solve an LP using Gurobi.

    Args:
        builder: LPBuilder with the problem specification.

    Returns:
        Tuple of (obj_val, var_values, constraint_duals).
    """
    import gurobipy as gp

    model = gp.Model()
    model.Params.LogToConsole = 0
    model.Params.OutputFlag = 0

    gvars = []
    for i in range(builder.n_vars):
        gvars.append(model.addVar(lb=builder.var_lbs[i], obj=builder.var_objs[i]))

    model.ModelSense = -1  # maximize
    model.update()

    gcons = []
    for var_indices, coeffs, sense, rhs in builder.constraints:
        expr = gp.quicksum(coeffs[k] * gvars[var_indices[k]] for k in range(len(var_indices)))
        if sense == '<=':
            gcons.append(model.addConstr(expr <= rhs))
        else:
            gcons.append(model.addConstr(expr == rhs))

    model.optimize()

    obj_val = model.ObjVal
    var_values = [v.X for v in gvars]
    duals = [c.Pi for c in gcons]

    return obj_val, var_values, duals


def _solve_highs(builder: LPBuilder):
    """Solve an LP using HiGHS.

    Converts the maximization problem to minimization (HiGHS convention)
    and negates duals back to match the maximization interpretation.

    Args:
        builder: LPBuilder with the problem specification.

    Returns:
        Tuple of (obj_val, var_values, constraint_duals).
    """
    import highspy

    h = highspy.Highs()
    h.silent()

    inf = highspy.kHighsInf

    # Add variables with costs (HiGHS minimizes, so negate objective coefficients)
    for i in range(builder.n_vars):
        h.addVar(builder.var_lbs[i], inf)
        h.changeColCost(i, -builder.var_objs[i])

    # Add constraints
    for var_indices, coeffs, sense, rhs in builder.constraints:
        if sense == '<=':
            h.addRow(-inf, rhs, len(var_indices), var_indices, coeffs)
        else:
            h.addRow(rhs, rhs, len(var_indices), var_indices, coeffs)

    h.run()

    solution = h.getSolution()
    _, obj_val_min = h.getInfoValue("objective_function_value")
    obj_val = -obj_val_min  # negate back to maximization

    var_values = list(solution.col_value)

    # Dual sign convention: HiGHS solves min, so for <= constraints the duals
    # are non-positive. For our max problem, shadow prices = -1 * HiGHS duals.
    duals = [-d for d in solution.row_dual]

    return obj_val, var_values, duals


# Solver registry
_SOLVERS = {
    'gurobi': _solve_gurobi,
    'highs': _solve_highs,
}


class MathPrograms:
    """Collection of LP formulations for fulfillment optimization.

    Provides fluid (expected demand) and offline (sample-based) linear programs
    with either fixed or variable inventory placement. Supports multiple LP
    solver backends.

    Args:
        graph: The bipartite supply-demand graph.
        solver: LP solver backend to use ('gurobi' or 'highs').
    """

    def __init__(self, graph: Graph, solver: str = 'gurobi'):
        self.graph = graph
        if solver not in _SOLVERS:
            raise ValueError(f"Unknown solver '{solver}'. Available: {list(_SOLVERS.keys())}")
        self.solver = solver
        self._solve = _SOLVERS[solver]

    def _build_and_solve(self, builder: LPBuilder):
        """Solve an LP and return (obj_val, var_values, constraint_duals)."""
        return self._solve(builder)

    def fluid_linear_program_fixed_inventory(self, average_demand: Dict[int, float], inventory,
                                             scaling_factor: float = 1.0,
                                             demand_node_id_to_add_1: int = None):
        """Build and solve a fluid LP that maximizes reward given fixed inventory.

        Args:
            average_demand: Expected demand per demand node.
            inventory: Inventory object with fixed initial_inventory per supply node.
            scaling_factor: Multiplier applied to demand (for rescaling).
            demand_node_id_to_add_1: If set, adds 1 to this demand node's capacity
                (used for marginal value computation).

        Returns:
            Tuple of (LPResult, inventory_constraints) where inventory_constraints
            maps supply_node_id to a ConstraintResult with shadow price.
        """
        builder = LPBuilder()

        # Flow variables y[supply, demand]
        y_idx = {}
        for supply_node_id, demand_node_id in self.graph.edges:
            y_idx[supply_node_id, demand_node_id] = builder.add_var(
                lb=0, obj=self.graph.edges[supply_node_id, demand_node_id].reward
            )

        # Demand constraints
        for demand_node_id in self.graph.demand_nodes:
            demand_node = self.graph.demand_nodes[demand_node_id]
            var_indices = [y_idx[s, demand_node_id] for s in demand_node.neighbors]
            coeffs = [1.0] * len(var_indices)
            rhs = average_demand[demand_node_id] * scaling_factor
            if demand_node.id == demand_node_id_to_add_1:
                rhs += 1
            builder.add_le_constraint(var_indices, coeffs, rhs)

        # Inventory constraints (these are the ones we need duals for)
        inv_constr_indices = {}
        for supply_node_id in self.graph.supply_nodes:
            supply_node = self.graph.supply_nodes[supply_node_id]
            var_indices = [y_idx[supply_node_id, d] for d in supply_node.neighbors]
            coeffs = [1.0] * len(var_indices)
            inv_constr_indices[supply_node_id] = builder.add_le_constraint(
                var_indices, coeffs, inventory.initial_inventory[supply_node_id]
            )

        obj_val, var_values, duals = self._build_and_solve(builder)

        result = LPResult(obj_val)
        inventory_constraints = {
            s: ConstraintResult(duals[inv_constr_indices[s]])
            for s in self.graph.supply_nodes
        }

        return result, inventory_constraints

    def fluid_linear_program_variable_inventory(self, average_demand: Dict[int, float],
                                                total_inventory: int,
                                                scaling_factor: float = 1.0):
        """Build and solve a fluid LP that jointly optimizes inventory placement.

        Args:
            average_demand: Expected demand per demand node.
            total_inventory: Total units of inventory to distribute across supply nodes.
            scaling_factor: Multiplier applied to demand.

        Returns:
            Tuple of (LPResult, x) where x maps supply_node_id to a
            VariableResult with the optimal inventory level.
        """
        builder = LPBuilder()

        # Inventory variables x[supply]
        x_idx = {}
        for supply_node_id in self.graph.supply_nodes:
            x_idx[supply_node_id] = builder.add_var(lb=0, obj=0.0)

        # Flow variables y[supply, demand]
        y_idx = {}
        for supply_node_id, demand_node_id in self.graph.edges:
            y_idx[supply_node_id, demand_node_id] = builder.add_var(
                lb=0, obj=self.graph.edges[supply_node_id, demand_node_id].reward
            )

        # Total inventory constraint
        builder.add_eq_constraint(
            list(x_idx.values()), [1.0] * len(x_idx), total_inventory
        )

        # Demand constraints
        for demand_node_id in self.graph.demand_nodes:
            demand_node = self.graph.demand_nodes[demand_node_id]
            var_indices = [y_idx[s, demand_node_id] for s in demand_node.neighbors]
            coeffs = [1.0] * len(var_indices)
            builder.add_le_constraint(var_indices, coeffs, average_demand[demand_node_id] * scaling_factor)

        # Inventory capacity constraints: sum of flows from supply <= x[supply]
        for supply_node_id in self.graph.supply_nodes:
            supply_node = self.graph.supply_nodes[supply_node_id]
            var_indices = [y_idx[supply_node_id, d] for d in supply_node.neighbors] + [x_idx[supply_node_id]]
            coeffs = [1.0] * len(supply_node.neighbors) + [-1.0]
            builder.add_le_constraint(var_indices, coeffs, 0.0)

        obj_val, var_values, duals = self._build_and_solve(builder)

        result = LPResult(obj_val)
        x = {s: VariableResult(var_values[x_idx[s]]) for s in self.graph.supply_nodes}

        return result, x

    def offline_linear_program_fixed_inventory(self, demand_samples: List[Sequence], inventory):
        """Build and solve an offline LP over multiple demand samples with fixed inventory.

        Args:
            demand_samples: List of Sequence objects (each with aggregate_demand).
            inventory: Inventory object with fixed initial_inventory.

        Returns:
            Tuple of (LPResult, inventory_constraints) where inventory_constraints
            maps (supply_node_id, sample_index) to a ConstraintResult.
        """
        builder = LPBuilder()
        n_samples = len(demand_samples)

        # Flow variables y[supply, demand, sample]
        y_idx = {}
        for supply_node_id, demand_node_id in self.graph.edges:
            for sample_index in range(n_samples):
                y_idx[supply_node_id, demand_node_id, sample_index] = builder.add_var(
                    lb=0, obj=self.graph.edges[supply_node_id, demand_node_id].reward
                )

        # Demand constraints
        for demand_node_id in self.graph.demand_nodes:
            demand_node = self.graph.demand_nodes[demand_node_id]
            for sample_index, demand_sample in enumerate(demand_samples):
                var_indices = [y_idx[s, demand_node_id, sample_index] for s in demand_node.neighbors]
                coeffs = [1.0] * len(var_indices)
                builder.add_le_constraint(var_indices, coeffs, demand_sample.aggregate_demand[demand_node_id])

        # Inventory constraints
        inv_constr_indices = {}
        for supply_node_id in self.graph.supply_nodes:
            supply_node = self.graph.supply_nodes[supply_node_id]
            for sample_index in range(n_samples):
                var_indices = [y_idx[supply_node_id, d, sample_index] for d in supply_node.neighbors]
                coeffs = [1.0] * len(var_indices)
                inv_constr_indices[supply_node_id, sample_index] = builder.add_le_constraint(
                    var_indices, coeffs, inventory.initial_inventory[supply_node_id]
                )

        obj_val, var_values, duals = self._build_and_solve(builder)

        result = LPResult(obj_val)
        inventory_constraints = {
            (s, k): ConstraintResult(duals[inv_constr_indices[s, k]])
            for s in self.graph.supply_nodes
            for k in range(n_samples)
        }

        return result, inventory_constraints

    def offline_linear_program_fixed_inventory_partial_demand(self, demand_samples: List[Sequence],
                                                             current_inventory, time_step,
                                                             current_demand_node=-1):
        """Build and solve an offline LP using only future (leftover) demand.

        Used by re-solving policies to compute dual variables mid-sequence.

        Args:
            demand_samples: List of Sequence objects (each with leftover_aggregate_demand).
            current_inventory: Current inventory levels per supply node.
            time_step: Time step from which to count remaining demand.
            current_demand_node: If >= 0, adds 1 to this node's demand.

        Returns:
            Tuple of (LPResult, inventory_constraints).
        """
        builder = LPBuilder()
        n_samples = len(demand_samples)

        # Flow variables y[supply, demand, sample]
        y_idx = {}
        for supply_node_id, demand_node_id in self.graph.edges:
            for sample_index in range(n_samples):
                y_idx[supply_node_id, demand_node_id, sample_index] = builder.add_var(
                    lb=0, obj=self.graph.edges[supply_node_id, demand_node_id].reward
                )

        # Demand constraints
        for demand_node_id in self.graph.demand_nodes:
            demand_node = self.graph.demand_nodes[demand_node_id]
            for sample_index, demand_sample in enumerate(demand_samples):
                var_indices = [y_idx[s, demand_node_id, sample_index] for s in demand_node.neighbors]
                coeffs = [1.0] * len(var_indices)
                rhs = demand_sample.leftover_aggregate_demand[time_step][demand_node_id]
                if demand_node_id == current_demand_node:
                    rhs += 1
                builder.add_le_constraint(var_indices, coeffs, rhs)

        # Inventory constraints
        inv_constr_indices = {}
        for supply_node_id in self.graph.supply_nodes:
            supply_node = self.graph.supply_nodes[supply_node_id]
            for sample_index in range(n_samples):
                var_indices = [y_idx[supply_node_id, d, sample_index] for d in supply_node.neighbors]
                coeffs = [1.0] * len(var_indices)
                inv_constr_indices[supply_node_id, sample_index] = builder.add_le_constraint(
                    var_indices, coeffs, current_inventory[supply_node_id]
                )

        obj_val, var_values, duals = self._build_and_solve(builder)

        result = LPResult(obj_val)
        inventory_constraints = {
            (s, k): ConstraintResult(duals[inv_constr_indices[s, k]])
            for s in self.graph.supply_nodes
            for k in range(n_samples)
        }

        return result, inventory_constraints

    def offline_linear_program_variable_inventory(self, demand_samples: List[Sequence], total_inventory: int):
        """Build and solve an offline LP that jointly optimizes inventory placement.

        Args:
            demand_samples: List of Sequence objects.
            total_inventory: Total inventory budget to allocate.

        Returns:
            Tuple of (LPResult, x) where x maps supply_node_id to a VariableResult.
        """
        builder = LPBuilder()
        n_samples = len(demand_samples)

        # Inventory variables x[supply]
        x_idx = {}
        for supply_node_id in self.graph.supply_nodes:
            x_idx[supply_node_id] = builder.add_var(lb=0, obj=0.0)

        # Flow variables y[supply, demand, sample]
        y_idx = {}
        for supply_node_id, demand_node_id in self.graph.edges:
            for sample_index in range(n_samples):
                y_idx[supply_node_id, demand_node_id, sample_index] = builder.add_var(
                    lb=0, obj=self.graph.edges[supply_node_id, demand_node_id].reward
                )

        # Total inventory constraint
        builder.add_eq_constraint(
            list(x_idx.values()), [1.0] * len(x_idx), total_inventory
        )

        # Demand constraints
        for demand_node_id in self.graph.demand_nodes:
            demand_node = self.graph.demand_nodes[demand_node_id]
            for sample_index, demand_sample in enumerate(demand_samples):
                var_indices = [y_idx[s, demand_node_id, sample_index] for s in demand_node.neighbors]
                coeffs = [1.0] * len(var_indices)
                builder.add_le_constraint(var_indices, coeffs, demand_sample.aggregate_demand[demand_node_id])

        # Inventory capacity constraints
        for supply_node_id in self.graph.supply_nodes:
            supply_node = self.graph.supply_nodes[supply_node_id]
            for sample_index in range(n_samples):
                var_indices = [y_idx[supply_node_id, d, sample_index] for d in supply_node.neighbors] + [x_idx[supply_node_id]]
                coeffs = [1.0] * len(supply_node.neighbors) + [-1.0]
                builder.add_le_constraint(var_indices, coeffs, 0.0)

        obj_val, var_values, duals = self._build_and_solve(builder)

        result = LPResult(obj_val)
        x = {s: VariableResult(var_values[x_idx[s]]) for s in self.graph.supply_nodes}

        return result, x
