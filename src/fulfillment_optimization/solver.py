"""LP solver dispatch layer supporting Gurobi and HiGHS backends."""

from .lp import LPBuilder


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


def solve(builder: LPBuilder, solver: str):
    """Solve an LP using the specified backend.

    Args:
        builder: LPBuilder with the problem specification.
        solver: Backend name ('gurobi' or 'highs').

    Returns:
        Tuple of (obj_val, var_values, constraint_duals).

    Raises:
        ValueError: If solver name is not recognized.
    """
    if solver not in _SOLVERS:
        raise ValueError(f"Unknown solver '{solver}'. Available: {list(_SOLVERS.keys())}")
    return _SOLVERS[solver](builder)
