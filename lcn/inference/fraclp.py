# coding=utf-8
# Copyright 2023-present the International Business Machines.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fractional Linear Program with Charnes-Cooper Transformation

import numpy as np
from pyomo.environ import (
    ConcreteModel,
    Set,
    Var,
    Objective,
    Constraint,
    ConstraintList,
    NonNegativeReals,
    minimize,
    maximize,
    SolverFactory,
    SolverStatus,
    TerminationCondition,
    value,
)

# # Problem data (8 decision variables)
# # Numerator: c^T x + d
# c = np.array([2.0, 3.0, 1.0, 4.0, 2.0, 1.0, 3.0, 2.0])
# d = 1.0

# # Denominator: e^T x + f
# e = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
# f = 5.0

# # Constraints: A x <= b, x >= 0
# A = np.array([
#     [ 1,  2,  0,  1,  0,  0,  1,  0],
#     [ 0,  1,  1,  0,  2,  0,  0,  1],
#     [ 1,  0,  0,  1,  0,  1,  0,  1],
#     [ 0,  0,  1,  1,  1,  0,  1,  0],
#     [ 1,  1,  0,  0,  0,  1,  0,  1],
# ])
# b = np.array([10.0, 8.0, 7.0, 6.0, 9.0])


# Problem data (8 decision variables)
# Numerator: c^T x + d
c = np.array([0, 0, 0, 0, 0, 0, 0, 1])
d = 0.0

# Denominator: e^T x + f
e = np.array([0, 0, 0, 1, 0, 0, 0, 1])
f = 0.0

# Constraints: A x <= b, x >= 0
A = np.array([
    [ 0,  0,  1,  1,  0,  0,  1,  1],
    [ 0,  0,  -1,  -1,  0,  0,  -1,  1],
    [ 0,  1,  0,  1,  0,  1,  0,  1],
    [ 0,  -1,  0,  -1,  0,  -1,  0,  -1],
    [ 0,  0,  0,  0,  1,  1,  1,  1],
    [ 0,  0,  0,  0,  -1,  -1,  -1,  -1],
    [ 0,  -0.9,  -0.9,  -0.9,  0,  0.1,  0.1,  0.1],
    [ 0,  0.6,  0.6,  0.6,  0,  -0.4,  -0.4,  -0.4],
])
b = np.array([0.2, -0.1, 0.1, -0.05, 0.1, -0.05, 0, 0])

n = len(c)  # number of decision variables
m = len(b)  # number of constraints


def build_fractional_lp(sense: str):
    """
    Build the original fractional linear program (for reference).

    The FLP has the form:
        minimize   (c^T x + d) / (e^T x + f)
        subject to  A x <= b,  x >= 0

    Note: This model has a fractional objective and is not directly solvable
    by a standard LP solver.

    Returns:
        A Pyomo ConcreteModel representing the original FLP.
    """
    model = ConcreteModel()
    model.VARS = Set(initialize=range(n))
    model.x = Var(model.VARS, within=NonNegativeReals)
    model.constr = ConstraintList()

    # Constraints: A x <= b
    for i in range(m):
        model.constr.add(
            sum(A[i, j] * model.x[j] for j in model.VARS) <= b[i]
        )

    # Simplex constraint: sum(x) == 1
    model.constr.add(sum(model.x[j] for j in model.VARS) == 1)

    # Fractional objective (not solvable by LP solvers)
    numerator = sum(c[j] * model.x[j] for j in model.VARS) + d
    denominator = sum(e[j] * model.x[j] for j in model.VARS) + f

    if sense == "min":
        model.objective = Objective(expr=numerator / denominator, sense=minimize)
    else:
        model.objective = Objective(expr=numerator / denominator, sense=maximize)

    return model


def build_charnes_cooper_lp(sense: str):
    """
    Build the Charnes-Cooper transformed LP equivalent of the fractional LP.

    The transformation introduces t = 1 / (e^T x + f) and y = t * x,
    converting the FLP into an equivalent LP:
        minimize   c^T y + d * t
        subject to  A y - b * t <= 0
                    e^T y + f * t == 1
                    y >= 0,  t >= 0

    Returns:
        A Pyomo ConcreteModel representing the transformed LP.
    """
    model = ConcreteModel()
    model.VARS = Set(initialize=range(n))
    model.y = Var(model.VARS, within=NonNegativeReals)
    model.t = Var(within=NonNegativeReals)
    model.constr = ConstraintList()

    # Transformed constraints: A y - b * t <= 0
    for i in range(m):
        model.constr.add(
            sum(A[i, j] * model.y[j] for j in model.VARS) - b[i] * model.t <= 0
        )

    # Simplex constraint (transformed): sum(y) == t
    model.constr.add(
        sum(model.y[j] for j in model.VARS) == model.t
    )

    # Normalization constraint: e^T y + f * t == 1
    model.constr.add(
        sum(e[j] * model.y[j] for j in model.VARS) + f * model.t == 1
    )

    # Transformed objective: minimize c^T y + d * t
    if sense == "min":
        model.objective = Objective(
            expr=sum(c[j] * model.y[j] for j in model.VARS) + d * model.t,
            sense=minimize,
        )
    else:
        model.objective = Objective(
            expr=sum(c[j] * model.y[j] for j in model.VARS) + d * model.t,
            sense=maximize,
        )

    return model


def solve_fraclp(sense: str, verbosity=1):
    """
    Solve the fractional linear program using the Charnes-Cooper transformation.

    Builds the transformed LP, solves it with CPLEX, and recovers the original
    decision variables and objective value.

    Args:
        verbosity: int
            Verbosity level (0 is silent).

    Returns:
        A dict with keys:
            - x: list of original decision variable values
            - objective_value: the optimal fractional objective value
            - solver_status: the solver status string
            - optimal: bool indicating whether the solution is optimal
    """
    model = build_charnes_cooper_lp(sense)

    try:
        opt = SolverFactory('ipopt')
        results = opt.solve(model, tee=(verbosity > 1))

        if (results.solver.status == SolverStatus.ok) and \
            (results.solver.termination_condition == TerminationCondition.optimal):
            t_val = value(model.t)
            y_vals = [value(model.y[j]) for j in range(n)]

            # Recover original variables: x = y / t
            x_vals = [y_vals[j] / t_val for j in range(n)]

            # Compute original fractional objective
            numerator_val = sum(c[j] * x_vals[j] for j in range(n)) + d
            denominator_val = sum(e[j] * x_vals[j] for j in range(n)) + f
            objective_value = numerator_val / denominator_val

            if verbosity > 0:
                print(f"Solver status: {results.solver.status}")
                print(f"Termination condition: {results.solver.termination_condition}")
                print(f"Transformed objective (c^T y + d*t): {value(model.objective):.6f}")
                print(f"t = {t_val:.6f}")
                for j in range(n):
                    print(f"  y[{j}] = {y_vals[j]:.6f},  x[{j}] = {x_vals[j]:.6f}")
                print(f"Original fractional objective (c^T x + d) / (e^T x + f) = {objective_value:.6f}")

            return {
                "x": x_vals,
                "objective_value": objective_value,
                "solver_status": str(results.solver.status),
                "optimal": True,
            }

        elif results.solver.termination_condition == TerminationCondition.infeasible:
            if verbosity > 0:
                print(f"Solver status: {results.solver.status}")
                print("Problem is infeasible.")
            return {
                "x": None,
                "objective_value": None,
                "solver_status": str(results.solver.status),
                "optimal": False,
            }

        else:
            if verbosity > 0:
                print(f"Solver status: {results.solver.status}")
                print(f"Termination condition: {results.solver.termination_condition}")
            return {
                "x": None,
                "objective_value": None,
                "solver_status": str(results.solver.status),
                "optimal": False,
            }

    except Exception as ex:
        if verbosity > 0:
            print(f"Exception during LP solve: {ex}")
        return {
            "x": None,
            "objective_value": None,
            "solver_status": "error",
            "optimal": False,
        }


if __name__ == "__main__":
    result = solve_fraclp("max", verbosity=1)
    if result["optimal"]:
        print(f"\nOptimal solution found.")
        print(f"Objective value: {result['objective_value']:.6f}")
        print(f"Decision variables: {result['x']}")
    else:
        print(f"\nNo optimal solution. Status: {result['solver_status']}")
