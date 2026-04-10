
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

# Exact and Approximate marginal inference algorithms for LCNs

import itertools
from pyomo.environ import (
    ConcreteModel,
    Set, NonNegativeReals,
    Var, ConstraintList,
    Objective, minimize, maximize,
    SolverFactory,
    SolverStatus,
    value,
    TerminationCondition
)

# Local
from lcn.core.model import LCN, SentenceType, Formula
from lcn.inference.utils.common import make_conjunction, check_consistency

infinity = float('inf')
max_iter = 1000
acceptable_tol = 1e-9
max_cpu_time = 100
debug = True
verbosity = 1


class Factorization:
    def __init__(self, lcn: LCN):
        self.lcn = lcn
        self.factors = []

    def solve_submodel(self, scope, literals, child, parents, sentences, sense):
        items = list(itertools.product([0, 1], repeat=len(scope)))
        index = {k:v for k, v in enumerate(items)}
        N = len(items)

        # Build constraint rows: each row is (coeffs, rhs) for a <= constraint
        constraint_rows = []

        # Probability simplex: sum(p) = 1 encoded as two <= rows
        ones = [1.0] * N
        constraint_rows.append(([-1.0] * N, -1.0))  # -sum(p) <= -1 i.e. sum(p) >= 1
        constraint_rows.append(([1.0] * N, 1.0))     # sum(p) <= 1

        for sid in sentences:
            s = self.lcn.sentences.get(sid)
            if s.type == SentenceType.Type1: # P(phi)
                A = [0.0] * N
                lobo = s.get_lower_bound()
                upbo = s.get_upper_bound()
                for j in range(N):
                    config = dict(zip(scope, index[j]))
                    A[j] = 1.0 if s.phi_formula.evaluate(table=config) == True else 0.0
                # sum(A*p) >= lobo  =>  -sum(A*p) <= -lobo
                constraint_rows.append(([-A[j] for j in range(N)], -lobo))
                # sum(A*p) <= upbo
                constraint_rows.append(([A[j] for j in range(N)], upbo))
            else: # Type 2 sentence P(phi | psi)
                Aqr = [0.0] * N
                Ar = [0.0] * N
                lobo = s.get_lower_bound()
                upbo = s.get_upper_bound()
                for j in range(N):
                    config = dict(zip(scope, index[j]))
                    Aqr[j] = 1.0 if s.phi_and_psi_formula.evaluate(table=config) == True else 0.0
                    Ar[j] = 1.0 if s.psi_formula.evaluate(table=config) == True else 0.0
                # sum((Aqr - lobo*Ar)*p) >= 0  =>  sum((lobo*Ar - Aqr)*p) <= 0
                constraint_rows.append(([lobo*Ar[j] - Aqr[j] for j in range(N)], 0.0))
                # sum((Aqr - upbo*Ar)*p) <= 0
                constraint_rows.append(([Aqr[j] - upbo*Ar[j] for j in range(N)], 0.0))

        # Build objective numerator vector
        Fq = make_conjunction(variables=scope, literals=literals)
        A = [0.0] * N
        for j in range(N):
            config = dict(zip(scope, index[j]))
            A[j] = 1.0 if Fq.evaluate(table=config) == True else 0.0

        if len(parents) == 0:
            # No parents: linear objective, standard LP
            return self._solve_linear_lp(N, A, constraint_rows, sense)
        else:
            # With parents: fractional objective, use Charnes-Cooper
            Fe = make_conjunction(variables=parents, literals=literals)
            E = [0.0] * N
            AE = [0.0] * N
            for j in range(N):
                config = dict(zip(scope, index[j]))
                E[j] = 1.0 if Fe.evaluate(table=config) == True else 0.0
                if A[j] == 1.0 and E[j] == 1.0:
                    AE[j] = 1.0  # bug fix: was == instead of =

            # c = numerator coefficients, d = 0
            # e = denominator coefficients, f = 0
            c = AE
            d = 0.0
            e = E
            f = 0.0
            return self._solve_charnes_cooper_lp(N, c, d, e, f, constraint_rows, sense)

    def _solve_linear_lp(self, N, c, constraint_rows, sense):
        """Solve a standard LP: min/max c^T p subject to constraints."""
        model = ConcreteModel()
        model.ITEMS = Set(initialize=range(N))
        model.p = Var(model.ITEMS, within=NonNegativeReals)
        model.constr = ConstraintList()

        for row_coeffs, row_rhs in constraint_rows:
            model.constr.add(
                sum(row_coeffs[j] * model.p[j] for j in model.ITEMS) <= row_rhs
            )

        obj_expr = sum(c[j] * model.p[j] for j in model.ITEMS)
        if sense == 'min':
            model.objective = Objective(expr=obj_expr, sense=minimize)
        else:
            model.objective = Objective(expr=obj_expr, sense=maximize)

        return self._solve_and_extract(model)

    def _solve_charnes_cooper_lp(self, N, c, d, e, f, constraint_rows, sense):
        """
        Solve a fractional LP via the Charnes-Cooper transformation.

        Original: min/max (c^T p + d) / (e^T p + f)
        Transformed: min/max c^T y + d*t
            subject to: A_row y - b_row * t <= 0  (for each original row A_row p <= b_row)
                        sum(y) == t                (simplex transformed)
                        e^T y + f*t == 1           (normalization)
                        y >= 0, t >= 0
        """
        model = ConcreteModel()
        model.ITEMS = Set(initialize=range(N))
        model.y = Var(model.ITEMS, within=NonNegativeReals)
        model.t = Var(within=NonNegativeReals)
        model.constr = ConstraintList()

        # Transformed constraints: A_row y - b_row * t <= 0
        for row_coeffs, row_rhs in constraint_rows:
            model.constr.add(
                sum(row_coeffs[j] * model.y[j] for j in model.ITEMS) - row_rhs * model.t <= 0
            )

        # Simplex constraint (transformed): sum(y) == t
        # (This is redundant with the equality rows from the simplex, but the
        #  simplex was encoded as two <= rows; the CC transform handles them.
        #  We still add the explicit transformed simplex for correctness.)
        # Actually, the two simplex rows (-sum <= -1 and sum <= 1) become:
        #   -sum(y) - (-1)*t <= 0  =>  -sum(y) + t <= 0
        #   sum(y) - 1*t <= 0      =>  sum(y) - t <= 0
        # Together these give sum(y) == t, so it's already captured.

        # Normalization constraint: e^T y + f*t == 1
        model.constr.add(
            sum(e[j] * model.y[j] for j in model.ITEMS) + f * model.t == 1
        )

        # Transformed objective: min/max c^T y + d*t
        obj_expr = sum(c[j] * model.y[j] for j in model.ITEMS) + d * model.t
        if sense == 'min':
            model.objective = Objective(expr=obj_expr, sense=minimize)
        else:
            model.objective = Objective(expr=obj_expr, sense=maximize)

        result = self._solve_and_extract(model)
        if result is not None:
            # The transformed objective equals the original fractional objective
            return result
        return None

    def _solve_and_extract(self, model):
        """Solve a Pyomo model and return the objective value."""
        try:
            opt = SolverFactory('ipopt')
            opt.options['max_iter'] = max_iter
            opt.options['max_cpu_time'] = max_cpu_time
            if acceptable_tol is not None:
                opt.options['acceptable_tol'] = acceptable_tol
            tee_flag = True if debug else False
            results = opt.solve(model, tee=tee_flag)
            if (results.solver.status == SolverStatus.ok) and \
                (results.solver.termination_condition == TerminationCondition.optimal):
                if verbosity > 0:
                    print(f"Solver status: {results.solver.status}")
                objective_value = value(model.objective)
                objective_optimal = True
            elif (results.solver.termination_condition == TerminationCondition.infeasible):
                if verbosity > 0:
                    print(f"Solver status: {results.solver.status}")
                objective_value = value(model.objective)
                objective_optimal = False
            else:
                if verbosity > 0:
                    print(f"Solver status: {results.solver.status}")
                objective_value = value(model.objective)
                objective_optimal = False

        except Exception as ex:
            if verbosity > 0:
                print(f"Exception during solver: {str(ex)}")
            objective_value = None
            objective_optimal = False

        if verbosity > 0:
            print(f"[Solver] objective={objective_value}, optimal={objective_optimal}")
        return objective_value

    def build(self):
        """
        Process the factorization
        """

        # Ensure that the LCN has been postprocessed (structure, etc.)
        assert self.lcn.structure_graph is not None
        assert self.lcn.simplified_structure_graph is not None
        assert self.lcn.families is not None

        # Process each family
        self.factors = []
        for family in self.lcn.families:
            print(f"Processing family: {family}")
            child = family["child"]
            parents = family["parents"]
            sentences = family["sentences"]
            vars = [child] if "-" not in child else child.split("-")
            for par in parents:
                if "-" in par:
                    vars.append(par.split("-"))
                else:
                    vars.append(par)

            print(f"Processing family: {child} <-- {parents}")
            print(f"Full scope: {vars}")

            # Iterate over all interpretations of the scope
            factor = {}
            interpretations = list(itertools.product([0, 1], repeat=len(vars)))
            for i, interpretation in enumerate(interpretations):
                literals = dict(zip(vars, interpretation))
                lobo = self.solve_submodel(vars, literals, child, parents, sentences, sense="min")
                upbo = self.solve_submodel(vars, literals, child, parents, sentences, sense="max")
                factor[i] = {
                    "interpretation": interpretation,
                    "scope": vars,
                    "child": child,
                    "parents": parents,
                    "lobo": lobo,
                    "upbo": upbo
                }

            self.factors.append(factor)

        return self.factors

if __name__ == "__main__":

    # Load the LCN
    file_name = "examples/alarm.lcn"
    l = LCN()
    l.from_lcn(file_name=file_name)
    print(l)

    # Check consistency
    ok = check_consistency(l)
    if ok:
        print("CONSISTENT")
    else:
        print("INCONSISTENT")

    # Factorize
    l.build_primal_graph(formula_labels=True)
    l.build_structure_graph()

    # check if the LCN is a chain graph
    ok = l.is_chain_graph()
    print(f"Is the LCN a chain graph? {ok}")

    # get the families of each node in the chain graph
    families = l.process_chain_graph()
    print("Families of each node:")
    for family in families:
        node = family["child"]
        print(f"{node}: {family}")

    fact = Factorization(l)
    factors = fact.build()

    for factor in factors:
        child = factor[0]["child"]
        parents = factor[0]["parents"]
        scope = factor[0]["scope"]
        print(f"\nFactor: {child} | parents={parents}, scope={scope}")
        for i, entry in factor.items():
            interp = entry["interpretation"]
            lobo = entry["lobo"]
            upbo = entry["upbo"]
            literals = dict(zip(scope, interp))
            lit_str = ", ".join(f"{k}={v}" for k, v in literals.items())
            lo_str = f"{lobo:.4f}" if lobo is not None else "None"
            up_str = f"{upbo:.4f}" if upbo is not None else "None"
            print(f"  {lit_str}  =>  [{lo_str}, {up_str}]")
