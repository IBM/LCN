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

# Exact marginal inference for LCNs (optimized version)

import itertools
import time
import numpy as np
from pyomo.environ import *
from typing import Tuple

# Local
from lcn.model import LCN, SentenceType, Formula
from lcn.independencies import Independencies
from lcn.inference.utils import make_conjunction, check_consistency

_ACCEPTABLE_TOL = 1e-9
_HESSIAN_APPROX = "limited-memory"

def _eval_indicator(formula: Formula, interpretations: list) -> np.ndarray:
    """Evaluate formula on all interpretations, return binary numpy vector."""
    return np.array([1.0 if formula.evaluate(table=interp) else 0.0
                     for interp in interpretations])


def _dot(vec: np.ndarray, model, items):
    """Build Pyomo linear expression: vec @ model.p"""
    return sum(float(vec[i]) * model.p[i] for i in items)


def solve_exact_model(
        lcn: LCN,
        query_formula: str,
        independencies: Independencies,
        evidence: dict = {},
        sense: str = 'min',
        debug: bool = False,
        verbosity: int = 1,
        max_iter: int = 10000,
        max_cpu_time: int = 7200,
        acceptable_tol: float = None,
        hessian_approximation: str = None
) -> Tuple:
    """
    Compute exact lower/upper bounds on the probability of the query formula
    by solving the corresponding non-linear constraint program (for the input
    LCN and independencies given by the Local Markov Condition).

    Args:
        lcn: LCN
            The input LCN model.
        query_formula: str
            A string representing the query formula.
        independencies: Independencies
            The independencies given by the Local Markov Condition.
        evidence: dict
            A dict containing the observed evidence variables.
        sense: str
            The sense of the optimization problem. It is either `min` or `max`.
        debug: bool
            A flag indicating the debugging mode.
        verbosity: int
            Verbosity level (0 is silent).
        max_iter: int
            Maximum number of iterations used by the ipopt solver (default 10000).
        max_cpu_time: int
            Maximum CPU time in seconds used by the ipopt solver (default 7200 sec).
        acceptable_tol: float
            Acceptable tolerance value used by the ipopt solver (default 0.00001).
        hessian_approximation: str
            The Hessian approximation used by the ipopt solver (default 'limited-memory').

    Returns:
        A tuple representing the objective value and a flag indicating its optimality.
    """

    # Step 1: Precompute interpretation table and indicator vectors
    vars_list = [k for k, _ in lcn.atoms.items()]
    items_tuples = list(itertools.product([0, 1], repeat=len(vars_list)))
    interpretations = [dict(zip(vars_list, t)) for t in items_tuples]
    N = len(interpretations)

    # Create the Pyomo model and variables
    model = ConcreteModel()
    model.ITEMS = Set(initialize=range(N))
    model.p = Var(model.ITEMS, within=NonNegativeReals)
    model.constr = ConstraintList()

    # Probability distribution constraint
    model.constr.add(sum(model.p[i] for i in model.ITEMS) == 1.0)

    # Step 2: Build sentence constraints using precomputed indicators
    for sid, s in lcn.sentences.items():
        lobo = s.get_lower_bound()
        upbo = s.get_upper_bound()
        if s.type == SentenceType.Type1:  # P(q)
            A = _eval_indicator(s.phi_formula, interpretations)
            expr = _dot(A, model, model.ITEMS)
            model.constr.add(expr >= lobo)
            model.constr.add(expr <= upbo)
        else:  # P(q|r)
            Aqr = _eval_indicator(s.phi_and_psi_formula, interpretations)
            Ar = _eval_indicator(s.psi_formula, interpretations)
            expr_qr = _dot(Aqr, model, model.ITEMS)
            expr_r = _dot(Ar, model, model.ITEMS)
            model.constr.add(expr_qr >= lobo * expr_r)
            model.constr.add(expr_qr <= upbo * expr_r)

    # Step 3: Build independence constraints using precomputed indicators
    for indep in independencies.get_assertions():
        X, T, S = list(indep.event1), list(indep.event2), list(indep.event3)
        if verbosity > 0:
            print(f"adding constraints for independence: {indep}")
        configs_S = [()] if len(S) == 0 else list(itertools.product([0, 1], repeat=len(S)))
        if len(S) > 0:
            for t in T:
                x = X[0]
                literals = {x: 1, t: 1}
                for s in configs_S:
                    literals.update(dict(zip(S, list(s))))
                    Fa = make_conjunction(variables=X + S + [t], literals=literals)
                    Fb = make_conjunction(variables=S, literals=literals)
                    Fc = make_conjunction(variables=X + S, literals=literals)
                    Fd = make_conjunction(variables=S + [t], literals=literals)
                    Aa = _eval_indicator(Fa, interpretations)
                    Ab = _eval_indicator(Fb, interpretations)
                    Ac = _eval_indicator(Fc, interpretations)
                    Ad = _eval_indicator(Fd, interpretations)
                    val1 = _dot(Aa, model, model.ITEMS) * _dot(Ab, model, model.ITEMS)
                    val2 = _dot(Ac, model, model.ITEMS) * _dot(Ad, model, model.ITEMS)
                    model.constr.add(val1 - val2 == 0.0)
        else:
            for t in T:
                x = X[0]
                literals = {x: 1, t: 1}
                Fa = make_conjunction(variables=X + [t], literals=literals)
                Fb = make_conjunction(variables=X, literals=literals)
                Fc = make_conjunction(variables=[t], literals=literals)
                Aa = _eval_indicator(Fa, interpretations)
                Ab = _eval_indicator(Fb, interpretations)
                Ac = _eval_indicator(Fc, interpretations)
                val1 = _dot(Aa, model, model.ITEMS)
                val2 = _dot(Ab, model, model.ITEMS) * _dot(Ac, model, model.ITEMS)
                model.constr.add(val1 - val2 == 0.0)

    # Step 4: Build objective (with evidence bug fix)
    obj_formula = Formula(label="obj", formula=query_formula)
    A_query = _eval_indicator(obj_formula, interpretations)

    if len(evidence) == 0:
        obj = _dot(A_query, model, model.ITEMS)
    else:
        ev = [k for k, _ in evidence.items()]
        Fe = make_conjunction(variables=ev, literals=evidence)
        E = _eval_indicator(Fe, interpretations)
        AE = A_query * E  # element-wise numpy multiply (fixes == vs = bug)
        obj = _dot(AE, model, model.ITEMS) / _dot(E, model, model.ITEMS)

    if sense == 'min':
        model.objective = Objective(expr=obj, sense=minimize)
    else:
        model.objective = Objective(expr=obj, sense=maximize)

    # Solve the non-linear model
    try:
        opt = SolverFactory('ipopt')
        opt.options['max_iter'] = max_iter
        opt.options['max_cpu_time'] = max_cpu_time
        if acceptable_tol is not None:
            opt.options['acceptable_tol'] = acceptable_tol
        if hessian_approximation is not None:
            opt.options['hessian_approximation'] = hessian_approximation
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

    except Exception as e:
        if verbosity > 0:
            print(f"Exception during ipopt: {str(e)}")
        objective_value = None
        objective_optimal = False

    if verbosity > 0:
        print(f"[Ipopt] objective={objective_value}, optimal={objective_optimal}")
    return objective_value, objective_optimal


class ExactInference:
    """
    The exact marginal inference algorithm for LCNs.
    See [Marinescu et al. Logical Credal Networks. NeurIPS 2022]
    """

    def __init__(
            self,
            lcn: LCN
    ):
        """
        Constructor of the Exact Inference solver.

        Args:
            lcn: LCN
                The input LCN model.
        """
        self.lcn = lcn
        self.lower_bound = None
        self.upper_bound = None
        self.feasible = None

    def run(
            self,
            query_formula: str,
            evidence: dict = {},
            debug: bool = False,
            verbosity: int = 1
    ):
        """
        Run the exact inference algorithm.

        Args:
            query_formula: str
                A string representing the query formula.
            evidence: dict
                A dictionary containing the observed evidence variables.
            debug: bool
                A flag indicating that ipopt is run in debugging mode.
            verbosity: int
                Verbosity level (0 is silent)
        """

        assert self.lcn is not None, "Make sure the LCN model exists."
        assert self.lcn.independencies is not None, "Make sure the LMC is applied."

        # Start the timer
        t_start = time.time()

        # Get the independencies from the Local Markov Condition
        if verbosity > 0:
            num_indep = len(self.lcn.independencies.get_assertions())
            print(f"[Local Markov Condition: {num_indep} independencies]")
            for indep in self.lcn.independencies.get_assertions():
                print(indep)

        lower_bound, feasible_lb = solve_exact_model(
            lcn=self.lcn,
            query_formula=query_formula,
            evidence=evidence,
            independencies=self.lcn.independencies,
            sense='min',
            debug=debug,
            verbosity=verbosity,
            acceptable_tol=_ACCEPTABLE_TOL,
        )
        upper_bound, feasible_ub = solve_exact_model(
            lcn=self.lcn,
            query_formula=query_formula,
            independencies=self.lcn.independencies,
            evidence=evidence,
            sense='max',
            debug=debug,
            verbosity=verbosity,
            acceptable_tol=_ACCEPTABLE_TOL,
        )

        t_end = time.time()
        self.lower_bound = .0 if not feasible_lb else max(abs(lower_bound), 0.0)
        self.upper_bound = .0 if not feasible_ub else min(abs(upper_bound), 1.0)
        self.feasible = feasible_lb and feasible_ub

        if verbosity > 0:
            print(f"[ExactInference] Result for {query_formula} is: [ {self.lower_bound:.4f}, {self.upper_bound:.4f} ]")
            print(f"[ExactInference] Feasibility: lb={feasible_lb}, ub={feasible_ub}, all={self.feasible}")
            print(f"[ExactInference] Time elapsed: {t_end - t_start} sec")


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

    # Run exact marginal inference
    query = "(!A)"
    evidence = {"B": 0, "E": 0}
    algo = ExactInference(lcn=l)
    algo.run(query_formula=query, evidence=evidence, debug=False)
