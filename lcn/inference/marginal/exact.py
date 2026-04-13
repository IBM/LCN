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
from typing import Dict, Tuple

# Local
from lcn.core.model import LCN, SentenceType, Formula
from lcn.core.independencies import Independencies
from lcn.inference.utils.common import make_conjunction, check_consistency

_ACCEPTABLE_TOL = 1e-9
_MAX_ITER = 1000
_MAX_CPU_TIME = 7200
_HESSIAN_APPROX = "limited-memory"


def _eval_indicator(formula: Formula, interpretations: list) -> np.ndarray:
    """Evaluate formula on all interpretations, return binary numpy vector."""
    return np.array([1.0 if formula.evaluate(table=interp) else 0.0
                     for interp in interpretations])


def _dot(vec: np.ndarray, model, items):
    """Build Pyomo linear expression: vec @ model.p"""
    return sum(float(vec[i]) * model.p[i] for i in items)


def _build_base_model(
        lcn: LCN,
        independencies: Independencies,
        evidence: dict = {},
        verbosity: int = 0
) -> Tuple:
    """
    Build the base Pyomo model with all constraints (probability distribution,
    sentence bounds, independence) but no objective. Pre-compute indicator
    vectors for all atoms and the evidence indicator.

    Args:
        lcn: The LCN model.
        independencies: Independence assertions from the LMC.
        evidence: {variable_name: value} for observed variables.
        verbosity: 0=silent, 1+=print details.

    Returns:
        (model, atom_indicators, evidence_indicator, interpretations, N)
        - model: Pyomo ConcreteModel with constraints, no objective
        - atom_indicators: dict atom_name -> indicator numpy array
        - evidence_indicator: numpy array (or None if no evidence)
        - interpretations: list of assignment dicts
        - N: number of interpretations
    """
    # Precompute interpretation table
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

    # Sentence constraints using precomputed indicators
    for sid, s in lcn.sentences.items():
        lobo = s.get_lower_bound()
        upbo = s.get_upper_bound()
        if s.type == SentenceType.Type1:
            A = _eval_indicator(s.phi_formula, interpretations)
            expr = _dot(A, model, model.ITEMS)
            model.constr.add(expr >= lobo)
            model.constr.add(expr <= upbo)
        else:
            Aqr = _eval_indicator(s.phi_and_psi_formula, interpretations)
            Ar = _eval_indicator(s.psi_formula, interpretations)
            expr_qr = _dot(Aqr, model, model.ITEMS)
            expr_r = _dot(Ar, model, model.ITEMS)
            model.constr.add(expr_qr >= lobo * expr_r)
            model.constr.add(expr_qr <= upbo * expr_r)

    # Independence constraints
    for indep in independencies.get_assertions():
        X, T, S = list(indep.event1), list(indep.event2), list(indep.event3)
        if verbosity > 1:
            print(f"adding constraints for independence: {indep}")
        configs_S = [()] if len(S) == 0 else list(
            itertools.product([0, 1], repeat=len(S)))
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

    # Pre-compute indicator vectors for all atoms
    atom_indicators = {}
    for atom_name in vars_list:
        atom_indicators[atom_name] = _eval_indicator(
            Formula(label=atom_name, formula=atom_name), interpretations)

    # Pre-compute evidence indicator (once)
    evidence_indicator = None
    if len(evidence) > 0:
        ev_vars = [k for k in evidence.keys()]
        Fe = make_conjunction(variables=ev_vars, literals=evidence)
        evidence_indicator = _eval_indicator(Fe, interpretations)

    return model, atom_indicators, evidence_indicator, interpretations, N


def _solve_with_objective(model, obj_expr, sense, solver, debug=False):
    """
    Attach an objective to the model, solve, and return the result.
    Removes any existing objective before adding the new one.

    Args:
        model: Pyomo ConcreteModel with constraints.
        obj_expr: Pyomo expression for the objective.
        sense: 'min' or 'max'.
        solver: Reusable SolverFactory instance.
        debug: If True, show solver output.

    Returns:
        (objective_value, feasible) tuple.
    """
    # Remove existing objective if present
    if hasattr(model, 'objective'):
        model.del_component('objective')

    if sense == 'min':
        model.objective = Objective(expr=obj_expr, sense=minimize)
    else:
        model.objective = Objective(expr=obj_expr, sense=maximize)

    try:
        results = solver.solve(model, load_solutions=True, tee=debug)
        if (results.solver.status == SolverStatus.ok) and \
            (results.solver.termination_condition == TerminationCondition.optimal):
            objective_value = value(model.objective)
            feasible = True
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            objective_value = value(model.objective)
            feasible = False
        else:
            objective_value = value(model.objective)
            feasible = False
    except Exception as e:
        if debug:
            print(f"Exception during ipopt: {str(e)}")
        objective_value = None
        feasible = False

    return objective_value, feasible


# -----------------------------------------------------------------------
# Legacy function kept for backward compatibility
# -----------------------------------------------------------------------

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


# -----------------------------------------------------------------------
# ExactInference class
# -----------------------------------------------------------------------

class ExactInference:
    """
    The exact marginal inference algorithm for LCNs.
    See [Marinescu et al. Logical Credal Networks. NeurIPS 2022]
    """

    def __init__(
            self,
            lcn: LCN
    ):
        self.lcn = lcn
        self.marginals = None
        self.feasible = None

    def run(
            self,
            evidence: dict = {},
            debug: bool = False,
            verbosity: int = 2
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Run exact inference to compute marginals for ALL singleton variables.

        Builds the constraint model once, then solves min/max for each atom
        by swapping the objective. Evidence is handled via conditional
        probability in the objective: P(atom AND e) / P(e).

        Args:
            evidence: dict
                A dictionary containing the observed evidence variables.
            debug: bool
                A flag indicating that ipopt is run in debugging mode.
            verbosity: int
                Verbosity level (0 is silent).

        Returns:
            Dict mapping variable name to (lower_bounds, upper_bounds)
            numpy arrays.
        """
        assert self.lcn is not None, "Make sure the LCN model exists."
        assert self.lcn.independencies is not None, "Make sure the LMC is applied."

        t_start = time.time()
        independencies = self.lcn.independencies
        evidence_set = set(evidence.keys())

        if verbosity > 0:
            num_indep = len(independencies.get_assertions())
            print(f"[ExactInference] Computing all marginals")
            print(f"[ExactInference] Evidence: {evidence}")
            print(f"[ExactInference] Local Markov Condition: {num_indep} independencies")

        # Build the base model once (constraints only, no objective)
        model, atom_indicators, evidence_indicator, interpretations, N = \
            _build_base_model(self.lcn, independencies, evidence, verbosity)

        # Pre-compute P(evidence) expression (reused across all atoms)
        ev_expr = None
        if evidence_indicator is not None:
            ev_expr = _dot(evidence_indicator, model, model.ITEMS)

        # Create a shared solver instance
        solver = SolverFactory('ipopt')
        solver.options['max_iter'] = _MAX_ITER
        solver.options['max_cpu_time'] = _MAX_CPU_TIME
        # solver.options['hessian_approximation'] = _HESSIAN_APPROX
        solver.options['acceptable_tol'] = _ACCEPTABLE_TOL
        if not debug:
            solver.options['print_level'] = 0

        # Compute marginals for each atom
        self.marginals = {}
        self.feasible = True
        atom_names = [k for k, _ in self.lcn.atoms.items()]

        for atom_name in atom_names:
            if atom_name in evidence_set:
                # Evidence variable: point distribution
                ev_val = evidence[atom_name]
                lo_arr = np.zeros(2)
                hi_arr = np.zeros(2)
                lo_arr[ev_val] = 1.0
                hi_arr[ev_val] = 1.0
                self.marginals[atom_name] = (lo_arr, hi_arr)
                continue

            if verbosity > 1:
                print(f"[ExactInference] Optimizing: {atom_name}")

            A = atom_indicators[atom_name]

            # Build objective: P(atom) unconditionally, or
            # P(atom AND evidence) / P(evidence) with evidence
            if ev_expr is None:
                obj_expr = _dot(A, model, model.ITEMS)
            else:
                AE = A * evidence_indicator  # element-wise numpy multiply
                obj_expr = _dot(AE, model, model.ITEMS) / ev_expr

            # Minimize P(atom=1)
            lo_val, feasible_lo = _solve_with_objective(
                model, obj_expr, 'min', solver, debug)
            # Maximize P(atom=1)
            hi_val, feasible_hi = _solve_with_objective(
                model, obj_expr, 'max', solver, debug)

            if not feasible_lo or not feasible_hi:
                self.feasible = False

            lo_1 = max(abs(lo_val), 0.0) if feasible_lo else 0.0
            hi_1 = min(abs(hi_val), 1.0) if feasible_hi else 1.0

            lo_arr = np.array([1.0 - hi_1, lo_1])
            hi_arr = np.array([1.0 - lo_1, hi_1])
            self.marginals[atom_name] = (lo_arr, hi_arr)

        t_end = time.time()

        if verbosity > 0:
            print(f"[ExactInference] Singleton variable marginals:")
            for atom_name in sorted(self.marginals):
                lo, hi = self.marginals[atom_name]
                for val in range(len(lo)):
                    print(f"  P({atom_name}={val}): "
                          f"[{lo[val]:.6f}, {hi[val]:.6f}]")
            print(f"[ExactInference] Feasible: {self.feasible}")
            print(f"[ExactInference] Time elapsed: {t_end - t_start:.4f} sec")

        return self.marginals


if __name__ == "__main__":

    def print_singleton_marginals(results):
        """Print only singleton variable marginals from the results."""
        print("  Singleton variable marginals:")
        for var in sorted(results):
            lo, hi = results[var]
            for val in range(len(lo)):
                print(f"    P({var}={val}): [{lo[val]:.6f}, {hi[val]:.6f}]")

    # Load the LCN
    file_name = "benchmarks/polytree/polytree_n10_1.lcn" #"examples/alarm.lcn"
    l = LCN()
    l.from_lcn(file_name=file_name)
    print(l)

    # Check consistency
    # ok = check_consistency(l)
    # if ok:
    #     print("CONSISTENT")
    # else:
    #     print("INCONSISTENT")

    # Run exact marginal inference (no evidence)
    print("\n=== ExactInference (no evidence) ===")
    algo = ExactInference(lcn=l)
    results = algo.run(evidence={}, debug=False, verbosity=2)
    print_singleton_marginals(results)

    # Run exact marginal inference (with evidence)
    # print("\n=== ExactInference (B=0, E=0) ===")
    # algo2 = ExactInference(lcn=l)
    # results = algo2.run(evidence={"B": 0, "E": 0}, debug=False)
    # print_singleton_marginals(results)
