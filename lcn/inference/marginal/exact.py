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
import logging
import time
import numpy as np
from tqdm import tqdm
from pyomo.environ import (
    ConcreteModel,
    ConstraintList,
    NonNegativeReals,
    Objective,
    Set,
    TerminationCondition,
    Var,
    maximize,
    minimize,
    value,
)
from typing import Dict, Tuple

# Local
from lcn.core.model import LCN, SentenceType, Formula
from lcn.core.independencies import Independencies
from lcn.inference.utils.common import (
    make_conjunction, check_consistency, make_ipopt,
    lmc_constraint_groups_vec, build_truth_table,
    find_feasible_points, optimize_marginal_slsqp,
    eval_indicator, dot
)

_N_RESTARTS = 4             # random-restart budget on a failed/vacuous solve


# The ipopt configuration is shared across the whole inference suite and lives
# in lcn.inference.utils.common; alias it here for the internal call sites.
_make_ipopt = make_ipopt


def _init_p(model, N: int, rng=None) -> None:
    """
    Initialize the joint-distribution variables to a feasible starting point.

    A fresh start is set before every solve so the result does not depend on the
    previous solve's solution (the shared model would otherwise warm-start each
    objective from the last one, making bounds order-dependent and wrong).

    Args:
        model: Pyomo model with ``model.p`` over ``model.ITEMS``.
        N: int
            Number of joint-distribution variables (interpretations).
        rng: optional numpy Generator
            If None, use the uniform point p[i] = 1/N. Otherwise draw a random
            point on the probability simplex (used for restarts).
    """
    if rng is None:
        for i in model.ITEMS:
            model.p[i].value = 1.0 / N
    else:
        x = rng.random(N)
        x /= x.sum()
        for i in model.ITEMS:
            model.p[i].value = float(x[i])


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
    model.p = Var(model.ITEMS, within=NonNegativeReals, bounds=(0.0, 1.0))
    model.constr = ConstraintList()

    # Constraint residual callables over a numpy solution vector ``p`` --- the
    # same constraints as the Pyomo model, used by the SLSQP robustness fallback
    # (see solve_marginal_slsqp). 'eq' must be ~0; 'ineq' must be >= 0.
    checks = [("eq", lambda p: float(p.sum()) - 1.0)]

    # Probability distribution constraint
    model.constr.add(sum(model.p[i] for i in model.ITEMS) == 1.0)

    # Sentence constraints using precomputed indicators
    for sid, s in lcn.sentences.items():
        lobo = s.get_lower_bound()
        upbo = s.get_upper_bound()
        if s.type == SentenceType.Type1:
            A = eval_indicator(s.phi_formula, interpretations)
            expr = dot(A, model, model.ITEMS)
            model.constr.add(expr >= lobo)
            model.constr.add(expr <= upbo)
            checks.append(("ineq", lambda p, A=A, lobo=lobo: float(A @ p) - lobo))
            checks.append(("ineq", lambda p, A=A, upbo=upbo: upbo - float(A @ p)))
        else:
            Aqr = eval_indicator(s.phi_and_psi_formula, interpretations)
            Ar = eval_indicator(s.psi_formula, interpretations)
            expr_qr = dot(Aqr, model, model.ITEMS)
            expr_r = dot(Ar, model, model.ITEMS)
            model.constr.add(expr_qr >= lobo * expr_r)
            model.constr.add(expr_qr <= upbo * expr_r)
            checks.append(("ineq", lambda p, Aqr=Aqr, Ar=Ar, lobo=lobo: float(Aqr @ p) - lobo * float(Ar @ p)))
            checks.append(("ineq", lambda p, Aqr=Aqr, Ar=Ar, upbo=upbo: upbo * float(Ar @ p) - float(Aqr @ p)))

    # Independence constraints. Each LMC assertion (X |= Y | S) is encoded as
    # the correct *joint* factorization over all configurations of the Y block
    # (see lmc_constraint_groups_vec); a per-element decomposition would
    # under-constrain the model when |Y| >= 2. The vectorized builder uses numpy
    # column masks over a precomputed truth table (bit-identical to the
    # Formula-based path) to avoid the dense ~2^|Y| * 2^n Formula.evaluate cost.
    table = build_truth_table(len(vars_list))
    col_of = {v: i for i, v in enumerate(vars_list)}

    for indep in independencies.get_assertions():
        if verbosity > 1:
            print(f"adding constraints for independence: {indep}")
        for group in lmc_constraint_groups_vec(indep, table, col_of):
            if group[0] == 'conditional':
                _, Aa, Ab, Ac, Ad = group
                val1 = dot(Aa, model, model.ITEMS) * dot(Ab, model, model.ITEMS)
                val2 = dot(Ac, model, model.ITEMS) * dot(Ad, model, model.ITEMS)
                model.constr.add(val1 - val2 == 0.0)
                checks.append(("eq", lambda p, Aa=Aa, Ab=Ab, Ac=Ac, Ad=Ad:
                               float(Aa @ p) * float(Ab @ p) - float(Ac @ p) * float(Ad @ p)))
            else:
                _, Aa, Ab, Ac = group
                val1 = dot(Aa, model, model.ITEMS)
                val2 = dot(Ab, model, model.ITEMS) * dot(Ac, model, model.ITEMS)
                model.constr.add(val1 - val2 == 0.0)
                checks.append(("eq", lambda p, Aa=Aa, Ab=Ab, Ac=Ac:
                               float(Aa @ p) - float(Ab @ p) * float(Ac @ p)))

    # Pre-compute indicator vectors for all atoms
    atom_indicators = {}
    for atom_name in vars_list:
        atom_indicators[atom_name] = eval_indicator(
            Formula(label=atom_name, formula=atom_name), interpretations)

    # Pre-compute evidence indicator (once)
    evidence_indicator = None
    if len(evidence) > 0:
        ev_vars = [k for k in evidence.keys()]
        Fe = make_conjunction(variables=ev_vars, literals=evidence)
        evidence_indicator = eval_indicator(Fe, interpretations)

    return model, atom_indicators, evidence_indicator, interpretations, N, checks


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
        (objective_value, feasible) tuple. A solve is considered feasible/usable
        when ipopt reports an ``optimal``, ``locallyOptimal``, ``feasible`` or
        ``acceptable`` termination --- ipopt's "acceptable" point is a valid
        solution and must not be discarded as if it were infeasible. Only a
        genuine ``infeasible`` termination or an exception yields feasible=False.
    """
    # Remove existing objective if present
    if hasattr(model, 'objective'):
        model.del_component('objective')

    if sense == 'min':
        model.objective = Objective(expr=obj_expr, sense=minimize)
    else:
        model.objective = Objective(expr=obj_expr, sense=maximize)

    # Termination conditions that correspond to a usable solution. ipopt's
    # "Solved to acceptable level" is surfaced as the string 'acceptable' by
    # some Pyomo versions and folded into 'optimal' by others; accept both.
    _usable = {
        TerminationCondition.optimal,
        TerminationCondition.locallyOptimal,
        TerminationCondition.feasible,
    }

    try:
        results = solver.solve(model, load_solutions=True, tee=debug)
        tc = results.solver.termination_condition
        if tc in _usable or str(tc).lower() == 'acceptable':
            objective_value = value(model.objective)
            feasible = True
        elif tc == TerminationCondition.infeasible:
            objective_value = value(model.objective)
            feasible = False
        else:
            # maxIterations / maxTimeLimit / solverFailure / other: the loaded
            # point is not trustworthy --- report None so the caller can restart.
            if debug:
                print(f"ipopt non-usable termination: status={results.solver.status}, "
                      f"termination={tc}")
            objective_value = None
            feasible = False
    except Exception as e:
        if debug:
            print(f"Exception during ipopt: {str(e)}")
        objective_value = None
        feasible = False

    return objective_value, feasible


def _robust_solve(model, obj_expr, sense, solver, N, atom, debug=False,
                  checks=None, obj_vec=None, seeds_provider=None):
    """
    Solve min/max of ``obj_expr`` robustly on the (nonconvex) marginal NLP.

    Each call uses a fresh uniform start (no warm-start carryover between atoms
    or senses). If the primary solve fails, returns None, or returns a vacuous
    bound (max at 1, min at 0 --- typically a sign ipopt stalled at a trivial
    stationary point), the solve is retried from several random simplex starts.

    ipopt (interior-point) is unreliable on the dense, nonconvex joint-LMC
    equality system and can return vacuous/failed bounds even when a valid
    optimum exists. When ``checks`` and ``obj_vec`` are supplied and the ipopt
    result is still suspicious, an SQP feasibility/optimization fallback
    (scipy SLSQP, see solve_marginal_slsqp) computes the bound directly.

    Args:
        model: Pyomo model with constraints and ``model.p``.
        obj_expr: Pyomo objective expression (the atom marginal P(atom=1)).
        sense: 'min' or 'max'.
        solver: configured ipopt solver (see _make_ipopt).
        N: int, number of joint-distribution variables.
        atom: str, the atom name (used only to seed restarts deterministically).
        debug: bool.
        checks: optional list of (kind, residual_fn) for the SLSQP fallback.
        obj_vec: optional numpy objective vector for the SLSQP fallback.
        seeds_provider: optional zero-arg callable returning feasible seed
            distributions; invoked lazily only when the SLSQP fallback is
            actually needed (so easy instances pay nothing).

    Returns:
        (best_value, feasible, used_fallback) tuple. feasible is False only if
        every attempt failed (best_value is then None); used_fallback is True
        iff the SLSQP fallback produced the usable value (for reporting stats).
    """
    def _is_suspicious(v, ok):
        if not ok or v is None:
            return True
        if sense == 'max' and v >= 1.0 - 1e-6:
            return True
        if sense == 'min' and v <= 1e-6:
            return True
        return False

    # Primary solve from the uniform feasible point.
    _init_p(model, N)
    best, ok = _solve_with_objective(model, obj_expr, sense, solver, debug)

    if _is_suspicious(best, ok):
        for k in range(_N_RESTARTS):
            # Deterministic per-(atom, sense, restart) seed -> reproducible runs.
            seed = abs(hash((atom, sense, k))) % (2 ** 32)
            _init_p(model, N, rng=np.random.default_rng(seed))
            v, okk = _solve_with_objective(model, obj_expr, sense, solver, debug)
            if okk and v is not None:
                if best is None or not ok:
                    best, ok = v, True
                elif sense == 'max':
                    best = max(best, v)
                else:
                    best = min(best, v)

    # Two-phase SLSQP fallback when ipopt still looks stuck at a vacuous/failed
    # point. ipopt cannot reliably navigate the dense joint-LMC system, but
    # SLSQP launched from a feasible seed (found once by find_feasible_points and
    # passed in via `seeds`) optimizes the linear marginal objective reliably.
    # Only applies to the linear (no-evidence) objective; the evidence ratio
    # objective passes obj_vec=None and stays ipopt-only.
    used_fallback = False
    if (checks is not None and obj_vec is not None
            and seeds_provider is not None and _is_suspicious(best, ok)):
        seeds = seeds_provider()
        v, okk = optimize_marginal_slsqp(N, obj_vec, checks, sense, seeds) \
            if seeds else (None, False)
        if okk and v is not None:
            used_fallback = True
            if best is None or not ok:
                best, ok = v, True
            elif sense == 'max':
                best = max(best, v)
            else:
                best = min(best, v)

    return best, ok, used_fallback


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
            verbosity: int = 2,
            mode: str = "exact"
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
            mode: str
                ipopt solver mode (see make_ipopt): ``"exact"`` (default) drives
                each bound to high accuracy; ``"fast"`` stops ipopt at the first
                acceptable point, trading accuracy for speed. The SLSQP fallback
                still backstops suspicious/failed solves in both modes.

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
            print("[ExactInference] Computing all marginals")
            print(f"[ExactInference] Evidence: {evidence}")
            print(f"[ExactInference] Local Markov Condition: {num_indep} independencies")

        # Build the base model once (constraints only, no objective)
        model, atom_indicators, evidence_indicator, interpretations, N, checks = \
            _build_base_model(self.lcn, independencies, evidence, verbosity)

        # Pre-compute P(evidence) expression (reused across all atoms)
        ev_expr = None
        if evidence_indicator is not None:
            ev_expr = dot(evidence_indicator, model, model.ITEMS)

        # Create a shared, correctly-configured ipopt solver instance
        solver = _make_ipopt(debug=debug, mode=mode)

        # Compute marginals for each atom
        self.marginals = {}
        self.feasible = True
        atom_names = [k for k, _ in self.lcn.atoms.items()]
        solve_atoms = [a for a in atom_names if a not in evidence_set]

        # Feasible seed distributions for the SLSQP bound-optimization fallback.
        # Computed lazily once (only if ipopt proves unreliable on this LCN) and
        # shared across all atoms / both senses, so easy instances that ipopt
        # already solves pay nothing.
        _seed_cache = {"seeds": None, "done": False}

        def _get_seeds():
            if not _seed_cache["done"]:
                _seed_cache["done"] = True
                try:
                    _seed_cache["seeds"] = find_feasible_points(N, checks)
                except Exception as e:
                    if debug:
                        print(f"seed search failed: {e}")
                    _seed_cache["seeds"] = []
            return _seed_cache["seeds"]

        # Suppress Pyomo's routine "Loading a SolverResults object with a warning
        # status" spam (emitted on every non-optimal ipopt termination, which is
        # normal here -- the restart/SLSQP fallback handles those). Keep it under
        # debug. Restore the prior level afterwards so we don't mute pyomo
        # globally as a side effect.
        pyomo_logger = logging.getLogger('pyomo')
        prev_level = pyomo_logger.level
        if not debug:
            pyomo_logger.setLevel(logging.ERROR)

        # Progress bar over the atoms actually solved (skip evidence). Only shown
        # when verbose; verbosity == 0 stays completely silent (safe for inner
        # loops such as MAP search).
        n_fallback = 0
        n_infeasible = 0
        pbar = tqdm(total=len(solve_atoms), desc="[ExactInference] atoms",
                    disable=(verbosity == 0))
        try:
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

                A = atom_indicators[atom_name]

                # Build objective: P(atom) unconditionally, or
                # P(atom AND evidence) / P(evidence) with evidence. The SLSQP
                # fallback handles a *linear* objective, so it is wired in only
                # for the no-evidence case (obj_vec = A); the evidence ratio
                # objective is nonlinear and stays ipopt-only.
                if ev_expr is None:
                    obj_expr = dot(A, model, model.ITEMS)
                    obj_vec = A
                else:
                    AE = A * evidence_indicator  # element-wise numpy multiply
                    obj_expr = dot(AE, model, model.ITEMS) / ev_expr
                    obj_vec = None

                # Minimize / maximize P(atom=1) robustly: fresh start per solve,
                # with a random-restart fallback when a solve fails or stalls at
                # a vacuous (0/1) stationary point on this nonconvex problem,
                # then an SLSQP fallback when ipopt still cannot find the bound.
                lo_val, feasible_lo, fb_lo = _robust_solve(
                    model, obj_expr, 'min', solver, N, atom_name, debug,
                    checks=checks, obj_vec=obj_vec, seeds_provider=_get_seeds)
                hi_val, feasible_hi, fb_hi = _robust_solve(
                    model, obj_expr, 'max', solver, N, atom_name, debug,
                    checks=checks, obj_vec=obj_vec, seeds_provider=_get_seeds)

                if not feasible_lo or not feasible_hi:
                    self.feasible = False
                    n_infeasible += 1
                if fb_lo or fb_hi:
                    n_fallback += 1

                lo_1 = max(abs(lo_val), 0.0) if feasible_lo else 0.0
                hi_1 = min(abs(hi_val), 1.0) if feasible_hi else 1.0

                lo_arr = np.array([1.0 - hi_1, lo_1])
                hi_arr = np.array([1.0 - lo_1, hi_1])
                self.marginals[atom_name] = (lo_arr, hi_arr)

                # Intermediate stats: current atom's P=1 bound + running counters.
                pbar.set_postfix_str(
                    f"{atom_name}=[{lo_1:.3f},{hi_1:.3f}] "
                    f"fallback={n_fallback} infeasible={n_infeasible}")
                pbar.update(1)
        finally:
            pbar.close()
            pyomo_logger.setLevel(prev_level)

        t_end = time.time()

        if verbosity > 0:
            print("[ExactInference] Singleton variable marginals:")
            for atom_name in sorted(self.marginals):
                lo, hi = self.marginals[atom_name]
                for val in range(len(lo)):
                    print(f"  P({atom_name}={val}): "
                          f"[{lo[val]:.6f}, {hi[val]:.6f}]")
            print(f"[ExactInference] Feasible: {self.feasible}")
            print(f"[ExactInference] Atoms solved: {len(solve_atoms)} | "
                  f"SLSQP fallback used: {n_fallback} | "
                  f"infeasible: {n_infeasible}")
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
    file_name = "examples/alarm.lcn"
    lcn_model = LCN()
    lcn_model.from_lcn(file_name=file_name)
    lcn_model.summary()
    print(lcn_model)

    # Check consistency
    print(f"\n=== Consistency check for {file_name} ===")
    ok = check_consistency(lcn_model)

    # Run exact marginal inference (no evidence)
    print("\n=== ExactInference (no evidence) ===")
    algo = ExactInference(lcn=lcn_model)
    results = algo.run(evidence={}, debug=False, verbosity=2, mode="exact")
    print_singleton_marginals(results)

    # Run exact marginal inference (with evidence)
    # print("\n=== ExactInference (B=0, E=0) ===")
    # algo2 = ExactInference(lcn=l)
    # results = algo2.run(evidence={"B": 0, "E": 0}, debug=False)
    # print_singleton_marginals(results)
