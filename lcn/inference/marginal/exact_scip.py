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

# Exact marginal inference for LCNs backed by the SCIP GLOBAL solver.
#
# This mirrors ExactInference (exact.py) but swaps the underlying solver: instead
# of ipopt + an SLSQP fallback (both *local* methods on the nonconvex bilinear-LMC
# NLP), it uses SCIP, a spatial branch-and-bound *global* solver. SCIP either
# proves a global optimum (optimality gap -> 0, "confirmed optimal") or, when it
# hits the time limit, returns the best feasible bound found so far together with
# the remaining gap ("unconfirmed").
#
# Both objective shapes are handled:
#   * no evidence  -> linear objective   P(atom=1)            = A @ p
#   * with evidence -> fractional objective P(atom=1, e)/P(e) = (A*E)@p / (E@p)
# SCIP optimizes the ratio directly through Pyomo's AMPL/NL interface; no
# Charnes-Cooper reformulation is needed. A small denominator floor P(e) >= eps
# keeps the ratio well-defined (a degenerate near-zero denominator would make the
# fractional objective ill-posed).
#
# Requires the SCIP CLI binary on PATH (`brew install scip`); Pyomo drives it via
# SolverFactory('scip'), exactly like ipopt. SCIP is an OPTIONAL dependency used
# only by this engine and the verify_scip.py checker, not by the core exact.py.

import itertools
import logging
import math
import time

import numpy as np
from tqdm import tqdm
from pyomo.environ import (
    ConcreteModel,
    ConstraintList,
    NonNegativeReals,
    Objective,
    Set,
    SolverFactory,
    TerminationCondition,
    Var,
    maximize,
    minimize,
    value,
)
from typing import Dict, Tuple

# Local
from lcn.core.model import LCN, SentenceType, Formula
from lcn.inference.utils.common import (
    eval_indicator,
    dot,
    build_truth_table,
    lmc_constraint_groups_vec,
    make_conjunction,
    check_consistency,
)


# Termination conditions that mean SCIP proved optimality. A time/iteration limit
# that still produced an incumbent yields a valid-but-unproven bound; that case is
# detected from the optimality gap, not enumerated here.
_OPTIMAL = {
    TerminationCondition.optimal,
    TerminationCondition.locallyOptimal,
    TerminationCondition.globallyOptimal,
    TerminationCondition.feasible,
}


def make_scip(time_limit: float = 3600.0, gap_tol: float = 0.0):
    """
    Return a configured SCIP solver, or raise a clear error if SCIP is missing.

    Args:
        time_limit: per-solve wall-clock limit in seconds (SCIP ``limits/time``);
            default 3600 (one hour).
        gap_tol: relative optimality gap to stop at (SCIP ``limits/gap``); 0 means
            prove global optimality.
    """
    solver = SolverFactory('scip')
    if not solver.available(exception_flag=False):
        raise RuntimeError(
            "SCIP solver not found. Install the SCIP CLI binary and ensure it is "
            "on PATH (macOS: `brew install scip`). Pyomo drives it via the "
            "AMPL/NL interface, like ipopt.")
    solver.options['limits/time'] = float(time_limit)
    if gap_tol is not None:
        solver.options['limits/gap'] = float(gap_tol)
    return solver


def _read_gap(results):
    """Best-effort optimality gap from a Pyomo results object (or None)."""
    gap = getattr(results.solver, 'gap', None)
    try:
        if gap is not None and not (isinstance(gap, float) and math.isnan(gap)):
            return float(gap)
    except (TypeError, ValueError):
        pass
    return None


class ExactInferenceSCIP:
    """
    Exact marginal inference for LCNs using SCIP as the global solver.

    Interface-compatible with ``ExactInference`` (exact.py): construct with an
    ``LCN``, call ``run()``, get back ``Dict[name -> (lower, upper)]``. In
    addition, ``self.status`` records the per-atom SCIP verdict (best value,
    optimality gap, and ``"confirmed"``/``"unconfirmed"``/``"unsolved"``).
    """

    def __init__(self, lcn: LCN):
        self.lcn = lcn
        self.marginals = None
        self.feasible = None
        # {atom: {'min': (value, gap, status, secs), 'max': (...)}}
        self.status = None

    # ------------------------------------------------------------------
    # Model construction (mirrors exact._build_base_model; reuses helpers)
    # ------------------------------------------------------------------
    def _build_base_model(self, interpretations):
        """
        Build a fresh Pyomo model with the simplex, sentence-bound and bilinear
        LMC constraints (no objective). A fresh model is built per solve so the
        objective swap never inherits a stale incumbent; cheap at small n.
        """
        vars_list = [k for k, _ in self.lcn.atoms.items()]
        N = len(interpretations)

        model = ConcreteModel()
        model.ITEMS = Set(initialize=range(N))
        model.p = Var(model.ITEMS, within=NonNegativeReals, bounds=(0.0, 1.0))
        model.constr = ConstraintList()

        # Probability distribution.
        model.constr.add(sum(model.p[i] for i in model.ITEMS) == 1.0)

        # Sentence-bound constraints.
        for _, s in self.lcn.sentences.items():
            lobo = s.get_lower_bound()
            upbo = s.get_upper_bound()
            if s.type == SentenceType.Type1:
                A = eval_indicator(s.phi_formula, interpretations)
                expr = dot(A, model, model.ITEMS)
                model.constr.add(expr >= lobo)
                model.constr.add(expr <= upbo)
            else:
                Aqr = eval_indicator(s.phi_and_psi_formula, interpretations)
                Ar = eval_indicator(s.psi_formula, interpretations)
                expr_qr = dot(Aqr, model, model.ITEMS)
                expr_r = dot(Ar, model, model.ITEMS)
                model.constr.add(expr_qr >= lobo * expr_r)
                model.constr.add(expr_qr <= upbo * expr_r)

        # LMC independence constraints (bilinear equalities) -- the nonconvex part.
        table = build_truth_table(len(vars_list))
        col_of = {v: i for i, v in enumerate(vars_list)}
        for indep in self.lcn.independencies.get_assertions():
            for group in lmc_constraint_groups_vec(indep, table, col_of):
                if group[0] == 'conditional':
                    _, Aa, Ab, Ac, Ad = group
                    model.constr.add(
                        dot(Aa, model, model.ITEMS) * dot(Ab, model, model.ITEMS)
                        - dot(Ac, model, model.ITEMS) * dot(Ad, model, model.ITEMS)
                        == 0.0)
                else:
                    _, Aa, Ab, Ac = group
                    model.constr.add(
                        dot(Aa, model, model.ITEMS)
                        - dot(Ab, model, model.ITEMS) * dot(Ac, model, model.ITEMS)
                        == 0.0)

        return model

    # ------------------------------------------------------------------
    # Single SCIP solve for one (atom, sense)
    # ------------------------------------------------------------------
    def _scip_solve(self, interpretations, A, evidence_indicator, sense, solver,
                    gap_tol, den_floor, tee=False):
        """
        Globally optimize the marginal of one atom with SCIP.

        Linear objective when ``evidence_indicator`` is None (P(atom=1) = A@p);
        otherwise the fractional objective P(atom=1, e)/P(e) = (A*E)@p / (E@p),
        with an optional denominator floor P(e) >= den_floor.

        Args:
            tee: when True, stream SCIP's search log (per-node bound/gap progress)
                to stdout for this solve.

        Returns ``(value_or_None, gap_or_None, status, seconds)`` where status is
        ``"confirmed"`` (proven global optimum), ``"unconfirmed"`` (best feasible
        bound found, gap > tol / time limit hit) or ``"unsolved"`` (no incumbent).
        Never raises on a timeout.
        """
        model = self._build_base_model(interpretations)

        if evidence_indicator is None:
            obj_expr = dot(A, model, model.ITEMS)
        else:
            AE = A * evidence_indicator  # element-wise numpy multiply
            den = dot(evidence_indicator, model, model.ITEMS)
            if den_floor is not None and den_floor > 0.0:
                model.constr.add(den >= den_floor)
            obj_expr = dot(AE, model, model.ITEMS) / den

        model.objective = Objective(
            expr=obj_expr, sense=(minimize if sense == 'min' else maximize))

        t0 = time.time()
        results = solver.solve(model, load_solutions=False, tee=tee)
        elapsed = time.time() - t0

        tc = results.solver.termination_condition
        gap = _read_gap(results)

        # Robust incumbent read: the SCIPAMPL/NL interface always reports one
        # "solution" slot even on a pure timeout (no variable values), so
        # len(results.solution) is not a reliable signal. Attempt the load and
        # read with exception=False (returns None on uninitialized vars instead
        # of logging an error and raising); None means "no incumbent".
        obj_val = None
        try:
            model.solutions.load_from(results)
            v = value(model.objective, exception=False)
            obj_val = float(v) if v is not None else None
        except Exception:
            obj_val = None

        if obj_val is None:
            return None, gap, "unsolved", elapsed

        confirmed = (tc in _OPTIMAL) or (gap is not None and gap <= gap_tol + 1e-12)
        status = "confirmed" if confirmed else "unconfirmed"
        return obj_val, gap, status, elapsed

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(
            self,
            evidence: dict = {},
            debug: bool = False,
            verbosity: int = 2,
            time_limit: float = 3600.0,
            gap_tol: float = 0.0,
            den_floor: float = 1e-6,
            progress_bar: bool = True,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Run SCIP-backed exact inference to compute marginals for ALL singleton
        variables.

        For each non-evidence atom, SCIP globally minimizes and maximizes
        P(atom=1) (linear) or P(atom=1 | evidence) (fractional). Evidence atoms
        are returned as point masses.

        Args:
            evidence: dict {variable: value} of observed variables.
            debug: if True, show solver output and keep Pyomo's warning logs.
            verbosity: 0 silent; >0 prints the LMC + a results table; ``2``
                additionally streams SCIP's per-solve search log (node bound/gap
                progress) to stdout.
            time_limit: SCIP per-solve wall-clock limit in seconds (default 3600
                = one hour).
            gap_tol: relative optimality gap SCIP stops at; 0 = prove optimality.
                A returned bound is flagged ``confirmed`` only when the achieved
                gap is <= gap_tol (i.e. proven), else ``unconfirmed``.
            den_floor: floor on P(evidence) for the fractional objective, keeping
                the ratio well-defined. Set 0/None to disable. Ignored without
                evidence.
            progress_bar: when True (default) show the tqdm atom progress bar.
                Set False to suppress it (e.g. when streaming SCIP's search log
                at verbosity 2, where the bar would interleave with the log).

        Returns:
            Dict mapping variable name to (lower_bounds, upper_bounds) numpy
            arrays, each [P(=0), P(=1)] (interface-compatible with
            ExactInference.run). Per-atom solver verdicts are in ``self.status``.
        """
        assert self.lcn is not None, "Make sure the LCN model exists."
        assert self.lcn.independencies is not None, "Make sure the LMC is applied."

        t_start = time.time()
        evidence_set = set(evidence.keys())
        solver = make_scip(time_limit=time_limit, gap_tol=gap_tol)

        assertions = self.lcn.independencies.get_assertions()
        if verbosity > 0:
            print("[ExactInferenceSCIP] Computing all marginals (SCIP global)")
            print(f"[ExactInferenceSCIP] Evidence: {evidence}")
            print(f"[ExactInferenceSCIP] Local Markov Condition: "
                  f"{len(assertions)} independencies")
            for indep in assertions:
                print(f"  {indep}")
            print(f"[ExactInferenceSCIP] Per-solve time limit: {time_limit:.0f}s, "
                  f"gap tolerance: {gap_tol}")

        # Interpretation table (shared across all solves).
        vars_list = [k for k, _ in self.lcn.atoms.items()]
        items = list(itertools.product([0, 1], repeat=len(vars_list)))
        interpretations = [dict(zip(vars_list, t)) for t in items]

        # Pre-compute atom indicators and the evidence indicator (once).
        atom_indicators = {
            v: eval_indicator(Formula(label=v, formula=v), interpretations)
            for v in vars_list}
        evidence_indicator = None
        if len(evidence) > 0:
            Fe = make_conjunction(variables=list(evidence.keys()), literals=evidence)
            evidence_indicator = eval_indicator(Fe, interpretations)

        self.marginals = {}
        self.status = {}
        self.feasible = True
        solve_atoms = [a for a in vars_list if a not in evidence_set]

        # Suppress Pyomo's routine warning-status spam (emitted on every non-
        # optimal termination, which is expected here -- we report gap/status
        # ourselves). Restore afterwards.
        pyomo_logger = logging.getLogger('pyomo')
        prev_level = pyomo_logger.level
        if not debug:
            pyomo_logger.setLevel(logging.ERROR)

        # Stream SCIP's search log at verbosity 2. The tqdm bar is controlled
        # independently by progress_bar; both can be on, but the streamed log
        # then interleaves with the bar -- pass progress_bar=False to avoid that.
        tee = (verbosity == 2)

        n_confirmed = n_unconfirmed = n_unsolved = 0
        pbar = tqdm(total=len(solve_atoms), desc="[ExactInferenceSCIP] atoms",
                    disable=(verbosity == 0 or not progress_bar))
        try:
            for atom_name in vars_list:
                if atom_name in evidence_set:
                    ev_val = evidence[atom_name]
                    lo_arr = np.zeros(2)
                    hi_arr = np.zeros(2)
                    lo_arr[ev_val] = 1.0
                    hi_arr[ev_val] = 1.0
                    self.marginals[atom_name] = (lo_arr, hi_arr)
                    continue

                A = atom_indicators[atom_name]
                if tee:
                    print(f"\n[ExactInferenceSCIP] === SCIP search: "
                          f"minimize P({atom_name}=1) ===")
                lo = self._scip_solve(interpretations, A, evidence_indicator,
                                      'min', solver, gap_tol, den_floor, tee=tee)
                if tee:
                    print(f"\n[ExactInferenceSCIP] === SCIP search: "
                          f"maximize P({atom_name}=1) ===")
                hi = self._scip_solve(interpretations, A, evidence_indicator,
                                      'max', solver, gap_tol, den_floor, tee=tee)
                self.status[atom_name] = {'min': lo, 'max': hi}

                lo_val, _, lo_status, _ = lo
                hi_val, _, hi_status, _ = hi

                # Vacuous fallbacks when a side is unsolved (no incumbent).
                lo_1 = min(max(lo_val, 0.0), 1.0) if lo_val is not None else 0.0
                hi_1 = min(max(hi_val, 0.0), 1.0) if hi_val is not None else 1.0
                if lo_status == "unsolved" or hi_status == "unsolved":
                    self.feasible = False

                lo_arr = np.array([1.0 - hi_1, lo_1])
                hi_arr = np.array([1.0 - lo_1, hi_1])
                self.marginals[atom_name] = (lo_arr, hi_arr)

                for st in (lo_status, hi_status):
                    if st == "confirmed":
                        n_confirmed += 1
                    elif st == "unconfirmed":
                        n_unconfirmed += 1
                    else:
                        n_unsolved += 1

                pbar.set_postfix_str(
                    f"{atom_name}=[{lo_1:.3f},{hi_1:.3f}] "
                    f"conf={n_confirmed} unconf={n_unconfirmed} "
                    f"unsolved={n_unsolved}")
                pbar.update(1)
        finally:
            pbar.close()
            pyomo_logger.setLevel(prev_level)

        t_end = time.time()

        if verbosity > 0:
            self._print_report(evidence_set, t_end - t_start,
                               n_confirmed, n_unconfirmed, n_unsolved)

        return self.marginals

    def _print_report(self, evidence_set, elapsed, n_conf, n_unconf, n_unsolved):
        """Print the per-atom bound/gap/status table and a summary line."""
        print("[ExactInferenceSCIP] Singleton variable marginals "
              "(P=1 bound | gap | status):")
        for atom_name in sorted(self.status):
            (lo_v, lo_g, lo_s, lo_t) = self.status[atom_name]['min']
            (hi_v, hi_g, hi_s, hi_t) = self.status[atom_name]['max']
            print(f"  {atom_name}: "
                  f"min={_fmt(lo_v, lo_g, lo_s, lo_t)}  "
                  f"max={_fmt(hi_v, hi_g, hi_s, hi_t)}")
        print(f"[ExactInferenceSCIP] Feasible: {self.feasible}")
        print(f"[ExactInferenceSCIP] Solves: confirmed optimal={n_conf} | "
              f"unconfirmed (best-so-far)={n_unconf} | unsolved={n_unsolved}")
        if n_unconf or n_unsolved:
            print("[ExactInferenceSCIP] Note: unconfirmed/unsolved bounds are "
                  "NOT proven optimal -- raise time_limit to certify.")
        print(f"[ExactInferenceSCIP] Time elapsed: {elapsed:.4f} sec")


def _n(v):
    return f"{v:.6f}" if v is not None else "  n/a   "


def _fmt(v, gap, status, secs):
    g = f"gap={gap:.2g}" if gap is not None else "gap=?"
    return f"{_n(v)} [{g}, {status}, {secs:.1f}s]"


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

    # Run SCIP-backed exact marginal inference (no evidence). A short per-solve
    # time limit keeps the demo responsive; the library default is 3600s (1h).
    print("\n=== ExactInferenceSCIP (no evidence) ===")
    algo = ExactInferenceSCIP(lcn=lcn_model)
    results = algo.run(evidence={}, debug=False, verbosity=2, progress_bar=False)
    print_singleton_marginals(results)
