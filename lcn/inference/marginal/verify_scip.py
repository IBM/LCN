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

# SCIP-based GLOBAL verifier for the marginal bounds computed by exact.py.
#
# ExactInference (exact.py) solves the nonconvex bilinear-LMC NLP with ipopt + an
# SLSQP fallback -- both *local* methods, which can return over-tight (or just
# wrong) bounds (see the seed-clustering bug fixed on alarm.lcn). SCIP is a
# spatial branch-and-bound *global* solver: it builds McCormick-envelope convex
# relaxations of the bilinear products and closes the optimality gap, so when it
# terminates with gap 0 the reported min/max is a CERTIFIED global bound.
#
# This script formulates exactly the same model exact.py optimizes (the 2^n joint
# simplex + sentence-bound inequalities + bilinear LMC equalities, built with the
# shared common.py helpers) and asks SCIP for the global min and max of each
# singleton marginal P(atom=1), then prints them side-by-side with ExactInference.
#
# CAVEAT (the price of global optimality): bilinear B&B is NP-hard. Some
# sub-problems do not certify within a time budget -- SCIP then returns either a
# valid-but-uncertified incumbent (PARTIAL) or nothing (UNSOLVED). The code
# handles both without crashing and reports the optimality gap so you can tell
# which bounds are proven.
#
# Requires the SCIP CLI binary on PATH (`brew install scip`). Pyomo's
# SolverFactory('scip') drives it via the AMPL/NL interface, like ipopt.

import argparse
import itertools
import math
import time

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

# Local
from lcn.core.model import LCN, SentenceType, Formula
from lcn.inference.marginal.exact import ExactInference
from lcn.inference.utils.common import (
    eval_indicator,
    dot,
    build_truth_table,
    lmc_constraint_groups_vec,
)


# Termination conditions that mean SCIP proved optimality. A time/iteration
# limit that still loaded an incumbent yields a valid-but-unproven (PARTIAL)
# bound; that case is detected via the gap, not enumerated here.
_OPTIMAL = {
    TerminationCondition.optimal,
    TerminationCondition.locallyOptimal,
    TerminationCondition.globallyOptimal,
    TerminationCondition.feasible,
}


def build_base_model(lcn: LCN):
    """
    Build the Pyomo model exact.py optimizes: the 2^n joint-probability simplex
    with the LCN sentence bounds and the bilinear LMC equality constraints. No
    objective is attached (the caller sets one per solve). Returns
    ``(model, interpretations, N)``.

    Mirrors exact._build_base_model / exact_map.solve_exact_model exactly, reusing
    the shared indicator/LMC helpers so this is the *same* feasible set, not a
    re-derivation.
    """
    vars_list = [k for k, _ in lcn.atoms.items()]
    items = list(itertools.product([0, 1], repeat=len(vars_list)))
    interps = [dict(zip(vars_list, t)) for t in items]
    N = len(interps)

    model = ConcreteModel()
    model.ITEMS = Set(initialize=range(N))
    model.p = Var(model.ITEMS, within=NonNegativeReals, bounds=(0.0, 1.0))
    model.constr = ConstraintList()

    # Probability distribution constraint.
    model.constr.add(sum(model.p[i] for i in model.ITEMS) == 1.0)

    # Sentence-bound constraints.
    for _, s in lcn.sentences.items():
        lobo = s.get_lower_bound()
        upbo = s.get_upper_bound()
        if s.type == SentenceType.Type1:
            A = eval_indicator(s.phi_formula, interps)
            expr = dot(A, model, model.ITEMS)
            model.constr.add(expr >= lobo)
            model.constr.add(expr <= upbo)
        else:
            Aqr = eval_indicator(s.phi_and_psi_formula, interps)
            Ar = eval_indicator(s.psi_formula, interps)
            expr_qr = dot(Aqr, model, model.ITEMS)
            expr_r = dot(Ar, model, model.ITEMS)
            model.constr.add(expr_qr >= lobo * expr_r)
            model.constr.add(expr_qr <= upbo * expr_r)

    # LMC independence constraints (bilinear equalities) -- the nonconvex part.
    table = build_truth_table(len(vars_list))
    col_of = {v: i for i, v in enumerate(vars_list)}
    for indep in lcn.independencies.get_assertions():
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

    return model, interps, N


def make_scip(time_limit=60.0, gap_tol=0.0):
    """
    Return a configured SCIP solver, or raise a clear error if SCIP is missing.

    Args:
        time_limit: per-solve wall-clock limit in seconds (SCIP ``limits/time``).
        gap_tol: relative optimality gap to stop at (SCIP ``limits/gap``); 0
            means prove global optimality.
    """
    solver = SolverFactory('scip')
    if not solver.available(exception_flag=False):
        raise RuntimeError(
            "SCIP solver not found. Install the SCIP CLI binary and ensure it is "
            "on PATH (macOS: `brew install scip`). Pyomo drives it via the "
            "AMPL/NL interface.")
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


def scip_bound(lcn: LCN, atom: str, sense: str, solver):
    """
    Globally optimize P(atom=1) over the LCN+LMC model with SCIP.

    Returns ``(value_or_None, gap_or_None, status, seconds)`` where status is:
      "CERTIFIED" - optimal termination (or a limit with gap ~0): proven bound.
      "PARTIAL"   - hit a time/iteration limit but an incumbent was loaded: the
                    value is a valid feasible bound, not proven optimal.
      "UNSOLVED"  - no usable point found (e.g. timed out before any incumbent).

    Never raises on a timeout: solve with ``load_solutions=False`` and only read
    ``value`` once we know a solution was loaded (a bare ``value()`` on an
    uninitialized model raises).
    """
    model, interps, _ = build_base_model(lcn)
    A = eval_indicator(Formula(label=atom, formula=atom), interps)
    model.objective = Objective(
        expr=dot(A, model, model.ITEMS),
        sense=(minimize if sense == 'min' else maximize))

    t0 = time.time()
    results = solver.solve(model, load_solutions=False, tee=False)
    elapsed = time.time() - t0

    tc = results.solver.termination_condition
    gap = _read_gap(results)

    # Try to read the incumbent. The SCIPAMPL/NL interface always reports one
    # "solution" slot even on a pure timeout (status stoppedByLimit), and that
    # slot may carry NO variable values -- so len(results.solution) is not a
    # reliable signal. The robust test is to attempt the load and then read with
    # exception=False (returns None on uninitialized vars instead of logging an
    # error and raising), treating None as "no incumbent".
    obj_val = None
    try:
        model.solutions.load_from(results)
        v = value(model.objective, exception=False)
        obj_val = float(v) if v is not None else None
    except Exception:
        obj_val = None

    if obj_val is None:
        return None, gap, "UNSOLVED", elapsed

    # A point exists. It is CERTIFIED when SCIP proved optimality (optimal
    # termination, or a limit reached with gap ~0); otherwise it is a valid but
    # unproven feasible bound (PARTIAL).
    certified = (tc in _OPTIMAL) or (gap is not None and gap <= 1e-6)
    status = "CERTIFIED" if certified else "PARTIAL"
    return obj_val, gap, status, elapsed


def run(lcn: LCN, time_limit=60.0, gap_tol=0.0, compare=True, tol=1e-3):
    """
    Compute SCIP global marginal bounds for every singleton atom and (optionally)
    compare them against ExactInference. Prints a table and returns the per-atom
    results dict ``{atom: {'min': (..), 'max': (..)}}``.
    """
    solver = make_scip(time_limit=time_limit, gap_tol=gap_tol)
    atom_names = [k for k, _ in lcn.atoms.items()]

    print(f"Solving {len(atom_names)} atoms x 2 senses with SCIP "
          f"(per-solve limit {time_limit:.0f}s, gap_tol {gap_tol})...")
    scip_res = {}
    for atom in atom_names:
        lo = scip_bound(lcn, atom, 'min', solver)
        hi = scip_bound(lcn, atom, 'max', solver)
        scip_res[atom] = {'min': lo, 'max': hi}
        print(f"  {atom}: min={_fmt(lo)}  max={_fmt(hi)}")

    if not compare:
        return scip_res

    print("\nRunning ExactInference (the oracle under test)...")
    exact = ExactInference(lcn=lcn).run(evidence={}, verbosity=0)

    print(f"\n{'atom':<6} {'SCIP [lo, hi]':<24} {'status (lo/hi)':<22} "
          f"{'exact [lo, hi]':<22} flag")
    print("-" * 84)
    n_ok = n_mismatch = n_uncert = 0
    for atom in sorted(scip_res):
        lo_v, _, lo_s, _ = scip_res[atom]['min']
        hi_v, _, hi_s, _ = scip_res[atom]['max']
        elo, ehi = exact[atom][0][1], exact[atom][1][1]

        scip_str = (f"[{_n(lo_v)}, {_n(hi_v)}]")
        stat_str = f"{lo_s}/{hi_s}"
        ex_str = f"[{elo:.4f}, {ehi:.4f}]"

        # Verdict only where SCIP certified that side.
        lo_cert = lo_s == "CERTIFIED"
        hi_cert = hi_s == "CERTIFIED"
        mism = ((lo_cert and abs(lo_v - elo) > tol)
                or (hi_cert and abs(hi_v - ehi) > tol))
        if not (lo_cert and hi_cert):
            flag = "UNCERTIFIED"
            n_uncert += 1
        elif mism:
            flag = "MISMATCH"
            n_mismatch += 1
        else:
            flag = "OK"
            n_ok += 1
        print(f"{atom:<6} {scip_str:<24} {stat_str:<22} {ex_str:<22} {flag}")
    print("-" * 84)
    print(f"summary: {n_ok} OK | {n_mismatch} MISMATCH | "
          f"{n_uncert} UNCERTIFIED")
    if n_mismatch:
        print("  -> MISMATCH: a CERTIFIED global bound disagrees with exact.py "
              "beyond tol; exact.py's bound is wrong on that side.")
    if n_uncert:
        print("  -> UNCERTIFIED: SCIP could not prove the bound within the time "
              "limit (PARTIAL/UNSOLVED); raise --time-limit to certify.")
    return scip_res


def _n(v):
    return f"{v:.4f}" if v is not None else "  n/a "


def _fmt(res):
    v, gap, status, secs = res
    g = f", gap={gap:.2g}" if gap is not None else ""
    return f"{_n(v)} [{status}{g}, {secs:.1f}s]"


def main():
    parser = argparse.ArgumentParser(
        description="SCIP global verifier for exact.py marginal bounds.")
    parser.add_argument("--file", default="examples/alarm.lcn",
                        help="LCN file (default examples/alarm.lcn).")
    parser.add_argument("--time-limit", type=float, default=60.0,
                        help="SCIP per-solve time limit in seconds (default 60).")
    parser.add_argument("--gap", type=float, default=0.0,
                        help="SCIP relative gap tolerance; 0 = prove global "
                             "optimality (default 0).")
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip the ExactInference comparison.")
    args = parser.parse_args()

    lcn = LCN()
    lcn.from_lcn(file_name=args.file)
    assert lcn.independencies is not None, "LMC independencies not populated."

    print(f"=== SCIP global verifier for {args.file} ===")
    print(f"Variables: {[k for k, _ in lcn.atoms.items()]}")

    run(lcn, time_limit=args.time_limit, gap_tol=args.gap,
        compare=not args.no_compare)


if __name__ == "__main__":
    main()
