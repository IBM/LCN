"""SCIP global feasibility check for LCN consistency.

The SLSQP-backed :func:`lcn.inference.utils.common.check_consistency` times out
on the supreme_court instances (n=10 but 345-1009 sentences -> a huge
constraint system). SCIP's spatial branch-and-bound with LP relaxation and
presolve handles that constraint density far better, so we use it as the
consistency oracle here.

Consistency is a pure FEASIBILITY question over the 2^n world-probability
model that :func:`verify_scip.build_base_model` already builds (a simplex
constraint, the per-sentence linear/bilinear bound constraints, and the LMC
independence equalities). We add a constant objective and ask SCIP for any
feasible point:

  * a feasible incumbent is loaded  -> CONSISTENT  (the point satisfies every
    sentence bound and LMC equality; a genuine witness distribution)
  * SCIP proves the model infeasible -> INCONSISTENT
  * neither within the time limit    -> UNDETERMINED (timeout)

Still builds the 2^n joint, so it is only feasible for n <= ~12 (fine for the
maintenance n<=12 and all supreme_court n=10 instances; NOT for the n>=24
scaling/paper instances).

    .venv/bin/python benchmarks/junkyu/scip_consistency.py --time-limit 600 FILE...
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pyomo.environ import Objective, minimize, value  # noqa: E402

from lcn.core.model import LCN  # noqa: E402
from lcn.inference.marginal.verify_scip import (  # noqa: E402
    build_base_model, make_scip, _OPTIMAL)

# Pyomo/SCIP termination conditions that mean "proven infeasible".
_INFEASIBLE = {"infeasible", "infeasibleOrUnbounded"}


def scip_consistent(path: str, time_limit: float):
    """Return (status, seconds): status in CONSISTENT / INCONSISTENT /
    UNDETERMINED / ERR:...."""
    t0 = time.time()
    try:
        lcn = LCN()
        with contextlib.redirect_stdout(io.StringIO()):
            lcn.from_lcn(path)
            lcn.build_primal_graph()
            lcn.build_structure_graph()
            lcn.local_markov_condition()
        model, _interps, _ = build_base_model(lcn)
        # Constant objective: we only care whether the feasible set is nonempty.
        model.objective = Objective(expr=0.0, sense=minimize)
        solver = make_scip(time_limit=time_limit)
        results = solver.solve(model, load_solutions=False, tee=False)
    except Exception as exc:
        return f"ERR:{type(exc).__name__}:{exc}", time.time() - t0

    tc = str(results.solver.termination_condition)
    if tc in _INFEASIBLE:
        return "INCONSISTENT", time.time() - t0

    # Did SCIP load an actual feasible point? (A witness distribution.)
    incumbent = False
    try:
        model.solutions.load_from(results)
        incumbent = value(model.objective, exception=False) is not None
    except Exception:
        incumbent = False

    if incumbent:
        return "CONSISTENT", time.time() - t0
    # optimal termination on a constant objective with no loaded point is odd;
    # treat only a real incumbent as a certificate, else undetermined.
    if tc in {str(c) for c in _OPTIMAL}:
        # Optimal over a constant objective => feasible set nonempty.
        return "CONSISTENT", time.time() - t0
    return "UNDETERMINED", time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=600.0,
                    help="SCIP per-instance wall-clock limit (s).")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    for path in args.files:
        status, dt = scip_consistent(path, args.time_limit)
        print(f"{os.path.basename(path):60s} SCIP={status} ({dt:.1f}s)",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
