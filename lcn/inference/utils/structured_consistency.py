"""Structure-exploiting LCN consistency (feasibility) check via a junction tree.

The exact :func:`lcn.inference.utils.common.check_consistency` builds a ``2^n``
joint model and is capped at ``n <= 10``. This module decides the SAME
feasibility question over JUNCTION-TREE CLUSTERS of size ``2^treewidth``
instead, so it scales to any bounded-treewidth LCN (the large chain/DAG/ktree
benchmarks, n = 24..40, and beyond).

It reuses the CredalJT (scheme D5) machinery, which already builds exactly the
constraint polytope we need:

  * ``CredalNetworkVertices.from_lcn(..., solve_families=False)`` -- the cheap
    structure-only chain-graph build (no per-family interval solves).
  * ``junction_nlp._build_jt_and_hosts`` -- the running-intersection junction
    tree plus the assignment of every sentence / non-RIP-implied LMC assertion
    to a host cluster (and the ``over_budget`` / ``fallback`` flags).
  * ``junction_nlp._build_constraint_model`` -- one variable per (cluster, local
    world), per-cluster simplex, Lauritzen separator-consistency equalities,
    sentence bound rows, and LMC bilinear rows. NO objective: a feasible SET.

Consistency = is that feasible set nonempty? We add a constant objective and ask
a solver for any feasible point.

SOUNDNESS (why the verdict is trustworthy):

  * The JT model imposes a SUBSET of the true joint constraints (only hosted
    sentences/LMCs; separators enforce running-intersection-implied
    independence structurally; unhostable LMCs are dropped only when RIP-implied
    or chain-graph-inert). It is therefore a RELAXATION of the true feasible set.
  * INFEASIBLE => the relaxation is empty => the true feasible set is empty
    => genuinely INCONSISTENT. Sound even under over_budget/fallback/dropped
    LMCs. Requires a GLOBAL solver (SCIP) because the LMC rows are bilinear
    (nonconvex); an ipopt "infeasible" is NOT a proof.
  * FEASIBLE and NOT over_budget and NOT fallback => a feasible solution's
    separator-consistent cluster marginals compose (running intersection) into a
    global joint satisfying every hosted constraint, and the dropped LMCs are
    inert => a genuine witness distribution exists => CONSISTENT. An ipopt
    incumbent is enough here (a real point is a real point).
  * FEASIBLE but over_budget/fallback => the relaxation was satisfiable but not
    exact => UNDETERMINED (reported honestly, never claimed consistent).

Unlike the generator's product-distribution witness (sound but CONSERVATIVE --
it false-rejects instances consistent only via a non-product distribution), this
check is both sound AND complete whenever the junction tree stays within the
treewidth budget (the common case for the bounded-treewidth generator
topologies), so it accepts every genuinely consistent instance.
"""

from __future__ import annotations

import contextlib
import io
import time
from dataclasses import dataclass
from typing import Optional

from lcn.core.model import LCN

# Verdict labels.
CONSISTENT = "CONSISTENT"
INCONSISTENT = "INCONSISTENT"
UNDETERMINED = "UNDETERMINED"
ERROR = "ERROR"

# Pyomo/SCIP termination conditions that mean "proven infeasible".
_INFEASIBLE = {"infeasible", "infeasibleOrUnbounded"}


@dataclass
class StructuredConsistencyResult:
    """Outcome of :func:`check_consistency_structured`.

    Attributes:
        status: one of ``CONSISTENT`` / ``INCONSISTENT`` / ``UNDETERMINED`` /
            ``ERROR``.
        treewidth: junction-tree treewidth (max cluster atoms - 1), or ``None``
            if the tree could not be built.
        seconds: wall-clock time for the whole check.
        note: short human-readable explanation (e.g. the reason for
            UNDETERMINED, or the exception text for ERROR).
    """
    status: str
    treewidth: Optional[int]
    seconds: float
    note: str = ""

    @property
    def consistent(self) -> Optional[bool]:
        """True/False for a definitive verdict, ``None`` if not determined."""
        if self.status == CONSISTENT:
            return True
        if self.status == INCONSISTENT:
            return False
        return None


def check_consistency_structured(
    lcn: LCN,
    *,
    max_cluster_atoms: int = 16,
    time_limit: float = 600.0,
    solver: str = "scip",
) -> StructuredConsistencyResult:
    """Decide LCN consistency over a junction tree (bounded treewidth).

    Args:
        lcn: the model to check (loaded via ``LCN.from_lcn``; the primal graph /
            LMC are built lazily by the junction-tree builder if absent).
        max_cluster_atoms: treewidth budget as the maximum number of atoms per
            junction-tree cluster (cost is ``sum_c 2^|cluster atoms|``). A tree
            whose widest cluster exceeds this yields ``UNDETERMINED``.
        time_limit: per-solve wall-clock limit in seconds.
        solver: ``"scip"`` (default; global, needed for a trustworthy
            INCONSISTENT) or ``"ipopt"`` (local; a CONSISTENT verdict is still
            sound, but INCONSISTENT is not a proof and is downgraded to
            UNDETERMINED with a note).

    Returns:
        A :class:`StructuredConsistencyResult`.
    """
    # Deferred imports: junction_nlp imports lcn.inference.utils.common, and the
    # generator imports common, so importing these at module top level would
    # create a cycle. Import inside the function instead (matches the lazy-import
    # idiom used elsewhere in utils).
    from pyomo.environ import Objective, minimize, value
    from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
    from lcn.inference.marginal.cn.junction_nlp import (
        _build_jt_and_hosts, _build_constraint_model)
    from lcn.inference.marginal.exact import make_scip
    from lcn.inference.utils.common import make_ipopt

    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            # Structure-only credal network (no per-family interval solves).
            cnv = CredalNetworkVertices.from_lcn(
                lcn, method="linear", solve_families=False,
                enumerate_vertices=False, cache=False, verbosity=0)
            jt, sents_by_host, lmc_by_host, _, over_budget, fallback = \
                _build_jt_and_hosts(cnv, None, {}, max_cluster_atoms)
        treewidth = jt.max_cluster_size() - 1

        if over_budget:
            return StructuredConsistencyResult(
                UNDETERMINED, treewidth, time.time() - t0,
                f"treewidth {treewidth} exceeds budget {max_cluster_atoms - 1}")

        with contextlib.redirect_stdout(io.StringIO()):
            model, _csize = _build_constraint_model(
                jt, cnv, sents_by_host, lmc_by_host)
            # Constant objective: we only ask whether the feasible set is nonempty.
            model.objective = Objective(expr=0.0, sense=minimize)
            if solver == "scip":
                opt = make_scip(time_limit=time_limit)
            else:
                opt = make_ipopt(debug=False, mode="exact")
            results = opt.solve(model, load_solutions=False, tee=False)
    except Exception as exc:  # pragma: no cover - defensive
        return StructuredConsistencyResult(
            ERROR, None, time.time() - t0, f"{type(exc).__name__}: {exc}")

    tc = str(results.solver.termination_condition)
    if tc in _INFEASIBLE:
        if solver != "scip":
            # A local solver's "infeasible" is not a proof of emptiness.
            return StructuredConsistencyResult(
                UNDETERMINED, treewidth, time.time() - t0,
                "ipopt reported infeasible (not a proof; re-run with --solver scip)")
        return StructuredConsistencyResult(
            INCONSISTENT, treewidth, time.time() - t0, "")

    # Did the solver load an actual feasible point (a witness distribution)?
    incumbent = False
    try:
        model.solutions.load_from(results)
        incumbent = value(model.objective, exception=False) is not None
    except Exception:
        incumbent = False

    if not incumbent:
        return StructuredConsistencyResult(
            UNDETERMINED, treewidth, time.time() - t0,
            f"no feasible point within limit ({tc})")

    # A feasible point exists. Exact iff every sentence was hosted (not fallback).
    if fallback:
        return StructuredConsistencyResult(
            UNDETERMINED, treewidth, time.time() - t0,
            "relaxation feasible but a sentence was unhostable")
    return StructuredConsistencyResult(
        CONSISTENT, treewidth, time.time() - t0, "witness distribution found")


def is_consistent_structured(
    lcn: LCN,
    *,
    max_cluster_atoms: int = 16,
    time_limit: float = 600.0,
    solver: str = "scip",
) -> Optional[bool]:
    """Boolean-gate wrapper around :func:`check_consistency_structured`.

    Returns ``True`` (consistent), ``False`` (inconsistent), or ``None`` when the
    check could not reach a definitive verdict (UNDETERMINED / ERROR). Callers
    that need a hard gate (e.g. the random-instance generator) treat ``None`` as
    "not certified" and reject.
    """
    return check_consistency_structured(
        lcn, max_cluster_atoms=max_cluster_atoms,
        time_limit=time_limit, solver=solver).consistent
