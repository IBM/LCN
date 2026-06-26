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
from lcn.core.independencies import Independencies
from lcn.inference.utils.common import (
    make_conjunction, check_consistency, make_ipopt,
    lmc_constraint_groups_vec, build_truth_table,
    find_feasible_points, optimize_marginal_slsqp,
    eval_indicator, dot
)

_N_RESTARTS = 4             # random-restart budget on a failed/vacuous solve

# "lightning" speed preset: clamp the per-solve time limit to this many seconds
# (and, for the local solver, drop to the fast ipopt config). A quick, best-effort
# pass -- bounds may be loose (local) or returned as "unconfirmed" (global).
LIGHTNING_TIME_LIMIT = 10.0


# The ipopt configuration is shared across the whole inference suite and lives
# in lcn.inference.utils.common; alias it here for the internal call sites.
_make_ipopt = make_ipopt


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


def _n(v):
    return f"{v:.6f}" if v is not None else "  n/a   "


def _fmt(v, gap, status, secs):
    g = f"gap={gap:.2g}" if gap is not None else "gap=?"
    return f"{_n(v)} [{g}, {status}, {secs:.1f}s]"


def _point_mass(ev_val: int) -> Tuple[np.ndarray, np.ndarray]:
    """Marginal of an evidence variable: a point mass at its observed value.

    Returns the ``(lower, upper)`` pair (both [P(=0), P(=1)]) used for an atom
    fixed by evidence -- identical for the local and global backends.
    """
    lo_arr = np.zeros(2)
    hi_arr = np.zeros(2)
    lo_arr[ev_val] = 1.0
    hi_arr[ev_val] = 1.0
    return lo_arr, hi_arr


# A P(atom=1) bound is "vacuous" when it is the whole unit interval [0, 1]: the
# solver returned no information about the atom. A solve that lands here is
# either a genuine [0,1] marginal or (far more often) a sign the solver could
# not pin the bound -- when EVERY non-evidence atom is vacuous, the model is
# almost certainly infeasible (e.g. inconsistent evidence).
_VACUOUS_EPS = 1e-6


def _is_vacuous(lo_1: float, hi_1: float, eps: float = _VACUOUS_EPS) -> bool:
    """True if the P(atom=1) bound spans the entire [0, 1] interval."""
    return lo_1 <= eps and hi_1 >= 1.0 - eps


def _degenerate_warning(evidence: dict, reason: str) -> str:
    """Build the all-vacuous / all-infeasible diagnostic message.

    ``reason`` is a short phrase describing what was observed (e.g. "all
    marginals are vacuous [0,1]"). When evidence is present the message points
    at it as the likely culprit, since inconsistent evidence is the most common
    cause of a uniformly degenerate result.
    """
    msg = f"[ExactInference] WARNING: {reason} -- the result is uninformative."
    if evidence:
        msg += (f"\n[ExactInference] The evidence {evidence} is most likely "
                f"INCONSISTENT with the LCN (no distribution satisfies the "
                f"constraints together with this evidence). Check the evidence "
                f"or run check_consistency on the model.")
    else:
        msg += ("\n[ExactInference] The LCN is most likely INCONSISTENT "
                "(over-constrained); run check_consistency on the model.")
    return msg


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


def _solve_with_objective(model, obj_expr, sense, solver, debug=False, tee=None):
    """
    Attach an objective to the model, solve, and return the result.
    Removes any existing objective before adding the new one.

    Args:
        model: Pyomo ConcreteModel with constraints.
        obj_expr: Pyomo expression for the objective.
        sense: 'min' or 'max'.
        solver: Reusable SolverFactory instance.
        debug: If True, show diagnostics (exception prints below).
        tee: If True, stream the solver's search log to stdout. When None
            (default), falls back to ``debug`` -- this keeps existing 5-argument
            positional callers (e.g. sccp.py) behaving exactly as before.

    Returns:
        (objective_value, feasible) tuple. A solve is considered feasible/usable
        when ipopt reports an ``optimal``, ``locallyOptimal``, ``feasible`` or
        ``acceptable`` termination --- ipopt's "acceptable" point is a valid
        solution and must not be discarded as if it were infeasible. Only a
        genuine ``infeasible`` termination or an exception yields feasible=False.
    """
    if tee is None:
        tee = debug

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
        results = solver.solve(model, load_solutions=True, tee=tee)
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
                  checks=None, obj_vec=None, seeds_provider=None,
                  use_slsqp_fallback=True, tee=None):
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
        use_slsqp_fallback: bool, when False the SLSQP fallback is disabled and
            the bound comes from ipopt (primary + random restarts) only. The
            seeds_provider is then never invoked (no feasible-seed search).
        tee: optional bool forwarded to ipopt to stream its search log (used at
            verbosity 2). None defers to ``debug``.

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
    best, ok = _solve_with_objective(model, obj_expr, sense, solver, debug, tee)

    if _is_suspicious(best, ok):
        for k in range(_N_RESTARTS):
            # Deterministic per-(atom, sense, restart) seed -> reproducible runs.
            seed = abs(hash((atom, sense, k))) % (2 ** 32)
            _init_p(model, N, rng=np.random.default_rng(seed))
            v, okk = _solve_with_objective(model, obj_expr, sense, solver, debug, tee)
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
    if (use_slsqp_fallback and checks is not None and obj_vec is not None
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

    Computes lower/upper marginal bounds for every singleton variable by solving,
    for each atom, a min and a max of P(atom=1) over the joint distributions
    consistent with the LCN. Two solver backends are available, selected by the
    ``solver`` argument of ``run``:

      * ``"local"`` (default): ipopt with a two-phase SLSQP fallback. Fast, but a
        *local* method on this nonconvex bilinear-LMC NLP -- bounds may be loose.
      * ``"global"``: SCIP (spatial branch-and-bound). Either certifies the global
        optimum or, on a time-out, returns the best bound found so far together
        with the optimality gap. Requires the optional SCIP CLI on PATH.

    After ``run`` with ``solver="global"``, ``self.status`` holds the per-atom SCIP
    verdict; it stays ``None`` for the local solver. ``self.solver_used`` records
    which backend ran.
    """

    def __init__(
            self,
            lcn: LCN
    ):
        self.lcn = lcn
        self.marginals = None
        # Last query bounds, set by run_query(); None until then.
        self.lower_bound = None
        self.upper_bound = None
        self.feasible = None
        # True when the run produced a uniformly uninformative result -- every
        # non-evidence marginal is the vacuous [0,1] bound, or every solve was
        # infeasible/unsolved. A strong signal that the LCN (or the LCN together
        # with the supplied evidence) is inconsistent. None until run().
        self.degenerate = None
        # Per-atom global-solver verdict {atom: {'min': (value, gap, status,
        # secs), 'max': (...)}}; populated only by the "global" (SCIP) solver.
        self.status = None
        # Which backend the last run() used ("local" or "global").
        self.solver_used = None

    def run(
            self,
            evidence: dict = {},
            debug: bool = False,
            verbosity: int = 2,
            solver: str = "local",
            mode: str = "slow",
            use_slsqp_fallback: bool = True,
            time_limit: float = 3600.0,
            lightning: bool = False,
            progress_bar: bool = True,
            gap_tol: float = 0.0,
            den_floor: float = 1e-6,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Run exact inference to compute marginals for ALL singleton variables.

        For each non-evidence atom, minimizes and maximizes P(atom=1) (linear,
        no evidence) or P(atom=1 | evidence) (fractional). Evidence atoms are
        returned as point masses.

        Args:
            evidence: dict {variable: value} of observed variables.
            debug: bool
                Show solver diagnostics and keep Pyomo's warning logs.
            verbosity: int
                0 silent; >0 prints the LMC + a results table; ``2`` additionally
                streams the solver's per-solve search log to stdout and (because
                that would interleave with the bar) suppresses the progress bar.
            solver: str
                Which backend to use: ``"local"`` (default; ipopt + SLSQP
                fallback) or ``"global"`` (SCIP spatial branch-and-bound, the
                optional SCIP CLI must be on PATH).
            mode: str
                Local solver only. ipopt accuracy config: ``"slow"`` (default,
                high accuracy) or ``"fast"`` (stops at the first acceptable point,
                trading accuracy for speed). Ignored by the global solver.
            use_slsqp_fallback: bool
                Local solver only. When True (default) a two-phase SLSQP fallback
                backstops ipopt on suspicious/failed solves. Set False to use
                ipopt alone (primary + random restarts).
            time_limit: float
                Per-solve time limit in seconds (default 3600 = one hour),
                applied to BOTH backends. For the local solver this is ipopt CPU
                time (``max_cpu_time``, with a best-effort wall cap); for the
                global solver it is SCIP wall time (``limits/time``).
            lightning: bool
                Speed preset. Clamps ``time_limit`` to a few seconds; for the
                local solver it also forces ``mode="fast"`` and disables the SLSQP
                fallback (unless the caller explicitly set it True); for the
                global solver it relaxes ``gap_tol`` to 0.1 when left at 0. A
                quick best-effort pass -- bounds may be loose / unconfirmed.
            progress_bar: bool
                Show the tqdm atom progress bar (default True). Forced off at
                verbosity 2 (see above) and at verbosity 0.
            gap_tol: float
                Global solver only. Relative optimality gap SCIP stops at; 0 =
                prove optimality. A bound is ``confirmed`` only when the achieved
                gap is <= gap_tol, else ``unconfirmed``.
            den_floor: float
                Global solver only. Floor on P(evidence) for the fractional
                objective, keeping the ratio well-defined. 0/None disables it.

        Returns:
            Dict mapping variable name to (lower_bounds, upper_bounds) numpy
            arrays, each [P(=0), P(=1)]. Per-atom global-solver verdicts (if any)
            are in ``self.status``. ``self.degenerate`` is set True when the
            whole result is uninformative -- every non-evidence marginal is the
            vacuous [0,1] bound or every solve was infeasible/unsolved -- which
            almost always signals an inconsistent model or (with evidence)
            inconsistent evidence; a warning is also printed when verbosity > 0.
        """
        assert self.lcn is not None, "Make sure the LCN model exists."
        assert self.lcn.independencies is not None, "Make sure the LMC is applied."

        solver = solver.lower()
        if solver not in ("local", "global"):
            raise ValueError(f"unknown solver: {solver!r} (use 'local' or 'global')")
        if mode not in ("slow", "fast"):
            raise ValueError(f"unknown mode: {mode!r} (use 'slow' or 'fast')")

        # Lightning preset: clamp the time budget and (local) drop to fast ipopt.
        if lightning:
            time_limit = min(time_limit, LIGHTNING_TIME_LIMIT)
            if solver == "local":
                mode = "fast"
                if use_slsqp_fallback:  # only auto-disable the default-on case
                    use_slsqp_fallback = False
            elif gap_tol == 0.0:
                gap_tol = 0.1

        # Verbosity 2 streams the solver log; the bar would interleave with it.
        effective_pbar = progress_bar and verbosity != 2

        self.solver_used = solver
        independencies = self.lcn.independencies
        if verbosity > 0:
            print(f"[ExactInference] Computing all marginals (solver={solver})")
            print(f"[ExactInference] Evidence: {evidence}")
            print(f"[ExactInference] Local Markov Condition: "
                  f"{len(independencies.get_assertions())} independencies")

        if solver == "global":
            return self._run_global(evidence, debug, verbosity, time_limit,
                                    gap_tol, den_floor, effective_pbar)
        return self._run_local(evidence, debug, verbosity, mode,
                               use_slsqp_fallback, time_limit, effective_pbar)

    # ==================================================================
    # Query inference: bounds on an arbitrary propositional formula
    # ==================================================================
    def _parse_query(self, query: str) -> Formula:
        """
        Parse a query string into a Formula and validate that every atom it
        references is a variable of the LCN.

        Raises ValueError on a malformed formula (from the Formula parser) or
        when the query mentions an unknown atom.
        """
        if not isinstance(query, str) or len(query.strip()) == 0:
            raise ValueError("query must be a non-empty propositional formula string.")
        q_formula = Formula(label="query", formula=query)  # raises on malformed input
        unknown = [a for a in q_formula.atoms.values() if a not in self.lcn.atoms]
        if unknown:
            raise ValueError(
                f"query references unknown atom(s) {sorted(unknown)}; "
                f"the LCN's variables are {sorted(self.lcn.atoms.keys())}.")
        return q_formula

    def _query_determined(self, q_formula: Formula, evidence: dict):
        """
        Return the exact value of P(query | evidence) when it is fixed by logic
        alone, else None.

        The query indicator is evaluated over every interpretation; restricted to
        the rows consistent with the evidence, if it is uniformly 1 the
        conditional probability is exactly 1.0, if uniformly 0 it is 0.0
        (independent of the feasible distribution). When the restricted rows are a
        mix of 0/1 the value depends on the distribution and None is returned so
        the caller runs the optimizer. With no evidence the same test collapses to
        "is the query a tautology / contradiction over all interpretations".
        """
        vars_list = [k for k, _ in self.lcn.atoms.items()]
        items = list(itertools.product([0, 1], repeat=len(vars_list)))
        interpretations = [dict(zip(vars_list, t)) for t in items]
        A_q = eval_indicator(q_formula, interpretations)
        if len(evidence) > 0:
            Fe = make_conjunction(variables=list(evidence.keys()), literals=evidence)
            mask = eval_indicator(Fe, interpretations).astype(bool)
            restricted = A_q[mask]
        else:
            restricted = A_q
        if restricted.size == 0 or restricted.min() != restricted.max():
            return None
        return float(restricted[0])

    def run_query(
            self,
            query: str,
            evidence: dict = {},
            debug: bool = False,
            verbosity: int = 1,
            solver: str = "local",
            mode: str = "slow",
            use_slsqp_fallback: bool = True,
            time_limit: float = 3600.0,
            lightning: bool = False,
            gap_tol: float = 0.0,
            den_floor: float = 1e-6,
    ) -> Tuple[float, float]:
        """
        Compute exact posterior lower/upper bounds on a query formula.

        Without evidence this returns bounds on ``P(query)``; with evidence it
        returns bounds on ``P(query | evidence)``. ``query`` is a propositional
        logic formula string over the LCN's variables (same syntax as the .lcn
        sentences, e.g. ``"B and !C"`` or ``"A or B"``).

        The query indicator is a linear functional of the joint distribution, so
        this reuses the same constrained model and min/max solving machinery as
        ``run`` -- only the objective points at the query instead of a singleton
        atom. The two solver backends mirror ``run``:

          * ``"local"`` (default): ipopt + two-phase SLSQP fallback (the
            unconditional objective is linear, so the fallback applies; the
            evidence-conditioned objective is fractional and stays ipopt-only).
          * ``"global"``: SCIP spatial branch-and-bound (certified bounds / gap).

        Args:
            query: str
                The query formula (a propositional logic string).
            evidence: dict
                {variable: value} of observed variables. When non-empty the
                bounds are on the conditional probability P(query | evidence).
            debug, verbosity, solver, mode, use_slsqp_fallback, time_limit,
            lightning, gap_tol, den_floor:
                Same meaning as in ``run`` (see that method's docstring). There
                is no progress bar -- a single query is one min + one max solve.

        Returns:
            A ``(lower, upper)`` tuple of floats. The same values are stored on
            ``self.lower_bound`` / ``self.upper_bound``; ``self.feasible`` records
            whether both solves were feasible. For ``solver="global"``,
            ``self.status`` holds ``{'min': (...), 'max': (...)}`` SCIP verdicts
            (``None`` for the local backend). ``self.solver_used`` records the
            backend.
        """
        assert self.lcn is not None, "Make sure the LCN model exists."
        assert self.lcn.independencies is not None, "Make sure the LMC is applied."

        solver = solver.lower()
        if solver not in ("local", "global"):
            raise ValueError(f"unknown solver: {solver!r} (use 'local' or 'global')")
        if mode not in ("slow", "fast"):
            raise ValueError(f"unknown mode: {mode!r} (use 'slow' or 'fast')")

        q_formula = self._parse_query(query)

        # Lightning preset: same clamping as run().
        if lightning:
            time_limit = min(time_limit, LIGHTNING_TIME_LIMIT)
            if solver == "local":
                mode = "fast"
                if use_slsqp_fallback:
                    use_slsqp_fallback = False
            elif gap_tol == 0.0:
                gap_tol = 0.1

        self.solver_used = solver
        self.status = None
        independencies = self.lcn.independencies
        if verbosity > 0:
            kind = f"P({query} | {evidence})" if evidence else f"P({query})"
            print(f"[ExactInference] Query {kind} (solver={solver})")
            print(f"[ExactInference] Local Markov Condition: "
                  f"{len(independencies.get_assertions())} independencies")

        # Logically-determined short-circuit: if the query is constant over every
        # interpretation consistent with the evidence (always true / always false),
        # then P(query | evidence) is exactly 1.0 / 0.0 regardless of the feasible
        # distribution -- no solve needed. This also cleanly handles trivial cases
        # (e.g. the query atom is itself an evidence variable) that the
        # fractional NLP path can otherwise stall on.
        det = self._query_determined(q_formula, evidence)
        if det is not None:
            self.feasible = True
            self.lower_bound = self.upper_bound = det
            if verbosity > 0:
                print("[ExactInference] Query is logically determined by the "
                      "evidence (no solve needed).")
                self._print_query_report(evidence, 0.0)
            return det, det

        if solver == "global":
            return self._query_global(q_formula, evidence, debug, verbosity,
                                      time_limit, gap_tol, den_floor)
        return self._query_local(q_formula, evidence, debug, verbosity, mode,
                                 use_slsqp_fallback, time_limit)

    def _query_local(self, q_formula, evidence, debug, verbosity, mode,
                     use_slsqp_fallback, time_limit) -> Tuple[float, float]:
        """Local (ipopt + SLSQP) bounds on the query. See run_query."""
        t_start = time.time()
        independencies = self.lcn.independencies

        # Reuse the shared constrained model + SLSQP residual checks + indicators.
        model, _atom_indicators, evidence_indicator, interpretations, N, checks = \
            _build_base_model(self.lcn, independencies, evidence, verbosity)

        # The query indicator is just another linear functional over the joint.
        A_q = eval_indicator(q_formula, interpretations)

        # Objective: P(query) (linear) or P(query, e) / P(e) (fractional).
        if evidence_indicator is None:
            obj_expr = dot(A_q, model, model.ITEMS)
            obj_vec = A_q
        else:
            ev_expr = dot(evidence_indicator, model, model.ITEMS)
            AE = A_q * evidence_indicator  # element-wise numpy multiply
            obj_expr = dot(AE, model, model.ITEMS) / ev_expr
            obj_vec = None  # fractional -> SLSQP fallback not applicable

        ipopt_mode = "exact" if mode == "slow" else "fast"
        solver = _make_ipopt(debug=debug, mode=ipopt_mode)
        solver.options['max_cpu_time'] = float(time_limit)
        solver.options['max_wall_time'] = float(time_limit)  # ignored if unsupported
        tee = bool(debug) or (verbosity == 2)

        # Lazy feasible seeds for the SLSQP fallback (computed once, only if
        # ipopt proves unreliable on this instance).
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

        pyomo_logger = logging.getLogger('pyomo')
        prev_level = pyomo_logger.level
        if not debug:
            pyomo_logger.setLevel(logging.ERROR)
        try:
            lo_val, feasible_lo, _ = _robust_solve(
                model, obj_expr, 'min', solver, N, "query", debug,
                checks=checks, obj_vec=obj_vec, seeds_provider=_get_seeds,
                use_slsqp_fallback=use_slsqp_fallback, tee=tee)
            hi_val, feasible_hi, _ = _robust_solve(
                model, obj_expr, 'max', solver, N, "query", debug,
                checks=checks, obj_vec=obj_vec, seeds_provider=_get_seeds,
                use_slsqp_fallback=use_slsqp_fallback, tee=tee)
        finally:
            pyomo_logger.setLevel(prev_level)

        self.feasible = bool(feasible_lo and feasible_hi)
        lower = max(min(lo_val, 1.0), 0.0) if feasible_lo else 0.0
        upper = min(max(hi_val, 0.0), 1.0) if feasible_hi else 1.0
        self.lower_bound = lower
        self.upper_bound = upper

        if verbosity > 0:
            self._print_query_report(evidence, time.time() - t_start)
        return lower, upper

    def _query_global(self, q_formula, evidence, debug, verbosity, time_limit,
                      gap_tol, den_floor) -> Tuple[float, float]:
        """Global (SCIP) bounds on the query. See run_query."""
        t_start = time.time()
        solver = make_scip(time_limit=time_limit, gap_tol=gap_tol)

        if verbosity > 0:
            for indep in self.lcn.independencies.get_assertions():
                print(f"  {indep}")
            print(f"[ExactInference] Per-solve time limit: {time_limit:.0f}s, "
                  f"gap tolerance: {gap_tol}")

        # Interpretation table + query/evidence indicators (built once).
        vars_list = [k for k, _ in self.lcn.atoms.items()]
        items = list(itertools.product([0, 1], repeat=len(vars_list)))
        interpretations = [dict(zip(vars_list, t)) for t in items]
        A_q = eval_indicator(q_formula, interpretations)
        evidence_indicator = None
        if len(evidence) > 0:
            Fe = make_conjunction(variables=list(evidence.keys()), literals=evidence)
            evidence_indicator = eval_indicator(Fe, interpretations)

        pyomo_logger = logging.getLogger('pyomo')
        prev_level = pyomo_logger.level
        if not debug:
            pyomo_logger.setLevel(logging.ERROR)
        tee = (verbosity == 2)
        try:
            lo = self._scip_solve(interpretations, A_q, evidence_indicator,
                                  'min', solver, gap_tol, den_floor, tee=tee)
            hi = self._scip_solve(interpretations, A_q, evidence_indicator,
                                  'max', solver, gap_tol, den_floor, tee=tee)
        finally:
            pyomo_logger.setLevel(prev_level)

        self.status = {'min': lo, 'max': hi}
        lo_val, _, lo_status, _ = lo
        hi_val, _, hi_status, _ = hi
        self.feasible = (lo_status != "unsolved" and hi_status != "unsolved")
        lower = min(max(lo_val, 0.0), 1.0) if lo_val is not None else 0.0
        upper = min(max(hi_val, 0.0), 1.0) if hi_val is not None else 1.0
        self.lower_bound = lower
        self.upper_bound = upper

        if verbosity > 0:
            (lv, lg, ls, lt) = lo
            (hv, hg, hs, ht) = hi
            print("[ExactInference] Query bound (value | gap | status):")
            print(f"  min={_fmt(lv, lg, ls, lt)}")
            print(f"  max={_fmt(hv, hg, hs, ht)}")
            self._print_query_report(evidence, time.time() - t_start)
        return lower, upper

    def _print_query_report(self, evidence, elapsed):
        """Common verbose footer for run_query (both backends)."""
        print(f"[ExactInference] Result: "
              f"[{self.lower_bound:.6f}, {self.upper_bound:.6f}]")
        print(f"[ExactInference] Feasible: {self.feasible}")
        if _is_vacuous(self.lower_bound, self.upper_bound):
            print("[ExactInference] WARNING: the query bound is vacuous [0,1] -- "
                  "uninformative. The model (or, with evidence, the evidence) may "
                  "be inconsistent; run check_consistency on the model.")
        print(f"[ExactInference] Time elapsed: {elapsed:.4f} sec")

    # ------------------------------------------------------------------
    # Local backend: ipopt + SLSQP fallback (one reused model)
    # ------------------------------------------------------------------
    def _run_local(self, evidence, debug, verbosity, mode, use_slsqp_fallback,
                   time_limit, effective_pbar):
        t_start = time.time()
        independencies = self.lcn.independencies
        evidence_set = set(evidence.keys())

        # Build the base model once (constraints only, no objective).
        model, atom_indicators, evidence_indicator, interpretations, N, checks = \
            _build_base_model(self.lcn, independencies, evidence, verbosity)

        # Pre-compute P(evidence) expression (reused across all atoms).
        ev_expr = None
        if evidence_indicator is not None:
            ev_expr = dot(evidence_indicator, model, model.ITEMS)

        # Shared ipopt solver. Translate the public mode ("slow"/"fast") to the
        # shared make_ipopt vocabulary ("exact"/"fast"), then bound its runtime by
        # the caller's time_limit (ipopt CPU time, plus a best-effort wall cap).
        ipopt_mode = "exact" if mode == "slow" else "fast"
        solver = _make_ipopt(debug=debug, mode=ipopt_mode)
        solver.options['max_cpu_time'] = float(time_limit)
        solver.options['max_wall_time'] = float(time_limit)  # ignored if unsupported

        # Stream ipopt's log at verbosity 2 (or under debug).
        tee = bool(debug) or (verbosity == 2)

        self.marginals = {}
        self.feasible = True
        self.status = None  # local solver has no per-atom gap/status concept
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
        n_vacuous = 0
        pbar = tqdm(total=len(solve_atoms), desc="[ExactInference] atoms",
                    disable=(not effective_pbar))
        try:
            for atom_name in atom_names:
                if atom_name in evidence_set:
                    self.marginals[atom_name] = _point_mass(evidence[atom_name])
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
                    checks=checks, obj_vec=obj_vec, seeds_provider=_get_seeds,
                    use_slsqp_fallback=use_slsqp_fallback, tee=tee)
                hi_val, feasible_hi, fb_hi = _robust_solve(
                    model, obj_expr, 'max', solver, N, atom_name, debug,
                    checks=checks, obj_vec=obj_vec, seeds_provider=_get_seeds,
                    use_slsqp_fallback=use_slsqp_fallback, tee=tee)

                if not feasible_lo or not feasible_hi:
                    self.feasible = False
                    n_infeasible += 1
                if fb_lo or fb_hi:
                    n_fallback += 1

                lo_1 = max(abs(lo_val), 0.0) if feasible_lo else 0.0
                hi_1 = min(abs(hi_val), 1.0) if feasible_hi else 1.0
                if _is_vacuous(lo_1, hi_1):
                    n_vacuous += 1

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

        # Degenerate result: every non-evidence atom came back vacuous [0,1], or
        # every solve was infeasible. With evidence this almost always means the
        # evidence is inconsistent with the LCN; without it, the LCN itself is.
        n_solved = len(solve_atoms)
        self.degenerate = n_solved > 0 and (
            n_vacuous == n_solved or n_infeasible == n_solved)

        if verbosity > 0:
            print("[ExactInference] Singleton variable marginals:")
            for atom_name in sorted(self.marginals):
                lo, hi = self.marginals[atom_name]
                for val in range(len(lo)):
                    print(f"  P({atom_name}={val}): "
                          f"[{lo[val]:.6f}, {hi[val]:.6f}]")
            print(f"[ExactInference] Feasible: {self.feasible}")
            print(f"[ExactInference] Atoms solved: {n_solved} | "
                  f"SLSQP fallback used: {n_fallback} | "
                  f"infeasible: {n_infeasible} | vacuous: {n_vacuous}")
            if self.degenerate:
                reason = ("all marginals are vacuous [0,1]"
                          if n_infeasible < n_solved
                          else "all solves were infeasible")
                print(_degenerate_warning(evidence, reason))
            print(f"[ExactInference] Time elapsed: {t_end - t_start:.4f} sec")

        return self.marginals

    # ------------------------------------------------------------------
    # Global backend: SCIP (fresh model per solve, certified bounds)
    # ------------------------------------------------------------------
    def _build_scip_model(self, interpretations):
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

    def _scip_solve(self, interpretations, A, evidence_indicator, sense, solver,
                    gap_tol, den_floor, tee=False):
        """
        Globally optimize the marginal of one atom with SCIP.

        Linear objective when ``evidence_indicator`` is None (P(atom=1) = A@p);
        otherwise the fractional objective P(atom=1, e)/P(e) = (A*E)@p / (E@p),
        with an optional denominator floor P(e) >= den_floor.

        Returns ``(value_or_None, gap_or_None, status, seconds)`` where status is
        ``"confirmed"`` (proven global optimum), ``"unconfirmed"`` (best feasible
        bound found, gap > tol / time limit hit) or ``"unsolved"`` (no incumbent).
        Never raises on a timeout.
        """
        model = self._build_scip_model(interpretations)

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

    def _run_global(self, evidence, debug, verbosity, time_limit, gap_tol,
                    den_floor, effective_pbar):
        t_start = time.time()
        evidence_set = set(evidence.keys())
        solver = make_scip(time_limit=time_limit, gap_tol=gap_tol)

        if verbosity > 0:
            for indep in self.lcn.independencies.get_assertions():
                print(f"  {indep}")
            print(f"[ExactInference] Per-solve time limit: {time_limit:.0f}s, "
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

        # Suppress Pyomo's routine warning-status spam (expected on time-outs --
        # we report gap/status ourselves). Restore afterwards.
        pyomo_logger = logging.getLogger('pyomo')
        prev_level = pyomo_logger.level
        if not debug:
            pyomo_logger.setLevel(logging.ERROR)

        # Stream SCIP's search log at verbosity 2.
        tee = (verbosity == 2)

        n_confirmed = n_unconfirmed = n_unsolved = 0
        n_vacuous = 0
        n_atoms_unsolved = 0
        pbar = tqdm(total=len(solve_atoms), desc="[ExactInference] atoms",
                    disable=(not effective_pbar))
        try:
            for atom_name in vars_list:
                if atom_name in evidence_set:
                    self.marginals[atom_name] = _point_mass(evidence[atom_name])
                    continue

                A = atom_indicators[atom_name]
                if tee:
                    print(f"\n[ExactInference] === SCIP search: "
                          f"minimize P({atom_name}=1) ===")
                lo = self._scip_solve(interpretations, A, evidence_indicator,
                                      'min', solver, gap_tol, den_floor, tee=tee)
                if tee:
                    print(f"\n[ExactInference] === SCIP search: "
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
                    n_atoms_unsolved += 1
                if _is_vacuous(lo_1, hi_1):
                    n_vacuous += 1

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

        # Degenerate result: every non-evidence atom is vacuous [0,1], or every
        # atom had at least one unsolved side. With evidence this almost always
        # means the evidence is inconsistent with the LCN; without it, the LCN.
        n_solved = len(solve_atoms)
        self.degenerate = n_solved > 0 and (
            n_vacuous == n_solved or n_atoms_unsolved == n_solved)

        if verbosity > 0:
            self._print_global_report(evidence, t_end - t_start, n_confirmed,
                                      n_unconfirmed, n_unsolved,
                                      n_vacuous, n_atoms_unsolved, n_solved)

        return self.marginals

    def _print_global_report(self, evidence, elapsed, n_conf, n_unconf,
                             n_unsolved, n_vacuous, n_atoms_unsolved, n_solved):
        """Print the per-atom bound/gap/status table and a summary line."""
        print("[ExactInference] Singleton variable marginals "
              "(P=1 bound | gap | status):")
        for atom_name in sorted(self.status):
            (lo_v, lo_g, lo_s, lo_t) = self.status[atom_name]['min']
            (hi_v, hi_g, hi_s, hi_t) = self.status[atom_name]['max']
            print(f"  {atom_name}: "
                  f"min={_fmt(lo_v, lo_g, lo_s, lo_t)}  "
                  f"max={_fmt(hi_v, hi_g, hi_s, hi_t)}")
        print(f"[ExactInference] Feasible: {self.feasible}")
        print(f"[ExactInference] Solves: confirmed optimal={n_conf} | "
              f"unconfirmed (best-so-far)={n_unconf} | unsolved={n_unsolved}")
        if n_unconf or n_unsolved:
            print("[ExactInference] Note: unconfirmed/unsolved bounds are "
                  "NOT proven optimal -- raise time_limit to certify.")
        if self.degenerate:
            reason = ("all marginals are vacuous [0,1]"
                      if n_atoms_unsolved < n_solved
                      else "all solves were unsolved (no incumbent)")
            print(_degenerate_warning(evidence, reason))
        print(f"[ExactInference] Time elapsed: {elapsed:.4f} sec")


class ExactInferenceSCIP(ExactInference):
    """
    Back-compat shim: ``ExactInferenceSCIP(lcn).run(...)`` is equivalent to
    ``ExactInference(lcn).run(solver="global", ...)``. The SCIP global solver now
    lives in ``ExactInference``; this subclass preserves the old global-by-default
    entry point for any code still referencing the former exact_scip.py class.
    """

    def run(self, *args, **kwargs):
        kwargs.setdefault("solver", "global")
        return super().run(*args, **kwargs)


if __name__ == "__main__":

    def print_singleton_marginals(results):
        """Print only singleton variable marginals from the results."""
        print("  Singleton variable marginals:")
        for var in sorted(results):
            lo, hi = results[var]
            for val in range(len(lo)):
                print(f"    P({var}={val}): [{lo[val]:.6f}, {hi[val]:.6f}]")

    # Load the LCN
    file_name = "examples/d4_biting.lcn"
    lcn_model = LCN()
    lcn_model.from_lcn(file_name=file_name)
    lcn_model.summary()
    lcn_model.show_graphs()
    print(lcn_model)

    # Check consistency
    print(f"\n=== Consistency check for {file_name} ===")
    ok = check_consistency(lcn_model)

    # Run exact marginal inference with the LOCAL solver (ipopt + SLSQP), no evidence.
    print("\n=== ExactInference (local, no evidence) ===")
    algo = ExactInference(lcn=lcn_model)
    results = algo.run(evidence={}, debug=False, verbosity=1, solver="local", mode="slow")
    print_singleton_marginals(results)

    # Run exact marginal inference with the GLOBAL solver (SCIP), no evidence. A
    # short per-solve time limit + loose gap tolerance keep the demo responsive
    # (some alarm atoms are hard for SCIP and would otherwise run to the limit);
    # the library default is a 3600s (1h) limit and gap_tol=0 (prove optimality).
    print("\n=== ExactInference (global / SCIP, no evidence) ===")
    algo2 = ExactInference(lcn=lcn_model)
    results = algo2.run(evidence={}, debug=False, verbosity=2, solver="global",
                        time_limit=3600, gap_tol=0.0, progress_bar=True)
    print_singleton_marginals(results)

    # Query inference: bounds on an arbitrary propositional formula (local solver).
    # print("\n=== ExactInference.run_query (local, no evidence) ===")
    # algo3 = ExactInference(lcn=lcn_model)
    # lb, ub = algo3.run_query("B and !C", verbosity=1, solver="local")
    # print(f"P(B and !C) in [{lb:.6f}, {ub:.6f}]")

    # Conditional query: bounds on P(query | evidence).
    # print("\n=== ExactInference.run_query (local, evidence B=1) ===")
    # algo4 = ExactInference(lcn=lcn_model)
    # lb, ub = algo4.run_query("D", evidence={"B": 1}, verbosity=1, solver="local")
    # print(f"P(D | B=1) in [{lb:.6f}, {ub:.6f}]")

    # Run exact marginal inference (with evidence)
    # print("\n=== ExactInference (B=0, E=0) ===")
    # algo5 = ExactInference(lcn=lcn_model)
    # results = algo5.run(evidence={"B": 0, "E": 0}, debug=False)
    # print_singleton_marginals(results)
