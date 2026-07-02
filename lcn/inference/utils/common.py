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
import logging
import os
import numpy as np
from pyomo.environ import (
    ConcreteModel,
    ConstraintList,
    NonNegativeReals,
    Objective,
    Set,
    SolverFactory,
    Var,
    maximize,
    value,
)
from typing import List, Dict

# Local
from lcn.core.model import LCN, Formula, SentenceType


# Subprocess isolation for the SLSQP fallback --------------------------------
#
# On some platforms (observed: RHEL 9, glibc with hardened heap checks) the
# scipy SLSQP path in the ExactInference *local* solver can trip a native
# "double free or corruption" abort that raises SIGABRT. A Python try/except
# cannot catch a SIGABRT -- the whole process dies -- so a single bad solve
# would take down an entire run even though ExactInference-local is only a
# best-effort, re-verified backend (see docs/properties.tex).
#
# The guard below runs each SLSQP-driving helper in a short-lived child
# process. If the child aborts natively, the parent observes a non-zero /
# signal exitcode and degrades gracefully (returns the helper's documented
# "no result" value) instead of crashing. A clean child returns its result
# over a pipe.
#
# We use the "fork" start method on POSIX so the child inherits the parent's
# memory: the `checks` argument is a list of closures (capturing numpy arrays)
# that is NOT picklable, and fork avoids having to serialize it. Only the
# RETURN value must be picklable (numpy arrays / floats / bools -- all are).
#
# Controlled by the LCN_ISOLATE_SLSQP environment variable:
#   unset / "auto" (default): isolate on Linux (where the crash occurs), and
#                             only when a forking start method is available;
#   "1"/"true"/"yes"        : force isolation on;
#   "0"/"false"/"no"        : force isolation off (run in-process).
import multiprocessing as _mp
import sys as _sys


def _isolation_enabled() -> bool:
    """Whether to run SLSQP helpers in an isolated child process."""
    flag = os.environ.get("LCN_ISOLATE_SLSQP", "auto").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    if flag in ("0", "false", "no", "off"):
        return False
    # auto: fork-based isolation is only meaningful/cheap on POSIX, and the
    # crash it guards against is Linux-specific.
    return _sys.platform.startswith("linux") and \
        "fork" in _mp.get_all_start_methods()


def _isolated_target(_fn, _q, *args, **kwargs):
    """Child entry point: run _fn and ship its result back over the queue."""
    try:
        _q.put(("ok", _fn(*args, **kwargs)))
    except BaseException as e:  # pragma: no cover - surfaced to parent below
        _q.put(("err", f"{type(e).__name__}: {e}"))


def run_isolated(fn, args=(), kwargs=None, on_crash=None, verbosity: int = 0,
                 timeout: float = None):
    """
    Run ``fn(*args, **kwargs)`` in a forked child process, returning its result.

    If isolation is disabled (see :func:`_isolation_enabled`) the function is
    called directly in-process. If the child dies from a native abort (SIGABRT,
    SIGSEGV, ...) -- which no Python ``try/except`` could catch -- the parent
    detects the abnormal exit and returns ``on_crash`` (a caller-supplied
    "no result" sentinel) instead of propagating the crash.

    Args:
        fn: callable to run (inherited via fork; need not be picklable).
        args, kwargs: arguments for ``fn`` (inherited via fork; the RESULT must
            be picklable, which every SLSQP helper's return value is).
        on_crash: value returned if the child crashes, errors, or times out.
        verbosity: if > 0, print a diagnostic when the child does not return
            a clean result.
        timeout: optional wall-clock seconds after which a still-running child
            is terminated and ``on_crash`` returned (guards a hung SLSQP).

    Returns:
        ``fn``'s return value, or ``on_crash`` on native crash / error / timeout.
    """
    if kwargs is None:
        kwargs = {}
    if not _isolation_enabled():
        return fn(*args, **kwargs)

    ctx = _mp.get_context("fork")
    q = ctx.Queue()
    proc = ctx.Process(target=_isolated_target,
                       args=(fn, q, *args), kwargs=kwargs)
    proc.start()

    # Poll for a result while watching liveness. We must NOT do a blocking
    # q.get() (a natively-aborting child never writes, so it would deadlock)
    # nor join() before draining (a large numpy result can fill the pipe and
    # deadlock the child on flush). So: try a short non-blocking read each
    # tick; stop once we have a payload, or the child has exited and the queue
    # is drained, or the optional timeout elapses.
    import queue as _queue
    import time as _time
    payload = None
    deadline = None if timeout is None else _time.monotonic() + timeout
    while True:
        try:
            payload = q.get(timeout=0.05)
            break
        except _queue.Empty:
            pass
        if not proc.is_alive():
            # Child exited; make one last non-blocking drain in case the result
            # landed between our get() and this check.
            try:
                payload = q.get_nowait()
            except _queue.Empty:
                payload = None
            break
        if deadline is not None and _time.monotonic() > deadline:
            break

    if proc.is_alive():
        proc.terminate()
    proc.join()

    result = on_crash
    if payload is not None and payload[0] == "ok":
        result = payload[1]
    elif verbosity > 0:
        if payload is not None and payload[0] == "err":
            print(f"[run_isolated] SLSQP child raised {payload[1]}; "
                  f"degrading to fallback result.")
        else:
            ec = proc.exitcode
            sig = -ec if (ec is not None and ec < 0) else None
            reason = (f"signal {sig}" if sig
                      else "timeout" if (deadline is not None) else
                      f"exit code {ec}")
            print(f"[run_isolated] SLSQP child terminated abnormally "
                  f"({reason}); degrading to fallback result.")
    return result


# ipopt configuration shared across all LCN inference algorithms ------------
_TOL = 1e-8                 # primary ipopt convergence tolerance
_ACCEPTABLE_TOL = 1e-8      # tolerance for an "acceptable" termination
_MAX_ITER = 100000
_MAX_CPU_TIME = 3600

# "fast" feasibility mode: loosen the *acceptable* termination criteria and stop
# at the first acceptable iterate, so ipopt returns a feasible-ish point quickly
# instead of driving to full optimality. ipopt has no pure-feasibility switch;
# the acceptable_* family + acceptable_iter is the documented early-stop lever
# (see https://coin-or.github.io/Ipopt/OPTIONS.html).
_FAST_ACCEPTABLE_TOL = 1e-4
_FAST_ACCEPTABLE_CONSTR_VIOL_TOL = 1e-4
_FAST_MAX_ITER = 500
_FAST_MAX_CPU_TIME = 60


def make_ipopt(debug: bool = False, mode: str = "exact"):
    """
    Create an ipopt solver instance configured for the (nonconvex) LCN NLPs.

    The LCN inference NLPs are nonconvex (conditional-probability and quadratic
    Markov constraints), so ipopt is a *local* solver here. These options keep
    it numerically well-behaved and make it respect variable box bounds exactly
    (``bound_relax_factor = 0``). This is the single shared configuration used
    by every LCN inference algorithm that relies on ipopt.

    Two modes:
      - ``"exact"`` (default): drive to high accuracy (tight tol / acceptable_tol,
        adaptive barrier, exact Hessian, generous budgets). Use this whenever a
        reported probability bound is computed.
      - ``"fast"``: loosen the *acceptable* termination thresholds and set
        ``acceptable_iter = 1`` so ipopt stops at the first acceptable point, plus
        a cheaper per-iteration setup (monotone barrier, limited-memory Hessian)
        and smaller budgets. Use this only where a quick feasible-ish point is
        enough (e.g. a constant-objective feasibility solve whose result is
        re-verified independently). Note: ``linear_solver`` is left at the build
        default (MUMPS here) -- it is not a useful speed lever on this install.

    Args:
        debug: bool
            If True, raise the ipopt print level for diagnostics.
        mode: str
            ``"exact"`` (default) or ``"fast"``.

    Returns:
        A configured Pyomo ``SolverFactory('ipopt')`` instance.
    """
    s = SolverFactory('ipopt')
    # Load-bearing in BOTH modes: honour variable box bounds exactly.
    s.options['bound_relax_factor'] = 0.0
    s.options['print_level'] = 5 if debug else 0

    if mode == "exact":
        s.options['tol'] = _TOL
        s.options['acceptable_tol'] = _ACCEPTABLE_TOL
        s.options['max_iter'] = _MAX_ITER
        s.options['max_cpu_time'] = _MAX_CPU_TIME
        s.options['mu_strategy'] = 'adaptive'
    elif mode == "fast":
        s.options['tol'] = _TOL  # keep tight primary target; stop early via acceptable_*
        s.options['acceptable_tol'] = _FAST_ACCEPTABLE_TOL
        s.options['acceptable_constr_viol_tol'] = _FAST_ACCEPTABLE_CONSTR_VIOL_TOL
        s.options['acceptable_iter'] = 1
        s.options['max_iter'] = _FAST_MAX_ITER
        s.options['max_cpu_time'] = _FAST_MAX_CPU_TIME
        s.options['mu_strategy'] = 'monotone'
        s.options['hessian_approximation'] = 'limited-memory'
    else:
        raise ValueError(f"unknown ipopt mode: {mode!r}")
    return s


def make_init_config(vars: List):
    return [1 if np.random.random() > 0.5 else 0 for _ in vars]

def select_neighbor(elements: List):
    sel = np.random.randint(0, len(elements))
    return elements[sel]

def find_neighbors(config: List) -> List[List]:
    neighbors = []
    for pos in range(len(config)):
        neighbor = []
        for i, val in enumerate(config):
            if i != pos: # leave value unchanged
                neighbor.append(val)
            else: # flip the value at position pos
                neighbor.append(1 if val == 0 else 0)
        neighbors.append(neighbor)
    return neighbors

def make_conjunction(variables: List[str], literals: Dict[str, int]) -> Formula:
    """
    Returns the conjunction of the input literals
    """
    
    assert len(variables) > 0, "Variables list cannot be empty."
    assert len(literals) > 0, "Literals dict cannot be empty."

    lits = []
    for x in variables:
        if literals[x] == 1:
            lits.append(x)
        else:
            lits.append(f"!{x}")
    conjunction_str = " and ".join(lits)
    return Formula(label="conjunction", formula=conjunction_str)


def lmc_constraint_groups(indep, eval_fn) -> List[tuple]:
    """
    Build the indicator-vector groups encoding a single Local Markov Condition
    independence assertion ``(X |= Y | S)`` as quadratic equality constraints.

    For binary atoms, conditional independence ``X |= Y | S`` is equivalent to
    the joint factorization (see docs/scc_factor_graph.tex)

        P(x, y, z) * P(z) = P(x, z) * P(y, z)

    holding for the single ``x = 1`` slice of the (singleton) variable ``X`` but
    for **every** joint configuration of the ``Y`` block and the ``S`` block.
    The ``x = 1`` slice alone is sufficient because, with the (y, s) cell fixed,
    the binary variable ``X`` has only one degree of freedom (the complementary
    ``x = 0`` equality follows from the fixed marginals). Iterating all ``Y``
    configurations is required: encoding ``Y`` element-by-element (pairwise
    ``X |= t | S``) is strictly weaker than independence of the joint ``Y`` when
    ``|Y| >= 2`` and would under-constrain the model.

    Args:
        indep: IndependenceAssertion
            An assertion with ``event1`` (X, a single atom), ``event2`` (Y, a
            set of atoms) and ``event3`` (S, the conditioning set).
        eval_fn: callable
            Maps a Formula to its 0/1 indicator vector over the interpretations
            (e.g. a closure over the fixed interpretation table). Passed in so
            this helper has no dependency on a particular inference module.

    Returns:
        A list of tuples. Each is either
        ``('conditional', Aa, Ab, Ac, Ad)`` meaning ``P(Aa)*P(Ab) == P(Ac)*P(Ad)``
        (i.e. ``P(x,y,z)P(z) == P(x,z)P(y,z)``), or
        ``('marginal', Aa, Ab, Ac)`` meaning ``P(Aa) == P(Ab)*P(Ac)``
        (i.e. ``P(x,y) == P(x)P(y)``) when ``S`` is empty.
    """
    X = list(indep.event1)
    Y = list(indep.event2)
    S = list(indep.event3)
    x = X[0]

    groups = []
    configs_Y = list(itertools.product([0, 1], repeat=len(Y)))
    if len(S) > 0:
        configs_S = list(itertools.product([0, 1], repeat=len(S)))
        for y_cfg in configs_Y:
            for s_cfg in configs_S:
                literals = {x: 1}
                literals.update(dict(zip(Y, y_cfg)))
                literals.update(dict(zip(S, s_cfg)))
                Fa = make_conjunction(variables=X + Y + S, literals=literals)
                Fb = make_conjunction(variables=S, literals=literals)
                Fc = make_conjunction(variables=X + S, literals=literals)
                Fd = make_conjunction(variables=Y + S, literals=literals)
                groups.append(('conditional', eval_fn(Fa), eval_fn(Fb),
                               eval_fn(Fc), eval_fn(Fd)))
    else:
        for y_cfg in configs_Y:
            literals = {x: 1}
            literals.update(dict(zip(Y, y_cfg)))
            Fa = make_conjunction(variables=X + Y, literals=literals)
            Fb = make_conjunction(variables=X, literals=literals)
            Fc = make_conjunction(variables=Y, literals=literals)
            groups.append(('marginal', eval_fn(Fa), eval_fn(Fb), eval_fn(Fc)))

    return groups


def eval_indicator(formula: Formula, interpretations: list) -> np.ndarray:
    """Evaluate formula on all interpretations, return a binary numpy vector.

    Shared utility used by every Pyomo-based inference engine (exact marginal,
    ariel, sccp, factorization, and the MAP scripts) to turn a Formula into its
    0/1 indicator over the joint interpretation table.
    """
    return np.array([1.0 if formula.evaluate(table=interp) else 0.0
                     for interp in interpretations])


def dot(vec: np.ndarray, model, items):
    """Build the Pyomo linear expression ``vec @ model.p`` over ``items``."""
    return sum(float(vec[i]) * model.p[i] for i in items)


def build_truth_table(num_vars: int) -> np.ndarray:
    """
    Build the truth table of all 2^num_vars interpretations.

    Returns a (2^num_vars, num_vars) int8 array whose rows enumerate
    ``itertools.product([0, 1], repeat=num_vars)`` in order (so row j matches
    ``index[j]`` everywhere the inference code builds interpretations, and
    column i is the i-th variable). This is the vectorized counterpart of the
    per-interpretation tables used with ``Formula.evaluate``.
    """
    return np.array(list(itertools.product([0, 1], repeat=num_vars)),
                    dtype=np.int8)


def conjunction_indicator(table: np.ndarray, col_of: Dict[str, int],
                          literals: Dict[str, int]) -> np.ndarray:
    """
    Indicator vector (float 0/1) of the conjunction of the given literals over
    the precomputed truth ``table``.

    Equivalent to ``_eval_indicator(make_conjunction(list(literals), literals))``
    but computed with numpy column masks instead of parsing/evaluating a
    Formula per interpretation. ``literals`` maps variable name -> 0/1; the
    indicator is 1 on rows where every listed variable equals its literal value.
    """
    mask = np.ones(table.shape[0], dtype=bool)
    for v, val in literals.items():
        mask &= (table[:, col_of[v]] == val)
    return mask.astype(float)


def lmc_constraint_groups_vec(indep, table: np.ndarray,
                              col_of: Dict[str, int]) -> List[tuple]:
    """
    Vectorized form of :func:`lmc_constraint_groups`: identical group math and
    tuple shapes, but each conjunction indicator is built with
    ``conjunction_indicator`` over the precomputed ``table`` (no Formula
    objects). Produces bit-identical indicator vectors to the Formula-based
    path, so it is a drop-in for the dense LMC encoding while avoiding the
    ~2^|Y| * 2^n Formula.evaluate calls that dominate the n=10 cost.

    Args:
        indep: IndependenceAssertion with event1 (X, singleton), event2 (Y),
            event3 (S).
        table: truth table from ``build_truth_table``.
        col_of: mapping variable name -> column index in ``table``.

    Returns:
        Same as ``lmc_constraint_groups``: list of ``('conditional', Aa, Ab,
        Ac, Ad)`` and/or ``('marginal', Aa, Ab, Ac)`` tuples.
    """
    X = list(indep.event1)
    Y = list(indep.event2)
    S = list(indep.event3)
    x = X[0]

    groups = []
    configs_Y = list(itertools.product([0, 1], repeat=len(Y)))
    if len(S) > 0:
        configs_S = list(itertools.product([0, 1], repeat=len(S)))
        for y_cfg in configs_Y:
            for s_cfg in configs_S:
                lit = {x: 1}
                lit.update(dict(zip(Y, y_cfg)))
                lit.update(dict(zip(S, s_cfg)))
                # Fa = conj(X+Y+S), Fb = conj(S), Fc = conj(X+S), Fd = conj(Y+S)
                Aa = conjunction_indicator(table, col_of, lit)
                Ab = conjunction_indicator(table, col_of,
                                           {v: lit[v] for v in S})
                Ac = conjunction_indicator(table, col_of,
                                           {v: lit[v] for v in X + S})
                Ad = conjunction_indicator(table, col_of,
                                           {v: lit[v] for v in Y + S})
                groups.append(('conditional', Aa, Ab, Ac, Ad))
    else:
        for y_cfg in configs_Y:
            lit = {x: 1}
            lit.update(dict(zip(Y, y_cfg)))
            Aa = conjunction_indicator(table, col_of, lit)            # conj(X+Y)
            Ab = conjunction_indicator(table, col_of, {x: 1})         # conj(X)
            Ad = {v: lit[v] for v in Y}
            Ac = conjunction_indicator(table, col_of, Ad)            # conj(Y)
            groups.append(('marginal', Aa, Ab, Ac))

    return groups


def _checks_feasible(p, checks, tol):
    """True iff solution vector ``p`` satisfies every constraint in ``checks``."""
    for kind, fn in checks:
        g = fn(p)
        if kind == "eq":
            if abs(g) > tol:
                return False
        elif g < -tol:
            return False
    return True


def find_feasible_points(N, checks, n_points=12, restarts=200, seed=0,
                         tol=1e-6, spread=True, spread_per_point=4,
                         spread_sigma=0.4):
    """
    Isolated wrapper around :func:`_find_feasible_points_impl` -- see
    :func:`run_isolated`. Runs the SLSQP feasibility search in a child process
    so a native abort degrades to an empty result (no feasible points) instead
    of crashing the run. Returns ``[]`` on child crash.
    """
    return run_isolated(
        _find_feasible_points_impl,
        args=(N, checks, n_points, restarts, seed, tol, spread,
              spread_per_point, spread_sigma),
        on_crash=[])


def _find_feasible_points_impl(N, checks, n_points=12, restarts=200, seed=0,
                               tol=1e-6, spread=True, spread_per_point=4,
                               spread_sigma=0.4):
    """
    Find feasible distributions over the 2^N world-probability simplex that
    satisfy a list of constraint residuals, via an SLSQP *feasibility* search
    (minimize the total squared constraint violation from many restarts).

    ipopt (interior-point) is unreliable at *finding feasibility* in the dense,
    nonconvex joint-LMC equality system; SLSQP handles it well. The points
    returned here serve both as a consistency certificate and as warm-start
    seeds for the bound-optimization phase (``optimize_marginal_slsqp``).

    On a nonconvex bilinear-LMC manifold the plain violation-minimizing search
    tends to converge to ONE central basin: every collected point can be nearly
    identical (e.g. all with ``P(atom)=0.5``). A clustered seed set caps the
    downstream local bound optimization. When ``spread`` is True (default) the
    collected points are therefore diversified by a perturb-and-reproject pass:
    each point is jittered with Gaussian noise and re-projected onto the feasible
    manifold, yielding feasible seeds spread across the region. This widens the
    bounds the local SLSQP can reach for atoms whose extreme is interior; far
    corner extremes are additionally covered by the objective-biased seeds in
    ``optimize_marginal_slsqp`` / ``find_biased_feasible_points``.

    Args:
        N: int
            Number of world-probability variables.
        checks: list of (kind, callable(p)->float)
            Constraint residuals; ``'eq'`` must be ~0, ``'ineq'`` must be >= 0.
            (The simplex sum-to-one is added internally.)
        n_points: int
            Stop the initial search once this many feasible points are collected.
        restarts: int
            Maximum number of SLSQP restarts (the first is the uniform point).
        seed: int
            RNG seed for reproducible restarts.
        tol: float
            Feasibility tolerance for accepting a point.
        spread: bool
            When True, append perturb-and-reproject diversified seeds.
        spread_per_point: int
            Number of jittered re-projections attempted per collected point.
        spread_sigma: float
            Std-dev of the Gaussian jitter applied before re-projection.

    Returns:
        A list of feasible solution vectors (numpy arrays); empty if none found.
    """
    from scipy.optimize import minimize as _sp_minimize

    def _sq_violation(p):
        tot = 0.0
        for kind, fn in checks:
            g = fn(p)
            tot += g * g if kind == "eq" else max(0.0, -g) ** 2
        return tot

    simplex = [{"type": "eq", "fun": lambda p: float(p.sum()) - 1.0}]
    rng = np.random.default_rng(seed)
    points = []
    for k in range(restarts):
        x0 = (np.full(N, 1.0 / N) if k == 0 else rng.random(N))
        x0 = x0 / x0.sum()
        res = _sp_minimize(_sq_violation, x0, method="SLSQP",
                           bounds=[(0.0, 1.0)] * N, constraints=simplex,
                           options={"maxiter": 800, "ftol": 1e-16})
        p = np.asarray(res.x, dtype=float)
        if _checks_feasible(p, checks, tol):
            points.append(p)
            if len(points) >= n_points:
                break

    # Diversify: jitter each feasible point and re-project to feasibility, so the
    # seed set spreads across the manifold instead of clustering in one basin.
    if spread and points:
        spread_pts = []
        for base in points:
            for _ in range(spread_per_point):
                x0 = np.abs(base + rng.normal(0.0, spread_sigma, N))
                s = x0.sum()
                x0 = x0 / s if s > 0 else np.full(N, 1.0 / N)
                res = _sp_minimize(_sq_violation, x0, method="SLSQP",
                                   bounds=[(0.0, 1.0)] * N, constraints=simplex,
                                   options={"maxiter": 800, "ftol": 1e-16})
                q = np.asarray(res.x, dtype=float)
                if _checks_feasible(q, checks, tol):
                    spread_pts.append(q)
        points.extend(spread_pts)
    return points


def find_biased_feasible_points(N, checks, obj_vec, sense, n_points=1,
                                restarts=2, seed=0, tol=1e-6, bias=0.5):
    """
    Isolated wrapper around :func:`_find_biased_feasible_points_impl` -- see
    :func:`run_isolated`. Returns ``[]`` on child crash.
    """
    return run_isolated(
        _find_biased_feasible_points_impl,
        args=(N, checks, obj_vec, sense, n_points, restarts, seed, tol, bias),
        on_crash=[])


def _find_biased_feasible_points_impl(N, checks, obj_vec, sense, n_points=1,
                                      restarts=2, seed=0, tol=1e-6, bias=0.5):
    """
    Find feasible distributions *biased toward an objective extreme*, for use as
    warm-start seeds in ``optimize_marginal_slsqp``.

    Identical to ``find_feasible_points`` except the SLSQP search minimizes the
    total squared constraint violation **minus** a reward term pulling
    ``obj_vec @ p`` toward its extreme (large for ``sense='max'``, small for
    ``'min'``). This steers the feasibility search out of the single central
    basin that the unbiased search funnels into, landing seeds near the true
    objective extreme.

    Why this is needed: ``find_feasible_points`` only minimizes violation, so on
    a nonconvex bilinear-LMC equality manifold all its random starts can converge
    to one interior feasible point (e.g. every seed has ``P(atom)=0.5``). SLSQP is
    a local solver, so the downstream bound optimization then cannot reach the
    true extreme. Biasing the *search* toward the objective fixes this at the
    source while still only accepting genuinely feasible points.

    Args:
        N: int
            Number of world-probability variables.
        checks: list of (kind, callable(p)->float)
            Constraint residuals; ``'eq'`` must be ~0, ``'ineq'`` must be >= 0.
        obj_vec: numpy array of length N
            Linear objective coefficients (the marginal indicator).
        sense: str
            ``'min'`` or ``'max'`` --- the extreme to bias the search toward.
        n_points: int
            Stop once this many distinct feasible points have been collected.
        restarts: int
            Maximum number of SLSQP restarts (kept very small: the bias makes the
            uniform start reach the extreme on the first try in practice, and the
            search stops as soon as ``n_points`` feasible seeds are collected).
        seed: int
            RNG seed for reproducible restarts.
        tol: float
            Feasibility tolerance for *accepting* a point (the accept test is the
            unbiased ``_checks_feasible``; the bias only steers the search).
        bias: float
            Strength of the objective-pull term. Large enough to escape the
            central basin, small relative to the violation penalty so feasibility
            still dominates.

    Returns:
        A list of feasible solution vectors (numpy arrays); empty if none found.
    """
    from scipy.optimize import minimize as _sp_minimize

    obj_vec = np.asarray(obj_vec, dtype=float)
    sign = 1.0 if sense == "max" else -1.0

    def _biased(p):
        tot = 0.0
        for kind, fn in checks:
            g = fn(p)
            tot += g * g if kind == "eq" else max(0.0, -g) ** 2
        # Subtract a reward for moving the objective toward its extreme, so a
        # lower _biased value corresponds to (feasible AND closer to the extreme).
        return tot - sign * bias * float(obj_vec @ p)

    simplex = [{"type": "eq", "fun": lambda p: float(p.sum()) - 1.0}]
    rng = np.random.default_rng(seed)
    points = []
    for k in range(restarts):
        x0 = (np.full(N, 1.0 / N) if k == 0 else rng.random(N))
        x0 = x0 / x0.sum()
        res = _sp_minimize(_biased, x0, method="SLSQP",
                           bounds=[(0.0, 1.0)] * N, constraints=simplex,
                           options={"maxiter": 800, "ftol": 1e-16})
        p = np.asarray(res.x, dtype=float)
        if _checks_feasible(p, checks, tol):
            points.append(p)
            if len(points) >= n_points:
                break
    return points


def optimize_marginal_slsqp(N, obj_vec, checks, sense, seeds, feas_tol=1e-6,
                            diversify=True):
    """
    Isolated wrapper around :func:`_optimize_marginal_slsqp_impl` -- see
    :func:`run_isolated`. Runs the SLSQP bound optimization in a child process
    so a native abort degrades to ``(None, False)`` ("no feasible optimum
    found") instead of crashing the run.
    """
    return run_isolated(
        _optimize_marginal_slsqp_impl,
        args=(N, obj_vec, checks, sense, seeds, feas_tol, diversify),
        on_crash=(None, False))


def _optimize_marginal_slsqp_impl(N, obj_vec, checks, sense, seeds,
                                  feas_tol=1e-6, diversify=True):
    """
    Optimize a linear objective ``obj_vec @ p`` over the 2^N simplex subject to
    ``checks``, starting SLSQP from each feasible seed in ``seeds`` and keeping
    the best feasible result. This is phase 2 of the robust bound computation:
    SLSQP reliably stays in the feasible region when launched *from* a feasible
    point (unlike from a random start), so good seeds (from
    ``find_feasible_points``) are essential.

    Because SLSQP is a *local* solver on the nonconvex bilinear-LMC manifold, a
    seed set clustered in one basin caps the achievable bound (e.g. all seeds at
    ``P(atom)=0.5`` make the search report ~0.5 as the extreme even when the true
    bound is 0 or 1). Two complementary seed-diversification mechanisms address
    this: (1) the supplied ``seeds`` should already be spread across the manifold
    (``find_feasible_points(..., spread=True)``), which covers atoms whose extreme
    is *interior*; (2) when ``diversify`` is True this function additionally
    generates objective-biased feasible seeds (``find_biased_feasible_points``)
    that sit near the requested extreme, covering *far corner* extremes (e.g.
    P(atom)=0/1) that a jitter radius would not reach. Both only ever add seeds:
    every accepted optimum is re-checked against the real constraints, so the
    bound can only tighten toward the true extreme, never become infeasible.

    Args:
        N: int
            Number of world-probability variables.
        obj_vec: numpy array of length N
            Linear objective coefficients.
        checks: list of (kind, callable(p)->float)
            Constraint residuals (see find_feasible_points).
        sense: str
            ``'min'`` or ``'max'``.
        seeds: list of numpy arrays
            Feasible warm-start points.
        feas_tol: float
            Feasibility tolerance for accepting a returned point.
        diversify: bool
            When True (default), generate extra objective-biased feasible seeds
            and union them with ``seeds`` before optimizing. Set False to use
            only the supplied seeds.

    Returns:
        (best_value, feasible) where feasible is True iff some seed produced a
        feasible optimum.
    """
    from scipy.optimize import minimize as _sp_minimize

    obj_vec = np.asarray(obj_vec, dtype=float)
    sign = 1.0 if sense == "min" else -1.0

    # Augment the (possibly central-basin-clustered) seeds with feasible points
    # biased toward the requested objective extreme, so the local SLSQP search
    # can actually reach it. Call the *impl* (not the isolated wrapper): we are
    # already inside the isolated child here, so a nested fork must be avoided.
    if diversify:
        seeds = list(seeds) + _find_biased_feasible_points_impl(
            N, checks, obj_vec, sense)

    cons = [{"type": "eq", "fun": lambda p: float(p.sum()) - 1.0}]
    for kind, fn in checks:
        if kind == "eq":
            cons.append({"type": "eq", "fun": (lambda p, fn=fn: fn(p))})
        else:
            cons.append({"type": "ineq", "fun": (lambda p, fn=fn: fn(p))})

    best = None
    for s in seeds:
        res = _sp_minimize(lambda p: sign * float(obj_vec @ p),
                           np.asarray(s, dtype=float), method="SLSQP",
                           bounds=[(0.0, 1.0)] * N, constraints=cons,
                           options={"maxiter": 800, "ftol": 1e-12})
        p = np.asarray(res.x, dtype=float)
        if _checks_feasible(p, checks, feas_tol):
            val = float(obj_vec @ p)
            if best is None:
                best = val
            else:
                best = min(best, val) if sense == "min" else max(best, val)
    return (best, best is not None)


def optimize_marginal_ratio_slsqp(N, num_vec, den_vec, checks, sense, seeds,
                                  feas_tol=1e-6, den_floor=1e-9,
                                  diversify=True):
    """
    Isolated wrapper around :func:`_optimize_marginal_ratio_slsqp_impl` -- see
    :func:`run_isolated`. Degrades to ``(None, False)`` on child crash.
    """
    return run_isolated(
        _optimize_marginal_ratio_slsqp_impl,
        args=(N, num_vec, den_vec, checks, sense, seeds, feas_tol, den_floor,
              diversify),
        on_crash=(None, False))


def _optimize_marginal_ratio_slsqp_impl(N, num_vec, den_vec, checks, sense,
                                        seeds, feas_tol=1e-6, den_floor=1e-9,
                                        diversify=True):
    """
    Optimize a *fractional* objective ``(num_vec @ p) / (den_vec @ p)`` over the
    2^N simplex subject to ``checks``, starting SLSQP from each feasible seed and
    keeping the best feasible result. This is the conditional-probability analogue
    of :func:`optimize_marginal_slsqp`: it computes a bound on
    ``P(child, parents) / P(parents)`` for the "linear" factorization's parent
    case directly on the simplex ``p`` (rather than the Charnes-Cooper ``(y, t)``
    space), so the existing simplex-based seed machinery applies unchanged.

    Points where the denominator falls below ``den_floor`` are rejected: the
    conditional is undefined there and SLSQP would otherwise chase a spurious
    extreme. As with the linear variant, every accepted optimum is re-checked
    against the real constraints, so the bound can only tighten toward the true
    extreme, never become infeasible.

    Args:
        N: int
            Number of world-probability variables.
        num_vec: numpy array of length N
            Numerator coefficients (indicator of child AND parents match).
        den_vec: numpy array of length N
            Denominator coefficients (indicator of parents match).
        checks: list of (kind, callable(p)->float)
            Constraint residuals (see find_feasible_points).
        sense: str
            ``'min'`` or ``'max'``.
        seeds: list of numpy arrays
            Feasible warm-start points.
        feas_tol: float
            Feasibility tolerance for accepting a returned point.
        den_floor: float
            Minimum denominator value; points below are treated as infeasible.
        diversify: bool
            When True (default), generate extra feasible seeds biased toward the
            numerator extreme and union them with ``seeds`` before optimizing.

    Returns:
        (best_value, feasible) where feasible is True iff some seed produced a
        feasible optimum with a well-defined (above-floor) denominator.
    """
    from scipy.optimize import minimize as _sp_minimize

    num_vec = np.asarray(num_vec, dtype=float)
    den_vec = np.asarray(den_vec, dtype=float)
    sign = 1.0 if sense == "min" else -1.0

    # Augment seeds with feasible points biased toward the numerator extreme; the
    # numerator is the dominant lever on the ratio when the denominator is bounded
    # away from zero, so this helps the local search reach far/corner extremes.
    # Call the *impl* (not the isolated wrapper): already inside the child here.
    if diversify:
        seeds = list(seeds) + _find_biased_feasible_points_impl(
            N, checks, num_vec, sense)

    cons = [{"type": "eq", "fun": lambda p: float(p.sum()) - 1.0}]
    for kind, fn in checks:
        if kind == "eq":
            cons.append({"type": "eq", "fun": (lambda p, fn=fn: fn(p))})
        else:
            cons.append({"type": "ineq", "fun": (lambda p, fn=fn: fn(p))})

    def _ratio(p):
        den = float(den_vec @ p)
        if den < den_floor:
            # Penalize: drive the search away from the undefined region.
            return sign * 1e6
        return sign * (float(num_vec @ p) / den)

    best = None
    for s in seeds:
        res = _sp_minimize(_ratio, np.asarray(s, dtype=float), method="SLSQP",
                           bounds=[(0.0, 1.0)] * N, constraints=cons,
                           options={"maxiter": 800, "ftol": 1e-12})
        p = np.asarray(res.x, dtype=float)
        den = float(den_vec @ p)
        if den >= den_floor and _checks_feasible(p, checks, feas_tol):
            val = float(num_vec @ p) / den
            if best is None:
                best = val
            else:
                best = min(best, val) if sense == "min" else max(best, val)
    return (best, best is not None)


def check_consistency_product_witness(lcn: LCN, restarts: int = 40,
                                      seed: int = 0, tol: float = 1e-7) -> bool:
    """
    Fast, SOUND (but conservative) consistency check for the random generator.

    A product distribution ``P(world) = prod_i q_i`` over per-atom marginals
    ``q`` satisfies EVERY Local Markov Condition independence ``X |= Y | S``
    automatically (full factorization implies all conditional independencies).
    So instead of solving over the 2^n world-probabilities subject to the dense
    joint-LMC constraint system (minutes at n=10), we search only the ``n``
    marginals ``q in [0,1]^n`` and ask whether some product distribution
    satisfies the LCN's SENTENCE bounds. If one does, it is an explicit witness
    that the LCN is consistent -> return True. If none is found within the
    restart budget, return False.

    This is SOUND (every True is a genuine consistency certificate) but
    CONSERVATIVE: an LCN consistent only via a non-product distribution is
    rejected. That is acceptable for rejection sampling -- rejected candidates
    are simply regenerated, and accepted ones are guaranteed consistent. This is
    NOT a replacement for ``check_consistency`` (the full joint-LMC oracle used
    by the inference path), which must remain unweakened.

    Args:
        lcn: LCN
            The input LCN model.
        restarts: int
            Number of marginal-vector restarts (the first is q = 0.5).
        seed: int
            RNG seed for reproducible restarts.
        tol: float
            Per-sentence-bound feasibility tolerance for accepting a witness.

    Returns:
        `True` if a product-distribution witness satisfying all sentence bounds
        is found, otherwise `False`.
    """
    from scipy.optimize import minimize as _sp_minimize

    vars = [k for k, _ in lcn.atoms.items()]
    n = len(vars)
    if n == 0 or n > 10:
        return True  # defensive: caller gates at <= 10

    table = build_truth_table(n)                     # (2^n, n) int8
    interpretations = [dict(zip(vars, row)) for row in table]
    table_one = (table == 1)

    # Precompute sentence indicator vectors once (~n sentences).
    sentences = []
    for _, s in lcn.sentences.items():
        lo, hi = s.get_lower_bound(), s.get_upper_bound()
        if s.type == SentenceType.Type1:
            sentences.append(("t1", eval_indicator(s.phi_formula, interpretations), lo, hi))
        else:
            sentences.append(("t2", eval_indicator(s.phi_and_psi_formula, interpretations),
                              eval_indicator(s.psi_formula, interpretations), lo, hi))

    def _make_p(q):
        # p[j] = prod_i (q_i if world j has atom i true else 1 - q_i)
        return np.prod(np.where(table_one, q, 1.0 - q), axis=1)

    def _sq_violation(q):
        p = _make_p(q)
        tot = 0.0
        for entry in sentences:
            if entry[0] == "t1":
                _, A, lo, hi = entry
                v = float(A @ p)
                tot += max(0.0, lo - v) ** 2 + max(0.0, v - hi) ** 2
            else:
                _, Aqr, Ar, lo, hi = entry
                ppsi = float(Ar @ p)
                pjoint = float(Aqr @ p)
                tot += max(0.0, lo * ppsi - pjoint) ** 2 \
                    + max(0.0, pjoint - hi * ppsi) ** 2
        return tot

    def _witness_ok(q):
        p = _make_p(q)
        for entry in sentences:
            if entry[0] == "t1":
                _, A, lo, hi = entry
                v = float(A @ p)
                if v < lo - tol or v > hi + tol:
                    return False
            else:
                _, Aqr, Ar, lo, hi = entry
                ppsi = float(Ar @ p)
                pjoint = float(Aqr @ p)
                if pjoint < lo * ppsi - tol or pjoint > hi * ppsi + tol:
                    return False
        return True

    if not sentences:
        return True  # no bounds to satisfy; any product distribution works

    rng = np.random.default_rng(seed)
    bounds = [(0.0, 1.0)] * n
    for k in range(restarts):
        q0 = np.full(n, 0.5) if k == 0 else rng.random(n)
        res = _sp_minimize(_sq_violation, q0, method="L-BFGS-B",
                           bounds=bounds,
                           options={"maxiter": 500, "ftol": 1e-16})
        if _witness_ok(np.asarray(res.x, dtype=float)):
            return True
    return False


def check_consistency(lcn: LCN, max_slsqp_restarts: int = 300) -> bool:
    """
    Check if the LCN is consistent or not. An LCN is consistent
    if there exists a model (i.e., interpretation) that satisfies
    the LCN's sentences and the Local Markov Condition independencies.

    Args:
        lcn: LCN
            The input LCN model.
        max_slsqp_restarts: int
            Restart budget for the SLSQP feasibility fallback used when ipopt
            cannot find a feasible point. The default (300) is thorough but slow
            at the n=10 boundary (the search is over 2^n world-probabilities).
            Callers that only need a quick yes/no answer (e.g. the random
            instance generator's rejection loop) can pass a much smaller value
            to trade exhaustiveness for speed. A consistent LCN may then
            occasionally be reported inconsistent if the smaller budget misses a
            feasible point, so use a reduced budget only where that is
            acceptable (rejection sampling simply discards and regenerates).

    Returns:
        `True` if the LCN is consistent and `False` otherwise.
    """
    # Create the interpretations
    vars = [k for k, _ in lcn.atoms.items()]
    items = list(itertools.product([0, 1], repeat=len(vars)))
    index = {k:v for k, v in enumerate(items)}
    N = len(items)

    # Check consistency for small enough LCNs (up to 10 atoms)
    if len(vars) > 10:
        print("LCN is too large for exact consistency checking.")
        return True

    # Create the model and variables
    model = ConcreteModel()
    model.ITEMS = Set(initialize=index.keys())
    model.p = Var(model.ITEMS, within=NonNegativeReals, bounds=(0.0, 1.0))
    model.constr = ConstraintList()

    # We also record each constraint as a residual callable over a solution
    # vector ``pv`` (numpy). ``eq`` residuals must be ~0; ``ineq`` residuals
    # must be >= 0. This lets us verify feasibility of a returned point directly
    # instead of trusting only ipopt's termination flag (the corrected, denser
    # LMC constraint system makes a single un-seeded ipopt solve unreliable, so
    # we solve from several starts and accept any genuinely feasible point).
    checks = []  # list of (kind, callable(pv)->float)

    # Create the constraint ensuring a probability distribution over interpretations
    model.constr.add(sum(model.p[i] for i in model.ITEMS) == 1.0)
    checks.append(("eq", lambda pv: float(pv.sum()) - 1.0))

    # Create the constraints for the sentences
    for sid, s in lcn.sentences.items():
        if s.type == SentenceType.Type1: # Type 1 sentence P(phi)
            A = np.array([1.0 if s.phi_formula.evaluate(table=dict(zip(vars, index[j]))) else 0.0
                          for j in range(N)])
            lobo = s.get_lower_bound()
            upbo = s.get_upper_bound()
            model.constr.add(sum(A[i]*model.p[i] for i in model.ITEMS) >= lobo)
            model.constr.add(sum(A[i]*model.p[i] for i in model.ITEMS) <= upbo)
            checks.append(("ineq", lambda pv, A=A, lobo=lobo: float(A @ pv) - lobo))
            checks.append(("ineq", lambda pv, A=A, upbo=upbo: upbo - float(A @ pv)))
        else: # Type 2 sentence: P(phi|psi)
            Aqr = np.array([1.0 if s.phi_and_psi_formula.evaluate(table=dict(zip(vars, index[j]))) else 0.0
                            for j in range(N)])
            Ar = np.array([1.0 if s.psi_formula.evaluate(table=dict(zip(vars, index[j]))) else 0.0
                           for j in range(N)])
            lobo = s.get_lower_bound()
            upbo = s.get_upper_bound()
            val = sum(Ar[i]*model.p[i] for i in model.ITEMS)
            model.constr.add(sum(Aqr[i]*model.p[i] for i in model.ITEMS) >= lobo*val)
            model.constr.add(sum(Aqr[i]*model.p[i] for i in model.ITEMS) <= upbo*val)
            checks.append(("ineq", lambda pv, Aqr=Aqr, Ar=Ar, lobo=lobo: float(Aqr @ pv) - lobo * float(Ar @ pv)))
            checks.append(("ineq", lambda pv, Aqr=Aqr, Ar=Ar, upbo=upbo: upbo * float(Ar @ pv) - float(Aqr @ pv)))

    # Constraints corresponding to the independence assumptions
    # Atom x is conditionaly independent of non-parents non-descendants (T) 
    # given its parents (S) in the primal graph of the LCN
    # Namely, we consider independence assertions [X, Y=T, Z=S]
    # i.e., P(x|S,T) = P(x|S)
    #   P(x,S,T)P(S) = P(x,S)P(S,T)
    # Here, independencies are coming from LCN's Local Markov Condition
    independencies = lcn.local_markov_condition()
    print(f"Local Markov Condition yields {len(independencies.get_assertions())} independencies.")

    # Vectorized LMC indicator construction: build the truth table once and use
    # numpy column masks instead of Formula.evaluate per interpretation. At n=10
    # an assertion's joint encoding expands to thousands of conjunction groups,
    # so the Formula-based path was the dominant cost (~minutes); this is
    # bit-identical but ~3 orders of magnitude faster.
    table = build_truth_table(len(vars))
    col_of = {v: i for i, v in enumerate(vars)}

    for indep in independencies.get_assertions():
        print(f"independence: {indep}")
        # Correct joint encoding of (X |= Y | S): see lmc_constraint_groups.
        for group in lmc_constraint_groups_vec(indep, table, col_of):
            if group[0] == 'conditional':
                _, Aa, Ab, Ac, Ad = group
                val1 = sum(Aa[i]*model.p[i] for i in model.ITEMS) * sum(Ab[i]*model.p[i] for i in model.ITEMS)
                val2 = sum(Ac[i]*model.p[i] for i in model.ITEMS) * sum(Ad[i]*model.p[i] for i in model.ITEMS)
                model.constr.add(val1 - val2 == 0.0)
                checks.append(("eq", lambda pv, Aa=Aa, Ab=Ab, Ac=Ac, Ad=Ad:
                               float(Aa @ pv) * float(Ab @ pv) - float(Ac @ pv) * float(Ad @ pv)))
            else:
                _, Aa, Ab, Ac = group
                val1 = sum(Aa[i]*model.p[i] for i in model.ITEMS)
                val2 = sum(Ab[i]*model.p[i] for i in model.ITEMS) * sum(Ac[i]*model.p[i] for i in model.ITEMS)
                model.constr.add(val1 - val2 == 0.0)
                checks.append(("eq", lambda pv, Aa=Aa, Ab=Ab, Ac=Ac:
                               float(Aa @ pv) - float(Ab @ pv) * float(Ac @ pv)))

    # Pure feasibility problem (constant objective).
    model.objective = Objective(expr=1.0, sense=maximize)

    # The corrected joint-LMC system is nonconvex and dense; a single un-seeded
    # ipopt solve frequently stalls at an infeasible point even when the LCN is
    # consistent (a feasible distribution can require many restarts to land on).
    # Try ipopt from a few starts first (cheap when it works), then verify the
    # returned point against `checks` directly rather than trusting only its
    # termination flag. Use the "fast" mode: this is a constant-objective
    # feasibility solve and any returned point is re-verified below at 1e-6, so a
    # looser termination cannot produce a wrong verdict -- it just returns sooner.
    opt = make_ipopt(mode="fast")
    rng = np.random.default_rng(0)
    consistent = False
    # Silence Pyomo's routine "Loading a SolverResults object with a warning
    # status" messages: a non-optimal ipopt termination is expected here (the
    # point is re-verified against `checks` and an SLSQP fallback follows).
    pyomo_logger = logging.getLogger('pyomo')
    _prev_level = pyomo_logger.level
    pyomo_logger.setLevel(logging.ERROR)
    try:
        for k in range(8):
            start = np.full(N, 1.0 / N) if k == 0 else rng.random(N)
            start = start / start.sum()
            for i in model.ITEMS:
                model.p[i].value = float(start[i])
            try:
                opt.solve(model, tee=False)
                pv = np.array([value(model.p[i]) for i in model.ITEMS], dtype=float)
            except Exception as e:
                print(f"Exception during ipopt (restart {k}): {str(e)}")
                continue
            if not np.any(np.isnan(pv)) and _checks_feasible(pv, checks, 1e-6):
                consistent = True
                break
    finally:
        pyomo_logger.setLevel(_prev_level)

    # Fallback: ipopt is unreliable at *finding feasibility* in this dense
    # nonconvex equality system and can miss a feasible point that exists. Use
    # the SLSQP feasibility search; any returned point is an explicit certificate.
    if not consistent:
        try:
            points = find_feasible_points(N, checks, n_points=1,
                                          restarts=max_slsqp_restarts)
            consistent = len(points) > 0
        except Exception as e:
            print(f"Exception during SLSQP fallback: {str(e)}")

    print(f"[check_consistency] consistent={consistent}")
    return consistent

