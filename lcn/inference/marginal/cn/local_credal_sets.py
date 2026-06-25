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

# Local credal-set solver for the chain-graph factorization.
#
# Given a symbolic factor P(child | parents) and one interpretation of its
# scope, these routines solve the (fractional) optimization problem induced
# by the LCN constraints to obtain the lower/upper bound of the corresponding
# local credal set. The solving was previously embedded in the factorization
# class; it now lives here so that `ChainGraphFactorization` stays purely
# symbolic and `CredalNetwork` calls this module to compute the intervals.
#
# Two factorization methods are supported:
#   - "linear": treat each family in isolation. With no parents this is a plain
#     LP; with parents it is a linear-fractional program P(child,parents) /
#     P(parents), solved exactly via the Charnes-Cooper transformation. Note
#     that the family already carries every LCN sentence whose atom scope lies
#     inside the family scope (process_chain_graph collects these), so "linear"
#     is the in-scope-sentence LP; what it omits are the LMC independence
#     equalities, which couple states the per-family LP cannot see.
#   - "linear-tight" (scheme D1 of docs/tighter_approximation.tex): the "linear"
#     program PLUS the Local Markov Condition bilinear equalities for every LMC
#     assertion whose full atom set is contained in the family scope. Those
#     equalities make the program nonconvex (bilinear), so it is solved on the
#     hardened ipopt path or, under solver="scip", to certified global optimality
#     (scheme D3). Families with no in-scope LMC assertion fall back to the fast
#     pure-"linear" path, so "linear-tight" == "linear" there.
#
# Two solver backends are supported:
#   - "ipopt" (default): a local NLP solver. Because the "linear-tight" program
#     (and the nonconvex corners of the "linear" fractional program) can trap a
#     local solver at a wrong/vacuous point, the ipopt path is *hardened* with
#     the same multi-restart + vacuous-bound detection + SLSQP-fallback strategy
#     as `ExactInference._robust_solve` (lcn/inference/marginal/exact.py).
#   - "scip": SCIP's spatial branch-and-bound global solver (Pyomo AMPL/NL
#     interface), which solves the fractional / bilinear programs to certified
#     global optimality. SCIP is fed the raw fractional objective directly (no
#     Charnes-Cooper needed). Requires the SCIP CLI on PATH (`brew install scip`).

import itertools

import numpy as np
from pyomo.environ import (
    ConcreteModel,
    Set, NonNegativeReals,
    Var, ConstraintList,
    Objective, minimize, maximize,
    SolverStatus,
    value,
    TerminationCondition
)

# Local
from lcn.core.model import LCN, SentenceType
from lcn.inference.marginal.exact import make_scip, _read_gap, _OPTIMAL
from lcn.inference.utils.common import (
    make_conjunction, make_ipopt,
    eval_indicator, dot,
    find_feasible_points,
    optimize_marginal_slsqp, optimize_marginal_ratio_slsqp,
    build_truth_table, lmc_constraint_groups_vec,
)

# Number of random restarts for the hardened ipopt path before giving up.
_N_RESTARTS = 4
# A solved bound this close to 0 (min) or 1 (max) is treated as suspicious
# (likely a vacuous local-solver artifact) and triggers the fallback.
_VACUOUS_TOL = 1e-6
# Minimum denominator P(parents) for the SCIP raw-fractional objective. The
# conditional P(child | parents) is undefined when P(parents)=0, and SCIP's
# spatial branch-and-bound will otherwise drive the denominator to ~0 and report
# an unbounded ratio. Constraining P(parents) >= floor restricts the search to
# the region where the conditional is defined (cf. exact.py's den_floor).
_SCIP_DEN_FLOOR = 1e-6


def _dot_var(vec: np.ndarray, var, items):
    """Build Pyomo linear expression: vec @ var over items."""
    return sum(float(vec[i]) * var[i] for i in items)


def _init_p(model, N, rng=None):
    """Initialize the simplex variables: uniform, or a random point if rng given."""
    if rng is None:
        start = np.full(N, 1.0 / N)
    else:
        start = rng.random(N)
        start = start / start.sum()
    for i in model.ITEMS:
        model.p[i].value = float(start[i])


class LocalCredalSetSolver:
    """
    Solves the per-interpretation optimization problems that define the local
    credal sets of a chain-graph factorization. One solver instance is bound to
    an :class:`LCN`, a factorization ``method`` ("linear" or "linear-tight") and
    a ``solver`` backend ("ipopt" or "scip"); its :meth:`solve` method returns
    the min/max bound for a single factor interpretation.

    The instance holds only the LCN and small config, so it can be pickled and
    shipped to worker processes for parallel per-family solving.
    """

    def __init__(self, lcn: LCN, method: str = "linear", solver: str = "ipopt",
                 time_limit: float = None, gap_tol: float = 0.0,
                 verbosity: int = 1):
        assert method in ("linear", "linear-tight"), \
            f"Unknown method '{method}'. Use 'linear' or 'linear-tight'."
        assert solver in ("ipopt", "scip"), \
            f"Unknown solver '{solver}'. Use 'ipopt' or 'scip'."
        self.lcn = lcn
        self.method = method
        self.solver = solver
        self.time_limit = time_limit
        self.gap_tol = gap_tol
        self.verbosity = verbosity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, scope, literals, child, parents, sentences, sense):
        """
        Solve a single factor interpretation for the configured method/solver.

        Args:
            scope: full (flattened) atom scope of the family.
            literals: dict mapping each scope atom to its 0/1 value.
            child: child node name (possibly compound "A-B").
            parents: flattened list of parent atoms.
            sentences: sentence ids attached to the family.
            sense: "min" or "max".
        """
        if self.method == "linear":
            return self._solve_linear(
                scope, literals, child, parents, sentences, sense)
        else:  # "linear-tight"
            return self._solve_linear_tight(
                scope, literals, child, parents, sentences, sense)

    # ------------------------------------------------------------------
    # Linear factorization (LP / Charnes-Cooper fractional LP)
    # ------------------------------------------------------------------

    def _linear_rows_and_obj(self, scope, literals, parents, sentences):
        """
        Build the shared pieces of the "linear" model for one interpretation:
        the <= constraint rows (coeffs, rhs), the objective numerator A, and
        (when there are parents) the denominator E. Returns
        (N, interpretations, constraint_rows, A, E_or_None).
        """
        items_tuples = list(itertools.product([0, 1], repeat=len(scope)))
        interpretations = [dict(zip(scope, t)) for t in items_tuples]
        N = len(interpretations)

        # Build constraint rows: each row is (coeffs, rhs) for a <= constraint
        constraint_rows = []

        # Probability simplex: sum(p) = 1 encoded as two <= rows
        constraint_rows.append((-np.ones(N), -1.0))  # -sum(p) <= -1 i.e. sum(p) >= 1
        constraint_rows.append((np.ones(N), 1.0))     # sum(p) <= 1

        for sid in sentences:
            s = self.lcn.sentences.get(sid)
            if s.type == SentenceType.Type1:  # P(phi)
                A = eval_indicator(s.phi_formula, interpretations)
                lobo = s.get_lower_bound()
                upbo = s.get_upper_bound()
                # sum(A*p) >= lobo  =>  -sum(A*p) <= -lobo
                constraint_rows.append((-A, -lobo))
                # sum(A*p) <= upbo
                constraint_rows.append((A, upbo))
            else:  # Type 2 sentence P(phi | psi)
                Aqr = eval_indicator(s.phi_and_psi_formula, interpretations)
                Ar = eval_indicator(s.psi_formula, interpretations)
                lobo = s.get_lower_bound()
                upbo = s.get_upper_bound()
                # sum((Aqr - lobo*Ar)*p) >= 0  =>  sum((lobo*Ar - Aqr)*p) <= 0
                constraint_rows.append((lobo * Ar - Aqr, 0.0))
                # sum((Aqr - upbo*Ar)*p) <= 0
                constraint_rows.append((Aqr - upbo * Ar, 0.0))

        # Objective numerator vector: indicator of the full interpretation
        Fq = make_conjunction(variables=scope, literals=literals)
        A = eval_indicator(Fq, interpretations)

        E = None
        if len(parents) > 0:
            Fe = make_conjunction(variables=parents, literals=literals)
            E = eval_indicator(Fe, interpretations)

        return N, interpretations, constraint_rows, A, E

    def _solve_linear(self, scope, literals, child, parents, sentences, sense):
        N, interpretations, constraint_rows, A, E = self._linear_rows_and_obj(
            scope, literals, parents, sentences)

        if self.solver == "scip":
            return self._scip_linear(N, constraint_rows, A, E, sense)

        # ipopt (hardened)
        if E is None:
            # No parents: standard LP, linear objective A @ p
            return self._robust_linear_lp(N, A, constraint_rows, sense)
        else:
            # With parents: fractional objective (A*E @ p) / (E @ p)
            AE = A * E
            return self._robust_fractional_lp(N, AE, E, constraint_rows, sense)

    # ------------------------------------------------------------------
    # Tight factorization (D1: in-scope sentences + scope-restricted LMC,
    # bilinear)
    # ------------------------------------------------------------------

    def _inscope_lmc_assertions(self, scope):
        """
        The LMC independence assertions (X |= Y | S) whose *entire* atom set is
        contained in ``scope``. lmc_constraint_groups_vec references every atom
        of X, Y and S through the family-scope truth table, so an assertion can
        only be encoded locally if all of its atoms are columns of that table.
        Assertions spanning families are skipped here -- that residual is what
        schemes D2/D3 close.

        The Local Markov Condition is computed lazily and memoized on the LCN
        (so repeated family solves reuse it). Under n_jobs>1 each worker holds
        its own pickled LCN copy and computes it independently; the result is
        deterministic, so this needs no shared state.
        """
        if self.lcn.primal_graph is None:
            self.lcn.build_primal_graph()
        if self.lcn.independencies is None:
            self.lcn.local_markov_condition()
        scope_set = set(scope)
        return [a for a in self.lcn.independencies.get_assertions()
                if a.all_vars.issubset(scope_set)]

    def _build_linear_tight_model(self, scope, literals, parents, sentences,
                                  assertions, sense):
        """
        Build the nonconvex (bilinear) "linear-tight" model for one factor
        interpretation: the same simplex + in-scope sentence rows as "linear",
        PLUS the Local Markov Condition equalities of every in-scope assertion.

        The LMC rows are bit-identical to ExactInference's joint encoding
        (exact.py): for each assertion (X |= Y | S), and each group produced by
        lmc_constraint_groups_vec over the family-scope truth table, add either
        the conditional equality P(x,y,s)P(s) = P(x,s)P(y,s) or, when S is empty,
        the marginal equality P(x,y) = P(x)P(y).
        """
        items_tuples = list(itertools.product([0, 1], repeat=len(scope)))
        interpretations = [dict(zip(scope, t)) for t in items_tuples]
        N = len(interpretations)

        model = ConcreteModel()
        model.ITEMS = Set(initialize=range(N))
        model.p = Var(model.ITEMS, within=NonNegativeReals, bounds=(0.0, 1.0))
        model.constr = ConstraintList()

        # Probability simplex: sum(p) = 1
        model.constr.add(sum(model.p[i] for i in model.ITEMS) == 1.0)

        # Sentence constraints (every sentence whose scope is in the family scope
        # -- process_chain_graph already collected these into `sentences`).
        for sid in sentences:
            s = self.lcn.sentences.get(sid)
            if s.type == SentenceType.Type1:
                A = eval_indicator(s.phi_formula, interpretations)
                lobo = s.get_lower_bound()
                upbo = s.get_upper_bound()
                expr = dot(A, model, model.ITEMS)
                model.constr.add(expr >= lobo)
                model.constr.add(expr <= upbo)
            else:
                Aqr = eval_indicator(s.phi_and_psi_formula, interpretations)
                Ar = eval_indicator(s.psi_formula, interpretations)
                lobo = s.get_lower_bound()
                upbo = s.get_upper_bound()
                expr_qr = dot(Aqr, model, model.ITEMS)
                expr_r = dot(Ar, model, model.ITEMS)
                model.constr.add(expr_qr >= lobo * expr_r)
                model.constr.add(expr_qr <= upbo * expr_r)

        # Local Markov Condition equalities, restricted to the family scope.
        # The truth table / col_of use the SAME scope ordering as the
        # interpretations above, so the indicator vectors index model.p directly.
        if assertions:
            table = build_truth_table(len(scope))
            col_of = {v: i for i, v in enumerate(scope)}
            for indep in assertions:
                for group in lmc_constraint_groups_vec(indep, table, col_of):
                    if group[0] == 'conditional':
                        _, Aa, Ab, Ac, Ad = group
                        val1 = dot(Aa, model, model.ITEMS) * dot(Ab, model, model.ITEMS)
                        val2 = dot(Ac, model, model.ITEMS) * dot(Ad, model, model.ITEMS)
                        model.constr.add(val1 - val2 == 0.0)
                    else:
                        _, Aa, Ab, Ac = group
                        val1 = dot(Aa, model, model.ITEMS)
                        val2 = dot(Ab, model, model.ITEMS) * dot(Ac, model, model.ITEMS)
                        model.constr.add(val1 - val2 == 0.0)

        # Objective
        Fq = make_conjunction(variables=scope, literals=literals)
        A = eval_indicator(Fq, interpretations)

        if len(parents) == 0:
            obj_expr = dot(A, model, model.ITEMS)
        else:
            # Fractional objective via auxiliary variable:
            # obj = P(child AND parents) / P(parents). The aux variable is the
            # conditional probability, hence bounded to [0, 1] — this both is
            # correct and prevents the global solver (SCIP) from reporting an
            # unbounded ratio when P(parents) -> 0. For the global solver we also
            # floor the denominator so the conditional stays well-defined.
            Fe = make_conjunction(variables=parents, literals=literals)
            E = eval_indicator(Fe, interpretations)
            AE = A * E
            AE_expr = dot(AE, model, model.ITEMS)
            E_expr = dot(E, model, model.ITEMS)
            if self.solver == "scip":
                model.constr.add(E_expr >= _SCIP_DEN_FLOOR)
            model.obj_var = Var(within=NonNegativeReals, bounds=(0.0, 1.0))
            model.constr.add(model.obj_var * E_expr == AE_expr)
            obj_expr = model.obj_var

        model.objective = Objective(
            expr=obj_expr, sense=(minimize if sense == 'min' else maximize))
        return model, N

    def _solve_linear_tight(self, scope, literals, child, parents, sentences, sense):
        # When no LMC assertion fits inside the family scope, the enriched
        # program reduces to the pure "linear" program -- solve it on the fast
        # LP / Charnes-Cooper path (so linear-tight == linear for that family).
        assertions = self._inscope_lmc_assertions(scope)
        if not assertions:
            return self._solve_linear(
                scope, literals, child, parents, sentences, sense)

        model, N = self._build_linear_tight_model(
            scope, literals, parents, sentences, assertions, sense)
        if self.solver == "scip":
            return self._scip_extract(model, sense)
        # ipopt: multi-restart only (the bilinear LMC equalities are not
        # simplex-expressible as the SLSQP-fallback `checks`, so no SLSQP
        # fallback -- matching ExactInference's obj_vec=None ratio handling).
        return self._ipopt_multistart(model, N, sense, key=("linear-tight", sense))

    # ------------------------------------------------------------------
    # Hardened ipopt: linear LP (parentless)
    # ------------------------------------------------------------------

    def _linear_lp_model(self, N, c, constraint_rows, sense):
        """Build a standard LP model: min/max c^T p subject to constraint rows."""
        model = ConcreteModel()
        model.ITEMS = Set(initialize=range(N))
        model.p = Var(model.ITEMS, within=NonNegativeReals, bounds=(0.0, 1.0))
        model.constr = ConstraintList()
        for row_coeffs, row_rhs in constraint_rows:
            model.constr.add(dot(row_coeffs, model, model.ITEMS) <= row_rhs)
        obj_expr = dot(c, model, model.ITEMS)
        model.objective = Objective(
            expr=obj_expr, sense=(minimize if sense == 'min' else maximize))
        return model

    def _robust_linear_lp(self, N, c, constraint_rows, sense):
        """Solve the parentless LP robustly: ipopt multi-restart + SLSQP fallback."""
        model = self._linear_lp_model(N, c, constraint_rows, sense)
        best, ok = self._ipopt_multistart_raw(model, N, sense, key=("lin", sense))

        if self._is_suspicious(best, ok, sense):
            checks = self._checks_from_rows(constraint_rows)
            seeds = self._feasible_seeds(N, checks)
            if seeds:
                v, okk = optimize_marginal_slsqp(
                    N, np.asarray(c, dtype=float), checks, sense, seeds)
                best, ok = self._merge(best, ok, v, okk, sense)
        return best if ok else None

    # ------------------------------------------------------------------
    # Hardened ipopt: fractional LP (with parents)
    # ------------------------------------------------------------------

    def _fractional_lp_model(self, N, AE, E, constraint_rows, sense):
        """
        Build the Charnes-Cooper transform of (AE @ p)/(E @ p):
        min/max AE^T y s.t. row_coeffs^T y - rhs*t <= 0, E^T y == 1, y,t >= 0.
        """
        model = ConcreteModel()
        model.ITEMS = Set(initialize=range(N))
        model.y = Var(model.ITEMS, within=NonNegativeReals)
        model.t = Var(within=NonNegativeReals)
        model.constr = ConstraintList()
        for row_coeffs, row_rhs in constraint_rows:
            model.constr.add(
                _dot_var(row_coeffs, model.y, model.ITEMS) - row_rhs * model.t <= 0)
        model.constr.add(_dot_var(E, model.y, model.ITEMS) == 1)
        obj_expr = _dot_var(AE, model.y, model.ITEMS)
        model.objective = Objective(
            expr=obj_expr, sense=(minimize if sense == 'min' else maximize))
        return model

    def _robust_fractional_lp(self, N, AE, E, constraint_rows, sense):
        """
        Solve the fractional LP robustly. Primary path: ipopt on the
        Charnes-Cooper LP (an exact reformulation, so a single solve is usually
        enough). Fallback: optimize the ratio directly on the simplex p with
        SLSQP (optimize_marginal_ratio_slsqp), which the simplex-based seed
        machinery supports.
        """
        model = self._fractional_lp_model(N, AE, E, constraint_rows, sense)
        best, ok = self._ipopt_multistart_raw(
            model, N, sense, key=("frac", sense), init_var=model.y)

        if self._is_suspicious(best, ok, sense):
            checks = self._checks_from_rows(constraint_rows)
            seeds = self._feasible_seeds(N, checks)
            if seeds:
                v, okk = optimize_marginal_ratio_slsqp(
                    N, np.asarray(AE, dtype=float), np.asarray(E, dtype=float),
                    checks, sense, seeds)
                best, ok = self._merge(best, ok, v, okk, sense)
        return best if ok else None

    # ------------------------------------------------------------------
    # ipopt solve drivers
    # ------------------------------------------------------------------

    def _make_ipopt(self):
        # verbosity >= 2 shows the solver's own progress (ipopt print_level).
        opt = make_ipopt(debug=(self.verbosity >= 2), mode="exact")
        if self.time_limit is not None:
            opt.options['max_cpu_time'] = float(self.time_limit)
            opt.options['max_wall_time'] = float(self.time_limit)
        return opt

    def _ipopt_solve_once(self, model):
        """Solve once with ipopt; return (objective_value_or_None, optimal_bool)."""
        opt = self._make_ipopt()
        try:
            results = opt.solve(model, tee=(self.verbosity >= 2))
            tc = results.solver.termination_condition
            status = results.solver.status
            val = value(model.objective, exception=False)
            val = float(val) if val is not None else None
            optimal = (status == SolverStatus.ok and tc == TerminationCondition.optimal)
            return val, optimal
        except Exception as ex:
            if self.verbosity > 1:
                print(f"[LocalCredalSetSolver] ipopt exception: {ex}")
            return None, False

    def _ipopt_multistart_raw(self, model, N, sense, key, init_var=None):
        """
        Solve `model` with ipopt from a uniform start, then from a few random
        restarts if the first result is suspicious. Returns (best, ok). When
        `init_var` is given (Charnes-Cooper y), only the uniform start is seeded
        on model.p; restarts perturb the named var instead.
        """
        # Primary solve from uniform start (seed model.p if present).
        if hasattr(model, 'p'):
            _init_p(model, N)
        best, ok = self._ipopt_solve_once(model)

        if self._is_suspicious(best, ok, sense):
            for k in range(_N_RESTARTS):
                seed = abs(hash((key, k))) % (2 ** 32)
                rng = np.random.default_rng(seed)
                if hasattr(model, 'p'):
                    _init_p(model, N, rng=rng)
                elif init_var is not None:
                    for i in model.ITEMS:
                        init_var[i].value = float(rng.random())
                v, okk = self._ipopt_solve_once(model)
                best, ok = self._merge(best, ok, v, okk, sense)
                if not self._is_suspicious(best, ok, sense):
                    break
        return best, ok

    def _ipopt_multistart(self, model, N, sense, key):
        """Multi-restart ipopt returning the bound (or None)."""
        best, ok = self._ipopt_multistart_raw(model, N, sense, key)
        return best if ok else None

    # ------------------------------------------------------------------
    # SCIP solve drivers
    # ------------------------------------------------------------------

    def _scip_linear(self, N, constraint_rows, A, E, sense):
        """
        Build and globally solve the linear-method model with SCIP.

        For the parentless case this is a plain LP. For the parent case we feed
        SCIP the Charnes-Cooper transform (also a plain LP) rather than a raw
        division or a bilinear aux-variable objective: the homogenized LP encodes
        the conditional exactly and avoids the spurious feasible region that a
        denominator-floored ratio introduces when P(parents) -> 0 (where the
        ratio is unconstrained but the sentence bounds are still satisfiable).
        SCIP solves the resulting LP to global optimality.
        """
        if E is None:
            model = self._linear_lp_model(N, A, constraint_rows, sense)
        else:
            AE = A * E
            model = self._fractional_lp_model(N, AE, E, constraint_rows, sense)
        return self._scip_extract(model, sense)

    def _scip_extract(self, model, sense):
        """Solve `model` with SCIP and return the objective value (or None)."""
        solver = make_scip(
            time_limit=(self.time_limit if self.time_limit is not None else 3600.0),
            gap_tol=self.gap_tol)
        try:
            results = solver.solve(model, load_solutions=False,
                                   tee=(self.verbosity >= 2))
        except Exception as ex:
            if self.verbosity > 1:
                print(f"[LocalCredalSetSolver] scip exception: {ex}")
            return None
        gap = _read_gap(results)
        obj_val = None
        try:
            model.solutions.load_from(results)
            v = value(model.objective, exception=False)
            obj_val = float(v) if v is not None else None
        except Exception:
            obj_val = None
        if self.verbosity > 1:
            tc = results.solver.termination_condition
            confirmed = (tc in _OPTIMAL) or (gap is not None and gap <= self.gap_tol + 1e-12)
            print(f"[LocalCredalSetSolver] scip {sense}: value={obj_val}, "
                  f"gap={gap}, confirmed={confirmed}")
        return obj_val

    # ------------------------------------------------------------------
    # Robustness helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_suspicious(value_, ok, sense):
        """A solve looks suspicious if it failed or sits at a vacuous extreme."""
        if not ok or value_ is None:
            return True
        if sense == 'max' and value_ >= 1.0 - _VACUOUS_TOL:
            return True
        if sense == 'min' and value_ <= _VACUOUS_TOL:
            return True
        return False

    @staticmethod
    def _merge(best, ok, v, okk, sense):
        """Merge a new (v, okk) into the running (best, ok), keeping the extreme."""
        if not okk or v is None:
            return best, ok
        if best is None or not ok:
            return v, True
        if sense == 'max':
            return max(best, v), True
        return min(best, v), True

    @staticmethod
    def _checks_from_rows(constraint_rows):
        """
        Convert the linear <= constraint rows into the (kind, fn) `checks` form
        used by the SLSQP helpers. Each row (coeffs, rhs) is the inequality
        coeffs @ p <= rhs, i.e. the residual rhs - coeffs @ p must be >= 0.
        The simplex sum-to-one rows are dropped here because the SLSQP helpers
        add `sum(p) == 1` themselves.
        """
        checks = []
        for coeffs, rhs in constraint_rows:
            coeffs = np.asarray(coeffs, dtype=float)
            # Skip the two simplex rows (+/- ones with rhs +/-1).
            if np.all(coeffs == 1.0) and rhs == 1.0:
                continue
            if np.all(coeffs == -1.0) and rhs == -1.0:
                continue
            checks.append(("ineq", (lambda p, c=coeffs, r=rhs: float(r - c @ p))))
        return checks

    def _feasible_seeds(self, N, checks):
        """Feasible warm-start seeds for the SLSQP fallback (best-effort)."""
        try:
            return find_feasible_points(N, checks)
        except Exception as ex:
            if self.verbosity > 1:
                print(f"[LocalCredalSetSolver] seed search failed: {ex}")
            return []
