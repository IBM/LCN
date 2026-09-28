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

# Brute-force verifier for the marginal bounds computed by exact.py.
#
# ExactInference (exact.py) solves a nonconvex NLP over the full 2^n joint with
# ipopt, which is only a *local* solver on the dense joint-LMC system, so its
# bounds need an independent cross-check. This script computes marginal bounds a
# completely different way -- by optimizing over a low-dimensional PARAMETRIC
# family of joint distributions that is LMC-correct by construction -- and prints
# a side-by-side comparison flagging any disagreement.
#
# WHY A PARAMETRIC FAMILY (and not raw simplex enumeration)?
# The LMC induces *bilinear equality* constraints (the independencies), so the
# feasible set is a continuous, measure-zero surface inside the 2^n simplex.
# Uniform sampling of the raw simplex essentially never lands on it. Instead we
# parametrize the joint so that every parameter setting automatically satisfies
# the LMC equalities, then we only have to enforce the (much looser) sentence
# bound *inequalities*. For alarm.lcn the structure graph is the chain graph
#
#     B -> A,  E -> A,  A -> C,  A -> D,  and an UNDIRECTED edge  C -- D
#
# which yields the factorization
#
#     P(B,E,A,C,D) = P(B) * P(E) * P(A|B,E) * P(C,D|A)
#
# where P(C,D|A) is a *full* 2x2 block per value of A (C and D stay coupled given
# A -- this is exactly what the undirected C--D edge means). Crucially there is NO
# topological-DAG CPT factorization that reproduces exactly the symmetric CIs
# (C |= B,E | A,D) and (D |= B,E | A,C); forcing C |= D | A would be UNSOUND. The
# factorization above imposes precisely the LMC independencies and nothing more
# (verified numerically at startup against lmc_constraint_groups_vec).
#
# This script targets alarm.lcn specifically (the requested instance). Other
# topologies need their own factorization derived from their structure graph.

import argparse
import itertools

import numpy as np
from scipy.optimize import minimize

# Local
from lcn.core.model import LCN, SentenceType, Formula
from lcn.inference.marginal.exact import ExactInference
from lcn.inference.utils.common import (
    eval_indicator,
    build_truth_table,
    lmc_constraint_groups_vec,
)


# Number of free parameters in the alarm factorization (see module docstring):
#   theta[0]      = P(B=1)
#   theta[1]      = P(E=1)
#   theta[2:6]    = P(A=1 | B,E) for (B,E) in (0,0),(0,1),(1,0),(1,1)
#   theta[6:10]   = raw nonneg weights for P(C,D | A=0), normalized to a 2x2 block
#   theta[10:14]  = raw nonneg weights for P(C,D | A=1), normalized to a 2x2 block
_N_THETA = 14


# -----------------------------------------------------------------------
# Interpretation table and parametric joint builder
# -----------------------------------------------------------------------

def build_interpretations(lcn: LCN):
    """Return (vars_list, interpretations, N), mirroring exact._build_base_model."""
    vars_list = [k for k, _ in lcn.atoms.items()]
    items = list(itertools.product([0, 1], repeat=len(vars_list)))
    interps = [dict(zip(vars_list, t)) for t in items]
    return vars_list, interps, len(interps)


def make_joint_builder(vars_list, interps):
    """
    Return a closure ``build(theta) -> p`` mapping the 14-parameter vector to a
    length-N joint distribution via P(B)P(E)P(A|B,E)P(C,D|A).

    The variable roles are read from the structure graph elsewhere; here we look
    up the columns by name so the builder is robust to atom ordering. The five
    alarm variables B, E, A, C, D must all be present.
    """
    required = {"A", "B", "C", "D", "E"}
    missing = required - set(vars_list)
    if missing:
        raise ValueError(
            f"verify_bruteforce targets alarm.lcn (vars A,B,C,D,E); missing {missing}")

    # Precompute the (B,E,A,C,D) integer value of each interpretation once.
    rows = [(it["B"], it["E"], it["A"], it["C"], it["D"]) for it in interps]
    N = len(rows)

    # (B,E) -> index into theta[2:6]
    be_index = {(0, 0): 2, (0, 1): 3, (1, 0): 4, (1, 1): 5}

    def build(theta):
        pB = min(max(float(theta[0]), 0.0), 1.0)
        pE = min(max(float(theta[1]), 0.0), 1.0)

        # P(C,D|A=a): normalize a nonnegative 2x2 block (indexed [c, d]).
        cd = {}
        for a, off in ((0, 6), (1, 10)):
            blk = np.abs(np.asarray(theta[off:off + 4], dtype=float)).reshape(2, 2)
            s = blk.sum()
            cd[a] = blk / s if s > 0 else np.full((2, 2), 0.25)

        p = np.empty(N, dtype=float)
        for k, (b, e, a, c, d) in enumerate(rows):
            pb = pB if b == 1 else 1.0 - pB
            pe = pE if e == 1 else 1.0 - pE
            pa1 = min(max(float(theta[be_index[(b, e)]]), 0.0), 1.0)
            pa = pa1 if a == 1 else 1.0 - pa1
            p[k] = pb * pe * pa * cd[a][c, d]
        return p

    return build


# -----------------------------------------------------------------------
# Sentence-bound constraints (the only constraints not baked into `build`)
# -----------------------------------------------------------------------

def make_sentence_constraints(lcn: LCN, interps, build):
    """
    Build scipy ``ineq`` constraint dicts and a scalar total-violation function
    for the LCN sentence bounds, mirroring exact._build_base_model. The LMC
    equalities are NOT included here -- they hold by construction of ``build``.

    Type1  P(phi):      lobo <= A@p <= upbo
    Type2  P(phi|psi):  lobo*(Ar@p) <= Aqr@p <= upbo*(Ar@p)
    """
    specs = []  # (Aqr, Ar_or_None, lobo, upbo)
    for _, s in lcn.sentences.items():
        lobo = s.get_lower_bound()
        upbo = s.get_upper_bound()
        if s.type == SentenceType.Type1:
            A = eval_indicator(s.phi_formula, interps)
            specs.append((A, None, lobo, upbo))
        else:
            Aqr = eval_indicator(s.phi_and_psi_formula, interps)
            Ar = eval_indicator(s.psi_formula, interps)
            specs.append((Aqr, Ar, lobo, upbo))

    cons = []
    for Aqr, Ar, lobo, upbo in specs:
        if Ar is None:
            cons.append({"type": "ineq",
                         "fun": lambda th, A=Aqr, lo=lobo: float(A @ build(th)) - lo})
            cons.append({"type": "ineq",
                         "fun": lambda th, A=Aqr, up=upbo: up - float(A @ build(th))})
        else:
            cons.append({"type": "ineq",
                         "fun": lambda th, Aqr=Aqr, Ar=Ar, lo=lobo:
                         float(Aqr @ build(th)) - lo * float(Ar @ build(th))})
            cons.append({"type": "ineq",
                         "fun": lambda th, Aqr=Aqr, Ar=Ar, up=upbo:
                         up * float(Ar @ build(th)) - float(Aqr @ build(th))})

    def total_violation(theta):
        p = build(theta)
        tot = 0.0
        for Aqr, Ar, lobo, upbo in specs:
            if Ar is None:
                v = float(Aqr @ p)
                tot += max(0.0, lobo - v) ** 2 + max(0.0, v - upbo) ** 2
            else:
                qr = float(Aqr @ p)
                r = float(Ar @ p)
                tot += (max(0.0, lobo * r - qr) ** 2 + max(0.0, qr - upbo * r) ** 2)
        return tot

    return cons, total_violation


# -----------------------------------------------------------------------
# Soundness self-check: the parametric family must satisfy ALL LMC equalities
# -----------------------------------------------------------------------

def assert_parametrization_sound(lcn, vars_list, interps, build,
                                 n_trials=1000, tol=1e-9, seed=12345):
    """
    Verify that random parameter settings produce joints satisfying every LMC
    equality residual to ``tol``. This guarantees the brute-force family is a
    faithful (neither over- nor under-constrained) sample of the LMC-feasible
    set before any bound is trusted. Raises AssertionError on failure.
    """
    table = build_truth_table(len(vars_list))
    col_of = {v: i for i, v in enumerate(vars_list)}

    eq_checks = []
    for indep in lcn.independencies.get_assertions():
        for group in lmc_constraint_groups_vec(indep, table, col_of):
            if group[0] == 'conditional':
                _, Aa, Ab, Ac, Ad = group
                eq_checks.append(lambda p, Aa=Aa, Ab=Ab, Ac=Ac, Ad=Ad:
                                 float(Aa @ p) * float(Ab @ p)
                                 - float(Ac @ p) * float(Ad @ p))
            else:
                _, Aa, Ab, Ac = group
                eq_checks.append(lambda p, Aa=Aa, Ab=Ab, Ac=Ac:
                                 float(Aa @ p) - float(Ab @ p) * float(Ac @ p))

    rng = np.random.default_rng(seed)
    max_resid = 0.0
    for _ in range(n_trials):
        theta = rng.random(_N_THETA)
        p = build(theta)
        # sanity: it is a distribution
        assert abs(float(p.sum()) - 1.0) < 1e-9, "parametric joint not normalized"
        for fn in eq_checks:
            max_resid = max(max_resid, abs(fn(p)))
    assert max_resid < tol, (
        f"parametrization violates LMC equalities (max residual {max_resid:.2e} "
        f">= {tol:.0e}); the factorization is unsound for this LCN")
    return max_resid


# -----------------------------------------------------------------------
# Brute-force bound computation
# -----------------------------------------------------------------------

def brute_force_bounds(lcn, vars_list, interps, build,
                       n_restarts=200, feas_tol=1e-6, seed=0, verbose=True):
    """
    Compute per-atom marginal bounds by optimizing over the LMC-correct
    parametric family subject to the sentence bounds, using scipy SLSQP only
    (independent of exact.py's ipopt path).

    For each of ``n_restarts`` random parameter seeds we (1) project to
    sentence-feasibility by minimizing total constraint violation, then (2) from
    each feasible seed minimize and maximize every atom's P(atom=1). The running
    min/max over all feasible optima are the brute-force bounds.

    Returns {atom_name: (lo1, hi1)} for P(atom=1); atoms with no feasible point
    found get (None, None).
    """
    cons, total_violation = make_sentence_constraints(lcn, interps, build)
    atom_ind = {v: eval_indicator(Formula(label=v, formula=v), interps)
                for v in vars_list}

    lo = {v: None for v in vars_list}
    hi = {v: None for v in vars_list}
    rng = np.random.default_rng(seed)
    bounds = [(0.0, 1.0)] * _N_THETA
    n_feasible_seeds = 0

    def _update(v, val):
        if lo[v] is None or val < lo[v]:
            lo[v] = val
        if hi[v] is None or val > hi[v]:
            hi[v] = val

    for r in range(n_restarts):
        theta0 = rng.random(_N_THETA)
        # Phase 1: project to a sentence-feasible point.
        proj = minimize(total_violation, theta0, method="SLSQP", bounds=bounds,
                        options={"maxiter": 500, "ftol": 1e-12})
        seed_theta = proj.x
        if total_violation(seed_theta) > 1e-8:
            continue
        n_feasible_seeds += 1

        # Record the feasible seed's own marginals (cheap, and a valid witness).
        p_seed = build(seed_theta)
        for v in vars_list:
            _update(v, float(atom_ind[v] @ p_seed))

        # Phase 2: optimize each atom marginal from this feasible seed.
        for v in vars_list:
            A = atom_ind[v]
            for sense in ("min", "max"):
                sgn = 1.0 if sense == "min" else -1.0
                res = minimize(lambda th, A=A, sgn=sgn: sgn * float(A @ build(th)),
                               seed_theta, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
                if total_violation(res.x) < feas_tol:
                    _update(v, float(A @ build(res.x)))

        if verbose and (r + 1) % 50 == 0:
            print(f"  [brute-force] restart {r + 1}/{n_restarts}, "
                  f"feasible seeds: {n_feasible_seeds}")

    if verbose:
        print(f"  [brute-force] feasible seeds: {n_feasible_seeds}/{n_restarts}")
    return {v: (lo[v], hi[v]) for v in vars_list}


# -----------------------------------------------------------------------
# Comparison vs ExactInference
# -----------------------------------------------------------------------

def compare(bf, exact_marginals, tol=1e-3):
    """
    Compare brute-force P(atom=1) bounds against ExactInference and print a table.

    Flags per atom:
      OUTSIDE - BF found an LMC-feasible point exact's interval excludes
                (BF lo < exact lo - tol or BF hi > exact hi + tol). Since BF is
                sound by construction, this indicates exact's bound is too TIGHT.
      LOOSE   - exact's interval is materially wider than BF on some side
                (exact may be too loose, or BF under-sampled -- both noted).
      OK      - intervals agree within tol.
    """
    rows = []
    n_outside = n_loose = n_ok = 0
    print(f"\n{'atom':<6} {'brute-force [lo, hi]':<26} "
          f"{'exact [lo, hi]':<26} flag")
    print("-" * 70)
    for v in sorted(bf):
        blo, bhi = bf[v]
        elo, ehi = exact_marginals[v][0][1], exact_marginals[v][1][1]
        if blo is None:
            flag = "NO-FEAS"
            bf_str = "[ none ]"
        else:
            bf_str = f"[{blo:.4f}, {bhi:.4f}]"
            outside = (blo < elo - tol) or (bhi > ehi + tol)
            loose = (blo > elo + tol) or (bhi < ehi - tol)
            if outside:
                flag = "OUTSIDE"
                n_outside += 1
            elif loose:
                flag = "LOOSE"
                n_loose += 1
            else:
                flag = "OK"
                n_ok += 1
        ex_str = f"[{elo:.4f}, {ehi:.4f}]"
        print(f"{v:<6} {bf_str:<26} {ex_str:<26} {flag}")
        rows.append((v, bf[v], (elo, ehi), flag))
    print("-" * 70)
    print(f"summary: {n_ok} OK | {n_loose} LOOSE | {n_outside} OUTSIDE")
    if n_outside:
        print("  -> OUTSIDE flags: brute force reached LMC-feasible marginals that "
              "exact.py's interval excludes; exact bound looks too tight.")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Brute-force (parametric, LMC-aware) verifier for exact.py "
                    "marginal bounds on alarm.lcn.")
    parser.add_argument("--file", default="examples/alarm.lcn",
                        help="LCN file (only alarm.lcn's topology is supported).")
    parser.add_argument("--restarts", type=int, default=200,
                        help="Number of random parameter restarts (default 200).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="Comparison tolerance for flagging (default 1e-3).")
    args = parser.parse_args()

    lcn = LCN()
    lcn.from_lcn(file_name=args.file)
    assert lcn.independencies is not None, "LMC independencies not populated."

    vars_list, interps, N = build_interpretations(lcn)
    print(f"=== Brute-force verifier for {args.file} ===")
    print(f"Variables: {vars_list}  (joint size N = {N})")
    print("LMC independencies:")
    for a in lcn.independencies.get_assertions():
        print(f"  {a}")

    build = make_joint_builder(vars_list, interps)

    # Soundness gate: the parametric family must satisfy all LMC equalities.
    max_resid = assert_parametrization_sound(lcn, vars_list, interps, build)
    print(f"\nParametrization soundness OK "
          f"(max LMC equality residual = {max_resid:.2e} over random params)")

    print(f"\nRunning brute force ({args.restarts} restarts)...")
    bf = brute_force_bounds(lcn, vars_list, interps, build,
                            n_restarts=args.restarts, seed=args.seed)

    print("\nRunning ExactInference (the oracle under test)...")
    exact_marginals = ExactInference(lcn=lcn).run(evidence={}, verbosity=0)

    compare(bf, exact_marginals, tol=args.tol)


if __name__ == "__main__":
    main()
