"""Independent verification of CredalJT on random small LCNs.

Two checks per instance:

(A) Reconstruction check (tests H1-H3 on the code's actual tree): solve the
    cluster program with a random linear objective, reconstruct a joint from the
    optimal cluster marginals with the RIP construction of Lemma rip, and verify
    numerically that it satisfies every LCN sentence and every LMC assertion.
    A violation means the cluster program admits a point outside D_LCN, i.e. CJT
    is NOT exact (an over-wide bound).

(B) Oracle check: compare CJT's certified endpoints against the direct 2^n
    global formulation's certified endpoints; flag any disagreement > tol.
"""
import contextlib
import io
import itertools
import json
import sys
import time

import numpy as np
from pyomo.environ import Objective, minimize, value

sys.path.insert(0, '/Users/radu/git/IBM/LCN')
from lcn.benchmarks.generator import Generator  # noqa: E402
from lcn.inference.marginal.cn import junction_nlp as jn  # noqa: E402
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices  # noqa: E402
from lcn.inference.marginal.exact import ExactInference, make_scip  # noqa: E402
from lcn.inference.utils.common import build_truth_table, eval_indicator  # noqa: E402
from lcn.core.model import SentenceType  # noqa: E402

TOL = 1e-4


def reconstruct(jt, qvals):
    """Joint over jt.atom_order from cluster marginals (Lemma rip), by leaf
    peeling from the root: p(x) = q_root * prod_{C != root} q_C / q_{S_C}."""
    order = jt.atom_order
    n = len(order)
    joint = np.ones(2 ** n)
    states = np.array(list(itertools.product([0, 1], repeat=n)))  # MSB-first

    def idx(atoms):
        cols = [order.index(a) for a in atoms]
        sub = states[:, cols]
        w = 2 ** np.arange(len(atoms))[::-1]
        return sub @ w if atoms else np.zeros(len(states), dtype=int)

    for cid in jt.cluster_ids:
        joint *= qvals[cid][idx(jt.atoms[cid])]
    for (c, p) in jt.edges:
        S = jt.sep[(c, p)]
        if not S:
            continue
        Mc = jn._marginal_matrix(jt.atoms[c], S)
        qs = Mc @ qvals[c]
        den = qs[idx(S)]
        with np.errstate(divide='ignore', invalid='ignore'):
            joint = np.where(den > 0, joint / np.where(den > 0, den, 1), 0.0)
    # Disconnected components (empty separators / several roots) multiply
    # independently, which is already the case above.
    return joint, states


def check_joint(lcn, joint, states, order):
    """Max violation of sentences and LMC assertions by the joint."""
    col = {a: i for i, a in enumerate(order)}
    interps = [dict(zip(order, row)) for row in states]
    worst = 0.0
    tot = joint.sum()
    worst = max(worst, abs(tot - 1))
    for s in lcn.sentences.values():
        lo, hi = s.get_lower_bound(), s.get_upper_bound()
        if s.type == SentenceType.Type1:
            v = float(eval_indicator(s.phi_formula, interps) @ joint)
            worst = max(worst, lo - v, v - hi)
        else:
            a = float(eval_indicator(s.phi_and_psi_formula, interps) @ joint)
            b = float(eval_indicator(s.psi_formula, interps) @ joint)
            worst = max(worst, lo * b - a, a - hi * b)
    for ind in lcn.independencies.get_assertions():
        X, Y, Z = sorted(ind.event1), sorted(ind.event2), sorted(ind.event3)
        allv = X + Y + Z
        for cfg in itertools.product([0, 1], repeat=len(allv)):
            asg = dict(zip(allv, cfg))
            def P(vs):
                m = np.ones(len(states), dtype=bool)
                for v in vs:
                    m &= states[:, col[v]] == asg[v]
                return float(joint[m].sum())
            lhs = P(X + Y + Z) * P(Z)
            rhs = P(X + Z) * P(Y + Z)
            worst = max(worst, abs(lhs - rhs))
    return worst


def build(lcn):
    with contextlib.redirect_stdout(io.StringIO()):
        cnv = CredalNetworkVertices.from_lcn(
            lcn, method="linear", solve_families=False,
            enumerate_vertices=False, cache=False, verbosity=0)
        jt, sbh, lbh, _qc, over, fb = jn._build_jt_and_hosts(cnv, None, {}, 16)
    return cnv, jt, sbh, lbh, over, fb


def reconstruction_check(lcn, cnv, jt, sbh, lbh, rng, n_obj=3):
    worst = 0.0
    for _ in range(n_obj):
        with contextlib.redirect_stdout(io.StringIO()):
            m, cs = jn._build_constraint_model(jt, cnv, sbh, lbh)
            obj = sum(float(rng.normal()) * m.q[(c, i)]
                      for c in jt.cluster_ids for i in range(cs[c]))
            m.objective = Objective(expr=obj, sense=minimize)
            res = make_scip(time_limit=30).solve(m, load_solutions=False)
            try:
                m.solutions.load_from(res)
            except Exception:
                continue
        q = {c: np.array([value(m.q[(c, i)]) for i in range(cs[c])])
             for c in jt.cluster_ids}
        if any(v is None or np.isnan(v).any() for v in q.values()):
            continue
        joint, states = reconstruct(jt, q)
        worst = max(worst, check_joint(lcn, joint, states, jt.atom_order))
    return worst


def oracle_check(lcn, cnv, tl):
    with contextlib.redirect_stdout(io.StringIO()):
        e = jn.CredalJT(cnv)
        e.run(evidence={}, solver="scip", time_limit=tl, verbosity=0)
        ei = ExactInference(lcn)
        ei.run(evidence={}, solver="global", verbosity=0, progress_bar=False,
               time_limit=tl)
    worst, n_cmp = 0.0, 0
    cert = {(s['atom'], s['sense']): s for s in e.solve_status}
    for a, (lo, hi) in e.singleton_marginals.items():
        st = ei.status.get(a)
        if st is None:
            continue
        for sense, mine in (('min', lo), ('max', hi)):
            v, _gap, status, _t = st[sense]
            c = cert.get((a, sense))
            if status == 'confirmed' and c and c['certified'] and v is not None:
                worst = max(worst, abs(mine - v))
                n_cmp += 1
    return worst, n_cmp, e


def main():
    n_inst = int(sys.argv[1])
    tl = float(sys.argv[2]) if len(sys.argv) > 2 else 30
    rng = np.random.default_rng(0)
    configs = [("tree-fr", {}), ("polytree-fr", {}), ("ktree-fr", {"k": 2}),
               ("dag-fr", {"max_treewidth": 3}), ("dag", {"max_treewidth": 3}),
               ("polytree", {}), ("chain", {}), ("random", {})]
    summary = []
    for gt, kw in configs:
        g = Generator(seed=7)
        with contextlib.redirect_stdout(io.StringIO()):
            insts = g.generate(num_vars=5, graph_type=gt, num_instances=n_inst,
                               num_extras=2, verbosity=0, **kw)
        for k, lcn in enumerate(insts):
            try:
                cnv, jt, sbh, lbh, over, fb = build(lcn)
            except Exception as ex:
                summary.append(dict(gt=gt, k=k, err=f"build: {ex}"[:80]))
                continue
            if over or fb:
                summary.append(dict(gt=gt, k=k, skip="fallback"))
                continue
            t0 = time.time()
            rec = reconstruction_check(lcn, cnv, jt, sbh, lbh, rng)
            ow, ncmp, e = oracle_check(lcn, cnv, tl)
            row = dict(gt=gt, k=k, mode=jt.h3.get('mode'), holds=jt.h3.get('holds'),
                       width=jt.max_cluster_size(), recon_viol=round(rec, 7),
                       oracle_maxdiff=round(ow, 7), n_cmp=ncmp,
                       cjt_cert=e.certified, secs=round(time.time() - t0, 1))
            summary.append(row)
            print(json.dumps(row), flush=True)
    bad = [r for r in summary if r.get('recon_viol', 0) > TOL or r.get('oracle_maxdiff', 0) > TOL]
    print("SUMMARY instances", len(summary), "bad", len(bad))
    for r in bad:
        print("BAD", r)


if __name__ == '__main__':
    main()
