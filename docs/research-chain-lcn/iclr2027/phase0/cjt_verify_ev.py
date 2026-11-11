"""Oracle check of CredalJT conditional queries (Charnes-Cooper form): compare
CJT's certified endpoints of P(a | e) against the direct 2^n global solver's
certified endpoints on random small LCNs with one random evidence atom."""
import contextlib
import io
import json
import random
import sys

sys.path.insert(0, '/Users/radu/git/IBM/LCN')
from lcn.benchmarks.generator import Generator  # noqa: E402
from lcn.inference.marginal.cn import junction_nlp as jn  # noqa: E402
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices  # noqa: E402
from lcn.inference.marginal.exact import ExactInference  # noqa: E402

TOL = 1e-3   # conditional values; the direct solver uses a 1e-6 denominator floor


def main():
    n_inst, tl = int(sys.argv[1]), float(sys.argv[2])
    rng = random.Random(0)
    configs = [("tree-fr", {}), ("polytree-fr", {}), ("ktree-fr", {"k": 2}),
               ("dag-fr", {"max_treewidth": 3}), ("chain", {}), ("random", {})]
    bad = n_total = 0
    for gt, kw in configs:
        with contextlib.redirect_stdout(io.StringIO()):
            insts = Generator(seed=11).generate(
                num_vars=5, graph_type=gt, num_instances=n_inst, num_extras=2,
                verbosity=0, **kw)
        for k, lcn in enumerate(insts):
            atom = rng.choice(sorted(lcn.atoms))
            ev = {atom: rng.randint(0, 1)}
            with contextlib.redirect_stdout(io.StringIO()):
                cnv = CredalNetworkVertices.from_lcn(
                    lcn, solve_families=False, enumerate_vertices=False,
                    cache=False, verbosity=0)
                e = jn.CredalJT(cnv)
                e.run(evidence=ev, solver="scip", time_limit=tl, verbosity=0)
                ei = ExactInference(lcn)
                ei.run(evidence=ev, solver="global", verbosity=0,
                       progress_bar=False, time_limit=tl)
            cert = {(s['atom'], s['sense']): s['certified'] for s in e.solve_status}
            worst, ncmp = 0.0, 0
            for a, (lo, hi) in e.singleton_marginals.items():
                st = ei.status.get(a)
                if not st:
                    continue
                for sense, mine in (('min', lo), ('max', hi)):
                    v, _g, status, _t = st[sense]
                    if status == 'confirmed' and cert.get((a, sense)) and v is not None:
                        worst = max(worst, abs(mine - v)); ncmp += 1
            n_total += 1
            row = dict(gt=gt, k=k, ev=ev, maxdiff=round(worst, 6), n_cmp=ncmp,
                       routes=sorted({s['route'] for s in e.atom_status.values()}))
            bad += worst > TOL
            print(json.dumps(row), flush=True)
    print("SUMMARY instances", n_total, "bad", bad)


if __name__ == '__main__':
    main()
