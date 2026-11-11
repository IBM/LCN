import contextlib, io, random, sys
sys.path.insert(0, '/Users/radu/git/IBM/LCN')
from lcn.benchmarks.generator import Generator
from lcn.inference.marginal.cn import junction_nlp as jn
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.marginal.exact import ExactInference
if __name__ == '__main__':
    rng = random.Random(0)
    # replay the RNG draws of cjt_verify_ev.py up to polytree-fr k=1
    with contextlib.redirect_stdout(io.StringIO()):
        t = Generator(seed=11).generate(num_vars=5, graph_type="tree-fr", num_instances=2, num_extras=2, verbosity=0)
        p = Generator(seed=11).generate(num_vars=5, graph_type="polytree-fr", num_instances=2, num_extras=2, verbosity=0)
    for lcn in t + [p[0]]:
        rng.choice(sorted(lcn.atoms)); rng.randint(0, 1)
    lcn = p[1]; atom = rng.choice(sorted(lcn.atoms)); ev = {atom: rng.randint(0, 1)}
    print("evidence", ev)
    for s in lcn.sentences.values(): print("  ", s)
    with contextlib.redirect_stdout(io.StringIO()):
        cnv = CredalNetworkVertices.from_lcn(lcn, solve_families=False, enumerate_vertices=False, cache=False, verbosity=0)
        e = jn.CredalJT(cnv); e.run(evidence=ev, solver="scip", time_limit=15, verbosity=0)
        ei = ExactInference(lcn); ei.run(evidence=ev, solver="global", verbosity=0, progress_bar=False, time_limit=15)
    for a in sorted(e.singleton_marginals):
        print(a, "CJT", e.singleton_marginals[a], e.atom_status[a])
        print("   direct", ei.status.get(a))
    for s in e.solve_status: print("   cjt-solve", {k: s[k] for k in ('atom','sense','value','primal','dual','termination','certified')})
