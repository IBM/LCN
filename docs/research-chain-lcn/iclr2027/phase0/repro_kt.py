import contextlib, io, sys
sys.path.insert(0, '/Users/radu/git/IBM/LCN')
from lcn.benchmarks.generator import Generator
from lcn.inference.marginal.cn import junction_nlp as jn
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.marginal.exact import ExactInference
if __name__ == '__main__':
    with contextlib.redirect_stdout(io.StringIO()):
        k = Generator(seed=11).generate(num_vars=5, graph_type="ktree-fr", k=2, num_instances=2, num_extras=2, verbosity=0)
    lcn = k[0]; ev = {"x3": 1}
    for s in lcn.sentences.values(): print("  ", s)
    with contextlib.redirect_stdout(io.StringIO()):
        cnv = CredalNetworkVertices.from_lcn(lcn, solve_families=False, enumerate_vertices=False, cache=False, verbosity=0)
        e = jn.CredalJT(cnv); e.run(evidence=ev, solver="scip", time_limit=30, verbosity=0)
        ei = ExactInference(lcn); ei.run(evidence=ev, solver="global", verbosity=0, progress_bar=False, time_limit=30)
    for a in sorted(e.singleton_marginals):
        print(a, "CJT", tuple(round(x,6) for x in e.singleton_marginals[a]), e.atom_status[a]['certified'], "| direct", {s:(ei.status[a][s][0], ei.status[a][s][2]) for s in ('min','max')})
