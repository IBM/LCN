import contextlib, io, random, sys
sys.path.insert(0, '/Users/radu/git/IBM/LCN')
from pyomo.environ import Objective, minimize, value
from lcn.benchmarks.generator import Generator
from lcn.inference.marginal.cn import junction_nlp as jn
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.marginal.exact import make_scip
if __name__ == '__main__':
    with contextlib.redirect_stdout(io.StringIO()):
        p = Generator(seed=11).generate(num_vars=5, graph_type="polytree-fr", num_instances=2, num_extras=2, verbosity=0)
    lcn = p[1]; ev = {"x3": 1}
    with contextlib.redirect_stdout(io.StringIO()):
        cnv = CredalNetworkVertices.from_lcn(lcn, solve_families=False, enumerate_vertices=False, cache=False, verbosity=0)
        jt, sbh, lbh, _, o, fb = jn._build_jt_and_hosts(cnv, None, ev, 16)
    print("tree", {c: jt.atoms[c] for c in jt.cluster_ids}, "edges", [(c,p,jt.sep[(c,p)]) for c,p in jt.edges])
    print("hosts", sbh, {k:[str(a) for a in v] for k,v in lbh.items()})
    for evd in [{}, ev]:
        m, cs = jn._build_constraint_model(jt, cnv, sbh, lbh, evidence=evd)
        m.objective = Objective(expr=0.0, sense=minimize)
        r = make_scip(time_limit=20).solve(m, load_solutions=False)
        print("evidence", evd, "->", r.solver.termination_condition, "ev_cluster", getattr(m, 'ev_cluster', None))
