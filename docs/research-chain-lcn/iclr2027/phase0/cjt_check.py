"""Run the fixed CredalJT on given instances; print marginals + guarantees."""
import sys, io, contextlib, json
sys.path.insert(0, '/Users/radu/git/IBM/LCN')
from lcn.core.model import LCN
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.marginal.cn.junction_nlp import CredalJT

def run(path, ev, solver='scip', tl=600):
    with contextlib.redirect_stdout(io.StringIO()):
        l = LCN(); l.from_lcn(path)
        cnv = CredalNetworkVertices.from_lcn(l, solve_families=False, enumerate_vertices=False, cache=False, verbosity=0)
        jt = CredalJT(cnv); jt.run(evidence=ev, solver=solver, time_limit=tl, verbosity=0)
    return jt

if __name__ == '__main__':
    ev = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    jt = run(sys.argv[1], ev)
    print({a: (round(lo, 4), round(hi, 4)) for a, (lo, hi) in sorted(jt.singleton_marginals.items())})
    print("h3", jt.h3, "| maxclu", jt.induced_width, "| d5_exact", jt.d5_exact, "certified", jt.certified, "EXACT", jt.exact)
