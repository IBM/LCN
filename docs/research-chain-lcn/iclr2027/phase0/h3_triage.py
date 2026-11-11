"""Phase-0 (S5/E2): per benchmark, how often does CJT drop an LMC assertion
(neither hosted nor separator-entailed, i.e. H3 fails), and how often does a
dropped assertion have a sub-independence (decomposition) that WOULD be
hostable -- the 'silently lost hostable independence' case."""
import sys, glob, io, contextlib, itertools, collections
sys.path.insert(0, '/Users/radu/git/IBM/LCN')
from lcn.core.model import LCN
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.marginal.cn.junction_nlp import _build_jt_and_hosts

def triage(path):
    with contextlib.redirect_stdout(io.StringIO()):
        l = LCN(); l.from_lcn(path)
        cnv = CredalNetworkVertices.from_lcn(l, enumerate_vertices=False,
                                             solve_families=False, cache=False,
                                             verbosity=0)
        jt, sbh, lbh, qc, over, fb = _build_jt_and_hosts(cnv, None, {}, 16)
    if over:
        return dict(over=1)
    n = ent = host = drop = drop_pair_hostable = 0
    for a in l.independencies.get_assertions():
        X, Y, Z = sorted(a.event1), sorted(a.event2), sorted(a.event3)
        if not (set(X) | set(Y) | set(Z)):
            continue
        n += 1
        if jt._rip_implied(X, Y, Z): ent += 1; continue
        if jt.host(sorted(set(X) | set(Y) | set(Z))) is not None: host += 1; continue
        drop += 1
        # any pairwise sub-independence x _|_ y | Z that is hostable and not entailed?
        if any(jt.host(sorted(set(X) | {y} | set(Z))) is not None
               and not jt._rip_implied(X, [y], Z) for y in Y):
            drop_pair_hostable += 1
    return dict(n=n, ent=ent, host=host, drop=drop, lost=drop_pair_hostable,
                fallback=int(fb), maxclu=jt.max_cluster_size())

def main():
    for bdir in sys.argv[1:]:
        agg = collections.Counter(); inst_drop = inst_lost = k = 0
        for f in sorted(glob.glob(f'{bdir}/*.lcn')):
            try: r = triage(f)
            except Exception as ex:
                agg['err'] += 1; continue
            k += 1; agg.update(r)
            inst_drop += r.get('drop', 0) > 0; inst_lost += r.get('lost', 0) > 0
        print(f"{bdir.split('/')[-1]:18s} inst={k:3d} assertions={agg['n']:5d} "
              f"entailed={agg['ent']:5d} hosted={agg['host']:4d} dropped={agg['drop']:4d} "
              f"| inst w/ drop={inst_drop:3d} inst w/ lost hostable sub-indep={inst_lost:3d} "
              f"over_budget={agg['over']} sentence_fallback={agg['fallback']} err={agg['err']}", flush=True)

if __name__ == '__main__':
    main()
