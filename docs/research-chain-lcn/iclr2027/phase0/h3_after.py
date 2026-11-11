"""Post-fix H3 census: per benchmark, how many instances CJT is structurally
exact on (H1-H4 hold), how much widening that cost, and max cluster growth."""
import sys, glob, io, contextlib, collections
sys.path.insert(0, '/Users/radu/git/IBM/LCN')
from lcn.core.model import LCN
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.marginal.cn import junction_nlp as jn

def width_without_widening(cnv):
    cc = jn.CouplingConstraints.from_lcn(cnv.lcn, cnv.cn.factorization.factors)
    ext = cc.constraint_node_sets(cnv.cn.node_atoms, kinds=("type1", "type2"))
    return jn._JTClusters(cnv, None, {}, ext).max_cluster_size()

def main():
    budget = int(sys.argv[1])
    for bdir in sys.argv[2:]:
        c = collections.Counter(); grow = []
        for f in sorted(glob.glob(f'{bdir}/*.lcn')):
            with contextlib.redirect_stdout(io.StringIO()):
                l = LCN(); l.from_lcn(f)
                cnv = CredalNetworkVertices.from_lcn(l, enumerate_vertices=False, solve_families=False, cache=False, verbosity=0)
                w0 = width_without_widening(cnv)
                jt, sbh, lbh, qc, over, fb = jn._build_jt_and_hosts(cnv, None, {}, budget)
            h = jt.h3; c['inst'] += 1
            c['exact'] += bool(h.get('holds')) and not over and not fb
            c['widened_inst'] += h.get('widened', 0) > 0
            c['relaxed_inst'] += h.get('relaxed', 0) > 0
            c['over'] += over; c['fb'] += fb
            grow.append(jt.max_cluster_size() - w0); c['mode_'+h.get('mode','?')] += 1
        print(f"{bdir.split('/')[-1]:18s} inst={c['inst']:3d} H3-exact={c['exact']:3d} "
              f"needed-widening={c['widened_inst']:3d} still-relaxed={c['relaxed_inst']:3d} "
              f"over-budget={c['over']} max-width-growth={max(grow)} mean-growth={sum(grow)/len(grow):.2f} modes={ {k[5:]:v for k,v in c.items() if k.startswith('mode_')} }", flush=True)

if __name__ == '__main__':
    main()
