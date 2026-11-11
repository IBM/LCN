"""Prototype: ordered-Markov junction tree. Eliminate nodes in REVERSE topological
order of the chain-graph node DAG; for every node x the elimination clique is
{x} u N_x with N_x subset of predecessors, and we impose (x _|_ N_x\\pa(x) | pa(x)),
checking it is ENTAILED by an LCN LMC assertion (soundness). Report width vs the
min-fill + widening tree."""
import sys, glob, io, contextlib, collections
sys.path.insert(0, '/Users/radu/git/IBM/LCN')
import networkx as nx
from lcn.core.model import LCN
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.marginal.cn import junction_nlp as jn

def node_dag(cnv):
    G = nx.DiGraph()
    for f in cnv.cn.factors:
        e = f[0]; G.add_node(e['child'])
        for p in e['parents']: G.add_edge(p, e['child'])
    return G

def ordered_width(cnv, extra_node_sets):
    G = node_dag(cnv); na = cnv.cn.node_atoms
    topo = list(nx.lexicographical_topological_sort(G))
    # interaction graph over nodes: family scopes + cross-family sentence node sets
    adj = {v: set() for v in G}
    scopes = [[x] + list(G.predecessors(x)) for x in G] + [list(s) for s in extra_node_sets]
    for s in scopes:
        for u in s:
            adj[u].update(w for w in s if w != u)
    width = 0; cliques = {}
    for x in reversed(topo):           # eliminate last-in-topo first
        nb = set(adj[x])
        cliques[x] = nb
        width = max(width, sum(len(na[v]) for v in nb | {x}))
        for u in nb:
            adj[u].update(nb - {u}); adj[u].discard(x)
        del adj[x]
    return width, cliques, G, topo

def entailed(lcn, x_atoms, rest_atoms, cond_atoms):
    """Is x _|_ rest | cond implied (by decomposition / the same conditioning set)
    by some LMC assertion of the LCN for an atom of x?  Conservative check:
    an assertion X'_|_Y'|Z' with X' subset x, Z' == cond, rest subset Y'."""
    if not rest_atoms: return True
    for a in lcn.independencies.get_assertions():
        E1, E2, E3 = set(a.event1), set(a.event2), set(a.event3)
        if E3 != cond_atoms: continue
        # X' _|_ Y' | Z  (decomposition on both sides; symmetry)
        if x_atoms <= E1 and rest_atoms <= E2: return True
        if x_atoms <= E2 and rest_atoms <= E1: return True
    # pairwise symmetric cover: x _|_ r | cond for each r from some r's own LMC
    # is NOT sufficient (pairwise != joint) -- only accept a single joint cover.
    return False

def main():
    for bdir in sys.argv[1:]:
        c = collections.Counter(); wo = []; ww = []
        for f in sorted(glob.glob(f'{bdir}/*.lcn')):
            with contextlib.redirect_stdout(io.StringIO()):
                l = LCN(); l.from_lcn(f)
                cnv = CredalNetworkVertices.from_lcn(l, enumerate_vertices=False, solve_families=False, cache=False, verbosity=0)
                cc = jn.CouplingConstraints.from_lcn(l, cnv.cn.factorization.factors)
                ext = cc.constraint_node_sets(cnv.cn.node_atoms, kinds=("type1", "type2"))
                jt, *_ = jn._build_jt_and_hosts(cnv, None, {}, 64)
            w, cl, G, topo = ordered_width(cnv, ext)
            na = cnv.cn.node_atoms
            ok = True
            for x, nb in cl.items():
                pa = set(G.predecessors(x)); extra = nb - pa
                xa = set(na[x]); ra = {a for v in extra for a in na[v]}; za = {a for v in pa for a in na[v]}
                if not entailed(l, xa, ra, za): ok = False
            c['inst'] += 1; c['sound'] += ok
            wo.append(w); ww.append(jt.max_cluster_size())
        print(f"{bdir.split('/')[-1]:18s} inst={c['inst']} ordered-entailed={c['sound']} "
              f"max width ordered={max(wo)} (mean {sum(wo)/len(wo):.1f}) | minfill+widen={max(ww)} (mean {sum(ww)/len(ww):.1f})", flush=True)

if __name__ == '__main__':
    main()
