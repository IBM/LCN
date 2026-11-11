"""Does _rip_implied accept X _|_ Y | Z using a separator S strictly larger than Z?
A separator S only enforces (sideA \ S) _|_ (sideB \ S) | S; with S != Z the claim
X _|_ Y | Z is NOT implied in general."""
import sys, glob, io, contextlib, collections
sys.path.insert(0, '/Users/radu/git/IBM/LCN')
from lcn.core.model import LCN
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.marginal.cn.junction_nlp import _build_jt_and_hosts

def witness(jt, X, Y, Z):
    Xs, Ys, Zs = set(X), set(Y), set(Z)
    exact = loose = False
    for (c, p) in jt.edges:
        S = set(jt.sep[(c, p)])
        if not Zs.issubset(S): continue
        A = jt._atoms_of(jt._subtree(c, p)) - S; B = jt._atoms_of(jt._subtree(p, c)) - S
        if (Xs <= A and Ys <= B) or (Xs <= B and Ys <= A):
            if S == Zs: exact = True
            else: loose = True
    return exact, loose

def main():
    for bdir in sys.argv[1:]:
        c = collections.Counter(); inst = 0
        for f in sorted(glob.glob(f'{bdir}/*.lcn')):
            with contextlib.redirect_stdout(io.StringIO()):
                l = LCN(); l.from_lcn(f)
                cnv = CredalNetworkVertices.from_lcn(l, enumerate_vertices=False, solve_families=False, cache=False, verbosity=0)
                jt, *_ = _build_jt_and_hosts(cnv, None, {}, 16)
            bad = 0
            for a in l.independencies.get_assertions():
                X, Y, Z = sorted(a.event1), sorted(a.event2), sorted(a.event3)
                if not jt._rip_implied(X, Y, Z): continue
                e, lo = witness(jt, X, Y, Z)
                c['entailed'] += 1
                if not e: c['only_via_larger_sep'] += 1; bad += 1
            inst += bad > 0
        print(f"{bdir.split('/')[-1]:18s} entailed={c['entailed']:4d} only-via-S⊋Z={c['only_via_larger_sep']:4d} instances affected={inst}", flush=True)

if __name__ == '__main__':
    main()
