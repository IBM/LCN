"""Phase-0 (S2): CVE with dominance pruning vs CVE with pruning disabled
(Potential.prune -> identity), with and without evidence. Unpruned CVE is the
exact strong-extension answer by definition; any difference => pruning unsound."""
import sys, glob, io, contextlib, random, json
sys.path.insert(0, '/Users/radu/git/IBM/LCN')
from lcn.core.model import LCN
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.marginal.cn.cve import CredalVE
from lcn.inference.marginal.cn import potentials

ORIG = potentials.Potential.prune

def marg(cnv, ev, prune):
    potentials.Potential.prune = ORIG if prune else (lambda self: self)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ve = CredalVE(cnv=cnv); ve.run(evidence=ev, verbosity=0)
        return {k: (float(v[0][1]), float(v[1][1])) if hasattr(v[0], '__len__') else v
                for k, v in ve.marginals.items()}
    finally:
        potentials.Potential.prune = ORIG

def main():
    files = sys.argv[2:]; nev = int(sys.argv[1]); rng = random.Random(0)
    worst = {'noev': 0.0, 'ev': 0.0}; cases = {'noev': 0, 'ev': 0}; bad = {'noev': 0, 'ev': 0}
    for f in files:
        with contextlib.redirect_stdout(io.StringIO()):
            l = LCN(); l.from_lcn(f)
            cnv = CredalNetworkVertices.from_lcn(l, lcn_file=f, cache=True, verbosity=0)
        atoms = sorted(l.atoms.keys()) if isinstance(l.atoms, dict) else sorted(l.atoms)
        evs = [('noev', {})] + [('ev', {a: rng.randint(0, 1)}) for a in rng.sample(atoms, min(nev, len(atoms)))]
        for tag, ev in evs:
            try:
                p, u = marg(cnv, ev, True), marg(cnv, ev, False)
            except Exception as ex:
                print(f"  skip {f.split('/')[-1]} {ev}: {type(ex).__name__}: {ex}"[:150]); continue
            cases[tag] += 1; d = 0.0
            for k in u:
                if k in p:
                    a, b = p[k], u[k]
                    try: d = max(d, abs(a[0]-b[0]), abs(a[1]-b[1]))
                    except TypeError: pass
            worst[tag] = max(worst[tag], d)
            if d > 1e-6:
                bad[tag] += 1
                print(f"  DIFF {d:.4f} {f.split('/')[-1]} ev={ev}\n     pruned  ={ {k:tuple(round(x,4) for x in v) for k,v in p.items()} }\n     unpruned={ {k:tuple(round(x,4) for x in v) for k,v in u.items()} }", flush=True)
    print(f"RESULT cases={cases} differing={bad} worst={worst}")

if __name__ == '__main__':
    main()
