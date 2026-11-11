"""A2 audit: what does Potential.prune remove, with and without evidence?"""
import sys, io, contextlib, collections, random
import numpy as np
sys.path.insert(0, '/Users/radu/git/IBM/LCN')
from lcn.core.model import LCN
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.marginal.cn.cve import CredalVE
from lcn.inference.marginal.cn import potentials as P

STATS = collections.Counter()
ORIG = P.Potential.prune

def audited(self):
    out = ORIG(self)
    if len(self.functions) > 1:
        STATS['prune_calls'] += 1
        kept = [f.ravel() for f in out.functions]
        for f in self.functions:
            if not any(np.allclose(f.ravel(), k, atol=1e-12) for k in kept):
                STATS['removed_nonduplicate'] += 1
    return out

def main():
    P.Potential.prune = audited
    import glob
    ev_mode, heur = sys.argv[1], sys.argv[2]
    files = sorted(f for pat in sys.argv[3:] for f in glob.glob(pat))
    rng = random.Random(0)
    for f in files:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                l = LCN(); l.from_lcn(f)
                cnv = CredalNetworkVertices.from_lcn(l, lcn_file=f, cache=True, verbosity=0)
                atoms = sorted(l.atoms.keys())
                ev = {} if ev_mode == 'none' else {rng.choice(atoms): rng.randint(0, 1)}
                CredalVE(cnv=cnv).run(evidence=ev, elim_heuristic=heur, verbosity=0)
            STATS['ok'] += 1
        except Exception as e:
            STATS['err'] += 1
            print('ERR', f.split('/')[-1], type(e).__name__, str(e)[:80], flush=True)
    print('RESULT', ev_mode, heur, dict(STATS), flush=True)

if __name__ == '__main__':
    main()
