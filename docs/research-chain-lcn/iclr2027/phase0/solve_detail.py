import sys, time
sys.path.insert(0, '/Users/radu/git/IBM/LCN/docs/research-chain-lcn/iclr2027/phase0')
from cjt_check import run
if __name__ == '__main__':
    t = time.time(); jt = run(sys.argv[1], {}, tl=float(sys.argv[2]))
    print(f"wall {time.time()-t:.1f}s certified={jt.certified} exact={jt.exact} h3={jt.h3}")
    for s in jt.solve_status:
        p, d = s.get('primal'), s.get('dual')
        diff = abs(p - d) if p is not None and d is not None else None
        if not s['certified'] or len(sys.argv) > 3:
            print(f"  {s['atom']} {s['sense']} val={s['value']!r} primal={p!r} dual={d!r} |p-d|={diff} gap%={s['gap']} tc={s['termination']} cert={s['certified']}")
