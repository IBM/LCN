"""Phase-0 helper: run a list of algorithms on one .lcn with given evidence."""
import json, sys
sys.path.insert(0, '/Users/radu/git/IBM/LCN')
from experiments.run_algorithm import run_single

def main():
    lcn, ev, algos = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3].split(',')
    for a in algos:
        r = run_single(lcn, a, evidence=ev, verbosity=0, time_limit=600, no_cache=True)
        m = r.get('marginals') or {}
        out = {k: (round(v['lower'][1], 4), round(v['upper'][1], 4)) for k, v in m.items()}
        print(f"{a:8s} {r['status']:7s} P(x=1): {out} {r.get('error') or ''}", flush=True)

if __name__ == '__main__':
    main()
