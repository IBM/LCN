"""E1 (part 1): run the certified CredalJT gate on every shipped instance of the
given benchmark dirs; write one JSON line per instance to certify_<set>.jsonl."""
import sys, glob, json, time, io, contextlib
sys.path.insert(0, '/Users/radu/git/IBM/LCN')
from lcn.core.model import LCN
from lcn.benchmarks.generator import cjt_exactness_report

def main():
    tl = float(sys.argv[1])
    for bdir in sys.argv[2:]:
        name = bdir.rstrip('/').split('/')[-1]
        out = open(f'certify_{name}.jsonl', 'a')
        done = set()
        try:
            done = {json.loads(l)['instance'] for l in open(f'certify_{name}.jsonl')}
        except Exception:
            pass
        for f in sorted(glob.glob(f'{bdir}/*.lcn')):
            if f in done: continue
            l = LCN()
            with contextlib.redirect_stdout(io.StringIO()):
                l.from_lcn(f)
            t = time.time()
            r = cjt_exactness_report(l, max_width=12, certify=True, time_limit=tl)
            r['instance'] = f; r['seconds'] = round(time.time() - t, 1)
            r['marginals'] = {a: list(v) for a, v in (r.get('marginals') or {}).items()}
            out.write(json.dumps(r, default=str) + '\n'); out.flush()
            print(name, f.split('/')[-1], r['exact'], r['width'], r['seconds'], r['reason'], flush=True)

if __name__ == '__main__':
    main()
