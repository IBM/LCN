"""Structure-exploiting consistency check for any LCN instance under benchmarks/.

Decides consistency over junction-tree clusters of size ``2^treewidth`` instead
of the intractable ``2^n`` joint, so it applies to every benchmark family
(real, junkyu, random chain/dag/tree/polytree/ktree, ...) regardless of ``n``,
as long as the moralized treewidth stays within ``--max-cluster-atoms``.

Wraps :func:`lcn.inference.utils.structured_consistency.check_consistency_structured`.
See that module for the soundness argument (INFEASIBLE => genuinely
inconsistent with a global solver; a feasible witness => consistent).

Examples
--------
    # A few explicit files
    .venv/bin/python benchmarks/check_consistency.py benchmarks/junkyu/*.lcn

    # A whole directory
    .venv/bin/python benchmarks/check_consistency.py --dir benchmarks/real

    # Fast local solver (CONSISTENT stays sound; INCONSISTENT -> UNDETERMINED)
    .venv/bin/python benchmarks/check_consistency.py --solver ipopt --jobs 4 --dir benchmarks/junkyu

Exit code is non-zero if any checked instance is INCONSISTENT, so this doubles
as a CI gate.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import glob
import io
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lcn.core.model import LCN  # noqa: E402
from lcn.inference.utils.structured_consistency import (  # noqa: E402
    check_consistency_structured, INCONSISTENT)


def _check_one(path, max_cluster_atoms, time_limit, solver):
    """Load one instance and run the structured check. Returns (path, result)."""
    lcn = LCN()
    with contextlib.redirect_stdout(io.StringIO()):
        lcn.from_lcn(path)
    r = check_consistency_structured(
        lcn, max_cluster_atoms=max_cluster_atoms,
        time_limit=time_limit, solver=solver)
    return path, r


def _collect_files(args):
    files = list(args.files)
    if args.dir:
        files += sorted(glob.glob(os.path.join(args.dir, "*.lcn")))
    # Expand any globs the shell did not, dedupe, keep order.
    expanded = []
    seen = set()
    for f in files:
        for g in (glob.glob(f) if any(c in f for c in "*?[") else [f]):
            if g not in seen:
                seen.add(g)
                expanded.append(g)
    return expanded


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="LCN files (or globs).")
    ap.add_argument("--dir", type=str, default=None,
                    help="Also check every *.lcn in this directory.")
    ap.add_argument("--solver", choices=("scip", "ipopt"), default="scip",
                    help="scip (default; global, trustworthy INCONSISTENT) or "
                         "ipopt (local; fast, CONSISTENT still sound).")
    ap.add_argument("--time-limit", type=float, default=600.0,
                    help="Per-instance solver wall-clock limit in seconds.")
    ap.add_argument("--max-cluster-atoms", type=int, default=16,
                    help="Treewidth budget as max atoms per JT cluster "
                         "(default 16; cost is sum_c 2^|cluster|).")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Check this many instances in parallel (separate "
                         "processes; default 1).")
    args = ap.parse_args()

    files = _collect_files(args)
    if not files:
        ap.error("no input files (pass paths/globs or --dir)")

    print(f"{'instance':52s} {'verdict':13s} {'tw':>3s} {'sec':>7s}  note")
    print("-" * 100)

    results = []

    def _emit(path, r):
        tw = "" if r.treewidth is None else str(r.treewidth)
        print(f"{os.path.basename(path):52s} {r.status:13s} {tw:>3s} "
              f"{r.seconds:7.1f}  {r.note}", flush=True)
        results.append((path, r))

    if args.jobs > 1:
        with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_check_one, f, args.max_cluster_atoms,
                              args.time_limit, args.solver): f for f in files}
            done = {}
            for fut in cf.as_completed(futs):
                path, r = fut.result()
                done[path] = r
            # Emit in the original file order for a stable table.
            for f in files:
                _emit(f, done[f])
    else:
        for f in files:
            _, r = _check_one(f, args.max_cluster_atoms, args.time_limit,
                              args.solver)
            _emit(f, r)

    # Summary.
    from collections import Counter
    counts = Counter(r.status for _, r in results)
    print("\n" + "  ".join(f"{k}={counts[k]}" for k in sorted(counts)))
    n_incon = counts.get(INCONSISTENT, 0)
    return 1 if n_incon > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
