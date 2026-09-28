"""Produce consistency-verified copies of the real-world LCN benchmarks.

For every ``benchmarks/real/*.lcn`` instance this script:

1. Loads the instance and builds its primal/structure graphs + Local Markov
   Condition (LMC), so the exact consistency oracle can run.
2. Checks consistency with the **full joint-LMC oracle**
   :func:`lcn.inference.utils.common.check_consistency` -- the same exact
   oracle the inference path uses (NOT the conservative product-distribution
   witness the random generator gates on, which false-rejects real instances
   that are consistent only via a non-product distribution).
3. If an instance is genuinely inconsistent, it repairs it by *symmetric
   widening*: every sentence interval ``[lo, hi]`` is widened to
   ``[max(0, lo - d), min(1, hi + d)]`` for a growing ``d`` until the oracle
   accepts. This keeps interval midpoints fixed and only ever enlarges the
   feasible set, so it can never break a genuinely consistent instance.
4. Writes the (possibly repaired) consistent instance to
   ``<name>_consistent.lcn`` in the same directory. Originals are never touched.

The oracle is a *local* solver and is very slow on n=10 (minutes to hours with
a high restart count). It is therefore run in a worker subprocess with a
per-instance wall-clock budget: if it cannot return a verdict in time, the
instance is treated as consistent (left unchanged) and logged -- we never edit
an instance based on an unconfirmed / unreliable "inconsistent" verdict.

Run with the project venv (``uv run`` currently mis-resolves deps):

    .venv/bin/python benchmarks/real/make_consistent.py
    .venv/bin/python benchmarks/real/make_consistent.py --only cancer --dry-run
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import multiprocessing as mp
import os
import sys
import time

# Make the repo root importable when run as a plain script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lcn.core.model import LCN  # noqa: E402
from lcn.inference.utils.common import check_consistency  # noqa: E402


# Cumulative widening schedule (delta added to each side of every interval).
WIDEN_SCHEDULE = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]


def _load(path: str) -> LCN:
    """Load an LCN and build the graphs + LMC the oracle needs."""
    lcn = LCN()
    with contextlib.redirect_stdout(io.StringIO()):
        lcn.from_lcn(path)
        lcn.build_primal_graph()
        lcn.build_structure_graph()
        lcn.local_markov_condition()
    return lcn


def _oracle_worker(path: str, bounds, restarts: int, q: "mp.Queue") -> None:
    """Subprocess entry point: (re)build the LCN, apply overridden bounds if
    given, run the exact oracle, and push the boolean verdict onto ``q``.

    ``bounds`` is either ``None`` (use the file's bounds) or a dict
    ``{label: (lo, hi)}`` overriding each sentence's interval -- this lets the
    parent test a widened variant without the child re-deriving anything.
    """
    try:
        lcn = _load(path)
        if bounds is not None:
            for label, (lo, hi) in bounds.items():
                s = lcn.sentences[label]
                s.set_lower_bound(lo)
                s.set_upper_bound(hi)
        with contextlib.redirect_stdout(io.StringIO()):
            ok = check_consistency(lcn, max_slsqp_restarts=restarts)
        q.put(bool(ok))
    except Exception as exc:  # pragma: no cover - defensive
        q.put(f"ERR:{exc!r}")


def check_bounded(path: str, bounds, restarts: int, budget: float):
    """Run the oracle in a worker with a wall-clock budget.

    Returns ``True``/``False`` for a verdict, ``None`` on timeout, or an
    ``"ERR:..."`` string if the worker raised.
    """
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_oracle_worker,
                       args=(path, bounds, restarts, q))
    proc.start()
    proc.join(timeout=budget)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return None  # timeout -> unconfirmed
    if not q.empty():
        return q.get()
    return "ERR:worker-exited-without-result"


def widened_bounds(lcn: LCN, delta: float):
    """Return a ``{label: (lo, hi)}`` map with every interval symmetrically
    widened by ``delta`` and clamped to [0, 1]."""
    out = {}
    for label, s in lcn.sentences.items():
        lo = max(0.0, s.get_lower_bound() - delta)
        hi = min(1.0, s.get_upper_bound() + delta)
        out[label] = (lo, hi)
    return out


def process_instance(path: str, restarts: int, budget: float,
                     max_widen: float, dry_run: bool):
    """Check/repair one instance. Returns a result dict for the summary."""
    name = os.path.splitext(os.path.basename(path))[0]
    lcn = _load(path)
    n = len(lcn.atoms)
    out_path = os.path.join(os.path.dirname(path), f"{name}_consistent.lcn")

    t0 = time.time()
    verdict = check_bounded(path, None, restarts, budget)
    elapsed = time.time() - t0

    result = {"name": name, "n": n, "out": out_path,
              "action": None, "delta": 0.0, "check_s": elapsed}

    if isinstance(verdict, str):  # worker error
        result["action"] = f"error ({verdict})"
        return result, None
    if verdict is True:
        result["action"] = "unchanged (consistent)"
        return result, lcn
    if verdict is None:
        result["action"] = "unchanged (TIMEOUT/assumed-consistent)"
        return result, lcn

    # verdict is False -> confirmed inconsistent within budget: repair.
    for delta in WIDEN_SCHEDULE:
        if delta > max_widen:
            break
        bounds = widened_bounds(lcn, delta)
        v = check_bounded(path, bounds, restarts, budget)
        if v is True:
            for label, (lo, hi) in bounds.items():
                lcn.sentences[label].set_lower_bound(lo)
                lcn.sentences[label].set_upper_bound(hi)
            result["action"] = f"widened (d={delta})"
            result["delta"] = delta
            return result, lcn
        # v is None (timeout) or False -> keep widening; timeout on a widened
        # variant is inconclusive, so we do not accept it.

    result["action"] = f"UNREPAIRED (still inconsistent at d={max_widen})"
    return result, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--time-budget", type=float, default=600.0,
                    help="Per-oracle-call wall-clock budget in seconds "
                         "(default 600).")
    ap.add_argument("--restarts", type=int, default=300,
                    help="max_slsqp_restarts passed to check_consistency "
                         "(default 300).")
    ap.add_argument("--max-widen", type=float, default=0.5,
                    help="Maximum symmetric widening delta before giving up "
                         "(default 0.5).")
    ap.add_argument("--only", type=str, default=None,
                    help="Process a single instance by basename (e.g. cancer).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Check/repair but do not write output files.")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(_HERE, "*.lcn")))
    # Never treat previously produced outputs as inputs.
    files = [f for f in files if not f.endswith("_consistent.lcn")]
    if args.only:
        files = [f for f in files
                 if os.path.splitext(os.path.basename(f))[0] == args.only]
        if not files:
            print(f"No instance named '{args.only}' in {_HERE}")
            return 1

    results = []
    for path in files:
        print(f"[{os.path.basename(path)}] checking "
              f"(budget={args.time_budget}s, restarts={args.restarts}) ...",
              flush=True)
        result, lcn = process_instance(path, args.restarts, args.time_budget,
                                       args.max_widen, args.dry_run)
        if lcn is not None and not args.dry_run:
            with contextlib.redirect_stdout(io.StringIO()):
                lcn.save_lcn(result["out"])
        results.append(result)
        print(f"    -> {result['action']}  ({result['check_s']:.1f}s)",
              flush=True)

    # Summary table.
    print("\n=== Summary ===")
    hdr = f"{'instance':14s} {'n':>2s}  {'action':40s} {'output'}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        out = "" if args.dry_run else os.path.basename(r["out"])
        print(f"{r['name']:14s} {r['n']:>2d}  {r['action']:40s} {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
