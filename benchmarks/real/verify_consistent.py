"""Independent re-verification of the ``*_consistent.lcn`` outputs.

Separate from ``make_consistent.py``: for every produced ``<name>_consistent.lcn``
this loads it fresh, rebuilds graphs + LMC, and asserts the full joint-LMC
oracle (:func:`lcn.inference.utils.common.check_consistency`) reports it
consistent. It also reports, for each pair, whether the consistent copy is
numerically identical to the original (bounds unchanged) or which sentences
were widened and by how much.

    .venv/bin/python benchmarks/real/verify_consistent.py --time-budget 900
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lcn.core.model import LCN  # noqa: E402
from lcn.inference.utils.common import check_consistency  # noqa: E402


def _load(path: str) -> LCN:
    lcn = LCN()
    with contextlib.redirect_stdout(io.StringIO()):
        lcn.from_lcn(path)
        lcn.build_primal_graph()
        lcn.build_structure_graph()
        lcn.local_markov_condition()
    return lcn


def _worker(path: str, restarts: int, q: "mp.Queue") -> None:
    try:
        lcn = _load(path)
        with contextlib.redirect_stdout(io.StringIO()):
            ok = check_consistency(lcn, max_slsqp_restarts=restarts)
        q.put(bool(ok))
    except Exception as exc:  # pragma: no cover
        q.put(f"ERR:{exc!r}")


def check_bounded(path: str, restarts: int, budget: float):
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(path, restarts, q))
    proc.start()
    proc.join(timeout=budget)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return None
    return q.get() if not q.empty() else "ERR:no-result"


def bound_diff(orig: str, cons: str):
    """Return list of (label, (olo, ohi), (nlo, nhi)) for changed sentences."""
    lo = LCN()
    lc = LCN()
    with contextlib.redirect_stdout(io.StringIO()):
        lo.from_lcn(orig)
        lc.from_lcn(cons)
    changed = []
    for label, s in lo.sentences.items():
        t = lc.sentences[label]
        o = (s.get_lower_bound(), s.get_upper_bound())
        n = (t.get_lower_bound(), t.get_upper_bound())
        if abs(o[0] - n[0]) > 1e-12 or abs(o[1] - n[1]) > 1e-12:
            changed.append((label, o, n))
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-budget", type=float, default=900.0)
    ap.add_argument("--restarts", type=int, default=300)
    args = ap.parse_args()

    outputs = sorted(glob.glob(os.path.join(_HERE, "*_consistent.lcn")))
    if not outputs:
        print("No *_consistent.lcn files found. Run make_consistent.py first.")
        return 1

    all_ok = True
    for cons in outputs:
        name = os.path.basename(cons)[: -len("_consistent.lcn")]
        orig = os.path.join(_HERE, f"{name}.lcn")
        t0 = time.time()
        verdict = check_bounded(cons, args.restarts, args.time_budget)
        dt = time.time() - t0

        changed = bound_diff(orig, cons) if os.path.exists(orig) else []
        tag = "IDENTICAL to original" if not changed \
            else f"{len(changed)} sentence(s) widened"

        status = {True: "CONSISTENT", False: "INCONSISTENT",
                  None: "TIMEOUT"}.get(verdict, str(verdict))
        mark = "OK " if verdict is True else "!! "
        if verdict is not True:
            all_ok = False
        print(f"{mark}{name:14s} {status:12s} {tag:26s} ({dt:.1f}s)")
        for label, o, n in changed:
            print(f"      {label}: [{o[0]},{o[1]}] -> [{n[0]},{n[1]}]")

    print("\nRESULT:", "all consistent" if all_ok
          else "SOME NOT VERIFIED CONSISTENT")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
