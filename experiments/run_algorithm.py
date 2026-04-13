"""Run a single inference algorithm on a single LCN instance.

Core building block for the experimental pipeline. Can be used standalone
or called from run_experiment.py.

Usage:
    python experiments/run_algorithm.py \
        --instance benchmarks/chain/chain_n10_1.lcn \
        --algorithm ccte
"""

import os


def set_num_threads(n):
    """Pin BLAS/LAPACK/OpenMP thread count. Must be called before
    importing numpy, pyomo, or any numerical library."""
    t = str(n)
    os.environ["OMP_NUM_THREADS"] = t
    os.environ["OPENBLAS_NUM_THREADS"] = t
    os.environ["MKL_NUM_THREADS"] = t
    os.environ["VECLIB_MAXIMUM_THREADS"] = t
    os.environ["NUMEXPR_NUM_THREADS"] = t


# Default to 1 thread for reproducible benchmarking.
# Can be overridden by calling set_num_threads() before _ensure_imports(),
# or via the --num-threads CLI flag.
_DEFAULT_NUM_THREADS = 1
if "OMP_NUM_THREADS" not in os.environ:
    set_num_threads(_DEFAULT_NUM_THREADS)

import argparse
import json
import multiprocessing
import time
import numpy as np

from lcn.core.model import LCN
from lcn.inference.marginal.exact import ExactInference
from lcn.inference.marginal.ariel import ArielInference
from lcn.inference.marginal.cve import CredalVE
from lcn.inference.marginal.ibp import IntervalBP
from lcn.inference.marginal.ccte import CredalCTE
from lcn.inference.marginal.approxlp import ApproxLP

ALGORITHMS = ["exact", "ariel", "ibp", "ccte", "ccte_e", "approxlp"]


_CVE_ALGORITHMS = {"ibp", "ccte", "ccte_e", "approxlp"}


def _compute_induced_width(cve):
    """Compute the induced width (treewidth upper bound) from a built CredalVE.

    Replays the min-fill elimination on the interaction graph built from
    the potential scopes.  The induced width is the maximum number of
    neighbours a variable has at the moment it is eliminated.
    """
    bn = cve.bn_min
    cards = {}
    for nid in bn.nodes():
        cards[bn.variable(nid).name()] = bn.variable(nid).domainSize()

    # Collect scopes from the extreme-point potentials
    scopes = []
    for node_name in cve.extreme_points:
        nid = bn.idFromName(node_name)
        parent_ids = sorted(bn.parents(nid))
        parent_names = [bn.variable(pid).name() for pid in parent_ids]
        scopes.append([node_name] + parent_names)

    # Build interaction graph
    all_vars = set()
    for s in scopes:
        all_vars.update(s)
    adj = {v: set() for v in all_vars}
    for s in scopes:
        for i, u in enumerate(s):
            for v in s[i + 1:]:
                adj[u].add(v)
                adj[v].add(u)

    # Min-fill elimination, tracking max cluster size
    remaining = set(all_vars)
    max_width = 0
    for _ in range(len(all_vars)):
        # Pick variable with fewest fill edges
        best_var = None
        best_fill = float('inf')
        for v in remaining:
            nbrs = [u for u in adj[v] if u in remaining]
            fill = sum(1 for i, u in enumerate(nbrs)
                       for w in nbrs[i + 1:] if w not in adj[u])
            if fill < best_fill or (fill == best_fill and
                    (best_var is None or v < best_var)):
                best_fill = fill
                best_var = v

        # The cluster at this step = {best_var} + its remaining neighbors
        nbrs = [u for u in adj[best_var] if u in remaining]
        max_width = max(max_width, len(nbrs))

        # Add fill edges
        for i, u in enumerate(nbrs):
            for w in nbrs[i + 1:]:
                adj[u].add(w)
                adj[w].add(u)

        # Remove variable
        remaining.remove(best_var)
        for u in adj[best_var]:
            adj[u].discard(best_var)
        del adj[best_var]

    return max_width


def _filter_singletons(results):
    """Keep only singleton variables (no '-' in name) from a results dict."""
    return {k: v for k, v in results.items() if "-" not in k}


def _marginals_to_dict(results):
    """Convert numpy marginals to JSON-serializable dict."""
    out = {}
    for var, (lo, hi) in results.items():
        out[var] = {
            "lower": [round(float(x), 8) for x in lo],
            "upper": [round(float(x), 8) for x in hi],
        }
    return out


def _run_single_worker(queue, lcn_file, algorithm, evidence, verbosity, kwargs):
    """Worker function for multiprocessing timeout enforcement."""
    result = _run_single_impl(lcn_file, algorithm, evidence, verbosity, **kwargs)
    queue.put(result)


def run_single(lcn_file, algorithm, evidence=None, verbosity=0,
               time_limit=None, **kwargs):
    """
    Run one algorithm on one LCN instance, with optional time limit.

    Args:
        lcn_file: path to .lcn file
        algorithm: one of "exact", "ariel", "ibp", "ccte", "ccte_e", "approxlp"
        evidence: dict of evidence (default: {})
        verbosity: 0=silent
        time_limit: max wall-clock seconds (None=unlimited). Enforced by
                    running the algorithm in a subprocess that is killed
                    if it exceeds the limit.
        **kwargs: algorithm-specific params

    Returns:
        dict with keys: algorithm, build_time, run_time, total_time,
        induced_width, status, marginals, error, epsilon
    """
    if evidence is None:
        evidence = {}

    if time_limit is None:
        return _run_single_impl(
            lcn_file, algorithm, evidence, verbosity, **kwargs)

    # Run in a subprocess with timeout
    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_run_single_worker,
        args=(queue, lcn_file, algorithm, evidence, verbosity, kwargs))
    proc.start()
    proc.join(timeout=time_limit)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return {
            "algorithm": algorithm,
            "build_time": 0.0,
            "run_time": 0.0,
            "total_time": round(time_limit, 4),
            "induced_width": None,
            "status": "timeout",
            "marginals": {},
            "error": f"Time limit exceeded ({time_limit}s)",
            "epsilon": kwargs.get("epsilon", None),
        }

    if not queue.empty():
        return queue.get_nowait()

    return {
        "algorithm": algorithm,
        "build_time": 0.0,
        "run_time": 0.0,
        "total_time": 0.0,
        "induced_width": None,
        "status": "error",
        "marginals": {},
        "error": "Worker process exited without result",
        "epsilon": kwargs.get("epsilon", None),
    }


def _run_single_impl(lcn_file, algorithm, evidence=None, verbosity=0, **kwargs):
    """Run one algorithm on one LCN instance (no timeout enforcement)."""
    if evidence is None:
        evidence = {}

    # Load LCN
    l = LCN()
    l.from_lcn(file_name=lcn_file)

    result = {
        "algorithm": algorithm,
        "build_time": 0.0,
        "run_time": 0.0,
        "total_time": 0.0,
        "induced_width": None,
        "status": "ok",
        "marginals": {},
        "error": None,
        "epsilon": kwargs.get("epsilon", None),
    }

    try:
        t_start = time.time()

        if algorithm == "exact":
            algo = ExactInference(lcn=l)
            raw = algo.run(evidence=evidence, verbosity=verbosity, debug=True)
            marginals = _filter_singletons(raw)
            t_end = time.time()
            result["run_time"] = round(t_end - t_start, 4)

        elif algorithm == "ariel":
            n_iters = kwargs.get("n_iters", 10)
            threshold = kwargs.get("threshold", 1e-6)
            algo = ArielInference(lcn=l)
            raw = algo.run(
                n_iters=n_iters, threshold=threshold,
                evidence=evidence, verbosity=verbosity)
            marginals = _filter_singletons(raw)
            t_end = time.time()
            result["run_time"] = round(t_end - t_start, 4)

        elif algorithm in _CVE_ALGORITHMS:
            # Build factorization (CredalVE) — timed separately
            t_build_start = time.time()
            cve = CredalVE(lcn=l)
            cve.build(verbosity=verbosity)
            t_build_end = time.time()
            result["build_time"] = round(t_build_end - t_build_start, 4)
            result["induced_width"] = _compute_induced_width(cve)

            # Run the algorithm — timed separately
            t_run_start = time.time()
            if algorithm == "ibp":
                n_iters = kwargs.get("n_iters", 100)
                threshold = kwargs.get("threshold", 1e-6)
                algo = IntervalBP(cve=cve)
                raw = algo.run(
                    evidence=evidence, n_iters=n_iters,
                    threshold=threshold, method="interval",
                    verbosity=verbosity)
            elif algorithm == "ccte":
                epsilon = kwargs.get("epsilon", None)
                algo = CredalCTE(cve=cve)
                raw = algo.run(
                    evidence=evidence, epsilon=epsilon,
                    verbosity=verbosity)
            elif algorithm == "ccte_e":
                epsilon = kwargs.get("epsilon", None)
                assert epsilon is not None, "epsilon must be provided for ccte_e"
                algo = CredalCTE(cve=cve)
                raw = algo.run(
                    evidence=evidence, epsilon=epsilon,
                    verbosity=verbosity)
            elif algorithm == "approxlp":
                n_iters = kwargs.get("n_iters", 50)
                algo = ApproxLP(cve=cve)
                raw = algo.run(
                    evidence=evidence, n_iters=n_iters,
                    verbosity=verbosity)
            t_run_end = time.time()
            result["run_time"] = round(t_run_end - t_run_start, 4)

            marginals = _filter_singletons(raw)

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        result["total_time"] = round(
            result["build_time"] + result["run_time"], 4)
        result["marginals"] = _marginals_to_dict(marginals)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["total_time"] = round(time.time() - t_start, 4)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run a single inference algorithm on an LCN instance.")
    parser.add_argument(
        "--instance", type=str, required=True,
        help="Path to the .lcn file")
    parser.add_argument(
        "--algorithm", type=str, required=True, choices=ALGORITHMS,
        help="Algorithm to run")
    parser.add_argument(
        "--evidence", type=str, default="{}",
        help="Evidence as JSON string (default: {})")
    parser.add_argument(
        "--epsilon", type=float, default=None,
        help="Epsilon for ccte_e algorithm (required for ccte_e)")
    parser.add_argument(
        "--time-limit", type=float, default=None,
        help="Time limit in seconds per instance (default: unlimited)")
    parser.add_argument(
        "--num-threads", type=int, default=1,
        help="Number of threads for BLAS/LAPACK/ipopt (default: 1)")
    parser.add_argument(
        "--verbosity", type=int, default=2,
        help="Verbosity level (default: 2)")
    args = parser.parse_args()

    set_num_threads(args.num_threads)

    evidence = json.loads(args.evidence)
    kwargs = {}
    if args.epsilon is not None:
        kwargs["epsilon"] = args.epsilon
    result = run_single(
        args.instance, args.algorithm,
        evidence=evidence, verbosity=args.verbosity,
        time_limit=args.time_limit, **kwargs)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
