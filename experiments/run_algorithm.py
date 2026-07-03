"""Run a single inference algorithm on a single LCN instance.

Core building block for the experimental pipeline. Can be used standalone
or called from run_experiment.py.

Usage:
    python experiments/run_algorithm.py \
        --instance benchmarks/chain/chain_n10_1.lcn \
        --algorithm ccte
"""

import os
import argparse
import json
import multiprocessing
import time

from lcn.core.model import LCN
from lcn.inference.marginal.exact import ExactInference
from lcn.inference.marginal.ariel import ArielInference
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.marginal.cn.ibp import IntervalBP
from lcn.inference.marginal.cn.ccte import CredalCTE
from lcn.inference.marginal.cn.approxlp import ApproxLP
from lcn.inference.marginal.cn.ijgp import CredalIJGP
from lcn.inference.marginal.cn.cve import CredalVE
from lcn.inference.marginal.cn.junction_nlp import CredalJT


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

ALGORITHMS = ["exact_l", "exact_g", "ariel", "ibp",  "ccte", "ccte_e", "ccte_cm", "approxlp", "cve", "cve_e", "cve_cp", "cve_cm", "cve_d4", "cjt"]

_CVE_ALGORITHMS = {"ibp", "ccte", "ccte_e", "ccte_cm", "approxlp", "cve", "cve_e", "cve_cp", "cve_cm", "cve_d4", "cjt"}

def _compute_induced_width(cnv):
    """Compute the induced width (treewidth upper bound) from a built
    CredalNetworkVertices.

    Replays the min-fill elimination on the interaction graph built from
    the potential scopes.  The induced width is the maximum number of
    neighbours a variable has at the moment it is eliminated.
    """
    # Collect the per-node CPT scopes [node] + parents. Derived from the
    # CredalNetwork factors (cnv.cn) so this also works for a vertex-free build
    # (enumerate_vertices=False, e.g. CredalJT) where extreme_points is None.
    # The factor/child order and parents match the pyAgrum bn_min arcs, so the
    # induced width is identical either way.
    scopes = []
    for factor in cnv.cn.factors:
        entry = factor[0]
        scopes.append([entry["child"]] + list(entry["parents"]))

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
        algorithm: one of "exact_l", "exact_g", "ariel", "ibp", "ccte",
                   "ccte_e", "approxlp", "cve", "cjt" (see ALGORITHMS)
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

    # For the exact backends, give the engine its own per-solve limit matching
    # the subprocess watchdog so SCIP/ipopt stop cleanly with partial bounds
    # instead of being hard-killed. (No-op for other algorithms.)
    if (time_limit is not None and algorithm in ("exact_l", "exact_g")
            and "exact_time_limit" not in kwargs):
        kwargs["exact_time_limit"] = float(time_limit)

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
    lcn_model = LCN()
    lcn_model.from_lcn(file_name=lcn_file)

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

        if algorithm in ("exact_l", "exact_g"):
            # Two backends of the same full-joint NLP:
            #   exact_l -> "local"  (ipopt + SLSQP fallback; fast, may be loose)
            #   exact_g -> "global" (SCIP spatial branch-and-bound; certified)
            exact_solver = "local" if algorithm == "exact_l" else "global"
            # The per-solve time limit (also used by the subprocess watchdog) so
            # the engine terminates cleanly with partial bounds rather than being
            # hard-killed. Defaults to ExactInference's own default when absent.
            exact_time_limit = kwargs.get("exact_time_limit", 3600.0)
            algo = ExactInference(lcn=lcn_model)
            raw = algo.run(
                evidence=evidence, solver=exact_solver, verbosity=verbosity,
                debug=(exact_solver == "local"), time_limit=exact_time_limit)
            marginals = _filter_singletons(raw)
            t_end = time.time()
            result["run_time"] = round(t_end - t_start, 4)

        elif algorithm == "ariel":
            n_iters = kwargs.get("n_iters", 10)
            threshold = kwargs.get("threshold", 1e-6)
            algo = ArielInference(lcn=lcn_model)
            raw = algo.run(
                n_iters=n_iters, threshold=threshold,
                evidence=evidence, verbosity=verbosity)
            marginals = _filter_singletons(raw)
            t_end = time.time()
            result["run_time"] = round(t_end - t_start, 4)

        elif algorithm in _CVE_ALGORITHMS:
            # Build the credal network vertices (chain-graph factorization +
            # interval local credal sets + extreme-point enumeration). The
            # whole pipeline is timed by CredalNetworkVertices.from_lcn via
            # perf_counter; build_time is reported in the experiments.
            fact_method = kwargs.get("factorization_method", "linear")
            n_jobs = kwargs.get("n_jobs", 1)
            cn_solver = kwargs.get("solver", "ipopt")
            cn_time_limit = kwargs.get("solver_time_limit", None)
            cn_gap_tol = kwargs.get("gap_tol", 0.0)
            merge_budget = kwargs.get("merge_budget", 1)
            # CredalJT (D5) formulates its NLP from the interval local credal
            # sets and never consumes the enumerated extreme points, so skip
            # the (potentially expensive) LRS vertex enumeration for it.
            enumerate_vertices = (algorithm != "cjt")
            cnv = CredalNetworkVertices.from_lcn(
                lcn_model, method=fact_method, solver=cn_solver,
                time_limit=cn_time_limit, gap_tol=cn_gap_tol,
                n_jobs=n_jobs, merge_budget=merge_budget,
                enumerate_vertices=enumerate_vertices, verbosity=verbosity)
            result["build_time"] = round(cnv.build_time, 4)
            result["induced_width"] = _compute_induced_width(cnv)

            # Run the algorithm — timed separately
            t_run_start = time.time()
            if algorithm == "ibp":
                n_iters = kwargs.get("n_iters", 100)
                threshold = kwargs.get("threshold", 1e-6)
                algo = IntervalBP(cnv=cnv)
                raw = algo.run(
                    evidence=evidence, n_iters=n_iters,
                    threshold=threshold, method="interval",
                    verbosity=verbosity)
            elif algorithm == "ijgp":
                i_bound = kwargs.get("i_bound", 2)
                n_iters = kwargs.get("n_iters", 100)
                threshold = kwargs.get("threshold", 1e-6)
                epsilon = None
                algo = CredalIJGP(cnv=cnv)
                raw = algo.run(
                    evidence=evidence, i_bound=i_bound,
                    n_iters=n_iters, threshold=threshold,
                    epsilon=epsilon, verbosity=verbosity)
            elif algorithm == "ijgp_e":
                i_bound = kwargs.get("ibound", 2)
                n_iters = kwargs.get("n_iters", 100)
                threshold = kwargs.get("threshold", 1e-6)
                epsilon = kwargs.get("epsilon", None)
                algo = CredalIJGP(cnv=cnv)
                raw = algo.run(
                    evidence=evidence, i_bound=i_bound,
                    n_iters=n_iters, threshold=threshold,
                    epsilon=epsilon, verbosity=verbosity)
            elif algorithm == "ijgp_cp":
                i_bound = kwargs.get("i_bound", 2)
                n_iters = kwargs.get("n_iters", 100)
                threshold = kwargs.get("threshold", 1e-6)
                epsilon = kwargs.get("epsilon", None)
                n_clusters = kwargs.get("n_clusters", 10)
                cluster_rep = kwargs.get("cluster_representative", "plub")
                algo = CredalIJGP(cnv=cnv)
                raw = algo.run(
                    evidence=evidence, i_bound=i_bound,
                    n_iters=n_iters, threshold=threshold,
                    epsilon=epsilon, n_clusters=n_clusters,
                    cluster_representative=cluster_rep,
                    verbosity=verbosity)
            elif algorithm == "ijgp_cm":
                i_bound = kwargs.get("i_bound", 2)
                n_iters = kwargs.get("n_iters", 100)
                threshold = kwargs.get("threshold", 1e-6)
                epsilon = kwargs.get("epsilon", None)
                n_clusters = kwargs.get("n_clusters", 10)
                cluster_rep = kwargs.get("cluster_representative", "mean")
                algo = CredalIJGP(cnv=cnv)
                raw = algo.run(
                    evidence=evidence, i_bound=i_bound,
                    n_iters=n_iters, threshold=threshold,
                    epsilon=epsilon, n_clusters=n_clusters,
                    cluster_representative=cluster_rep,
                    verbosity=verbosity)
            elif algorithm == "ccte":
                epsilon = None
                algo = CredalCTE(cnv=cnv)
                raw = algo.run(
                    evidence=evidence, epsilon=epsilon,
                    verbosity=verbosity)
            elif algorithm == "ccte_e":
                epsilon = kwargs.get("epsilon", None)
                assert epsilon is not None, "epsilon must be provided for ccte_e"
                algo = CredalCTE(cnv=cnv)
                raw = algo.run(
                    evidence=evidence, epsilon=epsilon,
                    verbosity=verbosity)
            elif algorithm == "ccte_cp":
                epsilon = kwargs.get("epsilon", None)
                n_clusters = kwargs.get("n_clusters", 10)
                cluster_rep = kwargs.get("cluster_representative", "plub")
                algo = CredalCTE(cnv=cnv)
                raw = algo.run(
                    evidence=evidence, epsilon=epsilon,
                    n_clusters=n_clusters,
                    cluster_representative=cluster_rep,
                    verbosity=verbosity)
            elif algorithm == "ccte_cm":
                epsilon = kwargs.get("epsilon", None)
                n_clusters = kwargs.get("n_clusters", 10)
                cluster_rep = kwargs.get("cluster_representative", "mean")
                algo = CredalCTE(cnv=cnv)
                raw = algo.run(
                    evidence=evidence, epsilon=epsilon,
                    n_clusters=n_clusters,
                    cluster_representative=cluster_rep,
                    verbosity=verbosity)
            elif algorithm == "approxlp":
                n_iters = kwargs.get("n_iters", 50)
                algo = ApproxLP(cnv=cnv)
                raw = algo.run(
                    evidence=evidence, n_iters=n_iters,
                    verbosity=verbosity)
            elif algorithm == "cve":
                algo = CredalVE(cnv=cnv)
                raw = algo.run(
                    evidence=evidence, coupling="off", verbosity=verbosity)
            elif algorithm == "cve_e":
                epsilon = kwargs.get("epsilon", None)
                assert epsilon is not None, "epsilon required for cve_e"
                algo = CredalVE(cnv=cnv)
                raw = algo.run(
                    evidence=evidence, epsilon=epsilon, coupling="off",
                    verbosity=verbosity)
            elif algorithm == "cve_cp":
                epsilon = kwargs.get("epsilon", None)
                n_clusters = kwargs.get("n_clusters", 10)
                cluster_rep = kwargs.get("cluster_representative", "plub")
                algo = CredalVE(cnv=cnv)
                raw = algo.run(
                    evidence=evidence, epsilon=epsilon, coupling="off",
                    n_clusters=n_clusters,
                    cluster_representative=cluster_rep,
                    verbosity=verbosity)
            elif algorithm == "cve_cm":
                epsilon = kwargs.get("epsilon", None)
                n_clusters = kwargs.get("n_clusters", 10)
                cluster_rep = kwargs.get("cluster_representative", "mean")
                algo = CredalVE(cnv=cnv)
                raw = algo.run(
                    evidence=evidence, epsilon=epsilon, coupling="off",
                    n_clusters=n_clusters,
                    cluster_representative=cluster_rep,
                    verbosity=verbosity)
            elif algorithm == "cve_d4":
                algo = CredalVE(cnv=cnv)
                raw = algo.run(
                    evidence=evidence, coupling="cross-family",
                    verbosity=verbosity)
            elif algorithm == "cjt":
                cjt_solver = kwargs.get("solver", "scip")
                algo = CredalJT(cnv=cnv)
                raw = algo.run(
                    evidence=evidence, solver=cjt_solver, verbosity=verbosity)
                result["induced_width"] = algo.induced_width
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
        help="Epsilon for ccte_e and ijgp_e algorithms")
    parser.add_argument(
        "--n-clusters", type=int, default=None,
        help="Number of clusters for ijgp_c algorithm (default: 10)")
    parser.add_argument(
        "--cluster-representative", type=str, default="plub",
        choices=["plub", "mean"],
        help="Cluster representative for ijgp_c: plub or mean (default: plub)")
    parser.add_argument(
        "--factorization-method", type=str, default="linear",
        choices=["linear", "linear-tight"],
        help="Factorization method: linear (LP) or linear-tight (LP + scope-restricted LMC equalities) (default: linear)")
    parser.add_argument(
        "--merge-budget", type=int, default=1,
        help="D2 scope-merge budget: max flattened scope of a merged super-family "
             "(1 = no merging; >= total atoms = exact). Default: 1.")
    parser.add_argument(
        "--solver", type=str, default="ipopt",
        choices=["ipopt", "scip"],
        help="Local credal-set solver backend: ipopt (local, default) or scip (global) (default: ipopt)")
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
    if args.n_clusters is not None:
        kwargs["n_clusters"] = args.n_clusters
    if args.cluster_representative in ["plub", "mean"]:
        kwargs["cluster_representative"] = args.cluster_representative
    if args.factorization_method != "linear":
        kwargs["factorization_method"] = args.factorization_method
    if args.merge_budget != 1:
        kwargs["merge_budget"] = args.merge_budget
    if args.solver != "ipopt":
        kwargs["solver"] = args.solver
    result = run_single(
        args.instance, args.algorithm,
        evidence=evidence, verbosity=args.verbosity,
        time_limit=args.time_limit, **kwargs)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
