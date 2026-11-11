"""Run a single inference algorithm on a single LCN instance.

Core building block for the experimental pipeline. Can be used standalone
or called from run_experiment.py.

Algorithms (``ALGORITHMS``):
    exact     direct optimization over the 2^n joint   (--solver scip|ipopt)
    ariel     ARIEL message passing on the LCN factor graph
    cjt       Credal Junction Tree, exact cluster NLP   (--solver scip|ipopt)
    ivbp      interval belief propagation on the compiled credal network
    cve       credal variable elimination, exact; --epsilon E = FPTAS (inner, factor 1+E)
    approxlp  coordinate descent over extreme points (reported as CDVE)

Build stages (``STAGES``, run with --stage instead of --algorithm) write the
caches the credal-network algorithms load: ``compile`` (the .cn interval
credal network) and ``enumerate`` (the .vtx extreme points).

Usage:
    python experiments/run_algorithm.py \
        --instance benchmarks/tree_small_fr/tree_fr_n5_1.lcn --algorithm cjt
    python experiments/run_algorithm.py --instance ... --algorithm cve \
        --epsilon 0.1                               # CVE FPTAS -> cve_e0.1
"""

import os
import argparse
import json
import multiprocessing
import signal
import time

from lcn.core.model import LCN
from lcn.inference.marginal.exact import ExactInference
from lcn.inference.marginal.ariel import ArielInference
from lcn.inference.marginal.cn.credal_network import CredalNetwork
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.marginal.cn.ivbp import IntervalBP
from lcn.inference.marginal.cn.approxlp import ApproxLP
from lcn.inference.marginal.cn.cve import CredalVE
from lcn.inference.marginal.cn.cjt import CredalJT


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

ALGORITHMS = ["exact", "ariel", "cjt", "ivbp", "cve", "approxlp"]
STAGES = ["compile", "enumerate"]

# Algorithms that run on a (compiled) credal network built from the chain-graph
# factorization. CJT uses only its structure (no interval solves, no vertices).
_CN_ALGORITHMS = {"cjt", "ivbp", "cve", "approxlp"}

# Engine backends for the algorithms that solve a nonlinear program.
ENGINE_SOLVERS = ["scip", "ipopt"]
_DEFAULT_ENGINE_SOLVER = "scip"

# Per-algorithm defaults of the variant parameters. A run whose parameters
# differ from these is a VARIANT: its output file carries a suffix (see
# ``variant_suffix``) and its record a ``variant`` dict, so variants never
# overwrite one another.
VARIANT_DEFAULTS = {
    "exact": {"solver": "scip"},
    "cjt": {"solver": "scip"},
    "cve": {"epsilon": None},
    "ivbp": {"n_iters": 100},
    "approxlp": {"n_iters": 50, "n_restarts": 8},
    "ariel": {"n_iters": 10},
}


def variant_of(algorithm, kwargs):
    """The variant parameters of a run: the algorithm's VARIANT_DEFAULTS
    overridden by any matching keys of ``kwargs``."""
    v = dict(VARIANT_DEFAULTS.get(algorithm, {}))
    for k in v:
        if kwargs.get(k) is not None:
            v[k] = kwargs[k]
    return v


def variant_suffix(algorithm, kwargs):
    """Filename suffix encoding the non-default variant parameters, e.g.
    ``cve`` with --epsilon 0.1 -> ``_e0.1``; ``cjt`` with ipopt ->
    ``_ipopt``; defaults -> ``""``."""
    v = variant_of(algorithm, kwargs)
    d = VARIANT_DEFAULTS.get(algorithm, {})
    parts = []
    if algorithm in ("exact", "cjt") and v["solver"] != d["solver"]:
        parts.append(v["solver"])
    if algorithm == "cve" and v["epsilon"]:
        parts.append(f"e{v['epsilon']:g}")
    if algorithm in ("ivbp", "approxlp", "ariel") and v["n_iters"] != d["n_iters"]:
        parts.append(f"it{v['n_iters']}")
    if algorithm == "approxlp" and v["n_restarts"] != d["n_restarts"]:
        parts.append(f"r{v['n_restarts']}")
    return ("_" + "_".join(parts)) if parts else ""


def _compute_induced_width(cn):
    """Compute the induced width (treewidth upper bound) from a CredalNetwork.

    Accepts either a CredalNetwork or a CredalNetworkVertices (the latter's
    ``.cn`` is used), so it works for a vertex-free build (e.g. CredalJT) and
    for the compile-only path that never builds vertices.

    Replays the min-fill elimination on the interaction graph built from
    the potential scopes.  The induced width is the maximum number of
    neighbours a variable has at the moment it is eliminated.
    """
    # Unwrap a CredalNetworkVertices to its CredalNetwork.
    if hasattr(cn, "cn"):
        cn = cn.cn

    # Collect the per-node CPT scopes [node] + parents from the CredalNetwork
    # factors. The factor/child order and parents match the pyAgrum bn_min arcs,
    # so the induced width is identical to a vertex-based derivation.
    scopes = []
    for factor in cn.factors:
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


def _set_memory_limit(memory_limit_gb):
    """Cap this process's virtual address space (RLIMIT_AS) at
    ``memory_limit_gb`` gigabytes. Best-effort: on platforms where RLIMIT_AS is
    unsupported or ignored (some macOS configurations) this silently degrades to
    no limit. Authoritative on Linux/RHEL 9.

    The limit is enforced *per process*. For ``compile``/``enumerate``'s
    per-family interval solves this now bounds the whole build directly: those
    solves run on a ``ThreadPoolExecutor`` in this one interpreter (not a process
    pool), so ``--n-jobs K`` no longer multiplies the resident Python heap. The
    ``enumerate`` step's LRS vertex enumeration also runs in-process (serial), so
    it too is bounded. The one out-of-band allocator RLIMIT_AS does not cover is
    the ipopt subprocess each solve spawns.

    Exceeding the limit makes Python allocations raise ``MemoryError``; native
    allocations may instead abort the process (SIGKILL/SIGSEGV/SIGABRT), which
    the parent classifies as an out-of-memory outcome.
    """
    if memory_limit_gb is None:
        return
    try:
        import resource
        nbytes = int(memory_limit_gb * (1024 ** 3))
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        # Do not raise the hard limit; clamp the soft limit to it if needed.
        if hard != resource.RLIM_INFINITY:
            nbytes = min(nbytes, hard)
        resource.setrlimit(resource.RLIMIT_AS, (nbytes, hard))
    except (ImportError, ValueError, OSError) as e:
        print(f"[warn] could not set memory limit ({memory_limit_gb} GB): {e}")


def _run_single_worker(queue, lcn_file, algorithm, evidence, verbosity, kwargs,
                       memory_limit_gb=None):
    """Worker function for multiprocessing time/memory limit enforcement."""
    # Set the address-space cap FIRST, before any numerical allocation.
    _set_memory_limit(memory_limit_gb)
    try:
        result = _run_single_impl(
            lcn_file, algorithm, evidence, verbosity, **kwargs)
    except MemoryError:
        result = {
            "algorithm": algorithm,
            "variant": variant_of(algorithm, kwargs),
            "build_time": 0.0,
            "run_time": 0.0,
            "total_time": 0.0,
            "induced_width": None,
            "status": "memory",
            "marginals": {},
            "error": f"Memory limit exceeded ({memory_limit_gb} GB)",
            "epsilon": kwargs.get("epsilon", None),
        }
    queue.put(result)


def per_solve_limit(lcn_file, time_limit, margin=0.9):
    """Per-solve time limit that keeps all 2 * n_atoms endpoint solves of an
    exact engine within ``margin * time_limit`` seconds of wall clock."""
    import contextlib
    import io
    n_atoms = 1
    try:
        lcn = LCN()
        with contextlib.redirect_stdout(io.StringIO()):
            lcn.from_lcn(file_name=lcn_file)
        n_atoms = max(1, len(lcn.atoms))
    except Exception:  # noqa: BLE001 -- fall back to one atom
        pass
    return max(1.0, margin * float(time_limit) / (2 * n_atoms))


def run_single(lcn_file, algorithm, evidence=None, verbosity=0,
               time_limit=None, memory_limit_gb=None, **kwargs):
    """
    Run one algorithm on one LCN instance, with optional time limit.

    Args:
        lcn_file: path to .lcn file
        algorithm: one of ALGORITHMS, or a build stage of STAGES
        evidence: dict of evidence (default: {})
        verbosity: 0=silent
        time_limit: max wall-clock seconds (None=unlimited). Enforced by
                    running the algorithm in a subprocess that is killed
                    if it exceeds the limit.
        memory_limit_gb: max virtual address space per process, in GB
                    (None=unlimited). Enforced via RLIMIT_AS inside the
                    subprocess (best-effort; authoritative on Linux). The
                    per-family solves are now threaded (one interpreter), so
                    n_jobs>1 no longer multiplies the resident Python heap;
                    ipopt subprocesses and the enumerate step's process-based
                    LRS remain out of RLIMIT_AS scope.
        **kwargs: algorithm-specific params

    Returns:
        dict with keys: algorithm, build_time, run_time, total_time,
        induced_width, status, marginals, error, epsilon. status is one of
        {ok, error, timeout, memory}.
    """
    if evidence is None:
        evidence = {}

    # The exact engines take a PER-SOLVE limit and solve two endpoints per atom.
    # Split the wall-clock budget across those solves (keeping a 10% margin
    # for building the model) so the engine always stops cleanly -- returning
    # sound partial bounds, marked uncertified -- before the watchdog kills the
    # run and discards everything. (No-op for other algorithms.)
    if (time_limit is not None and algorithm in ("exact", "cjt")
            and "engine_time_limit" not in kwargs):
        kwargs["engine_time_limit"] = per_solve_limit(lcn_file, time_limit)

    # Fast path: no isolation needed when neither limit is set.
    if time_limit is None and memory_limit_gb is None:
        return _run_single_impl(
            lcn_file, algorithm, evidence, verbosity, **kwargs)

    # Run in a subprocess so both the wall-clock watchdog (parent-side join) and
    # the memory cap (RLIMIT_AS set inside the worker) can be enforced.
    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_run_single_worker,
        args=(queue, lcn_file, algorithm, evidence, verbosity, kwargs,
              memory_limit_gb))
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
            "variant": variant_of(algorithm, kwargs),
            "build_time": 0.0,
            "run_time": 0.0,
            "total_time": round(time_limit, 4) if time_limit else 0.0,
            "induced_width": None,
            "status": "timeout",
            "marginals": {},
            "error": (f"Time limit exceeded ({time_limit}s)"
                      if time_limit else "Time limit exceeded"),
            "epsilon": kwargs.get("epsilon", None),
        }

    if not queue.empty():
        return queue.get_nowait()

    # The worker exited without putting a result. When a memory limit is in
    # effect and the process was killed by a signal (native allocation abort or
    # the OOM killer), classify it as an out-of-memory outcome; RLIMIT_AS
    # breaches typically surface as SIGKILL/SIGSEGV/SIGABRT.
    exitcode = proc.exitcode
    killed_by_signal = exitcode is not None and exitcode < 0
    oom_signals = {-signal.SIGKILL, -signal.SIGSEGV, -signal.SIGABRT}
    if memory_limit_gb is not None and killed_by_signal \
            and exitcode in oom_signals:
        return {
            "algorithm": algorithm,
            "variant": variant_of(algorithm, kwargs),
            "build_time": 0.0,
            "run_time": 0.0,
            "total_time": 0.0,
            "induced_width": None,
            "status": "memory",
            "marginals": {},
            "error": (f"Memory limit exceeded ({memory_limit_gb} GB); "
                      f"worker killed by signal {-exitcode}"),
            "epsilon": kwargs.get("epsilon", None),
        }

    sig_note = (f" (killed by signal {-exitcode})" if killed_by_signal
                else f" (exit code {exitcode})")
    return {
        "algorithm": algorithm,
        "variant": variant_of(algorithm, kwargs),
        "build_time": 0.0,
        "run_time": 0.0,
        "total_time": 0.0,
        "induced_width": None,
        "status": "error",
        "marginals": {},
        "error": f"Worker process exited without result{sig_note}",
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
        "variant": variant_of(algorithm, kwargs),
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

        if algorithm == "compile":
            # Compile the LCN into its credal network (chain-graph factorization
            # + interval local credal sets) and save it alongside the .lcn as a
            # .cn, recording the per-family solve wall-clock as compile_time.
            # The expensive per-family interval solves parallelize over n_jobs
            # worker threads (each min/max solve is one task; ipopt runs as a
            # GIL-releasing subprocess).
            fact_method = kwargs.get("factorization_method", "linear")
            n_jobs = kwargs.get("n_jobs", 1)
            cn_solver = kwargs.get("cn_solver", "ipopt")
            cn_time_limit = kwargs.get("solver_time_limit", None)
            cn_gap_tol = kwargs.get("gap_tol", 0.0)
            merge_budget = kwargs.get("merge_budget", 1)
            use_highs = kwargs.get("use_highs", True)

            t_compile = time.perf_counter()
            cn = CredalNetwork.from_lcn(
                lcn_model, method=fact_method, solver=cn_solver,
                time_limit=cn_time_limit, gap_tol=cn_gap_tol,
                n_jobs=n_jobs, merge_budget=merge_budget,
                solve_families=True, verbosity=verbosity, use_highs=use_highs)
            compile_time = time.perf_counter() - t_compile

            cn_path = os.path.splitext(lcn_file)[0] + ".cn"
            cn.save_cn(cn_path, method=fact_method, merge_budget=merge_budget,
                       solver=cn_solver, compile_time=round(compile_time, 4),
                       n_jobs=n_jobs)

            result["build_time"] = round(compile_time, 4)
            result["run_time"] = 0.0
            result["induced_width"] = _compute_induced_width(cn)
            result["cn_path"] = cn_path
            result["compile_time"] = round(compile_time, 4)
            result["total_time"] = round(compile_time, 4)
            # No marginals for a compile step.
            result["marginals"] = {}
            return result

        elif algorithm == "enumerate":
            # Enumerate (LRS) the extreme points of every local credal set and
            # save them alongside the .lcn as a .vtx, recording the enumeration
            # wall-clock. The LRS enumeration runs serially (it is cheap and
            # treewidth-bounded, and pyAgrum's LRS holds the GIL and is not
            # thread-safe); n_jobs affects only the compile-time per-family
            # solves. Records both compile_time and enumeration_time so downstream
            # engines can report build_time = compile_time + enumeration_time.
            fact_method = kwargs.get("factorization_method", "linear")
            n_jobs = kwargs.get("n_jobs", 1)
            cn_solver = kwargs.get("cn_solver", "ipopt")
            cn_time_limit = kwargs.get("solver_time_limit", None)
            cn_gap_tol = kwargs.get("gap_tol", 0.0)
            merge_budget = kwargs.get("merge_budget", 1)
            use_cache = kwargs.get("cache", True)
            use_highs = kwargs.get("use_highs", True)

            cn_path = os.path.splitext(lcn_file)[0] + ".cn"

            # Step 1: ensure a matching .cn exists on disk (so both artifacts are
            # produced and the .vtx carries a real compile_time). Reuse a valid
            # cached .cn; otherwise compile once and save it.
            compile_time = None
            cn_meta = (CredalNetwork.cn_metadata(cn_path) if use_cache else None)
            if cn_meta is not None and cn_meta.get("method") == fact_method \
                    and (cn_meta.get("merge_budget") or 1) == (merge_budget or 1) \
                    and cn_meta.get("solver") == cn_solver:
                compile_time = cn_meta.get("compile_time")
            else:
                t_compile = time.perf_counter()
                cn_obj = CredalNetwork.from_lcn(
                    lcn_model, method=fact_method, solver=cn_solver,
                    time_limit=cn_time_limit, gap_tol=cn_gap_tol,
                    n_jobs=n_jobs, merge_budget=merge_budget,
                    solve_families=True, verbosity=verbosity, use_highs=use_highs)
                compile_time = time.perf_counter() - t_compile
                cn_obj.save_cn(cn_path, method=fact_method,
                               merge_budget=merge_budget, solver=cn_solver,
                               compile_time=round(compile_time, 4), n_jobs=n_jobs)

            # Step 2: enumerate. With the .cn now present, from_lcn hits the .cn
            # cache (sets cnv.compile_time) and runs the parallel LRS.
            cnv = CredalNetworkVertices.from_lcn(
                lcn_model, method=fact_method, solver=cn_solver,
                time_limit=cn_time_limit, gap_tol=cn_gap_tol,
                n_jobs=n_jobs, merge_budget=merge_budget,
                enumerate_vertices=True, solve_families=True,
                lcn_file=lcn_file, cache=use_cache, verbosity=verbosity,
                use_highs=use_highs)
            # Fall back to the locally-measured compile_time if the cache path
            # did not populate it (e.g. --no-cache).
            if cnv.compile_time is None:
                cnv.compile_time = (round(compile_time, 4)
                                    if compile_time is not None else None)

            vtx_path = os.path.splitext(lcn_file)[0] + ".vtx"
            cnv.save_vtx(
                vtx_path, method=fact_method, merge_budget=merge_budget,
                solver=cn_solver,
                enumeration_time=(round(cnv.enumeration_time, 4)
                                  if cnv.enumeration_time is not None else None),
                compile_time=(round(cnv.compile_time, 4)
                              if cnv.compile_time is not None else None),
                n_jobs=n_jobs)

            n_vertices = sum(len(vs) for cfgs in cnv.extreme_points.values()
                             for vs in cfgs.values())
            result["build_time"] = round(cnv.build_time, 4)
            result["run_time"] = 0.0
            result["induced_width"] = _compute_induced_width(cnv)
            result["vtx_path"] = vtx_path
            result["enumeration_time"] = (round(cnv.enumeration_time, 4)
                                          if cnv.enumeration_time is not None
                                          else None)
            result["compile_time"] = (round(cnv.compile_time, 4)
                                      if cnv.compile_time is not None else None)
            result["n_vertices"] = n_vertices
            result["loaded_vertices_from_cache"] = bool(
                cnv.loaded_vertices_from_cache)
            result["total_time"] = round(cnv.build_time, 4)
            result["marginals"] = {}
            return result

        elif algorithm == "exact":
            # Direct optimization over the 2^n joint: --solver scip (spatial
            # branch-and-bound, certified per endpoint) or ipopt (local, with
            # the SLSQP fallback; uncertified).
            solver = kwargs.get("solver", _DEFAULT_ENGINE_SOLVER)
            exact_solver = "global" if solver == "scip" else "local"
            algo = ExactInference(lcn=lcn_model)
            raw = algo.run(
                evidence=evidence, solver=exact_solver, verbosity=verbosity,
                debug=(exact_solver == "local"),
                time_limit=kwargs.get("engine_time_limit", 3600.0))
            marginals = _filter_singletons(raw)
            if exact_solver == "global" and getattr(algo, "status", None):
                # Per-endpoint certification, used by the analysis to restrict
                # a reference to certified bounds.
                result["exact_status"] = {
                    a: {s: algo.status[a][s][2] for s in ("min", "max")}
                    for a in algo.status}
            t_end = time.time()
            result["run_time"] = round(t_end - t_start, 4)

        elif algorithm == "ariel":
            n_iters = kwargs.get("n_iters") or 10
            threshold = kwargs.get("threshold", 1e-6)
            algo = ArielInference(lcn=lcn_model)
            raw = algo.run(
                n_iters=n_iters, threshold=threshold,
                evidence=evidence, verbosity=verbosity)
            marginals = _filter_singletons(raw)
            t_end = time.time()
            result["run_time"] = round(t_end - t_start, 4)

        elif algorithm in _CN_ALGORITHMS:
            # Build the credal network vertices (chain-graph factorization +
            # interval local credal sets + extreme-point enumeration). The
            # whole pipeline is timed by CredalNetworkVertices.from_lcn via
            # perf_counter; build_time is reported in the experiments.
            fact_method = kwargs.get("factorization_method", "linear")
            n_jobs = kwargs.get("n_jobs", 1)
            cn_solver = kwargs.get("cn_solver", "ipopt")
            cn_time_limit = kwargs.get("solver_time_limit", None)
            cn_gap_tol = kwargs.get("gap_tol", 0.0)
            merge_budget = kwargs.get("merge_budget", 1)
            # CredalJT formulates its NLP from the LCN sentences and LMC
            # equalities and needs only the chain-graph STRUCTURE -- not the
            # per-family interval local credal sets and not the enumerated
            # extreme points. So for cjt skip both the per-family min/max
            # interval solves (solve_families=False) and the LRS vertex
            # enumeration (which solve_families=False forces off anyway). This
            # keeps the reported build_time to just the symbolic factorization.
            is_cjt = algorithm == "cjt"
            enumerate_vertices = not is_cjt
            solve_families = not is_cjt
            # Transparent .cn cache: reuse a compiled network next to the .lcn
            # when its (method, merge_budget, solver) match. Disabled via
            # --no-cache (kwargs["cache"]=False) and inherently off for cjt
            # (structure-only build carries no cacheable intervals).
            use_cache = kwargs.get("cache", True) and solve_families
            use_highs = kwargs.get("use_highs", True)
            cnv = CredalNetworkVertices.from_lcn(
                lcn_model, method=fact_method, solver=cn_solver,
                time_limit=cn_time_limit, gap_tol=cn_gap_tol,
                n_jobs=n_jobs, merge_budget=merge_budget,
                enumerate_vertices=enumerate_vertices,
                solve_families=solve_families,
                lcn_file=lcn_file, cache=use_cache, verbosity=verbosity,
                use_highs=use_highs)
            result["build_time"] = round(cnv.build_time, 4)
            result["induced_width"] = _compute_induced_width(cnv)
            # Surface cache provenance for analysis (both cache layers).
            result["loaded_from_cache"] = bool(cnv.loaded_from_cache)
            result["loaded_vertices_from_cache"] = bool(
                cnv.loaded_vertices_from_cache)
            if cnv.compile_time is not None:
                result["compile_time"] = round(cnv.compile_time, 4)
            if cnv.enumeration_time is not None:
                result["enumeration_time"] = round(cnv.enumeration_time, 4)

            # Run the algorithm — timed separately
            t_run_start = time.time()
            if algorithm == "ivbp":
                algo = IntervalBP(cnv=cnv)
                raw = algo.run(
                    evidence=evidence,
                    n_iters=kwargs.get("n_iters") or 100,
                    threshold=kwargs.get("threshold", 1e-6),
                    verbosity=verbosity)
                # Exact iff the credal network is a polytree (and the sweeps
                # reached the fixed point); otherwise loopy, no guarantee.
                result["ivbp_exact"] = bool(algo.exact)
                result["ivbp_converged"] = bool(algo.converged)
                result["ivbp_iterations"] = int(algo.iterations)
            elif algorithm == "approxlp":
                v = variant_of("approxlp", kwargs)
                algo = ApproxLP(cnv=cnv)
                raw = algo.run(
                    evidence=evidence, n_iters=v["n_iters"],
                    n_restarts=v["n_restarts"], verbosity=verbosity)
                result["approxlp_sweeps"] = int(algo.sweeps)
                result["approxlp_eliminations"] = int(algo.n_eliminations)
            elif algorithm == "cve":
                # epsilon None/0 -> exact CVE; > 0 -> the FPTAS (inner bounds
                # within a factor 1 + epsilon of the exact ones).
                v = variant_of("cve", kwargs)
                algo = CredalVE(cnv=cnv)
                raw = algo.run(evidence=evidence, epsilon=v["epsilon"],
                               verbosity=verbosity)
                result["cve_alpha"] = algo.alpha
                result["cve_quotient_ops"] = algo.n_quotient_ops
                result["cve_guarantee"] = algo.guarantee
                result["max_potential_size"] = algo.max_potential_size
            elif algorithm == "cjt":
                algo = CredalJT(cnv=cnv)
                cjt_kw = {}
                if kwargs.get("engine_time_limit") is not None:
                    cjt_kw["time_limit"] = kwargs["engine_time_limit"]
                if kwargs.get("cert_tol") is not None:
                    cjt_kw["cert_tol"] = kwargs["cert_tol"]
                raw = algo.run(
                    evidence=evidence,
                    solver=kwargs.get("solver", _DEFAULT_ENGINE_SOLVER),
                    verbosity=verbosity, **cjt_kw)
                result["induced_width"] = algo.induced_width
                # Size of the shared constraint NLP (None on exact fallback).
                result["nlp_stats"] = algo.nlp_stats
                # Guarantees: H3 triage, per-solve certification, and whether
                # the reported bounds are certified EXACT over D_LCN.
                result["cjt_h3"] = algo.h3
                result["cjt_structurally_exact"] = algo.d5_exact
                result["cjt_certified"] = algo.certified
                result["cjt_exact"] = algo.exact
                result["cjt_solves"] = [
                    {k: s.get(k) for k in ("atom", "sense", "value", "dual",
                                           "gap", "termination", "certified")}
                    for s in algo.solve_status]
            t_run_end = time.time()
            result["run_time"] = round(t_run_end - t_run_start, 4)

            marginals = _filter_singletons(raw)

        else:
            raise ValueError(f"Unknown algorithm: {algorithm} "
                             f"(choose from {ALGORITHMS} or stages {STAGES})")

        result["total_time"] = round(
            result["build_time"] + result["run_time"], 4)
        result["marginals"] = _marginals_to_dict(marginals)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["total_time"] = round(time.time() - t_start, 4)

    return result


def add_algorithm_arguments(parser):
    """Variant / build flags shared by run_algorithm.py and run_experiment.py."""
    g = parser.add_argument_group("engine and variant parameters")
    g.add_argument(
        "--solver", type=str, default=_DEFAULT_ENGINE_SOLVER,
        choices=ENGINE_SOLVERS,
        help="NLP backend for exact and cjt: scip (global, certified per "
             "endpoint; default) or ipopt (local, uncertified)")
    g.add_argument(
        "--cert-tol", type=float, default=None,
        help="cjt: primal-dual tolerance under which a SCIP endpoint counts as "
             "certified (default: 1e-4)")
    g.add_argument(
        "--epsilon", type=float, default=None,
        help="cve: FPTAS of Maua, de Campos & Zaffalon (2012) -- inner bounds "
             "within a factor 1 + EPSILON of the exact ones (default: exact)")
    g.add_argument(
        "--n-iters", type=int, default=None,
        help="ivbp / approxlp / ariel: iteration cap (defaults 100 / 50 / 10)")
    g.add_argument(
        "--n-restarts", type=int, default=None,
        help="approxlp: random-vertex restarts besides the centre start "
             "(default: 8; 0 = centre start only)")
    b = parser.add_argument_group(
        "credal-network build (ivbp, cve, approxlp, stages)")
    b.add_argument(
        "--factorization-method", type=str, default="linear",
        choices=["linear", "linear-tight"],
        help="Factorization method: linear (LP) or linear-tight (LP + "
             "scope-restricted LMC equalities) (default: linear)")
    b.add_argument(
        "--merge-budget", type=int, default=1,
        help="Scope-merge budget: max flattened scope of a merged "
             "super-family (1 = no merging). Default: 1.")
    b.add_argument(
        "--cn-solver", type=str, default="ipopt", choices=["ipopt", "scip"],
        help="Solver for the per-family local credal-set problems "
             "(default: ipopt)")
    b.add_argument(
        "--n-jobs", type=int, default=1,
        help="Worker threads for the per-family credal-set solves (default: 1)")
    b.add_argument(
        "--no-cache", action="store_true",
        help="Ignore any compiled .cn/.vtx next to the .lcn and recompute")
    b.add_argument(
        "--no-highs", action="store_true",
        help="Use the ipopt subprocess instead of in-process HiGHS for the "
             "per-family LPs (numerically identical, slower)")


def kwargs_from_args(args):
    """Translate the shared CLI flags into run_single keyword arguments."""
    kwargs = {"solver": args.solver}
    for name in ("cert_tol", "epsilon", "n_iters", "n_restarts"):
        if getattr(args, name) is not None:
            kwargs[name] = getattr(args, name)
    if args.factorization_method != "linear":
        kwargs["factorization_method"] = args.factorization_method
    if args.merge_budget != 1:
        kwargs["merge_budget"] = args.merge_budget
    if args.cn_solver != "ipopt":
        kwargs["cn_solver"] = args.cn_solver
    if args.n_jobs != 1:
        kwargs["n_jobs"] = args.n_jobs
    if args.no_cache:
        kwargs["cache"] = False
    if args.no_highs:
        kwargs["use_highs"] = False
    return kwargs


def main():
    parser = argparse.ArgumentParser(
        description="Run a single inference algorithm on an LCN instance.")
    parser.add_argument(
        "--instance", type=str, required=True,
        help="Path to the .lcn file")
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument(
        "--algorithm", type=str, choices=ALGORITHMS,
        help="Inference algorithm to run")
    what.add_argument(
        "--stage", type=str, choices=STAGES,
        help="Build stage to run instead (writes the .cn / .vtx cache)")
    parser.add_argument(
        "--evidence", type=str, default="{}",
        help="Evidence as JSON string (default: {})")
    add_algorithm_arguments(parser)
    parser.add_argument(
        "--time-limit", type=float, default=None,
        help="Wall-clock time limit in seconds per instance (default: unlimited)")
    parser.add_argument(
        "--memory-limit", type=float, default=None,
        help="Memory limit in GB per process, enforced via RLIMIT_AS "
             "(default: unlimited; best-effort off Linux)")
    parser.add_argument(
        "--num-threads", type=int, default=1,
        help="Number of threads for BLAS/LAPACK/ipopt (default: 1)")
    parser.add_argument(
        "--verbosity", type=int, default=2,
        help="Verbosity level (default: 2)")
    args = parser.parse_args()

    set_num_threads(args.num_threads)
    # stdout carries exactly one JSON document. The library prints progress
    # (e.g. LCN parsing) unconditionally, and with a time/memory limit the run
    # happens in a child process, so stdout is redirected to stderr at the file
    # descriptor level (inherited by children) and restored for the result.
    import sys
    sys.stdout.flush()
    saved_fd = os.dup(1)
    os.dup2(2, 1)
    try:
        result = run_single(
            args.instance, args.algorithm or args.stage,
            evidence=json.loads(args.evidence), verbosity=args.verbosity,
            time_limit=args.time_limit, memory_limit_gb=args.memory_limit,
            **kwargs_from_args(args))
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
