"""Rewrite experiments/run_algorithm.py to the six-algorithm harness (one-off)."""
import sys

P = sys.argv[1]
s = open(P).read()


def sub(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:70], s.count(old))
    s = s.replace(old, new)


def cut(start, end, new):
    global s
    a = s.index(start)
    b = s.index(end, a)
    s = s[:a] + new + s[b:]


sub('''"""Run a single inference algorithm on a single LCN instance.

Core building block for the experimental pipeline. Can be used standalone
or called from run_experiment.py.

Usage:
    python experiments/run_algorithm.py \\
        --instance benchmarks/chain/chain_n10_1.lcn \\
        --algorithm ccte
"""''', '''"""Run a single inference algorithm on a single LCN instance.

Core building block for the experimental pipeline. Can be used standalone
or called from run_experiment.py.

Algorithms (``ALGORITHMS``):
    exact     direct optimization over the 2^n joint   (--solver scip|ipopt)
    ariel     ARIEL message passing on the LCN factor graph
    cjt       Credal Junction Tree, exact cluster NLP   (--solver scip|ipopt)
    ivbp      interval belief propagation on the compiled credal network
    cve       credal variable elimination               (--n-clusters, --cluster-rep, --epsilon)
    approxlp  coordinate descent over extreme points (reported as CDVE)

Build stages (``STAGES``, run with --stage instead of --algorithm) write the
caches the credal-network algorithms load: ``compile`` (the .cn interval
credal network) and ``enumerate`` (the .vtx extreme points).

Usage:
    python experiments/run_algorithm.py \\
        --instance benchmarks/tree_small_fr/tree_fr_n5_1.lcn --algorithm cjt
    python experiments/run_algorithm.py --instance ... --algorithm cve \\
        --n-clusters 10 --cluster-rep mean          # CVE_cm
"""''')

sub('''from lcn.inference.marginal.cn.ivbp import IntervalBP
from lcn.inference.marginal.cn.ccte import CredalCTE
from lcn.inference.marginal.cn.approxlp import ApproxLP
from lcn.inference.marginal.cn.ijgp import CredalIJGP
from lcn.inference.marginal.cn.cve import CredalVE''', '''from lcn.inference.marginal.cn.ivbp import IntervalBP
from lcn.inference.marginal.cn.approxlp import ApproxLP
from lcn.inference.marginal.cn.cve import CredalVE''')

cut('''ALGORITHMS = ["compile", "enumerate",''', '''def _compute_induced_width(cn):''',
    '''ALGORITHMS = ["exact", "ariel", "cjt", "ivbp", "cve", "approxlp"]
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
    "cve": {"n_clusters": 0, "cluster_rep": "mean", "epsilon": None},
    "ivbp": {"n_iters": 100},
    "approxlp": {"n_iters": 50},
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
    ``cve`` with --n-clusters 10 -> ``_c10mean``; ``cjt`` with ipopt ->
    ``_ipopt``; defaults -> ``""``."""
    v = variant_of(algorithm, kwargs)
    d = VARIANT_DEFAULTS.get(algorithm, {})
    parts = []
    if algorithm in ("exact", "cjt") and v["solver"] != d["solver"]:
        parts.append(v["solver"])
    if algorithm == "cve":
        if v["n_clusters"]:
            parts.append(f"c{v['n_clusters']}{v['cluster_rep']}")
        if v["epsilon"] is not None:
            parts.append(f"e{v['epsilon']:g}")
    if algorithm in ("ivbp", "approxlp", "ariel") and v["n_iters"] != d["n_iters"]:
        parts.append(f"it{v['n_iters']}")
    return ("_" + "_".join(parts)) if parts else ""


''')

sub('''    Args:
        lcn_file: path to .lcn file
        algorithm: one of "exact_l", "exact_g", "ariel", "ibp", "ccte",
                   "ccte_e", "approxlp", "cve", "cjt_l", "cjt_g" (see
                   ALGORITHMS; "cjt" is a back-compat alias for "cjt_g")''',
    '''    Args:
        lcn_file: path to .lcn file
        algorithm: one of ALGORITHMS, or a build stage of STAGES''')
sub('''    if (time_limit is not None and algorithm in ("exact_l", "exact_g")
            and "exact_time_limit" not in kwargs):
        kwargs["exact_time_limit"] = float(time_limit)''',
    '''    if (time_limit is not None and algorithm in ("exact", "cjt")
            and "engine_time_limit" not in kwargs):
        kwargs["engine_time_limit"] = float(time_limit)''')

sub('''    result = {
        "algorithm": algorithm,
        "build_time": 0.0,''', '''    result = {
        "algorithm": algorithm,
        "variant": variant_of(algorithm, kwargs),
        "build_time": 0.0,''')

# The credal-network build solver is "cn_solver" everywhere it is read.
sub('''cn_solver = kwargs.get("solver", "ipopt")''',
    '''cn_solver = kwargs.get("cn_solver", "ipopt")''', cnt=3)

cut('''        elif algorithm in ("exact_l", "exact_g"):''', '''        elif algorithm == "ariel":''',
    '''        elif algorithm == "exact":
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

''')

sub('''            n_iters = kwargs.get("n_iters", 10)
            threshold = kwargs.get("threshold", 1e-6)
            algo = ArielInference(lcn=lcn_model)''',
    '''            n_iters = kwargs.get("n_iters") or 10
            threshold = kwargs.get("threshold", 1e-6)
            algo = ArielInference(lcn=lcn_model)''')

sub('''        elif algorithm in _CVE_ALGORITHMS:''', '''        elif algorithm in _CN_ALGORITHMS:''')
sub('''            # CredalJT (D5) formulates''', '''            # CredalJT formulates''')
sub('''            is_cjt = algorithm in _CJT_ALGORITHMS''', '''            is_cjt = algorithm == "cjt"''')

cut('''            if algorithm == "ibp":''', '''            t_run_end = time.time()''',
    '''            if algorithm == "ivbp":
                algo = IntervalBP(cnv=cnv)
                raw = algo.run(
                    evidence=evidence,
                    n_iters=kwargs.get("n_iters") or 100,
                    threshold=kwargs.get("threshold", 1e-6),
                    method="interval", verbosity=verbosity)
            elif algorithm == "approxlp":
                algo = ApproxLP(cnv=cnv)
                raw = algo.run(
                    evidence=evidence, n_iters=kwargs.get("n_iters") or 50,
                    verbosity=verbosity)
            elif algorithm == "cve":
                # n_clusters = 0 -> exact CVE; > 0 -> clustered potentials
                # (CVE_cm with cluster_rep="mean", CVE_cp with "plub").
                v = variant_of("cve", kwargs)
                algo = CredalVE(cnv=cnv)
                raw = algo.run(
                    evidence=evidence, epsilon=v["epsilon"], coupling="off",
                    n_clusters=v["n_clusters"],
                    cluster_representative=v["cluster_rep"],
                    verbosity=verbosity)
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
''')

sub('''        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")''',
    '''        else:
            raise ValueError(f"Unknown algorithm: {algorithm} "
                             f"(choose from {ALGORITHMS} or stages {STAGES})")''')

a = s.index('''def main():''')
s = s[:a] + '''def add_algorithm_arguments(parser):
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
        "--n-clusters", type=int, default=None,
        help="cve: cluster each intermediate potential into N functions "
             "(0 = exact CVE, default; CVE_cm = --n-clusters 10 --cluster-rep mean)")
    g.add_argument(
        "--cluster-rep", type=str, default=None, choices=["mean", "plub"],
        help="cve: cluster representative (default: mean)")
    g.add_argument(
        "--epsilon", type=float, default=None,
        help="cve: epsilon-approximate dominance pruning (default: exact)")
    g.add_argument(
        "--n-iters", type=int, default=None,
        help="ivbp / approxlp / ariel: iteration cap (defaults 100 / 50 / 10)")
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
    for name in ("cert_tol", "n_clusters", "epsilon", "n_iters"):
        if getattr(args, name) is not None:
            kwargs[name] = getattr(args, name)
    if args.cluster_rep is not None:
        kwargs["cluster_rep"] = args.cluster_rep
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
    result = run_single(
        args.instance, args.algorithm or args.stage,
        evidence=json.loads(args.evidence), verbosity=args.verbosity,
        time_limit=args.time_limit, memory_limit_gb=args.memory_limit,
        **kwargs_from_args(args))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
'''
open(P, 'w').write(s)
print("ok")
