"""Experiment runner: run inference algorithms on benchmark instances.

Each (benchmark, algorithm) pair writes to its own JSONL file so that
multiple benchmarks and algorithms can run in parallel without conflicts.

Output structure:
    {output-dir}/{benchmark-name}/{algorithm}.jsonl

Usage:
    # Run all algorithms on all benchmarks
    python experiments/run_experiment.py --input-dir benchmarks/chain

    # Parallel across benchmarks and algorithms (safe — separate files)
    python experiments/run_experiment.py --input-dir benchmarks/chain --algorithms ccte &
    python experiments/run_experiment.py --input-dir benchmarks/chain --algorithms ibp &
    python experiments/run_experiment.py --input-dir benchmarks/polytree --algorithms ccte &
    python experiments/run_experiment.py --input-dir benchmarks/polytree --algorithms ibp &
"""

import argparse
import glob
import json
import os
import re
import sys

from run_algorithm import run_single, set_num_threads, ALGORITHMS as ALL_ALGORITHMS

APPROX_ALGORITHMS = ["ariel", "ibp", "ccte", "ccte_e", "approxlp"]


def _parse_instance_info(filepath):
    """Extract graph_type and num_vars from filename like chain_n10_1.lcn."""
    basename = os.path.basename(filepath)
    match = re.match(r"(\w+)_n(\d+)_(\d+)\.lcn$", basename)
    if match:
        return match.group(1), int(match.group(2))
    return "unknown", 0


def _load_completed(output_file):
    """Load already-completed instance paths from a JSONL file."""
    completed = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    completed.add(rec["instance"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def _benchmark_name(input_dir):
    """Derive a benchmark name from the input directory path.
    e.g. 'benchmarks/chain' -> 'chain', 'benchmarks' -> 'benchmarks'
    """
    name = os.path.basename(os.path.normpath(input_dir))
    return name if name else "default"


def main():
    parser = argparse.ArgumentParser(
        description="Run inference algorithms on benchmark LCN instances.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Parallel usage — each combination gets its own output file:
  python experiments/run_experiment.py --input-dir benchmarks/chain --algorithms ccte &
  python experiments/run_experiment.py --input-dir benchmarks/chain --algorithms ibp &
  python experiments/run_experiment.py --input-dir benchmarks/polytree --algorithms ccte &
  python experiments/run_experiment.py --input-dir benchmarks/polytree --algorithms ibp &
""")
    parser.add_argument(
        "--input-dir", type=str, default="benchmarks",
        help="directory with .lcn files (default: benchmarks)")
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="root output directory (default: results)")
    parser.add_argument(
        "--benchmark", type=str, default=None,
        help="benchmark name for output subdir (default: derived from input-dir)")
    parser.add_argument(
        "--algorithms", type=str, nargs="+", default=APPROX_ALGORITHMS,
        choices=ALL_ALGORITHMS,
        help="algorithms to run (default: all approximate)")
    parser.add_argument(
        "--exact-threshold", type=int, default=15,
        help="max num_vars for running exact inference (default: 15)")
    parser.add_argument(
        "--epsilon", type=float, default=None,
        help="epsilon for ccte_e algorithm (default: None)")
    parser.add_argument(
        "--ibound", type=int, default=2,
        help="ibound for ijgp, ijgp_e algorithms (default: 2)")
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
        help="Factorization method: linear or linear-tight (default: linear)")
    parser.add_argument(
        "--merge-budget", type=int, default=1,
        help="D2 scope-merge budget: max flattened scope of a merged "
             "super-family (1 = no merging). Default: 1.")
    parser.add_argument(
        "--solver", type=str, default="ipopt",
        choices=["ipopt", "scip"],
        help="Local credal-set solver backend: ipopt (local, default) or scip (global)")
    parser.add_argument(
        "--evidence", type=str, default="{}",
        help="evidence as JSON string (default: {})")
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="worker processes for the per-family credal-set solves during "
             "compilation / credal-network build (default: 1)")
    parser.add_argument(
        "--no-cache", action="store_true",
        help="ignore any compiled .cn next to each .lcn and (re)compute the "
             "credal network in memory (default: use the cache when present)")
    parser.add_argument(
        "--time-limit", type=float, default=None,
        help="time limit in seconds per instance per algorithm (default: unlimited)")
    parser.add_argument(
        "--num-threads", type=int, default=1,
        help="number of threads for BLAS/LAPACK/ipopt (default: 1)")
    parser.add_argument(
        "--verbosity", type=int, default=2,
        help="0=silent, 1=progress, 2=detailed (default: 2)")
    args = parser.parse_args()

    set_num_threads(args.num_threads)
    evidence = json.loads(args.evidence)

    # Derive benchmark name
    bench_name = args.benchmark or _benchmark_name(args.input_dir)

    # Find all .lcn files, sorted by (num_vars, filename) so that
    # instances are processed in increasing problem size
    pattern = os.path.join(args.input_dir, "**", "*.lcn")
    instances = glob.glob(pattern, recursive=True)
    instances.sort(key=lambda p: (_parse_instance_info(p)[1],
                                  os.path.basename(p)))
    if not instances:
        print(f"No .lcn files found under {args.input_dir}")
        sys.exit(1)

    total = len(instances)

    for algo in args.algorithms:
        # Output: {output_dir}/{benchmark}/{algorithm}.jsonl
        bench_dir = os.path.join(args.output_dir, bench_name)
        os.makedirs(bench_dir, exist_ok=True)
        out_path = os.path.join(bench_dir, f"{algo}.jsonl")

        # Load already-completed instances for resume
        completed = _load_completed(out_path)
        if completed and args.verbosity > 0:
            print(f"[{bench_name}/{algo}] Resuming: {len(completed)} "
                  f"instances already in {out_path}")

        outf = open(out_path, "a")
        try:
            for idx, instance in enumerate(instances, 1):
                graph_type, num_vars = _parse_instance_info(instance)
                basename = os.path.basename(instance)

                # Skip exact (both backends) for large instances
                if algo in ("exact_l", "exact_g") and \
                        num_vars > args.exact_threshold:
                    continue

                # Skip already completed
                if instance in completed:
                    if args.verbosity > 0:
                        print(f"[{bench_name}/{algo}] [{idx}/{total}] "
                              f"{basename} | skip")
                    continue

                # Build kwargs for algorithm-specific params
                kwargs = {}
                if args.epsilon is not None:
                    kwargs["epsilon"] = args.epsilon
                if args.ibound is not None:
                    kwargs["ibound"] = args.ibound
                if args.factorization_method != "linear":
                    kwargs["factorization_method"] = args.factorization_method
                if args.merge_budget != 1:
                    kwargs["merge_budget"] = args.merge_budget
                if args.solver != "ipopt":
                    kwargs["solver"] = args.solver
                if args.n_jobs != 1:
                    kwargs["n_jobs"] = args.n_jobs
                if args.no_cache:
                    kwargs["cache"] = False

                result = run_single(
                    instance, algo, evidence=evidence,
                    verbosity=args.verbosity,
                    time_limit=args.time_limit, **kwargs)

                # Enrich with instance metadata
                result["instance"] = instance
                result["benchmark"] = bench_name
                result["graph_type"] = graph_type
                result["num_vars"] = num_vars
                result["evidence"] = evidence

                # Write to JSONL
                outf.write(json.dumps(result) + "\n")
                outf.flush()

                if args.verbosity > 0:
                    status = result["status"]
                    bt = result["build_time"]
                    rt = result["run_time"]
                    tt = result["total_time"]
                    if bt > 0:
                        print(f"[{bench_name}/{algo}] [{idx}/{total}] "
                              f"{basename} | build={bt:.2f}s run={rt:.2f}s "
                              f"total={tt:.2f}s {status}")
                    else:
                        print(f"[{bench_name}/{algo}] [{idx}/{total}] "
                              f"{basename} | {tt:.2f}s {status}")

        finally:
            outf.close()

        if args.verbosity > 0:
            print(f"[{bench_name}/{algo}] Done. Results in {out_path}\n")


if __name__ == "__main__":
    main()
