"""Experiment runner: run inference algorithms on benchmark instances.

Each (benchmark, algorithm) pair writes to its own JSONL file so that
multiple benchmarks and algorithms can run in parallel without conflicts.

Output structure:
    {output-dir}/{benchmark-name}/{algorithm}{variant-suffix}.jsonl

where the suffix encodes non-default variant flags (e.g. ``cve_e0.1``,
``cjt_ipopt``), so variants of one algorithm never share a file.

Usage:
    # The reference and the approximate methods on one benchmark set
    python experiments/run_experiment.py --input-dir benchmarks/tree_small_fr \
        --algorithms cjt ariel ivbp cve approxlp

    # CVE FPTAS (inner bounds within a factor 1 + 0.1) -- written to cve_e0.1.jsonl
    python experiments/run_experiment.py --input-dir benchmarks/tree_small_fr \
        --algorithms cve --epsilon 0.1

    # Build stages (write the .cn / .vtx caches the credal-network methods load)
    python experiments/run_experiment.py --input-dir benchmarks/tree_small_fr \
        --stages compile enumerate

    # Parallel across benchmarks and algorithms (safe: separate files)
    python experiments/run_experiment.py --input-dir benchmarks/dag_small_fr --algorithms cjt &
    python experiments/run_experiment.py --input-dir benchmarks/dag_small_fr --algorithms ivbp &
"""

import argparse
import glob
import json
import os
import re
import sys

from run_algorithm import (ALGORITHMS as ALL_ALGORITHMS, STAGES,
                           add_algorithm_arguments, kwargs_from_args,
                           run_single, set_num_threads, variant_suffix)

APPROX_ALGORITHMS = ["ariel", "ivbp", "cve", "approxlp"]


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
Parallel usage -- each (benchmark, algorithm, variant) gets its own file:
  python experiments/run_experiment.py --input-dir benchmarks/dag_small_fr --algorithms cjt &
  python experiments/run_experiment.py --input-dir benchmarks/dag_small_fr --algorithms ivbp &
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
    what = parser.add_mutually_exclusive_group()
    what.add_argument(
        "--algorithms", type=str, nargs="+", default=None,
        choices=ALL_ALGORITHMS,
        help=f"algorithms to run (default: {' '.join(APPROX_ALGORITHMS)})")
    what.add_argument(
        "--stages", type=str, nargs="+", default=None, choices=STAGES,
        help="build stages to run instead of algorithms")
    parser.add_argument(
        "--exact-threshold", type=int, default=15,
        help="max num_vars for running the exact algorithm (default: 15)")
    add_algorithm_arguments(parser)
    parser.add_argument(
        "--evidence", type=str, default="{}",
        help="evidence as JSON string (default: {})")
    parser.add_argument(
        "--time-limit", type=float, default=None,
        help="wall-clock time limit in seconds per instance per algorithm "
             "(default: unlimited)")
    parser.add_argument(
        "--memory-limit", type=float, default=None,
        help="memory limit in GB per process, enforced via RLIMIT_AS "
             "(default: unlimited). Per process: the per-family solves are "
             "threaded (one interpreter), so --n-jobs K does not multiply the "
             "resident Python heap; ipopt subprocesses and the process-based "
             "enumerate LRS remain outside this cap. Best-effort off Linux.")
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

    base_kwargs = kwargs_from_args(args)
    todo = args.stages or args.algorithms or APPROX_ALGORITHMS
    for algo in todo:
        # Output: {output_dir}/{benchmark}/{algorithm}{variant}.jsonl
        bench_dir = os.path.join(args.output_dir, bench_name)
        os.makedirs(bench_dir, exist_ok=True)
        out_name = algo + variant_suffix(algo, base_kwargs)
        out_path = os.path.join(bench_dir, f"{out_name}.jsonl")

        # Load already-completed instances for resume
        completed = _load_completed(out_path)
        if completed and args.verbosity > 0:
            print(f"[{bench_name}/{out_name}] Resuming: {len(completed)} "
                  f"instances already in {out_path}")

        outf = open(out_path, "a")
        try:
            for idx, instance in enumerate(instances, 1):
                graph_type, num_vars = _parse_instance_info(instance)
                basename = os.path.basename(instance)

                # The 2^n exact formulation is skipped on large instances.
                if algo == "exact" and num_vars > args.exact_threshold:
                    continue

                # Skip already completed
                if instance in completed:
                    if args.verbosity > 0:
                        print(f"[{bench_name}/{out_name}] [{idx}/{total}] "
                              f"{basename} | skip")
                    continue

                kwargs = dict(base_kwargs)

                result = run_single(
                    instance, algo, evidence=evidence,
                    verbosity=args.verbosity,
                    time_limit=args.time_limit,
                    memory_limit_gb=args.memory_limit, **kwargs)

                # Enrich with instance metadata
                result["instance"] = instance
                result["benchmark"] = bench_name
                result["graph_type"] = graph_type
                result["num_vars"] = num_vars
                result["evidence"] = evidence
                result["memory_limit_gb"] = args.memory_limit

                # Write to JSONL
                outf.write(json.dumps(result) + "\n")
                outf.flush()

                if args.verbosity > 0:
                    status = result["status"]
                    bt = result["build_time"]
                    rt = result["run_time"]
                    tt = result["total_time"]
                    if bt > 0:
                        print(f"[{bench_name}/{out_name}] [{idx}/{total}] "
                              f"{basename} | build={bt:.2f}s run={rt:.2f}s "
                              f"total={tt:.2f}s {status}")
                    else:
                        print(f"[{bench_name}/{out_name}] [{idx}/{total}] "
                              f"{basename} | {tt:.2f}s {status}")

        finally:
            outf.close()

        if args.verbosity > 0:
            print(f"[{bench_name}/{out_name}] Done. Results in {out_path}\n")


if __name__ == "__main__":
    main()
