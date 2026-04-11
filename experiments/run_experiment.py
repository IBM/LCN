"""Experiment runner: run inference algorithms on benchmark instances.

Each algorithm writes to its own JSONL file (results_{algorithm}.jsonl)
so that multiple algorithm runs can execute in parallel without conflicts.

Usage:
    # Run all algorithms sequentially
    python experiments/run_experiment.py --input-dir benchmarks/

    # Run a single algorithm (safe to launch multiple in parallel)
    python experiments/run_experiment.py --input-dir benchmarks/ --algorithms ccte
    python experiments/run_experiment.py --input-dir benchmarks/ --algorithms ibp
    python experiments/run_experiment.py --input-dir benchmarks/ --algorithms ariel
    python experiments/run_experiment.py --input-dir benchmarks/ --algorithms approxlp

    # Exact inference for small instances only
    python experiments/run_experiment.py --input-dir benchmarks/ --algorithms exact --exact-threshold 15
"""

import argparse
import glob
import json
import os
import re
import sys

from run_algorithm import run_single

APPROX_ALGORITHMS = ["ariel", "ibp", "ccte", "approxlp"]
ALL_ALGORITHMS = ["exact"] + APPROX_ALGORITHMS


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


def _output_path(output_dir, algorithm):
    """Return the per-algorithm JSONL output path."""
    return os.path.join(output_dir, f"results_{algorithm}.jsonl")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference algorithms on benchmark LCN instances.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Parallel usage (launch in separate terminals/processes):
  python experiments/run_experiment.py --algorithms exact --exact-threshold 15
  python experiments/run_experiment.py --algorithms ariel
  python experiments/run_experiment.py --algorithms ccte
  python experiments/run_experiment.py --algorithms ibp
  python experiments/run_experiment.py --algorithms approxlp
""")
    parser.add_argument(
        "--input-dir", type=str, default="benchmarks",
        help="root directory with .lcn files (default: benchmarks)")
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="output directory for per-algorithm JSONL files (default: results)")
    parser.add_argument(
        "--algorithms", type=str, nargs="+", default=APPROX_ALGORITHMS,
        choices=ALL_ALGORITHMS,
        help="algorithms to run (default: ariel ibp ccte approxlp)")
    parser.add_argument(
        "--exact-threshold", type=int, default=15,
        help="max num_vars for running exact inference (default: 15)")
    parser.add_argument(
        "--evidence", type=str, default="{}",
        help="evidence as JSON string (default: {})")
    parser.add_argument(
        "--verbosity", type=int, default=1,
        help="0=silent, 1=progress (default: 1)")
    args = parser.parse_args()

    evidence = json.loads(args.evidence)
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all .lcn files
    pattern = os.path.join(args.input_dir, "**", "*.lcn")
    instances = sorted(glob.glob(pattern, recursive=True))
    if not instances:
        print(f"No .lcn files found under {args.input_dir}")
        sys.exit(1)

    total = len(instances)

    for algo in args.algorithms:
        out_path = _output_path(args.output_dir, algo)

        # Load already-completed instances for this algorithm (resume)
        completed = _load_completed(out_path)
        if completed and args.verbosity > 0:
            print(f"[{algo}] Resuming: {len(completed)} instances "
                  f"already in {out_path}")

        outf = open(out_path, "a")
        try:
            for idx, instance in enumerate(instances, 1):
                graph_type, num_vars = _parse_instance_info(instance)
                basename = os.path.basename(instance)

                # Skip exact for large instances
                if algo == "exact" and num_vars > args.exact_threshold:
                    continue

                # Skip already completed
                if instance in completed:
                    if args.verbosity > 0:
                        print(f"[{algo}] [{idx}/{total}] {basename} | skip")
                    continue

                result = run_single(
                    instance, algo, evidence=evidence, verbosity=0)

                # Enrich with instance metadata
                result["instance"] = instance
                result["graph_type"] = graph_type
                result["num_vars"] = num_vars
                result["evidence"] = evidence

                # Write to per-algorithm JSONL
                outf.write(json.dumps(result) + "\n")
                outf.flush()

                if args.verbosity > 0:
                    status = result["status"]
                    bt = result["build_time"]
                    rt = result["run_time"]
                    tt = result["total_time"]
                    if bt > 0:
                        print(f"[{algo}] [{idx}/{total}] {basename} | "
                              f"build={bt:.2f}s run={rt:.2f}s "
                              f"total={tt:.2f}s {status}")
                    else:
                        print(f"[{algo}] [{idx}/{total}] {basename} | "
                              f"{tt:.2f}s {status}")

        finally:
            outf.close()

        if args.verbosity > 0:
            print(f"[{algo}] Done. Results in {out_path}\n")


if __name__ == "__main__":
    main()
