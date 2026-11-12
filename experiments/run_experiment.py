"""Experiment runner: run the evaluated runs on benchmark instances.

Each (benchmark, evidence level, run) triple writes to its own JSONL file, so
benchmarks, evidence levels and runs can execute in parallel without
conflicts.

Output structure:
    {output-dir}/{benchmark}/ev{K}/{run}.jsonl

where ``run`` is a canonical run name of experiments/runs.py (exact_scip,
exact_ipopt, ariel, cjt_scip, cjt_ipopt, cve, cve_eps_0.1, ivbp, cdve, ...) and
K the number of observed atoms per instance (``ev0``: no evidence). The
evidence of an instance is drawn deterministically from (instance file name,
--evidence-seed, K), so every run of an instance sees the same evidence.

Usage:
    # The whole evaluation grid on one benchmark set, without evidence
    python experiments/run_experiment.py --input-dir benchmarks/tree_small_fr \
        --runs all

    # Selected runs with 3 observed atoms per instance
    python experiments/run_experiment.py --input-dir benchmarks/tree_small_fr \
        --runs cjt_scip cve cve_eps_0.1 ivbp cdve --evidence-atoms 3

    # Build stages (write the .cn / .vtx caches the credal-network runs load)
    python experiments/run_experiment.py --input-dir benchmarks/tree_small_fr \
        --stages compile enumerate

    # Parallel across benchmarks and runs (safe: separate files)
    python experiments/run_experiment.py --input-dir benchmarks/dag_small_fr --runs cjt_scip &
    python experiments/run_experiment.py --input-dir benchmarks/dag_small_fr --runs ivbp &
"""

import argparse
import contextlib
import glob
import hashlib
import io
import json
import os
import random
import re
import sys

from run_algorithm import (STAGES, add_algorithm_arguments, kwargs_from_args,
                           run_single, set_num_threads)
from runs import RUNS, parse_run, run_name

from lcn.core.model import LCN


def sample_evidence(lcn_file, k, seed=0):
    """Deterministic evidence for one instance: ``k`` atoms drawn uniformly
    without replacement and a uniform 0/1 value for each, from a generator
    seeded by (instance file name, seed, k). Every run of the instance gets
    the same evidence. All engines condition only on the distributions that
    give it positive probability (regular conditioning)."""
    if k <= 0:
        return {}
    lcn = LCN()
    with contextlib.redirect_stdout(io.StringIO()):
        lcn.from_lcn(file_name=lcn_file)
    atoms = sorted(lcn.atoms)
    key = f"{os.path.basename(lcn_file)}|{seed}|{k}".encode()
    rng = random.Random(int(hashlib.sha256(key).hexdigest()[:16], 16))
    chosen = rng.sample(atoms, min(k, len(atoms)))
    return {a: rng.randint(0, 1) for a in sorted(chosen)}


def _parse_instance_info(filepath):
    """Extract graph_type and num_vars from filename like chain_n10_1.lcn."""
    basename = os.path.basename(filepath)
    match = re.match(r"(\w+)_n(\d+)_(\d+)\.lcn$", basename)
    if match:
        return match.group(1), int(match.group(2))
    return "unknown", 0


def _instance_info(filepath):
    """(graph_type, num_vars) of an instance. num_vars is the number of atoms
    of the parsed LCN, because the file name does not always carry it (the
    ``real`` and some ``junkyu`` files have no ``_n<N>_`` tag, which made the
    exact-engine size threshold see 0 variables)."""
    graph_type, _ = _parse_instance_info(filepath)
    lcn = LCN()
    with contextlib.redirect_stdout(io.StringIO()):
        lcn.from_lcn(file_name=filepath)
    return graph_type, len(lcn.atoms)


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
Parallel usage -- each (benchmark, evidence level, run) gets its own file:
  python experiments/run_experiment.py --input-dir benchmarks/dag_small_fr --runs cjt_scip &
  python experiments/run_experiment.py --input-dir benchmarks/dag_small_fr --runs ivbp &
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
        "--runs", type=str, nargs="+", default=None, metavar="RUN",
        help="runs to execute: canonical names (" + ", ".join(RUNS) +
             "; cve_eps_<E> for any E, tuning suffixes _it<N>/_r<R>) or "
             "'all' for the evaluation grid (default: all)")
    what.add_argument(
        "--stages", type=str, nargs="+", default=None, choices=STAGES,
        help="build stages to run instead of runs")
    parser.add_argument(
        "--exact-threshold", type=int, default=15,
        help="max num_vars for the exact_* runs (default: 15)")
    add_algorithm_arguments(parser)
    parser.add_argument(
        "--evidence-atoms", type=int, default=0, metavar="K",
        help="observed atoms per instance, drawn deterministically "
             "(default: 0 = no evidence); results go to ev{K}/")
    parser.add_argument(
        "--evidence-seed", type=int, default=0,
        help="seed of the per-instance evidence draw (default: 0)")
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
    parser.add_argument(
        "--shard", type=str, default=None, metavar="I/N",
        help="run only shard I of N (0-based): every N-th instance of the "
             "size-sorted list, starting at I. Output goes to "
             "{run}.shard{I}of{N}.jsonl so shards never share a file "
             "(default: no sharding)")
    args = parser.parse_args()

    set_num_threads(args.num_threads)

    shard = None
    if args.shard is not None:
        m = re.fullmatch(r"(\d+)/(\d+)", args.shard)
        if not m or int(m.group(2)) < 1 or int(m.group(1)) >= int(m.group(2)):
            parser.error(f"--shard expects I/N with 0 <= I < N, "
                         f"got {args.shard!r}")
        shard = (int(m.group(1)), int(m.group(2)))

    # Derive benchmark name
    bench_name = args.benchmark or _benchmark_name(args.input_dir)

    # Find all .lcn files, sorted by (num_vars, filename) so that
    # instances are processed in increasing problem size
    pattern = os.path.join(args.input_dir, "**", "*.lcn")
    instances = glob.glob(pattern, recursive=True)
    if not instances:
        print(f"No .lcn files found under {args.input_dir}")
        sys.exit(1)
    info = {p: _instance_info(p) for p in instances}
    instances.sort(key=lambda p: (info[p][1], os.path.basename(p)))
    if shard is not None:
        i, n = shard
        instances = instances[i::n]

    total = len(instances)
    shard_tag = f".shard{shard[0]}of{shard[1]}" if shard else ""

    base_kwargs = kwargs_from_args(args)
    if args.stages:
        todo = [(st, st, {}) for st in args.stages]
    else:
        names = RUNS if (not args.runs or args.runs == ["all"]) else args.runs
        todo = []
        for name in names:
            try:
                algo, params = parse_run(name)
            except ValueError as e:
                parser.error(str(e))
            todo.append((name, algo, params))
    k_ev = max(0, args.evidence_atoms)
    ev_dir = f"ev{k_ev}"
    for out_name, algo, params in todo:
        kw = dict(base_kwargs)
        kw.update({k: v for k, v in params.items() if v is not None})
        if algo not in STAGES:
            assert run_name(algo, kw) == out_name, (out_name, kw)
        # Output: {output_dir}/{benchmark}/ev{K}/{run}.jsonl (stages: no
        # evidence level -- they build caches)
        bench_dir = os.path.join(args.output_dir, bench_name) if algo in STAGES \
            else os.path.join(args.output_dir, bench_name, ev_dir)
        os.makedirs(bench_dir, exist_ok=True)
        out_path = os.path.join(bench_dir, f"{out_name}{shard_tag}.jsonl")

        # Load already-completed instances for resume
        completed = _load_completed(out_path)
        if completed and args.verbosity > 0:
            print(f"[{bench_name}/{out_name}] Resuming: {len(completed)} "
                  f"instances already in {out_path}")

        outf = open(out_path, "a")
        try:
            for idx, instance in enumerate(instances, 1):
                graph_type, num_vars = info[instance]
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

                kwargs = dict(kw)
                evidence = {} if algo in STAGES else \
                    sample_evidence(instance, k_ev, args.evidence_seed)

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
                result["evidence_atoms"] = k_ev
                result["evidence_seed"] = args.evidence_seed
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
