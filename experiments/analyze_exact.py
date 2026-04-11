"""Analyze inference results against exact bounds.

For instances where exact inference results are available (small instances),
compute error metrics for each approximate algorithm.

Loads all per-algorithm JSONL files from the results directory.

Metrics per variable:
  - Lower bound error: exact_lower - approx_lower (positive = approx looser)
  - Upper bound error: approx_upper - exact_upper (positive = approx looser)
  - Interval width ratio: approx_width / exact_width

Usage:
    python experiments/analyze_exact.py --results-dir results
    python experiments/analyze_exact.py --results-dir results --output analysis_exact.csv
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict


def _std(vals):
    """Compute population standard deviation."""
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))


def _load_results(results_dir):
    """Load all results from per-algorithm JSONL files in a directory."""
    records = []
    pattern = os.path.join(results_dir, "results_*.jsonl")
    for path in sorted(glob.glob(pattern)):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _group_by_instance(records):
    """Group records by instance path -> {algorithm: record}."""
    groups = defaultdict(dict)
    for rec in records:
        if rec["status"] != "ok":
            continue
        groups[rec["instance"]][rec["algorithm"]] = rec
    return groups


def analyze(records, output_file=None):
    """Compute error metrics for approximate algorithms vs exact."""
    groups = _group_by_instance(records)

    # Collect per (graph_type, num_vars, algorithm) stats
    stats = defaultdict(lambda: {
        "lb_errors": [], "ub_errors": [], "width_ratios": [],
        "build_times": [], "run_times": [], "total_times": []
    })

    for instance, algos in groups.items():
        if "exact" not in algos:
            continue

        exact = algos["exact"]
        exact_marg = exact["marginals"]

        for algo_name, rec in algos.items():
            if algo_name == "exact":
                continue

            approx_marg = rec["marginals"]
            key = (rec["graph_type"], rec["num_vars"], algo_name)
            s = stats[key]
            s["build_times"].append(rec.get("build_time", 0.0))
            s["run_times"].append(rec.get("run_time", 0.0))
            s["total_times"].append(rec.get("total_time",
                                            rec.get("time_seconds", 0.0)))

            for var in exact_marg:
                if var not in approx_marg:
                    continue

                e_lo = exact_marg[var]["lower"]
                e_hi = exact_marg[var]["upper"]
                a_lo = approx_marg[var]["lower"]
                a_hi = approx_marg[var]["upper"]

                # Use P(var=1) bounds (index 1)
                lb_err = e_lo[1] - a_lo[1]  # positive = approx is looser
                ub_err = a_hi[1] - e_hi[1]  # positive = approx is looser
                exact_width = e_hi[1] - e_lo[1]
                approx_width = a_hi[1] - a_lo[1]

                s["lb_errors"].append(lb_err)
                s["ub_errors"].append(ub_err)
                if exact_width > 1e-12:
                    s["width_ratios"].append(approx_width / exact_width)

    # Also collect exact timing
    exact_stats = defaultdict(list)
    for instance, algos in groups.items():
        if "exact" in algos:
            rec = algos["exact"]
            exact_stats[(rec["graph_type"], rec["num_vars"])].append(
                rec.get("total_time", rec.get("time_seconds", 0.0)))

    if not stats:
        print("No instances found with both exact and approximate results.")
        return

    # Print summary table
    header = (f"{'type':<12} {'n':>4} {'algo':<10} "
              f"{'mean_lb':>9} {'std_lb':>9} {'max_lb':>9} "
              f"{'mean_ub':>9} {'std_ub':>9} {'max_ub':>9} "
              f"{'mean_wr':>9} {'std_wr':>9} "
              f"{'build_t':>8} {'run_t':>8} {'total_t':>8} "
              f"{'std_tt':>8} {'exact_t':>8}")
    print(header)
    print("-" * len(header))

    rows = []
    for (graph_type, num_vars, algo), s in sorted(stats.items()):
        if not s["lb_errors"]:
            continue

        mean_lb = sum(s["lb_errors"]) / len(s["lb_errors"])
        std_lb = _std(s["lb_errors"])
        max_lb = max(abs(x) for x in s["lb_errors"])
        mean_ub = sum(s["ub_errors"]) / len(s["ub_errors"])
        std_ub = _std(s["ub_errors"])
        max_ub = max(abs(x) for x in s["ub_errors"])
        mean_wr = (sum(s["width_ratios"]) / len(s["width_ratios"])
                   if s["width_ratios"] else float("nan"))
        std_wr = (_std(s["width_ratios"])
                  if len(s["width_ratios"]) >= 2 else 0.0)
        mean_bt = sum(s["build_times"]) / len(s["build_times"])
        mean_rt = sum(s["run_times"]) / len(s["run_times"])
        mean_tt = sum(s["total_times"]) / len(s["total_times"])
        std_tt = _std(s["total_times"])

        et_list = exact_stats.get((graph_type, num_vars), [])
        exact_t = sum(et_list) / len(et_list) if et_list else float("nan")

        print(f"{graph_type:<12} {num_vars:>4} {algo:<10} "
              f"{mean_lb:>9.6f} {std_lb:>9.6f} {max_lb:>9.6f} "
              f"{mean_ub:>9.6f} {std_ub:>9.6f} {max_ub:>9.6f} "
              f"{mean_wr:>9.4f} {std_wr:>9.4f} "
              f"{mean_bt:>8.3f} {mean_rt:>8.3f} {mean_tt:>8.3f} "
              f"{std_tt:>8.3f} {exact_t:>8.3f}")

        rows.append({
            "graph_type": graph_type,
            "num_vars": num_vars,
            "algorithm": algo,
            "mean_lb_error": round(mean_lb, 8),
            "std_lb_error": round(std_lb, 8),
            "max_lb_error": round(max_lb, 8),
            "mean_ub_error": round(mean_ub, 8),
            "std_ub_error": round(std_ub, 8),
            "max_ub_error": round(max_ub, 8),
            "mean_width_ratio": round(mean_wr, 6),
            "std_width_ratio": round(std_wr, 6),
            "mean_build_time": round(mean_bt, 4),
            "mean_run_time": round(mean_rt, 4),
            "mean_total_time": round(mean_tt, 4),
            "std_total_time": round(std_tt, 4),
            "exact_time": round(exact_t, 4),
        })

    # Save CSV
    if output_file and rows:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze inference results against exact bounds.")
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Directory with results_*.jsonl files (default: results)")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV file (optional)")
    args = parser.parse_args()

    records = _load_results(args.results_dir)
    if not records:
        print(f"No results found in {args.results_dir}/")
        sys.exit(1)

    print(f"Loaded {len(records)} results from {args.results_dir}/\n")
    analyze(records, args.output)


if __name__ == "__main__":
    main()
