"""Analyze inference results against a reference algorithm.

For large instances where exact inference is not available, use a reference
algorithm (default: ARIEL) to compute absolute error metrics for other algorithms.

Loads all per-algorithm JSONL files from the results directory.

Metrics per variable (absolute errors on P(var=1) bounds):
  - |ref_lower - approx_lower|
  - |approx_upper - ref_upper|
  - Interval width ratio: approx_width / ref_width

Usage:
    python experiments/analyze_reference.py --results-dir results
    python experiments/analyze_reference.py --results-dir results --reference ariel --output analysis_ref.csv
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict


def _mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def _std(vals):
    """Compute population standard deviation."""
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))


def _rmse(vals):
    """Compute root mean squared error from a list of absolute errors."""
    if not vals:
        return float("nan")
    return math.sqrt(sum(x * x for x in vals) / len(vals))


def _load_results(results_dir):
    """Load all results from JSONL files under the results directory.
    Globs recursively: results/{benchmark}/{algorithm}.jsonl
    """
    records = []
    pattern = os.path.join(results_dir, "**", "*.jsonl")
    for path in sorted(glob.glob(pattern, recursive=True)):
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


def analyze(records, reference="ariel", output_file=None, latex_file=None):
    """Compute absolute error metrics vs a reference algorithm."""
    groups = _group_by_instance(records)

    # Collect per (graph_type, num_vars, algorithm) stats
    stats = defaultdict(lambda: {
        "abs_lb_errors": [], "abs_ub_errors": [], "width_ratios": [],
        "contained": [],
        "build_times": [], "run_times": [], "total_times": [],
        "induced_widths": []
    })

    ref_stats = defaultdict(list)

    for instance, algos in groups.items():
        if reference not in algos:
            continue

        ref_rec = algos[reference]
        num_vars = ref_rec["num_vars"]

        ref_marg = ref_rec["marginals"]
        ref_stats[(ref_rec["graph_type"], num_vars)].append(
            ref_rec.get("total_time", ref_rec.get("time_seconds", 0.0)))

        for algo_name, rec in algos.items():
            if algo_name == reference or algo_name == "exact":
                continue

            approx_marg = rec["marginals"]
            key = (rec["graph_type"], num_vars, algo_name)
            s = stats[key]
            s["build_times"].append(rec.get("build_time", 0.0))
            s["run_times"].append(rec.get("run_time", 0.0))
            s["total_times"].append(rec.get("total_time",
                                            rec.get("time_seconds", 0.0)))
            iw = rec.get("induced_width")
            if iw is not None:
                s["induced_widths"].append(iw)

            for var in ref_marg:
                if var not in approx_marg:
                    continue

                r_lo = ref_marg[var]["lower"]
                r_hi = ref_marg[var]["upper"]
                a_lo = approx_marg[var]["lower"]
                a_hi = approx_marg[var]["upper"]

                # Absolute errors on P(var=1) bounds (index 1)
                s["abs_lb_errors"].append(abs(r_lo[1] - a_lo[1]))
                s["abs_ub_errors"].append(abs(a_hi[1] - r_hi[1]))

                # Containment: approx interval contains reference interval
                tol = 1e-9
                s["contained"].append(
                    1 if (a_lo[1] <= r_lo[1] + tol and
                          a_hi[1] >= r_hi[1] - tol) else 0)

                ref_width = r_hi[1] - r_lo[1]
                approx_width = a_hi[1] - a_lo[1]
                if ref_width > 1e-12:
                    s["width_ratios"].append(approx_width / ref_width)

    if not stats:
        print(f"No results found for reference algorithm '{reference}'.")
        return

    # Print summary table
    print(f"Reference: {reference}")
    print()
    header = (f"{'type':<12} {'n':>4} {'algo':<10} "
              f"{'mae_lb':>9} {'rmse_lb':>9} {'max_lb':>9} "
              f"{'mae_ub':>9} {'rmse_ub':>9} {'max_ub':>9} "
              f"{'contain':>8} {'mean_wr':>9} "
              f"{'build_t':>8} {'run_t':>8} {'total_t':>8} "
              f"{'std_tt':>8} {'ref_t':>8} {'avg_iw':>7}")
    print(header)
    print("-" * len(header))

    rows = []
    for (graph_type, num_vars, algo), s in sorted(stats.items()):
        if not s["abs_lb_errors"]:
            continue

        mae_lb = _mean(s["abs_lb_errors"])
        rmse_lb = _rmse(s["abs_lb_errors"])
        max_lb = max(s["abs_lb_errors"])
        mae_ub = _mean(s["abs_ub_errors"])
        rmse_ub = _rmse(s["abs_ub_errors"])
        max_ub = max(s["abs_ub_errors"])
        contain = _mean(s["contained"])
        mean_wr = _mean(s["width_ratios"])
        std_wr = _std(s["width_ratios"])
        mean_bt = _mean(s["build_times"])
        mean_rt = _mean(s["run_times"])
        mean_tt = _mean(s["total_times"])
        std_tt = _std(s["total_times"])

        rt_list = ref_stats.get((graph_type, num_vars), [])
        ref_t = _mean(rt_list)
        avg_iw = _mean(s["induced_widths"])

        iw_str = f"{avg_iw:>7.1f}" if s["induced_widths"] else f"{'n/a':>7}"
        print(f"{graph_type:<12} {num_vars:>4} {algo:<10} "
              f"{mae_lb:>9.6f} {rmse_lb:>9.6f} {max_lb:>9.6f} "
              f"{mae_ub:>9.6f} {rmse_ub:>9.6f} {max_ub:>9.6f} "
              f"{contain:>8.4f} {mean_wr:>9.4f} "
              f"{mean_bt:>8.3f} {mean_rt:>8.3f} {mean_tt:>8.3f} "
              f"{std_tt:>8.3f} {ref_t:>8.3f} {iw_str}")

        rows.append({
            "graph_type": graph_type,
            "num_vars": num_vars,
            "algorithm": algo,
            "reference": reference,
            "mae_lb_error": round(mae_lb, 8),
            "rmse_lb_error": round(rmse_lb, 8),
            "max_lb_error": round(max_lb, 8),
            "mae_ub_error": round(mae_ub, 8),
            "rmse_ub_error": round(rmse_ub, 8),
            "max_ub_error": round(max_ub, 8),
            "containment_rate": round(contain, 6),
            "mean_width_ratio": round(mean_wr, 6),
            "std_width_ratio": round(std_wr, 6),
            "mean_build_time": round(mean_bt, 4),
            "mean_run_time": round(mean_rt, 4),
            "mean_total_time": round(mean_tt, 4),
            "std_total_time": round(std_tt, 4),
            "ref_time": round(ref_t, 4),
            "avg_induced_width": round(avg_iw, 2) if s["induced_widths"] else None,
        })

    # Save CSV
    if output_file and rows:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV to {output_file}")

    # Save LaTeX
    if latex_file and rows:
        _save_latex(rows, latex_file, reference,
                    caption=f"Error metrics vs.\\ {reference} reference")
        print(f"Saved LaTeX to {latex_file}")


def _save_latex(rows, path, reference, caption="Results"):
    """Write rows as a LaTeX table."""
    cols = [
        ("type", "Type", "l"),
        ("num_vars", "$n$", "r"),
        ("algorithm", "Algorithm", "l"),
        ("mae_lb_error", "MAE$_{\\text{lb}}$", "r"),
        ("rmse_lb_error", "RMSE$_{\\text{lb}}$", "r"),
        ("max_lb_error", "Max$_{\\text{lb}}$", "r"),
        ("mae_ub_error", "MAE$_{\\text{ub}}$", "r"),
        ("rmse_ub_error", "RMSE$_{\\text{ub}}$", "r"),
        ("max_ub_error", "Max$_{\\text{ub}}$", "r"),
        ("containment_rate", "Contain", "r"),
        ("mean_width_ratio", "W-ratio", "r"),
        ("mean_build_time", "Build", "r"),
        ("mean_run_time", "Run", "r"),
        ("mean_total_time", "Total", "r"),
        ("ref_time", "Ref", "r"),
        ("avg_induced_width", "IW", "r"),
    ]
    keys = [c[0] for c in cols]
    headers = [c[1] for c in cols]
    aligns = "".join(c[2] for c in cols)

    with open(path, "w") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\scriptsize\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\begin{{tabular}}{{{aligns}}}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            vals = []
            for k in keys:
                v = row.get(k)
                if v is None:
                    vals.append("--")
                elif isinstance(v, float):
                    if k.startswith("mae") or k.startswith("rmse") or k.startswith("max"):
                        vals.append(f"{v:.4f}")
                    elif k == "containment_rate":
                        vals.append(f"{v:.3f}")
                    elif "time" in k:
                        vals.append(f"{v:.2f}")
                    elif k == "mean_width_ratio":
                        vals.append(f"{v:.3f}")
                    else:
                        vals.append(f"{v:.2f}")
                else:
                    vals.append(str(v))
            f.write(" & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze inference results against a reference algorithm.")
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Directory with results_*.jsonl files (default: results)")
    parser.add_argument(
        "--reference", type=str, default="ariel",
        help="Reference algorithm (default: ariel)")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV file (optional)")
    parser.add_argument(
        "--latex", type=str, default=None,
        help="Output LaTeX table file (optional)")
    args = parser.parse_args()

    records = _load_results(args.results_dir)
    if not records:
        print(f"No results found in {args.results_dir}/")
        sys.exit(1)

    print(f"Loaded {len(records)} results from {args.results_dir}/\n")
    analyze(records, args.reference, args.output, args.latex)


if __name__ == "__main__":
    main()
