"""Analyze inference results against a user-selectable reference algorithm.

Loads all per-algorithm JSONL files from the results directory
(``results/{benchmark}/{algorithm}.jsonl``), groups them by instance, and
computes error metrics for every other algorithm relative to the chosen
reference.

The reference is selected with ``--reference``:
  * ``ariel`` (default): use the ARIEL algorithm's record as the reference.
    ARIEL is exact on singly-connected LCNs and available at any scale, so it
    is the practical baseline when exact inference is out of reach.
  * ``exact``: use the ground-truth exact bounds per instance, preferring
    ``exact_g`` (SCIP, certified global) and falling back to ``exact_l``
    (ipopt, local) when only that is present. Only instances that were solved
    exactly contribute.
  * any other algorithm name (e.g. ``cve``, ``cjt``, ``ibp``): use that
    algorithm's record as the reference.

Reported metrics
----------------
Errors are measured on ``P(var=1)`` (the state-1 marginal), per variable, then
aggregated (mean/max) over all variables of all instances in each group. Let
``[r_lo, r_hi]`` be the reference bound and ``[a_lo, a_hi]`` the algorithm's
bound for a variable. The script reports, for each (group, algorithm):

  * ``mae_lb`` -- mean absolute lower-bound error, ``mean |r_lo - a_lo|``.
  * ``mae_ub`` -- mean absolute upper-bound error, ``mean |a_hi - r_hi|``.
        (These two MAEs are the primary accuracy metrics.)
  * ``std_lb`` / ``std_ub`` -- standard deviation of the same per-variable
        absolute lower-/upper-bound errors within the group (the spread that
        pairs with ``mae_lb`` / ``mae_ub``).
  * ``max_lb`` / ``max_ub`` -- the corresponding worst-case absolute errors,
        ``max |r_lo - a_lo|`` and ``max |a_hi - r_hi|``.
  * ``contain`` -- containment rate: the fraction of variables whose algorithm
        interval *contains* the reference interval
        (``a_lo <= r_lo`` and ``a_hi >= r_hi``, within a 1e-9 tolerance). 1.0
        means the algorithm is a valid outer bound on every variable; values
        below 1.0 flag intervals that are too tight (inner / unsound).
  * ``mean_wr`` -- mean interval-width ratio ``(a_hi - a_lo) / (r_hi - r_lo)``
        over variables with a non-degenerate reference width (``std_wr`` in the
        size grouping). >1 wider than the reference, <1 tighter.
  * Timings (means): ``build_t``, ``run_t``, ``total_t`` for the algorithm
        (``std_tt`` = std-dev of total time in the size grouping), and
        ``ref_t`` = mean total time of the reference on the same group.
  * ``avg_iw`` -- mean reported induced width (``n/a`` if unavailable).

The reference algorithm itself, and the exact backends when they are not the
reference, are excluded from the scored rows.

The ``--latex`` file contains two tables: the first reports the mean metrics
(``mae_lb`` / ``mae_ub`` and timings), the second reports the corresponding
standard deviations (``std_lb`` / ``std_ub``).

Usage:
    python experiments/analyze_results.py --results-dir results
    python experiments/analyze_results.py --results-dir results --reference exact \\
        --output analysis.csv --latex analysis.tex --group-by size
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict


# The algorithm names produced by run_algorithm.py (see its ALGORITHMS list),
# plus the special "exact" reference that resolves to exact_g/exact_l per
# instance.
_ALGORITHMS = ["exact_l", "exact_g", "ariel", "ibp", "ccte", "ccte_e",
               "ccte_cm", "approxlp", "cve", "cve_e", "cve_d4", "cjt",
               "cjt_l", "cjt_g"]
_REFERENCE_CHOICES = ["exact"] + _ALGORITHMS

# The exact backends, in preference order when the reference is "exact": the
# certified global solver (exact_g) is trusted over the local one (exact_l).
_EXACT_ALGORITHMS = ("exact_g", "exact_l")


def _mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def _std(vals):
    """Compute population standard deviation."""
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))


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


def _make_stats_key(rec, instance, group_by):
    """Build the grouping key depending on mode."""
    if group_by == "instance":
        return (os.path.basename(instance), rec["algorithm"])
    else:
        return (rec["graph_type"], rec["num_vars"], rec["algorithm"])


def _reference_record(algos, reference):
    """Return the reference record for an instance's {algorithm: record} map.

    For ``reference == "exact"`` prefer exact_g (certified global) then exact_l
    (local). Otherwise return the named algorithm's record. None if absent.
    """
    if reference == "exact":
        for name in _EXACT_ALGORITHMS:
            if name in algos:
                return algos[name]
        return None
    return algos.get(reference)


def _is_reference_or_exact(algo_name, reference):
    """Whether algo_name should be excluded from the scored (approximate) set.

    Always exclude the reference itself. Also exclude the exact backends when
    they are not the reference -- they are ground truth, not approximations to
    score against the reference.
    """
    if reference == "exact":
        return algo_name in _EXACT_ALGORITHMS
    # A named reference: exclude it, and still exclude the exact backends (they
    # are the ground truth and are analyzed separately with --reference exact).
    return algo_name == reference or algo_name in _EXACT_ALGORITHMS


def analyze(records, reference="ariel", output_file=None, latex_file=None,
            group_by="size"):
    """Compute absolute-error metrics vs the chosen reference algorithm.

    Args:
        records: list of result dicts loaded from JSONL.
        reference: reference algorithm name, or "exact" for the exact bounds.
        output_file: optional CSV output path.
        latex_file: optional LaTeX table output path.
        group_by: "size" to aggregate by (graph_type, num_vars, algorithm),
                  "instance" for per-instance rows.
    """
    groups = _group_by_instance(records)

    # Collect stats using the chosen grouping key
    stats = defaultdict(lambda: {
        "abs_lb_errors": [], "abs_ub_errors": [], "width_ratios": [],
        "contained": [],
        "build_times": [], "run_times": [], "total_times": [],
        "induced_widths": []
    })

    ref_stats = defaultdict(list)

    for instance, algos in groups.items():
        ref_rec = _reference_record(algos, reference)
        if ref_rec is None:
            continue

        ref_marg = ref_rec["marginals"]
        if group_by == "instance":
            ref_key = os.path.basename(instance)
        else:
            ref_key = (ref_rec["graph_type"], ref_rec["num_vars"])
        ref_stats[ref_key].append(
            ref_rec.get("total_time", ref_rec.get("time_seconds", 0.0)))

        for algo_name, rec in algos.items():
            if _is_reference_or_exact(algo_name, reference):
                continue

            approx_marg = rec["marginals"]
            key = _make_stats_key(rec, instance, group_by)
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
        print(f"No results found for reference '{reference}'.")
        return

    # Print summary table
    print(f"Reference: {reference}")
    print()
    if group_by == "instance":
        header = (f"{'instance':<30} {'algo':<10} "
                  f"{'mae_lb':>9} {'std_lb':>9} {'max_lb':>9} "
                  f"{'mae_ub':>9} {'std_ub':>9} {'max_ub':>9} "
                  f"{'contain':>8} {'mean_wr':>9} "
                  f"{'build_t':>8} {'run_t':>8} {'total_t':>8} "
                  f"{'ref_t':>8} {'avg_iw':>7}")
    else:
        header = (f"{'type':<12} {'n':>4} {'algo':<10} "
                  f"{'mae_lb':>9} {'std_lb':>9} {'max_lb':>9} "
                  f"{'mae_ub':>9} {'std_ub':>9} {'max_ub':>9} "
                  f"{'contain':>8} {'mean_wr':>9} "
                  f"{'build_t':>8} {'run_t':>8} {'total_t':>8} "
                  f"{'std_tt':>8} {'ref_t':>8} {'avg_iw':>7}")
    print(header)
    print("-" * len(header))

    rows = []
    for key, s in sorted(stats.items()):
        if not s["abs_lb_errors"]:
            continue

        mae_lb = _mean(s["abs_lb_errors"])
        std_lb = _std(s["abs_lb_errors"])
        max_lb = max(s["abs_lb_errors"])
        mae_ub = _mean(s["abs_ub_errors"])
        std_ub = _std(s["abs_ub_errors"])
        max_ub = max(s["abs_ub_errors"])
        contain = _mean(s["contained"])
        mean_wr = _mean(s["width_ratios"])
        std_wr = _std(s["width_ratios"])
        mean_bt = _mean(s["build_times"])
        mean_rt = _mean(s["run_times"])
        mean_tt = _mean(s["total_times"])
        std_tt = _std(s["total_times"])
        avg_iw = _mean(s["induced_widths"])
        iw_str = f"{avg_iw:>7.1f}" if s["induced_widths"] else f"{'n/a':>7}"

        if group_by == "instance":
            inst_name, algo = key
            rt_list = ref_stats.get(inst_name, [])
            ref_t = _mean(rt_list)

            print(f"{inst_name:<30} {algo:<10} "
                  f"{mae_lb:>9.6f} {std_lb:>9.6f} {max_lb:>9.6f} "
                  f"{mae_ub:>9.6f} {std_ub:>9.6f} {max_ub:>9.6f} "
                  f"{contain:>8.4f} {mean_wr:>9.4f} "
                  f"{mean_bt:>8.3f} {mean_rt:>8.3f} {mean_tt:>8.3f} "
                  f"{ref_t:>8.3f} {iw_str}")

            rows.append({
                "instance": inst_name,
                "algorithm": algo,
                "reference": reference,
                "mae_lb_error": round(mae_lb, 8),
                "std_lb_error": round(std_lb, 8),
                "max_lb_error": round(max_lb, 8),
                "mae_ub_error": round(mae_ub, 8),
                "std_ub_error": round(std_ub, 8),
                "max_ub_error": round(max_ub, 8),
                "containment_rate": round(contain, 6),
                "mean_width_ratio": round(mean_wr, 6),
                "build_time": round(mean_bt, 4),
                "run_time": round(mean_rt, 4),
                "total_time": round(mean_tt, 4),
                "ref_time": round(ref_t, 4),
                "avg_induced_width": round(avg_iw, 2) if s["induced_widths"] else None,
            })
        else:
            graph_type, num_vars, algo = key
            rt_list = ref_stats.get((graph_type, num_vars), [])
            ref_t = _mean(rt_list)

            print(f"{graph_type:<12} {num_vars:>4} {algo:<10} "
                  f"{mae_lb:>9.6f} {std_lb:>9.6f} {max_lb:>9.6f} "
                  f"{mae_ub:>9.6f} {std_ub:>9.6f} {max_ub:>9.6f} "
                  f"{contain:>8.4f} {mean_wr:>9.4f} "
                  f"{mean_bt:>8.3f} {mean_rt:>8.3f} {mean_tt:>8.3f} "
                  f"{std_tt:>8.3f} {ref_t:>8.3f} {iw_str}")

            rows.append({
                "graph_type": graph_type,
                "num_vars": num_vars,
                "algorithm": algo,
                "reference": reference,
                "mae_lb_error": round(mae_lb, 8),
                "std_lb_error": round(std_lb, 8),
                "max_lb_error": round(max_lb, 8),
                "mae_ub_error": round(mae_ub, 8),
                "std_ub_error": round(std_ub, 8),
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
        _save_latex(rows, latex_file, reference, group_by,
                    caption=f"error metrics vs.\\ {reference} reference")
        print(f"Saved LaTeX to {latex_file}")


def _latex_safe(s):
    """Escape a string for LaTeX. Wrap in math text if it contains underscores."""
    s = str(s)
    if "_" in s:
        return "$\\text{" + s.replace("_", "\\_") + "}$"
    return s


def _write_latex_table(f, rows, cols, caption):
    """Write a single LaTeX table block (given its column spec) to file f."""
    keys = [c[0] for c in cols]
    headers = [c[1] for c in cols]
    aligns = "".join(c[2] for c in cols)

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
                if k.startswith(("mae", "max", "std_lb", "std_ub")):
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
                vals.append(_latex_safe(v))
        f.write(" & ".join(vals) + " \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")


def _save_latex(rows, path, reference, group_by="size", caption="Results"):
    """Write two LaTeX tables to ``path``: one with mean error metrics and one
    with the corresponding standard deviations."""
    if group_by == "instance":
        id_cols = [
            ("instance", "Instance", "l"),
            ("algorithm", "Algorithm", "l"),
        ]
        time_cols = [
            ("build_time", "Build", "r"),
            ("run_time", "Run", "r"),
            ("total_time", "Total", "r"),
            ("ref_time", "Ref", "r"),
            ("avg_induced_width", "IW", "r"),
        ]
    else:
        id_cols = [
            ("graph_type", "Type", "l"),
            ("num_vars", "$n$", "r"),
            ("algorithm", "Algorithm", "l"),
        ]
        time_cols = [
            ("mean_build_time", "Build", "r"),
            ("mean_run_time", "Run", "r"),
            ("mean_total_time", "Total", "r"),
            ("ref_time", "Ref", "r"),
            ("avg_induced_width", "IW", "r"),
        ]

    mean_cols = id_cols + [
        ("mae_lb_error", "MAE$_{\\text{lb}}$", "r"),
        ("mae_ub_error", "MAE$_{\\text{ub}}$", "r"),
    ] + time_cols
    std_cols = id_cols + [
        ("std_lb_error", "SD$_{\\text{lb}}$", "r"),
        ("std_ub_error", "SD$_{\\text{ub}}$", "r"),
    ] + time_cols

    with open(path, "w") as f:
        _write_latex_table(f, rows, mean_cols,
                           caption=f"Mean {caption}")
        f.write("\n")
        _write_latex_table(f, rows, std_cols,
                           caption=f"Standard deviation of {caption}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze inference results against a selectable "
                    "reference algorithm.")
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Directory with results_*.jsonl files (default: results)")
    parser.add_argument(
        "--reference", type=str, default="ariel", choices=_REFERENCE_CHOICES,
        metavar="ALGO",
        help="Reference algorithm (default: ariel). Use 'exact' for the "
             "ground-truth exact bounds (prefers exact_g then exact_l per "
             "instance), or one of " + ", ".join(_ALGORITHMS))
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV file (optional)")
    parser.add_argument(
        "--latex", type=str, default=None,
        help="Output LaTeX table file (optional)")
    parser.add_argument(
        "--group-by", type=str, default="size",
        choices=["size", "instance"],
        help="Group results by problem size or per instance (default: size)")
    args = parser.parse_args()

    records = _load_results(args.results_dir)
    if not records:
        print(f"No results found in {args.results_dir}/")
        sys.exit(1)

    print(f"Loaded {len(records)} results from {args.results_dir}/\n")
    analyze(records, args.reference, args.output, args.latex,
            group_by=args.group_by)


if __name__ == "__main__":
    main()
