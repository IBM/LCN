"""Visualize inference results from CSV files.

Produces two figures:
  1. MAE (lower bound & upper bound) vs. problem size, one line per algorithm.
  2. Total runtime vs. problem size, one line per algorithm (log-log scale).

Usage:
    python experiments/plot_results.py --exact results-polytree-exact.csv \
                                       --ref results-polytree-ref.csv \
                                       --output plots/polytree
"""

import argparse
import os

import matplotlib
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Consistent styling per algorithm
ALGO_STYLE = {
    "ariel":    {"marker": "o", "color": "#1f77b4", "label": "ARIEL"},
    "approxlp": {"marker": "s", "color": "#2ca02c", "label": "CDVE"},
    "ibp":      {"marker": "^", "color": "#d62728", "label": "IBP"},
    "ccte":     {"marker": "D", "color": "#9467bd", "label": "CCTE"},
    "ccte_e":   {"marker": "d", "color": "#8c564b", "label": r"CCTE-$\epsilon$"},
    "ccte_cp":  {"marker": "p", "color": "#e377c2", "label": "CCTE-cp"},
    "ccte_cm":  {"marker": "h", "color": "#7f7f7f", "label": "CCTE-cm"},
    "ijgp":     {"marker": "v", "color": "#ff7f0e", "label": "IJGP"},
    "ijgp_e":   {"marker": "<", "color": "#bcbd22", "label": r"IJGP-$\epsilon$"},
    "ijgp_cp":  {"marker": ">", "color": "#17becf", "label": "IJGP-cp"},
    "ijgp_cm":  {"marker": "P", "color": "#aec7e8", "label": "IJGP-cm"},
}


def _style(algo):
    """Return marker/color/label for an algorithm, with fallback."""
    return ALGO_STYLE.get(algo, {"marker": "x", "color": "gray",
                                  "label": algo})


def load_and_merge(exact_path, ref_path):
    """Load exact and reference CSVs. Deduplicate: prefer exact source
    when the same (algorithm, num_vars) appears in both."""
    frames = []
    if exact_path and os.path.exists(exact_path):
        df = pd.read_csv(exact_path)
        df["source"] = "exact"
        frames.append(df)
    if ref_path and os.path.exists(ref_path):
        df = pd.read_csv(ref_path)
        df["source"] = "reference"
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No CSV files found.")
    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate: keep exact row when both exist for same (algo, n)
    combined["_priority"] = combined["source"].map(
        {"exact": 0, "reference": 1})
    combined = combined.sort_values("_priority") \
        .drop_duplicates(subset=["algorithm", "num_vars"], keep="first") \
        .drop(columns=["_priority"])

    return combined


def _extract_baselines(exact_path, ref_path):
    """Extract exact and reference baseline times as separate series."""
    exact_times, ref_times = None, None
    if exact_path and os.path.exists(exact_path):
        df = pd.read_csv(exact_path)
        if "exact_time" in df.columns:
            et = df.drop_duplicates("num_vars")[["num_vars", "exact_time"]] \
                   .dropna().sort_values("num_vars")
            if not et.empty:
                exact_times = et
    if ref_path and os.path.exists(ref_path):
        df = pd.read_csv(ref_path)
        if "ref_time" in df.columns:
            rt = df.drop_duplicates("num_vars")[["num_vars", "ref_time"]] \
                   .dropna().sort_values("num_vars")
            if not rt.empty:
                ref_times = rt
    return exact_times, ref_times


def _set_log_xaxis(ax, sizes):
    """Configure a log-scale x-axis with ticks at the actual data sizes."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.minorticks_off()


def _plot_mae_panel(df, output_path, suptitle):
    """
    Two side-by-side panels: MAE lower bound (left) and upper bound (right).
    """
    algos = sorted(df["algorithm"].unique(),
                   key=lambda a: list(ALGO_STYLE.keys()).index(a)
                   if a in ALGO_STYLE else 999)
    sizes = sorted(df["num_vars"].unique())

    fig, (ax_lb, ax_ub) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for algo in algos:
        sub = df[df["algorithm"] == algo].sort_values("num_vars")
        if sub.empty:
            continue
        s = _style(algo)
        ax_lb.plot(sub["num_vars"], sub["mae_lb_error"],
                   marker=s["marker"], color=s["color"], label=s["label"],
                   linewidth=1.4, markersize=6)
        ax_ub.plot(sub["num_vars"], sub["mae_ub_error"],
                   marker=s["marker"], color=s["color"], label=s["label"],
                   linewidth=1.4, markersize=6)

    for ax, title in [(ax_lb, "Lower bound error"),
                      (ax_ub, "Upper bound error")]:
        _set_log_xaxis(ax, sizes)
        ax.set_xlabel("Number of variables ($n$)")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_ylim(bottom=-0.01)

    ax_lb.set_ylabel("MAE")

    handles, labels = ax_lb.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=min(len(algos), 6), frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle(suptitle, fontsize=12, y=1.08)
    fig.tight_layout(rect=[0, 0, 1, 0.91])

    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_mae(exact_path, ref_path, output_prefix):
    """
    Generate separate MAE plots for exact and reference comparisons.
    """
    if exact_path and os.path.exists(exact_path):
        df_exact = pd.read_csv(exact_path)
        _plot_mae_panel(df_exact, f"{output_prefix}_mae_exact.pdf",
                        "MAE vs. Exact")

    if ref_path and os.path.exists(ref_path):
        df_ref = pd.read_csv(ref_path)
        _plot_mae_panel(df_ref, f"{output_prefix}_mae_ref.pdf",
                        "MAE vs. Reference (ARIEL)")


def plot_runtime(df, output_prefix, exact_times, ref_times):
    """
    Figure 2: Two side-by-side panels (log-log scale).
      Left  -- Algorithm run time only (excludes build/factorization).
      Right -- Total time (build + run), showing end-to-end cost.
    Exact and reference baselines plotted as separate dashed lines.
    """
    algos = sorted(df["algorithm"].unique(),
                   key=lambda a: list(ALGO_STYLE.keys()).index(a)
                   if a in ALGO_STYLE else 999)
    sizes = sorted(df["num_vars"].unique())

    fig, (ax_run, ax_total) = plt.subplots(1, 2, figsize=(10, 4),
                                           sharey=True)

    for algo in algos:
        sub = df[df["algorithm"] == algo].sort_values("num_vars")
        if sub.empty:
            continue
        s = _style(algo)

        # Run time (algorithm only, no build)
        if "mean_run_time" in sub.columns:
            ax_run.plot(sub["num_vars"], sub["mean_run_time"],
                        marker=s["marker"], color=s["color"],
                        label=s["label"], linewidth=1.4, markersize=6)

        # Total time (build + run)
        time_col = "mean_total_time" if "mean_total_time" in sub.columns \
            else "total_time"
        if time_col in sub.columns:
            ax_total.plot(sub["num_vars"], sub[time_col],
                          marker=s["marker"], color=s["color"],
                          label=s["label"], linewidth=1.4, markersize=6)

    # Plot baselines on both panels
    for ax in (ax_run, ax_total):
        if exact_times is not None and not exact_times.empty:
            ax.plot(exact_times["num_vars"], exact_times["exact_time"],
                    linestyle="--", color="black", linewidth=1.5,
                    marker="*", markersize=7, label="Exact", alpha=0.7)
        if ref_times is not None and not ref_times.empty:
            ax.plot(ref_times["num_vars"], ref_times["ref_time"],
                    linestyle=":", color="dimgray", linewidth=1.5,
                    marker=".", markersize=7, label="Reference (ARIEL)",
                    alpha=0.7)

    for ax, title in [(ax_run, "Run time (algorithm only)"),
                      (ax_total, "Total time (build + run)")]:
        _set_log_xaxis(ax, sizes)
        ax.set_xlabel("Number of variables ($n$)")
        ax.set_title(title, fontsize=11)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25, which="both", linewidth=0.5)

    ax_run.set_ylabel("Time (seconds)")

    # Shared legend at the top
    handles, labels = ax_total.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=min(len(algos) + 2, 6), frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.91])

    path = f"{output_prefix}_runtime.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot inference results from CSV files.")
    parser.add_argument(
        "--exact", type=str, default=None,
        help="Path to exact-comparison CSV (e.g. results-polytree-exact.csv)")
    parser.add_argument(
        "--ref", type=str, default=None,
        help="Path to reference-comparison CSV (e.g. results-polytree-ref.csv)")
    parser.add_argument(
        "--output", type=str, default="plots/results",
        help="Output prefix for PDF files (default: plots/results)")
    args = parser.parse_args()

    df = load_and_merge(args.exact, args.ref)
    exact_times, ref_times = _extract_baselines(args.exact, args.ref)

    print(f"Loaded {len(df)} rows, "
          f"algorithms: {sorted(df['algorithm'].unique())}, "
          f"sizes: {sorted(df['num_vars'].unique())}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    plot_mae(args.exact, args.ref, args.output)
    plot_runtime(df, args.output, exact_times, ref_times)


if __name__ == "__main__":
    main()
