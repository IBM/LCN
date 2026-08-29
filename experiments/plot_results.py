"""Visualize inference results from CSV files.

Produces two figures:
  1. MAE (lower bound & upper bound) vs. problem size, one line per algorithm.
  2. Mean total time (build + run) vs. problem size, one line per algorithm
     (log-log scale).

With ``--std``, a lighter shade band of +/- one standard deviation is drawn
around every curve, read from the ``std_*`` columns written by
``analyze_results.py``. The band is skipped for any series whose std column is
absent (older CSVs predate those columns).

By default only the five algorithms in ``DEFAULT_ALGORITHMS`` are plotted
(CVE, CVE-cm, ARIEL, IBP and approxlp, which is labelled CDVE); override with
``--algorithms``.

Usage:
    python experiments/plot_results.py --exact results-polytree-exact.csv \
                                       --ref results-polytree-ref.csv \
                                       --std --output plots/polytree
"""

import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"


# Consistent styling per algorithm
ALGO_STYLE = {
    "ariel":    {"marker": "o", "color": "#1f77b4", "label": "ARIEL"},
    "approxlp": {"marker": "s", "color": "#2ca02c", "label": "CDVE"},
    "ibp":      {"marker": "^", "color": "#d62728", "label": "IBP"},
    "cve":      {"marker": "D", "color": "#9467bd", "label": "CVE"},
    "cve_cp":   {"marker": "p", "color": "#e377c2", "label": "CVE-cp"},
    "cve_cm":   {"marker": "h", "color": "#7f7f7f", "label": "CVE-cm"},
    "cjt":      {"marker": "v", "color": "#ff7f0e", "label": "CJT"},
    "cjt_g":    {"marker": "v", "color": "#ff7f0e", "label": "CJT-G"},
    "cjt_l":    {"marker": "<", "color": "#bcbd22", "label": "CJT-L"},
}

# The algorithms plotted by default, in plot/legend order.
DEFAULT_ALGORITHMS = ["cve", "cve_cm", "ariel", "ibp", "approxlp"]

# Shading of the +/- 1 std band.
_BAND_ALPHA = 0.18

# On a log axis the band's lower edge must stay positive. Rather than a fixed
# absolute floor (which drags the axis down over many empty decades whenever the
# std exceeds the mean), clip to this fraction of the smallest positive value
# actually plotted, so bands stay within the data's own range.
_LOG_FLOOR_FRACTION = 0.5


def _style(algo):
    """Return marker/color/label for an algorithm, with fallback."""
    return ALGO_STYLE.get(algo, {"marker": "x", "color": "gray",
                                  "label": algo})


def _algos_to_plot(df, requested):
    """Requested algorithms that are actually present, in requested order."""
    present = set(df["algorithm"].unique())
    return [a for a in requested if a in present]


def _log_band_floor(df, algos, mean_col):
    """Positive lower clip for std bands on a log axis, from the data itself.

    Returns a fraction of the smallest positive mean among the series that will
    actually be plotted, so a band whose std exceeds its mean is truncated just
    below the lowest curve instead of plunging to an arbitrary absolute floor
    and stretching the axis over empty decades. None if no positive value
    exists (the caller then leaves the axis to matplotlib).
    """
    if mean_col not in df.columns:
        return None
    vals = pd.to_numeric(df.loc[df["algorithm"].isin(algos), mean_col],
                         errors="coerce").dropna()
    vals = vals[vals > 0]
    if vals.empty:
        return None
    return float(vals.min()) * _LOG_FLOOR_FRACTION


def _plot_series(ax, sub, mean_col, std_col, style, show_std, log_floor=None):
    """Plot one algorithm's curve, optionally with a +/- 1 std shade band.

    ``log_floor`` clips the band's lower edge to a positive value (for log-scale
    axes, where a non-positive edge is silently dropped); pass the value from
    ``_log_band_floor`` so the band stays within the plotted data range. When it
    is None the lower edge is clipped at 0 instead, since a negative MAE is
    meaningless.
    """
    if mean_col not in sub.columns:
        return

    ax.plot(sub["num_vars"], sub[mean_col],
            marker=style["marker"], color=style["color"], label=style["label"],
            linewidth=1.4, markersize=6)

    if not show_std or std_col is None or std_col not in sub.columns:
        return

    band = sub[["num_vars", mean_col, std_col]].dropna()
    if band.empty:
        return

    lo = band[mean_col] - band[std_col]
    hi = band[mean_col] + band[std_col]
    lo = lo.clip(lower=log_floor if log_floor is not None else 0.0)
    ax.fill_between(band["num_vars"], lo, hi,
                    color=style["color"], alpha=_BAND_ALPHA, linewidth=0)


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


def _plot_mae_panel(df, output_path, algorithms, show_std=False):
    """
    Two side-by-side panels: MAE lower bound (left) and upper bound (right).
    """
    algos = _algos_to_plot(df, algorithms)
    sizes = sorted(df["num_vars"].unique())

    fig, (ax_lb, ax_ub) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for algo in algos:
        sub = df[df["algorithm"] == algo].sort_values("num_vars")
        if sub.empty:
            continue
        s = _style(algo)
        _plot_series(ax_lb, sub, "mae_lb_error", "std_lb_error", s, show_std)
        _plot_series(ax_ub, sub, "mae_ub_error", "std_ub_error", s, show_std)

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
               bbox_to_anchor=(0.5, 1.02), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.91])

    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_mae(exact_path, ref_path, output_prefix, algorithms, show_std=False):
    """
    Generate the MAE plot from whichever CSV was supplied.
    """
    path = exact_path if (exact_path and os.path.exists(exact_path)) \
        else ref_path
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        _plot_mae_panel(df, f"{output_prefix}_mae.pdf", algorithms,
                        show_std=show_std)


def plot_runtime(df, output_prefix, exact_times, ref_times, algorithms,
                 show_std=False):
    """
    Figure 2: a single panel with the mean total time (build + run) vs. problem
    size on a log-log scale, one line per algorithm, optionally with a +/- 1
    standard deviation shade band (``std_total_time``).

    Where the std exceeds the mean the band's lower edge is clipped to just
    below the lowest plotted curve (see ``_log_band_floor``) so the log axis
    stays on the data's own range.

    Exact and reference baselines are drawn as separate dashed lines.
    """
    algos = _algos_to_plot(df, algorithms)
    sizes = sorted(df["num_vars"].unique())

    time_col = "mean_total_time" if "mean_total_time" in df.columns \
        else "total_time"
    # Clip bands to the plotted data's own range rather than a fixed floor.
    floor = _log_band_floor(df, algos, time_col)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.2))

    for algo in algos:
        sub = df[df["algorithm"] == algo].sort_values("num_vars")
        if sub.empty:
            continue
        s = _style(algo)

        _plot_series(ax, sub, time_col, "std_total_time", s, show_std,
                     log_floor=floor)

    if exact_times is not None and not exact_times.empty:
        ax.plot(exact_times["num_vars"], exact_times["exact_time"],
                linestyle="--", color="black", linewidth=1.5,
                marker="*", markersize=7, label="Exact", alpha=0.7)
    if ref_times is not None and not ref_times.empty:
        ax.plot(ref_times["num_vars"], ref_times["ref_time"],
                linestyle=":", color="dimgray", linewidth=1.5,
                marker=".", markersize=7, label="Reference", alpha=0.7)

    _set_log_xaxis(ax, sizes)
    ax.set_xlabel("Number of variables ($n$)")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Total time (build + run)", fontsize=11)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, which="both", linewidth=0.5)

    # Shared legend at the top
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=min(len(algos) + 2, 5), frameon=False,
               bbox_to_anchor=(0.5, 1.03), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    path = f"{output_prefix}_runtime.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot inference results from CSV files.")
    parser.add_argument(
        "--exact", type=str, default=None,
        help="Path to exact-comparison CSV (e.g. results-polytree-small.csv)")
    parser.add_argument(
        "--ref", type=str, default=None,
        help="Path to reference-comparison CSV (e.g. results-polytree-small.csv)")
    parser.add_argument(
        "--output", type=str, default="plots/results",
        help="Output prefix for PDF files (default: plots/results)")
    parser.add_argument(
        "--std", action="store_true",
        help="Shade a +/- 1 standard deviation band around each curve")
    parser.add_argument(
        "--algorithms", type=str, default=",".join(DEFAULT_ALGORITHMS),
        help="Comma-separated algorithms to plot, in plot order "
             f"(default: {','.join(DEFAULT_ALGORITHMS)})")
    args = parser.parse_args()

    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]

    df = load_and_merge(args.exact, args.ref)
    exact_times, ref_times = _extract_baselines(args.exact, args.ref)

    print(f"Loaded {len(df)} rows, "
          f"algorithms: {sorted(df['algorithm'].unique())}, "
          f"sizes: {sorted(df['num_vars'].unique())}")
    print(f"Plotting: {_algos_to_plot(df, algorithms)}"
          f"{' with std bands' if args.std else ''}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    plot_mae(args.exact, args.ref, args.output, algorithms,
             show_std=args.std)
    plot_runtime(df, args.output, exact_times, ref_times, algorithms,
                 show_std=args.std)


if __name__ == "__main__":
    main()
