"""Plot the CSV written by analyze_results.py.

Figures (one PDF each, prefix ``--output``):
  * ``{prefix}_mae.pdf``      -- MAE of the lower / upper bound vs problem size,
                                 one line per run;
  * ``{prefix}_runtime.pdf``  -- mean total time (build + run) vs problem size,
                                 log-log, plus the reference run's time;
  * ``{prefix}_evidence.pdf`` -- (with ``--by-evidence``) MAE vs the number of
                                 observed atoms, averaged over sizes.

The CSV must be grouped by size (``--group-by ...,size``); rows are selected
with ``--benchmark`` and ``--evidence K`` when the CSV has those columns.
Run names, styles and labels come from experiments/runs.py. ``--std`` shades a
+/- one standard deviation band.

Usage:
    python experiments/analyze_results.py --results-dir results \\
        --group-by benchmark,evidence,size --output analysis.csv
    python experiments/plot_results.py --csv analysis.csv \\
        --benchmark tree_small_fr --evidence 0 --std --output plots/tree_small
"""

import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runs import label, order_key, style  # noqa: E402

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"

# The runs plotted by default, in legend order (the reference is drawn as its
# own time baseline).
DEFAULT_RUNS = ["ariel", "cve", "cve_eps_0.1", "ivbp", "cdve", "cjt_ipopt",
                "exact_ipopt"]

_BAND_ALPHA = 0.18
_LOG_FLOOR_FRACTION = 0.5


def _runs_present(df, requested):
    present = set(df["run"].unique())
    return sorted([r for r in requested if r in present], key=order_key)


def _series(ax, sub, x, mean_col, std_col, st, show_std, log_floor=None):
    sub = sub.sort_values(x)
    ax.plot(sub[x], sub[mean_col], marker=st["marker"], color=st["color"],
            label=st["label"], linewidth=1.4, markersize=6)
    if show_std and std_col in sub.columns:
        band = sub[[x, mean_col, std_col]].dropna()
        if not band.empty:
            lo = (band[mean_col] - band[std_col]).clip(
                lower=log_floor if log_floor is not None else 0.0)
            ax.fill_between(band[x], lo, band[mean_col] + band[std_col],
                            color=st["color"], alpha=_BAND_ALPHA, linewidth=0)


def _legend(fig, ax, n):
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=min(n, 5), frameon=False,
               bbox_to_anchor=(0.5, 1.03), fontsize=12)


def _log_x(ax, sizes):
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.minorticks_off()


_XLABEL = {"size": "Number of variables ($n$)",
           "nr_m": "Added non-realizable sentences ($m$)",
           "nr_delta": "Half-width of the added intervals ($\\delta$)"}


def _x_axis(ax, x, xs):
    if x == "size":
        _log_x(ax, xs)
    else:
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_xlabel(_XLABEL.get(x, x))


def plot_mae(df, runs, path, show_std, x="size"):
    xs = sorted(df[x].unique())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for r in runs:
        sub = df[df["run"] == r]
        _series(a1, sub, x, "mae_lb", "std_lb", style(r), show_std)
        _series(a2, sub, x, "mae_ub", "std_ub", style(r), show_std)
    for ax, t in ((a1, "Lower bound error"), (a2, "Upper bound error")):
        _x_axis(ax, x, xs)
        ax.set_title(t, fontsize=11)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_ylim(bottom=-0.01)
    a1.set_ylabel("MAE")
    _legend(fig, a1, len(runs))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_runtime(df, runs, path, show_std, x="size"):
    xs = sorted(df[x].unique())
    vals = pd.to_numeric(df[df["run"].isin(runs)]["total_t"], errors="coerce")
    vals = vals[vals > 0]
    floor = float(vals.min()) * _LOG_FLOOR_FRACTION if not vals.empty else None
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.2))
    for r in runs:
        _series(ax, df[df["run"] == r], x, "total_t", "std_total_t",
                style(r), show_std, log_floor=floor)
    ref = df.drop_duplicates(x)[[x, "ref_t"]].dropna()
    if not ref.empty:
        name = df["reference"].iloc[0]
        ax.plot(ref[x], ref["ref_t"], linestyle="--",
                color=style(name)["color"], marker=style(name)["marker"],
                linewidth=1.5, alpha=0.8, label=f"{label(name)} (reference)")
    _x_axis(ax, x, xs)
    ax.set_yscale("log")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Total time (build + run)", fontsize=11)
    ax.grid(True, alpha=0.25, which="both", linewidth=0.5)
    _legend(fig, ax, len(runs) + 1)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_evidence(df, runs, path):
    agg = df.groupby(["run", "evidence"], as_index=False)[["mae_lb",
                                                           "mae_ub"]].mean()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for r in runs:
        sub = agg[agg["run"] == r]
        _series(a1, sub, "evidence", "mae_lb", None, style(r), False)
        _series(a2, sub, "evidence", "mae_ub", None, style(r), False)
    for ax, t in ((a1, "Lower bound error"), (a2, "Upper bound error")):
        ax.set_xlabel("Observed atoms per instance")
        ax.set_xticks(sorted(agg["evidence"].unique()))
        ax.set_title(t, fontsize=11)
        ax.grid(True, alpha=0.25, linewidth=0.5)
    a1.set_ylabel("MAE (mean over sizes)")
    _legend(fig, a1, len(runs))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", required=True, help="analyze_results.py CSV")
    p.add_argument("--output", default="plots/results",
                   help="output prefix (default: plots/results)")
    p.add_argument("--runs", default=",".join(DEFAULT_RUNS),
                   help=f"comma list of runs (default: {','.join(DEFAULT_RUNS)})")
    p.add_argument("--benchmark", default=None,
                   help="keep only this benchmark's rows")
    p.add_argument("--evidence", type=int, default=None,
                   help="keep only rows with this many observed atoms "
                        "(MAE / runtime figures)")
    p.add_argument("--by-evidence", action="store_true",
                   help="also plot MAE vs the number of observed atoms")
    p.add_argument("--std", action="store_true",
                   help="shade +/- 1 std bands")
    p.add_argument("--x", default="size",
                   help="x-axis column: size (default), nr_m or nr_delta "
                        "(the CSV must be grouped by it)")
    p.add_argument("--type", default=None,
                   help="keep only rows of this topology (column 'type')")
    p.add_argument("--size", type=int, default=None,
                   help="keep only rows of this size (column 'size')")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if args.x not in df.columns:
        p.error(f"the CSV must be grouped by {args.x} (--group-by ...,{args.x})")
    if args.benchmark is not None and "benchmark" in df.columns:
        df = df[df["benchmark"] == args.benchmark]
    if args.type is not None and "type" in df.columns:
        df = df[df["type"] == args.type]
    if args.size is not None and "size" in df.columns:
        df = df[df["size"] == args.size]
    runs = _runs_present(df, [r.strip() for r in args.runs.split(",")
                              if r.strip()])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.by_evidence and "evidence" in df.columns:
        plot_evidence(df, runs, f"{args.output}_evidence.pdf")
    sel = df
    if "evidence" in df.columns:
        k = args.evidence if args.evidence is not None else \
            int(sorted(df["evidence"].unique())[0])
        sel = df[df["evidence"] == k]
    print(f"{len(sel)} rows; runs {runs}")
    plot_mae(sel, runs, f"{args.output}_mae.pdf", args.std, x=args.x)
    plot_runtime(sel, runs, f"{args.output}_runtime.pdf", args.std, x=args.x)


if __name__ == "__main__":
    main()
