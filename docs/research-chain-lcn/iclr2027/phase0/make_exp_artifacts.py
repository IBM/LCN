"""Regenerate every experimental table and figure of the ICLR paper from results/.

    python docs/research-chain-lcn/iclr2027/phase0/make_exp_artifacts.py [--results-dir results]

Writes (relative to the paper directory):
  outputs/results-{tree,polytree,ktree,dag}-{small,large}.tex   per-size tables (appendix)
  outputs/results-nr.tex, outputs/results-nr-ablation.tex       almost-realizable tables
  outputs/results-reference.tex                                 CJT certification table
  outputs/results-summary.tex                                   main-text summary table
  outputs/results-real.tex                                      realistic LCNs (if present)
  outputs/numbers.txt                                           every number quoted in the text
  figs/{set}_mae.pdf, figs/{set}_runtime.pdf                    per-set curves
  figs/main_results.pdf                                         main-text figure
CSVs from experiments/analyze_results.py are kept in outputs/csv/.
"""
import argparse
import collections
import glob
import json
import math
import os
import re
import statistics as st
import subprocess
import sys

PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(PAPER, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments"))
from runs import label, style  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"

TOPOS = ["tree", "polytree", "ktree", "dag"]
FR_SETS = [f"{t}_{s}_fr" for t in TOPOS for s in ("small", "large")]
NR_SETS = ["nr_tree", "nr_dag", "nr_dag_ablation"]
APPROX = ["ariel", "cjt_ipopt", "cve", "cve_eps_0.01", "ivbp", "cdve"]
EXACT = ["exact_scip", "exact_ipopt"]
PLOT_RUNS = ["ariel", "cjt_ipopt", "cve", "cve_eps_0.01", "ivbp", "cdve"]
NUMBERS = []


def note(key, value):
    NUMBERS.append(f"{key}: {value}")


# ----------------------------------------------------------------- loading
def load_raw(results, bench):
    out = collections.defaultdict(list)
    for f in glob.glob(os.path.join(results, bench, "ev0", "*.jsonl")):
        for line in open(f):
            line = line.strip()
            if line:
                r = json.loads(line)
                if "run" in r:
                    out[r["run"]].append(r)
    return out


def failure_kind(r):
    """ok / timeout / memory / error, with allocation failures (and the bare
    MemoryError raised under RLIMIT_AS, whose message is empty) as memory."""
    s = r.get("status")
    if s in ("ok", "timeout", "memory"):
        return s
    e = str(r.get("error") or "")
    if e == "" or "allocate" in e or "MemoryError" in e:
        return "memory"
    return "error"


def analyze(results, group_by, out_csv, runs=None):
    cmd = [sys.executable, os.path.join(REPO, "experiments", "analyze_results.py"),
           "--results-dir", results, "--group-by", group_by, "--output", out_csv]
    if runs:
        cmd += ["--runs", *runs]
    subprocess.run(cmd, check=True, cwd=REPO, stdout=subprocess.DEVNULL)
    return pd.read_csv(out_csv)


# ----------------------------------------------------------------- helpers
def f4(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if abs(x) < 5e-5:
        return "0"
    return f"{x:.4f}".lstrip("0") if x < 1 else f"{x:.3f}"


def f3(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.3f}"


def ft(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if x < 10:
        return f"{x:.2f}"
    if x < 100:
        return f"{x:.1f}"
    return f"{x:.0f}"


def lab(run):
    return label(run, latex=True)


def solved_counts(raw, run, n=None):
    c = collections.Counter(failure_kind(r) for r in raw.get(run, [])
                            if n is None or r["num_vars"] == n)
    return c


def solved_str(c):
    tot = sum(c.values())
    s = f"{c['ok']}/{tot}"
    extra = []
    if c["timeout"]:
        extra.append(f"{c['timeout']}t")
    if c["memory"]:
        extra.append(f"{c['memory']}m")
    if c["error"]:
        extra.append(f"{c['error']}e")
    return s + (f" ({','.join(extra)})" if extra else "")


def median_time(raw, run, n=None):
    t = [r["total_time"] for r in raw.get(run, [])
         if r["status"] == "ok" and (n is None or r["num_vars"] == n)]
    return st.median(t) if t else float("nan")


def cert_stats(raw, n=None):
    """(certified endpoints, endpoints, fully certified instances, instances)."""
    ce = ne = full = 0
    rs = [r for r in raw.get("cjt_scip", []) if n is None or r["num_vars"] == n]
    for r in rs:
        s = r.get("cjt_solves") or [] if r["status"] == "ok" else []
        c = sum(1 for x in s if x.get("certified") and
                (r.get("cjt_atom_structural") or {}).get(x["atom"], True))
        ce += c
        ne += len(s) if s else 2 * r["num_vars"]
        full += bool(s) and c == len(s)
    return ce, ne, full, len(rs)


# ----------------------------------------------------------------- tables
def per_set_table(df, raw, bench, path):
    topo, scale = bench.split("_")[:2]
    sub = df[df["benchmark"] == bench]
    runs = (EXACT if scale == "small" else ["exact_scip"]) + APPROX
    sizes = sorted(sub["size"].unique())
    with open(path, "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\scriptsize\n")
        f.write(f"\\caption{{Random \\texttt{{{topo}}} LCNs, {scale} scale: errors against the "
                "certified CJT (SCIP) endpoints, per size ($10$ instances each). "
                "MAE and SD are over atoms; max is the largest error of either endpoint; "
                "out/in are the fractions of atoms whose interval contains / is contained "
                "in the reference interval (tolerance $10^{-4}$); solved counts give "
                "timeouts (t) and memory-outs (m); time is the median total time in seconds "
                "over the solved instances. Exact inference over the $2^n$ joint is run only "
                "for $n \\le 15$; its \\emph{solved} instances may hold uncertified (dual) "
                "endpoints, hence its large errors.}\n")
        f.write(f"\\label{{tab:full-{topo}-{scale}}}\n")
        f.write("\\begin{tabular}{rlrrrrrrrlr}\n\\toprule\n")
        f.write("$n$ & Method & MAE$_l$ & SD$_l$ & MAE$_u$ & SD$_u$ & max & out & in "
                "& solved & time \\\\\n\\midrule\n")
        for n in sizes:
            ce, ne, full, ni = cert_stats(raw, n)
            f.write(f"{n} & CJT (SCIP), reference & \\multicolumn{{7}}{{l}}{{certified endpoints "
                    f"{ce}/{ne}, instances {full}/{ni}}} & "
                    f"{solved_str(solved_counts(raw, 'cjt_scip', n))} & "
                    f"{ft(median_time(raw, 'cjt_scip', n))} \\\\\n")
            for run in runs:
                c = solved_counts(raw, run, n)
                if not c:
                    continue
                row = sub[(sub["size"] == n) & (sub["run"] == run)]
                if len(row):
                    r = row.iloc[0]
                    cells = [f4(r.mae_lb), f4(r.std_lb), f4(r.mae_ub), f4(r.std_ub),
                             f4(max(r.max_lb, r.max_ub)), f3(r.outer), f3(r.inner)]
                else:
                    cells = ["--"] * 7
                f.write(" & " + lab(run) + " & " + " & ".join(cells) + " & "
                        + solved_str(c) + " & " + ft(median_time(raw, run, n)) + " \\\\\n")
            if n != sizes[-1]:
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def reference_table(raws, path):
    with open(path, "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\footnotesize\n")
        f.write("\\caption{The CJT (SCIP) reference on every synthetic set: endpoints certified "
                "(proven optimal or primal--dual gap $\\le 10^{-4}$, on a structurally exact route), "
                "instances with every endpoint certified, runs finished, median and maximum total time "
                "(s), the largest cluster (atoms) of the tree CJT builds and the median and maximum number of "
                "nonlinear (LMC) rows of its program; with none, the program is a linear program. Every "
                "instance uses the ordered triage and none relaxes an assertion.}\n\\label{tab:reference}\n")
        f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\nSet & endpoints & instances & finished & "
                "median t & max t & cluster & nonlinear \\\\\n\\midrule\n")
        for bench in FR_SETS + NR_SETS:
            raw = raws[bench]
            ce, ne, full, ni = cert_stats(raw)
            rs = [r for r in raw["cjt_scip"] if r["status"] == "ok"]
            t = [r["total_time"] for r in rs]
            w = max((r.get("nlp_stats") or {}).get("max_cluster_atoms", 0) for r in rs)
            modes = collections.Counter((r.get("cjt_h3") or {}).get("mode") for r in rs)
            nl = [(r.get("nlp_stats") or {}).get("n_nonlinear", 0) for r in rs]
            relaxed = sum(((r.get("cjt_h3") or {}).get("relaxed", 0) > 0) for r in rs)
            note(f"ref {bench}", f"cert {ce}/{ne}={100*ce/ne:.1f}% full {full}/{ni} "
                 f"tmed {st.median(t):.1f} tmax {max(t):.0f} width {w} modes {dict(modes)} relaxed {relaxed} nonlinear med {st.median(nl)} max {max(nl)}")
            f.write(f"\\texttt{{{bench.replace('_', chr(92) + '_')}}} & {ce}/{ne} ({100*ce/ne:.1f}\\%) & "
                    f"{full}/{ni} & {len(rs)}/{len(raw['cjt_scip'])} & {ft(st.median(t))} & "
                    f"{ft(max(t))} & {w} & {st.median(nl):g} / {max(nl)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def nr_table(df, path):
    runs = ["ariel", "cjt_ipopt", "cve", "ivbp", "cdve"]
    with open(path, "w") as f:
        for typ, g0 in df.groupby("type"):
            f.write("\\begin{table}[htbp]\n\\centering\n\\scriptsize\n")
            f.write(f"\\caption{{Almost-realizable LCNs on \\texttt{{{typ}}} bases with $m$ added "
                    "non-realizable sentences ($\\delta=0.1$, $10$ instances per cell), against the certified "
                    "CJT (SCIP) endpoints. wr is the mean width ratio (method / reference). "
                    "CVE$_\\varepsilon$ ($\\varepsilon{=}0.01$) coincides with CVE to four decimals in every "
                    "cell and is omitted. Time is the mean total time (s); ref.\\ is the reference's.}\n")
            f.write(f"\\label{{tab:nr-{typ}}}\n")
            f.write("\\begin{tabular}{rrlrrrrrr}\n\\toprule\n$n$ & $m$ & Method & MAE$_l$ & MAE$_u$ "
                    "& out & in & wr & time \\\\\n\\midrule\n")
            first = True
            for n, g in g0.groupby("size"):
                for m in sorted(g["nr_m"].unique()):
                    if not first:
                        f.write("\\midrule\n" if m == 0 else "\\cmidrule(l){2-9}\n")
                    first = False
                    gg = g[g["nr_m"] == m]
                    ref_t = gg["ref_t"].iloc[0]
                    for i, run in enumerate(runs):
                        r = gg[gg["run"] == run]
                        if not len(r):
                            continue
                        r = r.iloc[0]
                        head = f"{n} & {m}" if i == 0 else " & "
                        tail = f" (ref.\\ {ft(ref_t)})" if i == 0 else ""
                        f.write(f"{head} & {lab(run)} & {f4(r.mae_lb)} & {f4(r.mae_ub)} & {f3(r.outer)} & "
                                f"{f3(r.inner)} & {r.wr:.2f} & {ft(r.total_t)}{tail} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")


def ablation_table(df, path):
    runs = ["ariel", "cjt_ipopt", "cve", "ivbp", "cdve"]
    kinds = {"1": "\\texttt{nr1} (non-root marginal)", "2": "\\texttt{nr2} (mixture conditional)",
             "3": "\\texttt{nr3} (cross-family)"}
    with open(path, "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\scriptsize\n")
        f.write("\\caption{Ablation by kind: \\texttt{dag} bases with $n=10$ and $m=4$ added sentences of a "
                "single kind, for two half-widths $\\delta$ ($10$ instances per cell), against the certified "
                "CJT (SCIP) endpoints. Columns as in Table~\\ref{tab:nr-tree}.}\n\\label{tab:nr-ablation}\n")
        f.write("\\begin{tabular}{lrlrrrrrr}\n\\toprule\nKind & $\\delta$ & Method & MAE$_l$ & MAE$_u$ & "
                "out & in & wr & time \\\\\n\\midrule\n")
        keys = sorted({(str(k), d) for k, d in zip(df["nr_kind"], df["nr_delta"])})
        for j, (k, d) in enumerate(keys):
            if j:
                f.write("\\midrule\n")
            gg = df[(df["nr_kind"].astype(str) == k) & (df["nr_delta"] == d)]
            for i, run in enumerate(runs):
                r = gg[gg["run"] == run]
                if not len(r):
                    continue
                r = r.iloc[0]
                head = f"{kinds.get(k, k)} & {d:g}" if i == 0 else " & "
                f.write(f"{head} & {lab(run)} & {f4(r.mae_lb)} & {f4(r.mae_ub)} & {f3(r.outer)} & "
                        f"{f3(r.inner)} & {r.wr:.2f} & {ft(r.total_t)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def summary_table(dfb, raws, path):
    """Main text: per set, MAE = mean of MAE_l and MAE_u, out/in, solved, median t at the largest n."""
    sets = [("tree_large_fr", "tree"), ("polytree_large_fr", "polytree"),
            ("ktree_large_fr", "ktree"), ("dag_large_fr", "dag"),
            ("nr_tree", "nr-tree"), ("nr_dag", "nr-dag")]
    runs = ["cjt_scip"] + APPROX
    with open(path, "w") as f:
        f.write("\\begin{tabular}{l" + "rr" * len(sets) + "}\n\\toprule\n")
        f.write(" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{\\texttt{{{s}}}}}" for _, s in sets)
                + " \\\\\n")
        f.write("".join(f"\\cmidrule(lr){{{2 + 2*i}-{3 + 2*i}}}" for i in range(len(sets))) + "\n")
        f.write("Method" + " & MAE & time" * len(sets) + " \\\\\n\\midrule\n")
        for run in runs:
            cells = []
            for bench, _ in sets:
                raw = raws[bench]
                nmax = max(r["num_vars"] for r in raw["cjt_scip"])
                if run == "cjt_scip":
                    ce, ne, _, _ = cert_stats(raw)
                    cells += [f"({100*ce/ne:.0f}\\%)", ft(median_time(raw, run, nmax))]
                    continue
                row = dfb[(dfb["benchmark"] == bench) & (dfb["run"] == run)]
                c = solved_counts(raw, run)
                if not len(row) or not c:
                    cells += ["--", "--"]
                    continue
                r = row.iloc[0]
                mae = 0.5 * (r.mae_lb + r.mae_ub)
                frac = c["ok"] / sum(c.values())
                mark = "" if frac == 1 else f"$^{{{100*frac:.0f}}}$"
                t = median_time(raw, run, nmax)
                cells += [f4(mae) + mark, ft(t)]
            name = "CJT (SCIP), reference" if run == "cjt_scip" else lab(run)
            f.write(name + " & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def real_table(df, raws, benches, path):
    runs = ["exact_scip", "exact_ipopt", "ariel", "cjt_ipopt", "cve", "cve_eps_0.01", "ivbp", "cdve"]
    with open(path, "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\scriptsize\n")
        f.write("\\caption{Realistic LCNs, per instance, against the certified CJT (SCIP) endpoints. "
                "Columns as in the random tables; `--' in the error columns means the run did not finish "
                "or the reference certified no endpoint.}\n\\label{tab:real-full}\n")
        f.write("\\begin{tabular}{llrrrrrlr}\n\\toprule\nInstance & Method & MAE$_l$ & MAE$_u$ & max & out "
                "& in & solved & time \\\\\n\\midrule\n")
        first = True
        for bench in benches:
            raw = raws[bench]
            insts = sorted({r["instance"] for r in raw.get("cjt_scip", [])})
            for inst in insts:
                if not first:
                    f.write("\\midrule\n")
                first = False
                base = os.path.basename(inst)[:-4].replace("_consistent", "")
                ref = [r for r in raw["cjt_scip"] if r["instance"] == inst][0]
                s = ref.get("cjt_solves") or []
                ce = sum(1 for x in s if x.get("certified"))
                name = "\\texttt{" + base.replace("_", "\\_") + "}"
                f.write(f"\\multicolumn{{9}}{{l}}{{{name} ($n{{=}}{ref['num_vars']}$); reference: "
                        f"{ce}/{len(s) or 2*ref['num_vars']} endpoints certified, "
                        f"{ft(ref['total_time'])} s, status {ref['status']}}} \\\\\n")
                for run in runs:
                    rr = [r for r in raw.get(run, []) if r["instance"] == inst]
                    if not rr:
                        continue
                    row = df[(df["instance"] == os.path.basename(inst)) & (df["run"] == run)] \
                        if "instance" in df.columns else []
                    if len(row):
                        r = row.iloc[0]
                        cells = [f4(r.mae_lb), f4(r.mae_ub), f4(max(r.max_lb, r.max_ub)),
                                 f3(r.outer), f3(r.inner)]
                    else:
                        cells = ["--"] * 5
                    k = failure_kind(rr[0])
                    t = ft(rr[0]["total_time"]) if k == "ok" else "--"
                    f.write(f" & {lab(run)} & " + " & ".join(cells) + f" & {k} & {t} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# ----------------------------------------------------------------- figures
def _ax_series(ax, sub, x, col, run, std=None):
    sub = sub.sort_values(x)
    s = style(run)
    ax.plot(sub[x], sub[col], marker=s["marker"], color=s["color"], label=s["label"],
            linewidth=1.4, markersize=5)
    if std is not None and std in sub:
        ax.fill_between(sub[x], (sub[col] - sub[std]).clip(lower=0), sub[col] + sub[std],
                        color=s["color"], alpha=0.15, linewidth=0)


def set_figures(df, raws, bench, figdir):
    sub = df[df["benchmark"] == bench].copy()
    sub["mae"] = 0.5 * (sub["mae_lb"] + sub["mae_ub"])
    sizes = sorted(sub["size"].unique())
    runs = [r for r in PLOT_RUNS if r in set(sub["run"])]
    # error
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for run in runs:
        g = sub[sub["run"] == run]
        _ax_series(a1, g, "size", "mae_lb", run, "std_lb")
        _ax_series(a2, g, "size", "mae_ub", run, "std_ub")
    for ax, t in ((a1, "Lower bound"), (a2, "Upper bound")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.minorticks_off()
        ax.set_xlabel("Number of atoms ($n$)")
        ax.set_title(t, fontsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_ylim(bottom=-0.002)
    a1.set_ylabel("MAE vs. reference")
    h, lb = a1.get_legend_handles_labels()
    fig.legend(h, lb, loc="upper center", ncol=len(runs), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(figdir, f"{bench}_mae.pdf"), bbox_inches="tight")
    if os.environ.get("PREVIEW_DIR"):
        fig.savefig(os.path.join(os.environ["PREVIEW_DIR"], f"{bench}_mae.png"), bbox_inches="tight", dpi=70)
    plt.close(fig)
    # time (median over solved instances, from the raw records)
    raw = raws[bench]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for run in ["cjt_scip"] + runs:
        xs, ys = [], []
        for n in sizes:
            t = median_time(raw, run, n)
            if not math.isnan(t):
                xs.append(n)
                ys.append(t)
        s = style(run)
        ax.plot(xs, ys, marker=s["marker"], color=s["color"],
                label=s["label"] + (" (ref.)" if run == "cjt_scip" else ""),
                linestyle="--" if run == "cjt_scip" else "-", linewidth=1.4, markersize=5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.minorticks_off()
    ax.set_yscale("log")
    ax.set_xlabel("Number of atoms ($n$)")
    ax.set_ylabel("Median total time (s)")
    ax.grid(True, alpha=0.25, which="both", linewidth=0.5)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"{bench}_runtime.pdf"), bbox_inches="tight")
    if os.environ.get("PREVIEW_DIR"):
        fig.savefig(os.path.join(os.environ["PREVIEW_DIR"], f"{bench}_runtime.png"), bbox_inches="tight", dpi=70)
    plt.close(fig)


def main_figure(df, dfnr, raws, figdir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
    # (a) dag_large MAE (mean of both endpoints)
    a = axes[0]
    sub = df[df["benchmark"] == "dag_large_fr"].copy()
    sub["mae"] = 0.5 * (sub["mae_lb"] + sub["mae_ub"])
    sizes = sorted(sub["size"].unique())
    for run in PLOT_RUNS:
        _ax_series(a, sub[sub["run"] == run], "size", "mae", run)
    a.set_title("(a) \\texttt{dag} (large): error", fontsize=10)
    a.set_ylabel("MAE vs. reference")
    # (b) dag_large time
    b = axes[1]
    raw = raws["dag_large_fr"]
    for run in ["cjt_scip"] + PLOT_RUNS:
        xs = [n for n in sizes if not math.isnan(median_time(raw, run, n))]
        s = style(run)
        b.plot(xs, [median_time(raw, run, n) for n in xs], marker=s["marker"], color=s["color"],
               linestyle="--" if run == "cjt_scip" else "-", linewidth=1.4, markersize=5,
               label=s["label"] + (" (reference)" if run == "cjt_scip" else ""))
    b.set_yscale("log")
    b.set_title("(b) \\texttt{dag} (large): median time (s)", fontsize=10)
    for ax in (a, b):
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.minorticks_off()
        ax.set_xlabel("Number of atoms ($n$)")
        ax.grid(True, alpha=0.25, which="both", linewidth=0.5)
    # (c) nr: MAE vs m, n = 10, solid = dag, dashed = tree
    c = axes[2]
    for typ, ls in (("dag", "-"), ("tree", ":")):
        g = dfnr[(dfnr["type"] == typ) & (dfnr["size"] == 10)].copy()
        g["mae"] = 0.5 * (g["mae_lb"] + g["mae_ub"])
        for run in ["ariel", "cjt_ipopt", "cve", "ivbp"]:
            gg = g[g["run"] == run].sort_values("nr_m")
            s = style(run)
            c.plot(gg["nr_m"], gg["mae"], marker=s["marker"], color=s["color"], linestyle=ls,
                   linewidth=1.4, markersize=5)
    c.set_xticks([0, 1, 2, 4, 8])
    c.set_xlabel("Added non-realizable sentences ($m$), $n=10$")
    c.set_title("(c) almost-realizable: error (solid dag, dotted tree)", fontsize=10)
    c.grid(True, alpha=0.25, linewidth=0.5)
    h, lb = b.get_legend_handles_labels()
    fig.legend(h, lb, loc="upper center", ncol=len(lb), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ax in axes:
        ax.title.set_text(ax.get_title().replace("\\texttt{", "").replace("}", ""))
    fig.savefig(os.path.join(figdir, "main_results.pdf"), bbox_inches="tight")
    if os.environ.get("PREVIEW_DIR"):
        fig.savefig(os.path.join(os.environ["PREVIEW_DIR"], "main_results.png"), bbox_inches="tight", dpi=80)
    plt.close(fig)


def nr_figures(dfnr, figdir):
    for typ in ("tree", "dag"):
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for n, ls in ((10, "-"), (20, "--")):
            g = dfnr[(dfnr["type"] == typ) & (dfnr["size"] == n)].copy()
            g["mae"] = 0.5 * (g["mae_lb"] + g["mae_ub"])
            for run in ["ariel", "cjt_ipopt", "cve", "ivbp", "cdve"]:
                gg = g[g["run"] == run].sort_values("nr_m")
                s_ = style(run)
                ax.plot(gg["nr_m"], gg["mae"], marker=s_["marker"], color=s_["color"], linestyle=ls,
                        linewidth=1.3, markersize=5, label=s_["label"] if n == 10 else None)
        ax.set_xticks([0, 1, 2, 4, 8])
        ax.set_xlabel("Added non-realizable sentences ($m$)")
        ax.set_ylabel("MAE vs. reference")
        ax.set_title(f"{typ} bases", fontsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, f"nr_{typ}_mae_m.pdf"), bbox_inches="tight")
        if os.environ.get("PREVIEW_DIR"):
            fig.savefig(os.path.join(os.environ["PREVIEW_DIR"], f"nr_{typ}_mae_m.png"), bbox_inches="tight", dpi=70)
        plt.close(fig)


# ----------------------------------------------------------------- numbers
def collect_numbers(dfb, dfs, dfnr, dfab, raws):
    for bench in FR_SETS + NR_SETS:
        raw = raws[bench]
        for run in EXACT + APPROX:
            c = solved_counts(raw, run)
            if c:
                note(f"solved {bench} {run}", solved_str(c))
        for run in APPROX:
            r = dfb[(dfb["benchmark"] == bench) & (dfb["run"] == run)]
            if len(r):
                r = r.iloc[0]
                note(f"agg {bench} {run}", f"mae {r.mae_lb:.5f}/{r.mae_ub:.5f} max {r.max_lb:.4f}/"
                     f"{r.max_ub:.4f} out {r.outer:.3f} in {r.inner:.3f} wr {r.wr:.3f} viol {r.viol}")
    for bench in FR_SETS:
        raw = raws[bench]
        for n in sorted({r["num_vars"] for r in raw["cjt_scip"]}):
            note(f"tmed {bench} n={n}", " ".join(f"{run}={median_time(raw, run, n):.2f}"
                                                 for run in ["cjt_scip"] + APPROX + EXACT))
        rs = [r for r in raw["ivbp"] if r["status"] == "ok"]
        note(f"ivbp {bench}", f"exact {sum(bool(r.get('ivbp_exact')) for r in rs)}/{len(rs)} "
             f"converged {sum(bool(r.get('ivbp_converged')) for r in rs)} "
             f"iters median {st.median([r.get('ivbp_iterations') or 0 for r in rs])}")
        ex = collections.defaultdict(collections.Counter)
        for r in raw.get("exact_scip", []):
            if r["status"] != "ok":
                ex[r["num_vars"]][failure_kind(r)] += 1
                continue
            for v in (r.get("exact_status") or {}).values():
                for s_ in v.values():
                    ex[r["num_vars"]][s_] += 1
        note(f"exact_scip status {bench}", {n: dict(v) for n, v in sorted(ex.items())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(REPO, "results"))
    args = ap.parse_args()
    res = args.results_dir
    out = os.path.join(PAPER, "outputs")
    figdir = os.path.join(PAPER, "figs")
    csvdir = os.path.join(out, "csv")
    os.makedirs(csvdir, exist_ok=True)

    benches = sorted(d.split(os.sep)[-2] for d in glob.glob(os.path.join(res, "*", "ev0")))
    raws = {b: load_raw(res, b) for b in benches}

    dfs = analyze(res, "benchmark,size", os.path.join(csvdir, "by_size.csv"))
    dfb = analyze(res, "benchmark", os.path.join(csvdir, "by_benchmark.csv"))
    dfnr = analyze(res, "benchmark,type,size,nr_m", os.path.join(csvdir, "nr_by_m.csv"))
    dfnr = dfnr[dfnr["benchmark"].isin(["nr_tree", "nr_dag"])]
    dfab = analyze(res, "benchmark,nr_kind,nr_delta", os.path.join(csvdir, "nr_ablation.csv"))
    dfab = dfab[dfab["benchmark"] == "nr_dag_ablation"]

    for bench in FR_SETS:
        topo, scale = bench.split("_")[:2]
        per_set_table(dfs, raws[bench], bench, os.path.join(out, f"results-{topo}-{scale}.tex"))
        set_figures(dfs, raws, bench, figdir)
    reference_table(raws, os.path.join(out, "results-reference.tex"))
    nr_table(dfnr, os.path.join(out, "results-nr.tex"))
    ablation_table(dfab, os.path.join(out, "results-nr-ablation.tex"))
    summary_table(dfb, raws, os.path.join(out, "results-summary.tex"))
    main_figure(dfs, dfnr, raws, figdir)
    nr_figures(dfnr, figdir)
    collect_numbers(dfb, dfs, dfnr, dfab, raws)

    real = [b for b in ("real", "junkyu") if b in raws and raws[b].get("cjt_scip")]
    if real:
        dfi = analyze(res, "benchmark,instance", os.path.join(csvdir, "real_by_instance.csv"))
        dfi = dfi[dfi["benchmark"].isin(real)]
        dfr = analyze(res, "benchmark", os.path.join(csvdir, "real_by_benchmark.csv"))
        real_table(dfi, raws, real, os.path.join(out, "results-real.tex"))
        for b in real:
            for run in EXACT + APPROX:
                c = solved_counts(raws[b], run)
                note(f"solved {b} {run}", solved_str(c))
                r = dfr[(dfr["benchmark"] == b) & (dfr["run"] == run)]
                if len(r):
                    r = r.iloc[0]
                    note(f"agg {b} {run}", f"mae {r.mae_lb:.4f}/{r.mae_ub:.4f} out {r.outer:.3f} "
                         f"in {r.inner:.3f} wr {r.wr:.3f} t {r.total_t:.2f}")
            ce, ne, full, ni = cert_stats(raws[b])
            note(f"ref {b}", f"cert {ce}/{ne} full {full}/{ni}")
    with open(os.path.join(out, "numbers.txt"), "w") as f:
        f.write("\n".join(NUMBERS) + "\n")
    print(f"wrote tables to {out}, figures to {figdir}")


if __name__ == "__main__":
    main()
