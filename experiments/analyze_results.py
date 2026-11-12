"""Analyze inference results against a reference run.

Loads every record under the results directory
(``results/{benchmark}/ev{K}/{run}.jsonl``, any depth), groups the records by
(benchmark, evidence level, instance), and scores every run against the
reference run on the same instance.

Runs are the canonical names of experiments/runs.py (``exact_scip``,
``exact_ipopt``, ``ariel``, ``cjt_scip``, ``cjt_ipopt``, ``cve``,
``cve_eps_0.1``, ``ivbp``, ``cdve``, ...), read from each record's ``run``
field. Records without one (the pre-revision logs in results_legacy/) are
skipped and counted.

Reference
---------
``--reference RUN`` (default ``cjt_scip``). For the certified engines
(``cjt_scip``, ``exact_scip``) only **certified** endpoints are compared
(for CJT: certified solve AND structurally exact route, ``cjt_atom_structural``): an
endpoint the solver did not certify (its value is a sound dual bound, not the
optimum) is left out of the lower- or upper-bound statistics. Every other run
is scored, including the ipopt runs (local solvers, no guarantee) and the other
certified engine, which then serves as a cross-check. Atoms observed in the
evidence are not scored.

Metrics, on P(atom = 1), per atom, aggregated over the group
-------------------------------------------------------------------------
  * ``mae_lb`` / ``mae_ub`` (with ``std_*``, ``max_*``) -- absolute error of
        the lower / upper endpoint vs the reference (certified endpoints only).
  * ``outer`` -- fraction of atoms whose interval contains the reference
        interval; ``inner`` -- fraction contained in it (both within --tol).
  * ``wr`` -- mean width ratio (run width / reference width, reference width
        > 1e-12).
  * ``viol`` -- number of atoms violating the run's declared guarantee (see
        runs.GUARANTEES). This is checked against the reference for
        ``certified`` / ``exact_se`` / ``exact_poly`` (when the record reports
        ``ivbp_exact``) and for ``inner``; ``inner_eps`` runs are checked against
        the ``cve`` run of the same instance when present (inner, within a
        factor 1+eps), else against the reference (inner). Strong-extension
        guarantees equal LCN guarantees only on family-realizable instances,
        so a violation on a non-family-realizable instance is expected, not a
        bug. The first violations are printed; ``--violations FILE`` writes all
        of them.
  * ``solved`` -- ok / timeout / memory / error counts of the run in the group
        (over all its records, not only the scored ones).
  * times: mean build / run / total time, ``ref_t`` (reference total time),
        mean induced width.

Grouping: ``--group-by`` is a comma list of benchmark, evidence, type, size,
instance (default ``benchmark,evidence,size``), and for the perturbed
(almost-realizable) instances of ``lcn/benchmarks/gen_nonrealizable.py`` also
nr_kind, nr_m, nr_delta (read from the file name; ``type`` is then the base
topology), e.g. ``--group-by type,size,nr_m``.

Usage:
    python experiments/analyze_results.py --results-dir results
    python experiments/analyze_results.py --results-dir results \\
        --reference exact_scip --group-by evidence,size \\
        --output analysis.csv --latex analysis.tex
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runs import RUNS, guarantee, label, order_key, parse_run  # noqa: E402

DEFAULT_REFERENCE = "cjt_scip"
# Tolerance of containment / guarantee tests vs a certified reference
# (CJT's default certification tolerance).
DEFAULT_TOL = 1e-4
# Tolerance of the CVE_eps vs CVE check (same engine, exact arithmetic).
_EPS_TOL = 1e-9
_GROUP_FIELDS = ("benchmark", "evidence", "type", "size", "instance",
                 "nr_kind", "nr_m", "nr_delta")

# Perturbed instances (lcn/benchmarks/gen_nonrealizable.py):
# {topology}_nr{kind}_m{m}_d{100*delta}_n{n}_{i}.lcn
_NR_RE = re.compile(r"^(\w+?)_nr(\w+?)_m(\d+)_d(\d+)_n\d+_\d+\.lcn$")


def _nr_info(basename):
    """(topology, kind, m, delta) of a perturbed instance, else None."""
    mt = _NR_RE.match(basename)
    if not mt:
        return None
    return mt.group(1), mt.group(2), int(mt.group(3)), int(mt.group(4)) / 100


def _mean(v):
    return sum(v) / len(v) if v else float("nan")


def _std(v):
    if len(v) < 2:
        return 0.0 if v else float("nan")
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def load_records(results_dir):
    """All records under ``results_dir`` that carry a ``run`` field, and
    the number of records skipped for lacking one."""
    records, skipped = [], 0
    for path in sorted(glob.glob(os.path.join(results_dir, "**", "*.jsonl"),
                                 recursive=True)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "run" not in rec:
                    skipped += 1
                    continue
                _normalize_status(rec)
                records.append(rec)
    return records, skipped


def _normalize_status(rec):
    """Count a numpy allocation failure as a memory-out.

    A ``MemoryError`` raised inside the worker (e.g. under the RLIMIT_AS cap)
    is logged by the runner as ``status: "error"`` with numpy's
    ``Unable to allocate ...`` message, or (newer records, which carry the
    exception type) as ``MemoryError: ...``; only a worker killed by a signal
    gets ``status: "memory"``. All are the same outcome.
    """
    err = rec.get("error") or ""
    if rec.get("status") == "error" and \
            ("Unable to allocate" in err or "MemoryError" in err):
        rec["status"] = "memory"


def _instance_key(rec):
    return (rec.get("benchmark", "?"), rec.get("evidence_atoms", 0),
            rec["instance"])


def _group_key(rec, fields):
    base = os.path.basename(rec.get("instance", "?"))
    nr = _nr_info(base)
    vals = {"benchmark": rec.get("benchmark", "?"),
            "evidence": rec.get("evidence_atoms", 0),
            "type": nr[0] if nr else rec.get("graph_type", "?"),
            "size": rec.get("num_vars", 0),
            "instance": base,
            "nr_kind": nr[1] if nr else "-",
            "nr_m": nr[2] if nr else 0,
            "nr_delta": nr[3] if nr else 0.0}
    return tuple(vals[f] for f in fields)


# ----------------------------------------------------------------------
# Certification and intervals
# ----------------------------------------------------------------------

def certified_endpoints(rec):
    """{atom: (lower certified, upper certified)} for a certified engine's
    record; None means "treat every endpoint as certified" (runs without a
    certification notion)."""
    if rec.get("cjt_solves"):
        # A CJT endpoint is a valid reference only if its solve is certified
        # AND the atom's route is structurally exact (H1-H4): a certified
        # optimum of a relaxed program (H3 relaxed, or a tree violating H1)
        # is only an outer bound. Records predating the per-atom field fall
        # back to the instance-level structural flag.
        structural = rec.get("cjt_atom_structural")
        default = bool(rec.get("cjt_structurally_exact", False))
        cert = {}
        for s in rec["cjt_solves"]:
            ok_struct = (structural.get(s["atom"], False)
                         if structural is not None else default)
            lo, hi = cert.get(s["atom"], (True, True))
            if s["sense"] == "min":
                lo = lo and bool(s.get("certified")) and ok_struct
            else:
                hi = hi and bool(s.get("certified")) and ok_struct
            cert[s["atom"]] = (lo, hi)
        return cert
    if rec.get("exact_status"):
        return {a: (st.get("min") == "confirmed", st.get("max") == "confirmed")
                for a, st in rec["exact_status"].items()}
    return None


def _interval(rec, atom):
    m = rec.get("marginals", {}).get(atom)
    if not m:
        return None
    return float(m["lower"][1]), float(m["upper"][1])


def _cert(rec, atom):
    cert = certified_endpoints(rec) if guarantee(rec["run"]) == "certified" \
        else None
    if cert is None:
        return True, True
    return cert.get(atom, (False, False))


# ----------------------------------------------------------------------
# Guarantee checks
# ----------------------------------------------------------------------

def check_guarantee(rec, ref, cve_rec, atom, tol=DEFAULT_TOL):
    """Violations of ``rec``'s declared guarantee on ``atom`` as a list of
    (endpoint, value, bound) tuples; [] if none or not checkable."""
    g = guarantee(rec["run"])
    a = _interval(rec, atom)
    r = _interval(ref, atom)
    if a is None or r is None or g == "none":
        return []
    rlo_ok, rhi_ok = _cert(ref, atom)
    out = []
    if g in ("certified", "exact_se") or \
            (g == "exact_poly" and rec.get("ivbp_exact")):
        alo_ok, ahi_ok = _cert(rec, atom)
        if rlo_ok and alo_ok and abs(a[0] - r[0]) > tol:
            out.append(("lower", a[0], r[0]))
        if rhi_ok and ahi_ok and abs(a[1] - r[1]) > tol:
            out.append(("upper", a[1], r[1]))
    elif g == "inner":
        if rlo_ok and a[0] < r[0] - tol:
            out.append(("lower", a[0], r[0]))
        if rhi_ok and a[1] > r[1] + tol:
            out.append(("upper", a[1], r[1]))
    elif g == "inner_eps":
        eps = parse_run(rec["run"])[1]["epsilon"]
        c = _interval(cve_rec, atom) if cve_rec is not None else None
        if c is not None:            # vs exact CVE, same engine: tight test
            if not (c[0] - _EPS_TOL <= a[0] <= (1 + eps) * c[0] + _EPS_TOL):
                out.append(("lower", a[0], c[0]))
            if not (c[1] / (1 + eps) - _EPS_TOL <= a[1] <= c[1] + _EPS_TOL):
                out.append(("upper", a[1], c[1]))
        else:                        # vs the reference: inner only
            if rlo_ok and a[0] < r[0] - tol:
                out.append(("lower", a[0], r[0]))
            if rhi_ok and a[1] > r[1] + tol:
                out.append(("upper", a[1], r[1]))
    return out


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------

def _new_stats():
    return {k: [] for k in ("lb", "ub", "outer", "inner", "wr", "build",
                            "run", "total", "iw")} | {"viol": 0, "atoms": 0,
                                                       "instances": set()}


def analyze(records, reference=DEFAULT_REFERENCE, group_by=None,
            tol=DEFAULT_TOL, runs=None, verbose=True):
    """Score every run against ``reference``. Returns (rows, violations):
    rows are dicts (one per group and run, registry order), violations
    (benchmark, K, instance, run, atom, endpoint, value, bound) tuples."""
    parse_run(reference)                       # validates the name
    fields = list(group_by or ("benchmark", "evidence", "size"))
    for f in fields:
        if f not in _GROUP_FIELDS:
            raise ValueError(f"unknown group field {f!r}; use {_GROUP_FIELDS}")

    by_inst = defaultdict(dict)
    solved = defaultdict(Counter)
    for rec in records:
        if runs is not None and rec["run"] not in runs \
                and rec["run"] != reference:
            continue
        solved[_group_key(rec, fields) + (rec["run"],)][rec["status"]] += 1
        if rec["status"] == "ok":
            by_inst[_instance_key(rec)][rec["run"]] = rec

    stats = defaultdict(_new_stats)
    ref_times = defaultdict(list)
    violations = []
    for (bench, k, inst), recs in sorted(by_inst.items()):
        ref = recs.get(reference)
        if ref is None:
            continue
        gk = _group_key(ref, fields)
        ref_times[gk].append(ref.get("total_time", 0.0))
        evidence = set(ref.get("evidence") or {})
        cve_rec = recs.get("cve")
        for name, rec in recs.items():
            if name == reference:
                continue
            s = stats[gk + (name,)]
            s["instances"].add(inst)
            s["build"].append(rec.get("build_time", 0.0))
            s["run"].append(rec.get("run_time", 0.0))
            s["total"].append(rec.get("total_time", 0.0))
            if rec.get("induced_width") is not None:
                s["iw"].append(rec["induced_width"])
            for atom in ref.get("marginals", {}):
                if atom in evidence:
                    continue
                r, a = _interval(ref, atom), _interval(rec, atom)
                if r is None or a is None:
                    continue
                s["atoms"] += 1
                lo_ok, hi_ok = _cert(ref, atom)
                if lo_ok:
                    s["lb"].append(abs(r[0] - a[0]))
                if hi_ok:
                    s["ub"].append(abs(a[1] - r[1]))
                if lo_ok and hi_ok:
                    s["outer"].append(int(a[0] <= r[0] + tol and
                                          a[1] >= r[1] - tol))
                    s["inner"].append(int(a[0] >= r[0] - tol and
                                          a[1] <= r[1] + tol))
                    if r[1] - r[0] > 1e-12:
                        s["wr"].append((a[1] - a[0]) / (r[1] - r[0]))
                for v in check_guarantee(rec, ref, cve_rec, atom, tol):
                    s["viol"] += 1
                    violations.append((bench, k, os.path.basename(inst), name,
                                       atom) + v)

    rows = []
    keys = sorted(stats, key=lambda k: (k[:-1], order_key(k[-1])))
    for key in keys:
        s = stats[key]
        gk, name = key[:-1], key[-1]
        c = solved.get(key, Counter())
        row = {f: v for f, v in zip(fields, gk)}
        row.update({
            "run": name, "label": label(name), "guarantee": guarantee(name),
            "reference": reference, "n_instances": len(s["instances"]),
            "n_atoms": s["atoms"],
            "mae_lb": _mean(s["lb"]), "std_lb": _std(s["lb"]),
            "max_lb": max(s["lb"]) if s["lb"] else float("nan"),
            "mae_ub": _mean(s["ub"]), "std_ub": _std(s["ub"]),
            "max_ub": max(s["ub"]) if s["ub"] else float("nan"),
            "outer": _mean(s["outer"]), "inner": _mean(s["inner"]),
            "wr": _mean(s["wr"]), "viol": s["viol"],
            "ok": c["ok"], "timeout": c["timeout"], "memory": c["memory"],
            "error": c["error"],
            "build_t": _mean(s["build"]), "run_t": _mean(s["run"]),
            "total_t": _mean(s["total"]), "std_total_t": _std(s["total"]),
            "ref_t": _mean(ref_times[gk]),
            "iw": _mean(s["iw"]) if s["iw"] else None,
        })
        rows.append(row)
    if verbose:
        _print(rows, fields, reference, violations)
    return rows, violations


def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def _print(rows, fields, reference, violations, max_viol=20):
    print(f"Reference: {reference}  ({label(reference)}; certified endpoints "
          f"only)\n")
    head = [f[:5] for f in fields] + ["run", "inst", "atoms", "mae_lb",
                                      "mae_ub", "max_lb", "max_ub", "outer",
                                      "inner", "wr", "viol", "solved",
                                      "total_t", "ref_t"]
    lines = []
    for r in rows:
        solved = f"{r['ok']}/{r['ok'] + r['timeout'] + r['memory'] + r['error']}"
        lines.append([str(r[f]) for f in fields] + [
            r["run"], str(r["n_instances"]), str(r["n_atoms"]),
            _fmt(r["mae_lb"]), _fmt(r["mae_ub"]), _fmt(r["max_lb"]),
            _fmt(r["max_ub"]), _fmt(r["outer"], 3), _fmt(r["inner"], 3),
            _fmt(r["wr"], 3), str(r["viol"]), solved, _fmt(r["total_t"], 3),
            _fmt(r["ref_t"], 3)])
    widths = [max(len(h), *(len(x[i]) for x in lines)) if lines else len(h)
              for i, h in enumerate(head)]
    print("  ".join(h.rjust(w) for h, w in zip(head, widths)))
    print("-" * (sum(widths) + 2 * len(widths)))
    for x in lines:
        print("  ".join(c.rjust(w) for c, w in zip(x, widths)))
    if violations:
        print(f"\n{len(violations)} guarantee violation(s) (first "
              f"{min(max_viol, len(violations))}); expected only on "
              f"non-family-realizable instances:")
        for v in violations[:max_viol]:
            b, k, inst, name, atom, end, val, bound = v
            print(f"  {b}/ev{k} {inst} {name} {atom} {end}: {val:.6f} vs "
                  f"{bound:.6f}")


# ----------------------------------------------------------------------
# Output files
# ----------------------------------------------------------------------

def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 8) if isinstance(v, float) else v)
                        for k, v in r.items()})


def write_violations(violations, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "evidence_atoms", "instance", "run", "atom",
                    "endpoint", "value", "bound"])
        w.writerows(violations)


def _tex(v, nd):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def write_latex(rows, path, fields, reference):
    """Two tables: means (MAE, containment, time) and spreads (std)."""
    group_cols = [f for f in fields if f != "instance"] or fields
    def table(f, cols, caption):
        f.write("\\begin{table}[ht]\n\\centering\n\\scriptsize\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write("\\begin{tabular}{" + "l" * (len(group_cols) + 1) +
                "r" * len(cols) + "}\n\\toprule\n")
        f.write(" & ".join([c.capitalize() for c in group_cols] + ["Method"] +
                           [c[1] for c in cols]) + " \\\\\n\\midrule\n")
        prev = None
        for r in rows:
            g = tuple(r[c] for c in group_cols)
            if prev is not None and g != prev:
                f.write("\\midrule\n")
            cells = [str(r[c]) if g != prev else "" for c in group_cols]
            prev = g
            cells.append(label(r["run"], latex=True))
            cells += [_tex(r[c[0]], c[2]) for c in cols]
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")
    with open(path, "w") as f:
        ref = label(reference, latex=True)
        table(f, [("mae_lb", "MAE$_{lb}$", 4), ("mae_ub", "MAE$_{ub}$", 4),
                  ("outer", "outer", 3), ("inner", "inner", 3),
                  ("viol", "viol.", 0), ("total_t", "time (s)", 2)],
              f"Error metrics vs.\\ {ref} (certified endpoints).")
        table(f, [("std_lb", "std$_{lb}$", 4), ("std_ub", "std$_{ub}$", 4),
                  ("max_lb", "max$_{lb}$", 4), ("max_ub", "max$_{ub}$", 4),
                  ("std_total_t", "std time", 2)],
              f"Spread of the errors vs.\\ {ref}.")


def main():
    p = argparse.ArgumentParser(
        description="Score runs against a reference run.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--reference", default=DEFAULT_REFERENCE, metavar="RUN",
                   help=f"reference run (default {DEFAULT_REFERENCE}); one of "
                        + ", ".join(RUNS) + ", ...")
    p.add_argument("--runs", nargs="+", default=None, metavar="RUN",
                   help="restrict to these runs (default: all present)")
    p.add_argument("--group-by", default="benchmark,evidence,size",
                   help="comma list of " + ", ".join(_GROUP_FIELDS))
    p.add_argument("--tol", type=float, default=DEFAULT_TOL)
    p.add_argument("--output", default=None, help="CSV of the rows")
    p.add_argument("--latex", default=None, help="LaTeX tables")
    p.add_argument("--violations", default=None,
                   help="CSV of every guarantee violation")
    args = p.parse_args()
    try:
        parse_run(args.reference)
        for r in args.runs or []:
            parse_run(r)
    except ValueError as e:
        p.error(str(e))

    records, skipped = load_records(args.results_dir)
    if skipped:
        print(f"[note] skipped {skipped} record(s) without a 'run' field "
              f"(pre-revision logs)\n")
    fields = [f.strip() for f in args.group_by.split(",") if f.strip()]
    rows, viol = analyze(records, reference=args.reference, group_by=fields,
                         tol=args.tol, runs=args.runs)
    if not rows:
        print(f"No instance has a successful '{args.reference}' record.")
        return
    if args.output:
        write_csv(rows, args.output)
        print(f"\nSaved CSV to {args.output}")
    if args.latex:
        write_latex(rows, args.latex, fields, args.reference)
        print(f"Saved LaTeX to {args.latex}")
    if args.violations:
        write_violations(viol, args.violations)
        print(f"Saved violations to {args.violations}")


if __name__ == "__main__":
    main()
