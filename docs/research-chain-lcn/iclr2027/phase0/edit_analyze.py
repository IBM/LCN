"""Rewrite experiments/analyze_results.py for the six-algorithm harness."""
import sys

P = sys.argv[1]
s = open(P).read()


def sub(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:70], s.count(old))
    s = s.replace(old, new)


def cut(start, end, new):
    global s
    a = s.index(start)
    b = s.index(end, a)
    s = s[:a] + new + s[b:]


cut('''Loads all per-algorithm JSONL files from the results directory''',
    '''Reported metrics''',
    '''Loads all per-run JSONL files from the results directory
(``results/{benchmark}/{run}.jsonl``), groups them by instance, and computes
error metrics for every other run relative to the chosen reference.

A *run* is an algorithm together with its variant flags; its name is the
algorithm plus the variant suffix written by run_experiment.py (e.g. ``cve``,
``cve_c10mean``, ``cjt``, ``cjt_ipopt``, ``exact``). Records are keyed by run
name, so variants of one algorithm are scored separately.

The reference is selected with ``--reference`` (a run name; default ``cjt``).
For the two exact engines -- ``cjt`` and ``exact``, solved by SCIP -- only
**certified** endpoints are used: an endpoint the solver did not certify
(its reported value is then a sound dual bound, not the optimum) is excluded
from the lower- or upper-bound error statistics of that variable. Any other
run name (e.g. ``ariel``) is used as-is.

''')

cut('''# The algorithm names produced by run_algorithm.py (see its ALGORITHMS list),''',
    '''def _mean(vals):''',
    '''# The algorithms of run_algorithm.py (run names may add a variant suffix).
_ALGORITHMS = ["exact", "ariel", "cjt", "ivbp", "cve", "approxlp"]

# Engines whose endpoints carry a certification flag. When one of them is the
# reference only its certified endpoints are compared; when it is not, it is
# excluded from the scored rows (it is ground truth, not an approximation).
_EXACT_ENGINES = ("cjt", "exact")

# Display names used in the LaTeX tables (the raw run names stay in the CSV).
_LATEX_ALGO_NAMES = {
    "approxlp": "CDVE",
    "ariel": "ARIEL",
    "ivbp": "IVBP",
    "cve": "CVE",
    "cve_c10mean": "CVE$_{cm}$",
    "cve_c10plub": "CVE$_{cp}$",
    "cjt": "CJT",
    "cjt_ipopt": "CJT (ipopt)",
    "exact": "Exact",
    "exact_ipopt": "Exact (ipopt)",
}

# Runs omitted from the LaTeX tables (still present in the CSV).
_LATEX_EXCLUDE = ("cve_c10plub",)


def run_name(rec):
    """Run name of a record: the algorithm plus the suffix its variant implies
    (identical to the output file name written by run_experiment.py)."""
    algo = rec["algorithm"]
    variant = rec.get("variant") or {}
    try:
        from run_algorithm import variant_suffix
    except ImportError:  # analysis run outside experiments/
        return algo
    return algo + variant_suffix(algo, variant)


def _engine_of(name):
    """The exact engine a run name belongs to (``cjt``/``exact``), or None."""
    for e in _EXACT_ENGINES:
        if name == e or name.startswith(e + "_"):
            return e
    return None


def _certified_endpoints(rec):
    """{atom: (lower_certified, upper_certified)} for a CJT or exact record;
    None if the record carries no certification (treat all as certified)."""
    if rec.get("cjt_solves"):
        cert = {}
        for s in rec["cjt_solves"]:
            lo, hi = cert.get(s["atom"], (True, True))
            if s["sense"] == "min":
                lo = bool(s["certified"])
            else:
                hi = bool(s["certified"])
            cert[s["atom"]] = (lo, hi)
        return cert
    if rec.get("exact_status"):
        return {a: (st["min"] == "confirmed", st["max"] == "confirmed")
                for a, st in rec["exact_status"].items()}
    return None


''')

sub('''def _group_by_instance(records):
    """Group records by instance path -> {algorithm: record}."""
    groups = defaultdict(dict)
    for rec in records:
        if rec["status"] != "ok":
            continue
        groups[rec["instance"]][rec["algorithm"]] = rec
    return groups''',
    '''def _group_by_instance(records):
    """Group records by instance path -> {run name: record}."""
    groups = defaultdict(dict)
    for rec in records:
        if rec["status"] != "ok":
            continue
        groups[rec["instance"]][run_name(rec)] = rec
    return groups''')
sub('''        return (os.path.basename(instance), rec["algorithm"])
    else:
        return (rec["graph_type"], rec["num_vars"], rec["algorithm"])''',
    '''        return (os.path.basename(instance), run_name(rec))
    else:
        return (rec["graph_type"], rec["num_vars"], run_name(rec))''')

cut('''def _reference_record(algos, reference):''', '''def analyze(''',
    '''def _reference_record(algos, reference):
    """The reference record of an instance's {run name: record} map, or None."""
    return algos.get(reference)


def _is_excluded(name, reference):
    """Whether run ``name`` is left out of the scored rows: the reference
    itself, and any run of an exact engine (ground truth, not an
    approximation)."""
    return name == reference or _engine_of(name) is not None


''')

sub('''def analyze(records, reference="ariel", output_file=None, latex_file=None,''',
    '''def analyze(records, reference="cjt", output_file=None, latex_file=None,''')
sub('''        reference: reference algorithm name, or "exact" for the exact bounds.''',
    '''        reference: reference run name (e.g. "cjt", "exact", "ariel").''')
sub('''        ref_marg = ref_rec["marginals"]''',
    '''        ref_marg = ref_rec["marginals"]
        ref_cert = (_certified_endpoints(ref_rec)
                    if _engine_of(reference) else None)''')
sub('''        for algo_name, rec in algos.items():
            if _is_reference_or_exact(algo_name, reference):
                continue''',
    '''        for algo_name, rec in algos.items():
            if _is_excluded(algo_name, reference):
                continue''')
sub('''                # Absolute errors on P(var=1) bounds (index 1)
                s["abs_lb_errors"].append(abs(r_lo[1] - a_lo[1]))
                s["abs_ub_errors"].append(abs(a_hi[1] - r_hi[1]))''',
    '''                # Only certified reference endpoints are compared.
                lo_ok, hi_ok = ((True, True) if ref_cert is None
                                else ref_cert.get(var, (False, False)))
                if not (lo_ok and hi_ok):
                    s["n_uncertified"] = s.get("n_uncertified", 0) + 1
                # Absolute errors on P(var=1) bounds (index 1)
                if lo_ok:
                    s["abs_lb_errors"].append(abs(r_lo[1] - a_lo[1]))
                if hi_ok:
                    s["abs_ub_errors"].append(abs(a_hi[1] - r_hi[1]))
                if not (lo_ok and hi_ok):
                    continue  # containment / width need both endpoints''')

sub('''        if not s["abs_lb_errors"]:
            continue''',
    '''        if not s["abs_lb_errors"] or not s["abs_ub_errors"]:
            continue''')

sub('''        "--reference", type=str, default="ariel", choices=_REFERENCE_CHOICES,
        metavar="ALGO",
        help="Reference algorithm (default: ariel). Use 'exact' for the "
             "ground-truth exact bounds (prefers exact_g then exact_l per "
             "instance), 'cjt_any' for CredalJT (prefers cjt_l then cjt per "
             "instance), or one of " + ", ".join(_ALGORITHMS))''',
    '''        "--reference", type=str, default="cjt", metavar="RUN",
        help="Reference run name (default: cjt). For cjt and exact only "
             "certified endpoints are compared. Run names are the algorithm "
             "plus its variant suffix, e.g. " + ", ".join(_ALGORITHMS)
             + ", cve_c10mean, cjt_ipopt")''')

open(P, 'w').write(s)
print("ok")
