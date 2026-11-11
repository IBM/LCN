"""The registry of evaluated runs: one naming scheme for the whole harness.

A *run* is an algorithm with its parameters. Its canonical **run name** is
used for the results file (``{run}.jsonl``), the ``run`` field of every
record, the analysis CSV / LaTeX rows and the plot legends:

    run name        algorithm  parameters        guarantee (vs D_LCN)
    exact_scip      exact      solver=scip       certified endpoints exact
    exact_ipopt     exact      solver=ipopt      none (local NLP)
    ariel           ariel      (ipopt)           none
    cjt_scip        cjt        solver=scip       certified endpoints exact
    cjt_ipopt       cjt        solver=ipopt      none (local NLP)
    cve             cve                          exact vs D_SE
    cve_eps_{E}     cve        epsilon=E         inner vs D_SE, factor 1+E
    ivbp            ivbp                         exact vs D_SE on polytrees
    cdve            cdve                         inner vs D_SE

Rules: the solver is ALWAYS part of the name of the two NLP engines (``--solver``
is required for them); the CVE epsilon value is part of the name
(``cve_eps_0.1``; ``cve_eps`` alone is not a run). Tuning parameters that do
not change the method (``n_iters`` of ariel / ivbp / cdve, ``n_restarts`` of
cdve) keep their defaults; a non-default value is appended as a suffix
(``cdve_r0``, ``ariel_it20``) so different settings never share a file.
"""

import re

ALGORITHMS = ["exact", "ariel", "cjt", "cve", "ivbp", "cdve"]
SOLVERS = ["scip", "ipopt"]
SOLVER_ALGORITHMS = ("exact", "cjt")

# Defaults of the parameters that a run name can encode.
DEFAULT_PARAMS = {
    "exact": {"solver": None},
    "cjt": {"solver": None},
    "ariel": {"n_iters": 10},
    "cve": {"epsilon": None},
    "ivbp": {"n_iters": 100},
    "cdve": {"n_iters": 50, "n_restarts": 8},
}

# Default epsilon of the CVE_eps run of the evaluation grid.
DEFAULT_EPSILON = 0.1

# The nine runs of the evaluation grid, in table / legend order.
RUNS = ["exact_scip", "exact_ipopt", "ariel", "cjt_scip", "cjt_ipopt",
        "cve", f"cve_eps_{DEFAULT_EPSILON:g}", "ivbp", "cdve"]

# Guarantee classes (checked by analyze_results):
#   certified  -- exact on every endpoint the solver certifies
#   exact_se   -- exact w.r.t. the strong extension (= D_LCN when the instance
#                 is family-realizable)
#   exact_poly -- exact w.r.t. the strong extension when the record reports
#                 ivbp_exact (polytree credal network); none otherwise
#   inner_eps  -- inner w.r.t. the strong extension, within a factor 1 + eps
#   inner      -- inner w.r.t. the strong extension
#   none       -- no guarantee (local solvers, loopy message passing)
GUARANTEES = ("certified", "exact_se", "exact_poly", "inner_eps", "inner",
              "none")

_STYLE = {
    "exact_scip":  ("*", "#000000"),
    "exact_ipopt": ("x", "#7f7f7f"),
    "ariel":       ("o", "#1f77b4"),
    "cjt_scip":    ("v", "#ff7f0e"),
    "cjt_ipopt":   ("<", "#bcbd22"),
    "cve":         ("D", "#9467bd"),
    "cve_eps":     ("h", "#e377c2"),
    "ivbp":        ("^", "#d62728"),
    "cdve":        ("s", "#2ca02c"),
}


def params_of(algorithm, kwargs):
    """The name-relevant parameters of a run: the algorithm's defaults
    overridden by matching keys of ``kwargs`` (None means 'not given')."""
    if algorithm not in DEFAULT_PARAMS:
        raise ValueError(f"unknown algorithm {algorithm!r}; "
                         f"use one of {ALGORITHMS}")
    p = dict(DEFAULT_PARAMS[algorithm])
    for k in p:
        if kwargs.get(k) is not None:
            p[k] = kwargs[k]
    if algorithm == "cve" and p["epsilon"] is not None \
            and float(p["epsilon"]) == 0.0:
        p["epsilon"] = None                  # epsilon 0 is exact CVE
    return p


def run_name(algorithm, kwargs):
    """Canonical run name of ``algorithm`` with parameters ``kwargs``."""
    p = params_of(algorithm, kwargs)
    d = DEFAULT_PARAMS[algorithm]
    if algorithm in SOLVER_ALGORITHMS:
        if p["solver"] not in SOLVERS:
            raise ValueError(f"{algorithm} needs an explicit solver "
                             f"({' or '.join(SOLVERS)}); got {p['solver']!r}")
        return f"{algorithm}_{p['solver']}"
    if algorithm == "cve":
        return "cve" if p["epsilon"] is None else \
            f"cve_eps_{float(p['epsilon']):g}"
    name = algorithm
    if p["n_iters"] != d["n_iters"]:
        name += f"_it{p['n_iters']}"
    if algorithm == "cdve" and p["n_restarts"] != d["n_restarts"]:
        name += f"_r{p['n_restarts']}"
    return name


_RE_SOLVER = re.compile(r"^(exact|cjt)_(scip|ipopt)$")
_RE_EPS = re.compile(r"^cve_eps_([0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?)$")
_RE_TUNED = re.compile(r"^(ariel|ivbp|cdve)((?:_it[0-9]+)?)((?:_r[0-9]+)?)$")


def parse_run(name):
    """(algorithm, params) of a canonical run name; ValueError otherwise."""
    m = _RE_SOLVER.match(name)
    if m:
        return m.group(1), {"solver": m.group(2)}
    if name == "cve":
        return "cve", {"epsilon": None}
    m = _RE_EPS.match(name)
    if m:
        eps = float(m.group(1))
        if eps <= 0:
            raise ValueError("cve_eps needs a positive epsilon")
        return "cve", {"epsilon": eps}
    m = _RE_TUNED.match(name)
    if m:
        algo = m.group(1)
        if m.group(3) and algo != "cdve":
            raise ValueError(f"{name!r}: restarts apply to cdve only")
        p = dict(DEFAULT_PARAMS[algo])
        if m.group(2):
            p["n_iters"] = int(m.group(2)[3:])
        if m.group(3):
            p["n_restarts"] = int(m.group(3)[2:])
        if run_name(algo, p) != name:          # e.g. an explicit default
            raise ValueError(f"{name!r} is not canonical; "
                             f"use {run_name(algo, p)!r}")
        return algo, p
    raise ValueError(
        f"unknown run {name!r}; runs are {', '.join(RUNS)} (cve_eps_<E> for "
        f"any epsilon E > 0, optional _it<N> / _r<R> tuning suffixes)")


def _base(name):
    """The run family of a name: strips tuning suffixes and the epsilon."""
    if name.startswith("cve_eps_"):
        return "cve_eps"
    return re.sub(r"(_it[0-9]+)?(_r[0-9]+)?$", "", name)


def guarantee(name):
    """Guarantee class of a run (see GUARANTEES)."""
    algo, p = parse_run(name)
    if algo in SOLVER_ALGORITHMS:
        return "certified" if p["solver"] == "scip" else "none"
    if algo == "cve":
        return "exact_se" if p["epsilon"] is None else "inner_eps"
    return {"ariel": "none", "ivbp": "exact_poly", "cdve": "inner"}[algo]


def label(name, latex=False):
    """Display label of a run (LaTeX form for tables, plain/mathtext for
    plots)."""
    algo, p = parse_run(name)
    if algo in SOLVER_ALGORITHMS:
        base = {"exact": "Exact", "cjt": "CJT"}[algo]
        return f"{base} ({'SCIP' if p['solver'] == 'scip' else 'ipopt'})"
    if algo == "cve":
        if p["epsilon"] is None:
            return "CVE"
        e = f"{p['epsilon']:g}"
        return (f"CVE$_\\varepsilon$ ($\\varepsilon{{=}}{e}$)" if latex
                else f"CVE$_\\varepsilon$ ($\\varepsilon$={e})")
    base = {"ariel": "ARIEL", "ivbp": "IVBP", "cdve": "CDVE"}[algo]
    extra = []
    if p["n_iters"] != DEFAULT_PARAMS[algo]["n_iters"]:
        extra.append(f"T={p['n_iters']}")
    if algo == "cdve" and p["n_restarts"] != DEFAULT_PARAMS[algo]["n_restarts"]:
        extra.append(f"R={p['n_restarts']}")
    return base + (f" ({', '.join(extra)})" if extra else "")


def style(name):
    """{marker, color, label} for plots."""
    marker, color = _STYLE.get(_base(name), ("x", "gray"))
    return {"marker": marker, "color": color, "label": label(name)}


def order_key(name):
    """Sort key placing runs in registry (table / legend) order."""
    base = _base(name)
    bases = [_base(r) for r in RUNS]
    return (bases.index(base) if base in bases else len(bases), name)
