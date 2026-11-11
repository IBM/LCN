"""Uniform notation: bold X = binary LCN propositions; bold V = credal-network
variables (possibly compound). One-off; prints a per-region change count.

Regions are located by section headings, not line numbers, so the script is
robust to earlier edits. Usage: notation.py FILE [--dry]"""
import re
import sys

P = sys.argv[1]
DRY = "--dry" in sys.argv
s = open(P).read()
count = {}


def region(start, end):
    a = s.index(start)
    b = s.index(end, a)
    return a, b


def apply(name, a, b, rules):
    global s
    seg = s[a:b]
    n = 0
    for pat, rep in rules:
        seg, k = re.subn(pat, rep, seg)
        n += k
    count[name] = n
    s = s[:a] + seg + s[b:]


# Element-level CN rename: X_i/X_j/X_{...} -> V_*, bold sets -> \bV, but keep
# the LCN input lists "\{X_1, \ldots, X_n\}" in algorithm \Require lines.
KEEP = "@@LCNVARS@@"
CN_RULES = [
    (r"\\Require LCN \$\\cL\$ with variables \$\\\{X_1, \\ldots, X_n\\\}\$",
     r"\\Require LCN $\\cL$ with variables " + KEEP),
    (r"\\mathbf\{X\}'", r"\\bV'"),
    (r"\\mathbf\{X\}", r"\\bV"),
    (r"\\bX\b", r"\\bV"),
    (r"(?<![A-Za-z\\])X_\{([^}]*)\}", r"V_{\1}"),
    (r"(?<![A-Za-z\\])X_([ijk0-9n])", r"V_\1"),
    (KEEP.replace("@", r"\@"), r"$\\{X_1, \\ldots, X_n\\}$"),
]

# Atom-level appendix rename: \mathbf V (binary atoms) -> \bX.
ATOM_RULES = [
    (r"\\mathbf V'", r"\\bX'"),
    (r"\\mathbf V\b", r"\\bX"),
    (r"\\mathbf \\bX", r"\\bX"),
]

# ---- Main text ----
# 2.1 Credal networks: CN variables.
a, b = region(r"\subsection{Credal Networks}", r"\subsection{Logical Credal Networks}")
apply("main 2.1 CN", a, b, CN_RULES)
# 2.2 LCN: tidy \mathbf \bX.
a, b = region(r"\subsection{Logical Credal Networks}", r"\subsection{Chain Graphs and Factorization}")
apply("main 2.2 LCN", a, b, ATOM_RULES)
# 4 compilation + algorithms: CN variables, except the LCN input sentence.
a, b = region(r"\section{Approximate Inference via Credal Network Compilation}",
              r"\section{Experiments}")
apply("main 4", a, b, CN_RULES)
# Restore the first sentence of 4.1, which is about the LCN's atoms.
s = s.replace(r"Given a chain-graph LCN $\mathcal{L}$ over variables $\bV=\{V_1,\ldots,V_n\}$",
              r"Given a chain-graph LCN $\mathcal{L}$ over propositions $\bX=\{X_1,\ldots,X_n\}$")

# ---- Appendix ----
a, b = region(r"\section{Preliminaries}", r"\section{Detailed Description of Credal Variable Elimination}")
apply("app A-E (atoms)", a, b, ATOM_RULES)
a, b = region(r"\section{Detailed Description of Credal Variable Elimination}",
              r"\section{Detailed Experimental Evaluation}")
apply("app F-H (CN)", a, b, CN_RULES)

print(count)
if not DRY:
    open(P, "w").write(s)
    print("written")
