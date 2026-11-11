"""A3 appendix: IBP algorithm box, Evidence subsection, evidence example."""
import sys

P = sys.argv[1]
s = open(P).read()


def sub(old, new):
    global s
    assert s.count(old) == 1, ("NOT UNIQUE/FOUND", old[:90], s.count(old))
    s = s.replace(old, new)


def cut(head, end_marker, new):
    """Replace from `head` up to (not incl.) the next `end_marker`."""
    global s
    a = s.index(head)
    b = s.index(end_marker, a + len(head))
    s = s[:a] + new + s[b:]


# ---- Algorithm box ----
sub(r'''\Require LCN $\cL$ with variables $\{X_1, \ldots, X_n\}$, evidence $\mathbf{E} = \mathbf{e}$, iterations $T$, threshold $\tau$
\Ensure Bounds $[\underline{P}(Y_i{=}y|\be),\; \overline{P}(Y_i{=}y|\be)]$ for
        non-evidence variables $Y_i$ and states $y$''',
    r'''\Require LCN $\cL$ with variables $\{X_1, \ldots, X_n\}$, iterations $T$, threshold $\tau$
\Ensure Bounds $[\underline{P}(Y_i{=}y),\; \overline{P}(Y_i{=}y)]$ for
        every variable $Y_i$ and state $y$ (no evidence)''')
sub(r'''\For{each evidence variable $E_i = e_i$}
  \State Set $\mu_{E_i \to f} \gets (\delta_{e_i}, \delta_{e_i})$ $\forall f \in \mathcal{N}(Y_j)$
\EndFor
''', '')
sub(r'''  \For{each $v$ ($\notin \bE$), each $f \in \mathcal{N}(v)$}''',
    r'''  \For{each $v$, each $f \in \mathcal{N}(v)$}''')
sub(r'''  \For{each $f_i$, each variable $v \in \mathrm{scope}(f_i)$}
    \State $\mathbf{l}_{\mathrm{new}} \gets \mathbf{1}, \quad \mathbf{u}_{\mathrm{new}} \gets \mathbf{0}$''',
    r'''  \For{each $f_i$ and each parent $w$ of $X_i$}
    \State $\mu_{f_i \to w} \gets (\mathbf 0,\mathbf 1)$ \Comment{vacuous upward}
  \EndFor
  \For{each $f_i$, child $v = X_i$}
    \State $\mathbf{l}_{\mathrm{new}} \gets \mathbf{1}, \quad \mathbf{u}_{\mathrm{new}} \gets \mathbf{0}$''')
sub(r'''\For{each non evidence variable $Y_i$}
  \For{each value $y \in D_{Y_i}$}
    \State $\underline{P}(Y_i{=}y \mid \mathbf{E}{=}\mathbf{e}) \gets \max_{f \in \mathcal{N}(Y_i)} l_{f \to Y_i}(y)$
    \State $\overline{P}(Y_i{=}x \mid \mathbf{E}{=}\mathbf{e}) \gets \min_{f \in \mathcal{N}(Y_i)} u_{f \to Y_i}(y)$
  \EndFor
\EndFor
\State \Return $[\underline{P}(Y_i{=}y \mid \mathbf{E}{=}\mathbf{e}),\; \overline{P}(Y_i{=}y \mid \mathbf{E}{=}\mathbf{e})]$''',
    r'''\For{each variable $Y_i$}
  \For{each value $y \in D_{Y_i}$}
    \State $\underline{P}(Y_i{=}y) \gets \max_{f \in \mathcal{N}(Y_i)} l_{f \to Y_i}(y)$
    \State $\overline{P}(Y_i{=}y) \gets \min_{f \in \mathcal{N}(Y_i)} u_{f \to Y_i}(y)$
  \EndFor
\EndFor
\State \Return $[\underline{P}(Y_i{=}y),\; \overline{P}(Y_i{=}y)]$''')

# ---- Evidence subsection -> scope statement ----
cut(r'''\subsection{Evidence}\label{sec:evidence}''',
    r'''\subsection{Factor-to-Variable Message Computation}''',
    r'''\subsection{Scope: no evidence}\label{sec:evidence}

IBP computes marginal bounds without evidence. A conditional table
$P(X_i\mid\mathbf{pa}(X_i))$ constrains nothing about its parents' marginals, so
the factor-to-parent messages are vacuous and IBP propagates information only
from parents to children. Conditioning on an observed descendant would require
upward (diagnostic) messages that combine by a likelihood product rather than
by intersection, which the interval algebra here does not provide. An earlier
version of the algorithm clamped evidence messages and normalized likelihoods
into parent messages; that injects a spurious constraint on the parents'
marginals and is not sound, so it was removed. For conditional queries use
\textsc{CJT}.

''')

# ---- Drop the evidence worked example ----
cut(r'''\subsection{Query with Evidence: $P(B{=}1 \mid E{=}1)$}\label{sec:example-evidence}''',
    r'''\subsection{Proofs of Theorem 3 and 4}''',
    '')

open(P, 'w').write(s)
print("ok")
