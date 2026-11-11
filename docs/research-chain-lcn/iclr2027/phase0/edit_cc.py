"""Paper: conditional queries now use the Charnes-Cooper scaling (no floor)."""
import sys

P = sys.argv[1]
s = open(P).read()


def sub(old, new):
    global s
    assert s.count(old) == 1, ("NOT UNIQUE/FOUND", old[:90], s.count(old))
    s = s.replace(old, new)


def cut(start, end, new):
    global s
    a = s.index(start)
    b = s.index(end, a)
    s = s[:a] + new + s[b:]


# ---- NLP objective ----
cut(r'''With evidence $e$ the ratio is reformulated with one auxiliary $t\in[0,1]$,''',
    r'''\paragraph{The program.}''',
    r'''With evidence $e$ the read-off $P(a\mid e)=\langle A_{a\wedge e},q\rangle/
\langle A_e,q\rangle$ is linear-fractional, and is made linear by the
Charnes--Cooper transformation \citep{charnes1962flp}: substitute
$y_C=r\,q_C$ for every cluster, with a scale $r\ge0$, and require
\begin{equation}\label{eq:nlp-obj-ev}
  \textstyle\sum_i y_{C,i}=r\quad\forall C,\qquad
  \bigl\langle A^{(C_a)}_{e},y_{C_a}\bigr\rangle=1,\qquad
  \readoff_a(y)=\bigl\langle A^{(C_a)}_{a=1}\!\cdot\!A^{(C_a)}_{e},\,y_{C_a}\bigr\rangle .
\end{equation}
The separator equalities, the cleared Type-2 rows and the assertion rows are
homogeneous in $q$ and are imposed on $y$ unchanged; the Type-1 rows become
$\ell\,r\le\langle A_\varphi,y\rangle\le u\,r$. A feasible $y$ corresponds to the
feasible $q=y/r$ with $P_q(e)=1/r>0$, and $\readoff_a(y)=P_q(a\mid e)$;
conversely every feasible $q$ with $P_q(e)>0$ yields such a $y$. The program is
therefore exact for the conditional query without any denominator floor, and it
is infeasible exactly when $P(e)=0$ for every feasible $q$, in which case the
conditional bound is undefined and is reported as vacuous.

''')
sub(r'''Collecting \eqref{eq:nlp-var}--\eqref{eq:nlp-obj-ev},''',
    r'''Collecting \eqref{eq:nlp-var}--\eqref{eq:nlp-obj}, for a marginal query,''')
sub(r'''&\text{\eqref{eq:nlp-lmc}} && \forall\text{ hosted assertion.}
\end{aligned}
\end{equation}''',
    r'''&\text{\eqref{eq:nlp-lmc}} && \forall\text{ hosted assertion.}
\end{aligned}
\end{equation}
For a conditional query the same constraints are imposed on $y$ as described
above, together with \eqref{eq:nlp-obj-ev}.''')

# ---- solving / complexity mentions ----
sub(r'''and under evidence so is the ratio equality \eqref{eq:nlp-obj-ev}. Exactness in''',
    r'''while the conditional read-off is linear after the scaling
\eqref{eq:nlp-obj-ev}. Exactness in''')
sub(r'''DAG, $|\mathcal C|\le m\le n$; under evidence add one auxiliary scalar per atom.''',
    r'''DAG, $|\mathcal C|\le m\le n$; under evidence add the single scale $r$.''')
sub(r'''(a) Only the assertion equalities \eqref{eq:nlp-lmc} and, under evidence, the
ratio equality \eqref{eq:nlp-obj-ev} are nonconvex.''',
    r'''(a) Only the assertion equalities \eqref{eq:nlp-lmc} are nonconvex; the
conditional objective is linear after the scaling \eqref{eq:nlp-obj-ev}.''')

# ---- theorem d5 ----
sub(r'''For a conditional query $P(a\mid e)$, solved with the denominator floor
\eqref{eq:nlp-floor}, the same equalities hold with $\DLCN$ replaced by
$\DLCN^{\varepsilon}=\{p\in\DLCN:P_p(e)\ge\varepsilon\}$; they are the exact
conditional bounds whenever $\min_{p\in\DLCN}P_p(e)\ge\varepsilon$
(Remark~\ref{rem:denfloor}).''',
    r'''For a conditional query $P(a\mid e)$, solved in the scaled form
\eqref{eq:nlp-obj-ev}, the same equalities hold for $\lo P(a\mid e)$ and
$\hi P(a\mid e)$ taken over $\DLCN^{+}=\{p\in\DLCN:P_p(e)>0\}$, the standard
definition of conditional lower and upper probabilities.''')
cut(r'''\emph{Conditional queries.} By H4 the read-off is a functional of the single''',
    r'''\end{proof}''',
    r'''\emph{Conditional queries.} By H4 the read-off is a functional of the single
cluster $C_a$, and both maps preserve $q_{C_a}$, hence $P(e)$. The two maps
therefore restrict to bijections-up-to-reconstruction between $\DLCN^{+}$ and
$\mathcal G^{+}=\{q\in\mathcal G:P_q(e)>0\}$, and the argument above applies to
these two sets. The scaling \eqref{eq:nlp-obj-ev} maps $\mathcal G^{+}$ onto the
feasible set of the scaled program, $q\mapsto(y,r)=(q/P_q(e),1/P_q(e))$, with
inverse $q=y/r$, under which $P_q(a\mid e)$ equals the linear read-off of $y$.
Every constraint of $\mathcal G$ is preserved: the homogeneous rows because they
are homogeneous, the simplex and Type-1 rows because both sides scale by $r$.
''')

# ---- remarks ----
sub(r'''H2 and the second half of H3 decide only tightness. For conditional queries this
holds as long as $\min_{p\in\DLCN}P_p(e)\ge\varepsilon$
(Remark~\ref{rem:denfloor}).''',
    r'''H2 and the second half of H3 decide only tightness; the same holds for
conditional queries, over $\DLCN^{+}$.''')
cut(r'''\begin{remark}[The denominator floor is a soundness hypothesis]\label{rem:denfloor}''',
    r'''\end{remark}''',
    r'''\begin{remark}[Why the conditional query is scaled]\label{rem:denfloor}
A natural alternative linearizes $P(a\mid e)$ by an auxiliary $t$ with
$t\,P(e)=P(a\wedge e)$ and a floor $P(e)\ge\varepsilon$ to exclude the
degenerate $P(e)=0$. The floor removes the members of $\DLCN$ with
$P(e)<\varepsilon$, and in floating point it fails in a worse way: once
$\varepsilon$ is near the solver's feasibility tolerance, $t\,P(e)=P(a\wedge e)$
holds within tolerance for every $t$. On the instance
$0\le P(E)\le0.3$, $0.2\le P(A\mid E)\le0.4$, $0.6\le P(A\mid\lnot E)\le0.8$
with $\varepsilon=10^{-6}$, SCIP certifies $P(A\mid E)\in[0,1]$, whereas the exact
bound is $[0.2,0.4]$. The scaled form \eqref{eq:nlp-obj-ev} has neither defect and
returns $[0.2,0.4]$.
''')

open(P, 'w').write(s)
print("ok")
