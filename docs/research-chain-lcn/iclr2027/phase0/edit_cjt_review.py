"""CJT section review: proof fixes + removal of notes about earlier versions."""
import sys

P = sys.argv[1]
s = open(P).read()


def sub(old, new):
    global s
    assert s.count(old) == 1, ("NOT UNIQUE/FOUND", old[:90], s.count(old))
    s = s.replace(old, new)


def cut(start, end, new, include_end=False):
    global s
    a = s.index(start)
    b = s.index(end, a)
    if include_end:
        b += len(end)
    s = s[:a] + new + s[b:]


# ---------------- B intro ----------------
sub(r'''\textsc{CJT} proceeds in four steps. It builds a cluster tree over the atoms
from the family scopes of the chain graph \eqref{eq:factor}; it assigns every
sentence and every non-redundant independence assertion to a cluster containing
its scope; it writes a single constraint system whose variables are one
distribution per cluster; and it reads the query off a cluster containing it,
minimizing and maximizing over that system.''',
    r'''\textsc{CJT} proceeds in four steps. It builds a cluster tree over the atoms
from the family scopes of the chain graph \eqref{eq:factor}; it assigns every
sentence to a cluster containing its atoms, and discharges every independence
assertion by separation, by an equivalent reduced assertion hosted on a cluster,
or by the rows of an ordered tree; it writes a single constraint system whose
variables are one distribution per cluster; and it reads the query off a cluster
containing it, minimizing and maximizing over that system.''')

sub(r'''so $w$ is the treewidth of the triangulated graph and $w+1$ the largest cluster
measured in atoms. Cluster sizes are always reported in atoms below.''',
    r'''so $w$ is the width of $T$ --- an upper bound on the treewidth of the
moralized chain graph --- and $w+1$ the largest cluster measured in atoms.
Cluster sizes are always reported in atoms below.''')

# ---------------- B.2: drop correction remark; neutral reduction example ----
cut(r'''\begin{remark}[A correction to the implementation]\label{rem:triagefix}''',
    r'''\begin{example}\label{ex:dropped}''', '')
sub(r'''\begin{example}\label{ex:dropped}
In \texttt{examples/cjt\_dropped\_lmc.lcn} the atoms $x_2,x_4$ are roots and
$s_4$ reads $0.096543\le P(x_3\mid x_2\wedge x_4)\le0.696543$. The LMC gives
$(x_2\indep x_0,x_1,x_4)$, whose four atoms exceed the largest cluster ($3$
atoms). In $G^\ast$ the atoms $x_0,x_1$ lie in a different connected component
from $x_2$, so the reduction yields the equivalent $(x_2\indep x_4)$, hosted on
$\{x_2,x_3,x_4\}$. The exact bound is then
$\lo P(x_3)=\lo P(x_2)\,\lo P(x_4)\cdot0.096543
=0.673737\cdot0.349412\cdot0.096543=0.022727$, which \textsc{CJT} and Credal VE
both return; without the assertion the program can correlate $x_2$ and $x_4$
negatively and reaches $0.096543\,(0.673737+0.349412-1)=0.0022$.
\end{example}''',
    r'''\begin{example}[Reduction]\label{ex:dropped}
Consider an LCN over $x_0,\dots,x_5$ in which $x_2$ and $x_4$ are roots, $x_3$
has parents $x_2,x_4$ with $0.096543\le P(x_3\mid x_2\wedge x_4)\le0.696543$, and
$x_0$, $x_1$ form a separate component. The LMC gives the assertion
$(x_2\indep x_0,x_1,x_4)$, whose four atoms exceed the largest cluster ($3$
atoms). In $G^\ast$ the atoms $x_0,x_1$ lie in a different connected component
from $x_2$, so the reduction yields the equivalent $(x_2\indep x_4)$, which is
hosted on $\{x_2,x_3,x_4\}$. With $0.673737\le P(x_2)$ and $0.349412\le P(x_4)$
the exact bound is
$\lo P(x_3)=0.673737\cdot0.349412\cdot0.096543=0.022727$; without the assertion
the program could correlate $x_2$ and $x_4$ negatively and would return only
$0.096543\,(0.673737+0.349412-1)=0.0022$.
\end{example}''')

# ---------------- B.3: fallback remark ----------------
sub(r'''so its output carries no global guarantee and can be over-tight. Conversely, on
instances where the $2^n$ formulation is solvable but its global solver fails to
certify some endpoint, the cluster program of Section~\ref{sec:nlp} is often the
more reliable route to a bound, even at $n$ small enough for the joint to be
written down.''',
    r'''so its output carries no global guarantee and can be over-tight. Conversely,
even at $n$ small enough for the joint to be written down, the cluster program is
often the easier one to certify: on the worked example of
Section~\ref{sec:cjt-example} it certifies all endpoints where the $2^n$
formulation certifies only seven of ten.''')

# ---------------- C.2: definition ----------------
sub(r'''Let $T$ be a clique tree over clusters $\mathcal C$, built by the standard
moralize~$\to$~triangulate~$\to$~clique-tree pipeline, and let $G^\ast$ be the
graph on the atoms in which two atoms are adjacent iff they co-occur in a
cluster. Every cluster is a clique of $G^\ast$, and every maximal clique of
$G^\ast$ is a cluster; a non-maximal cluster may remain after pruning
(Section~\ref{sec:algo-jt}), which is harmless below.''',
    r'''Let $T$ be the cluster tree built by bucket elimination
(Section~\ref{sec:algo-jt}) over clusters $\mathcal C$ with atom sets $A(C)$, and
let $G^\ast$ be the graph on the atoms in which two atoms are adjacent iff they
co-occur in a cluster; it is the triangulated moral graph of the chain graph,
augmented by the cross-family sentence scopes. Every cluster is a clique of
$G^\ast$ and every maximal clique of $G^\ast$ is a cluster; we identify a cluster
with its atom set.''')
sub(r'''    separates $X$ from $Y$ in $G^\ast$ (every path from $X$ to $Y$ meets $S$).
    Note that it is the \emph{conditioning set} $S$ that must separate: a tree
    separator strictly larger than $S$ does not suffice.''',
    r'''    separates $X$ from $Y$ in $G^\ast$ (every path from $X$ to $Y$ meets $S$).''')

# ---------------- C.2: lemma rip lead-in + proof ----------------
sub(r'''The reconstruction direction needs the following lemma, whose usual statement
contains two defects we must repair: it is normally asserted with a
\emph{uniqueness} claim that is false in the presence of zero separator cells,
and the division in the product formula is left unaddressed.''',
    r'''The reconstruction direction rests on the following lemma. We state and prove
it for distributions with zeros, where the joint is not unique and the familiar
quotient formula involves $0/0$.''')
sub(r'''Let $T'=T-C$; by the running-intersection property
$C\setminus S$ meets no cluster of $T'$, and $(q_{C'})_{C'\in T'}$ is still
separator-consistent, so the inductive hypothesis gives a joint $p'$ over''',
    r'''Let $T'=T-C$. Removing a leaf preserves the running-intersection property, the
remaining clusters are still separator-consistent, and by running intersection
$C\setminus S$ meets no cluster of $T'$; so the inductive hypothesis gives a
joint $p'$ over''')

# ---------------- C.2: ordered proposition ----------------
sub(r'''then $p=\prod_x p(x\mid\pa(x))$, so $p$ satisfies every $d$-separation of $D$
and in particular every LMC assertion of a chain-graph LCN whose LMC is the local
Markov property of $D$. Conversely every row \eqref{eq:ordered-row} is implied by
that local Markov property, since $N_x\setminus\pa(x)$ are non-descendants of $x$.''',
    r'''then $p=\prod_x p(x\mid\pa(x))$, so $p$ satisfies every $d$-separation of $D$.
Consequently, if every LMC assertion of the LCN is a $d$-separation of $D$
between unions of nodes (in particular, no assertion splits a compound node),
$p$ satisfies every LMC assertion. Conversely each row \eqref{eq:ordered-row}
follows from the local Markov property of $D$, since $N_x\setminus\pa(x)$
consists of non-descendants of $x$.''')
sub(r'''eliminated yet. The order is a perfect elimination order of $G^\ast_D$: when $x$
is eliminated it is simplicial in the graph induced on the remaining nodes, with
neighbourhood $N_x$.''',
    r'''eliminated yet. Since $G^\ast_D$ is the graph of the elimination cliques, the
order is a perfect elimination order of it: when $x$ is eliminated it is
simplicial in the graph induced on the remaining nodes, with neighbourhood
$N_x$. (Pruning a subsumed leaf does not change $G^\ast_D$, since the leaf's
clique lies inside its parent's.)''')
sub(r'''The implementation checks, per instance, that each row \eqref{eq:ordered-row} is
implied by a single LCN assertion (by symmetry and decomposition) and that every
LCN assertion is a $d$-separation of $D$; if either check fails, the ordered tree
is discarded and the min-fill triage is used. Both checks passed on every
benchmark instance of Section~\ref{sec-experiments}. They fail when an assertion
splits a compound node, as for the undirected component $F_1F_2F_3$ of the
\texttt{smokers} example; there the min-fill triage widens to a single
$9$-atom cluster, which is exact but no cheaper than the direct program.''',
    r'''The implementation checks, per instance, the two conditions that make
Proposition~\ref{prop:ordered} applicable: that each row \eqref{eq:ordered-row} is
implied by a single LCN assertion (by symmetry and decomposition), and that every
LCN assertion is a $d$-separation of $D$ between unions of nodes. If either fails
the ordered tree is discarded and the min-fill triage is used. They fail when an
assertion splits a compound node, as in the worked example of
Section~\ref{sec:cjt-example} and in the \texttt{smokers} example; for the latter
the min-fill triage widens to a single $9$-atom cluster, which is exact but no
cheaper than the direct program.''')

# ---------------- C.2: theorem d5 conditional queries ----------------
sub(r'''and likewise for a conditional query $P(a\mid e)$ subject to
Remark~\ref{rem:denfloor}. Moreover the \emph{projection} half holds without \textup{(H2)} and without''',
    r'''For a conditional query $P(a\mid e)$, solved with the denominator floor
\eqref{eq:nlp-floor}, the same equalities hold with $\DLCN$ replaced by
$\DLCN^{\varepsilon}=\{p\in\DLCN:P_p(e)\ge\varepsilon\}$; they are the exact
conditional bounds whenever $\min_{p\in\DLCN}P_p(e)\ge\varepsilon$
(Remark~\ref{rem:denfloor}). Moreover the \emph{projection} half holds without \textup{(H2)} and without''')
sub(r'''projection half, the optima coincide. For a conditional query, by H4 the read-off
is a functional of the single cluster $C_a$ and both maps preserve $q_{C_a}$, so
the same argument applies verbatim.''',
    r'''projection half, the optima coincide.

\emph{Conditional queries.} By H4 the read-off is a functional of the single
cluster $C_a$, and both maps preserve $q_{C_a}$, hence $P(e)$; the floor
\eqref{eq:nlp-floor} therefore restricts $\mathcal G$ exactly to the image of
$\DLCN^{\varepsilon}$, and the argument above applies to these two sets. On them
$P(e)\ge\varepsilon>0$, so the auxiliary equality \eqref{eq:nlp-obj-ev} gives
$t=P(a\mid e)$.''')

# ---------------- C.2: reading-the-proof remark ----------------
sub(r'''rows, or relaxed. With the triage of Section~\ref{sec:algo-host} it holds on every
example in \texttt{examples/} that is a chain graph and on all $556$ benchmark
instances; Remark~\ref{rem:triagefix} records the earlier failures.''',
    r'''rows, or relaxed.''')

open(P, 'w').write(s)
print("ok")
