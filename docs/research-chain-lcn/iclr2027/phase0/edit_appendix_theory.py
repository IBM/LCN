"""Exactness section edits for the CJT LMC-triage fix (one-off)."""
import sys

P = sys.argv[1]
s = open(P).read()


def sub(old, new, count=1):
    global s
    assert old in s, old[:80]
    s = s.replace(old, new, count)


# Definition: separator-entailment + imposed independence rows.
sub(r'''  \item An LMC assertion $(X\indep Y\mid S)$ is \emph{separator-entailed} if $S$
    separates $X$ from $Y$ in $G^\ast$.''',
    r'''  \item An LMC assertion $(X\indep Y\mid S)$ is \emph{separator-entailed} if $S$
    separates $X$ from $Y$ in $G^\ast$ (every path from $X$ to $Y$ meets $S$).
    Note that it is the \emph{conditioning set} $S$ that must separate: a tree
    separator strictly larger than $S$ does not suffice.
  \item The \emph{imposed independence rows} $\Lambda_T$ are the independence
    statements actually hosted on $T$: reduced LMC assertions
    (Lemma~\ref{lem:reduce}) on a min-fill tree, or ordered rows
    (Proposition~\ref{prop:ordered}) on an ordered tree. The LCN's own LMC set is
    denoted $\Lambda$.''')

# Restate the theorem hypotheses H2/H3.
sub(r'''Suppose \textup{(H1)} $T$ has the running-intersection property (true by
construction); \textup{(H2)} every LCN sentence is hosted; \textup{(H3)} every
LMC assertion is either hosted or separator-entailed in $G^\ast$; \textup{(H4)}
the query atom and all evidence atoms lie together in some cluster.''',
    r'''Suppose \textup{(H1)} $T$ has the running-intersection property (true by
construction); \textup{(H2)} every LCN sentence is hosted; \textup{(H3)} every
imposed row in $\Lambda_T$ is implied by $\Lambda$, and every assertion of
$\Lambda$ holds for every joint that factorizes over $G^\ast$ and satisfies
$\Lambda_T$; \textup{(H4)} the query atom and all evidence atoms lie together in
some cluster.''')

# Insert reduction lemma + ordered proposition before the theorem.
sub(r'''\begin{theorem}[\textsc{CJT} exactness]\label{thm:d5}''',
    r'''The two triages of Section~\ref{sec:algo-host} discharge H3 through the
following two results.

\begin{lemma}[Exact reduction of an assertion]\label{lem:reduce}
Let $p$ factorize over the cliques of $G^\ast$, let $(X\indep Y\mid Z)$ be an
assertion, let $K$ be the set of atoms reachable from $X$ in $G^\ast$ after
deleting $Z$, and let $Y_{\mathrm{in}}=Y\cap K$, $Y_{\mathrm{out}}=Y\setminus K$.
Then $X\indep Y\mid Z$ holds under $p$ if and only if
$X\indep Y_{\mathrm{in}}\mid Z$ does. In particular the assertion is
separator-entailed, and holds automatically, when $Y_{\mathrm{in}}=\varnothing$.
\end{lemma}

\begin{proof}
($\Rightarrow$) is decomposition. ($\Leftarrow$) By construction of $K$, every
path from $K$ to $Y_{\mathrm{out}}$ meets $Z$, so $Z$ separates $K\supseteq
X\cup Y_{\mathrm{in}}$ from $Y_{\mathrm{out}}$ in $G^\ast$. Factorization over an
undirected graph implies its global Markov property without any positivity
assumption, so $(X\cup Y_{\mathrm{in}})\indep Y_{\mathrm{out}}\mid Z$, and by weak
union $X\indep Y_{\mathrm{out}}\mid Z\cup Y_{\mathrm{in}}$. Contraction with
$X\indep Y_{\mathrm{in}}\mid Z$ yields $X\indep Y_{\mathrm{in}}\cup
Y_{\mathrm{out}}\mid Z$. Decomposition, weak union and contraction hold for every
probability distribution (they are the semi-graphoid axioms).
\end{proof}

\begin{proposition}[Ordered tree: H3 by construction]\label{prop:ordered}
Let $D$ be the node DAG of the chain-graph factorization \eqref{eq:factor}, let
$T$ be built by bucket elimination along a reverse topological order of $D$, and
for every node $x$ let $\{x\}\cup N_x$ be its elimination clique. Then
$N_x$ consists of topological predecessors of $x$ and contains $\pa(x)$. If
$p$ factorizes over $G^\ast$ and satisfies, for every node $x$,
\begin{equation}\label{eq:ordered-row}
  x\indep N_x\setminus\pa(x)\mid\pa(x),
\end{equation}
then $p=\prod_x p(x\mid\pa(x))$, so $p$ satisfies every $d$-separation of $D$
and in particular every LMC assertion of a chain-graph LCN whose LMC is the local
Markov property of $D$. Conversely every row \eqref{eq:ordered-row} is implied by
that local Markov property, since $N_x\setminus\pa(x)$ are non-descendants of $x$.
\end{proposition}

\begin{proof}
Bucket elimination along $x_n,\dots,x_1$ (reverse topological) removes $x$ only
after all its topological successors, so the neighbours of $x$ at that moment ---
$N_x$ --- are predecessors; $\pa(x)\subseteq N_x$ because the family scope
$\{x\}\cup\pa(x)$ is one of the covered scopes and none of $\pa(x)$ has been
eliminated yet. Reconstructing $p$ leaf by leaf as in Lemma~\ref{lem:rip}, in the
elimination order, writes $p=\prod_x p(x\mid N_x)$, where each conditional is
defined arbitrarily on $\{p(N_x)=0\}$ (where it is multiplied by zero). By
\eqref{eq:ordered-row}, $p(x\mid N_x)=p(x\mid\pa(x))$ wherever $p(N_x)>0$, hence
$p=\prod_x p(x\mid\pa(x))$. A distribution that factorizes along a DAG satisfies
its global (d-separation) Markov property, again without positivity. The
converse is the local Markov property of $D$ followed by decomposition.
\end{proof}

The implementation checks, per instance, that each row \eqref{eq:ordered-row} is
implied by a single LCN assertion (by symmetry and decomposition) and that every
LCN assertion is a $d$-separation of $D$; if either check fails, the ordered tree
is discarded and the min-fill triage is used. Both checks passed on every
benchmark instance of Section~\ref{sec-experiments}. They fail when an assertion
splits a compound node, as for the undirected component $F_1F_2F_3$ of the
\texttt{smokers} example; there the min-fill triage widens to a single
$9$-atom cluster, which is exact but no cheaper than the direct program.

\begin{theorem}[\textsc{CJT} exactness]\label{thm:d5}''')

# Projection half: imposed rows must hold for p in D_LCN.
sub(r'''Each hosted constraint $c$ holds: by
Lemma~\ref{lem:locality} its value depends on $p$ only through
$p\restr\scope(c)$, and since $\scope(c)\subseteq C_h$ we have
$q_{C_h}\restr\scope(c)=p\restr\scope(c)$, so $c$ holds on the host exactly
because it holds for $p$.''',
    r'''Each hosted constraint $c$ holds: by
Lemma~\ref{lem:locality} its value depends on $p$ only through
$p\restr\scope(c)$, and since $\scope(c)\subseteq C_h$ we have
$q_{C_h}\restr\scope(c)=p\restr\scope(c)$, so $c$ holds on the host exactly
because it holds for $p$. For a sentence this is immediate; for an imposed row
of $\Lambda_T$ it holds for $p$ because, by the first half of H3, it is implied by
an assertion of $\Lambda$, which $p\in\DLCN$ satisfies.''')

sub(r'''\emph{Projection $\DLCN\to\mathcal G$ (needs only H1, H4).}''',
    r'''\emph{Projection $\DLCN\to\mathcal G$ (needs H1, H4 and the first half of H3).}''')

# Reconstruction half: replace the separator-entailed paragraph.
a = s.index(r'''\emph{Separator-entailed LMC} (the other part of H3)''')
b = s.index(r'''Hence $p\in\DLCN$, and''', a)
s = s[:a] + r'''\emph{LMC assertions} (the second half of H3): $p$ factorizes over the cliques
of $G^\ast$ and, by the previous paragraph, satisfies every imposed row of
$\Lambda_T$; by H3 it therefore satisfies every assertion of $\Lambda$, i.e.\
\eqref{eq:lmc} holds for each. On a min-fill tree this is Lemma~\ref{lem:reduce}
(each assertion is equivalent to its imposed reduction, or separator-entailed);
on an ordered tree it is Proposition~\ref{prop:ordered}. Neither argument uses
positivity. Note that no assertion of $\Lambda$ need be imposed verbatim.

''' + s[b:]

# Outer-bound claim: now requires only the first half of H3.
sub(r'''Moreover the \emph{projection} half holds without
\textup{(H2)}/\textup{(H3)}, so \textsc{CJT} is always a valid \emph{outer}
bound.''',
    r'''Moreover the \emph{projection} half holds without \textup{(H2)} and without
the second half of \textup{(H3)}, so \textsc{CJT} is a valid \emph{outer} bound
whenever every imposed row is implied by $\Lambda$ --- which the relaxation step
of Section~\ref{sec:algo-host} preserves.''')
sub(r'''This is the outer-bound claim, and it used
neither H2 nor H3.''',
    r'''This is the outer-bound claim; it used neither H2 nor the second half of
H3.''')

# Remark (c).
a = s.index(r'''(c) H3 is \emph{sufficient, not necessary}''')
b = s.index(r'''\end{remark}''', a)
s = s[:a] + r'''(c) H3 is \emph{sufficient, not necessary}, and it can be checked
mechanically: the implementation records for each instance whether H3 holds, how
many assertions were separator-entailed, reduced and hosted, implied by ordered
rows, or relaxed. With the triage of Section~\ref{sec:algo-host} it holds on every
example in \texttt{examples/} that is a chain graph and on all $556$ benchmark
instances; Remark~\ref{rem:triagefix} records the earlier failures.
''' + s[b:]

open(P, 'w').write(s)
print("ok")
