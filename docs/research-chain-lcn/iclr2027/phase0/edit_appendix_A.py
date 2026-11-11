"""A1-A6 appendix edits: product set vs hull, positivity, no-Gap-A lemma,
CVE pruning lemma + Thm 2 proof, IBP Thm 4 proof (one-off)."""
import sys

P = sys.argv[1]
s = open(P).read()


def cut(head, new):
    """Replace from `head` up to (not incl.) the next '%% ====' line."""
    global s
    head = head.split("\n")[0]
    a = s.index(head)
    b = s.index("%% ====", a)
    s = s[:a] + new + "\n\n" + s[b:]


def sub(old, new):
    global s
    assert s.count(old) == 1, ("NOT UNIQUE/FOUND", old[:90], s.count(old))
    s = s.replace(old, new)


# ---- A1: the appendix set is the PRODUCT set; add hull + bound lemma ----
sub(r'''The credal-network based methods replace each factor by an interval local credal set
$K_{v,\pi}$ per parent configuration $\pi$ and propagate over the \emph{strong
extension}:
\begin{equation}\label{eq:se-app}
  \DSE=\Bigl\{\textstyle\prod_v P_v\;:\;P_v\in K_{v,\pi}
  \text{ chosen independently for each parent configuration }\pi\Bigr\}
  \;\supseteq\;\DLCN .
\end{equation}
The inclusion is generally strict, for two reasons that it is worth naming since
they are exactly what \textsc{CJT} avoids:''',
    r'''The credal-network based methods replace each factor by an interval local credal set
$K_{v,\pi}$ per parent configuration $\pi$ and work with the \emph{product set}
\begin{equation}\label{eq:se-app}
  \mathcal P_{\mathrm{SE}}=\Bigl\{\textstyle\prod_v P_v\;:\;P_v(\cdot\mid\pi)\in K_{v,\pi}
  \text{ chosen independently for each }v\text{ and parent configuration }\pi\Bigr\}
\end{equation}
and its convex hull, the \emph{strong extension} $\DSE=\mathrm{CH}(\mathcal
P_{\mathrm{SE}})$. Mixtures of products need not factorize, so $\DSE$ may contain
distributions violating the independences of the chain graph; but the two sets
have the same bounds.

\begin{lemma}[Hull invariance of bounds]\label{lem:hull}
Let $S$ be a compact set of distributions and $f(p)=\langle a,p\rangle/\langle
b,p\rangle$ with $\langle b,p\rangle>0$ on $\mathrm{CH}(S)$ (for a marginal,
$b=\mathbf 1$). Then $\min_{\mathrm{CH}(S)}f=\min_S f$ and
$\max_{\mathrm{CH}(S)}f=\max_S f$.
\end{lemma}

\begin{proof}
Let $p=\sum_k\lambda_k p_k$ be a convex combination of points of $S$. Then
$f(p)=\sum_k\mu_k f(p_k)$ with
$\mu_k=\lambda_k\langle b,p_k\rangle/\sum_j\lambda_j\langle b,p_j\rangle\ge0$
summing to $1$, so $f(p)$ lies between $\min_kf(p_k)$ and $\max_kf(p_k)$. The
extrema over $S$ exist by compactness and continuity.
\end{proof}

Every statement below about \emph{sets} is made for $\mathcal P_{\mathrm{SE}}$;
by Lemma~\ref{lem:hull} it transfers to the \emph{bounds} of $\DSE$. The
inclusion $\DLCN\subseteq\mathcal P_{\mathrm{SE}}$ is generally strict, for two
reasons that it is worth naming since they are exactly what \textsc{CJT} avoids:''')

sub(r'''    the free product in \eqref{eq:se-app} lets each family pick its row''',
    r'''    the free product in \eqref{eq:se-app} lets each family pick its row''') if False else None

# ---- A1/A5: inclusion proposition and its proof ----
sub(r'''\begin{proposition}[Inclusion]\label{prop:incl}
Let $\mathcal L$ be a chain-graph LCN. Then $\DLCN\subseteq\DSE$. Consequently, for every query,''',
    r'''Call a chain-graph LCN \emph{factorizing} if every $p\in\DLCN$ satisfies
\eqref{eq:factor} with the conditionals induced by $p$. This holds for every
DAG-structured LCN (all chain components are single atoms), because the local
Markov property of a DAG implies its factorization for every distribution
(chain rule along a topological order, then the local independences). It holds
for general chain graphs if $\DLCN$ is restricted to strictly positive
distributions \citep{cozman23a}. Positivity cannot simply be dropped for
multi-atom components: the atom-wise LMC assertions
$(C\indep\mathbf Y\mid D,\mathbf Z)$, $(D\indep\mathbf Y\mid C,\mathbf Z)$ of a
component $\{C,D\}$ give the component-wise $(\{C,D\}\indep\mathbf Y\mid\mathbf Z)$
only through the intersection property, which may fail when $p$ has zeros.

\begin{proposition}[Inclusion]\label{prop:incl}
Let $\mathcal L$ be a factorizing chain-graph LCN. Then $\DLCN\subseteq\mathcal P_{\mathrm{SE}}\subseteq\DSE$. Consequently, for every query,''')

sub(r'''Let $p\in\DLCN$. Since $\mathcal L$ is a chain graph and $p$ satisfies every LMC equality, the
chain-graph factorization theorem \cite{cozman23a} gives the factorization \eqref{eq:factor}:''',
    r'''Let $p\in\DLCN$. Since $\mathcal L$ is factorizing, $p$ satisfies \eqref{eq:factor}:''')

sub(r'''$K_{v,\pi}$, which is precisely membership in $\DSE$ as defined in \eqref{eq:se-app}. The bound
inclusion follows because minimizing and maximizing a fixed functional over a larger set yields a
wider interval.''',
    r'''$K_{v,\pi}$, which is precisely membership in $\mathcal P_{\mathrm{SE}}$ as defined in
\eqref{eq:se-app}; and $\mathcal P_{\mathrm{SE}}\subseteq\mathrm{CH}(\mathcal P_{\mathrm{SE}})=\DSE$.
The bound inclusion follows because minimizing and maximizing a fixed functional over a larger set
yields a wider interval.''')

# ---- A4: no-Gap-A lemma; sufficiency without (A) ----
sub(r'''\begin{proposition}[Sufficiency]\label{prop:suff-app}
Let $\mathcal L$ be a chain-graph LCN with local sets built by chain-graph factorization. Suppose
\begin{enumerate}
  \item[\textup{(A)}] \emph{no Gap~A:} for every family $v$ and parent configuration $\pi$,
    $K_{v,\pi}=K_{v,\pi}^{\mathrm{true}}$, the projection of $\DLCN$ onto that row; and
  \item[\textup{(B)}] \emph{full realizability:} every sentence of $\mathcal L$ is family-realizable
    in some family (Definition~\ref{def:realizable-app}).
\end{enumerate}
Then $\DSE=\DLCN$. \emph{Single-connectedness of the primal graph is not required.}
\end{proposition}''',
    r'''The sufficiency result needs every local row to be realized inside $\DLCN$
(``no Gap~A''). Under full realizability this is not an extra assumption.

\begin{lemma}[Realizability closes Gap~A]\label{lem:noGapA}
Let $\mathcal L$ be a consistent, factorizing chain-graph LCN in which every
sentence is family-realizable in some family. Then for every family $v$ and
parent configuration $\pi$, $K_{v,\pi}=K^{\mathrm{true}}_{v,\pi}$, the projection
of $\DLCN$ onto the row $P(\ch_v\mid\pa_v{=}\pi)$.
\end{lemma}

\begin{proof}
$K^{\mathrm{true}}_{v,\pi}\subseteq K_{v,\pi}$ is the inclusion argument of
Proposition~\ref{prop:incl}. Conversely, let $r\in K_{v,\pi}$. By consistency
pick $p^0\in\DLCN$, and let $q$ be the product of the conditional rows of $p^0$
with the single row $(v,\pi)$ replaced by $r$. We show $q\in\DLCN$; then $r$ is
the $(v,\pi)$ row of a member of $\DLCN$ (on $\{q(\pa_v{=}\pi)=0\}$ the row is
immaterial, and the cleared sentences are vacuous there), so
$r\in K^{\mathrm{true}}_{v,\pi}$.

\emph{Sentences.} By realizability each sentence $s$ is a function of one row of
one family (forms R1/R2). If that row is not $(v,\pi)$, $q$ and $p^0$ share it, so
$g_s(q)=g_s(p^0)\in[\ell_s,u_s]$. If it is $(v,\pi)$, then $\atoms(s)\subseteq
\ch_v\cup\pa_v$, so $s$ is among the in-scope sentences of the local program
defining $K_{v,\pi}$. That program ranges over joints on the family scope; by
R1/R2 the value of $s$ on any such joint is a function of its $(v,\pi)$ row
alone, so every row in the projection $K_{v,\pi}$ --- in particular $r$ ---
satisfies $s$. The other in-scope sentences of that program are realizable in
\emph{other} families and constrain only rows of those families (e.g.\ a root
marginal of a parent); they restrict the local program's parent marginal, not
its $(v,\pi)$ row, and are satisfied at $q$ because $q$ inherits those rows from
$p^0$.

\emph{LMC.} $q$ is a product of the form \eqref{eq:factor}; factorization implies
the global Markov property of the chain graph without positivity, so $q$
satisfies every LMC assertion. Hence $q\in\DLCN$.
\end{proof}

\begin{proposition}[Sufficiency]\label{prop:suff-app}
Let $\mathcal L$ be a consistent, factorizing chain-graph LCN with local sets built by chain-graph
factorization, in which every sentence is family-realizable in some family
(Definition~\ref{def:realizable-app}). Then $\mathcal P_{\mathrm{SE}}=\DLCN$, and by
Lemma~\ref{lem:hull} $\DSE$ and $\DLCN$ have the same lower and upper probabilities.
\emph{Single-connectedness of the primal graph is not required.}
\end{proposition}''')

sub(r'''Proposition~\ref{prop:incl} gives $\DLCN\subseteq\DSE$, so only $\DSE\subseteq\DLCN$ needs proof.
Let $q=\prod_v\prod_\pi P_{v,\pi}$ with each $P_{v,\pi}\in K_{v,\pi}$ be an arbitrary element of
$\DSE$.''',
    r'''Proposition~\ref{prop:incl} gives $\DLCN\subseteq\mathcal P_{\mathrm{SE}}$, so only
$\mathcal P_{\mathrm{SE}}\subseteq\DLCN$ needs proof. Let $q=\prod_v\prod_\pi P_{v,\pi}$ with each
$P_{v,\pi}\in K_{v,\pi}$ be an arbitrary element of $\mathcal P_{\mathrm{SE}}$.''')

sub(r'''$v$ alone. That table is the collection of rows $\{P_{v,\pi}\}_\pi$, and by (A) each''',
    r'''$v$ alone. That table is the collection of rows $\{P_{v,\pi}\}_\pi$, and by Lemma~\ref{lem:noGapA} each''')
sub(r'''row certified by (A) to be realized inside $\DLCN$ carries its constraint value with it --- so the''',
    r'''row realized inside $\DLCN$ (Lemma~\ref{lem:noGapA}) carries its constraint value with it --- so the''')

# ---- A2: CVE pruning lemma + Theorem 2 proof ----
cut(r'''\subsection{Proofs of Theorem 1 and 2}''',
    r'''\subsection{Proofs of Theorem 1 and 2}

The complexity bound of Theorem 1 follows from \cite{maua2020ijar}: each bucket
combines at most $C$ functions over at most $w^*+1$ variables of domain size at
most $d$, and there are $m$ buckets.

The exactness of CVE rests on the following property of the potentials it
produces without evidence.

\begin{lemma}[Evidence-free potentials are conditional tables]\label{lem:cvecond}
Run bucket elimination on the credal network with no evidence, along any
elimination order, starting from the potentials
$\phi_X=\{P(X\mid\mathbf{pa}(X))\}$ whose functions are the vertex choices of $X$'s
local credal sets. For every potential $\phi$ created during elimination there is a
partition of its scope into a \emph{child block} $C_\phi$ and a \emph{parent block}
$R_\phi$, determined by the elimination history alone, such that every $f\in\phi$
satisfies $\sum_{\mathbf c}f(\mathbf c,\mathbf r)=1$ for every assignment $\mathbf r$
of $R_\phi$. Consequently no function of $\phi$ is strictly dominated by another, and
$prune(\phi)$ removes only exact duplicates.
\end{lemma}

\begin{proof}
Call a function \emph{conditional} on $(C,R)$ if it sums to $1$ over $C$ for each
assignment of $R$, and assign to every potential the set $\mathcal A_\phi$ of
variables whose own CPT has been multiplied into it; let $C_\phi=\mathcal
A_\phi\cap\mathrm{scope}(\phi)$ and $R_\phi=\mathrm{scope}(\phi)\setminus C_\phi$.
Initially $\phi_X$ is conditional on $(\{X\},\mathbf{pa}(X))$.

\emph{Product.} If $f$ is conditional on $(C_1,R_1)$ and $g$ on $(C_2,R_2)$ with
$C_1\cap C_2=\varnothing$ --- each CPT enters exactly one bucket, so the
$\mathcal A$ sets of two potentials in a bucket are disjoint --- and no variable of
$C_1$ is a descendant of a variable of $C_2$ through $R_1$ and vice versa (both are
read off one DAG, so the combined child set is ancestrally ordered), then $fg$ is
conditional on $(C_1\cup C_2,(R_1\cup R_2)\setminus(C_1\cup C_2))$: summing first
over the $C$-block that is not a parent of the other, then over the other, gives $1$.

\emph{Elimination.} When the bucket of $v$ is processed, every potential mentioning
$v$ is in it, including the one carrying $v$'s own CPT, so $v\in C$ of the bucket
product. Summing out a child-block variable preserves conditionality, with child
block $C\setminus\{v\}$.

By induction every created potential is conditional on $(C_\phi,R_\phi)$, the same
split for all its functions. If $g\ge f$ pointwise with $g\ne f$, pick $\mathbf r$
with $g(\cdot,\mathbf r)\ne f(\cdot,\mathbf r)$; then
$\sum_{\mathbf c}g(\mathbf c,\mathbf r)>\sum_{\mathbf c}f(\mathbf c,\mathbf r)=1$,
a contradiction.
\end{proof}

With evidence the indicator potentials are not conditional tables, the lemma
fails, and dominance pruning can discard a function that attains the extremum of
the \emph{normalized} posterior. We observed this directly: comparing CVE with
pruning against CVE without pruning on $20$ small tree and polytree instances, the
intervals agree in all $20$ evidence-free cases and differ in $28$ of $40$ cases
with a single evidence atom, by up to $0.90$, the pruned interval being strictly
inside the exact one.

\begin{proof}[Proof of Theorem 2]
Without evidence the last bucket contains functions over the query $Z$ alone, each
a distribution over $Z$ (Lemma~\ref{lem:cvecond} with $R=\varnothing$), and by the
lemma the set of these functions is exactly the set of marginals on $Z$ of the
vertex products of $\mathcal P_{\mathrm{SE}}$ (pruning removed only duplicates;
combination and summation commute with the choice of vertices). Since
$P(Z{=}z)$ is multilinear in the local conditionals, its extrema over
$\mathcal P_{\mathrm{SE}}$ are attained at vertex products, so the componentwise
minimum and maximum are the bounds over $\mathcal P_{\mathrm{SE}}$, hence over
$\DSE$ by Lemma~\ref{lem:hull}. By Proposition~\ref{prop:suff-app}, which needs
consistency, factorization and full realizability, these equal the bounds over
$\DLCN$. For a compound query node, the singleton bounds are read off by the
linear programs of the previous subsection over the exact compound marginal
set, which preserves exactness.
\end{proof}''')

# ---- A3: IBP Theorem 4 proof ----
cut(r'''\subsection{Proofs of Theorem 3 and 4}''',
    r'''\subsection{Proofs of Theorem 3 and 4}

The complexity bound of Theorem 3 follows from the message computation: each
factor-to-child message enumerates at most $V_{\max}$ vertex combinations times
$2^{S_{\max}}$ corner combinations, each costing $O(k^{S_{\max}})$ to accumulate.

\begin{proof}[Proof of Theorem 4]
Let the structure be a polytree over binary atoms, with no evidence. Write
$I_X=[\lo P(X{=}1),\hi P(X{=}1)]$ for the exact interval over
$\mathcal P_{\mathrm{SE}}$. We show by induction along a topological order that,
at convergence, $\mu_{f_X\to X}$ encodes $I_X$ and every message
$\mu_{X\to f}$ into a child factor $f$ encodes $I_X$ too.

\emph{Messages into a child factor.} Upward factor-to-parent messages are vacuous,
so the variable-to-factor message from $X$ into a child factor intersects only
$\mu_{f_X\to X}$ with vacuous intervals and equals it.

\emph{Message from a family factor.} Let $X$ have parents $W_1,\dots,W_k$. In a
polytree, distinct parents have disjoint ancestral sets, so in every member of
$\mathcal P_{\mathrm{SE}}$ they are mutually independent, and the parameters
(local rows) that determine $P(W_j{=}1)$ are disjoint across $j$ and disjoint from
the rows of $X$. Hence
\[
  P(X{=}1)=\sum_{\pi}P(X{=}1\mid\pi)\prod_{j}P(W_j{=}\pi_j),
\]
where the rows $P(X{=}1\mid\pi)$ and the parent marginals $P(W_j{=}1)$ range
\emph{independently}: the rows over $K_{X,\pi}$, and each $P(W_j{=}1)$ over the
set of values it attains in $\mathcal P_{\mathrm{SE}}$, which is the interval
$I_{W_j}$ (it is the image of a connected set under a continuous map, and its
endpoints are $\lo P,\hi P$). The expression is linear in each row and in each
$P(W_j{=}1)$ separately, so its extrema over this product of convex sets are
attained at a vertex of each row's credal set and at an endpoint of each
$I_{W_j}$. Those are exactly the vertex and corner combinations enumerated by
\eqref{eq:compact-ftov}; without evidence the normalizing sum is $1$. By induction
the incoming parent intervals equal $I_{W_j}$, so $\mu_{f_X\to X}$ encodes $I_X$.

\emph{Extraction.} Only $f_X$ sends a non-vacuous message to $X$, so
\eqref{eq:compact-extract} returns $I_X$. A topological sweep reaches this fixed
point after a number of iterations equal to the depth of the polytree.

Thus IBP returns the exact bounds over $\mathcal P_{\mathrm{SE}}$, hence over
$\DSE$ (Lemma~\ref{lem:hull}). A polytree over binary atoms is DAG-structured and
so factorizing; by Proposition~\ref{prop:suff-app} these are the bounds over
$\DLCN$.
\end{proof}

Each hypothesis is used. With multi-valued variables the attainable set of a
parent's marginal is a polytope, not an interval, and its vertices are not the
two normalized endpoint vectors the message carries --- and exact polytree
inference is NP-hard already for ternary variables \citep{maua2014jair}. On a
loopy structure the parents of a node may be dependent and their marginals cannot
be chosen independently. With evidence on a descendant the correct posterior
requires upward (diagnostic) messages, which IBP does not send: on the two-node
network $A\to B$ with $P(A){=}0.5$, $P(B\mid A){=}0.9$, $P(B\mid\lnot A){=}0.1$
and evidence $B{=}1$, IBP returns $P(A{=}1)=0.5$ whereas the exact posterior is
$0.9$.''')

open(P, 'w').write(s)
print("ok")
