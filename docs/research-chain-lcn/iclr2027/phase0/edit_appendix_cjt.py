"""Rewrite the CJT appendix sections for the LMC-triage fix (one-off edit)."""
import sys

P = sys.argv[1]
s = open(P).read()


def replace_between(s, start, end, new, include_end=False):
    a = s.index(start)
    b = s.index(end, a)
    if include_end:
        b += len(end)
    return s[:a] + new + s[b:]


# ---------- Algorithm box ----------
NEW_ALG = r'''\begin{algorithm}[t]
\caption{\textsc{CJT}: marginal bounds at treewidth cost}
\label{alg:cjt}
\begin{algorithmic}[1]
\Require sentences $\Sigma$, LMC assertions $\Lambda$, node DAG $D$ with family
  scopes $\mathcal F$, evidence $e$, width budget $b$
\Ensure $[\lo P(a\mid e),\ \hi P(a\mid e)]$ for every atom $a\notin\atoms(e)$
\State $\mathcal S\gets\mathcal F\cup\{\atoms(s):s\in\Sigma\text{ cross-family}\}$
\State $T_o\gets\textsc{BucketTree}(\mathcal S,\ \text{reverse topological order of }D)$;
  \ $\Lambda_o\gets\textsc{Ordered}(T_o)$ \Comment{Prop.~\ref{prop:ordered}}
\State $T_m\gets\textsc{BucketTree}(\mathcal S,\textsc{MinFillOrder}(\mathcal S))$;
  \ $(T_m,\Lambda_m)\gets\textsc{Triage}(\Lambda,T_m,b)$ \Comment{\S\ref{sec:algo-host}}
\State $(T,\Lambda_T)\gets$ narrowest candidate on which H3 holds
  \Comment{RIP holds \hfill [H1]}
\ForAll{$s\in\Sigma$}
  \Comment{every sentence must be hosted \hfill [H2]}
  \State host $s$ on a cluster $C$ with $\atoms(s)\subseteq A(C)$, if one exists
\EndFor
\If{$\max_C|A(C)|>b$ \textbf{or} some $s\in\Sigma$ was not hosted}
  \State \Return $\textsc{Direct}(a,e)$ for every atom $a\notin\atoms(e)$
    \Comment{$2^n$, uncertified; Rem.~\ref{rem:oracle}}
\EndIf
\State build $\mathcal G$: \ $q_C\in\Delta(2^{|A(C)|})$,
  \ $q_C\restr S_{CD}=q_D\restr S_{CD}$, \ \eqref{eq:nlp-t1}--\eqref{eq:nlp-t2}
  on each host, \eqref{eq:nlp-lmc} for $\Lambda_T$ \Comment{once \hfill [H3]}
\ForAll{atoms $a\notin\atoms(e)$}
  \If{$\{a\}\cup\atoms(e)\subseteq A(C_a)$ for some $C_a$}
    \Comment{objective only \hfill [H4]}
    \State $\lo P\gets\min_{q\in\mathcal G}\readoff_a(q)$, \
      $\hi P\gets\max_{q\in\mathcal G}\readoff_a(q)$
      \Comment{certified or dual bound, Rem.~\ref{rem:dual}}
  \Else
    \State $[\lo P,\hi P]\gets\textsc{Direct}(a,e)$
  \EndIf
  \State clamp $[\lo P,\hi P]$ to $[0,1]$
\EndFor
\end{algorithmic}
\end{algorithm}'''
a = s.index(r'\begin{algorithm}[t]' + '\n' + r'\caption{\textsc{CJT}: marginal bounds at treewidth cost}')
b = s.index(r'\end{algorithm}', a) + len(r'\end{algorithm}')
s = s[:a] + NEW_ALG + s[b:]

# ---------- Constructing the cluster tree ----------
s = replace_between(
    s,
    r'Independence assertions are deliberately \emph{not} used to augment.',
    r'Each cluster' + "'" + r's atom set $A(C)$ is the union',
    r'''Independence assertions are \emph{not} used to augment wholesale: most are
entailed by the tree's topology (Section~\ref{sec:algo-host}), and adding their
full scopes as cliques would inflate the width for no gain --- a Markov chain
would collapse from width~$2$ to a single cluster containing all $n$ atoms.

Two elimination orders are tried. The \emph{min-fill} order is computed greedily
on the interaction graph --- the graph in which two nodes are adjacent when they
co-occur in some scope --- eliminating at each step a vertex whose elimination
adds fewest fill-in edges. The \emph{ordered} order eliminates the nodes of the
chain-graph node DAG $D$ in \emph{reverse topological} order; on the resulting
tree every LMC assertion is discharged by one hosted row per node, with no
widening (Proposition~\ref{prop:ordered}). The algorithm keeps the narrower of
the trees on which H3 holds. For either order, buckets are formed along the
order, each scope placed in the bucket of the first order variable it contains;
a bucket's cluster is the union of its scopes, and its parent is the first later
order variable appearing in the cluster after the bucket variable is removed.
This is standard bucket elimination, so \textbf{the resulting tree has the
running-intersection property by construction}, discharging hypothesis~H1 of
Theorem~\ref{thm:d5} without further argument.

''')

# ---------- Hosting and triage ----------
s = replace_between(
    s,
    r'Independence assertions admit a three-way triage',
    r'\subsection{Solving}\label{sec:algo-solve}',
    r'''Independence assertions are discharged in one of two ways, depending on the
tree.

\paragraph{Min-fill tree: reduce, host, widen, relax.}
For an assertion $(X\indep Y\mid Z)$ let $K$ be the set of atoms reachable from
$X$ in $G^\ast$ after deleting $Z$, and $Y_{\mathrm{in}}=Y\cap K$.
\begin{enumerate}
  \item \textbf{Reduce.} Replace the assertion by $(X\indep Y_{\mathrm{in}}\mid Z)$.
    By Lemma~\ref{lem:reduce} the two are \emph{equivalent} for every joint the
    cluster program can reconstruct, so nothing is lost. If
    $Y_{\mathrm{in}}=\varnothing$ the assertion is \emph{separator-entailed} ---
    $Z$ separates $X$ from $Y$ in $G^\ast$ --- and nothing is imposed; this is
    what keeps a chain at width~$2$.
  \item \textbf{Host.} If $X\cup Y_{\mathrm{in}}\cup Z\subseteq A(C)$ for some
    cluster $C$, impose the reduced assertion there as the bilinear equalities
    \eqref{eq:lmc}.
  \item \textbf{Widen.} Otherwise add $X\cup Y_{\mathrm{in}}\cup Z$ as an
    augmenting clique and rebuild the tree; repeat to a fixed point while the
    largest cluster stays within the budget $b$.
  \item \textbf{Relax.} An assertion still unhostable is replaced by its hostable
    consequences $(X\indep y\mid Z)$, $y\in Y_{\mathrm{in}}$. Each is implied by
    the original (decomposition), so this is sound; but H3 fails and the bound is
    a valid outer bound that may be loose.
\end{enumerate}

\paragraph{Ordered tree: one row per node.}
For every node $x$ with elimination clique $\{x\}\cup N_x$, impose the single
assertion $(x\indep N_x\setminus\pa(x)\mid\pa(x))$ on $x$'s cluster. Its scope is
that cluster, so it is always hosted; it is checked to follow from an LCN
assertion by symmetry and decomposition (soundness); and by
Proposition~\ref{prop:ordered} these rows imply \emph{every} LMC assertion, so H3
holds by construction. A compound child $x=(x_1,\dots,x_k)$ is imposed through the
equivalent chain $x_i\indep N_x\setminus\pa(x)\mid\pa(x),x_{<i}$, $i=1,\dots,k$
(contraction one way, weak union and decomposition the other; no positivity).

\begin{remark}[A correction to the implementation]\label{rem:triagefix}
An earlier implementation departed from this description in two ways, and the
reference bounds of a previous version of this paper were computed with it.
(i) It \emph{dropped} every assertion that had no host, instead of reducing,
widening or relaxing it. On the instance of Example~\ref{ex:dropped} this loses
$x_2\indep x_4$ and returns $\lo P(x_3)=0.0022$ instead of the exact $0.0227$.
(ii) It treated an assertion as separator-entailed whenever some tree separator
$S\supseteq Z$ split $X$ from $Y$; but a separator enforces only
$X\indep Y\mid S$, which does not imply $X\indep Y\mid Z$ when $S\supsetneq Z$.
Both errors only removed constraints, so the old bounds were valid outer bounds;
but H3 failed silently on most instances. Case~(i) occurred on $40/60$
\texttt{tree\_small}, $48/60$ \texttt{polytree\_small}, $20/60$
\texttt{ktree\_small}, $56/60$ \texttt{dag\_small}, $9/10$ Bayesian-network and
$7/16$ Supreme Court instances, and case~(ii) on $14$ further instances. With
the triage above H3 holds on all $556$ instances of
Section~\ref{sec-experiments}, at a mean width increase of at most one atom per
benchmark set.
\end{remark}

\begin{example}\label{ex:dropped}
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
\end{example}

''')

# ---------- Global solution / certification ----------
s = replace_between(
    s,
    r'\paragraph{Global solution.}',
    r'\begin{remark}[A local solver yields an inner bound]',
    r'''\paragraph{Global solution and certification.}
The system is nonconvex: the assertion equalities \eqref{eq:nlp-lmc} are bilinear,
and under evidence so is the ratio equality \eqref{eq:nlp-obj-ev}. Exactness in
Theorem~\ref{thm:d5} refers to the global optimum, so each endpoint is computed by
a spatial branch-and-bound solver (SCIP) and recorded with its certificate.

\begin{remark}[Uncertified endpoints are reported by their dual bound]\label{rem:dual}
When branch-and-bound stops before closing the gap (e.g.\ at its time limit),
the incumbent of a minimization lies \emph{above} the true minimum, so reporting
it would make the interval an unsound inner bound. \textsc{CJT} reports the
solver's \emph{dual} bound instead, which is $\le$ the true minimum (resp.\ $\ge$
the true maximum). An endpoint is \emph{certified} when the solver proves
optimality or its primal and dual bounds agree to within $10^{-4}$; the reported
value is the dual bound in either case. Every reported interval is therefore an
outer bound, and it is exact when H1--H4 hold and both endpoints are certified.
\end{remark}

''')

open(P, 'w').write(s)
print("ok")
