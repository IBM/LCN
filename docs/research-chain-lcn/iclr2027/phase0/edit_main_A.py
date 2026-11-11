"""A1-A7 main-text edits (one-off)."""
import sys

P = sys.argv[1]
s = open(P).read()


def sub(old, new):
    global s
    assert s.count(old) == 1, ("NOT UNIQUE/FOUND", old[:90], s.count(old))
    s = s.replace(old, new)


# ---------------- A5: positivity / factorization ----------------
sub(r'''Given a chain graph $G$ over variables $\bX$, it can be shown that for strictly positive probabilities the local and global Markov conditions are equivalent \citep{cowell99,cozman23a}. The implication is that the joint distribution $P(\bX)$ factorizes according to $G$ as:''',
    r'''Given a chain graph $G$ over variables $\bX$, it can be shown that for strictly positive probabilities the local and global Markov conditions are equivalent \citep{cowell99,cozman23a}. The implication is that a strictly positive joint distribution $P(\bX)$ satisfying LMC(C) factorizes according to $G$ as:''')
sub(r'''Consequently, for an LCN whose structure is a chain graph, the local Markov condition for LCNs (Definition~\ref{def:lmc-lcn}) coincides with Definition~\ref{def:lmc-cg} and every distribution in $\DLCN$ admits the factorization in \eqref{eq:factorization} \citep{cozman23a}.''',
    r'''Consequently, for an LCN whose structure is a chain graph, the local Markov condition for LCNs (Definition~\ref{def:lmc-lcn}) coincides with Definition~\ref{def:lmc-cg} and every strictly positive distribution in $\DLCN$ admits the factorization in \eqref{eq:factorization} \citep{cozman23a}. When every chain component is a single atom (the structure is a DAG) positivity is not needed: the local Markov property of a DAG implies its factorization for every distribution. For components with several atoms it is: the atom-wise assertions $(C\indep \bY\mid D,\bZ)$ and $(D\indep \bY\mid C,\bZ)$ imply the component-wise $(\{C,D\}\indep\bY\mid\bZ)$ only by the intersection property, which requires positivity. Throughout, the results that rely on \eqref{eq:factorization} are stated for \emph{factorizing} chain-graph LCNs: those that are DAG-structured, or whose $\DLCN$ is restricted to strictly positive distributions.''')

# ---------------- A1: strong extension = hull of the product set -------------
sub(r'''The \emph{strong extension} $\DSE$ of a credal network is the \emph{convex hull} (denoted CH) of all joint distributions that satisfy the following Markov property: every variable is strongly independent of its non-descendants conditional on its parents \citep{cozman2000credal,maua2020ijar}.
\begin{equation}\label{eq:se}
    \DSE=\textrm{CH}\{P(\bX)~:~ P(\bX)=\prod_{i=1}^{n} P(X_i\mid \pi_{ik}),~ P(X_i\mid \pi_{ik}) \textrm{~is a vertex of~} K(X_i\mid \pi_{ik})\}
\end{equation}''',
    r'''Let $\mathcal P_{\mathrm{SE}}$ denote the set of joint distributions that factorize along $G$ with every local conditional chosen, separately for each variable and parent configuration, from its credal set:
\begin{equation}\label{eq:pse}
    \mathcal P_{\mathrm{SE}}=\Bigl\{P(\bX)~:~ P(\bX)=\prod_{i=1}^{n} P(X_i\mid \mathbf{pa}(X_i)),~ P(X_i\mid \pi_{ik}) \in K(X_i\mid \pi_{ik})\ \forall i,k\Bigr\}.
\end{equation}
The \emph{strong extension} of the credal network is its convex hull \citep{cozman2000credal,maua2020ijar},
\begin{equation}\label{eq:se}
    \DSE=\textrm{CH}\bigl(\mathcal P_{\mathrm{SE}}\bigr),
\end{equation}
whose extreme points are products of vertices of the local credal sets. $\mathcal P_{\mathrm{SE}}$ is in general not convex, and $\DSE$ in general contains mixtures that violate the independences of $G$. The two sets nevertheless have the same lower and upper probabilities: a marginal $P(a)$ is linear in $P$, and a conditional $P(a\mid\mathbf e)$ is linear-fractional with positive denominator, and the extrema of such a function over a compact set and over its convex hull coincide (Lemma~\ref{lem:hull} in the Appendix). We therefore state set-level results for $\mathcal P_{\mathrm{SE}}$ and bound-level results for $\DSE$.''')

# ---------------- A1: inclusion + sufficiency ----------------
sub(r'''\begin{proposition}[inclusion]\label{prop:inclusion}
  For any chain-graph LCN, $\DLCN \subseteq \DSE$. Consequently, any inference algorithm that is \emph{exact} or \emph{outer} for $\DSE$ computes valid \emph{outer bounds} for $\DLCN$.
\end{proposition}''',
    r'''\begin{proposition}[inclusion]\label{prop:inclusion}
  For any factorizing chain-graph LCN, $\DLCN \subseteq \mathcal P_{\mathrm{SE}}\subseteq\DSE$. Consequently, any inference algorithm that is \emph{exact} or \emph{outer} for $\DSE$ computes valid \emph{outer bounds} for $\DLCN$.
\end{proposition}''')

sub(r'''Second, even with $K_v = K_v^{\mathrm{true}}$, the free product of \eqref{eq:se} lets each family pick its row independently, which can violate a sentence whose scope spans several families. We therefore introduce a class of LCN sentences, called \emph{family-realizable sentences}, and identify conditions when $\DSE = \DLCN$.''',
    r'''Second, even with $K_v = K_v^{\mathrm{true}}$, the free product of \eqref{eq:pse} lets each family pick its row independently, which can violate a sentence whose scope spans several families. We therefore introduce a class of LCN sentences, called \emph{family-realizable sentences}, and identify conditions under which $\mathcal P_{\mathrm{SE}} = \DLCN$, so that $\DSE$ and $\DLCN$ have the same bounds.''')

sub(r'''\begin{proposition}[sufficiency]\label{prop:suff}
Let $\mathcal L$ be a chain-graph LCN with families $\{(ch_v, pa_v)\}$ and local credal sets built by the chain-graph approximation. Suppose (i) for every family $v$ and parent configuration $\pi$, $K_{v,\pi}=K_{v,\pi}^{\mathrm{true}}$, the projection of $\DLCN$ onto that row; and (ii) every sentence of $\mathcal L$ is family-realizable in some family (Definition~\ref{def:realizable}). Then $\DSE=\DLCN$.
\end{proposition}''',
    r'''\begin{proposition}[sufficiency]\label{prop:suff}
Let $\mathcal L$ be a consistent, factorizing chain-graph LCN with families $\{(\ch_v, \pa_v)\}$ and local credal sets built by the chain-graph approximation. If every sentence of $\mathcal L$ is family-realizable in some family (Definition~\ref{def:realizable}), then $\mathcal P_{\mathrm{SE}}=\DLCN$; hence $\DSE$ and $\DLCN$ have the same lower and upper probabilities for every marginal and every conditional query with positive denominator.
\end{proposition}

The condition $K_{v,\pi}=K^{\mathrm{true}}_{v,\pi}$ (no local-set widening) is not an extra hypothesis: under full realizability it follows from consistency (Lemma~\ref{lem:noGapA} in the Appendix).''')

# ---------------- A6: inner/outer claims ----------------
sub(r'''We next present three algorithms that compute exact, outer and inner bounds for $\DSE$, respectively.''',
    r'''We next present three algorithms that compute, respectively, exact, outer and inner bounds for $\DSE$. By Proposition~\ref{prop:inclusion} the first two are outer bounds for $\DLCN$; an inner bound for $\DSE$ carries no guarantee for $\DLCN$ unless the two have the same bounds (Proposition~\ref{prop:suff}).''')

# ---------------- A2: CVE ----------------
sub(r'''\begin{theorem}[exactness]
Let $\cL$ be a family-realizable chain-graph LCN. Then CVE computes \emph{exact bounds} with respect to $\DLCN$.
\end{theorem}''',
    r'''Without evidence the pruning step is lossless: every intermediate function is a conditional table over a fixed split of its scope, and two distinct conditional tables over the same split cannot dominate one another, so $prune$ removes only duplicates (Lemma~\ref{lem:cvecond} in the Appendix). With evidence the indicator potentials break this normalization; dominance pruning can then discard functions that attain the extrema of the \emph{normalized} posterior, and the resulting interval may be strictly inner. Exact credal inference under evidence requires the generalized Bayes rule \citep{maua2020ijar}, which we do not implement.

\begin{theorem}[exactness]
Let $\cL$ be a consistent, factorizing, family-realizable chain-graph LCN. Then, for every marginal query without evidence, CVE computes \emph{exact bounds} with respect to $\DLCN$.
\end{theorem}''')

sub(r'''Then, for each cluster we replace its components with their mean values. This yields an approximation method denoted by CVE$_{cm}$.''',
    r'''Then, for each cluster we replace its components with their mean values. This yields an approximation method denoted by CVE$_{cm}$. The mean representative is not a vertex of any credal set and dominates no member of its cluster, so CVE$_{cm}$ is a \emph{heuristic}: its intervals are neither guaranteed inner nor guaranteed outer.''')

# ---------------- A3: IBP ----------------
sub(r'''Given a credal network compiled to a \emph{factor graph} \citep{koller2009probabilistic} whose factor nodes $\{f_i\}$ store the finite sets of extreme points $\mathrm{ext}\,K(X_i \mid \pi)$ (one per parent configuration $\pi$ of the associated variable $X_i$), plus evidence $\mathbf{Y} = \mathbf{y}$, our second approach called Interval Belief Propagation (IBP) computes bounds $[\underline{P}(X_i{=}x \mid \mathbf{y}),\, \overline{P}(X_i{=}x \mid \mathbf{y})]$ for every non-evidence variable $X_i$ and every state $x \in D_{X_i}$. Unlike previous algorithms like L2U \citep{l2u} and GL2U \citep{antonucci2010} which are restricted to binary variables, IBP natively handles multi-valued variables.''',
    r'''Given a credal network compiled to a \emph{factor graph} \citep{koller2009probabilistic} whose factor nodes $\{f_i\}$ store the finite sets of extreme points $\mathrm{ext}\,K(X_i \mid \pi)$ (one per parent configuration $\pi$ of the associated variable $X_i$), our second approach called Interval Belief Propagation (IBP) computes marginal bounds $[\underline{P}(X_i{=}x),\, \overline{P}(X_i{=}x)]$ for every variable $X_i$ and every state $x \in D_{X_i}$. IBP propagates \emph{downward} only, from parents to children, like the predictive half of 2U/L2U \citep{l2u}; it performs no upward (diagnostic) updates and is therefore not a method for queries with evidence. It accepts multi-valued (compound) variables, but its exactness guarantee is for binary variables only.''')

sub(r'''Each directed edge of the factor graph carries an interval-vector message
$\mu_{v \to f} = (\mathbf{l}_{v \to f}, \mathbf{u}_{v \to f}) \in
[0,1]^{|D_v|} \times [0,1]^{|D_v|}$ (and symmetrically $\mu_{f \to v}$) with
$\mathbf{l} \le \mathbf{u}$ componentwise. All messages are initialized to the
vacuous interval $(\mathbf{0}, \mathbf{1})$. For every evidence assignment
$Y_j = y_j$, the outgoing variable-to-factor messages from $Y_j$ are clamped to
the one-hot indicator
$\mu_{Y_j \to f} \gets (\delta_{y_j}, \delta_{y_j})$ and held fixed throughout.

At every iteration, two sweeps are performed. \emph{Variable-to-factor} messages
(skipped for evidence variables) tighten by intersecting the other incoming
bounds:''',
    r'''Each directed edge of the factor graph carries an interval-vector message
$\mu_{v \to f} = (\mathbf{l}_{v \to f}, \mathbf{u}_{v \to f}) \in
[0,1]^{|D_v|} \times [0,1]^{|D_v|}$ (and symmetrically $\mu_{f \to v}$) with
$\mathbf{l} \le \mathbf{u}$ componentwise. All messages are initialized to the
vacuous interval $(\mathbf{0}, \mathbf{1})$. Write $f_i$ for the family factor of
$X_i$, with scope $\{X_i\}\cup\mathbf{pa}(X_i)$.

At every iteration, two sweeps are performed. \emph{Variable-to-factor} messages
tighten by intersecting the other incoming bounds:''')

sub(r'''(component-wise, over all neighbor factors $\mathcal{N}(v)$ other than $f$). \emph{Factor-to-variable}
messages extremize the target's normalized marginal over two discrete choice
sets:
\begin{equation}\label{eq:compact-ftov}
  \mathbf{l}_{f_i \to v}(x) \;=\; \min_{(\mathbf{vc},\,\mathbf{cc})}
    \tilde{p}_{\mathbf{vc},\mathbf{cc}}(x),
  \qquad
  \mathbf{u}_{f_i \to v}(x) \;=\; \max_{(\mathbf{vc},\,\mathbf{cc})}
    \tilde{p}_{\mathbf{vc},\mathbf{cc}}(x),
\end{equation}''',
    r'''(component-wise, over all neighbor factors $\mathcal{N}(v)$ other than $f$); if the
intersection is empty ($\mathbf l>\mathbf u$ in some component) the upper bound is
raised to the lower one. \emph{Factor-to-variable} messages are vacuous towards a
parent, $\mu_{f_i\to w}=(\mathbf 0,\mathbf 1)$ for $w\in\mathbf{pa}(X_i)$: a
conditional table constrains nothing about its parents' marginals. Towards the
child they extremize the child's marginal over two discrete choice sets:
\begin{equation}\label{eq:compact-ftov}
  \mathbf{l}_{f_i \to X_i}(x) \;=\; \min_{(\mathbf{vc},\,\mathbf{cc})}
    \tilde{p}_{\mathbf{vc},\mathbf{cc}}(x),
  \qquad
  \mathbf{u}_{f_i \to X_i}(x) \;=\; \max_{(\mathbf{vc},\,\mathbf{cc})}
    \tilde{p}_{\mathbf{vc},\mathbf{cc}}(x),
\end{equation}''')

sub(r'''parent configuration $\pi$, a \emph{corner-point combination} $\mathbf{cc}$
picks $\mathbf{q}_w \in \{\mathbf{l}_{w \to f_i}, \mathbf{u}_{w \to f_i}\}$ for
each other-variable $w \in \mathrm{scope}(f_i) \setminus \{v\}$, and''',
    r'''parent configuration $\pi$, a \emph{corner-point combination} $\mathbf{cc}$
picks, for each parent $w\in\mathbf{pa}(X_i)$, a distribution $\mathbf q_w$ at an
endpoint of its incoming interval --- for binary $w$, $\mathbf q_w=(1-l_w,l_w)$ or
$(1-u_w,u_w)$ with $[l_w,u_w]$ the interval on $P(w{=}1)$ (for multi-valued $w$ the
normalized vectors $\mathbf l_{w\to f_i}$ and $\mathbf u_{w\to f_i}$, a heuristic
choice) --- and''')

sub(r'''The final conditional marginal bounds for every variable are obtained by
intersecting all factor-to-variable messages incident on it:
\begin{equation}\label{eq:compact-extract}
  \underline{P}(X_i{=}x \mid \mathbf{y}) = \max_{f \in \mathcal{N}(X_i)}
    \mathbf{l}_{f \to X_i}(x),
  \qquad
  \overline{P}(X_i{=}x \mid \mathbf{y}) = \min_{f \in \mathcal{N}(X_i)}
    \mathbf{u}_{f \to X_i}(x).
\end{equation}''',
    r'''The final marginal bounds for every variable are obtained by
intersecting all factor-to-variable messages incident on it:
\begin{equation}\label{eq:compact-extract}
  \underline{P}(X_i{=}x) = \max_{f \in \mathcal{N}(X_i)}
    \mathbf{l}_{f \to X_i}(x),
  \qquad
  \overline{P}(X_i{=}x) = \min_{f \in \mathcal{N}(X_i)}
    \mathbf{u}_{f \to X_i}(x).
\end{equation}
Since only $f_i$ sends a non-vacuous message to $X_i$, this is the message
$\mu_{f_i\to X_i}$ itself.''')

sub(r'''\begin{theorem}[exactness]
    Let $\mathcal{L}$ be a family-realizable chain-graph LCN. If $\mathcal{L}$'s structure is singly connected (i.e., a tree), then IBP computes \emph{exact bounds} with respect to $\DLCN$.
\end{theorem}''',
    r'''\begin{theorem}[exactness]
    Let $\mathcal{L}$ be a consistent, family-realizable chain-graph LCN whose structure is a polytree over binary atoms (no undirected component). Then, for every marginal query without evidence, IBP computes \emph{exact bounds} with respect to $\DLCN$.
\end{theorem}

The restrictions are necessary rather than technical. Without upward messages IBP cannot condition on evidence on a descendant. Exact inference in polytree-shaped credal networks is NP-hard already for ternary variables \citep{maua2014jair}, so no polynomial message-passing scheme is exact with multi-valued variables. On loopy structures, the parents of a node are no longer independent and corner enumeration over their marginals is only an outer approximation.''')

# ---------------- A6: CDVE ----------------
sub(r'''The output is \emph{inner bounds} for $\DSE$ --- coordinate descent explores only a subset of feasible vertex combinations, so the returned interval is contained in the true interval. ''',
    r'''The output is an \emph{inner} bound for $\DSE$ --- coordinate descent explores only a subset of feasible vertex combinations, so the returned interval is contained in the interval of $\DSE$. It is an inner bound for $\DLCN$ only when $\DSE$ and $\DLCN$ have the same bounds (Proposition~\ref{prop:suff}); otherwise its relation to the LCN interval is not determined. ''')

# ---------------- A7: CJT solver statement ----------------
sub(r'''The program is nonconvex through the
bilinear LMC assertions and is solved globally.''',
    r'''The program is nonconvex through the
bilinear LMC assertions. Each endpoint is solved by spatial branch-and-bound (SCIP); an endpoint the solver does not certify within its time limit is reported by its dual bound, so every reported interval is an outer bound, and an exact one when the hypotheses below hold and both endpoints are certified (Appendix~\ref{sec-full-cjt}).''')

open(P, 'w').write(s)
print("ok")
