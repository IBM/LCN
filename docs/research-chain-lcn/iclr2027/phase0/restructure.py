"""Reorganize the main text: Background = CN, LCN, chain graphs; Sec. 3 = CJT
(main contribution, no compilation); Sec. 4 = credal-network compilation and its
algorithms. Experiments/related/conclusion untouched (except \\ref numbering).
One-off; asserts that every source block is used exactly once."""
import sys

P = sys.argv[1]
s = open(P).read()


def between(a, b):
    i = s.index(a)
    j = s.index(b, i)
    return s[i:j]


# ---------------------------------------------------------------- slices
INTRO_HEAD = '%% ================================================================\n\\section{Introduction}'
BG_HEAD = '%% ================================================================\n\\section{Background}'
L2C_HEAD = '%% ================================================================\n\\section{From LCNs to Credal Networks}'
CJT_HEAD = '%% ================================================================\n\\section{Exact LCN Inference at Treewidth Cost}'
EXP_HEAD = '%% ================================================================\n\\section{Experiments}'

pre = s[:s.index(INTRO_HEAD)]
intro = between(INTRO_HEAD, BG_HEAD)
bg = between(BG_HEAD, L2C_HEAD)
l2c = between(L2C_HEAD, CJT_HEAD)
cjt = between(CJT_HEAD, EXP_HEAD)
rest = s[s.index(EXP_HEAD):]
assert pre + intro + bg + l2c + cjt + rest == s

# background pieces
bg_lcn = between(r'\subsection{Logical Credal Networks}', r'\subsection{Chain Graphs and Factorization}')
bg_cg = bg[bg.index(r'\subsection{Chain Graphs and Factorization}'):]

# l2c pieces
l2c_cn = between(r'\subsection{Credal Networks}', r'\subsection{The Chain Graph Approximation for LCNs}')
l2c_rest = l2c[l2c.index(r'\subsection{The Chain Graph Approximation for LCNs}'):]

# cjt pieces: keep the program, theorem; rewrite the motivation paragraph
cjt_prog = cjt[cjt.index(r'\[' + '\n' + r'\begin{aligned}'):]

# ---------------------------------------------------------------- intro
NEW_INTRO = r'''%% ================================================================
\section{Introduction}
\label{sec-intro}
%% ================================================================

Combining logical and probabilistic reasoning is a long-standing goal of Artificial Intelligence. Frameworks such as Markov Logic Networks \citep{mln}, ProbLog~\citep{problog}, and Probabilistic Soft Logic~\citep{bach2017psl} encode domain knowledge as logical rules augmented with uncertainty, but most of them require \emph{precise} probability values or impose strong structural assumptions such as acyclicity. In many real applications, however, domain expertise yields only \emph{ranges} of plausible probabilities and the rules of interest are neither directed nor acyclic.

\emph{Logical Credal Networks} (LCNs)~\citep{marinescu2022neurips} were introduced to address this gap: they allow \emph{imprecise} specifications of the form $\alpha \leq P(\phi\,|\,\varphi) \leq \beta$ over arbitrary propositional formulas, with no acyclicity restriction. An LCN defines a set of probability distributions consistent with its constraints and a local Markov condition, and inference amounts to computing tight lower and upper bounds on the probabilities of query formulas. This flexibility makes LCNs well suited to causal reasoning with partial knowledge, robust decision making, and trustworthy neuro-symbolic systems, but it also makes inference considerably harder than in Bayesian or classical credal networks \citep{pearl88,cozman2000credal}. The only exact method optimizes a nonconvex program over the full joint distribution, with one variable for each of the $2^n$ truth assignments \citep{marinescu2022neurips}, which confines it to a handful of propositions; the only approximate method is the ARIEL message-passing scheme~\citep{marinescu2023lcn}.

\paragraph{Contribution.}
We show that exact LCN inference need not pay the $2^n$ cost. Our main contribution is \emph{Credal Junction Tree} (CJT), an exact algorithm for LCNs whose structure is a chain graph. Building on the chain-graph factorization of LCNs established by \citet{cozman23a}, CJT moralizes and triangulates the chain graph, builds a junction tree, and replaces the global joint distribution by one local marginal per cluster, tied together by separator-consistency constraints. The LCN's probabilistic sentences are imposed on the clusters that contain them, and its independence assertions are either entailed by the tree's topology, reduced to equivalent smaller assertions that fit in a cluster, or implied by one constraint per node of a topologically ordered tree. The resulting nonlinear program has $O(n\,2^{w+1})$ variables for treewidth $w$ instead of $2^n$, and we prove that, under conditions we characterize and check mechanically, its global optimum is the exact LCN bound. CJT needs only the chain graph's \emph{structure}: unlike the approaches below, it never compiles the LCN into a credal network, never solves per-family local problems and never enumerates extreme points, so it is not affected by the relaxation those steps introduce.

Our second contribution is a systematic \emph{credal-network compilation} of chain-graph LCNs, which makes the well-developed machinery of credal networks \citep{cozman2000credal,maua2020ijar,l2u,antonucci2015approxlp} available for LCN inference. The compilation replaces the LCN's distribution set by the strong extension of a credal network over the chain components, which contains it; we identify a class of \emph{family-realizable} LCNs on which the two have the same bounds. On this basis we adapt three credal-network algorithms: credal variable elimination (CVE), exact for family-realizable LCNs without evidence, with a clustered variant CVE$_{cm}$; interval belief propagation (IBP), exact on family-realizable binary polytrees without evidence; and a coordinate-descent method (CDVE) that computes inner bounds. We evaluate all methods on synthetic and realistic LCN benchmarks against CJT as the exact reference.

The Appendix contains the proofs, detailed descriptions of all algorithms with worked examples, and the full experimental results.

'''

# ---------------------------------------------------------------- background
CN_BLOCK = l2c_cn.replace(r'\subsection{Credal Networks}', r'\subsection{Credal Networks}' + '\n' + r'\label{sec-cn}', 1)
# The hull lemma pointer now refers forward to Sec. 4 for its use.
NEW_BG = (r'''%% ================================================================
\section{Background}
\label{sec-background}
%% ================================================================

''' + CN_BLOCK + bg_lcn + bg_cg)

# Chain-graph section: add the family / node-DAG vocabulary used by Secs. 3-4.
CG_ADD = r'''

We call the chain components the \emph{nodes} of the factorization: a node $v$ is a
single atom, or a \emph{compound} node when its component contains several atoms,
and $\ch_v$ denotes its atoms and $\pa_v$ its parents. The pairs
$(\ch_v,\pa_v)$ are the \emph{families} of the LCN, the sets $\ch_v\cup\pa_v$ its
\emph{family scopes}, and contracting each component to its node turns the chain
graph into a DAG $D$ over the nodes, the \emph{node DAG}. An LCN whose structure
is a chain graph is called a \emph{chain-graph LCN}; we consider only these
hereafter.
'''
anchor = r'''results relying on \eqref{eq:factorization} are stated for such \emph{factorizing} LCNs.'''
assert NEW_BG.count(anchor) == 1
NEW_BG = NEW_BG.replace(anchor, anchor + CG_ADD)

# ---------------------------------------------------------------- CJT (Sec 3)
NEW_CJT = r'''%% ================================================================
\section{Exact LCN Inference at Treewidth Cost}
\label{sec-cjt}
%% ================================================================

Exact inference in an LCN computes $\lo P(a)=\min_{p \in \DLCN}P(a)$ and
$\hi P(a)=\max_{p \in \DLCN}P(a)$. Optimizing over $\DLCN$ directly requires a
nonlinear program with one variable per joint state, $2^n$ in total
\citep{marinescu2022neurips}. We introduce \emph{Credal Junction Tree} (CJT), which
solves the same problem exactly with a program whose size is exponential only in
the treewidth of the chain graph.

CJT uses the chain graph only through its family scopes
(Section~\ref{sec-background}). It moralizes and triangulates the chain graph,
producing a chordal graph $G^\ast$, and builds a junction tree $T$ whose
clusters $\mathcal{C}$ satisfy the running-intersection property
\citep{dechter-book}. Instead of one variable per joint state, it introduces a
local marginal $q_C\in\Delta(2^{|C|})$ for each cluster $C\in\mathcal{C}$ and solves
(see the Appendix for details and a worked example):
''' + cjt_prog

# Wording fixes inside the kept CJT program block: nothing refers to the CN.
# Complexity becomes a numbered theorem (JNVR), placed after the exactness one.
old_cplx = r'''The formulation needs $\sum_C2^{|C|}{=}O(n2^{w+1})$ variables and $O(n2^w)$ separator-consistency constraints making its complexity exponential in the treewidth $w$ rather than in the number of variables $n$. '''
assert NEW_CJT.count(old_cplx) == 1
NEW_CJT = NEW_CJT.replace(old_cplx, r'''\begin{theorem}[size]\label{thm:cjt-size}
Let $w+1$ be the largest cluster of $T$, in atoms. The CJT program has
$\sum_{C}2^{|C|}=O(n\,2^{w+1})$ variables, $O(n\,2^{w})$ separator-consistency
constraints, two rows per sentence and at most $2^{w+1}$ rows per imposed
independence assertion.
\end{theorem}

The size bound concerns the formulation, not the solving time: the program is
nonconvex, and global optimization of it is not polynomial in its size. Its
benefit is that the program is exponential in the treewidth $w$ rather than in
$n$, so that exact inference becomes possible on instances where the $2^n$
formulation cannot even be written down. CJT never compiles a credal network: it
uses neither per-family local credal sets nor their extreme points, and its
target is $\DLCN$ itself. Section~\ref{sec-lcn-to-credal} shows what is gained,
and what is lost, by compiling instead.
''')

# ---------------------------------------------------------------- compilation (Sec 4)
NEW_L2C_HEAD = r'''%% ================================================================
\section{Approximate Inference via Credal Network Compilation}
\label{sec-lcn-to-credal}
%% ================================================================

An alternative to solving the LCN program, exactly as CJT does or approximately,
is to compile the LCN into a credal network (Section~\ref{sec-cn}) and use the
algorithms developed for those. We describe the compilation, characterize when it
is exact, and adapt three credal-network algorithms to it.

'''
l2c_rest = l2c_rest.replace(
    r'''Given a chain-graph LCN $\mathcal{L}$ over variables $\bX=\{X_1,\ldots,X_n\}$, we approximate $\DLCN$ by the strong extension $\DSE$ of an induced credal network. The construction is based on the chain components $\mathcal{T}_1,\ldots,\mathcal{T}_m$ of the underlying chain graph. We first build a DAG $G'$ whose nodes correspond to these chain components. Each node represents the joint state space of the variables in the corresponding chain component, that is, the Cartesian product of their domains. Thus, a node corresponds to a compound variable whenever its chain component contains multiple variables, and to a single LCN variable otherwise. Let $v$ denote a family in $G'$, consisting of a child node $ch_v$ and its parent set $pa_v$.''',
    r'''Given a chain-graph LCN $\mathcal{L}$ over variables $\bX=\{X_1,\ldots,X_n\}$, we approximate $\DLCN$ by the strong extension $\DSE$ of an induced credal network over the node DAG $D$ (Section~\ref{sec-background}). Each node takes values in the joint state space of its chain component, so a compound node is a multi-valued variable. Let $v$ be a family, with child node $\ch_v$ and parent set $\pa_v$.''')
assert r'over the node DAG $D$' in l2c_rest
NEW_L2C = NEW_L2C_HEAD + l2c_rest

# ---------------------------------------------------------------- assemble
out = pre + NEW_INTRO + NEW_BG + NEW_CJT + NEW_L2C + rest
open(P, 'w').write(out)
print("ok", len(s), "->", len(out))
