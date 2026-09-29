# AND/OR search for MAP inference in chain graph LCNs

Detailed research plan — 29 September 2026.
Repository baseline: `bae09ed2174858581301c628553c9f82df39dd03`.

This is a proposal for new research and implementation. The code and literature
findings below are observations; proposed algorithmic contributions and theorems
are identified as research work. References [R1]–[R14], reading notes, and source
verification details are in [references.md](references.md).

## 1. Research direction

Develop depth-first branch-and-bound and best-first AND/OR search for **maximin
and maximax joint MAP in chain graph logical credal networks**. Extend the methods
to marginal MAP after establishing the full-assignment case. The central issue is
how to exploit graphical decomposition while retaining the logical constraints
that couple probability models across the decomposed subproblems.

The closest starting points are *Abductive Reasoning in Logical Credal Networks*
(NeurIPS 2024) [R1] and *Credal Marginal MAP* (NeurIPS 2023) [R2]. Both explicitly
identify branch-and-bound/best-first search with effective bounding heuristics as
future work. The AND/OR foundations [R3–R6] provide pseudo-trees, decomposition,
context caching, and search control; they do not by themselves justify replacing
precise factors with independently optimized probability intervals.

The proposed research has three main outputs:

1. A characterization of when chain LCN subproblems can be combined exactly at
   AND nodes, and what information a reusable search context must retain.
2. Admissible bound families for both maximin and maximax, including bounds from
   partial explanation events, relaxations, and feasible probability models.
3. Two algorithms using those bounds: depth-first branch-and-bound for low-memory
   anytime solutions, and best-first graph search for systematic reduction of
   the optimality gap, followed by a memory-bounded hybrid.

The research claim should be about preserving LCN semantics and improving
certified search. AND/OR search, credal MAP, mini-bucket heuristics, and generic
scenario generation are already established ideas.

## 2. Define the inference problem precisely

### 2.1 Primary target: the joint-probability convention in the LCN paper

Partition atoms into MAP variables $Y$, observed variables $E=e$, and hidden
variables $Z$. For a feasible joint distribution $p$, let

$$
 F(y,p)=P_p(y,e)=\sum_z p(y,z,e).
$$

Let $D$ be the set of distributions satisfying every LCN sentence and every Local
Markov Condition (LMC) equality. Define

$$
 s_-(y)=\inf_{p\in D}F(y,p),\quad
 s_+(y)=\sup_{p\in D}F(y,p),\qquad
 V_- =\max_y s_-(y),\quad V_+=\max_y s_+(y).
$$

These are maximin and maximax respectively. Full MAP means $Z$ is empty; marginal
MAP means $Z$ is nonempty. The primary task is to return an assignment, a certified
lower bound on its score, and an upper bound on the optimal score. Returning all
ties or an interval-dominance frontier is a separate output contract.

The phrase “given evidence” in [R1] uses the joint event $P(y,e)$ in Definitions
1–2. The existing `evaluate_config` also conjoins evidence and calls its scorer
without a separate conditional-evidence argument. Preserve this convention for
the main experiments and name it explicitly.

### 2.2 Posterior MAP is a distinct extension

The posterior variant replaces $F$ with $P_p(y,e)/P_p(e)$, for $P_p(e)>0$. For a single
precise model the common denominator does not affect the winning assignment.
For a credal set the denominator varies with $p$, so the two tasks can differ.

For example, consider the convex hull of these two distributions:

| Model | $P(A,e)$ | $P(\neg A,e)$ | $P(\neg e)$ |
| --- | ---: | ---: | ---: |
| p1 | 0.04 | 0.06 | 0.90 |
| p2 | 0.80 | 0.10 | 0.10 |

Joint maximin chooses $\neg A$: its lower joint mass is 0.06 rather than 0.04.
Posterior maximin chooses $A$: its lower posterior is 0.4 rather than 1/9.
The minima occur at the endpoints because the ratios on this segment are
linear-fractional with positive denominators.

Do not introduce a denominator floor silently. A restriction $P(e)>=\beta$ defines
a different feasible domain unless justified. Establish regular-extension and
zero-evidence behavior separately. The main joint-MAP work avoids this ratio
issue and matches the existing research baseline.

### 2.3 What a branch means

A partial MAP assignment $a$ identifies a set of complete explanations. Write
$C(a)=\{y:y \textrm{ extends } a\}$ and

$$
 V_\pm(a)=\max_{y\in C(a)}s_\pm(y).
$$

Search branches change the event being scored; they do **not** assert $P(a)=1$,
replace prior constraints with posterior constraints, or modify the LCN's LMC.
This distinction matters when conditioning factors for computational purposes.
A zero-score branch and an inconsistent input LCN are different conditions.

### 2.4 Chain graph representation

For chain components $C_j$, the candidate representation is

$$
 p_\theta(x)=\prod_j\theta_j(x_{C_j}\mid x_{pa(C_j)}).
$$

Use the component DAG for organization, but retain the atom-level structure,
sentence scopes, and internal independencies of noncomplete components. A table
for an entire component is not automatically unconstrained. The Markov and
factorization equivalences in [R11] require positivity in the general case;
deterministic models and boundaries need separate justification.

The search can always fall back to original-joint probability optimization on
small models. An exact claim for a reduced parameterization needs both soundness
and completeness relative to D. Probability intervals from compilation may be a
relaxation, even when each coordinate bound was computed exactly. See also the
[sampling research plan](../research-sampling/research-plan.md), Sections 2–3.

## 3. Literature findings and what remains open

| Research line | Established contribution | Role in this project |
| --- | --- | --- |
| LCN abduction, 2024 [R1] | Joint maximin/maximax MAP and marginal MAP; exhaustive DFS, LDS, SA, and approximate scoring. | Direct baseline and explicit motivation for pruning and best-first search. |
| Credal marginal MAP, 2023 [R2] | Set-valued variable elimination, DFS, credal mini-buckets, and local search. | Closest credal predecessor; inspect which guarantees depend on independent local model choices and maximax. |
| AND/OR spaces, 2007 [R3] | Pseudo-trees expose decomposition; context merging yields graph search whose size depends on width. | Search-space foundation, once residual probabilistic and credal dependencies are represented. |
| AOBB, 2009 [R4] | Depth-first AND/OR branch-and-bound, orderings, static/dynamic mini-bucket heuristics. | Low-memory search control and initial precise-model baseline. |
| Memory-intensive AND/OR, 2009 [R5] | Context caching, graph AOBB, and best-first search. | Cache semantics, solution graphs, and best-first versus DFS comparison. |
| AND/OR marginal MAP, 2018 [R6] | Constrained pseudo-trees, sum/max search, weighted mini-buckets, and anytime combinations. | Operator-order discipline and separation of assignment expansion from expensive evaluation. |
| Anyspace and recursive best-first, 2018–2019 [R7–R8] | Bounded-memory search and progressive refinement of marginal-MAP bounds. | Later hybrid design; memory-limited best-first is already prior art. |
| Mini-buckets and decomposition bounds [R9] | Bounded-width admissible relaxations for precise graphical models. | Heuristic construction, subject to proving the bound direction after credal lifting. |
| Probability–possibility credal MAP, 2017 [R10] | An alternative approximate approach to credal MAP. | Additional comparator after checking its target and semantics. |
| LCN factorization and credal-network foundations [R11–R12] | Chain graph semantics and distinctions among credal extensions. | Establish the actual optimization domain. |
| Robust optimization and rectangularity [R13–R14] | Scenario generation and conditions for separable robust dynamic programming. | Useful analogies; their theorems do not transfer automatically to LCNs. |

The core papers [R1–R8] were inspected in full-text form for this plan. The review
also checked the local factorization manuscript and bibliographic records for
the broader comparisons. This focused search does not establish an absence of
later or differently named related work. Before publication, complete backward
and forward citation searches from [R1], [R2], and [R6], and examine robust
abduction, robust combinatorial optimization, decision circuits, and influence
diagrams with imprecise probabilities.

Avoid treating the maximin case as a mechanical change of max to min in a
maximax algorithm. In particular, assignment choice must remain outside the
adversarial minimization. Similarly, a “best” set-valued message is insufficient
for reconstructing an assignment without coherent witness/backpointer tracking.

## 4. Current implementation and required foundations

| Repository component | Observation | Planned use or change |
| --- | --- | --- |
| [exact_map.py](../../lcn/inference/map/exact_map.py) | `_run_dfs` expands the binary assignment tree and scores leaves; no partial-assignment pruning, pseudo-tree, or context cache. Dispatch supports DFS, LDS, and SA. | Preserve as an enumeration baseline; add new search engines separately. |
| `solve_exact_model` in the same file | Uses IPOPT and an SLSQP fallback on a nonconvex joint-LMC problem. `evaluate_config` discards status flags. | Local feasible objectives cannot serve as global certificates; introduce an explicit bound-oracle result contract. |
| [approx_map.py](../../lcn/inference/map/approx_map.py) | Approximate assignment scoring and search can cheaply propose explanations. | Candidate generation and ordering only, unless a particular bound has been independently proved admissible. |
| [model.py](../../lcn/core/model.py), [factorization.py](../../lcn/inference/marginal/cn/factorization.py) | Component structure, scopes, LMC assertions, and optional family merging. | Build and validate the search graph; do not mistake symbolic factorization for independent credal specification. |
| [local_credal_sets.py](../../lcn/inference/marginal/cn/local_credal_sets.py), [vertices.py](../../lcn/inference/marginal/cn/vertices.py) | Compute coordinate intervals and independent row vertices. | Exact baseline on a verified separate subclass; relaxations or proposals elsewhere. |
| [coupling.py](../../lcn/inference/marginal/cn/coupling.py) | Identifies constraints whose scopes fit no family. | Reuse scope/indicator utilities, but include all lost couplings, including within-family and across-row ones. |
| [exact.py](../../lcn/inference/marginal/exact.py) | Has a global SCIP query path, reporting incumbent, gap, and status. Returned endpoints can remain incumbents on timeout. | Expose certified numerical objective bounds separately from feasible values and retain full-model witnesses. |
| [junction_nlp.py](../../lcn/inference/marginal/cn/junction_nlp.py) | Cluster constraints and separator consistency; main interface targets singleton marginals. | Investigate incremental arbitrary-event scoring. A growing conjunction may require wider clusters; reuse is not automatic. |
| [potentials.py](../../lcn/inference/marginal/cn/potentials.py), [ijgp.py](../../lcn/inference/marginal/cn/ijgp.py) | Factor operations, orderings, and mini-bucket infrastructure. | Reuse mechanics; re-prove each search bound for its score and target. |

An oracle result should carry `(certified_lower, certified_upper,
feasible_model, feasible_value, status, tolerance, target_id, cost)`.
For a minimization the feasible value is an **upper** bound on its optimum;
for a maximization it is a **lower** bound. Names such as `lower_bound` for an
estimated lower probability must not obscure that distinction.

Global-solver claims must use numerical certificates appropriate to the backend,
including tolerances. A local solver reporting `optimal` does not prove a global
optimum of the nonconvex LMC problem. Unknown, infeasible, time-limited, and
zero-probability outcomes must remain distinguishable throughout search.

## 5. Extending the AND/OR search space

### 5.1 Establish the simple exact reduction first

Suppose the original target is exactly the product of independent, nonempty,
compact sets $K_j$ of normalized component tables. Every combination of those
tables must define an admissible model. For a **complete** assignment $x=(y,e)$,
define local scalar factors

$$
 l_j(x_{C_j},x_{pa_j})=\min_{\theta_j\in K_j}\theta_j(x_{C_j}\mid x_{pa_j}),
 \quad
 u_j(x_{C_j},x_{pa_j})=\max_{\theta_j\in K_j}\theta_j(x_{C_j}\mid x_{pa_j}).
$$

Then independent attainment and nonnegativity give

$$
 s_-(y)=\prod_j l_j(y,e),\qquad s_+(y)=\prod_j u_j(y,e).
$$

Both tasks reduce to ordinary MAP over nonnegative factors. The endpoint tables
need not normalize; they are search potentials, not a new Bayesian network.
The same conclusion holds for linear event extrema over the convex hull of the
product models. This is a baseline reduction, not the intended novelty claim.

Whole-table restrictions within one $K_j$ are compatible with this reduction:
a complete assignment reads only one entry of that component table, and its
attaining table can be chosen independently of the other components. The
conditions fail when original LCN constraints link component choices or when
local coordinate calculations do not describe the actual attainable entries.
With hidden variables, the same table can contribute to several summed terms;
independent optimization of each term is generally invalid.

Implement ordinary AOBB/AOBF on these endpoint potentials first. It establishes
the search machinery and reveals how much of a benchmark collection already
reduces to known precise-model optimization.

### 5.2 Why probabilistic decomposition is insufficient

Suppose two residual factors are $t$ and $1-t$ with a shared feasible parameter
$t$ in $[0.2,0.8]$. Their extrema illustrate the danger:

$$
 \min_t t(1-t)=0.16\ne0.04=(\min_t t)(\min_t(1-t)),
$$
$$
 \max_t t(1-t)=0.25\ne0.64=(\max_t t)(\max_t(1-t)).
$$

These can be the probabilities of two independent events in each precise model.
The dependence is among admissible parameter choices, not among worlds within a
fixed model. Therefore a disconnected conditioned world graph alone is not a
certificate for multiplying independently optimized child values.

An AND decomposition needs a factorization of the **residual optimization
problem**. Conditional on its boundary $b$, a sufficient form is

$$
 F= w(b)\prod_j F_j(y_j,\theta_j;b),\qquad
 \Theta(b)=\prod_j\Theta_j(b),
$$

with disjoint decision sets, nonnegative factors, and no omitted constraint or
shared parameter. Under these conditions each child can optimize independently
and the results can be multiplied. If b includes unfixed probability parameters,
the child results remain functions or constrained objects over b until a valid
joint optimization eliminates it. World assignments to separators do not fix
separator probability distributions.

The first main theorem should give sufficient decomposition conditions for a
useful LCN subclass, together with counterexamples to weaker graph-only criteria.
It need not give a cheap necessary-and-sufficient test for arbitrary LCNs.

### 5.3 Graphs, pseudo-trees, and node types

Build a computational interaction representation from component family factors,
all active sentence constraints, the LMC constraints not already structurally
entailed, shared table parameters, and any boundary quantities required by the
chosen oracle. Audit graph recognition so contraction does not hide a forbidden
semidirected cycle.

An atom-scope hypergraph is a conservative starting point for cluster placement.
It is not automatically a sufficient graph for parameter optimization: a local
sentence can depend on upstream marginal probabilities. Either use a probability-
variable constraint graph directly or retain enough symbolic dependence to prove
that the resulting residual subproblems really separate.

Use a pseudo-tree whose non-tree interaction edges connect ancestors and
descendants. Investigate two representations:

- Component-state variables, with domain size $2^|C_j|$. These preserve component
  tables but can create large branching factors.
- Atom-level search with component/cluster factors retained. This gives binary
  branching, but may require more context and larger bound computations.

The search distinguishes four operations:

| Operation | Meaning | Exact backup when its conditions hold |
| --- | --- | --- |
| Decision OR | Choose a MAP variable value. | Maximum over alternative scores. |
| Decomposition AND | Solve independent residual subproblems jointly. | Product of child values and any separately owned factor. |
| Hidden-state SUM | Marginalize an unobserved variable in marginal MAP. | Sum under the same precise model/compatible parameter object. |
| Model optimization | Minimize or maximize over admissible distributions. | A constrained oracle or explicitly represented parameter search. |

Do not conflate decomposition AND with logical conjunction or adversarial choice.
An arbitrary continuous LCN feasible set is not a finite collection of MIN-node
branches. Begin with oracle-backed model optimization; use explicit model nodes
only for finite or rigorously bounded domains.

For marginal MAP, a safe starting pseudo-tree puts MAP decisions above hidden
summation variables, as in [R6]. Respect the order $\max_y$ $\inf_\theta$ $\sum_z$ for
maximin and $\max_y$ $\sup_\theta$ $\sum_z$ for maximax. The two maxima commute in the
second expression, but moving model optimization inside a sum can choose a
different model per hidden state. Likewise max-min interchange changes robust
decision semantics. Refine ordering constraints only after proving a valid
commutation or independence rule.

Two small examples make these ordering restrictions testable. A precise joint
table over $(Y,Z)$ with rows $(0.4,0)$ and $(0.3,0.3)$ has
$\max_y \sum_z p(y,z)=0.6$, whereas $\sum_z \max_y p(y,z)=0.7$. For model optimization,
let $p_t(Y=0,Z=0)=0.5t$ and $p_t(Y=0,Z=1)=0.5(1-t)$, with the remaining two
states each having mass $0.25$ and $t$ in $[0.2,0.8]$. Then $P_t(Y=0)=0.5$ for every
model. Summing independently minimized terms gives $0.2$; summing independently
maximized terms gives $0.8$. Those can be relaxation bounds, but neither is the
correct probability extremum of the event.

### 5.4 Context caching with coupled constraints

In a precise graphical model, ancestor assignments separating a subtree from
the rest often identify an equivalent residual problem. For LCNs the cache must
also preserve the feasible boundary-model choices and any coupling to the rest
of the explanation.

Start conservatively: key entries by residual subproblem identity, separator
assignments, relevant prefix information, evidence/score convention, constraint
and parameter-domain identity, and applicable scenario set. Store reusable
symbolic templates when numerical equivalence is unproved. Full-prefix keys are
a safe fallback, though they may yield little merging.

Research richer messages as functions of shared parameters or separator moments,
or as sets of compatible score vectors with witnesses. Only compress to a scalar
when a decomposition theorem permits it. Approximate equality of continuous
boundary quantities is not enough for exact cache merging; certified enclosing
regions could support a separate approximate scheme.

Store lower and upper certificates separately from exact values. A subproblem
pruned relative to one incumbent is not automatically solved exactly, and that
pruned status cannot be reused under a different threshold. Record the threshold
or cache only the reusable bound. Revisit entries when constraints are refined;
adding valid scenarios tightens a maximin master, so old upper bounds remain
safe but may be stale.

The second main theorem should state an equivalence criterion for cache reuse.
Measure the resulting context size rather than promising dependence only on
the component-DAG treewidth. Logical scopes, shared parameters, and the size of
set-valued boundary messages can dominate complexity.

## 6. Admissible bounds: the core algorithmic work

For both tasks, prune a branch only with an upper bound $U(a)>=V(a)$ compared
against a certified incumbent $L<=s(y_{inc})$. A heuristic that predicts promising
assignments can guide expansion without satisfying this requirement, but it
cannot justify pruning or an optimality claim.

### 6.1 Partial-event bounds

Let $E_a$ be the event $(a,e)$. Since each completion event is contained in $E_a$,

$$
 V_-(a)\le\inf_{p\in D}P_p(E_a),\qquad
 V_+(a)\le\sup_{p\in D}P_p(E_a).
$$

For maximin, even a **feasible model evaluation** $P_p(E_a)$ is a valid upper
bound on $V_-(a)$. A feasible objective from minimizing $P(E_a)$ can improve it
without requiring a global optimum. For maximax, that same feasible evaluation
does not upper-bound all admissible models; use a certified upper bound on the
supremum. These inexpensive-to-state bounds may be loose near the root but
provide a sound first branch-and-bound algorithm.

Evaluate events with objective indicators over the unchanged probability model.
Reuse the constraint system, warm starts, and factor contraction paths across
related prefixes. Count the cost of each bound call: an expensive solve at every
node can be slower than leaf enumeration despite pruning many nodes.

### 6.2 The polarity table

Assume $R$ is a proven outer relaxation of $D$ and $S$ is a nonempty collection of
validated models in $D$. For complete $y$:

| Quantity | Relation to the true score | Safe search use |
| --- | --- | --- |
| $\inf$ over $R$ of $F(y,p)$ | $<= s_-(y)$ | Maximin incumbent lower bound, using a certified lower bound on this minimization. |
| $F(y,p0)$, $p0$ feasible in $D$ | $>= s_-(y)$ | Upper bound on that candidate's maximin score; not a robust incumbent certificate. |
| $F(y,p0)$, $p0$ feasible in $D$ | $<= s_+(y)$ | Maximax incumbent lower bound, with a feasible model witness. |
| $\sup$ over $R$ of $F(y,p)$ | $>= s_+(y)$ | Maximax score upper bound, using a certified upper bound on this maximization. |

In particular, dropping LCN constraints before a **minimization** generally
lowers its value. That does not make the result an admissible upper bound for
maximin pruning. For example, relaxing $P(A)$ in $[0.4,0.6]$ to $[0,1]$ changes its
lower probability from $0.4$ to $0$; using $0$ as an upper bound is unsound.

Use validated feasible models from local NLP or other searches as witnesses in
the directions supported above. A locally minimized objective can be useful
without being a lower probability certificate.

### 6.3 Bound hierarchy for maximax

Investigate an increasingly expensive portfolio:

1. The universal bound $1$ for joint event probability and inherited parent bounds.
2. Certified coordinate upper envelopes $u_j$ of factor entries. Because the
   factors are nonnegative, their product upper-bounds every precise full-world
   mass. Maximizing or summing these envelopes gives a bound if the required
   factorization/inclusion has been proved.
3. Mini-bucket or weighted mini-bucket upper bounds on those nonnegative factors,
   preserving the appropriate sum/max inequalities. Their factors need not be
   normalized; resulting joint-probability bounds may be clipped at 1.
4. Credal set-valued or region-based bounds retaining selected local relationships,
   drawing on [R2] and rechecking the bound direction after every approximation.
5. LP/convex relaxations of the original joint/cluster problem, or certified
   global nonlinear upper bounds, applied selectively to difficult prefixes.

Exact arbitrary-coordinate bounds are unnecessary if certified enclosures are
available. An uncertified local maximum can underestimate a factor entry and
make the resulting pruning rule unsafe. Approximate message passing and
representative-based compression require their own admissibility proof.

For an incumbent, propose $y$ and find a globally feasible original-LCN model $p$.
Its $F(y,p)$ is already a valid maximax lower bound; optimize $p$ further if useful.
Filtering compiled vertices alone can miss feasible extrema introduced by
coupling constraints, so it is neither a completeness argument nor the only
candidate generator.

### 6.4 Bound hierarchy for maximin

Start with the partial-event bound from Section 6.1 and the safe, possibly weak,
maximax bound $V_-(a)<=V_+(a)$. Add two stronger constructions.

**A portfolio of feasible models.** For any nonempty $S$ subset of $D$,

$$
 V_-(a)\le
 M_S(a):=\max_{y\in C(a)}\min_{p\in S}F(y,p)
 \le\min_{p\in S}\max_{y\in C(a)}F(y,p).
$$

The rightmost bound can reuse precise-model MAP/AND-OR machinery. The finite-
scenario master M_S can be tighter, but all scenarios must share **one** assignment
y. Solving each scenario and combining its preferred assignment independently
does not solve the master. Retain compatible score vectors or use a constrained
master solver; bound the cost and cardinality of its frontier.

**Adversarial scenario generation.** Solve or bound the master, take its candidate
y, and minimize F(y,p) over the original LCN. A feasible adversarial model can be
added to S, tightening the master upper bound. A certified lower bound on that
inner minimum gives a robust incumbent certificate for y. A relaxed-domain
minimization can also provide a valid candidate lower bound, but its optimizer
cannot be added as a scenario unless it is feasible in D.

This adapts a standard robust-optimization strategy [R13]. The proposed novelty
is its integration with AND/OR contexts, reusable LCN constraints, and partial-
assignment bounds. A finite scenario subset need not capture the full optimum;
termination requires the certified master/incumbent gap to close.

For nonnegative full-MAP factors, certified local lower envelopes also give
candidate lower bounds by multiplying them. Under the exact rectangular
conditions of Section 5.1 these scores are exact. Otherwise their looseness is
measured rather than assumed away.

### 6.5 Refinement and monotonicity

At a fixed search state, combine valid upper bounds by taking their minimum;
combine valid lower bounds for a fixed feasible explanation by taking their
maximum. Track the provenance of each certificate. Children may inherit their
parent's upper bound. Bounds from unrelated contexts cannot be mixed as if they
belonged to one subproblem.

For nested outer relaxations, maximax upper bounds decrease as the domain
tightens, while maximin candidate lower bounds increase. For growing feasible
scenario sets, maximin master upper bounds decrease. These monotonic effects
suggest an adaptive choice among expanding assignments, improving a feasible
model, tightening a relaxation, and refining an inner solve.

## 7. Depth-first branch-and-bound

Implement an OR-space version first using the partial-event bounds. Then enable
AND decomposition and context caching only where the conditions in Section 5
are satisfied. This isolates the benefit of pruning from the benefit of a new
search space.

The abstract algorithm for either score is:

```text
establish a nonempty target domain, or report infeasible/unknown
obtain an initial complete explanation and a certified score lower bound L
initialize a depth-first frontier containing the root
while work remains and the total budget permits:
    select the next assignment state or unfinished bound task
    obtain/refine a certified branch upper bound U(a)
    if U(a) <= L:
        close this state with its pruning certificate
    else if a is complete:
        evaluate/refine its score interval [l(a), u(a)]
        update incumbent if l(a) improves L
        retain the leaf as unresolved if u(a) still exceeds L
    else if an exact AND decomposition is certified:
        solve children with compatible boundary information
        combine certificates and explanation witnesses; update ancestors
    else:
        branch on the next MAP variable; visit promising values first
    record the upper bound represented by all remaining work
return incumbent, L, global U, and termination status
```

“Retain as unresolved” is essential. Exhausting assignment nodes does not prove
optimality if a leaf's robust minimization or optimistic maximization remains
uncertified. A complete assignment may require additional oracle work instead
of further branching. Include those tasks in the global bound.

Start with a static pseudo-tree for reproducibility and caching. Compare value
ordering by cheap bound, feasible-model prediction, and approximate-MAP output.
Afterward study dynamic variable ordering and stronger bounds at selected depths.
Recompute the residual context when the ordering changes; a cache built for an
old pseudo-tree cannot be assumed valid unchanged.

For a product AND node, pruning a child must use the upper contribution of the
other children and any prefix factors. Compute the upper value of the containing
partial solution graph; do not compare a local conditional probability directly
against the root incumbent. Prefer log-domain products with explicit zero
handling. Avoid division by a sibling bound when that bound can be zero.

Offer tree search with bounded caches first, then context-minimal graph search.
Stack space can be small while local optimization, tables, and witness storage
are large, so report total memory rather than claiming linear total space.

## 8. Best-first and memory-bounded search

### 8.1 Initial best-first baseline

Use a priority queue over assignment branches with priority equal to their
certified upper bound. Keep incomplete leaf evaluations on the queue. Update
the incumbent using certified lower bounds and close nodes only when dominated
or solved to the requested tolerance. This is a transparent baseline before
introducing AND/OR graph search.

### 8.2 Best-first AND/OR graph search

Maintain a partial explicit AND/OR graph, lower/upper value records, witness
backpointers, and parent links for all cached subproblems. At a decision OR node,
the lower and upper backups are maxima of child bounds. At a certified product
AND node, they are products of compatible child bounds and owned factors.
Coupled nodes retain their constrained oracle or boundary representation rather
than forcing a scalar product backup.

Follow the current best partial solution graph and choose an unresolved tip whose
refinement can affect its bound. A high local upper bound need not make a tip
important to the root: outside factors and AND siblings determine its root
contribution. Compare subproblem ordering by this contribution, interval width,
and observed refinement cost, with [R5–R8] as the search-control baselines.

At a tip, the algorithm can choose among:

- Expanding a MAP variable or exposing a proven decomposition.
- Refining a complete assignment's probability optimization.
- Increasing a mini-bucket/cluster budget or adding valid relaxation constraints.
- Adding a globally feasible scenario to the maximin master.
- Constructing a complete explanation to improve the incumbent.

Update every dependent parent when a shared subproblem bound changes. Preserve
factor ownership so cached reuse does not duplicate probability factors. Mark an
OR node solved when an adequately evaluated child dominates every alternative;
mark an AND node solved when its required children and their combination are
certified. For coupled states, retain the oracle's unresolved gap explicitly.

With admissible upper bounds and fair refinement, the root record encloses the
true optimum. Do not assume classical best-first node-expansion optimality
survives heterogeneous NLP costs, adaptive heuristics, changing scenarios, or
approximate caching. The primary performance criterion is elapsed time to a
certified gap, not simply the number of expanded nodes.

### 8.3 Anytime behavior, memory limits, and stopping

Compare pure AOBF with periodic DFS dives that find complete candidates, then
with recursive best-first or bounded-memory strategies inspired by [R7–R8].
Eviction must retain a valid summary of unfinished work so discarded branches
can be regenerated. Discarding a state is not a proof that it is solved.

For a single returned explanation, stop when $U_{root}-L<=\epsilon_{abs}$, optionally
with an explicitly defined relative criterion. Joint probabilities may be tiny,
so report absolute and relative gaps and handle zero denominators in gap ratios.
Never call tolerance-based termination exact real-arithmetic optimality.

If all optimal assignments are required, equality pruning can discard ties;
change the pruning rule and output contract accordingly. Interval dominance,
maximality, and E-admissibility involve different comparisons and are deferred.

Distinguish `certified`, `time_limit`, `memory_limit`, `infeasible_input`,
`oracle_unresolved`, and `unsupported_semantics`. On interruption return the
best validated explanation, available bounds, and the unresolved work summary.

## 9. Proposed contributions and proof obligations

### C1 — Decomposition and contexts for chain LCN MAP

Characterize a useful subclass in which residual world factors **and** admissible
model choices separate. Develop an AND/OR representation that retains shared
probability parameters or separator constraints when they do not separate.
Establish sound combination and context-equivalence theorems, plus conservative
fallbacks to merged subproblems or OR search.

This extends the meaning of an AND/OR state beyond a world assignment. Its value
depends on whether retained constraints make local extrema jointly attainable.
The full-MAP rectangular reduction is the control case, not the contribution.

Proof obligations include factor ownership, compatibility of child witnesses,
internal component independencies, graph recognition, positive-model coverage,
and explicit handling of any supported hard-zero cases.

### C2 — A bound portfolio with correct maximin/maximax directions

Develop incremental event bounds, certified factor/cluster relaxations, and
model-based bounds, and combine them in a search oracle. Formalize the bound
direction for partial and complete assignments under both objectives. Reuse
constraint systems and solve only the endpoint needed for the current purpose.

The partial-event inequalities are elementary foundations. The research value
is in making them sufficiently tight and cheap using chain structure, coupled
constraints, and an adaptive allocation of solver effort. Compare them directly
with precise and credal mini-bucket baselines.

### C3 — Scenario-based maximin AND/OR search

Represent a partial explanation by a vector of scores across a finite set of
feasible original-LCN models. Combine scenario coordinates under their precise
factorizations, while keeping one shared explanation across scenarios. Apply
componentwise dominance only when the future operations and feasible continuation
sets make it sound; preserve explanation backpointers.

At the master objective, maximize the minimum scenario score. Add adversarial
models using the original LCN oracle and certify incumbents using lower bounds
on their worst-case scores. Investigate how shared subproblems and incremental
messages reduce the cost of successive masters.

Adding a scenario adds a coordinate. A vector dominated on the old coordinates
may no longer be dominated on the new one. Therefore old scalar upper bounds can
be retained as conservative bounds, but previously pruned vector frontiers
cannot simply be extended in place and assumed complete. Rebuild them or retain
enough provenance to restore alternatives; version the cache by scenario domain.

For instance, an explanation with profile (0.5,0.5) dominates one with profile
(0.4,0.4). A third scenario extending them to (0.5,0.5,0.1) and (0.4,0.4,0.4)
makes the previously discarded explanation better for the maximin master.

Scenario generation and Pareto/set-valued messages are prior art individually.
The proposed contribution is their correct combination with coupled LCN domains
and reusable AND/OR search. Frontier growth may limit this method to a modest
number of scenarios; quantify that limit.

### C4 — Search that allocates work to the optimality certificate

Unify assignment expansion and probability-oracle refinement in both DFS and
best-first control. A node can remain unresolved because of its unexplored
assignments, a loose relaxation, or an unfinished inner optimization. Study
policies that allocate effort among these causes according to expected root-gap
reduction per unit cost, while preserving fair refinement for completeness.

Establish correctness independently of the heuristic scheduling policy. Compare
the policy with fixed oracle budgets and uniform refinement. Frame improvements
as empirical unless a stronger cost theorem is actually proved.

### C5 — A measured structural complexity account

Study complexity in terms of MAP pseudo-tree depth, world-variable context width,
component cardinalities, constraint/parameter boundary size, and the size of
scenario or set-valued messages. Separate search complexity from oracle cost.

Conventional bounded-width AND/OR complexity applies on the precise/rectangular
baseline with bounded factors. It does not imply that general nonlinear credal
optimization becomes polynomial at bounded component-DAG width. A useful result
may identify conditions under which additional coupling remains localized, plus
families where one logical constraint destroys that advantage.

For the finite scalar baseline, let n be the number of search variables, k their
maximum domain size, h the pseudo-tree depth, and c the largest number of
variables in a cache context. Conservative familiar size bounds are $O(n k^h)$
for the AND/OR tree and $O(n k^(c+1))$ for a cached graph counting value nodes.
For component variables $k$ can be $2^b$, where $b$ is maximum component size. These
counts do not include generating tables or solving their probability extrema.
For the coupled algorithms, track a cost expression of the form

$$
 T=T_{\mathrm{setup}}+
 \sum_{v\ \mathrm{expanded}}T_{\mathrm{expand}}(v)+
 \sum_{q\ \mathrm{oracle\ calls}}T_{\mathrm{oracle}}(q)+T_{\mathrm{cache}},
$$

and account for boundary/profile cardinality in each term. A small context count
does not bound the cost of optimizing a nonconvex continuous boundary object.

Prioritize C1–C2 and two working search engines. Pursue C3 if robust bounds are
the bottleneck and C4 if oracle costs vary substantially. C5 organizes the claims
and experiments even if a broad fixed-parameter theorem proves unattainable.

## 10. Validation and experimental design

### 10.1 Small examples that every implementation must pass

1. **Precise-model reduction:** AOBB/AOBF agree with exhaustive full MAP and
   marginal MAP on tiny BNs and positive chain-factor models.
2. **Rectangular credal full MAP:** local endpoint potentials agree with exhaustive
   selection of feasible local models and assignments for both objectives.
3. **Coupled choices:** reproduce the t(1-t) example; forbid multiplying
   independently attained extrema when their shared parameter differs.
4. **Operator order:** for a two-scenario score table with rows (0.9,0.1) and
   (0.1,0.9), max_y min_s is 0.1 while min_s max_y is 0.9. The latter can be an
   upper bound, not the answer to the first task.
5. **Marginal MAP:** construct a hidden-state example where choosing a different
   model or MAP assignment for each summed state gives an incorrect answer.
6. **Joint versus posterior:** reproduce the probability table in Section 2.2.
7. **Polarity:** a relaxed lower probability and a local NLP minimum must never be
   used as a maximin upper bound and incumbent lower certificate, respectively,
   unless an independent argument establishes the required direction.
8. **Contexts:** construct two prefixes with identical world-separator values but
   different residual coupling; unsafe cache merging must be detected.
9. **Scenario updates:** a vector dominated before a new scenario arrives becomes
   nondominated afterward; the updated master must recover it.
10. **Interruptions and degeneracy:** exercise unresolved leaves, timeouts without
    a solver incumbent, zero-score branches, invalid input, ties, and cache eviction.

The numerical examples above are abstract residual/credal examples, not claims
that a particular `.lcn` file already encodes them. Create corresponding LCN
fixtures where appropriate and verify the graph and LMC after parsing. Adding
a sentence can change the structure; a hand-drawn intended graph is insufficient.

Use exhaustive assignments and independently certified joint optimization as the
reference on tiny LCNs. Cross-check full-joint and cluster formulations only
under verified hosting and reconstruction conditions. The existing local-solver
DFS is a baseline, not a universal exact oracle.

### 10.2 Benchmark families

Use matched families to distinguish structural benefits from easier semantics:

| Family | Purpose |
| --- | --- |
| Precise BNs and separately specified credal DAGs | Recover established AND/OR performance and the exact full-MAP endpoint reduction. |
| Independent complete chain-component tables | Measure the effect of larger state domains without global logical coupling. |
| Noncomplete components with internal LMC restrictions | Test whether component representation preserves the original independencies. |
| Controlled local, across-row, and across-family constraints | Measure where scalar decomposition fails and whether richer contexts recover savings. |
| Long-scope logical constraints | Stress context/cluster growth and conservative fallback behavior. |
| Positive versus deterministic models | Separate proven positive semantics from boundary/support extensions. |
| Full MAP versus 25%, 50%, and 75% MAP-variable subsets | Isolate hidden summation and constrained-order costs. |

Begin with 4–12 atoms for certified references. Scale next to 20, 50, and 100
atoms where component and constraint widths allow. Vary component size, credal
imprecision, evidence, logical-scope size, and coupling density independently
where possible. Record actual scopes and widths, not only atom count.

Reuse [examples](../../examples), [benchmarks](../../benchmarks), and the existing
[generator](../../lcn/benchmarks/generator.py), adding generators around a known
feasible witness for the coupled cases. Include examples from the 2023/2024 MAP
studies when available. Existing filename labels such as “chain” or “polytree”
do not substitute for checking the parsed model and optimization domain.

### 10.3 Baselines and ablations

Compare enumeration with certified scoring; the existing DFS/LDS/SA and approximate
MAP methods; OR branch-and-bound and OR best-first with identical oracles;
rectangular endpoint AOBB/AOBF; credal mini-bucket/DFS methods from [R2] where their
semantics apply; and the proposed AND/OR variants. Evaluate promising assignments
under the same original-LCN target before comparing quality.

Required ablations are AND versus OR space, caching on/off, component versus atom
branching, graph-only versus retained-coupling contexts, event-only versus stronger
bounds, static versus adaptive oracle effort, and DFS versus AOBF versus bounded-
memory hybrid. Use graph-only unsafe combinations solely as labeled diagnostic
counterexamples, never as certified competing solvers.

For maximin additionally compare a single feasible-model bound, several models
using min-of-individual-MAP bounds, the shared-assignment scenario master, and
adversarial scenario generation. For maximax compare coordinate envelopes,
mini-buckets, retained local constraints, and stronger global/cluster relaxations.

### 10.4 Measures, budget, and success criteria

Measure time to first certified incumbent, time to prescribed absolute/relative
gap, final gap, solved fraction, assignment quality, and peak memory. Also record
node expansions, genuine AND decompositions, cache hit rate, context sizes,
scenario/frontier sizes, and all oracle calls and statuses. Separate compilation,
heuristic construction, search, validation, and global-solver time.

Pruning fewer nodes with cheaper bounds can win in wall time. A smaller reported
gap is useful only if both sides are valid for the same original LCN objective.
On unresolved large instances report certificates and feasible explanations;
do not relabel agreement among approximations as ground truth.

Use a pilot to choose manageable instance counts and time limits, then freeze
tuning. A starting design is 10–20 instances per family, wall-time checkpoints
at 1, 10, 60, 300, and 1,800 seconds, and several memory caps. Use multiple seeds
for stochastic initialization; deterministic searches need repetitions for timing,
not artificial stochastic-confidence claims. Keep tuning instances separate.

Success means better time/memory to certified gaps than OR search with the same
bound machinery on a documented family. Report counter-regimes: large coupling
boundaries, expensive inner solves, ineffective scenarios, and determinism that
falls outside the proven representation. A negative result about decomposition
is scientifically useful if supported by a precise condition and counterexample.

## 11. Implementation plan and milestones

New code should live alongside the existing MAP engines, for example in a
proposed `lcn/inference/map/and_or/` package. These are suggested boundaries,
not existing interfaces:

| Module | Responsibility |
| --- | --- |
| `problem` | Objective convention, MAP/evidence/hidden sets, original-LCN identity, semantic support. |
| `structure` | Interaction representation, pseudo-tree, factor ownership, decomposition tests. |
| `bounds` | Certificates, partial-event bounds, relaxations, witnesses, incremental oracle reuse. |
| `context` | Residual identities, boundary objects, cache versions, exact versus bounded entries. |
| `profiles` | Feasible scenarios, shared-assignment score vectors, dominance, backpointers. |
| `dfbnb` / `best_first` | Search control over a shared state and bound contract. |
| `results` | Explanation, score interval, global gap, provenance, costs, unresolved state. |

Use persistent constraints with replaceable objectives where possible. The
arbitrary-conjunction case may force cluster widening, so measure this before
assuming `CredalJT` provides a cheap drop-in oracle. Thread global time/memory
budgets into every solve; a search timeout must not be checked only after an
unbounded inner optimization returns.

| Phase | Tentative duration | Concrete deliverable and gate |
| --- | --- | --- |
| A | Weeks 1–2 | Formal task contracts, literature comparison, exact examples, certified oracle API. |
| B | Weeks 3–4 | OR DFBnB/best-first plus rectangular endpoint AOBB/AOBF; agreement with exhaustive references. |
| C | Weeks 5–7 | Decomposition/context results for a useful coupled subclass; fallback for unsupported separations. |
| D | Weeks 8–10 | Stronger polarity-correct bounds, incremental oracle reuse, scenario-master prototype if justified. |
| E | Weeks 11–12 | Best-first refinement scheduling, bounded-memory hybrid, complete interruption/certificate handling. |
| F | Weeks 13–16 | Frozen benchmark evaluation, ablations, proofs, reproducibility package, and paper draft. |

This is a planning estimate for one research effort, not a promise that an open
theorem will be proved on schedule. If full LCN decomposition remains difficult,
publish the exact supported subclass and use certified OR search with structured
oracles for the general case. If scenario profiles explode, retain the cheaper
feasible-model bounds and report the tradeoff.

The first implementation deliverable should be **certified OR branch-and-bound
and best-first using partial-event bounds, together with ordinary AOBB/AOBF on
the rectangular full-MAP reduction**. This creates a reliable comparison point
for the new AND/OR state representation rather than attributing all gains to
multiple simultaneous changes.
