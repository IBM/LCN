# Global Markov conditions and structured marginal inference beyond chain LCNs

Detailed research plan — 29 September 2026.
Repository baseline: `f6a3ea4f1184967b351b7b5486521d68ce3a6045`.

This document proposes research; it does not report implemented algorithms or
experimental improvements. The literature findings and repository observations
are distinguished from proposed results. Annotated sources and their verification
status appear in [references.md](references.md).

## 1. Research direction and intended contributions

Develop algorithms that exploit **global Markov conditions (GMCs)** to compute
certified lower and upper marginal probabilities in logical credal networks
(LCNs) with structures more general than chain graphs. Study which graphical
factorizations preserve the intended set of probability distributions, how to
exploit their internal structure, and what guarantees remain when factorization
is incomplete or optimization is interrupted.

The immediate starting point is an existing result, not a new conjecture.
Cozman et al. (ISIPTA 2025) [R1] apply Koster's directed–undirected mixed graph
(DUMG) theory [R3] to LCNs, obtain a GMC/Gibbs equivalence under positivity, and
formulate inference as multilinear programming. The earlier IJAR paper [R2]
explains why local and global Markov conditions diverge beyond chain graphs.
Symbolic variable elimination and convex-relaxation branch-and-bound for credal
multilinear programs also predate this project [R8, R9].

The proposed contribution is a **semantically explicit, structurally compiled,
anytime marginal-bounding framework** built on these results. Its main outputs
should be:

1. A verified compiler from LCN assessments and a declared GMC to anterior
   subgraphs, normalized block factorizations, and probability constraints. It
   must preserve directed-edge provenance and all logical couplings.
2. Algorithms that exploit small cliques and separators **inside cyclic
   components**, rather than treating every strongly connected component (SCC)
   as an unrestricted exponential-size table.
3. A hierarchy of outer relaxations that works with zero probabilities and
   admits lazy GMC constraints, together with feasible-model methods that
   provide the complementary inner bounds.
4. Structural conditions for exact separator elimination, query reduction,
   and extensions to ancestral and sigma-Markov models, with explicit limits
   on transferring results between these semantics.

The first paper should focus on DUMGs and finite discrete marginals. General
ancestral factorizations and sigma-separation provide a second research track.
Causal identification, parameter learning, first-order grounding, and the
[separate MAP-search project](../research-map-lcn/research-plan.md) are outside
the initial deliverable.

## 2. Fix the semantic target before choosing an algorithm

### 2.1 Marginal inference and conditional assessments

Let $V$ be a finite set of discrete variables, initially Boolean atoms. Let
$\Delta(V)$ be the joint probability simplex. An assessment
$\ell_i\leq P(\phi_i\mid\psi_i)\leq u_i$ contributes

$$
 \ell_i P(\psi_i)\leq P(\phi_i\wedge\psi_i)
 \leq u_i P(\psi_i).
$$

Unconditional assessments take $\psi_i=\mathrm{true}$. These inequalities impose
no restriction when $P(\psi_i)=0$. This is the initial joint-constraint
convention; a requirement that every conditioning event be possible is a
different model and must be declared. The coupled/uncoupled annotation also
affects graph construction, independently of these numerical inequalities.

For a graph construction and separation rule $s$, define

$$
 \mathcal D_s=\{p\in\Delta(V):p\text{ satisfies every assessment and every CI
 implied by }s\}.
$$

For a query event $Q$, the primary outputs are

$$
 \underline P_s(Q)=\inf_{p\in\mathcal D_s}P_p(Q),\qquad
 \overline P_s(Q)=\sup_{p\in\mathcal D_s}P_p(Q).
$$

Under the zero-permitting convention, the full-joint constraints are polynomial
equalities and closed inequalities on a compact simplex. Thus nonempty
$\mathcal D_s$ attains unconditional extrema. It is generally **nonconvex**
because conditional independence is nonconvex. Do not treat it as a product of
independently selectable local credal sets without an additional theorem.

For posterior queries use regular extension:

$$
 \underline P_s(Q\mid e)=
 \inf_{p\in\mathcal D_s:\,P_p(e)>0}\frac{P_p(Q,e)}{P_p(e)},
 \qquad
 \overline P_s(Q\mid e)=
 \sup_{p\in\mathcal D_s:\,P_p(e)>0}\frac{P_p(Q,e)}{P_p(e)}.
$$

Distinguish an empty model, evidence impossible in every feasible model, a
nonempty model with unresolved optimization, and an ordinary wide interval.
None should silently become the same result `[0,1]`.

### 2.2 Semantic profiles to keep separate

| Profile | Graph and independence interpretation | Role in this project |
| --- | --- | --- |
| Original LCN LMC | Existing LCN construction and local Markov assertions [R2] | Compatibility baseline; retain its present meaning. |
| GMC on the original structure | Anterior moral separation after the original construction collapses opposing arrows | Controlled comparison; not automatically equal to either adjacent profile. |
| Mixed-DUMG GMC | Mixed-structure construction and Koster separation in [R1] | Primary new inference target. |
| Directed ancestral GMC | Directed or hyperedged directed graph, with the d-separation/aFP definitions of [R5] | Second factorization track, after proving the translation. |
| Sigma-GMC | Sigma-separation for a specified directed/hyperedged graph [R5] | Distinct extension for feedback and marginalization. |
| LMC plus a GMC | Intersection of two explicitly specified constraint sets | Optional stronger model; never an equivalence shortcut. |

There is **no general inclusion** between the original LMC and GMC model sets.
Long directed cycles can have vacuous LMCs but nontrivial GMCs. Conversely,
Example 10 of [R2] has assessments inconsistent with its LMC but compatible with
its GMC. Bound widths across these profiles measure different models; a narrower
interval is not by itself an algorithmic improvement.

Nor does a feedback graph automatically license a particular GMC. Section 5 of
[R1], Spirtes and Neal [R7], and the structural-equation results in [R5, R13] discuss
the needed assumptions and counterexamples. Treat the GMC as an explicit modeling
choice unless a structural-model theorem establishes it.

### 2.3 Zero probabilities and factorized subsets

Write $\mathcal D_G$ for the all-support DUMG-GMC target and $\mathcal F_G$ for
distributions satisfying the normalized Gibbs representation in Section 3 and
all assessments. The results in [R1] give

$$
 \mathcal F_G\subseteq\mathcal D_G,\qquad
 \mathcal F_G\cap\{p>0\}=\mathcal D_G\cap\{p>0\}.
$$

Equality on the whole nonnegative domain is not automatic. Consequently,
optimizing only over factorized distributions can miss the true lower or upper
marginal for $\mathcal D_G$. It supplies feasible inner information if its
witnesses are validated, not a general outer enclosure.

Maintain separate declarations for strict positivity, a known support, a floor
$p(x)\geq\epsilon$, and arbitrary nonnegative distributions. A floor changes the
target and can make deterministic assessments infeasible. Adding uniform noise
does not generally preserve either assessments or conditional independence.
The closure of a positive graphical family, its intersection with the assessment
constraints, and the closure of their intersection also need not coincide.

## 3. Factorizations to investigate

### 3.1 Koster's DUMG factorization: the primary route

Construct the mixed-structure directly from the assessments as in [R1]: create
one node per atom; add arrows from each conditioning-formula atom to each
conditioned-formula atom; add undirected links among conditioned-formula atoms
for coupled assessments. Retain opposing arrows introduced by different
assessments. An undirected logical connection and two opposing directed arrows
are different structures. Two opposing arrows are also **not** the bidirected
latent-confounding edge of an ADMG.

An anterior set is closed under predecessors reachable through paths that
respect directed arrows and may traverse undirected edges. Let
$\operatorname{an}_G(S)$ be the minimal anterior set containing $S$.
Moralization connects parents of each undirected path component and then
forgets arrow directions and duplicate edges. The GMC asserts

$$
 X\perp Y\mid Z
 \quad\text{if }Z\text{ separates }X\text{ and }Y
 \text{ in }\left(G_{\operatorname{an}_G(X\cup Y\cup Z)}\right)^m.
$$

One moralization of the whole graph is insufficient. For $A\to C\leftarrow B$,
the appropriate anterior graph for $A\perp B$ contains only $A,B$; the whole
moral graph instead connects them. A compiler that uses only the latter loses
this independence.

Let $J(G)$ be the join-irreducible anterior sets. For $A\in J(G)$, define

$$
 B_A=\langle A\rangle=
 \bigcup_{B\subsetneq A:\,B\text{ anterior}}B,\qquad
 D_A=A\setminus B_A.
$$

With $\operatorname{cl}_G(D)$ denoting $D$ together with its parents and
undirected neighbors, the factorization in [R1, Section 4.2] is

$$
 p(x)=\prod_{A\in J(G)}q_A(x_{D_A}\mid x_{B_A}),
 \qquad
 q_A(x_{D_A}\mid x_{B_A})=
 \prod_{C\in\mathcal C((G_A)^m)}
 \rho_{A,C}\!\left(x_{C\cap\operatorname{cl}_G(D_A)}\right).
$$

The potentials are nonnegative. The products must be valid conditional kernels:
retain their conditional normalization constraints and their allowed scopes.
For positive distributions every boundary configuration has positive mass. For
zeros, distinguish identities on reachable boundary configurations from choices
of conditional versions on null boundaries; require a representability argument
before enforcing any particular completion.

Arbitrary potentials followed by an arbitrary boundary-dependent normalizer are
not a free implementation choice: the normalizer can introduce a scope forbidden
by the theorem. Equally, a product of normalized node conditionals around a
cycle is not this representation. With two binary variables and both
conditionals assigning probability 0.9 to agreement, the naive product sums to
$2(0.9)^2+2(0.1)^2=1.64$.

Theorems 4.1–4.3 of [R1] separate three established facts:

1. GMC can be checked through the family of moralized anterior subgraphs,
   without a positivity assumption.
2. Positive GMC distributions admit the stated Gibbs factorization.
3. Distributions admitting the stated Gibbs factorization satisfy the GMC.

The computational opportunity is the internal clique product in $q_A$, which
may be much smaller than a table over the whole cyclic block. For example,
[R1, Example 4.4] has a block kernel

$$
 q(C,D,E,F\mid A,B)=
 \rho_1(A,B,F)\rho_2(A,C)\rho_3(B,D)\rho_4(C,D)
 \rho_5(D,F)\rho_6(C,E)\rho_7(E,F).
$$

Another kernel in that example is
$q(G,H\mid A,B,C,D,E,F,I,J)=\rho_8(F,G,H,J)\rho_9(G,H,I)$.
These are published factor scopes to reproduce, including their normalization
constraints. The proposed improvement is to exploit and optimize this structure,
while retaining assessments that couple its kernels.

### 3.2 Directed and hyperedged ancestral factorization

Forré and Mooij [R5] define an ancestral factorization property (aFP): each
ancestral marginal factors over complete subsets of its appropriate moral
graph. Their Corollary 3.6.9 gives dGMP $\Leftrightarrow$ aFP for positive
densities, or under their graph's perfect-elimination-order condition.
The latter is a property of the original graph under their definitions; making
some auxiliary graph chordal by adding edges does not establish it.

For ordinary directed graphs, their positive-density Lemma 3.6.14 yields
normalized SCC kernels with further internal clique factorizations:

$$
 p(x)=\prod_{S\in\operatorname{SCC}(G)}
 p(x_S\mid x_{\operatorname{pa}(S)\setminus S}),\qquad
 p(x_S\mid x_{\operatorname{pa}(S)\setminus S})
 =\prod_C k_{S,C}(x_C),
$$

where the permissible $C$ lie within $S\cup\operatorname{pa}(S)$ and are
complete in the moralized ancestral graph specified by the lemma.

Research task: identify the exact overlap with the DUMG compiler on directed
inputs and design a common normalized-block interface. For hyperedges, implement
the source's graph and marginalization definitions, rather than interpreting
existing LCN undirected edges as hidden common causes. Establish soundness and
completeness separately for each supported profile.

### 3.3 Decomposable moral/anterior regions with support-aware reconstruction

For a decomposable undirected graphical model, consistent clique marginals can
be assembled along a junction tree using full separator distributions [R11].
The positive case has the familiar expression

$$
 p(x)=\frac{\prod_C\mu_C(x_C)}{\prod_{(C,D)\in T}\mu_{C\cap D}(x_{C\cap D})}.
$$

At zero separator mass, use a conditional-kernel construction: consistency
forces the corresponding clique mass to zero, and arbitrary normalized
conditional versions can be chosen there. Do not numerically evaluate $0/0$.
This produces a joint Markov to the decomposable graph. It does not establish
that it satisfies every GMC of an arbitrary original mixed graph.

Research task: characterize families of anterior regions whose local
factorizations and separator constraints reconstruct a joint satisfying the
**entire** target GMC. This could give useful exact zero-permitting subclasses.
For arbitrary graphs, triangulated regions remain useful as outer representations
when every target distribution projects into them. Fill edges cannot simply
erase the original GMC assertions in a claimed exact model.

### 3.4 Sigma-separation and acyclification: a distinct second track

Sigma-separation handles cycles differently from ordinary d-separation and has
useful stability under the graph marginalization of [R5]. Appropriate
acyclification translates its separation statements into d-separation statements.
This is a route to a graph oracle, not by itself a theorem that arbitrary LCN
assessments compile into independent DAG kernels.

Bongers et al. [R13] give a later journal treatment of these foundations. Their
Appendix A, Proposition A.19 makes the sigma/d-separation equivalence under
acyclification explicit; Theorem A.21 derives sigma-GMC for SCMs uniquely
solvable with respect to each SCC. This supplies a concrete assumption to test
when a feedback application motivates the model. An arbitrary LCN has no such
solvability certificate merely by having a cyclic graph.

In the positive, ordinary-directed setting of [R5, Remark 3.6.15], sigma-GMC
supports an SCC kernel product but does not in general supply the finer internal
d-GMC clique restrictions. This predicts a real tradeoff: a sigma model may
retain useful decomposition between components while losing the within-component
sparsity that makes the primary DUMG route attractive.

Research tasks are to translate assessments without losing their joint meaning,
compare direct sigma-CI constraints with the appropriate acyclified graph, and
measure which separators survive. Ordinary GMCs on observed variables do not
automatically characterize distributions realizable by latent structural models.
Marginal-model and nested/fixing factorizations can impose additional restrictions;
they require a separately defined semantic project, not an implicit GMC extension.

### 3.5 Practical comparison

| Representation | Useful structure | Assumptions or unresolved issue | First algorithm |
| --- | --- | --- | --- |
| DUMG Gibbs blocks | Anterior blocks, small internal cliques | GMC equivalence under positivity; support and potential bounds otherwise | Compiled structured optimization |
| Directed/HEDG aFP | Ancestral marginals and SCC kernels | Exact graph definitions and aFP equivalence hypotheses | Shared compiler after translation proof |
| Decomposable anterior regions | Joint separator messages | Reconstruction must preserve all original GMCs | Exact elimination on proved subclasses |
| Sigma SCC kernels | Separation between feedback components | Distinct semantics; internal sparsity may disappear | Direct sigma-CI relaxation first |
| Direct GMC polynomial model | Any graph oracle and arbitrary support | Potentially exponential constraints and joint state space | Reference oracle and adaptive outer hierarchy |

## 4. Mathematical and computational reference models

### 4.1 Direct full-joint GMC oracle

For small instances enumerate joint states and every graph-implied CI needed
by the chosen profile. For disjoint variable sets $X,Y,Z$, encode

$$
 P(x,y,z)P(z)=P(x,z)P(y,z)
 \quad\text{for every }x,y,z.
$$

These equations remain valid at $P(z)=0$. Do not replace all set-valued CIs with
an unproved pairwise basis in the presence of zeros. Start with exhaustive
separation enumeration on very small graphs, then prove reduced bases for
specific classes. The full-joint model is a polynomial optimization problem
with a compact feasible set under Section 2.1, not a linear program. Its explicit
representation is still exponential in the number of variables.

Use a global solver's objective certificates, analytic examples, or exact
arithmetic where feasible. A local NLP optimum is a feasible candidate only
after validation; it is not a proof of the global marginal endpoint.

### 4.2 Output contract and bound directions

For $f(p)=P_p(Q)$, maintain an outer relaxation $\mathcal R\supseteq\mathcal D_s$
and a nonempty collection of validated feasible witnesses
$\mathcal W\subseteq\mathcal D_s$. Return four numbers:

$$
 L_{\rm out}\leq\underline P_s(Q)\leq L_{\rm in},\qquad
 U_{\rm in}\leq\overline P_s(Q)\leq U_{\rm out}.
$$

Here $L_{\rm out}$ is a certified lower bound on $\inf_{\mathcal R}f$,
$U_{\rm out}$ a certified upper bound on $\sup_{\mathcal R}f$,
$L_{\rm in}=\min_{\mathcal W}f$, and $U_{\rm in}=\max_{\mathcal W}f$.
An incumbent from minimizing over a relaxation need not be a target-feasible
witness. A maximization objective value is likewise not necessarily a certified
upper bound. Record solver primal and dual roles explicitly.

Every result includes the semantic profile, graph-construction version, support
policy, evidence convention, dropped/relaxed constraints, feasibility status,
certificate tolerances, and the endpoint gaps
$L_{\rm in}-L_{\rm out}$ and $U_{\rm out}-U_{\rm in}$ when available.
Numerical residuals alone are not mathematical feasibility certificates; use
exact/rational witnesses or validated residual correction where possible, and
label ordinary floating-point candidates as provisional.

## 5. Proposed algorithms and work packages

### WP0 — GMC graph oracle and factorization compiler

**Input:** LCN assessments with provenance, a semantic profile, and a support
policy. **Output:** a graph, separation oracle, factor/region scopes, and a
constraint ledger explaining how every assessment and GMC obligation is handled.

Implement the mixed-DUMG constructor separately from the existing original
structure constructor. Retain both directed and undirected edges where the
definition allows them. Audit overlapping conditioning/conditioned atoms and
self-loops, and map the parser's annotation to the published coupled/uncoupled
definition with explicit examples.

Avoid enumerating all anterior sets in the factor compiler. Replace undirected
links by arcs in both directions **for reachability computation only**, contract
SCCs, and examine the ancestor ideals of the resulting partial order. Principal
ideals are the join-irreducibles of a finite ideal lattice. Prove that this
construction matches Koster's anterior convention and gives exactly $J(G)$,
with $D_A$ the corresponding reachability block. This is an application of
established order theory, not a claim to invent the ideal-lattice theorem.
Moralization must still use the original edge types.

Compile each normalized block's internal clique scopes, then attach every
assessment using its **entire** atom scope. Scope locality does not guarantee
parameter locality: a constraint on $P(X)$ in a family $(X,\operatorname{pa}(X))$
can depend on the parents' distribution and couple several kernels. Record
couplings in the optimization graph instead of independently projecting them
onto local intervals.

Deliverables: small-graph separation truth tables; compiler equivalence tests;
the factorization in [R1, Example 4.4]; reduction to applicable chain/DAG cases;
and a report of actual factor scopes, boundary scopes, and assessment couplings.

### WP1 — Structured global optimization over normalized GMC factors

Build a sum-product circuit for each query and assessment event using the
compiled factors. Introduce auxiliary variables for intermediate contractions,
reuse shared subexpressions across marginals, and retain the normalization and
logical constraints in the same optimization problem. In contrast to solving
each SCC as a full joint table, elimination can exploit its internal clique
width.

This extends the established credal symbolic-elimination approach [R8, R9] to
the normalized, overlapping anterior factors of [R1]. The research questions
are the scope-correct compilation, tractable parameter bounds, coupling-aware
relaxations, and measured benefit on cyclic LCNs.

For bounded variables $a\in[a_L,a_U]$, $b\in[b_L,b_U]$, replace a product
$w=ab$ by its McCormick envelope [R10]:

$$
\begin{aligned}
w&\geq a_Lb+b_La-a_Lb_L,&
w&\geq a_Ub+b_Ua-a_Ub_U,\\
w&\leq a_Ub+b_La-a_Ub_L,&
w&\leq a_Lb+b_Ua-a_Lb_U.
\end{aligned}
$$

Lift longer products sequentially, tighten variable ranges, and add valid
products of existing constraints when useful. A spatial branch-and-bound solver
then refines these envelopes. Compare depth-first control, best-bound control,
and a memory-limited hybrid; these branch on continuous model/auxiliary domains,
not on MAP assignments.

**Parameter-bounding gate.** Clique potentials are not probabilities and cannot
simply be bounded by one. Scaling freedoms may yield unbounded potential
representations of perfectly bounded distributions. Investigate gauges on
proved subclasses, bounded normalized-kernel variables with consistency
constraints, and bounded marginal variables as an alternative. Every range
restriction needs a representation theorem. If none is available, WP2 remains
the general certified route; a truncated potential model cannot certify outer
bounds for the full target.

Use the factor route as exact only on an audited model class where both
representation equivalence and the optimization certificate apply. On a
general zero-permitting GMC model it can still search for feasible witnesses.
Multi-start coordinate optimization and ApproxLP ideas [R12] are useful there,
but updates must preserve the normalized block and all coupled assessments.

**Research hypotheses:** internal sparse factorization reduces optimization size
on large sparse SCCs; constraint-aware elimination beats an ordering based only
on the visual graph; branching on influential normalization/separator variables
improves endpoint-gap reduction relative to generic branching. Each hypothesis
requires ablation against the same feasible domain and solver settings.

### WP2 — Anterior-region outer relaxations with full separator distributions

Build a region family containing all query and assessment scopes, selected
anterior moral cliques, and useful separators. For each region $R$, introduce a
normalized nonnegative table $\mu_R(x_R)$. Enforce equality of the full marginals
on shared separators. Missing large assessment scopes must be added or their
constraints explicitly relaxed; never silently reinterpret a formula locally.

The initial linear model retains assessments and marginal consistency but
drops some or all GMC equations. Every true joint distribution projects into
this model, so it gives outer marginal bounds. Small regions need not reconstruct
a joint, and reconstructed joints need not satisfy the target GMC; neither
property is required for this initial outer guarantee.

Strengthen the model with hosted GMC equations, represented using bounded
marginals in $[0,1]$. For example, a CI residual is

$$
 r_{x,y,z}=\mu_{XYZ}(x,y,z)\mu_Z(z)
             -\mu_{XZ}(x,z)\mu_{YZ}(y,z).
$$

Enforce equality of the two lifted products with McCormick inequalities and
refined bounds. This is a sound linear outer relaxation of the polynomial CI,
not exact enforcement of independence. Use spatial branching or stronger
relaxations when the envelope is too weak. Add appropriate region tables when
the required joint marginal is absent.

Construct nested levels $\mathcal R_0\supseteq\mathcal R_1\supseteq\cdots$
in a common lifted representation or prove the corresponding projection
inclusions. Increasing region size, adding CIs, and tightening product domains
must preserve every target joint. Optimal lower outer bounds then increase and
optimal upper outer bounds decrease. With interrupted solves, retain the best
certificates across all previous levels to preserve the reported monotonicity.

At the limit, one full-joint region plus all GMC equations recovers the reference
polynomial model. Exactness additionally requires globally resolving its
nonconvexity. Merely reaching the largest region while keeping loose product
envelopes does not make the LP exact.

For a proved decomposable subclass, replace relaxation by exact separator
elimination. A separator message should represent the feasible relation among
its **joint distribution**, remaining shared model variables, and query-relevant
quantities. It may be a constrained set or an outer polyhedral approximation,
not just an interval for each atom. Even a tree of regions does not automatically
make credal set propagation small: nonrectangular feasible sets can have many
vertices or require nonconvex descriptions.

**Expected contribution:** a target-preserving hierarchy that covers zeros,
retains logical coupling, and can exploit GMC restrictions absent from ordinary
local-consistency LPs. General marginal-polytope relaxations [R11] and McCormick
lifting are prior art; the anterior/GMC selection, correctness conditions, and
empirical tradeoffs constitute the proposed extension.

### WP3 — Lazy GMC constraint generation and adaptive region selection

Enumerating all GMC assertions can overwhelm even a modest graph. Start from
WP2 and add assertions selectively:

```text
compile graph, assessments, initial regions, and query
repeat until the time/memory budget or certified endpoint tolerance is reached:
    solve/refine the current outer model for each unresolved endpoint
    retain its valid global objective certificates
    try to reconstruct and validate target-feasible models for inner bounds
    inspect available joint marginals for violations of graph-implied CIs
    select a violated CI, a missing region, or a product-domain refinement
    add the valid constraint/region, or branch to refine the relaxation
return outer bounds, validated inner bounds, and unresolved obligations
```

Separate two oracles: the graph oracle decides whether a CI belongs to the
chosen GMC; the probability oracle measures its violation in a candidate table.
A singleton message cannot evaluate a multivariate CI residual. Start with
minimal separators of query-relevant anterior graphs, then expand the candidate
pool when progress stalls.

Candidate priorities can combine violation magnitude, relaxation dual
information, circuit sensitivity, and estimated state-space cost. This is a
heuristic for which valid refinement to add, not a certificate of completeness.
Adding a violated nonlinear equality requires its exact global treatment or a
sound relaxation; an arbitrary tangent cut to a nonconvex CI is not valid.

Prove outer-bound preservation after every step. Claim convergence to the exact
reference only when a complete constraint/region schedule and a convergent
global optimization scheme are in place. Exhaustive small-graph separation is
the first completeness oracle; a polynomial-time complete oracle is a research
question, not an assumption. If a sampled CI search finds no violation, report
that search outcome without declaring exactness.

### WP4 — Query-aware projection and conditional probabilities

Normalized leaf blocks can disappear from a sum-product expression when their
variables are not queried. They cannot necessarily disappear from the feasible
model: an assessment on a descendant can constrain an ancestor's marginal.
For a simple directed example, encode
$P(B=1\mid A=1)=1$, $P(B=1\mid A=0)=0$, and $P(B=1)=0.8$.
These imply $B=A$ almost surely and $P(A=1)=0.8$, even for a query mentioning
only $A$. Dropping the descendant's assessments loses that restriction.

Develop an elimination rule with an extension condition: every retained feasible
model must extend to eliminated variables while satisfying all removed
assessments and GMCs. When this cannot be proved, retain their projected
constraints or a sound outer approximation. Build a relevance graph over
factors **and assessments**, and compare it with graph-ancestor pruning alone.
Distinguish algebraic simplification of the objective from feasible-set
projection in the compiler's ledger.

For posterior bounds, [R1] proposes a reciprocal lift $tP(e)=1$ with objective
$tP(Q,e)$. This is still a nonlinear optimization problem. If $P(e)$ can tend
to zero, $t$ is unbounded. Closedness of the lifted feasible set alone does not
prove attainment; neither does it supply the finite bounds needed for McCormick
relaxations. If a genuine model assumption supplies $P(e)\geq\beta>0$, then
$t\leq1/\beta$ is justified. A numerical floor introduced for convenience is
a different inference target.

Investigate division-free endpoint certification on the closed joint domain.
Provided some feasible model has $P(e)>0$, for any candidate lower bound $\tau$,

$$
 \tau\leq\underline P_s(Q\mid e)
 \quad\Longleftrightarrow\quad
 \inf_{p\in\mathcal D_s}\{P_p(Q,e)-\tau P_p(e)\}\geq0.
$$

Similarly, $\overline P_s(Q\mid e)\leq\tau$ iff the corresponding supremum
is at most zero. Models with $P(e)=0$ contribute zero and cause no false
counterexample. A validated negative/positive witness refutes the proposed
bound in the respective direction. These tests enable bisection without an
evidence floor, using the unconditional polynomial oracle. Their difficulty is
certifying a zero optimum numerically; analyze tolerances and retain unresolved
brackets instead of interpreting solver noise as a proof. Strictly positive
semantic profiles require their own infimum/closure analysis.

### WP5 — Support-sensitive exactness and alternative GMCs

Use the direct GMC hierarchy as the baseline for models with logical zeros.
Investigate sufficient support-localization conditions inspired by Moussouris
and Geiger–Meek–Sturmfels [R6], checking them on all relevant anterior models,
not only the full moral graph. Separately explore the genuine perfect-order
subclasses of [R5]. A graph condition, support condition, and positivity floor
are distinct hypotheses and should yield distinct theorems and result labels.

Develop a hybrid in which a block uses a compact factor representation only
when a **conditional** representation theorem justifies it; other blocks use
explicit regional marginals and GMC constraints. The combination needs a
gluing proof for cross-block CIs and assessments. Local factorability alone
does not establish the validity of the hybrid.

Finally, apply the same direct-CI machinery to a specified sigma-GMC before
attempting an optimized sigma kernel compiler. This isolates the semantic
question from solver behavior. Compare within-profile algorithms separately
from across-profile modeling sensitivity.

## 6. Proposed theoretical results and novelty tests

| Target result | Claim to establish | What must not be claimed without more work |
| --- | --- | --- |
| T1: DUMG compilation | Principal anterior blocks and compiled normalized scopes represent exactly the published factorization; all assessments remain intact | A new GMC factorization theorem, already supplied by [R1, R3] |
| T2: Exactness domains | Precise positivity, support, and graph hypotheses under which the compiled feasible set equals the target | Equality for arbitrary logical zeros |
| T3: Outer hierarchy | Projection containment at every level, monotone optimal bounds, full-reference limit | LP exactness merely from local consistency or large regions |
| T4: Lazy refinement | Every added CI/region/domain refinement preserves the target; completeness under an explicit exhaustive schedule | Completeness of a heuristic violation search |
| T5: Separator elimination | Conditional feasible-set gluing and reconstruction on a useful structural subclass | Independent selection of singleton interval endpoints |
| T6: Query projection | A checkable sufficient extension criterion and preservation of marginal bounds | Dropping every nonancestor assessment |
| T7: Structural complexity | Circuit/relaxation size governed by actual factor, assessment, and separator scopes | Polynomial-time exact credal inference solely from small graphical treewidth |
| T8: Posterior certification | Sound division-free comparisons and convergence conditions for posterior brackets | Attainment from closedness or an unstated evidence floor |

For T7, distinguish compilation/evaluation cost from global optimization cost.
For maximum variable cardinality $d$, elimination intermediates of width $w$
typically require $O(d^{w+1})$ storage per table, multiplied by the number of
compiled events and shared intermediates. Construct $w$ from the factor and
assessment hypergraph, including normalization constraints. Separately report
region width, separator width, number of retained CIs, and spatial search
complexity. A small SCC count or a DAG condensation is not a complexity bound.

Before publication, update the literature search around DUMG inference,
reciprocal/cyclic graphical models, credal multilinear solvers, polynomial CI
relaxations, support-aware graphical models, and sigma-Markov inference.
The present review identifies a credible program; it does not establish a
priority claim against every later paper or implementation.

## 7. Repository integration and constraints on reuse

The current repository offers useful building blocks, but their present graph
semantics and exactness claims need to be audited for the new target.

| Existing location | Useful component | Required audit or change in a future implementation |
| --- | --- | --- |
| [model.py](../../lcn/core/model.py), `build_structure_graph` | Atom/formula scopes and annotations | Step 3 collapses opposing arrows; construct mixed-DUMG edges from assessments instead. |
| [mixed_graph.py](../../lcn/core/mixed_graph.py) | Separate directed and undirected stores, simultaneous edge types | Preserve provenance; verify induced graphs, copies, self-loops, and moralization. |
| [exact.py](../../lcn/inference/marginal/exact.py) | Full-joint event expressions and solver interfaces | Replace LMC generation with the selected GMC oracle; audit certificate and evidence handling. |
| [sccp.py](../../lcn/inference/marginal/sccp.py), [sccp_ariel.py](../../lcn/inference/marginal/sccp_ariel.py) | SCC construction and local optimization | A condensation DAG need not induce a tree factor graph; singleton messages and local solves are not sufficient for global exactness. |
| [junction_nlp.py](../../lcn/inference/marginal/cn/junction_nlp.py) | Region tables, separator consistency, hosting, SCIP integration | Existing construction is chain-oriented; every unhosted GMC must be proved implied, explicitly relaxed, or hosted in a larger region. |
| [structured_consistency.py](../../lcn/inference/utils/structured_consistency.py) | Feasibility orchestration | Its current chain compiler does not establish arbitrary-GMC feasibility equivalence. |
| [strong_extension_exactness.tex](../research-chain-lcn/strong_extension_exactness.tex) | Examples of assessments coupling family kernels | Use its counterexamples; independently prove any general theorem transferred to the new profile. |

Two specific code risks deserve regression cases. First, sentence classification
in the SCC path uses the component of the first conditioned-formula atom for
linking assessments; full multi-atom scopes need independent validation.
Second, the junction path contains an “inert” rationale for dropping some
unhostable LMC assertions. Such an assertion cannot be dropped in a new exact
GMC backend without a profile-specific implication proof.

Proposed future modules under `lcn/inference/marginal/gmc/` are `semantics.py`,
`compile.py`, `joint_reference.py`, `regions.py`, `structured_global.py`,
`refine.py`, and `certificates.py`. Keep model construction separate from solver
choice, so the same semantic target can be compared under different algorithms.
Store constraint provenance and dropped-constraint counts in machine-readable
results. This research-plan task creates documentation only.

## 8. Experimental design

### 8.1 Mandatory semantic and algebraic fixtures

1. **Collider:** $A\to C\leftarrow B$ checks anterior restriction before
   moralization and detects loss of $A\perp B$.
2. **Positive directed four-cycle:** on binary states use
   $p(x)=2^{-4}[1+\tfrac12(-1)^{x_1+x_2+x_3+x_4}]$. Its minimum mass is $1/32$;
   it violates the opposite-node GMC CI with maximum polynomial residual
   $1/128$. It separates a positive joint allowed by a vacuous cycle LMC from
   the cycle GMC.
3. **Nonfactorizable zero-support cycle:** [R5, Example 3.6.10], attributed
   there to Lauritzen, assigns mass $1/8$ to
   `0000, 1000, 1100, 1110, 0001, 0011, 0111, 1111` and zero elsewhere.
   It satisfies $X_1\perp X_3\mid X_2,X_4$ and
   $X_2\perp X_4\mid X_1,X_3$ but does not factor over cycle edges.
   Positive states `0000`, `0011`, and `1110` force every edge factor needed by
   `0010` to be positive, contradicting its zero mass. The direct GMC route
   must accept this model; an unrestricted-GMC compiler must not exclude it.
4. **Opposing arrows versus undirected links:** reproduce the distinction in
   [R1, Examples 4.3–4.4], including the published internal factor scopes.
5. **Cyclic conditional-product failure:** the 1.64 normalizer in Section 3.1
   prevents accidentally using a Bayesian-network product around a cycle.
6. **Separator correlation:** fair binary $A,B$ can have all mass on `00,11`
   or on `01,10`. The singleton marginals agree, but the joint separator
   distributions conflict. Independent endpoint messages cannot certify gluing.
7. **Condensation diamond:** $A\to B,A\to C,B\to D,C\to D$ is a DAG, but its
   ordinary family factor graph has a loop. SCC contraction alone does not
   justify tree propagation.
8. **Constraints on eliminated descendants:** the $B=A$ example in WP4 checks
   feasible-set projection rather than query-only pruning.
9. **Inconsistency and evidence:** inconsistent assessments, an impossible
   evidence event, evidence approaching zero, and feasible deterministic cases
   exercise distinct statuses and posterior bracketing.

During preparation of this plan, fixtures 2, 3, and 5 were checked with exact
rational arithmetic in a temporary script. These checks validate the displayed
examples, not an implementation of the proposed algorithms.

### 8.2 Instance families

Use matched positive and zero-permitting families, with the semantic profile
fixed within each algorithm comparison:

- DAG and valid chain cases as compatibility controls; include family-local
  formulas that nevertheless couple parent distributions and cross-family
  assessments.
- Directed rings, sparse feedback ladders, and mixed semidirected cycles with
  one large SCC but small internal moral width.
- Networks of cyclic blocks with variable boundary size and condensation
  topology, including diamonds and multiply connected component graphs.
- Dense cyclic blocks as a stress case where factorization may offer little
  advantage; vary logical-sentence scope independently of graph density.
- Decomposable anterior structures and graphs satisfying the actual
  perfect-order hypotheses of [R5], as candidate exact subclasses.
- Deterministic logic and controlled support holes, including the non-Gibbs
  fixture; separately constructed sigma-GMC examples for the secondary track.
- Repository application instances, classified by the chosen construction
  rather than assumed to be valid beyond-chain GMC benchmarks.

For generated instances, keep a known witness and verify it against all
assessments and the chosen GMC. Sampling arbitrary positive clique potentials
and normalizing the whole joint need not generate a valid DUMG model with its
required normalized block structure. Reject or repair invalid generators.
Use held-out feasible models to form interval assessments without accidentally
changing the graph through newly introduced formula scopes; record the resulting
graph and verify it again. Include deliberately inconsistent instances separately.

### 8.3 Baselines and ablations

Compare against the direct full-joint **GMC** polynomial formulation on small
instances; the published factorized multilinear formulation with generic global
optimization on the same representation domain; a sentence-only LP; region
consistency without GMC refinements; and existing SCC/ARIEL/junction/ApproxLP
methods where their semantics can be aligned. Original-LMC runs belong in a
separate semantic-sensitivity comparison. Heuristic outputs without certificates
must be labeled as such.

Ablate internal SCC factorization, shared circuits, normalization tightening,
constraint-aware elimination orders, region size, full versus singleton
separator information, lazy versus eager CIs, adaptive versus fixed CI order,
support-aware switching, valid query projection, witness search, and search
control. An intentionally unsound singleton or dropped-constraint variant is a
failure demonstration, never a certified baseline.

### 8.4 Metrics and reproducibility

Measure endpoint gaps separately from interval width. A wide exact credal interval
is uncertainty in the model, not solver error. Record time to certified gaps
$10^{-2}$ and $10^{-3}$, final gap at fixed time, peak memory, compilation time,
number of LP/NLP/global subproblems, regions/CIs added, and certificate status.
Report graph size, SCC sizes, actual elimination and separator widths, sentence
scope, support policy, and evidence probability ranges.

On solved small instances, check outer containment and validated witness
feasibility against the reference. On larger instances, report certificates and
unresolved gaps without calling the best observed objective ground truth. Use
paired instances, fixed seeds and resource limits, and separate parameter tuning
from evaluation. Start with at least 20 generated instances per tractable design
cell, then adjust based on variance and compute budget; publish all seeds,
rejected-instance reasons, solver versions, tolerances, and timeout results.

### 8.5 Falsifiable success criteria

- **Correctness gate:** no containment violation on mandatory fixtures or
  certified small instances; exactness claims restricted to proved domains.
- **Structural benefit:** on sparse large-SCC families, the compiled approach
  improves time/memory at the same endpoint tolerance over full SCC tables and
  a generic version of the same factorized program.
- **Adaptive benefit:** at equal resources, lazy refinement reduces endpoint
  gaps more effectively than the initial LP and competitive fixed schedules.
- **Support benefit:** the direct/hybrid route handles non-Gibbs GMC examples
  without excluding feasible joints; any compact support theorem has a
  checkable hypothesis and a demonstrated useful family.
- **Honest negative result:** dense scopes, huge separators, poor potential
  bounds, or weak product relaxations may erase the advantage. Quantify these
  boundaries instead of hiding them through semantic restrictions.

## 9. Milestones, decision gates, and publication scope

The schedule below is a proposed 20-week program for one researcher with access
to the existing inference infrastructure. It is an estimate, not a claim about
the duration of unresolved proofs.

| Weeks | Work and reviewable deliverable | Decision gate |
| --- | --- | --- |
| 1–2 | Source comparison, semantic-profile specification, exact fixtures, full-joint GMC prototype design | Reproduce [R1] and [R5] counterexamples; settle annotation/edge translation. |
| 3–5 | Anterior-block compiler and direct reference implementation | Prove scope/normalization correspondence; retain every assessment. |
| 6–8 | Initial region LP and globally treated CI lifts; certificate interface | Establish projection containment and correct solver bound directions. |
| 9–11 | Structured factor optimization and bounded-parameter study | Use it for exact claims only if representation/bounding gates pass; otherwise retain it for witnesses and prioritize regions. |
| 12–14 | Lazy refinement, full separator messages, query-projection rules | Demonstrate improvement over fixed-region baselines on matched targets. |
| 15–17 | Support subclasses, posterior certification, hybrid proof attempts | Resolve zero/evidence fixtures and state remaining incompleteness explicitly. |
| 18–20 | Controlled experiments, ablations, manuscript and artifact | Publish supported claims; defer broad HEDG/sigma compilation if translation proofs remain open. |

The first concrete deliverable should be a **semantic and certification report**:
the same small LCN evaluated under original LMC, mixed-DUMG GMC, and factorized
GMC subsets, with graph scopes, exact CI residuals, feasible-set distinctions,
and correctly directed endpoint bounds. This precedes performance claims.

A strong first paper would combine T1–T4 with a practical anytime outer hierarchy
and sparse-cycle experiments. A second paper could address T5–T8, support-aware
exact subclasses, and sigma/ancestral extensions. If the general compact-factor
route remains numerically difficult, a correct zero-aware GMC relaxation with
adaptive separator constraints is still a substantial, clearly scoped result.

## 10. Reading order

Read [R1] Sections 4–5 and [R2] Section 6 first, then verify Koster's original
definitions and theorem [R3] before finalizing the compiler proof. Read [R5]
Sections 2.7–2.8 and 3.6–3.8 for alternative separation and factorization rules,
and [R13] Section 6 and Appendix A for their later SCM treatment.
Use [R6] for support boundaries, [R8–R10] for optimization precedents, and
[R11–R12] for separator representations and feasible-model search. The
[annotated references](references.md) identify which original full texts were
inspected and which claims currently rely on a published secondary treatment.
