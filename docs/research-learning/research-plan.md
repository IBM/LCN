# Learning logical credal networks from data

Detailed research plan — 29 September 2026.
Repository baseline: `f6a3ea4f1184967b351b7b5486521d68ce3a6045`.

This is a proposal for research and implementation, not a report of completed
learning algorithms. Literature findings, proposed contributions, and proof
obligations are distinguished below. Source notes and verification status are
in [references.md](references.md). The existing [GMC research plan](../research-gmc/research-plan.md)
and [sampling plan](../research-sampling/research-plan.md) provide complementary
inference directions; neither supplies a learning guarantee by itself.

## 1. Research direction

Develop methods for **parameter learning, structure learning, and joint
structure–parameter learning of LCNs**, with a particular emphasis on learning
models that support efficient, certified lower and upper probability inference.
Combine LCNs' logical assessments with statistical estimation of uncertainty and
the structural restrictions used by probabilistic circuits.

The central challenge is to learn three related objects without conflating them:

1. A representative distribution that predicts data well.
2. A set of distributions expressing a stated source of uncertainty.
3. A logical and computational structure that makes the required queries
   interpretable and, where possible, tractable.

The original LCN paper [R1] already uses empirical information to set rule
probabilities in an application. Learning local credal sets and uncertainty
over network structures is established research [R4–R7]. LearnSPN, LearnPSDD,
Strudel, and Einsum Networks provide structure/parameter learning for tractable
precise models [R10–R13]. Credal SPNs and credal sentential decision diagrams
(CSDDs) already extend circuits to imprecise probabilities and give important
tractability results [R15, R16].

LearnCSPN [R19] already learns credal circuit structure and parameters from
incomplete data. Circuit-based credal-network relaxations [R20], learning PCs
with domain constraints [R21], and newer parameter optimizers [R22] are also
direct predecessors. The proposed algorithms must be compared with these,
rather than positioned only against precise circuits with post-hoc intervals.

Accordingly, the contribution should be **LCN-specific learning with explicit
statistical and inference guarantees**, not the first use of data in LCNs, the
first learned credal model, or the first credal circuit. The proposed program
has four main outputs:

- Estimators of LCN probability assessments and compatible credal parameters,
  including a clear treatment of coherence, missing data, and sample scarcity.
- Search over sentence structures and their induced Markov assumptions, with
  uncertainty over structures when the data do not select one reliably.
- A joint learning algorithm that trades predictive fit, logical complexity,
  uncertainty quality, and measured inference cost.
- A hierarchy of learnable LCN fragments with query-specific tractability
  certificates, including a rigorously characterized interface to circuits.

Begin with finite discrete data, complete observations, and the original LCN
semantics. Add incomplete data and alternative GMC semantics in later stages.
Continuous variables, first-order relational structure, causal discovery, and
deep neural perception are extensions rather than prerequisites.

## 2. What is being learned?

### 2.1 Separate sentences, probabilities, and the Markov interpretation

Let the observed atoms be $V$, and let $D=\{x^{(1)},\ldots,x^{(N)}\}$ be
initially independent and identically distributed complete observations.
Allow known hard domain constraints $H$ and, later, explicitly modeled missing
values or latent variables $Z$.

An LCN structure is a finite sentence schema

$$
 S=\{(\phi_j,\psi_j,\tau_j):j=1,\ldots,m\},
$$

where $\phi_j,\psi_j$ are formulas and $\tau_j$ records the annotation that
affects graph construction. An unconditional sentence has
$\psi_j=\mathrm{true}$. The numerical specification
$\eta=\{(\ell_j,u_j)\}_{j=1}^m$ supplies the bounds. Let $s$ identify the graph
construction and Markov condition. Then

$$
 \mathcal D_{S,\eta,s}=
 \left\{p\in\Delta(V):
 \begin{array}{l}
 p(H)=1,\quad
 \ell_jp(\psi_j)\leq p(\phi_j\wedge\psi_j)\leq u_jp(\psi_j),\ \forall j,\\
 p\text{ satisfies the Markov assertions induced by }(S,s)
 \end{array}\right\}.
$$

This joint-inequality convention leaves a conditional assessment unrestricted
when its conditioning event has probability zero. It must be recorded in learned
models. A circuit representation has additional parameters $\theta$ and local
credal sets $K$; neither $\theta$ nor $K$ is automatically equivalent to the
sentence bounds $\eta$.

**Parameter learning** fixes $S,s,H$ and estimates a representative model and
$\eta$ or a justified alternative parameter-set representation. **Structure
learning** chooses formulas, conditioning scopes, annotations, and graph/circuit
structure. **Joint learning** interleaves these choices. The primary structure
search fixes $s$; comparing original LMC, chain-compatible factorization, and
mixed-DUMG GMC is a separately labeled modeling comparison [R2, R3].

### 2.2 LCN syntax is part of the structure

A vacuous numerical statement can still change an LCN's graphical semantics.
With two otherwise isolated fair atoms $A,B$, independence gives
$P(A=B)=1/2$. Adding the schema $0\leq P(B\mid A)\leq1$ introduces the arrow
$A\to B$ under the ordinary atomic construction; the two-variable model can
then allow both perfect agreement and perfect disagreement. Its equality-event
bounds become $[0,1]$ despite the new interval being numerically vacuous.

Thus a learner must score and record graph-bearing sentences, not simply prune
them because their bounds contain all probabilities. Likewise, a Boolean rewrite
that preserves a formula's truth table can change its syntactic atom occurrence
and induced graph if performed carelessly. Use canonical formula representations
for duplicate detection, while retaining the graph/provenance information needed
to prove that a rewrite preserves the LCN model.

The learned syntax will generally not be identifiable from observational data.
Equivalent formulas, Markov-equivalent graphs, alternative latent circuits, and
redundant assessments can represent the same observable distributions. Seek
predictive and semantic recovery under stated assumptions, and use minimum
description length to choose a representative. Do not promise recovery of a
unique true LCN program.

### 2.3 Four kinds of uncertainty

| Source | Possible representation | Interpretation to preserve |
| --- | --- | --- |
| Sampling uncertainty | Simultaneous confidence regions or justified Bayesian posterior sets | State the coverage or prior assumptions. |
| Prior imprecision | Imprecise Dirichlet or other prior families | A posterior-predictive envelope, not automatically a frequentist confidence interval. |
| Structural uncertainty | A set of learned structures or an explicit distribution over them | Union/envelope and model averaging are different operations. |
| Model mismatch or incomplete observations | Sensitivity classes, likelihood regions, completion sets | State the missingness or mismatch assumptions; do not hide them in arbitrary interval width. |

Keep numerical inference error separate from all four. A time-limited solver's
outer interval is not itself the learned credal set.

## 3. Literature map and research positioning

| Literature | Established contribution | How the proposed work extends it |
| --- | --- | --- |
| LCNs [R1–R3] | Logical assessments, graph-derived Markov semantics, inference, chain and DUMG factorizations | Learn sentence schemas and uncertainty while preserving those semantics. |
| Credal learning [R4–R7] | Local imprecise multinomial estimation, structural uncertainty, model selection/averaging | Account for LCN formula scopes and globally coupled assessments. |
| Tree/graph learning [R8] | Chow–Liu dependence trees and broader score/CI-based structure search | A controlled starting fragment and proposals for more expressive LCNs. |
| Structural EM [R9] | Alternate expected-data scoring with structure and parameter updates | Add LCN constraints and uncertainty estimation without assuming classical EM proofs still apply. |
| LearnSPN [R10] | Recursive variable partitioning and instance clustering | Propose scopes/latent mixtures, then check LCN representability and uncertainty. |
| LearnPSDD and Strudel [R11, R12] | Valid circuit splits/clones, vtree-guided structure search, efficient initialization | Search with credal inference costs and interval quality as additional criteria. |
| Einsum Networks [R13] | Efficient batched circuit evaluation, differentiation, and EM | Scale representative-model fitting after the semantics are validated. |
| Circuit operation atlas [R14] | Query-specific sufficient conditions and hardness boundaries | Check the whole training/query pipeline, including formula compilation and credal optimization. |
| Credal SPNs/CSDDs [R15, R16] | Local credal weights, bottom-up bound algorithms, topology-dependent posterior guarantees | Establish which LCNs they represent and learn within the corresponding restrictions. |
| 2U and credal complexity [R17] | Exact binary-polytree posterior inference and limits of generic low-width claims | A defensible first tractable LCN learning target and a warning against overgeneralization. |
| LearnCSPN [R19] | Missing-data-aware variable partitioning, clustering, and credal weight learning | Preserve LCN constraints and quantify statistical guarantees beyond circuit validity. |
| Circuit constraint relaxation [R20] | Faithful credal-BN compilation followed by efficient outer bounding | Track LCN assessment/Markov constraints and learn with the resulting inference contract. |
| Knowledge-guided PC learning [R21] | Parameter estimation using equality/inequality knowledge and penalties | Learn a feasible credal family and sentence structure, not only a constrained point estimate. |
| Recent optimization and structural uncertainty [R22, R23] | Improved mini-batch PC fitting; emerging probabilistic structure learning and structured credal learning | Stronger fitting baselines and explicit comparisons of uncertainty semantics. |

The closest direct predecessors include CSDD learning with local imprecise
Dirichlet estimates [R16] and LearnCSPN's joint learning from missing values
[R19]. Replacing probabilities with intervals or introducing missing-data-aware
circuit search is therefore not a sufficient novel contribution. The new
questions are LCN equivalence, calibrated learning after structure selection,
logical coupling, and preserving query tractability during joint search.

## 4. Define tractability by a query contract

### 4.1 Efficient precise evaluation is only the first requirement

For each learned model specify the supported query class $\mathcal Q$, the input
encoding, arithmetic accuracy, and the meaning of its bounds. Initial queries
are unconditional singleton probabilities and singleton posteriors given a
conjunction of observed variable assignments. Extend to partial-assignment
events where the selected backend supports them.

Even an independent distribution on fair bits does not make arbitrary Boolean
formula queries tractable: evaluating a general CNF's probability counts its
satisfying assignments. Formula compilation can be exponential. A tractability
claim must either restrict formula queries to a compatible compiled class or
include the compiled query size in its complexity statement.

Distinguish four levels:

1. Polynomial evaluation of one precise model.
2. Polynomial exact lower/upper inference over its stated credal set.
3. Polynomial certified outer approximation, without an exactness claim.
4. Learning cost, including structure search, compilation, and uncertainty
   estimation; tractable inference does not imply globally tractable learning.

### 4.2 Candidate tractable families

| Family | Initial supported exact task | Required restrictions |
| --- | --- | --- |
| Binary-polytree LCN fragment | Singleton lower/upper posteriors with assignment evidence, using 2U [R17] | Atomic CPT-style assessments for complete parent configurations; separately specified row sets; exact translation to the binary credal polytree; no extra cross-family coupling. |
| Smooth, decomposable credal SPN | Lower/upper probabilities of partial assignments [R15] | Local sum-weight polytopes separately specified as in the theorem; normalized leaves/weights; polynomial local LP descriptions. |
| Credal SPN with each internal node having at most one parent | Univariate conditional expectations/posteriors [R15] | The preceding conditions plus the theorem's topology and evidence assumptions; numerical precision included. |
| CSDD | Partial-assignment lower/upper probabilities [R16] | The source's local credal-set and support conditions. |
| Singly connected CSDD | Singleton conditional lower/upper probabilities [R16] | Its definition of singly connectedness, support positivity, and coherent evidence. |
| General constrained circuit or bounded-width LCN | Efficient precise subroutines; possibly certified outer bounds | Exact robust tractability remains a separate proof obligation. |

The circuit rows are **candidate representations for tractable LCNs**, not
already-proved LCN fragments. WP4 must establish the translation. The binary
polytree row is the first native-fragment target; verify its atomic LCN compiler
and Markov equivalence before using it as an exact learning backend.

An indegree bound controls the explicit CPT input size but does not replace the
polytree restriction. Bounded treewidth alone does not make arbitrary credal
inference tractable [R4, R17]. Nor does separate specification survive arbitrary
LCN assessments: a marginal constraint on a child may couple its parent
distribution and conditional table even when all mentioned atoms fit one family.

### 4.3 Circuit properties and their limits

Smoothness requires sum children to have the same variable scope;
decomposability requires product children to have disjoint scopes; determinism
requires sum children to have disjoint supports. Structured decomposability
coordinates decompositions through a vtree. These properties support different
operations; they are not interchangeable [R14].

For an observation $e$, separately specified credal SPNs admit bottom-up
extremization. A sum node solves a local LP over its weight vector, and a product
node combines nonnegative child values [R15, Theorem 1]. Shared subcircuits do
not by themselves invalidate that observation-probability theorem. It would be
too restrictive to demand a tree for every unconditional credal query.

Posterior bounds are different: numerator and denominator must use the same
model, and signed expressions appear in generalized Bayes tests. The tree
restriction in [R15] and the singly connectedness condition in [R16] matter
there. Multiply connected CSDDs can yield outer approximations for those
queries rather than exact results. Distinguish shared subcircuits permitted by
a theorem from arbitrary equality ties between parameters of distinct nodes,
which can introduce additional coupling.

CSDD Definition 5 [R16] requires strictly positive local lower probabilities
on allowed alternatives, except for structural impossibilities. An unseen
context under a vacuous imprecise Dirichlet model can violate this condition.
Do not insert a positive floor merely to retain an exactness label: either
declare a justified restricted prior, prove a boundary extension, or use a
different inference guarantee.

## 5. Parameter learning with a fixed structure

### WP1A — Representative-model fitting

For a fixed schema and hard knowledge, fit

$$
 \widehat\theta\in\arg\max_{\theta\in\Theta_{S,H,s}}
 \sum_{n=1}^N\log p_\theta(x^{(n)})-\lambda R(\theta).
$$

In a general LCN, $\Theta$ may be a joint probability table with nonlinear
Markov constraints. Likelihood is concave in that joint table, but the Markov
feasible set is generally nonconvex. Closed-form frequency estimates apply
only to proved factorized cases; a local NLP solution is not a globally optimal
fit certificate.

For complete data in the CPT fragment, use row counts and MLE or a declared
Dirichlet posterior mean. For a deterministic PSDD with fixed structure, use
the context/edge flow counts described in [R11]:

$$
 \widehat\theta_{vi}=\frac{N(\gamma_v\wedge\pi_{vi})}{N(\gamma_v)},
$$

where $\gamma_v$ is the node's context and $\pi_{vi}$ its branch event. Handle
empty contexts explicitly. In nondeterministic mixture circuits, branch
membership is latent; use EM or constrained gradient optimization, with the
batched computations of [R13] as an implementation option.

Compare full-batch EM, conventional mini-batch updates, and the Anemone method
in [R22] when the circuit family permits them. Its distribution-change
regularization is a stronger recent baseline than treating every minibatch as
a fresh complete dataset. For supplied knowledge, compare the increasing-penalty
parameter learner of [R21] with explicit constrained fitting; finite penalty
iterations do not themselves certify that all LCN assessments hold.

Retain all supplied probabilistic knowledge during fitting. If it is inconsistent
with the selected structural family, report that conflict rather than silently
projecting each assessment onto a separate local parameter interval.

### WP1B — Learn local credal sets and assessment bounds

Implement three estimators as distinct modes.

**Imprecise Bayesian mode.** For complete observed multinomial row counts
$n_1,\ldots,n_d$, $n_\cdot=\sum_i n_i$, and equivalent sample size $s_0>0$,
use the imprecise Dirichlet posterior-predictive set

$$
 K=\left\{\theta_i=\frac{n_i+s_0t_i}{n_\cdot+s_0}:
 t\in\Delta_d\right\},\qquad
 \theta_i\in\left[\frac{n_i}{n_\cdot+s_0},
                  \frac{n_i+s_0}{n_\cdot+s_0}\right].
$$

The displayed bounds require simplex normalization; they are not independent
free choices. At an empty context, $K$ is vacuous. Study the refinements and
alternative imprecise estimators in Masegosa–Moral [R5], rather than assuming
the basic IDM is optimal for all objectives. Structural zeros restrict the
allowed outcomes before updating; they should come from domain knowledge or
a separately justified support model.

**Frequentist mode.** Freeze a finite schema on data independent of the
calibration sample. For each conditional binary assessment compute a binomial
confidence interval conditional on the number of occurrences of $\psi_j$.
Use exact Clopper–Pearson intervals [R18] as a conservative baseline, assign error
budgets $\alpha_j$ with $\sum_j\alpha_j\leq\alpha$, and intersect appropriate
multinomial row bounds with their simplex. A zero count for the conditioning
event gives $[0,1]$. For a true fixed structural model, simultaneous row/event
coverage can imply $P^*\in\mathcal D$ and therefore simultaneous coverage of
all supported query probabilities by exact or sound outer inference.

This implication **depends on the true distribution satisfying the selected
Markov model and hard knowledge**. Confidence intervals for sentence probabilities
do not validate data-selected exact independence assumptions. Model misspecification
and structural selection need separate treatment in WP2.

**Likelihood-region mode.** Retain correlations between parameters through

$$
 K_r=\{\theta\in\Theta_{S,H,s}:
 2[\ell(\widehat\theta)-\ell(\theta)]\leq r\}.
$$

This is a candidate uncertainty family for globally coupled LCNs. Calibrate
$r$ under explicit assumptions; do not apply a routine chi-square threshold
to singular latent circuits, boundary parameters, or selected structures.
Bootstrap calibration is an empirical method unless an appropriate theorem is
proved. Such a region generally breaks separate specification, so it may require
the general certified inference route instead of a tractable credal circuit.

### WP1C — Coherence, projection, and assessment extraction

Individually reasonable intervals can be incompatible with one another or with
the chosen Markov structure. Solve a feasibility problem and retain a witness.
The final reported representative distribution should belong to the final
calibrated set. If the discovery fit lies outside it, refit or select a feasible
representative inside the set; this does not require shrinking the confidence
region. Alternatively, label the discovery predictor as a separate artifact.
Check all assessments on the joint model, including compound conditioning
events and family-local formulas that depend on parent marginals.

If widening estimated intervals is allowed, formulate explicit slack variables:

$$
 \min_{\delta^-,\delta^+\geq0}\sum_j c_j(\delta^-_j+\delta^+_j)
 \quad\text{such that}\quad
 \mathcal D_{S,\{[\ell_j-\delta^-_j,u_j+\delta^+_j]\},s}\neq\varnothing.
$$

Clip endpoints to $[0,1]$ and protect nonnegotiable domain constraints. Widening
confidence bands preserves their marginal coverage, but it does not repair an
incorrect Markov assumption. A global minimal-repair claim needs a global solver
certificate. Report every widened assessment and distinguish estimated rules
from expert-specified requirements.

Conversely, when fitting a circuit first, one can extract
$\ell_j=\inf_{p\in\mathcal C}p(\phi_j\mid\psi_j)$ and the corresponding upper
bound. The resulting finite collection of intervals need not characterize the
circuit family. Once the LCN-induced independencies are added, the resulting
model can be larger, smaller, or incomparable. Verify containment or equivalence
rather than calling this extraction an exact translation.

### WP1D — Incomplete data

For a record observed on $O_n$, the representative likelihood is

$$
 \ell(\theta)=\sum_n\log\sum_{x_{V\setminus O_n}}
 p_\theta(x_{O_n}^{(n)},x_{V\setminus O_n}).
$$

Use EM with exact circuit marginals where available, or an explicitly approximate
E-step elsewhere. State whether missingness is MCAR/MAR and ignorable; otherwise
model the missingness mechanism or perform a sensitivity analysis. Complete-case
counts and treating missing values as false are not general solutions.

Applying IDM formulas to fractional EM counts is a useful baseline heuristic,
not automatically a valid imprecise Bayesian posterior. With uncertain completions,
the posterior can be a mixture over count configurations, with dependencies that
expected counts discard. Research alternatives are completion-set optimization,
observed-data likelihood regions, and posterior-set updates over tractable
latent models. On small instances compare against enumeration of possible
completions. Analyze their computational and statistical costs separately.

## 6. Structure learning

### WP2A — A controlled language of candidate structures

Begin with a fixed atom dictionary. Permit bounded-size clauses/conjunctions,
bounded conditioning scopes, and a small set of annotation choices with known
semantics. Candidate sources include observed co-occurrence, mutual information,
association-rule proposals, and supplied domain templates. These are proposal
mechanisms, not certificates of independence or logical truth.

Search progressively through four levels:

1. Atomic directed trees/forests with CPT-row assessments.
2. Binary polytrees with bounded indegree and separately specified row sets.
3. Bounded-scope chain or general LCN sentence schemas, using a certified
   general inference backend when needed.
4. Circuit-compatible context and mixture structures whose LCN translation has
   been proved, with an explicit auxiliary-variable budget if required.

Moves include adding/removing/reversing an admissible directed dependency,
adding/removing a sentence template, changing a conditioning scope, changing
an annotation, splitting/merging a context, and circuit splits/clones or vtree
changes. Every move recomputes the affected graph and Markov assertions.
Do not assume changing one sentence only changes one likelihood term.

Use Chow–Liu [R8] as a transparent baseline for a representative dependence
tree. It solves a particular precise tree-approximation problem, not the full
credal-LCN selection objective. Extend with beam search over legal polytree
moves, then broader sentence search. Count the description length of formulas,
annotations, graph-bearing vacuous sentences, and auxiliary variables.

### WP2B — Scores that do not reward meaningless intervals

Neither optimistic nor worst-case training likelihood alone identifies a useful
credal set. If $\mathcal D_1\subseteq\mathcal D_2$, then

$$
 \sup_{p\in\mathcal D_1}\ell(p)\leq\sup_{p\in\mathcal D_2}\ell(p),\qquad
 \inf_{p\in\mathcal D_1}\ell(p)\geq\inf_{p\in\mathcal D_2}\ell(p).
$$

The first objective can favor unnecessary enlargement; maximizing the second
can favor collapsing the set. Separate predictive fitting from uncertainty
calibration. Use an explicitly chosen selection objective, for example

$$
 J(S,C)= -\ell_{\rm val}(\widehat p_{S,C})
 +\lambda\operatorname{DL}(S,C)
 +\mu\operatorname{Cost}_{\mathcal Q}(C,K)
 +\nu\widehat{\operatorname{Width}}_{\mathcal Q}(K),
$$

subject to feasibility, declared uncertainty-estimation rules, and any exact
tractability requirement. Here provisional intervals come only from training
data, and validation selects among candidates. Width is a secondary criterion
under a fixed calibration/prior rule, not an unconstrained knob that can be
reduced to zero. Use held-out representative log loss and Brier score; for
set-valued decisions report the chosen utility or selective-risk criterion
separately rather than calling it a universally proper score for credal sets.

BIC/MDL and Bayesian scores provide additional baselines, but their usual
local decomposition and regular asymptotics need not hold for coupled LCNs or
singular circuit mixtures. Cache local score changes only where a derivation
establishes locality. Else evaluate the affected global fit, with an optimization
gap or a documented approximation.

### WP2C — Structural uncertainty and identifiability

Constraint-based CI tests and stability selection can prioritize dependencies.
Failure to reject a conditional-independence null does not establish a
population Markov condition. Consistency claims require assumptions such as
membership in the candidate family, suitable identifiability/faithfulness, and
consistent tests.

Keep a bounded collection $\mathcal S$ of competitive structures when evidence
is insufficient. An envelope model is

$$
 \mathcal D_{\rm union}=\bigcup_{S\in\mathcal S}\mathcal D_{S,\eta_S,s},
 \qquad
 \underline P(Q)=\min_{S\in\mathcal S}\underline P_S(Q),\quad
 \overline P(Q)=\max_{S\in\mathcal S}\overline P_S(Q).
$$

For posteriors, take the corresponding extrema only over models in which the
evidence is possible. This remains polynomial in the number of retained models
when each model's query is tractable, but selecting a small adequate collection
is still a learning problem. A finite envelope has a coverage guarantee only
if the appropriate structural-retention and parameter-coverage events are
controlled. A bootstrap stability threshold alone does not prove this.

Bayesian/credal model averaging [R7] is a separate option with explicit model
weights or weight sets. It can produce observational mixtures that violate
every constituent graph's pointwise CIs. Do not describe such a mixture as the
same LCN structure merely because each component is an LCN.

Recent probabilistic circuit structure inference and structured credal-learning
proposals [R23] should inform this comparison. Their uncertainty sets and model
averages have their own semantics; neither can be adopted as an LCN coverage
theorem merely because it describes structural uncertainty.

## 7. Joint structure–parameter learning

### WP3A — Alternating search with inference-aware refinement

Use separate fitting, selection, calibration, and final-test partitions. The
first two may be inner folds to conserve data; the basic validity argument
uses a genuinely untouched calibration partition after structure selection.

```text
initialize a permitted structure using a tree model or valid circuit
fit a representative distribution on fitting data
repeat until the search/compilation budget is exhausted:
    form provisional local credal sets using the declared estimator
    propose graph, sentence, context-split, or circuit moves
    reject moves violating hard knowledge or the selected tractability contract
    refit affected parameters; use full refitting when locality is not proved
    check coherence and the actual model represented by any circuit translation
    evaluate fit, description length, interval utility, and inference cost
    keep a beam of competitive candidates, preserving semantic distinctions
select a structure or finite structure collection using selection data
freeze structures, support assumptions, query contract, and tuning choices
construct final uncertainty sets from independent calibration data
validate feasibility and inference certificates; evaluate once on test data
```

The main proposed improvement over ordinary likelihood-guided circuit growth
is an **uncertainty-aware split criterion**. A split can improve likelihood but
create rare contexts with almost vacuous credal parameters. Rank moves by fit
improvement together with resulting context sample sizes, provisional query
widths, and the cost of exact credal inference. Compare against LearnPSDD/Strudel
likelihood-per-size improvements and fixed minimum-count stopping rules.

A second proposed move resolves a circuit-sharing conflict. Sharing may reduce
model size and pool data but invalidate an exact conditional-bound algorithm.
Compare keeping the shared circuit with certified outer bounds against cloning
the relevant subcircuits and retraining their parameters. Cloning changes the
statistical family unless equality constraints are retained; retaining those
ties may restore the computational coupling. Do not claim the two choices are
semantically equivalent without proof.

### WP3B — Structural EM for incomplete data and latent circuits

Use Structural EM [R9] as the precise baseline: compute expected sufficient
statistics under the current representative model, search over valid structures,
and update parameters. Carry domain constraints through every step and reuse
tractable circuit E-steps when their query requirements are supported.

The classical monotonicity argument concerns a particular objective and an
appropriate E/M update. It does not automatically cover credal-set width
penalties, approximate inference, support changes, or an adversarial E-step.
For the proposed joint objective, accept a move only after evaluating its
specified objective or a proved surrogate bound. Use line search, a trust region,
or rollback if necessary. Global optimality remains a separate issue.

Investigate a robust extension in which expected sufficient statistics range
over a compatible set of latent completions. Preserve joint compatibility of
these statistics. Independently minimizing every row or every observation can
select mutually inconsistent models:

$$
 \inf_\theta\sum_n\log p_\theta(x^{(n)})
 \neq \sum_n\inf_\theta\log p_\theta(x^{(n)})
 \quad\text{in general}.
$$

The first implementation should use precise Structural EM for discovery and
separate uncertainty calibration. A fully set-valued Structural EM is a later
research contribution, with a new objective, convergence analysis, and small
completion-enumeration reference problems.

LearnCSPN [R19] is the direct incomplete-data circuit-learning comparison. It
modifies the independence-count calculation and clustering step, allows ambiguous
membership for incomplete records, and learns credal weights in a tree-shaped
internal circuit. Reproduce its valid-circuit and complete-data-reduction
properties, then separately measure whether its learned intervals cover the
intended generating model. Those structural properties alone are not a
finite-sample uncertainty guarantee.

### WP3C — Efficient updates and reusable computations

Cache formula truth vectors, row/context counts, circuit flows, and query
circuits. Use warm-started constrained parameter solves and reuse unaffected
subcircuits. Batch likelihood/gradient evaluation as in [R13], but include the
time spent on credal LPs, feasibility, and calibration in end-to-end comparisons.

Learned knowledge may cause a large compilation increase. Treat maximum formula
scope, circuit size, separator width, and number of coupled credal parameters
as hard budgets where required. A rejected structural move should retain its
statistical score for analysis, so one can quantify the accuracy cost of the
tractability restriction.

## 8. Learning tractable LCNs through probabilistic circuits

### WP4A — Prove the native tractable fragment first

Formalize the binary-polytree CPT fragment in Section 4.2. Restrict conditionals
to complete parent configurations, keep row uncertainty separately specified,
and prove that the LCN's pointwise distribution family equals the corresponding
factorized family. Check zero-probability parent configurations explicitly.

Then combine tree/polytree structure learning, row estimation, and 2U inference.
This gives a concrete end-to-end baseline for exact posterior bounds, with
runtime measured in the explicit model size and query accuracy. Additional
logical rules are accepted only if they can be represented without violating
the fragment's conditions; otherwise use the general mode with a changed
certificate. A numerically harmless-looking child marginal can invalidate
separate specification.

### WP4B — Two directions of LCN–circuit translation

**LCN to circuit.** Compile an LCN model family, preserving every assessment
and Markov condition. A circuit for a single fitted distribution is insufficient:
it must retain the family of allowed parameter choices and their dependencies.
Wijk et al. [R20] already give a faithful credal-BN/circuit transformation and
an outer relaxation that releases shared parameter constraints. Use that as a
direct compilation/bounding baseline. Extending it to LCNs requires retaining
the additional formula constraints and the chosen Markov semantics.
For a constrained circuit representation $\mathcal C_K$, establish

$$
 \{p_\theta:\theta\in K\}=\mathcal D_{S,\eta,s},
$$

or state the exact containment that is proved. If parameter compilation creates
cross-node constraints, efficient precise evaluation can survive while efficient
credal optimization fails. Track this as a valid constrained-circuit mode.

**Circuit to LCN.** Start from learned decomposable/deterministic circuits and
derive sentence templates, context conditions, and any required selector/gate
atoms. Prove that marginalizing those auxiliary atoms yields the intended
observable family. Deterministic gate constraints can introduce structural
zeros, cycles, or new LCN independencies; the proof must include them. Merely
exporting local sum-weight intervals is not enough.

First target tree-shaped context circuits and CSDD fragments with bounded
context formulas. Test larger shared circuits only after this bridge is
understood. If the existing LCN language cannot express the required
context-specific independencies or parameter ties compactly, present the result
as a **circuit-backed extension** or as a related model, with a precise language
extension. It must not be silently labeled an equivalent native LCN.

### WP4C — Pointwise model sets versus convex strong extensions

The original LCN semantics requires Markov CIs of every admitted distribution.
Some credal-circuit papers define a strong extension as a convex hull of
factorizing distributions [R16]. Convex mixtures need not preserve pointwise
independence: mixing equally two independent Bernoulli pairs with success
probabilities 0.1 and 0.9 gives marginal probabilities 0.5 but joint success
probability 0.41, not 0.25.

For linear marginal objectives, taking the convex hull does not change extrema.
For regular conditional ratios, a mixture's ratio is an evidence-weighted
average of its component ratios, so endpoint equivalence can also hold under
the appropriate evidence convention. This is **query-endpoint equivalence**,
not equality of semantic sets. Moreover,

$$
 \operatorname{conv}(\mathcal F\cap H')
 \neq \operatorname{conv}(\mathcal F)\cap H'
 \quad\text{in general},
$$

where $H'$ denotes additional assessment constraints. Intersecting after
convexification can admit mixtures of models that individually violate those
constraints. Establish whether constraints apply before or after the convex hull
in every translation and learning objective.

### WP4D — Certificate-preserving circuit search

Annotate every learned circuit with its scopes, supports, local credal-set
encoding, parameter ties, topology/multiplicity properties, and the supported
query algorithms. Each split, clone, mixture, parameter-sharing operation,
and added logical assessment must update this certificate.

The most promising new result is a grammar of learning moves that preserves
both an LCN representability theorem and a credal query theorem. LearnPSDD's
support-preserving splits/clones and Strudel's structured splits are precedents
for precise-model validity [R11, R12]; preserving exact **credal** posteriors
and LCN semantics is the additional task.

Use three explicit statuses: exact tractable fragment; tractable certified
outer approximation; and general constrained model with budgeted inference.
Never improve apparent speed by dropping global logical constraints or treating
learned parameters as independently selectable when they are not.

## 9. Statistical guarantees and theoretical targets

### 9.1 A first finite-sample result

For a structure chosen independently of the calibration sample, assume the
data-generating distribution $P^*$ belongs to its stated Markov/support family.
If simultaneous calibrated parameter/assessment regions satisfy

$$
 \Pr\{P^*\in\widehat{\mathcal D}\}\geq1-\alpha,
$$

then on the same event every query probability belongs to the corresponding
exact credal interval. A sound outer inference algorithm preserves this
containment. For posterior queries require $P^*(e)>0$ and the declared regular
extension convention. This is a model-coverage argument; it does not require
a separate union bound over subsequently chosen queries.

The central research work is establishing the premise for each estimator and
structure-selection procedure. The theorem cannot be obtained by attaching
confidence labels to independently learned intervals after unrestricted search.
For retained structures, allocate and justify structural and parameter error
budgets where possible. When only conditional-on-model validity is available,
report exactly that and test misspecification explicitly.

### 9.2 Proposed theorem program

| Target | Precise result sought | Main obstacle |
| --- | --- | --- |
| T1: Fixed-schema learning | Coherent parameter-set construction and consistency/coverage under stated assumptions | Markov constraints, overlapping formulas, sparse contexts. |
| T2: Structure-aware calibration | Conditions under which selection plus independent calibration yields valid query envelopes | A selected graph can be structurally wrong even with accurate local probabilities. |
| T3: Native tractable fragment | Exact LCN/credal-polytree equivalence and preserved 2U query complexity | Extra assessments and zero parent contexts. |
| T4: LCN–circuit bridge | Semantic equality, projection equality, or explicitly limited endpoint equivalence | Context-specific independence, latent gates, convexification, parameter coupling. |
| T5: Valid joint learning moves | Invariance of support, LCN meaning, and required credal tractability | Precise circuit validity is weaker than the desired guarantee. |
| T6: Joint objective progress | Monotonicity of a stated score or surrogate for the proposed alternating method | Structural changes, approximate E-steps, interval penalties. |
| T7: Sample/computation tradeoff | Bounds linking context counts, interval uncertainty, circuit size, and query error | Rare contexts and posterior evidence can amplify uncertainty. |
| T8: Incomplete-data uncertainty | Sound posterior/completion-set propagation for a useful tractable subclass | Fractional counts discard posterior dependence. |

Under correct identifiable finite models, positive context probabilities, and
appropriate regularity, local estimators should concentrate and representative
fits should be consistent. Prove each result for its fragment. Do not infer
uniform posterior concentration when evidence probabilities can approach zero,
or global structure-search consistency for a fixed-width heuristic beam.

## 10. Repository integration

The current package is organized around representation and inference. The code
review did not identify an existing dedicated learning subsystem. Build the
research prototypes separately from stable inference interfaces.

| Existing component | Proposed reuse | Required care |
| --- | --- | --- |
| [model.py](../../lcn/core/model.py) and [parser.py](../../lcn/core/parser.py) | Sentence construction, annotations, formula evaluation | Preserve syntactic graph effects and record learned versus supplied provenance. |
| [independencies.py](../../lcn/core/independencies.py) | Markov assertions and equivalence checks | Fix the semantic profile; avoid treating a graphical edit as purely numerical. |
| [compile_cn.py](../../lcn/inference/marginal/cn/compile_cn.py) | Candidate native-fragment compilation | Prove family/row representability; existing compilation is not an arbitrary LCN equivalence oracle. |
| [junction_nlp.py](../../lcn/inference/marginal/cn/junction_nlp.py) and [structured_consistency.py](../../lcn/inference/utils/structured_consistency.py) | Joint constraints and coherence checks | Audit hosted/dropped constraints and solver certificates for the selected learning target. |
| [exact.py](../../lcn/inference/marginal/exact.py) | Small-model joint reference and feasible witnesses | Distinguish local solutions from global bounds; likelihood fitting requires a new objective. |
| [cve.py](../../lcn/inference/marginal/cn/cve.py) and [approxlp.py](../../lcn/inference/marginal/cn/approxlp.py) | Credal inference/optimization baselines | Match represented sets; ApproxLP outputs do not by themselves certify outer bounds. |
| [strong_extension_exactness.tex](../research-chain-lcn/strong_extension_exactness.tex) | Counterexamples to local-parameter assumptions | Reuse concrete fixtures, independently check general claims transferred to learning. |

Proposed future modules under `lcn/learning/` are `data.py`, `schemas.py`,
`counts.py`, `parameter_sets.py`, `scores.py`, `structure_search.py`,
`joint_learning.py`, `circuit_bridge.py`, and `certificates.py`. A learned artifact
should include its LCN program, any circuit/auxiliary variables, a representative
model, uncertainty interpretation, data-partition identifiers, estimator settings,
support/Markov profile, and query contract.

For circuit baselines, evaluate suitable maintained implementations of LearnSPN,
LearnPSDD/Strudel, and a batched PC library after checking their supported
operations. Pin versions and compare equivalent workloads. Adding a dependency
does not resolve the LCN translation theorem. This task creates research
documentation only.

## 11. Experimental program

### 11.1 Mandatory correctness fixtures

1. The fair-atom/vacuous-sentence example in Section 2.2: numerical redundancy
   does not imply semantic redundancy.
2. Binomial/multinomial counts with an empty context, an unseen outcome, and a
   structural zero. For $n=2,N=10,s_0=2$, the IDM interval is $[1/6,1/3]$.
   An empty context gives $[0,1]$, not an invented precise probability.
3. Independent-model mixtures with $P(A,B)=0.41$ versus
   $P(A)P(B)=0.25$, testing strong-extension versus pointwise-CI semantics.
4. An assessment on a nonroot marginal and a formula-valued conditioning event
   that couple multiple CPT rows or families.
5. A circuit with shared subgraphs for which observation bounds are exact but
   a posterior routine only supplies an outer approximation, following [R16].
6. A zero-frequency allowed CSDD branch, testing whether its published support
   assumptions actually hold after the selected estimator.
7. A small missing-data problem whose posterior/completion sets can be enumerated,
   exposing differences from IDM on fractional EM counts.
8. Arbitrary-CNF probability on independent bits, making query-compilation cost
   visible; no generic formula-query tractability label is allowed.
9. Known-domain constraints contradicted by noisy observations, inconsistent
   learned assessments, and impossible evidence, all with distinct statuses.

The numerical values in fixtures 1–3 were checked with exact rational arithmetic
during preparation of this plan. Implementation tests and statistical simulation
remain future work.

### 11.2 Data families and sampling design

Generate complete-data samples from known binary trees/polytrees, chain models,
general LCN-compatible distributions, deterministic context circuits, and models
outside the candidate class. Vary sample size, dependency strength, formula
scope, rare-context frequency, interval/prior strength, and circuit sharing.
Use an explicit feasible generating distribution: an LCN's credal set alone
does not define a unique data-generating sampling process.

Evaluate several generating distributions from a given set, including boundary
and interior points. Treat random selection of a model once per dataset
differently from resampling model parameters for every observation; the latter
can change the observed joint family. Include replicated datasets per condition
to estimate frequentist coverage.

For real data, start with established discrete density-estimation benchmarks
used by LearnSPN/LearnPSDD/Strudel, then a logical-support task such as
seven-segment displays [R16] and one application where domain rules are available.
Separate supplied constraints from rules inferred using training labels. Use
entity/time splits for repeated or temporal records. The original LCN fraud
experiment [R1] demonstrates empirical rule construction, but the new evaluation
must estimate/tune all learned bounds without accessing final test outcomes.

For incomplete-data studies, apply controlled MCAR and MAR masks to synthetic
data with known truth, followed by sensitivity to nonignorable missingness.
Report discretization and preprocessing rules learned only from fitting data.

### 11.3 Baselines

- Empirical sentence frequencies with fixed widths; bootstrap ranges as an
  explicitly heuristic baseline; IDM; simultaneous conditional confidence bands;
  and a calibrated likelihood-region approach where justified.
- Fixed expert structure, Chow–Liu plus local credal sets, a polytree search,
  and ordinary BN score-based search followed by the same parameter estimator.
- LearnSPN, LearnPSDD, Strudel, and representative-model EM/batched PC fitting.
- Learned precise circuits with post-hoc parameter perturbation [R15], and
  CSDD-style local IDM learning [R16]. These are essential direct competitors.
- LearnCSPN [R19] on incomplete data, knowledge-guided PC parameter learning
  [R21], and recent PC optimization [R22]. For constrained-circuit outer bounds,
  compare the applicable credal-BN relaxation of [R20].
- Separate structure-then-parameter learning versus joint learning; precise
  Structural EM versus the proposed constrained/set-aware variants.
- General LCN fitting with certified inference on small instances, to measure
  the expressive cost of tractability restrictions.

Use the same data partitions, uncertainty conventions, and query workloads
where meaningful. Different semantic targets should be compared as modeling
alternatives rather than as competing approximations to the same truth.

### 11.4 Metrics

**Prediction:** test log loss and Brier score for the declared representative
distribution; missing-value likelihood; application-specific decisions.

**Uncertainty:** simultaneous containment of the known generating distribution
or query probabilities on synthetic repetitions; interval width conditional on
coverage; coverage by context frequency; and sensitivity to prior strength.
On real data, estimate event probabilities only in groups with adequate repeated
observations and uncertainty on those estimates. A single binary outcome is not
the unknown probability, so membership of that outcome in $[\ell,u]$ is not a
valid probability-interval calibration metric.

**Decisions:** selective risk versus abstention/coverage, set-valued prediction
size, and a prespecified utility. A robust interval's threshold decision has
different semantics from a conformal prediction set; do not equate them.

**Structure and logic:** graph recovery up to an appropriate equivalence class,
formula-semantic recovery where identifiable, support violations, feasibility
repair, auxiliary-variable count, and stability across samples.

**Computation:** learning time including calibration and compilation, peak memory,
circuit size, effective credal coupling, per-query latency, exactness status,
and outer-versus-inner optimization gaps. Report probability interval width
separately from numerical solver uncertainty.

Use at least 30 independently generated training datasets for initial synthetic
cells, increasing replication where coverage estimates are too noisy. Report
confidence intervals on coverage itself. Separate tuning seeds from evaluation
seeds and publish failures/timeouts as well as successful runs.

### 11.5 Ablations and falsifiable hypotheses

| Hypothesis | Comparison that can refute it |
| --- | --- |
| Uncertainty-aware structure search avoids harmful rare-context splits | Compare with likelihood-only search and a simple minimum-count rule at matched circuit size. |
| Logic improves data efficiency when correct | Compare with the same learner without rules; repeat under misspecified/noisy rules. |
| Joint learning beats post-hoc interval addition | Match estimator, search budget, and partitions; compare fit/coverage/width rather than accuracy alone. |
| A certified circuit fragment offers a useful accuracy–runtime tradeoff | Compare against native tractable LCNs and small general LCN references on the same query family. |
| Structural envelopes improve reliability at moderate data sizes | Compare single selected structures and retained collections, including their width and runtime costs. |
| Missing-data set propagation improves on fractional-count heuristics | Compare with enumerated small-case references and assess the scaling penalty. |

A negative outcome is informative: uncertainty-aware scores may add little
beyond minimum-count regularization; exact posterior topology restrictions may
require too many clones; or an LCN–circuit equivalence may need an impractical
number of auxiliary atoms. Quantify these limits and revise the proposed family.

## 12. Milestones and publication strategy

Proposed scope: a 24-week initial program, with theorem-dependent extensions
treated as research gates rather than promised outcomes.

| Weeks | Deliverable | Gate |
| --- | --- | --- |
| 1–3 | Formal learning contract, dataset partitions, native-fragment semantics, correctness fixtures | Agree on uncertainty meanings and supported queries; reproduce direct predecessors. |
| 4–6 | Complete-data fixed-structure estimators, feasibility checks, reference fitting | Demonstrate correct counts, coherence, and conditional-on-model coverage. |
| 7–9 | Tree/polytree structure search with final calibration | Establish the exact native-fragment translation and separate structure error from parameter error. |
| 10–13 | LearnSPN/PSDD/Strudel baselines and restricted LCN–circuit compiler | Prove equality/containment and query conditions before calling a learned circuit an LCN. |
| 14–17 | Joint uncertainty-aware search and certificate-preserving moves | Compare against post-hoc credalization and minimum-count baselines at matched resources. |
| 18–20 | Missing-data fitting and small robust-EM/completion references | State which updates are heuristic and which statistical or progress guarantees hold. |
| 21–24 | Full experiments, ablations, artifact, and manuscript | Restrict claims to proved fragments; report general-mode approximations explicitly. |

The first concrete deliverable should learn a binary-polytree LCN from complete
data, attach independently calibrated local sets, verify its translation, and
return exact singleton posterior bounds with a model-conditional coverage
statement. It should also demonstrate a rule that moves the model outside that
tractable fragment and correctly changes the inference contract.

A first paper can combine coherent LCN parameter/structure learning with the
native tractable fragment. A second can focus on the LCN–circuit bridge and
credal-tractability-preserving joint learning. A third, if justified by the
results, can address structural uncertainty and incomplete-data guarantees.
Before any priority claim, update the literature search on LCN learning, credal
circuits, uncertainty calibration, and constrained circuit learning beyond the
foundations reviewed here.
