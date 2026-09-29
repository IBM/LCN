# Sampling inference for chain graph logical credal networks

Research plan, 29 September 2026. Code reviewed at commit
`bae09ed2174858581301c628553c9f82df39dd03`.

This plan completes the literature and code review begun on 28 September.
It proposes research and implementation work; it does not report new algorithm
benchmarks or claim that the proposed methods have already been implemented.

## 1. Recommendation and scope

Develop an inference method that combines **search over feasible precise models**
with **importance sampling of worlds**, using the chain graph structure to reduce
both costs. Start with ordinary credal networks, where the semantics and reference
algorithms are established, then extend to chain graph LCNs while retaining their
logical constraints and Local Markov Condition (LMC).

Sampling for credal networks already exists. Random selection of local credal
vertices, simulated annealing, genetic search, and optimization using sampled
importance weights are relevant precedents [R1–R7]. Consequently, implementing
Monte Carlo on the existing compiled credal network is a useful baseline, but is
insufficient as the main research contribution.

The strongest research question is:

> Can shared-sample importance inference and structure-aware feasible-model
> search provide useful anytime estimates of lower and upper probabilities for
> chain graph LCNs, while preserving constraints lost by interval compilation
> and explicitly accounting for statistical and optimization error?

The initial inference task is a singleton marginal or posterior:

$$
 L=\inf_{p\in\mathcal D,\ p(e)>0}p(Q=1\mid e),\qquad
 U=\sup_{p\in\mathcal D,\ p(e)>0}p(Q=1\mid e).
$$

Use unconditional queries first, then ordinary evidence, then rare evidence and
zero-probability boundary cases. Formula queries can follow once their indicator
evaluation and scope costs are supported. MAP inference, learning, first-order
grounding, and structures with directed or semidirected cycles are later work.
A chain graph here may contain large undirected components and a branching or
multiply connected component DAG; it is not limited to a linear Markov chain.

## 2. What the repository provides

Paths below are relative to this document. These are reusable mechanisms, with
their limits stated explicitly.

| Existing code or document | Role in the research |
| --- | --- |
| [model.py](../../lcn/core/model.py), [mixed_graph.py](../../lcn/core/mixed_graph.py) | Build the primal and structure graphs, collapse undirected connected components, identify families, and derive LMC assertions. |
| [factorization.py](../../lcn/inference/marginal/cn/factorization.py) | Describe component/parent scopes and attached sentences; optionally merge families. It does not itself establish parameter feasibility. |
| [local_credal_sets.py](../../lcn/inference/marginal/cn/local_credal_sets.py) | Build sentence constraints and solve coordinate probability bounds. The linear path uses LP/Charnes–Cooper; `linear-tight` adds in-scope LMC equalities. |
| [vertices.py](../../lcn/inference/marginal/cn/vertices.py) | Convert interval factors into pyAgrum credal sets and enumerate local vertices. Useful for reproducing compiled-CN baselines. |
| [approxlp.py](../../lcn/inference/marginal/cn/approxlp.py) | Existing coordinate search and precise-model variable elimination. The current implementation uses enumerated vertices; it is not the complete general constraint-oracle implementation of the paper. |
| [coupling.py](../../lcn/inference/marginal/cn/coupling.py) | Sentence and LMC checkers for constraints whose atom scopes fit no family. Reuse the indicator machinery, but extend the coverage for a full LCN feasibility oracle. |
| [junction_nlp.py](../../lcn/inference/marginal/cn/junction_nlp.py), [exact.py](../../lcn/inference/marginal/exact.py) | Small-instance reference optimization, feasible witness extraction, and possible initialization. Global status, objective bounds, and residuals must be recorded. |
| [common.py](../../lcn/inference/utils/common.py), [structured_consistency.py](../../lcn/inference/utils/structured_consistency.py) | Truth tables, formula indicators, LMC residual construction, and consistency checks. |
| [run_algorithm.py](../../experiments/run_algorithm.py), [generator.py](../../lcn/benchmarks/generator.py) | Existing experiment timing, process limits, reporting, and synthetic instances. |
| [properties.tex](../research-chain-lcn/properties.tex), [strong_extension_exactness.tex](../research-chain-lcn/strong_extension_exactness.tex) | Existing analysis of compilation gaps and engine guarantees; audit their assumptions against the actual representation. |

The central implementation fact is that the usual compilation path retains
**coordinate intervals**, then permits independent choices from the resulting
row credal sets. Exact solution of each coordinate optimization does not preserve
the full feasible set of a component table or the relationships among tables.
Increasing the sampling budget cannot recover constraints absent from that model.

There are three distinct losses to investigate:

1. **Within-row loss:** coordinate intervals can discard constraints on sums of
   states, even in a root component.
2. **Across-row or across-family loss:** a sentence can depend on several CPT rows
   and on parent probabilities. Atom-scope containment in one family does not
   make independent row selection valid.
3. **Independence loss:** a free table for an undirected component need not satisfy
   all atom-level independencies inside that component. The `linear` local solver
   omits LMC equalities, and tighter coordinate bounds still do not encode them
   in every recombined candidate.

For example, consider a four-state component table
\(p=(p_{00},p_{01},p_{10},p_{11})\) with only the simplex constraints and
\(p_{00}+p_{01}\le0.6\). Its exact coordinate intervals are
\([0,0.6],[0,0.6],[0,1],[0,1]\). The table \((0.6,0.4,0,0)\) satisfies all
four intervals and normalization but violates the original constraint. Its
probability of the first atom being zero is 1, whereas the original maximum is
0.6. This is a table-level counterexample to treating interval projection as an
exact representation of arbitrary component constraints.

## 3. Semantics to fix before implementing sampling

### 3.1 The original LCN target

Let $\Omega=\{0,1\}^n$. Define $\mathcal D_{\mathrm{LCN}}$ as the joint
distributions on $\Omega$ satisfying normalization, nonnegativity, every
sentence, and every LMC equality. A Type-2 sentence uses the cleared convention
implemented in the repository:

$$
 \ell P(\psi)\le P(\varphi\land\psi)\le uP(\psi).
$$

These are **linear constraints in the joint probabilities**. LMC equalities such
as $P(x,y,s)P(s)=P(x,s)P(y,s)$ are generally quadratic in those probabilities.
The resulting LCN feasible set need not be convex. The description of the cleared
sentence itself as “bilinear in p” in `properties.tex` should be corrected in a
separate documentation change.

Distinguish zero conditioning mass in a sentence, where the cleared constraint
is vacuous, from zero evidence mass in a query, where the posterior is undefined.
For the first conditional experiments define the restricted target explicitly:

$$
 \mathcal D_\beta=\{p\in\mathcal D_{\mathrm{LCN}}:P(e)\ge\beta\},\quad\beta>0.
$$

This restriction makes uniform error analysis easier but changes the query
unless justified. Report its value and examine sensitivity as it decreases.
It is not interchangeable with the regular-extension target over all
\(P(e)>0\). Existing numerical denominator floors also require this accounting.

### 3.2 Component factorization and its limits

For chain components $C_1,\ldots,C_m$, the computational representation is

$$
 p_\theta(x)=\prod_{j=1}^m
 \theta_j(x_{C_j}\mid x_{\operatorname{pa}(C_j)}).
$$

Topological sampling of this component DAG is straightforward once the
conditional tables are fixed. It does not follow that arbitrary choices of
those tables describe exactly the original LCN.

The local revised LCN paper establishes the chain graph Markov/factorization
connection under positivity [R11, Section 5]. Its discussion also identifies
special cases allowing zeros and explains why general hard constraints need
care. For noncomplete components, the relevant conditional graphical
factorization must retain the internal independencies; treating the component
as an unrestricted super-variable can relax the model.

The first theory deliverable must therefore establish both directions for each
supported subclass:

- **Soundness:** every accepted parameter vector produces an LCN-feasible joint.
- **Completeness:** every target joint is represented, or the parameterization is
  dense enough to recover the desired infima and suprema.

Checking every original sentence and LMC assertion establishes soundness of a
particular product candidate. It cannot establish completeness of the product
representation. Begin with ordinary DAG credal networks and a verified class of
chain components; treat arbitrary positive chain LCNs and then zeros separately.
Do not silently smooth a hard-zero model and claim to solve the original one.

Also audit graph recognition against the mathematical chain graph definition:
contraction must not hide a directed edge internal to an undirected connected
component or another forbidden semidirected cycle.

### 3.3 The compiled credal-network target

For separately specified local credal sets, write

$$
 \mathcal S_{\mathrm{CN}}=\{p_\theta:\theta_{j,\pi}\in K_{j,\pi}\},\qquad
 \mathcal K_{\mathrm{CN}}=\operatorname{co}(\mathcal S_{\mathrm{CN}}).
$$

The second set is the convex strong extension in the usual terminology.
Repository notes sometimes use “strong extension” for the first set. Keep this
distinction explicit: mixtures of independent distributions need not retain
independence. Linear expectation extrema agree over a generating set and its
convex hull; posterior extrema also agree under the appropriate positive-evidence
conditions, because a mixture posterior is an evidence-weighted average.

Where compilation is a sound relaxation, its bounds enclose the original LCN
bounds. Verify that premise, including positivity, solver bounds, and denominator
restrictions, before labeling a computed interval an outer bound.

Family-local syntax alone is not enough to claim equality of the compiled and
original models. In particular, multi-state root-block constraints can suffer
the within-row loss above. A useful sufficient-condition theorem must require
that the retained local representation describes the full relevant constraints,
in addition to structural and independence conditions.

### 3.4 Two sampling levels and three errors

An **outer search** selects a precise model $\theta$; an **inner estimator**
draws worlds or integrates them to evaluate $p_\theta(Q\mid e)$. Sampling
models does not endow epistemic uncertainty with a canonical probability law.
Uniform draws depend on parameterization; their average is not a lower or upper
probability. Here randomness is a search mechanism.

Track these errors separately:

| Error | Meaning | How to isolate it |
| --- | --- | --- |
| Representation | Compiled or restricted feasible set differs from the original LCN. | Compare certified extrema of the two models on tiny instances. |
| Search | The selected feasible models miss the target extremizers. | Use exact inference for candidate evaluation and compare with a certified target. |
| Monte Carlo | A candidate's posterior is estimated inaccurately. | Freeze candidates and compare independent estimates with exact evaluation. |

With exact evaluation of feasible models $\theta_1,\ldots,\theta_M$,

$$
 L\le \min_i p_{\theta_i}(Q\mid e)
 \le \max_i p_{\theta_i}(Q\mid e)\le U.
$$

Thus candidate extrema form an inner interval for their stated target. This
deterministic guarantee does not apply directly to noisy sample extrema or to
compiled-model candidates that violate the original LCN.

## 4. Literature map and research positioning

| Work | Established idea | Use and limitation here |
| --- | --- | --- |
| Mauá and Cozman (2020) [R1] | Survey of credal-network semantics, complexity, exact and approximate inference. | Primary map of the field; Section 5.2 traces stochastic vertex search to earlier annealing and genetic methods. |
| Cano, Cano and Moral (1994); Cano and Moral (1996) [R2] | Simulated annealing and genetic algorithms over sets of probabilities. | Prior art for stochastic outer search; original papers should be obtained before making a detailed novelty comparison. |
| Cano et al. (2007) [R3] | Hill-climbing and branch-and-bound for credal inference. | Compare local search and distinguish feasible incumbents from global bounding. |
| Antonucci et al. (2015) [R4] | A-LP fixes other local models and optimizes a remaining model; randomized objectives and restarts support search. | Strong exact-inner-inference baseline and a template for constraint-oracle updates without full vertex enumeration. |
| aGrUM/pyAgrum `CNMonteCarloSampling` [R5] | Randomly choose local credal vertices, construct BNs, and run a BN inference engine; the inspected C++ template defaults to `LazyPropagation`. | Immediately relevant random-model baseline. Its name does not mean it necessarily samples worlds for each BN. |
| Baudrit, Destercke and Wuillemin (2016) [R6] | Dynamic credal-network modeling and Monte Carlo exploration of local vertices. | Demonstrates practical use of model sampling; dynamic repeated-parameter semantics need separate treatment. |
| Troffaes (2017, 2018) [R7] | Shared importance samples across a family of distributions, iterative proposal adaptation, and analysis of lower-envelope bias and consistency. | Closest methodological foundation. Applying importance sampling to credal sets alone is already known. |
| Cheng and Druzdzel (2000) [R8] | Adaptive importance sampling for Bayesian-network evidence queries. | Inner-inference baseline and proposal design for rare evidence; adapting to one BN need not cover all credal extremizers. |
| Bidyuk and Dechter (2007) [R9] | Cutset sampling and Rao–Blackwellization: sample some variables and integrate the rest. | Structural extension when component tables or whole-network elimination are expensive. |
| Sangalli, Krak and De Campos (2025) [R10] | Conservative inference for credal chains through belief functions. | Related outer-approximation comparison on its supported subclass, not a general chain graph LCN sampler. |
| LCN papers and local factorization notes [R11] | Logical constraints, LMC semantics, chain graph connections, and existing approximate inference. | Define the target; retain the positivity and constraint-hosting hypotheses. |

The recovered review included full-text excerpts of [R1], [R4], [R6], [R7],
[R9] and the local LCN papers, as well as aGrUM source inspection. The older
annealing/genetic entries are supported here through [R1]'s bibliography, and
[R3] through that survey and bibliographic metadata. This is a focused review,
not evidence that no other relevant algorithm exists.

Before a publication novelty claim, extend the citation search around [R3],
[R4], and [R7], including generalized Bayes, robust Bayesian sensitivity,
imprecise importance sampling, stochastic programming, and constrained credal
inference. Investigate additional recent work and software rather than searching
only for the exact phrase “chain graph LCN sampling.” Record model semantics and
which level is sampled for every comparison.

The proposed contribution is the combination of constraint-preserving LCN
parameter search, structure-aware shared-sample evaluation, and explicit error
accounting. Whether that combination is novel remains a research question.

## 5. Research work packages

### WP0 — Establish semantic and numerical reference cases

Before evaluating speed, construct a small collection for which feasibility and
query extrema can be checked independently. The collection should include:

1. A precise BN, followed by a binary separately specified credal tree and a
   multiply connected credal DAG.
2. A complete two-atom component with a nontrivial constraint on a sum of states,
   including the within-row counterexample in Section 2.
3. A noncomplete undirected component, such as an undirected path, with an
   internal conditional independence that an unrestricted component table can
   violate.
4. Constraints tying parent configurations: partial-parent evidence or a
   disjunction of parent assignments, and an unconditional non-root assessment.
5. An explicitly coupled multi-family example, using
   [d4_biting.lcn](../../examples/d4_biting.lcn) as one existing stress case.
6. A hard-zero example, a zero-probability sentence condition, impossible query
   evidence, and evidence tending to zero.

For every fixture record the actual parsed graph, component scopes, sentences,
LMC assertions, and the numerical target. Adding a logical sentence can change
the graph, so a hand-drawn intended graph is not sufficient evidence.

Use brute-force joint enumeration for tiny models and SCIP global optimization
where appropriate. Compare `ExactInference` and `CredalJT` with independent
residual checks and analytical results when available. A solver timeout or local
optimum is a reference interval or incumbent, not an exact answer. `CredalJT`
also needs its running-intersection and constraint/evidence-hosting assumptions
checked. Do not use CCTE as an unconditional exact oracle: the local consolidated
notes already document counterexamples to that claim.

**Deliverable:** a semantic support matrix, reference fixtures, and certificates
or explicit unresolved gaps. Proceed with an LCN subclass only when both its
representation and candidate validation are understood.

### WP1 — Reproduce credal sampling with exact candidate evaluation

Implement or wrap the following methods on the same separately specified CN:

- Randomly select a vertex for each local row and evaluate the resulting precise
  BN exactly. Keep endpoint witnesses and running extrema; cache repeated models.
- Run pyAgrum `CNMonteCarloSampling` as an external baseline, checking its installed
  API and evidence behavior.
- Compare randomized coordinate search or annealing with the existing ApproxLP.
- Add random linear-objective LP draws for explicitly constrained local polytopes
  when enumerating their vertices is expensive. These draws are not uniform over
  vertices; they preferentially select vertices with larger normal cones.

Use a reproducible generator for all model draws and preserve the best complete
candidate for each endpoint. Exact inference at this stage isolates outer-search
quality. A uniform random local-vertex baseline has eventual finite-space
coverage if every configuration has nonzero sampling probability, but this says
little about practical convergence in an exponentially large search space.

Keep two modes distinct: original separately specified CN input, and a CN
compiled from an LCN. Report the latter's result against the compiled target.

**Deliverable:** an anytime baseline with known inner-bound meaning and measured
time spent compiling, enumerating, selecting models, and evaluating them.

### WP2 — Retain constraints when searching chain LCN models

Introduce a constraint representation that retains the original sentence rows
and relevant independence conditions. Do not make `.vtx` files or coordinate
intervals the sole representation of an LCN search domain.

Use three increasingly difficult parameter domains:

| Domain | Search mechanism | Required caveat |
| --- | --- | --- |
| Separate linear local credal polytopes | LP-selected vertices, coordinate optimization, and optional interior draws. | Local validity implies global validity only for the verified separate specification. |
| Linear constraints coupling entries of a component table or several rows | Block LP or linear-fractional optimization when the other model parameters make the full constraints linear in that block. | Whole-family scope does not itself imply such linearity or separability. |
| General LCN constraints and residual LMC equalities | Feasible initialization, constrained nonlinear block moves, restarts, and full candidate validation. | A fixed-other-block sentence expectation is affine in a CPT block, but an LMC residual can remain quadratic; this is not automatically A-LP. |

Build a validator that checks normalization, nonnegativity, **every original
sentence**, and **every LMC assertion**, either numerically or by a documented
structural implication. The existing cross-family checker alone cannot do this:
it selects constraints by scope, which misses lost within-family couplings.

For small instances reconstruct the full joint. For larger ones compute required
marginals by cached elimination or a junction tree. Account for the induced width
of all validation scopes; a single long logical sentence can erase the expected
savings. Monte Carlo feasibility checks are statistical approximations and must
not silently replace exact checks in the mode claiming feasible witnesses.

Initialize from an existing feasible joint/cluster solve or a feasibility search
with multiple starts. Store residuals and the status of every accepted model.
Failure to find a model does not establish infeasibility. For equality-constrained
sets, rejection from a continuous box has probability zero of hitting the set
exactly; parameterize the equalities or use constrained proposals instead.

Study block updates and occasional coordinated moves between strongly coupled
families. Feasible regions can be disconnected, and single-block moves can be
trapped even when their local solves are exact. Penalty objectives can guide
search, but an infeasible penalty minimizer is not an endpoint witness.

Hit-and-run is an optional exploration tool for convex polytopes in their affine
hull, with equality reduction and conditioning first [R12]. It is not a generic
sampler for the nonconvex set induced by LMC equalities. Interior sampling also
needs boundary-directed optimization to find extrema efficiently.

Do not claim completeness from filtering existing interval vertices. For an
elementary illustration, the square $[0,1]^2$ intersected with
$x+y=1/2$ has a nonempty feasible segment, yet none of the square's four
vertices survives. Coupled constraints can create entirely new feasible extreme
points. Nonlinear independence constraints introduce further complications.

**Deliverable:** exact-evaluation search producing validated original-LCN
witnesses on a documented subclass. Measure validation cost and acceptance rate
before adding an approximate inner estimator.

### WP3 — Share importance samples across candidate models

For a normalized proposal $q$ on worlds, choose support so that
$q(x)>0$ whenever any admissible target model has $p_\theta(x)>0$.
Draw $X_1,\ldots,X_N\sim q$, and compute

$
 w_\theta(x)=\frac{p_\theta(x)}{q(x)},\qquad
 \widehat r_\theta=
 \frac{\sum_{k=1}^N w_\theta(X_k)\mathbf1_e(X_k)\mathbf1_Q(X_k)}
 {\sum_{k=1}^N w_\theta(X_k)\mathbf1_e(X_k)}.
$

The numerator and denominator can be estimated using the same cached worlds for
many models. Ordinary importance estimates of each expectation are unbiased for
a fixed model under the usual conditions, but their ratio generally is not.
For unconditional expectations of normalized models, also retain the ordinary
importance estimate instead of automatically self-normalizing.

A proposal over the unobserved variables with evidence clamped is often better
than sampling all worlds and discarding those inconsistent with evidence. Then
the target weight is $p_\theta(z,e)/q(z)$. A likelihood-weighting proposal
must account for partially observed compound components by summing or sampling
their remaining states correctly.

Compare three levels:

1. Independent ancestral/likelihood-weighting runs per precise model.
2. A fixed shared proposal and cached weights across candidate models.
3. An adaptive proposal built from current lower and upper endpoint candidates,
   with a defensive full-support component.

For example, a tractable mixture can combine several precise-model proposals
with a broad proposal: $q=\eta q_0+(1-\eta)\sum_j\alpha_jq_j$,
where $\eta>0$. Exact evaluation of the mixture density is required for the
weights. A mixture aimed only at the lower extremizer may miss the upper one.
Full support prevents impossible importance ratios but does not guarantee low
variance in high dimensions.

Begin adaptation in batches: fit a proposal on earlier data, freeze it, then
draw a fresh batch. Retain the proposal density associated with each draw.
Do not reweight old draws as if they came from a newly chosen proposal. More
elaborate multiple-importance or adaptive weighting requires its own analysis.

Use log probabilities, stable sums, cached component configurations, and
incremental updates of weights when a model changes one block. Under full-world
evaluation, a batch of N weights costs roughly O(Nm) table lookups for m
components after preprocessing; updating one table can reduce that arithmetic.
The number and size of table entries, validation costs, and proposal-density
evaluation remain separate costs.

The loop should be:

```text
initialize feasible precise models and a support-covering proposal
repeat within the budget:
    draw a fresh batch from the frozen proposal; cache densities and states
    propose model changes using shared-sample objective estimates
    validate candidates against the stated model constraints
    retain promising lower and upper candidates
    evaluate retained candidates exactly or on independent validation samples
    update endpoint records with witnesses, residuals, and uncertainty
    adapt the next proposal using training information
return endpoint records, target definition, and unresolved gaps
```

Optimization on noisy estimates can select unusually favorable noise.
Troffaes [R7] analyzes this issue for lower envelopes. In particular, for unbiased
fixed-model estimators the expected minimum is no greater than the minimum of
their expectations. This differs from the inward search error of a finite set of
exactly evaluated candidates. The two errors can cancel numerically without
either being small.

Keep search samples and final validation samples independent. Freeze the chosen
candidates before validation, and use simultaneous confidence accounting for
the candidates and queries being reported. If validation data are then used to
adapt or repeatedly stop, use fresh data, an allocated error budget, or a proven
time-uniform procedure.

Effective sample size,
$\mathrm{ESS}=(\sum_k v_k)^2/\sum_k v_k^2$ for the relevant nonnegative
weights $v_k$, is a useful diagnostic. Report it for evidence-weighted estimates,
along with maximum normalized weight, zero-denominator events, and variability
across independent runs. ESS alone is not a confidence interval or a global
optimization certificate.

**Deliverable:** a shared-sample method that improves runtime at matched accuracy
on CNs first, then on the validated LCN subclass, with separately measured
statistical error and search error.

### WP4 — Exploit structure with partial exact inference

After WP3, test cutset sampling/Rao–Blackwellization [R9]: select a subset of
variables or components to sample and integrate the remainder exactly. Choose
cutsets using residual induced width rather than component count alone. The
conditional exact computation must include the relevant evidence and query.

For undirected components, small tables may be sampled directly. Larger sparse
components can motivate junction-tree or blocked Gibbs calculations under a
verified conditional factorization. A Gibbs update must include all factors
affected by the updated variables, including descendant/evidence contributions;
the ancestral factor alone is not the posterior full conditional.

Avoid starting with nested MCMC for both model parameters and worlds. That adds
mixing, support, and selection issues at both levels. Introduce MCMC only after
the exact and importance baselines show a specific memory or variance bottleneck,
and then verify irreducibility and include autocorrelation-aware diagnostics.

**Deliverable:** a measured tradeoff among residual width, runtime, memory, and
variance. Retain the simpler estimator if structural integration does not pay
for its overhead.

## 6. Theoretical results to target

Separate standard facts we can rely on from proposed theorems that require proof.

1. **Feasible-candidate bounds — standard, with target identification.** Exact
   evaluation of feasible candidates gives the inner interval in Section 3.4.
   Formalize the validator-to-soundness implication for the supported LCN class.

2. **Representation theorem — research prerequisite.** Establish sufficient
   conditions for the component parameterization to cover the intended LCN set.
   Treat positive models and hard-zero models separately. Provide counterexamples
   where interval compilation or unrestricted component tables change the target.

3. **Outer-search consistency — limited standard result, then extension.** On a
   finite product of local vertex sets with exact inference, independent draws
   assigning positive mass to every choice eventually find all extrema almost
   surely. For continuous feasible search, require coverage near optimizers and
   continuity; local improvement alone is insufficient. Exact equalities and
   disconnected feasible sets require explicit treatment.

4. **Uniform inner-estimator convergence — research theorem for this algorithm.**
   For a finite world space and a fixed full-support q, normalized table products
   are continuous in the parameters and importance weights are uniformly bounded
   by a possibly enormous constant. With a uniform positive evidence bound,
   establish uniform convergence of the ratios. Combining that with sufficiently
   accurate global optimization or dense feasible search can give convergence of
   endpoint estimates. Pointwise Monte Carlo convergence alone cannot justify
   exchanging optimization and limits. Adaptive proposals need additional
   assumptions rather than inheriting this fixed-q argument automatically.

5. **Confidence statements — derive exactly what is covered.** A confidence
   interval for a selected model's posterior does not enclose the global LCN
   endpoint. If simultaneous candidate intervals are $[a_i,b_i]$, they enclose
   the sampled minimum in $[\min_i a_i,\min_i b_i]$ and the sampled maximum in
   $[\max_i a_i,\max_i b_i]$. They do not bound unsampled models. Valid outward
   endpoint bounds require a global relaxation, a uniform error bound combined
   with controlled optimization error, or another certified argument.

6. **Useful stopping certificate — combine bounds for the same target.** If a
   certified outer method gives $L_{\rm out}\le L$ and $U\le U_{\rm out}$,
   and exact feasible candidates give $L\le L_{\rm in}$ and
   $U_{\rm in}\le U$, stop when both endpoint gaps meet the tolerance.
   For statistical candidate evaluation replace these with appropriately oriented
   simultaneous confidence bounds. Without such information, report budget
   exhaustion or empirical stabilization, not convergence to the exact answer.

No polynomial-time general inference guarantee is expected: both precise BN
inference and credal optimization have hard cases [R1, R4]. The research target
is a useful accuracy/cost tradeoff on identifiable structural subclasses.

## 7. Experimental design

### Instance families

Use the WP0 cases as correctness gates, then expand along controlled axes:

| Axis | Proposed pilot settings |
| --- | --- |
| Graph structure | Trees, polytrees, multiply connected DAGs, and chain graphs with small complete/noncomplete components. |
| Number of atoms | 4–12 for joint reference calculations; then 20, 50, 100 where component and validation widths permit. |
| Component size | 1, 2, 4, 6 binary atoms initially; report the actual $2^k$ state cardinality. |
| Precision | Precise tables and progressively wider credal sets. |
| Logical coupling | Separate rows, within-row sums, across-row constraints, and across-family constraints. |
| Evidence | None, ordinary evidence, deliberately rare evidence, and structural-zero cases. |
| Numerical domain | Positive models first, then support-restricted and boundary cases. |

Reuse [benchmarks](../../benchmarks) and examples such as
[alarm.lcn](../../examples/alarm.lcn), [chain.lcn](../../examples/chain.lcn), and
[smokers.lcn](../../examples/smokers.lcn) after classifying their parsed structure.
The repository's “chain” filename or generator label alone does not identify
the difficulty or semantic subclass.

Use paired instances to isolate losses: retain the same graphical structure
while changing one assessment from a separate row constraint to a coupled one,
where possible. Verify that the syntax did not unexpectedly alter the graph.
For wider-scale synthetic data, construct sentences around a known feasible
witness and verify its LMC rather than assuming consistency.

### Baselines and ablations

The primary baseline set is exact/certified reference optimization on small
instances, existing ApproxLP, random-model sampling with exact inference,
pyAgrum CN Monte Carlo, and per-model likelihood weighting. Add the existing
ARIEL/IBP methods as empirical comparisons with each method's target and guarantee
documented. Use exact credal VE only where its representation and query mode are
appropriate; evidence and pruning require special care.

Required ablations are:

- Compiled intervals versus retained local constraints versus full LCN validation.
- Random outer search versus query-directed block search, with the same evaluator.
- Exact evaluation versus independent Monte Carlo versus shared samples.
- Fixed versus adaptive proposals; one-extremizer versus multiple-extremizer mixtures.
- Full-world sampling versus partial exact integration at several residual widths.
- Independent validation versus reuse of search samples, to measure selection bias.

Report both total wall time including preprocessing and inference-only time with
caches already built. Count LP/NLP solves, vertex enumeration, precise inference
calls, and feasibility checks. An apparent sampling speedup that excludes an
expensive `.vtx` build is not an end-to-end speedup.

### Metrics and statistical protocol

On models with certified L and U, report each endpoint's absolute error and
direction, interval width error, infeasible-witness rate, time to a specified
endpoint error, and memory. A narrower estimated interval is not automatically
better. On larger models report the remaining gap to available certified bounds
and empirical stability without substituting another approximation for truth.

For sampling methods also report evidence mass estimates, ESS, maximum normalized
weight, zero-denominator frequency, independent validation error, and confidence
coverage for the particular quantity covered. Track compilation, search, and
Monte Carlo contributions using the controlled comparisons in Section 3.4.

Use a small pilot to set budgets, followed by a frozen evaluation configuration.
An initial proposal is 10–20 instances per manageable family and 20 independent
seeds per stochastic method, with wall-time checkpoints such as 1, 10, 60, and
300 seconds. These are planning values, to be reduced or expanded after profiling.
Use a separate stress set for large components and rare evidence rather than a
full Cartesian product of every axis.

Paired seeds can reduce comparison noise where randomization is comparable;
independent replicate runs remain necessary. Tune proposal and search settings
on separate instances. Publish median and quantile performance, failures,
timeouts, solver gaps, and the distribution of constraint residuals.

### Hypotheses and decision rules

| Hypothesis | Supporting evidence | If it fails |
| --- | --- | --- |
| H1: shared samples amortize precise-model evaluation | Lower total runtime at matched endpoint error on held-out CNs, including validation. | Retain exact inference or independent sampling for that regime. |
| H2: adaptive mixtures help rare-evidence or widely varying models | Better endpoint accuracy and independent validation quality at equal time; ESS is supporting evidence only. | Use fixed proposals or cutset integration; narrow the claimed applicability. |
| H3: preserving constraints matters in practice | Smaller original-LCN error with feasible witnesses on paired coupling examples. | Recheck whether the benchmark constraints are active and whether compilation already captures them. |
| H4: graph structure reduces sampling/validation cost | A useful frontier in runtime, memory, and error as residual width varies. | Limit the method to bounded validation width or use a stronger structural representation. |

Define any numerical success threshold after the pilot and before the main
comparison. Do not choose it retrospectively to match favorable runs.

## 8. Proposed implementation boundaries

Implement new research code only after the semantic gates are met. A possible
location is `lcn/inference/marginal/sampling/`; the following names describe
proposed interfaces, not existing classes:

| Interface | Responsibility |
| --- | --- |
| `PreciseModel` | Component tables, stable log-probability evaluation, sampling, and immutable candidate identity. |
| `CredalDomain` | Supported parameterization and feasible candidate/block proposals; separate CN and original-LCN domains. |
| `ConstraintOracle` | Original sentences, LMC coverage, residuals, and exact/statistical validation status. |
| `QueryEvaluator` | Exact elimination, likelihood weighting, shared importance estimates, or partial exact integration. |
| `Proposal` | Sampling and evaluable density with explicit support and version identity. |
| `EndpointSearch` | Lower/upper search, candidate cache, budget allocation, and independent reevaluation. |
| `SamplingResult` | Endpoint estimates, witnesses, target, timing, residuals, uncertainty, and unresolved gaps. |

Each result should identify:

- Original LCN or compiled CN target; semantic subclass and any evidence floor.
- Exact, statistical, or unchecked feasibility status and validation tolerance.
- Estimator type, sample counts, seed, proposal version, and evidence handling.
- Endpoint witness IDs, confidence meaning, and any optimization certificate.
- Compilation, enumeration, search, evaluation, and validation costs separately.

Add experiment-runner entries after an API-level prototype works. Avoid changing
the meaning of existing `approxlp`, `cve`, or `.vtx` outputs. Reuse graph/indicator
code and backend libraries, while separating an LCN constraint domain from its
optional interval relaxation.

The first implementation should cover normalized tabular factors, binary atoms,
unconditional singleton queries, and exact candidate evaluation. Add evidence
before adaptive importance sampling; add large-component MCMC last, if needed.

Future tests should check meaningful invariants: reconstructed joints normalize,
component-state ordering is correct, candidate witnesses satisfy every original
constraint, precise-model estimates agree with analytical examples, impossible
evidence is distinguished from failure to observe evidence, and the two
counterexamples in this plan cannot be misclassified as exact compilation.
Statistical coverage requires a reproducible experiment, not a fragile single-seed
unit assertion.

## 9. Milestones, risks, and first concrete deliverable

The schedule is a tentative sequence for one researcher, not a runtime estimate
or commitment to finish unresolved proofs within a fixed week.

| Period | Work | Exit criterion |
| --- | --- | --- |
| Weeks 1–2 | WP0: semantic audit, tiny fixtures, representation statements, literature follow-up. | A verified initial subclass and independently checkable reference answers. |
| Weeks 3–4 | WP1: CN random-model and query-directed baselines with exact evaluation. | Reproducible anytime curves with feasible endpoint witnesses and full cost accounting. |
| Weeks 5–6 | WP2: retain constraints and validate original-LCN models. | Feasible witnesses on within-row, across-row, and across-family cases; known coverage limits. |
| Weeks 7–9 | WP3: shared importance samples and adaptive proposals. | Held-out accuracy/runtime comparisons and an independent validation protocol. |
| Weeks 10–11 | WP4 where justified, larger stress cases, theorem refinement. | Evidence for the applicable structural regime and honest failure cases. |
| Week 12 | Frozen evaluation, ablations, write-up, and reproducibility materials. | Claims traceable to proofs or experiments, with no ambiguity about the target set. |

The main risk is that enforcing all LCN constraints costs as much as the inference
being accelerated. Profile this in WP2 before investing in sophisticated
proposals. A second risk is that feasible search has poor coverage because the
model space is thin or disconnected; mitigate it with structured parameterizations,
coordinated blocks, and multiple feasible starts, and state any remaining limit.
A third risk is weight collapse across highly different extremizers; use mixtures,
independent endpoint proposals, or partial exact integration as the evidence
supports. None of these mechanisms guarantees efficient inference universally.

The recommended first implementation deliverable is deliberately small:
**random and query-directed precise-model search with exact evaluation, plus a
complete candidate validator on tiny chain LCNs**. That provides useful baselines
and establishes what is being optimized. Shared-sample importance inference is
the next deliverable once that foundation passes the semantic checks.

## 10. References and reading priorities

**[R1]** D. D. Mauá and F. G. Cozman (2020), *Thirty years of credal networks:
Specification, algorithms and complexity*, International Journal of Approximate
Reasoning 126, 133–157.
[DOI](https://doi.org/10.1016/j.ijar.2020.08.009).
The repository includes a [local copy](../1-s2.0-S0888613X20302152-main%20(1).pdf).
Read first, especially the strong-extension and inference sections.

**[R2]** A. Cano, J. E. Cano and S. Moral (1994), *Convex sets of probabilities
propagation by simulated annealing*, IPMU, pp. 4–8; A. Cano and S. Moral (1996),
*A genetic algorithm to approximate convex sets of probabilities*, IPMU,
pp. 859–864. Bibliographic details and methodological attribution here follow
[R1], references 63–64; original full-text verification remains follow-up work.

**[R3]** A. Cano, M. Gómez, S. Moral and J. Abellán (2007), *Hill-climbing and
branch-and-bound algorithms for exact and approximate inference in credal
networks*, International Journal of Approximate Reasoning 44, 261–280.
[DOI](https://doi.org/10.1016/j.ijar.2006.07.020).

**[R4]** A. Antonucci, C. P. de Campos, D. Huber and M. Zaffalon (2015),
*Approximate credal network updating by linear programming with applications
to decision making*, International Journal of Approximate Reasoning 58, 25–38.
[DOI](https://doi.org/10.1016/j.ijar.2014.10.003).
Read the optimization formulation, randomized initialization, and restart scheme.

**[R5]** aGrUM/pyAgrum, *Credal networks* documentation and
`CNMonteCarloSampling` implementation, inspected during the September 2026 review.
[Documentation](https://pyagrum.readthedocs.io/en/latest/credalNetwork.html),
[C++ interface](https://github.com/agrumery/aGrUM/blob/master/src/agrum/CN/inference/CNMonteCarloSampling.h),
[implementation](https://github.com/agrumery/aGrUM/blob/master/src/agrum/CN/inference/CNMonteCarloSampling_tpl.h).
These are moving links; pin a release and source revision for the experiments.

**[R6]** C. Baudrit, S. Destercke and P.-H. Wuillemin (2016), *Unifying parameter
learning and modelling complex systems with epistemic uncertainty using
probability interval*, Information Sciences 367–368, 630–647.
[DOI](https://doi.org/10.1016/j.ins.2016.07.003),
[accepted manuscript](https://hal.sorbonne-universite.fr/hal-01346202v1).
Section 4.2.2 describes Monte Carlo and guided vertex-search inference.

**[R7]** M. C. M. Troffaes (2017), *A Note on Imprecise Monte Carlo over Credal
Sets via Importance Sampling*, PMLR 62, 325–332.
[Paper](https://proceedings.mlr.press/v62/troffaes17a.html).
Also Troffaes (2018), *Imprecise Monte Carlo simulation and iterative importance
sampling for the estimation of lower previsions*.
[DOI](https://doi.org/10.1016/j.ijar.2018.06.009),
[preprint](https://arxiv.org/abs/1806.10404).
Read both before deriving estimator guarantees; distinguish fixed-model
unbiasedness, lower-envelope bias, self-normalized estimates, and optimization
assumptions. The iterative method is a precedent, not a new contribution here.

**[R8]** J. Cheng and M. J. Druzdzel (2000), *AIS-BN: An Adaptive Importance
Sampling Algorithm for Evidential Reasoning in Large Bayesian Networks*.
[DOI](https://doi.org/10.1613/jair.764).

**[R9]** B. Bidyuk and R. Dechter (2007), *Cutset Sampling for Bayesian Networks*,
Journal of Artificial Intelligence Research 28, 1–48.
[DOI](https://doi.org/10.1613/jair.2149).

**[R10]** M. Sangalli, T. Krak and C. De Campos (2025), *Towards conservative
inference in credal networks using belief functions: the case of credal chains*.
[Preprint](https://arxiv.org/abs/2507.07619).
Check its independence convention and chain subclass before using it as a bound.

**[R11]** Local LCN sources:
[Logical Credal Networks](../Logical_Credal_Networks.pdf),
[revised IJAR manuscript](../LCN_IJAR_Revised.pdf),
[ISIPTA 2025 manuscript](../ISIPTA2025_LCNs.pdf), and
[Approximate Inference in Logical Credal Networks](https://doi.org/10.24963/ijcai.2023/632)
([local copy](../LCN_ARIEL_IJCAI2023.pdf)).
Prioritize the revised manuscript's Section 5 on chain graphs and positivity,
then the repository's [consolidated properties](../research-chain-lcn/properties.tex).
Treat local draft claims as material to verify against their hypotheses and code.

**[R12]** L. Lovász and S. Vempala (2004), *Hit-and-run from a corner*, STOC.
[DOI](https://doi.org/10.1145/1007352.1007403).
Follow-up reading for convex-polytope exploration; its guarantees do not directly
transfer to nonconvex LCN parameter domains.
