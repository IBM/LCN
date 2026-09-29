# Literature and source notes

Companion to [the research plan](research-plan.md). Reviewed 29 September 2026.
This is a focused literature review, not an exhaustive priority search. Entries
state whether original full text, a primary paper's treatment of earlier work,
or bibliographic metadata was inspected. Version-specific theorem numbers refer
to the linked version.

## LCN foundations

**[R1] Radu Marinescu, Haifeng Qian, Alexander Gray, Debarun Bhattacharjya,
Francisco Barahona, Tian Gao, Ryan Riegel, and Pravinda Sahu (2022).
_Logical Credal Networks._ NeurIPS 35, 15325–15337.**

[Local paper](../Logical_Credal_Networks.pdf).

Primary local text inspected, especially the definitions and Section 4.3.
The fraud experiment combines a data-derived model with logical rules and
empirically estimated rule intervals. It demonstrates that learning-related
use of data predates this proposal; it is not a general structure-learning
algorithm or a finite-sample coverage theorem for learned LCNs.

The description estimates some rule intervals from the collections it calls
test sets. The new plan therefore specifies independent fitting, selection,
calibration, and final-test roles rather than copying that protocol. This is
an experimental-design distinction, not a claim that the original paper studied
the same statistical guarantee.

**[R2] Fabio G. Cozman et al. (2024). _Markov conditions and factorization in
logical credal networks._ International Journal of Approximate Reasoning 172,
109237.**

[DOI](https://doi.org/10.1016/j.ijar.2024.109237)
· [local revised manuscript](../LCN_IJAR_Revised.pdf).

The local full text was inspected in the preceding GMC research task and its
relevant findings reused here. The local manuscript has a later preprint date
than the journal reference. It explains the original LCN structure, the chain
factorization setting, and differences among local/global Markov conditions.
Use it to define the learning target and audit compiler equivalence; do not
assume every graph or factor edit preserves the original LCN family.

**[R3] Fabio G. Cozman, Radu Marinescu, Junkyu Lee, Alexander Gray, and
Denis D. Mauá (2025). _Dealing with cycles in graph-based probabilistic models:
the case of Logical Credal Networks._ ISIPTA, PMLR 290, 93–102.**

[Proceedings](https://proceedings.mlr.press/v290/cozman25a.html)
· [published paper](https://raw.githubusercontent.com/mlresearch/v290/main/assets/cozman25a/cozman25a.pdf)
· [local manuscript](../ISIPTA2025_LCNs.pdf).

The published text was inspected in the preceding GMC task. It extends the
factorization setting to directed–undirected mixed graphs using Koster's
theory and already formulates inference as multilinear optimization. Positivity,
normalization, and the distinction between LMC and GMC remain material for
learning. This is an optional semantic extension of the learning program,
not its default distribution family.

## Learning with imprecise probabilities

**[R4] Denis D. Mauá and Fabio G. Cozman (2020). _Thirty years of credal
networks: Specification, algorithms and complexity._ International Journal of
Approximate Reasoning 126, 133–157.**

[DOI](https://doi.org/10.1016/j.ijar.2020.08.009)
· [local paper](../1-s2.0-S0888613X20302152-main%20%281%29.pdf).

Primary text inspected, especially Section 7 on elicitation/learning and the
inference/complexity discussion. It explicitly separates learning network
structure from learning local credal sets, discusses whether structural
uncertainty should be represented by multiple graphs, and identifies [R5, R7]
as important predecessors. Its strong-extension and optimization discussions
also explain why compact factorization is not a general exact-tractability
guarantee.

**[R5] Andrés R. Masegosa and Serafín Moral (2014). _Imprecise probability
models for learning multinomial distributions from data. Applications to
learning credal networks._ International Journal of Approximate Reasoning
55(7), 1548–1569.**

[DOI](https://doi.org/10.1016/j.ijar.2013.09.019).

Author/title/venue metadata checked; the technical overview was read through
[R4, Section 7], not through an independently obtained original full text.
The survey reports refinements to the basic IDM and approaches involving sets
of network structures and extensively specified credal networks. Read the
original before selecting a final estimator or claiming a new treatment of
structural uncertainty. The LCN-specific gap is the induced logical/Markov
structure and cross-formula coupling, not learning multinomial credal sets
in isolation.

**[R6] Peter Walley (1996). _Inferences from Multinomial Data: Learning About
a Bag of Marbles._ Journal of the Royal Statistical Society, Series B 58(1),
3–34.**

[DOI](https://doi.org/10.1111/j.2517-6161.1996.tb02065.x).

Bibliographic metadata checked. The IDM update used in the plan was read in
[R16, Section 3.1, Equation 3] and contextualized by [R4]. The original full
text was not independently inspected here. The intervals are envelopes of
posterior-predictive probabilities under a set of Dirichlet priors. They must
not be described as distribution-free frequentist confidence intervals, and
their multinomial entries remain constrained by normalization.

**[R7] Serafín Moral (2019). _Learning with imprecise probabilities as model
selection and averaging._ International Journal of Approximate Reasoning 109,
111–124.**

[DOI](https://doi.org/10.1016/j.ijar.2019.04.001).

Metadata checked; methodological positioning inspected through [R4]. Read the
original when designing the structural-envelope/model-averaging comparison.
Also relevant is Giorgio Corani and Marco Zaffalon (2008), _Credal Model
Averaging: An Extension of Bayesian Model Averaging to Imprecise Probabilities_.
[DOI](https://doi.org/10.1007/978-3-540-87479-9_35).
This second title/DOI was verified through metadata and its citation in [R4].

A union of model sets, a Bayesian average with fixed weights, and an imprecise
set of mixing weights express different uncertainties. None automatically
retains the pointwise independencies of a single learned LCN graph.

## Structure learning and joint learning foundations

**[R8] Dependence trees and graph structure search.**

C. K. Chow and C. N. Liu (1968), _Approximating discrete probability
distributions with dependence trees_, IEEE Transactions on Information Theory
14(3), 462–467.
[DOI](https://doi.org/10.1109/TIT.1968.1054142).

Metadata checked; the tree-learning construction and its use in a circuit
learner were inspected in [R12, Section 4.1]. Maximum-weight spanning trees
using empirical mutual information provide the controlled initialization and
precise-model baseline. They do not optimize the plan's full logical/credal
objective.

David Maxwell Chickering (2002), _Optimal Structure Identification With Greedy
Search_, JMLR 3, 507–554.
[Journal page](https://www.jmlr.org/papers/v3/chickering02b.html).

Journal metadata and abstract inspected; the full proofs were not reviewed.
This is a baseline for searching equivalence classes of Bayesian-network
structures. Its assumptions and decomposable scores cannot be transferred
unchanged to general LCN sentence search. CI-based discovery and Bayesian
network scoring should receive a deeper implementation-stage review; the plan
does not make new consistency claims for those classical methods.

**[R9] Nir Friedman (1998). _The Bayesian Structural EM Algorithm._ UAI,
129–138.**

[ArXiv record](https://arxiv.org/abs/1301.7373)
· [paper](https://arxiv.org/pdf/1301.7373).

Original conference-paper scan inspected for its motivation, expected-score
updates, and convergence discussion, including Theorem 3.1. The arXiv upload
is from 2013; the record identifies the original as UAI 1998. Structural EM
already combines parameter estimation with structure search under incomplete
data. The proposed LCN extension must specify its objective and prove any
monotonicity claim after adding logical constraints, uncertainty penalties,
or approximate/set-valued E-steps.

## Learning probabilistic circuits

**[R10] Robert Gens and Pedro Domingos (2013). _Learning the Structure of
Sum-Product Networks._ ICML, PMLR 28.**

[Paper](https://proceedings.mlr.press/v28/gens13.pdf).

Primary full text inspected, especially the recursive learning scheme and
the discussion of parameter fitting. LearnSPN partitions variables into
approximately independent groups to form products, and clusters examples to
form sums. These are useful structure proposals. Approximate independence
tests used by a learner are not population-CI certificates, and learned latent
clusters do not supply observed multinomial counts without further assumptions.

The historical precursor is Daniel Lowd and Pedro Domingos (2008), _Learning
arithmetic circuits_, UAI, 383–392, verified through [R10, R16]. Its original
full text remains follow-up reading for detailed novelty comparisons of
learning with circuit-size penalties.

**[R11] Yitao Liang, Jessa Bekker, and Guy Van den Broeck (2017). _Learning
the Structure of Probabilistic Sentential Decision Diagrams._ UAI.**

[Official accepted-paper listing](https://www.auai.org/uai2017/accepted.php)
· [paper](https://www.auai.org/uai2017/proceedings/papers/291.pdf).

Primary full text inspected, particularly Sections 2–4. Complete-data parameter
MLEs are normalized context/branch counts. Vtree learning and split/clone moves
support structure learning while preserving the root's logical support.
Proposition 2 establishes validity/support preservation for those operations.
Their precise likelihood locality is useful, but does not establish locality
of LCN constraints or preservation of exact credal posterior inference.

The foundational PSDD paper is Doga Kisa, Guy Van den Broeck, Arthur Choi,
and Adnan Darwiche (2014), _Probabilistic Sentential Decision Diagrams_, KR,
as cited and explained by [R11, R16]. The original was not separately reviewed.

**[R12] Meihua Dang, Antonio Vergari, and Guy Van den Broeck (2020).
_Strudel: Learning Structured-Decomposable Probabilistic Circuits._ PGM,
PMLR 138.**

[Proceedings](https://proceedings.mlr.press/v138/dang20a.html)
· [paper](https://proceedings.mlr.press/v138/dang20a/dang20a.pdf).

Primary full text inspected selectively, especially the circuit properties and
Section 4. Strudel initializes from a Chow–Liu tree compiled into a structured
circuit, then applies greedy splits. It provides a particularly suitable
baseline for testing whether uncertainty-aware split selection improves on
ordinary precise learning at matched size and compute budgets.

**[R13] Robert Peharz et al. (2020). _Einsum Networks: Fast and Scalable
Learning of Tractable Probabilistic Circuits._ ICML, PMLR 119.**

[Proceedings](https://proceedings.mlr.press/v119/peharz20a.html)
· [paper](https://proceedings.mlr.press/v119/peharz20a/peharz20a.pdf).

Primary abstract and implementation/learning discussion inspected. The work
organizes circuit operations into batched einsum computations and simplifies
EM through automatic differentiation. It is an engineering and precise-fitting
precedent, not a theorem about optimizing over arbitrary credal parameter
regions. The plan proposes measuring credal optimization and calibration costs
in addition to ordinary circuit evaluation throughput.

**[R14] Antonio Vergari, YooJung Choi, Anji Liu, Stefano Teso, and Guy
Van den Broeck (2021). _A Compositional Atlas of Tractable Circuit Operations:
From Simple Transformations to Complex Information-Theoretic Queries._
arXiv:2102.06137v1.**

[Record and version history](https://arxiv.org/abs/2102.06137)
· [inspected version](https://arxiv.org/pdf/2102.06137v1).

The retrieved manuscript identifies itself as version 1. Its framework,
structural-property definitions, and tractability discussion were inspected
selectively. It treats inference as a composition of operations with separate
compatibility conditions, rather than assuming every query is easy because
a model is a circuit. This supports the plan's query contract and its treatment
of formula compilation, products, ratios, and learning objectives.

The broader overview by YooJung Choi, Antonio Vergari, and Guy Van den Broeck
(2020), _Probabilistic Circuits: A Unifying Framework for Tractable Probabilistic
Modeling_, is cited by this manuscript and [R12]. Attempts to retrieve a
standalone copy returned unavailable pages, so no independent full-text review
of that report is claimed here.

## Credal circuits and tractable credal inference

**[R15] Denis D. Mauá, Fabio G. Cozman, Diarmaid Conaty, and Cassio P.
de Campos (2017). _Credal Sum-Product Networks._ ISIPTA, PMLR 62, 205–216.**

[Proceedings](https://proceedings.mlr.press/v62/mau%C3%A117a.html)
· [paper](https://proceedings.mlr.press/v62/mau%C3%A117a/mau%C3%A117a.pdf).

Primary full text inspected, especially Section 3. Theorem 1 gives polynomial
lower/upper evaluation of observation indicators for separately specified local
weight polytopes, including shared circuit structures. Theorem 4 treats
univariate conditional expectations when each internal node has at most one
parent. The general conditional-expectation hardness result prevents extending
the observation theorem to arbitrary utilities or formula queries.

The experiments robustify learned SPNs by perturbing parameters, making
post-hoc credalization a necessary baseline for this research. Generalized
Bayes comparisons also require care with zero evidence and strict versus weak
inequalities; use the plan's explicit evidence convention rather than copying
an equivalence without its assumptions.

The journal continuation is Denis D. Mauá, Diarmaid Conaty, Fabio G. Cozman,
Katja Poppenhaeger, and Cassio P. de Campos (2018), _Robustifying sum-product
networks_, IJAR 101, 163–180.
[DOI](https://doi.org/10.1016/j.ijar.2018.07.003).
Its metadata and relationship to the earlier work were checked; the journal
full text was not independently reviewed. Compare its extensions before
finalizing algorithmic novelty claims.

**[R16] Lilith Mattei, Alessandro Antonucci, Denis D. Mauá, Alessandro
Facchini, and Julissa Villanueva Llerena (2020). _Tractable inference in credal
sentential decision diagrams._ IJAR 125, 26–48.**

[DOI](https://doi.org/10.1016/j.ijar.2020.06.005)
· [inspected manuscript](https://arxiv.org/pdf/2008.08524v1).

Journal metadata checked; the manuscript was inspected in detail for Sections
3–6, the support assumptions, parameter learning, and theorem statements.
Key results used in the plan are:

- Section 3.1 supplies the IDM multinomial update.
- Definition 5 imposes positive lower probabilities on allowed local branches,
  with zero mass reserved for specified logical impossibilities.
- The strong extension is a convex hull of compatible precise PSDDs.
- Example 3 learns credal parameters from logical-context counts using IDM.
- Theorem 3 gives exact marginal bounds without a singly connected topology
  restriction.
- Theorem 4 and the following discussion give exact singleton conditional
  inference for singly connected CSDDs; the multiply connected case can supply
  conservative outer bounds, with additional exactness analysis.

This is the closest predecessor for learning a logical credal circuit. The
research plan must go beyond reusing IDM estimates on a PSDD. Its proposed
additions are an LCN semantic bridge, calibration after structure selection,
coupled assessments, and learning moves that preserve the desired credal
query guarantees. The strict support assumptions also make empty or unseen
contexts a substantive boundary case.

**[R17] Binary-polytrees and complexity boundaries.**

Enrico Fagiuoli and Marco Zaffalon (1998), _2U: an exact interval propagation
algorithm for polytrees with binary variables_, Artificial Intelligence 106,
77–107.
[DOI](https://doi.org/10.1016/S0004-3702(98)00089-7).

Title/authors/venue/pages/DOI checked through metadata. The tractable binary
polytree setting was reviewed through [R4]; original full-text verification is
required before implementing the fragment's backend and its boundary cases.

Denis D. Mauá, Cassio P. de Campos, Alessio Benavoli, and Alessandro Antonucci
(2014), _Probabilistic Inference in Credal Networks: New Complexity Results_,
JAIR 50, 603–637.
[DOI](https://doi.org/10.1613/jair.4355).

Metadata and methodological relevance were checked through [R4] and the earlier
research review. The original proofs were not independently reread here.
The plan uses these works to distinguish a specific tractable binary-polytree
fragment from generic bounded-treewidth or compact-factorization claims.

## Statistical calibration baseline

**[R18] C. J. Clopper and E. S. Pearson (1934). _The Use of Confidence or
Fiducial Limits Illustrated in the Case of the Binomial._ Biometrika 26(4),
404–413.**

[DOI](https://doi.org/10.1093/biomet/26.4.404).

Bibliographic metadata checked; the original article was not separately
inspected. The standard exact binomial construction is the proposed conservative
baseline for fixed conditional-event probabilities. A union bound over allocated
error budgets supplies simultaneous coverage without independence among the
different estimated assessments. Independence of the calibration data from
schema selection and correctness of the Markov/support model remain separate
assumptions. Neither this construction nor the union-bound argument is a new
statistical contribution of the plan.

## Additional direct predecessors and recent work

**[R19] Amélie Levray and Vaishak Belle (2020). _Learning Credal Sum-Product
Networks._ Automated Knowledge Base Construction (AKBC).**

[Record](https://arxiv.org/abs/1901.05847)
· [inspected version](https://arxiv.org/pdf/1901.05847v2).

The retrieved version 2 identifies itself as an AKBC 2020 conference paper;
the initial arXiv submission was in 2019. Primary Sections 3–4 and theorem
statements were inspected. LearnCSPN adapts LearnSPN to missing values through
modified count-based independence tests, clustering based on lower/upper
likelihood quantities, ambiguous cluster membership, and credal sum weights.
Theorem 4 establishes its stated valid-CSPN property, Theorem 5 gives reduction
to LearnSPN on complete data, and the internal tree topology supports the
referenced tractable query regime.

This is a direct structure-plus-parameter learning predecessor, including
incomplete data. It prevents positioning the proposed research as the first
joint learner of credal circuits. Its interval construction and circuit-validity
results should be compared separately with the new plan's statistical-coverage
goals and LCN semantic requirements. Reproduce its normalization, count, and
clustering details before transferring implementation choices.

**[R20] Hjalmar Wijk, Benjie Wang, and Marta Kwiatkowska (2022).
_Robustness Guarantees for Credal Bayesian Networks via Constraint Relaxation
over Probabilistic Circuits._**

[Record](https://arxiv.org/abs/2205.05793)
· [inspected version](https://arxiv.org/pdf/2205.05793v1).

Primary abstract and Sections 4.2–4.4 inspected. The work faithfully transfers
a credal-BN maximum-event-probability problem into a constrained circuit and
then relaxes repeated-parameter constraints to obtain a tractable credal-SPN
upper bound. Theorem 2 establishes the bound direction; the following
analysis relates its looseness to a structurally enriched Bayesian network.
The paper retains the local LP cost in the general computation bound.

This is a direct precedent for a circuit representation with shared credal
parameters and a sound relaxation. The LCN extension must handle formula
assessments and its chosen Markov semantics; neither faithful compilation nor
releasing parameter ties should be claimed as a new general idea.

**[R21] Athresh Karanam, Saurabh Mathur, Sahil Sidheekh, and Sriraam
Natarajan (2025). _A Unified Framework for Human-Allied Learning of
Probabilistic Circuits._ AAAI 39(17), 17779–17787.**

[DOI](https://doi.org/10.1609/aaai.v39i17.33955)
· [manuscript record](https://arxiv.org/abs/2405.02413)
· [paper](https://arxiv.org/pdf/2405.02413).

AAAI metadata checked; the retrieved manuscript's constraint definitions and
learning algorithm were inspected. It expresses several forms of knowledge
through equality/inequality constraints on marginal and conditional queries,
then fits parameters with an increasing penalty. Algorithm 1 fixes circuit
structure and has a finite iteration limit; the discussion explicitly allows
incompatibility between knowledge and the circuit family.

Use this as the knowledge-guided point-estimation baseline. Assessing final
constraint residuals and learning a calibrated set of feasible distributions
are separate from optimizing a penalized point estimate. The 2024 arXiv
identifier should not be confused with the verified 2025 conference year.

**[R22] Anji Liu, Zilei Shao, and Guy Van den Broeck (2025). _Rethinking
Probabilistic Circuit Parameter Learning._**

[Record](https://arxiv.org/abs/2505.19982)
· [paper](https://arxiv.org/pdf/2505.19982).

Abstract and introductory optimization discussion inspected; the retrieved
record lists a revision dated 3 October 2025. The paper analyzes practical
mini-batch PC learning through the EM objective and proposes Anemone, with
adaptive updates intended to control changes induced by each batch. The plan
uses it as a recent representative-model fitting baseline. Its convergence
analysis was not reviewed in enough detail to transfer guarantees to constrained
or credal objectives; that comparison remains implementation-stage work.

**[R23] Recent adjacent work on credal learning theory and structural uncertainty.**

Michele Caprio, Maryam Sultana, Eleni Elia, and Fabio Cuzzolin (2024),
_Credal Learning Theory_.
[ArXiv record](https://arxiv.org/abs/2402.00957).

Author/title metadata and abstract inspected. It studies risk bounds when
variability in the data-generating distribution is modeled through credal sets,
including settings with multiple training sets. This is relevant to the theory
of learning under distributional uncertainty, but its stated sampling setup
must be compared with the plan's initial single-i.i.d.-sample setting.

Varun Venkatesh, Eyke Hüllermeier, Bernd Bischl, and Mina Rezaei (2026),
_Structured Credal Learning_.
[Record](https://arxiv.org/abs/2603.14070)
· [paper](https://arxiv.org/pdf/2603.14070).

Primary abstract and introduction inspected. The retrieved manuscript is labeled
as a March 2026 preprint. It separates covariate-shift uncertainty from
conditional-label ambiguity and studies resulting bounds and robust optimization.
Here “structured” refers to the uncertainty decomposition; it is not an LCN
graph-learning or graphical-factorization theorem. Use it to inform optional
shift/noisy-label experiments, without changing the initial learning objective.

Y. Sungtaek Ju (2026), _SymCircuit: Bayesian Structure Inference for Tractable
Probabilistic Circuits via Entropy-Regularized Reinforcement Learning_.
[ArXiv record](https://arxiv.org/abs/2603.20392).

Metadata and abstract inspected only. This recent preprint proposes a
grammar-constrained generative policy for circuit structure inference and
discusses model averaging and several uncertainty sources. It is a candidate
advanced structure-search comparison, not a verified replacement for the
initial beam-search baseline. Its reported posterior/optimization guarantees
require full-text examination before being adopted.

These 2024–2026 sources were screened in the broader title search. The plan
does not treat their abstracts as proofs of LCN calibration, tractable credal
inference, or a compact LCN–circuit translation.

## Reading and verification priorities

1. Read the original learning papers [R5, R7] before fixing the estimator and
   structural-uncertainty baseline.
2. Reproduce [R15, R16, R19] on their own semantic sets before adding LCN
   constraints, and compare the circuit compilation/relaxation in [R20].
3. Read 2U [R17] directly and prove the restricted LCN translation before
   labeling native learned models exact and tractable.
4. Compare full versions of circuit-operation and robust-SPN work when making
   detailed novelty claims; versioned arXiv statements may differ from later
   journal/conference versions.
5. Complete the deeper comparisons of [R21–R23] and extend the search to
   calibrated credal learning, constrained circuit learning, and LCN-specific
   learning. The searches used here identify relevant foundations and recent
   candidates but do not establish that no newer end-to-end LCN learner exists.
