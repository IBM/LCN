# Literature and source notes

Companion to [the research plan](research-plan.md). Reviewed 29 September 2026.
This is a focused review of relevant foundations and close predecessors, not
an exhaustive systematic or priority review. Entries distinguish inspection
of selected original sections, abstracts/metadata, and findings reused from
the preceding repository research reviews. A linked preprint year need not be
the year of the eventual conference or journal publication. Theorem numbers
refer to the versions identified below.

## LCN and credal foundations

**[N1] Radu Marinescu et al. (2022). _Logical Credal Networks._ NeurIPS 35,
15325–15337.**

[Local paper](../Logical_Credal_Networks.pdf).

The original definitions and inference/learning discussion were inspected in
the preceding LCN research tasks and reused here. LCNs combine interval-valued
logical probability assessments with graph-induced Markov assertions. The
original application already includes data-derived rule intervals. The present
proposal adds a neural conditional interface and jointly learned structure;
it should not claim the first use of data or uncertainty in LCNs.

**[N2] Fabio G. Cozman et al. (2024). _Markov conditions and factorization in
logical credal networks._ International Journal of Approximate Reasoning 172,
109237.**

[DOI](https://doi.org/10.1016/j.ijar.2024.109237)
· [local revised manuscript](../LCN_IJAR_Revised.pdf).

Relevant full-text findings are reused from the GMC and learning reviews.
The work distinguishes logical structure, Markov conditions, and chain
factorization. The local revised manuscript has a later date than the journal
reference. Check version-specific examples before using them in a submission.
Its central implication here is that a neural assessment edit must not silently
change or discard the model's structural constraints.

**[N3] Fabio G. Cozman, Radu Marinescu, Junkyu Lee, Alexander Gray, and Denis
D. Mauá (2025). _Dealing with cycles in graph-based probabilistic models:
the case of Logical Credal Networks._ ISIPTA, PMLR 290, 93–102.**

[Proceedings](https://proceedings.mlr.press/v290/cozman25a.html)
· [published paper](https://raw.githubusercontent.com/mlresearch/v290/main/assets/cozman25a/cozman25a.pdf)
· [local manuscript](../ISIPTA2025_LCNs.pdf).

Published Sections 4–5 were inspected in the preceding GMC task. The paper
already applies directed–undirected mixed-graph factorization and formulates
LCN inference as multilinear optimization. Positivity and the distinctions
between LMC, GMC, and cyclic structural models matter. A feedback neural
architecture does not inherit these probabilistic results merely because its
computation graph contains cycles.

**[N4] Denis D. Mauá and Fabio G. Cozman (2020). _Thirty years of credal
networks: Specification, algorithms and complexity._ International Journal
of Approximate Reasoning 126, 133–157.**

[DOI](https://doi.org/10.1016/j.ijar.2020.08.009)
· [local paper](../1-s2.0-S0888613X20302152-main%20%281%29.pdf).

Inference, complexity, and learning discussions were inspected in the previous
learning review. This supports the distinction between precise factorization
and optimization over credal parameters. Its discussion of 2U motivates an
optional binary-polytree fragment. Read the original 2U paper before implementing
its boundary cases: Fagiuoli and Zaffalon (1998),
[DOI](https://doi.org/10.1016/S0004-3702(98)00089-7). No new general credal
tractability claim is inferred from bounded graph width alone.

## Neural probabilistic and logical architectures

**[N5] Robin Manhaeve, Sebastijan Dumančić, Angelika Kimmig, Thomas Demeester,
and Luc De Raedt. _DeepProbLog: Neural Probabilistic Logic Programming_
(NeurIPS 2018), and _Neural Probabilistic Logic Programming in DeepProbLog_
(Artificial Intelligence 298, 103504, 2021).**

[Conference manuscript](https://arxiv.org/abs/1805.10872)
· [journal manuscript](https://arxiv.org/abs/1907.08194)
· [journal DOI](https://doi.org/10.1016/j.artint.2021.103504).

Selected original sections on neural annotated disjunctions, distribution
semantics, circuit inference, and gradient semirings were inspected. Neural
outputs already parameterize probabilistic logic and receive gradients from
logical queries. Independent probabilistic choices in the base semantics do
not imply every derived query is independent. The NLCN comparison should
address set-valued, coupled assessments and explicit LCN constraints, rather
than portray end-to-end neural/logical training as new.

**[N6] Zhun Yang, Adam Ishay, and Joohyung Lee (2020). _NeurASP: Embracing
Neural Networks into Answer Set Programming._ IJCAI.**

[Proceedings](https://www.ijcai.org/proceedings/2020/243)
· [later arXiv record](https://arxiv.org/abs/2307.07700).

Original abstract and introductory semantic/learning discussion inspected;
proceedings page checked. The arXiv upload is from 2023, not the original
conference year. Neural predictions participate in a probability model over
answer sets. This is a close neural-symbolic baseline, but stable-model
semantics and LCN world constraints must be aligned explicitly for a
same-problem inference comparison.

**[N7] Thomas Winters, Giuseppe Marra, Robin Manhaeve, and Luc De Raedt
(2021 preprint). _DeepStochLog: Neural Stochastic Logic Programming._**

[Record and paper](https://arxiv.org/abs/2106.12574).

The retrieved version 1 abstract and introduction were inspected. Neural
stochastic definite clause grammars support end-to-end learning and reasoning
over distributions on derivations. The paper explicitly distinguishes this
from the possible-world distribution semantics used by neural probabilistic
logic programs. Its scaling results should not be transferred to arbitrary
LCN marginal queries without a semantic reduction.

**[N8] Arseny Skryagin, Wolfgang Stammer, Daniel Ochs, Devendra Singh Dhami,
and Kristian Kersting (2021 preprint). _SLASH: Embracing Probabilistic
Circuits into Neural Answer Set Programming._**

[Record and paper](https://arxiv.org/abs/2110.03395).

Selected original introduction and architecture sections inspected in version
4, dated November 2021. Neural-probabilistic predicates combine neural modules
and PCs with an answer-set program. The object-centric example shares an
encoder and uses PCs for joint attribute/encoding models. SLASH already offers
unified end-to-end neural, probabilistic, and logical learning, making it a
more informative predecessor than a loosely coupled neural/rule pipeline.

**[N9] Ryan Riegel et al. (2020). _Logical Neural Networks._**

[Record and paper](https://arxiv.org/abs/2006.13155).

Selected original architectural and semantic sections inspected. Formula/neuron
alignment and bidirectional propagation of truth bounds are relevant design
ideas. The framework uses weighted real-valued logical semantics. Its truth
bounds are not automatically lower/upper probabilities over distributions
satisfying LCN assessments and independencies. An NLCN must derive probabilistic
conjunction bounds from its joint family rather than rename a fuzzy operator.

**[N10] Samy Badreddine, Artur d'Avila Garcez, Luciano Serafini, and Michael
Spranger (2022). _Logic Tensor Networks._ Artificial Intelligence 303, 103649.**

[Record](https://arxiv.org/abs/2012.13635)
· [DOI](https://doi.org/10.1016/j.artint.2021.103649).

Original introductory/semantic sections inspected and journal metadata checked.
Real Logic grounds first-order expressions in tensors with fuzzy truth
semantics, enabling differentiable learning from logical constraints. This
supports architectural and learning comparisons, not an identification of
truth satisfaction with probabilistic coherence. Temporal or interval-valued
fuzzy extensions also need this distinction.

**[N11] Jingyi Xu, Zilu Zhang, Tal Friedman, Yitao Liang, and Guy Van den
Broeck (2018). _A Semantic Loss Function for Deep Learning with Symbolic
Knowledge._ ICML, PMLR 80.**

[Paper](https://proceedings.mlr.press/v80/xu18h/xu18h.pdf).

Original abstract and introductory formulation inspected. Semantic loss
connects symbolic constraints with neural learning through the probability
mass assigned to satisfying outputs. It is a necessary knowledge-guided
training baseline. A loss encouraging satisfaction is not the same object as
an entire LCN family with verified lower/upper query probabilities.

**[N12] Kareem Ahmed, Stefano Teso, Kai-Wei Chang, Guy Van den Broeck,
and Antonio Vergari (2022). _Semantic Probabilistic Layers for Neuro-Symbolic
Learning._**

[Record and paper](https://arxiv.org/abs/2206.00426).

Selected original Sections 2–3 and Theorem 3.1 inspected. SPL combines a
neural-conditioned joint PC with a logical constraint circuit, then normalizes.
Smoothness, decomposability, and compatibility support the stated polynomial
product/evaluation result; determinism is additionally used for the MAP-state
result. This already handles correlated labels and hard logical support.
The proposed extension concerns coupled credal regions, endpoint optimization,
and their LCN semantics, not the first structured probabilistic neural layer.

**[N13] Xiaoting Shao et al. (2019 preprint). _Conditional Sum-Product
Networks: Imposing Structure on Deep Probabilistic Architectures._**

[Record and paper](https://arxiv.org/abs/1905.08550).

Original abstract and architecture/structure-learning motivation inspected.
The model conditions SPN parameters on inputs and also learns conditional
structure, since conditioning the parameters of an arbitrary fixed SPN can
misrepresent conditional dependencies. This is a direct predecessor for both
neural-conditioned circuits and their structure learning. Spell out
“conditional SPN” versus “credal SPN”: the acronym CSPN is used for both.

## Credal circuits and neural uncertainty

**[N14] Denis D. Mauá, Fabio G. Cozman, Diarmaid Conaty, and Cassio P.
de Campos (2017). _Credal Sum-Product Networks._ ISIPTA, PMLR 62, 205–216.**

[Proceedings](https://proceedings.mlr.press/v62/mau%C3%A117a.html)
· [paper](https://proceedings.mlr.press/v62/mau%C3%A117a/mau%C3%A117a.pdf).

Section 3 and its theorem statements were inspected in the preceding learning
review. Theorem 1 concerns lower/upper evaluation of observations with separately
specified local weight polytopes. Theorem 4 treats univariate conditional
expectations under an internal-tree restriction. Do not extrapolate these to
arbitrary posterior formulas or tied neural uncertainty. The post-hoc
robustification experiments motivate a baseline. The journal continuation,
_Robustifying sum-product networks_ (2018),
[DOI](https://doi.org/10.1016/j.ijar.2018.07.003), was checked bibliographically;
its extended results require a fresh original-text review before a novelty claim.

**[N15] Lilith Mattei, Alessandro Antonucci, Denis D. Mauá, Alessandro
Facchini, and Julissa Villanueva Llerena (2020). _Tractable inference in
credal sentential decision diagrams._ IJAR 125, 26–48.**

[DOI](https://doi.org/10.1016/j.ijar.2020.06.005)
· [inspected manuscript](https://arxiv.org/pdf/2008.08524v1).

Sections 3–6 were inspected in the previous learning task. The strong extension
is a convex hull of compatible precise PSDDs, with explicit support conditions.
Theorem 3 gives exact marginal evaluation in its stated setting. Theorem 4 and
the subsequent discussion distinguish exact singleton conditional inference
for singly connected CSDDs from conservative propagation in more general
topologies. Logical support, imprecise parameter learning, and selected exact
queries already coexist here; the NLCN interface must establish what it adds.

**[N16] Amélie Levray and Vaishak Belle (2020). _Learning Credal Sum-Product
Networks._ Automated Knowledge Base Construction.**

[Record](https://arxiv.org/abs/1901.05847)
· [inspected version](https://arxiv.org/pdf/1901.05847v2).

Original Sections 3–4 and theorem statements were inspected in the preceding
learning task. LearnCSPN adapts structure and parameter learning to incomplete
data using credal clustering/count information. Version 2 identifies the work
as AKBC 2020 despite the 2019 arXiv identifier. Learning a credal circuit's
structure and parameters is therefore established; the new proposal adds neural
alignment, LCN constraints, and a separately justified uncertainty protocol.

**[N17] Hjalmar Wijk, Benjie Wang, and Marta Kwiatkowska (2022).
_Robustness Guarantees for Credal Bayesian Networks via Constraint Relaxation
over Probabilistic Circuits._**

[Record](https://arxiv.org/abs/2205.05793)
· [inspected version](https://arxiv.org/pdf/2205.05793v1).

Original Sections 4.2–4.4 were inspected in the preceding learning task.
The work transfers credal-BN event optimization to a constrained circuit and
relaxes repeated-parameter constraints to obtain sound bounds. This is a
direct predecessor for preserving shared uncertainty before relaxing it.
NLCN extensions must address logical formula assessments and the chosen Markov
semantics; compilation with parameter ties and relaxation are not new in general.

**[N18] Michele Caprio, Souradeep Dutta, Kuk Jin Jang, Vivian Lin,
Radoslav Ivanov, Oleg Sokolsky, and Insup Lee. _Credal Bayesian Deep Learning._
arXiv record 2023; inspected version 5, October 2024.**

[Record and paper](https://arxiv.org/abs/2302.09656).

Original abstract and introductory construction inspected. Credal sets of
priors and likelihoods induce sets of posteriors over neural parameters and
predictive distributions. This already supplies a principled neural credal
perspective and decision rules based on lower probabilities. The proposal
should compare uncertainty interpretations and posterior approximations
carefully, rather than identify arbitrary interval heads with this construction.

**[N19] Kaizheng Wang, Keivan Shariatmadar, Shireen Kudukkil Manchingal,
Fabio Cuzzolin, David Moens, and Hans Hallez. _CreINNs: Credal-Set Interval
Neural Networks for Uncertainty Estimation in Classification Tasks._
arXiv record 2024; metadata lists a January 2025 revision.**

[Record](https://arxiv.org/abs/2401.05043).

Primary metadata and abstract inspected; original method/proofs were not
independently reviewed. The abstract describes interval neural weights and
lower/upper class probabilities defining a credal set. This is a close baseline
for interval neural prediction. Read the complete method before adopting its
uncertainty decomposition or asserting a formal containment guarantee for a
particular implementation.

**[N20] Matteo Tolloso and Davide Bacciu (2025 preprint). _Credal Graph
Neural Networks._**

[Record](https://arxiv.org/abs/2512.02722).

Primary metadata and abstract inspected. The work introduces set-valued GNN
predictions and studies layer-wise information, homophily, and distribution
shift. It is relevant to structured neural uncertainty and graph-based heads.
The inspected material does not establish LCN logical semantics or certified
LCN query optimization; neither equivalence nor an absence of further relevant
results is claimed without reading the full paper.

## Optimization, verification, and calibration

**[N21] Akshay Agrawal, Brandon Amos, Shane Barratt, Stephen Boyd,
Steven Diamond, and J. Zico Kolter (2019). _Differentiable Convex
Optimization Layers._ NeurIPS.**

[Record and paper](https://arxiv.org/abs/1910.12430).

Selected original formulation, differentiation discussion, and appendix
non-differentiability treatment inspected. Implicit differentiation depends on
regularity/invertibility, and the paper discusses approximate handling when
these fail. This supports differentiable convex NLCN relaxations; it is not a
general solution for global differentiation of nonconvex LCN extrema. Include
constraint-parameter derivatives and distinguish a value derivative from a
selected optimizer's derivative.

**[N22] Sven Gowal et al. (2018). _On the Effectiveness of Interval Bound
Propagation for Training Verifiably Robust Models._**

[Record and paper](https://arxiv.org/abs/1810.12715).

Original abstract and introductory verification discussion inspected.
Interval bound propagation encloses neural behavior under a specified input
perturbation model. It motivates the simplest neural-side enclosure for
composition with a logical credal relaxation. Certified perturbation bounds,
calibration of class probabilities, and confidence intervals from sampling are
different guarantees. Stronger neural verification methods should be reviewed
when implementing the joint robustness work package.

**[N23] Maxime Gasse, Didier Chételat, Nicola Ferroni, Laurent Charlin,
and Andrea Lodi (2019). _Exact Combinatorial Optimization with Graph
Convolutional Neural Networks._**

[Record](https://arxiv.org/abs/1906.01629).

Primary metadata and abstract inspected. A graph-convolutional policy learns
branch-and-bound variable selection from a strong-branching expert, using
variable/constraint graph structure. This establishes learned search control
as a predecessor. Read the original experimental and solver details before
reproducing it. In the proposed NLCN solver, learned control must preserve
verified pruning and fallback behavior.

**[N24] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger (2017).
_On Calibration of Modern Neural Networks._ ICML, PMLR 70.**

[Paper](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf).

Original calibration discussion inspected. Temperature scaling is an important
precise-prediction baseline, but empirical calibration does not imply
simultaneous coverage of conditional-probability functions. For the finite
stratum confidence construction, reuse the exact binomial/union-bound baseline
and source notes in the [learning bibliography](../research-learning/references.md).
A full conformal inference review remains follow-up work if label-set guarantees
become a core contribution.

## Grounding, rule learning, and circuit structure

**[N25] Emanuele Marconato, Samuele Bortolotti, Emile van Krieken,
Antonio Vergari, Andrea Passerini, and Stefano Teso (2024).
_BEARS Make Neuro-Symbolic Models Aware of their Reasoning Shortcuts._**

[Record and paper](https://arxiv.org/abs/2402.12240).

Original abstract and motivation inspected. BEARS uses an ensemble to represent
semantic ambiguity in concept predictions rather than assuming correct task
labels establish correct concepts. Its relationship to independence was also
read in [N26], which notes that mixing conditionally independent predictors
need not yield a conditionally independent concept distribution. Compare
mixtures and credal envelopes explicitly instead of describing them as identical.

**[N26] Emile van Krieken, Pasquale Minervini, Edoardo Ponti, and Antonio
Vergari (2025). _Neurosymbolic Reasoning Shortcuts under the Independence
Assumption._ PMLR 284.**

[Record and paper](https://arxiv.org/abs/2507.11357).

Original Sections 2–4 inspected, especially the XOR example and Theorems 7–8.
Under the paper's setup and assumptions, these characterize limitations of
conditionally independent concept models in representing mixtures over
reasoning shortcuts. This supplies a precise motivation for coupled concept
uncertainty. It does not imply that every dependent model learns the intended
grounding, or that an envelope over several models is the same as one mixture.

**[N27] Fan Yang, Zhilin Yang, and William W. Cohen (2017).
_Differentiable Learning of Logical Rules for Knowledge Base Reasoning._
NeurIPS.**

[Record and paper](https://arxiv.org/abs/1702.08367).

Original abstract and introductory architecture inspected in version 3.
Neural LP uses a controller to compose differentiable TensorLog operations
and learn rule parameters and structure together. This is a direct predecessor
for differentiable logical structure learning. Its soft composition/attention
parameters are not automatically coherent LCN assessments, and induced LCN
Markov changes require a separate discrete semantic check.

**[N28] Richard Evans and Edward Grefenstette (2018).
_Learning Explanatory Rules from Noisy Data._**

[Record and paper](https://arxiv.org/abs/1711.04574).

Original abstract, introductory differentiable-ILP formulation, and candidate
clause discussion inspected in version 2, January 2018. The initial arXiv
submission is from 2017. The method combines a restricted rule search space
with differentiable forward reasoning and supports integration with neural
perception. It motivates typed candidate grammars and small-data experiments;
it does not provide the probability-set semantics or credal certificates needed
by this proposal.

**[N29] Pang Wei Koh et al. (2020). _Concept Bottleneck Models._ ICML,
PMLR 119, 5338–5348.**

[Proceedings](https://proceedings.mlr.press/v119/koh20a.html).

Primary proceedings metadata and abstract inspected; full text was not
independently reviewed here. Concept-level representations and corrections
provide an important baseline for judging the value of interpretable neural
interfaces. Detailed intervention protocols should be read before reproduction.
The plan distinguishes manipulating predicted concepts from a claim about
causal interventions in the data-generating system.

**[N30] Antonio Vergari, YooJung Choi, Anji Liu, Stefano Teso, and Guy
Van den Broeck (2021). _A Compositional Atlas of Tractable Circuit Operations:
From Simple Transformations to Complex Information-Theoretic Queries._**

[Record](https://arxiv.org/abs/2102.06137)
· [inspected version](https://arxiv.org/pdf/2102.06137v1).

Selected structural definitions and tractability discussion were inspected in
the preceding learning review. Different circuit operations require different
compatibility conditions. This supports query-specific contracts for products,
conditioning, formula evaluation, and learning objectives. The cost of compiling
a new query or constraint must be counted alongside circuit evaluation.

**[N31] Probabilistic circuit structure and parameter learning.**

Robert Gens and Pedro Domingos (2013), _Learning the Structure of Sum-Product
Networks_, ICML, [paper](https://proceedings.mlr.press/v28/gens13.pdf).
Robert Peharz et al. (2020), _Einsum Networks: Fast and Scalable Learning of
Tractable Probabilistic Circuits_, ICML,
[proceedings](https://proceedings.mlr.press/v119/peharz20a.html).

The former's recursive structure-learning method and the latter's learning/
implementation discussion were inspected in the previous learning review.
Variable partitioning, mixture construction, batched evaluation, and efficient
parameter fitting are established components to adapt. Neither approximate
independence tests nor fast precise circuit evaluation provide a guarantee
for arbitrary neural-conditioned credal optimization. See the
[learning plan](../research-learning/research-plan.md) for complementary
LearnPSDD and Strudel structure-search directions.

**[N32] Athresh Karanam, Saurabh Mathur, Sahil Sidheekh, and Sriraam
Natarajan (2025). _A Unified Framework for Human-Allied Learning of
Probabilistic Circuits._ AAAI 39(17), 17779–17787.**

[DOI](https://doi.org/10.1609/aaai.v39i17.33955)
· [manuscript](https://arxiv.org/abs/2405.02413).

Original constraint definitions and learning algorithm were inspected in the
preceding learning review; journal/proceedings metadata was checked there.
The work fits PC parameters under several kinds of domain knowledge using
increasing penalties. This is a close constrained point-learning baseline.
A small penalty residual does not by itself define a calibrated credal family
or certify a global endpoint. Compare coherence, repair, and constraint
satisfaction separately from predictive fit.
