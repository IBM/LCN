# Literature and source notes

Companion to [the research plan](research-plan.md). Reviewed 29 September 2026.
The review is focused on the foundations and closest algorithmic predecessors;
it is not an exhaustive systematic review. Entries distinguish direct full-text
inspection, verification through a primary paper's treatment of earlier work,
and bibliographic verification. Theorem numbers refer to the versions identified
below.

## LCN semantics and the immediate beyond-chain predecessor

**[R1] Fabio G. Cozman, Radu Marinescu, Junkyu Lee, Alexander Gray, and
Denis D. Mauá (2025). _Dealing with cycles in graph-based probabilistic models:
the case of Logical Credal Networks._ ISIPTA, PMLR 290, 93–102.**

[Published proceedings page](https://proceedings.mlr.press/v290/cozman25a.html)
· [published paper](https://raw.githubusercontent.com/mlresearch/v290/main/assets/cozman25a/cozman25a.pdf)
· [local manuscript](../ISIPTA2025_LCNs.pdf).

The published full text and the local manuscript were inspected, particularly
Sections 4.1–4.3 and 5. The local copy is labeled as a submission; use the
published version for the final research manuscript. The factorization and
theorem numbering used in the plan were checked against that version.

This is the central predecessor. It distinguishes opposing directed arrows from
undirected connections, defines the mixed-structure DUMG, applies Koster's
factorization, and already gives a multilinear inference formulation. Theorems
4.1–4.3 separate positivity-free GMC equivalences, positive GMC-to-Gibbs, and
Gibbs-to-GMC. Example 4.4 is the compiler reference; Example 4.5 illustrates
algebraic elimination of variables from an inference expression. Section 5
explains why cyclic structural equation models need not satisfy the selected
GMC.

Two details motivate research checks rather than unqualified transfer. First,
conditional normalization and potential parameter bounds need an explicit
implementation treatment. Second, Section 4.3 describes a closed lifted
feasible region and then substitutes minima/maxima for infima/suprema.
Closedness alone is insufficient for that substitution; compactness or another
attainment argument is needed, especially when reciprocal evidence variables
can diverge. The plan preserves the formulation while making this proof
obligation explicit.

**[R2] Fabio G. Cozman et al. (2024). _Markov conditions and factorization in
logical credal networks._ International Journal of Approximate Reasoning 172,
109237.**

[DOI](https://doi.org/10.1016/j.ijar.2024.109237)
· [local revised manuscript](../LCN_IJAR_Revised.pdf).

The local revised full text was inspected, especially Section 6. The journal
year and article identifier are corroborated by the published reference in
[R1]. The local file has a later preprint date; its example numbering is used
when the plan refers to Example 10. Compare versions before citing those
example numbers in a submission.

The paper supplies the original structure/LMC framework, the chain-graph
factorization boundary, and a discussion of alternative global Markov
conditions. The long-cycle discussion and Example 10 show why neither a universal
LMC-to-GMC implication nor a universal tightening of LCN bounds can be assumed.
This is also the source for distinguishing the full directed construction from
the structure that collapses opposing arrows.

## Global Markov conditions and factorization beyond chain graphs

**[R3] Jan T. A. Koster (1997). _Gibbs and Markov properties of graphs._
Annals of Mathematics and Artificial Intelligence 21, 13–26.**

[DOI](https://doi.org/10.1023/A:1018948915264).

Title, author, venue, pages, and DOI were checked through publisher/Crossref
metadata. An attempted publisher PDF retrieval returned an HTML page, so the
original full text was **not** independently inspected. The plan's mathematical
use of Theorem 4.3 is supported by its explicit adaptation and proofs discussion
in the inspected published paper [R1].

Priority follow-up: read the original definitions of anterior sets, closure,
join-irreducibles, Gibbs kernels, and null conditioning configurations before
finalizing a compiler equivalence proof. This is the source of the DUMG theory;
its factorization should not be presented as a new result of this project.

**[R4] Jan T. A. Koster (1996). _Markov properties of nonrecursive causal
models._ Annals of Statistics 24(5), 2148–2177.**

[DOI](https://doi.org/10.1214/aos/1069362315).

Bibliographic metadata checked; original PDF retrieval returned HTML, so
technical claims here rely on the discussion in [R1, R2]. Read as historical
background on reciprocal/nonrecursive graphs and their separation rules.
Also relevant is Koster (2002), _Marginalizing and conditioning in graphical
models_, Bernoulli 8(6), 817–840, as cited by [R1]. Its full text remains a
follow-up for the query-projection and graph-marginalization work package.

**[R5] Patrick Forré and Joris M. Mooij (2017). _Markov Properties for Graphical
Models with Cycles and Latent Variables._ arXiv:1710.08775, version 1.**

[Abstract and version history](https://arxiv.org/abs/1710.08775)
· [versioned full text](https://arxiv.org/pdf/1710.08775v1).

The retrieved 131-page full text identifies itself as version 1, 24 October 2017.
Sections on ancestral factorization, acyclification/sigma-separation, and
marginal Markov properties were inspected selectively. This is a broad theory
paper; the plan does not assume that all its Markov properties are equivalent.

Specific results used:

- Corollary 3.6.9: dGMP and ancestral factorization are equivalent under
  positivity or the graph's defined perfect-elimination-order condition.
- Example 3.6.10: an eight-state support on four binary variables satisfies the
  cycle GMC but has no edge factorization. The authors attribute the underlying
  undirected example to Lauritzen (1996), Example 3.10.
- Theorem 3.6.11: ancestral factorization is stable under the specified graph
  marginalization operation. It is not a theorem about deleting nodes without
  updating the graph or retaining logical constraints.
- Lemma 3.6.14: a positive ancestral-factorizing directed model admits normalized
  SCC kernels with finer internal clique factors.
- Remark 3.6.15: in the stated positive directed setting, the d-GMC and sigma-GMC
  factorization comparison exposes the loss of internal SCC restrictions under
  the latter.
- Sections 2.7–2.8: acyclification and sigma-separation, including their carefully
  specified relation to d-separation and marginalization.
- Section 3.7: ordinary and marginalized Markov properties differ. In particular,
  a graph's observed-variable CIs do not generally establish latent-model
  realizability.

Read the relevant hypotheses in full before implementing a HEDG or sigma
compiler. Its bidirected/hyperedge conventions cannot be inferred from the
LCN code's use of the word “bidirected” for opposing arrows.

**[R6] Support boundaries: Moussouris (1974) and Geiger, Meek, and Sturmfels
(2006).**

John Moussouris, _Gibbs and Markov random systems with constraints_, Journal of
Statistical Physics 10(1), 11–33.
[DOI](https://doi.org/10.1007/BF01011714).

Dan Geiger, Christopher Meek, and Bernd Sturmfels, _On the toric algebra of
graphical models_, Annals of Statistics 34(3), 1463–1492.
[DOI](https://doi.org/10.1214/009053606000000263).

The relevance and bibliographic details were verified through [R1]; the
Geiger–Meek–Sturmfels title and DOI were also checked through metadata. Neither
original full text was inspected for this plan. They are priority reading for
the support-aware exactness project, not a basis for claiming that an arbitrary
zero pattern satisfies a ready-made sufficient condition.

[R1] discusses a barrier/support-localization property and limits of positive
graphical distributions. The plan treats checking and adapting those conditions
to all relevant anterior kernels as research work. It does not identify the
whole GMC set with either the Gibbs image or the closure of its positive part.

**[R7] Cyclic causal separation: Spirtes (1995) and Neal (2000).**

Peter Spirtes, _Directed cyclic graphical representations of feedback models_,
UAI, 491–498.

Radford M. Neal, _On deducing conditional independence from d-separation in
causal graphs with feedback (research note)_, JAIR 12, 87–91.
[DOI](https://doi.org/10.1613/jair.689).

Bibliographic details and their relevance were verified through [R1, R2, R5].
Original full-text comparison remains follow-up. These works motivate checking
the structural-model assumptions behind any separation semantics. A cyclic
diagram and conditional assessments alone do not justify importing every
acyclic causal theorem.

## Inference and optimization precedents

**[R8] Denis D. Mauá and Fabio G. Cozman (2020). _Thirty years of credal
networks: Specification, algorithms and complexity._ International Journal of
Approximate Reasoning 126, 133–157.**

[DOI](https://doi.org/10.1016/j.ijar.2020.08.009)
· [local paper](../1-s2.0-S0888613X20302152-main%20%281%29.pdf).

The local full text was inspected, particularly Section 5.2.1 and its references.
It explicitly describes symbolic variable elimination using intermediate
functions as multilinear constraints, tree-decomposition-guided formulations,
reciprocal normalization for conditioning, and branch-and-bound with convex
relaxations for outer bounds. These are established techniques and are not
novelty claims of the proposed research.

The survey also distinguishes local strong-extension specifications from more
general credal constraints. Its complexity discussion prevents inferring a
polynomial global optimization algorithm merely from a compact factor circuit.

**[R9] Cassio P. de Campos and Fabio G. Cozman (2004). _Inference in credal
networks using multilinear programming._ Second Starting AI Researcher
Symposium, 50–61.**

Bibliographic details and method were verified through [R8, Section 5.2.1 and
reference 57], rather than a separately retrieved original. This is the direct
algorithmic predecessor for symbolic elimination followed by global multilinear
optimization. Related work identified by the same survey includes:

- de Campos and Cozman (2007), _Inference in credal networks through integer
  programming_, ISIPTA, 145–154.
- J. C. F. da Rocha and F. G. Cozman (2005), _Inference in credal networks:
  branch-and-bound methods and the A/R+ algorithm_, IJAR 39, 279–296.
- A. Cano, M. Gómez, S. Moral, and J. Abellán (2007), _Hill-climbing and
  branch-and-bound algorithms for exact and approximate inference in credal
  networks_, IJAR 44, 261–280.
  [DOI](https://doi.org/10.1016/j.ijar.2006.07.020).

Read the original formulations before making a fine-grained novelty comparison
of branching, interval propagation, or constraint generation. The intended new
work is their extension to the DUMG/anterior representation with logical
coupling, normalization, zeros, and certified semantics.

**[R10] Garth P. McCormick (1976). _Computability of global solutions to
factorable nonconvex programs: Part I — Convex underestimating problems._
Mathematical Programming 10, 147–175.**

[DOI](https://doi.org/10.1007/BF01580665).

Publisher/Crossref metadata checked; the original full text was not inspected.
The four bilinear-envelope inequalities displayed in the plan are the standard
bounded-domain construction. Their use requires valid finite bounds on each
variable. Probability marginals lie in `[0,1]`; arbitrary Gibbs potentials do
not. Research on stronger reformulation-linearization and polynomial
optimization hierarchies should follow this baseline if weak envelopes limit
performance.

**[R11] Junction trees and marginal representations.**

Steffen L. Lauritzen and David J. Spiegelhalter (1988), _Local Computations with
Probabilities on Graphical Structures and Their Application to Expert Systems_,
Journal of the Royal Statistical Society, Series B 50(2), 157–194.
[DOI](https://doi.org/10.1111/j.2517-6161.1988.tb01721.x).

Martin J. Wainwright and Michael I. Jordan (2008), _Graphical Models, Exponential
Families, and Variational Inference_, Foundations and Trends in Machine Learning
1(1–2), 1–305.
[DOI](https://doi.org/10.1561/2200000001).

Bibliographic metadata checked; original full texts were not inspected in this
review. These are foundational reading for junction-tree reconstruction,
marginal versus local consistency, and graphical relaxations. Their precise-model
results do not automatically justify choosing interval endpoints independently
in credal messages, nor do they impose the GMCs of every anterior subgraph of
an arbitrary mixed graph.

Also read Steffen L. Lauritzen (1996), _Graphical Models_, especially the
support-sensitive factorization discussion cited by [R5].

**[R12] Alessandro Antonucci, Cassio P. de Campos, David Huber, and Marco
Zaffalon (2015). _Approximate credal network updating by linear programming with
applications to decision making._ International Journal of Approximate
Reasoning 58, 25–38.**

[DOI](https://doi.org/10.1016/j.ijar.2014.10.003).

Methodological relevance was checked through the survey [R8], the existing
[sampling research plan](../research-sampling/research-plan.md), and the
repository's [ApproxLP implementation](../../lcn/inference/marginal/cn/approxlp.py).
This plan does not report a new independent full-text review of the article.

ApproxLP provides a precedent for feasible-model coordinate optimization and
restarts. In the proposed GMC setting, local updates must also respect
normalization and cross-kernel logical constraints. Feasible models produce
inner information about endpoint optima; they do not replace certified outer
relaxations.

## Later treatment of cyclic structural models

**[R13] Stephan Bongers, Patrick Forré, Jonas Peters, and Joris M. Mooij (2021).
_Foundations of structural causal models with cycles and latent variables._
Annals of Statistics 49(5).**

[DOI](https://doi.org/10.1214/21-AOS2064)
· [arXiv record](https://arxiv.org/abs/1611.06221)
· [inspected full-text version](https://arxiv.org/pdf/1611.06221v6).

Journal metadata was checked, and the retrieved version 6 manuscript dated
22 November 2021 was inspected selectively: abstract/introduction, the Markov
section, and Appendix A.2.2. Proposition A.19 states the equivalence between
sigma-separation in the original directed mixed graph and d-separation in its
defined acyclification. Theorem A.21 supplies sigma-GMC under unique solvability
with respect to each SCC of the SCM graph. This is a later journal presentation
building on [R5], not an inference algorithm for imprecise logical assessments.

It strengthens the secondary track's motivation and clarifies which structural
assumptions should accompany feedback applications. The plan uses it for
separation and semantic hypotheses, without assuming that every GMC-compatible
joint distribution has an SCM realization.

## Source and novelty checklist for the next research phase

1. Obtain and read Koster [R3] directly before finalizing the DUMG compiler proof.
2. Compare the local and published versions of [R2] before relying on precise
   example numbering in a manuscript.
3. Read [R6] and the perfect-order definitions in [R5] before stating any new
   zero-permitting equivalence or hybrid-kernel theorem.
4. Compare the proposed relaxation/refinement scheme with the original credal
   optimization papers in [R9] and later work; the survey alone is insufficient
   for a first-of-its-kind claim.
5. Extend the search to cyclic/DUMG inference, polynomial conditional-independence
   optimization, and sigma-Markov inference published after these foundations.
   Record search dates and distinctions between separation equivalence,
   distribution representability, and inference guarantees.
