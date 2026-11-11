Official Review of Submission39159 by Reviewer A2AT
Official Reviewby Reviewer A2AT22 Oct 2026, 22:59 (modified: 06 Nov 2026, 12:31)EveryoneRevisions
Paper Summary:
This paper studies marginal inference in Logical Credal Networks (LCNs), an expressive probabilistic logic framework that combines propositional formulas with imprecise probability bounds to represent sets of probability distributions. The authors exploit the chain graph structure underlying LCNs and reformulate inference as a credal network problem. They also introduce Credal Junction Tree (CJT), an exact inference method that leverages a junction tree decomposition to construct and solve a substantially smaller nonlinear program.

Presentation: 2: Generally understandable, but requires substantial effort in places.
Informativeness: 2: I learned something interesting or useful.
Soundness: 1: I have significant concerns about the paper’s soundness that do not appear easy to address.
Critical Strengths:
The main contribution of this paper is to establish an algorithmic connection between chain-graph Logical Credal Networks (LCNs) and credal networks. In particular, the authors introduce Credal Junction Tree (CJT), an exact inference method that exploits the chain graph structure by moralizing and triangulating the graph and constructing a junction tree. This allows the original optimization problem over the full joint probability distribution to be reformulated as a substantially smaller nonlinear program over local clique marginals. The authors further prove that CJT computes exact lower and upper probability bounds for the original LCN under certain conditions, including that every probabilistic constraint can be assigned to a clique and all relevant conditional independence assertions are properly accounted for.

Critical Weaknesses:
The paper has several limitations that I think deserve further discussion. First, the proposed framework relies heavily on the underlying network structure, as the analysis mainly focuses on chain-graph LCNs, and some of the exactness guarantees require additional assumptions such as family-realizability. For more general LCNs, the strong extension can be substantially larger than the original feasible distribution set, potentially leading to loose probability bounds. The advantages of the approximate methods are not entirely consistent across different settings. Although IBP performs very well on synthetic benchmarks, these instances are specifically constructed to satisfy family-realizability, while on more realistic instances, the existing ARIEL method often produces more accurate bounds. This raises questions about how well the proposed approximations generalize to less structured problems. While CJT significantly reduces the number of variables in the nonlinear program, the resulting optimization problem remains nonconvex. The experiments mainly rely on IPOPT, which does not guarantee global optimality, so it is unclear whether the reported CJT results can always be regarded as certified exact solutions. And the methodological novelty seems somewhat limited, since the chain-graph factorization is based on previous work, and CVE, interval message passing, and coordinate descent are closely related to existing credal network inference techniques. Much of the contribution lies in adapting these methods to LCNs, with CJT being the main new algorithmic development.

Questions For Authors:
My main concern is the correctness of several theoretical claims. In particular, the dominance pruning rule used in CVE does not appear to preserve the extrema of normalized posterior probabilities, which calls the exactness guarantee in Theorem 2 into question. Similarly, the proposed IBP updates differ substantially from standard belief propagation, and the proof of Theorem 4 does not establish why these updates should be exact on singly connected structures. I am also concerned about the equality between the original LCN distribution set and the strong extension, since the latter is defined as a convex hull and may contain distributions that violate the original conditional independence constraints. Finally, while the CJT reconstruction argument appears more plausible under its stated assumptions, the paper does not clearly establish how often these assumptions hold in practice, and the descriptions of the optimization solver and global optimality guarantees are inconsistent across the main text and appendix. These issues directly affect the validity of the theoretical guarantees and the reliability of the experimental comparisons. I believe they require substantial clarification and, potentially, corrections to the stated results.

Flag For Ethics Review: No
Rating: 1: Clear rejection
Confidence: 4: I have substantial direct research experience on this topic.

Official Review of Submission39159 by Reviewer c7e5
Official Reviewby Reviewer c7e520 Oct 2026, 07:51 (modified: 06 Nov 2026, 12:31)EveryoneRevisions
Paper Summary:
This paper develops an algorithmic connection between LCNs and credal networks using an existing chain-graph factorization. It presents several inference methods, including CVE, IBP, CDVE, and CJT. The connection is conceptually appealing, and CJT is the clearest methodological contribution: under its stated structural conditions, it replaces optimization over the full joint distribution with a clique-marginal formulation containing (
) variables, rather than (
). This can substantially reduce the formulation size on low-treewidth instances, although it does not establish a corresponding bound on global optimization time. The evaluation covers a broad collection of synthetic and realistic benchmarks. Overall, the paper offers a potentially valuable synthesis, but several methods adapt existing credal-network techniques, so its methodological novelty is more concentrated than the number of proposed algorithms might suggest.

Presentation: 3: Clear and easy to follow.
Informativeness: 2: I learned something interesting or useful.
Soundness: 2: I have minor or fixable concerns about the paper’s soundness.
Critical Strengths:
Exploiting chain-graph structure to connect LCNs with existing credal-network algorithms is practically motivated and offers alternatives to optimization over the full joint distribution. CJT is particularly promising: under its stated structural conditions, replacing global joint-probability variables with consistent clique marginals can substantially reduce the formulation size on low treewidth graphs.
The empirical evaluation covers a broad collection of synthetic and realistic benchmarks. The reported results suggest computational benefits from CJT’s compact formulation and an attractive speed–accuracy trade-off for IBP on the synthetic instances, with accuracy measured against the CJT reference. The authors also report less favorable results for the credal-network approximations on non-family-realizable instances, helping readers assess their practical limitations.
Critical Weaknesses:
The global-optimality verification for CJT reference results needs clarification. CJT exactness depends on its stated structural conditions and on globally solving a potentially nonconvex optimization problem. The authors should state whether every CJT result used as a reference was accompanied by a global-optimality certificate and how uncertified cases were treated.
A more controlled analysis of the strong-extension relaxation would strengthen the empirical conclusions. The paper discusses relaxation error in Section 3.3 and evaluates non-family-realizable instances in Section 5, so this limitation is acknowledged. However, all synthetic benchmarks are family-realizable. On the realistic instances, Table 1 reports IBP upper-bound MAEs relative to CJT of 0.493 on MScaling and 0.263 on PScaling, compared with ARIEL’s 0.000 on both at the displayed precision. The authors also report that IBP and CDVE produce intervals 1.4–7.7 times as wide as the reference intervals. These results demonstrate a speed–accuracy trade-off, but leave the conditions under which the relaxation remains useful insufficiently characterized. Controlled experiments varying the proportion, type, and tightness of non-family-realizable constraints would help. On tractable instances, comparing exact inference over the strong extension with certified LCN bounds would isolate relaxation error, while comparisons with IBP and CDVE would reveal additional approximation error.
Proposition 2 appears to conflate the strong extension with the set of factorizing distributions that generates it. Equation (4) defines (
) as a convex hull, but the proof in Appendix C treats an arbitrary element of this hull as a product distribution, overlooking the fact that mixtures need not preserve independence.
Questions For Authors:
CDVE produces inner bounds relative to the strong extension; these are not automatically valid inner bounds for the original LCN when the two sets differ. Could the authors clarify this interpretation and state whether the cluster mean replacement in CVEcm preserves any rigorous lower or upper bound guarantees?
How do Equations (5)–(7) handle the family-realizable tree (
) with (
), (
, and (
)? With evidence (
), the factor messages yield ([0.5,0.5]) and ([0.9,0.9]), so Equation (7) appears to return an invalid interval ([0.9,0.5]), rather than the exact ([0.9,0.9]). Is an additional message-combination rule missing?
Equation (4) defines (
) as a convex hull, but the proof in Appendix C (Proposition 4) treats an arbitrary member as a factorizing distribution. Since mixtures of independent distributions need not preserve independence, how is the claimed equality (
) justified?
Flag For Ethics Review: No
Rating: 2: Weak rejection
Confidence: 3: I am familiar with this topic and its relevant literature.

Official Review of Submission39159 by Reviewer JNVR
Official Reviewby Reviewer JNVR17 Oct 2026, 16:01 (modified: 06 Nov 2026, 12:31)EveryoneRevisions
Paper Summary:
The paper takes advantage of the chain graph structure of Logical Credal Networks (LCNs) to reformulate inference as a credal network problem. Specifically, three approximate inference algorithms are put forth based on variable elimination, interval-valued message passage, and coordinate descent for computing posterior probability bounds. An exact inference method based on junction tree propagation (JTP) can solve substantially smaller nonlinear programs. Some theoretical results are obtained and experimental results are promising.

Presentation: 2: Generally understandable, but requires substantial effort in places.
Informativeness: 2: I learned something interesting or useful.
Soundness: 1: I have significant concerns about the paper’s soundness that do not appear easy to address.
Critical Strengths:
As only one approximate inference algorithm (ARIEL) exists for Logical Credal Networks, studying and proposing new exact and approximate inference algorithms is a worthy direction. The paper puts for not one, but multiple methods for LCN inference.

Critical Weaknesses:
Major Concerns

The paper does not provide a single formal algorithm specification or a proof of any theoretical claim in the main text, making it difficult to assess the novelty, correctness, and technical contributions of the proposed methods.
In S3.4 Credal Variable Elimination, no formal algorithm is provided. Instead, the proposed method is described using two paragraphs on p5.

In S3.5, Interval Belief Propagation, no formal algorithm is provided. Instead, the proposed method is described with one half of p6.

In S3.6, Coordinate Descent with Variable Elimination (CDVE), no formal algorithm is presented. Only one paragraph is used to describe the new approach.

In S4, Credal Junction Tree (CJT) is introduced without a formal algorithm. Instead, one half of a page is used to describe the method on p7.

Some of these appear incremental. For example, CDVE only requires one paragraph to describe. Similarly, CJT reads just like classical JTP. CJTP moralizes and then triangulates, producing a chordal graph. It constructs a junction tree satisfying the running intersection property. It assigns distributions over the cliques.

Likewise, no formal proofs of Theorems 1–6 are provided in the main text.

Experimental Evaluation

On p7, only Random LCNs with a moralized treewidth less than or equal to 3 (at most two parents per atom) are considered. While the text continues by stating that topologies both large and small are constructed, the critical issue is treewidth. Thus, having a "large" network but capping the treewidth at 3 is not interesting. Does this method scale to cases where the treewidth is larger than 3?

In [Darwiche, 2009], it is argued that treewidth can be overcome in some cases when the network involves a lot of determinism. In these cases, a network can be compiled into an arithmetic circuit even when the treewidth is high. How does the proposed method relate to this arithmetic circuit approach?

Weak Writing
The writing is not always precise. For instance, the Abstract simply states that the paper establishes theoretical properties of the proposed methods and shows substantial speedups over existing methods.

The organization could be better. It is strange to have Section S3.1 Credal Networks on p4, yet earlier have Section S2.1 Logical Credal Networks on p2. Perhaps it would be better to first define Credal Networks and then later define Logical Credal Networks.

Proposition 2 just appears. It is never introduced.

Similarly, there is no introduction to Theorem 1, which also just appears. There is no connecting sentence leading from Theorem 1 to Theorem 2. Instead, Theorem 2 just appears in the text after Theorem 1.

In fact, no formal theorem is ever introduced (Theorems 1-6). They all just appear in the text.

Should Credal Junction Tree (CJT) not instead be called Credal Junction Tree Propagation (CJTP)? The issue is that a junction tree is a type of graph, while junction tree propagation is a class of inference algorithms. The paper introduces CJT, which sounds like a type of graph, yet says it is for probabilistic inference.

Why is the complexity of CJT not given a numbered theorem like IBP and CVE?

The writing in S5 Experiments is choppy and unnatural. I counted 10 colons, not to mention the many semicolons and dashes.

I don't quite follow the significance of the lack of an acyclicity requirement mentioned on p1. In Fig 1, the cyclicity is between a pair of nodes C and D. Maybe it would be better to show a longer cycle. Then in the experimental results reported in Table 1, three of the six networks are classical Bayesian networks where acyclicity holds.

On p2, I find the phrases \psi to \psi and \phi to \phi strange. Maybe it would be clearer to introduce notation to represent formula nodes such as v_{\psi}.

Some citations are lacking or misattributed. For instance, while variable elimination (VE) is attributed to Dechter 2003, it is perhaps better to add (Zhang and Poole, 1996) at the very least. For example, (Zhang and Poole, 1996) is the sole citation for VE in the Koller and Friedman reference. Similarly, the running intersection property (RIP) was also attributed to Dechter 2003, whereas it could have been attributed to a number of references published in the 1980s.

References are listed inconsistently. Compare the first two references and you notice that even papers published in the same journal (IJAR) are shown differently. More generally, journal papers are listed differently (compare IJAR versus JMLR). Conference papers are listed inconsistently. Once again, two IJCAI papers are presented differently.

On a lesser note, calling the new method Interval Belief Propagation (IBP) will cause confusion in the community, since Iterative Belief Propagation (IBP) is already used in the literature [Darwiche, 2009].

Undefined Notation

p2 undefined notation: P
p2 undefined notation: l
p2 undefined notation: u
p2 undefined notation: (X \perp !!! \perp Y | Z)
p2 undefined notation: atoms(s)
p3 undefined notation: C
p3 undefined terminology: projection

Notation Drift

Boldface X is defined as a set of binary variables in S2.1 on p2, redefined as a set of random variables in S2.2 on p3, and changed again as a set of discrete variables with finite domains in S3.1 on p4.

G is a primal graph in S2.1 on p2, redefined as another type of graph in S2.2 on p3, and changed again as a DAG in S3.1 on p4.

pa(X_i) is defined as the parents of a propositional node in S2.1 on p2, then redefined as the parents of a node corresponding to a variable in S2.2 on p3, and then defined as the parents of a node in a DAG in S3.1 on p4.

Typos

missing period in Eq. (4)
Questions For Authors:
How does the proposed method relate to Darwiche's arithmetic circuit approach? In particular, discuss how your approach is better than the arithmetic circuit approach for overcoming high treewidth.
Flag For Ethics Review: No
Rating: 2: Weak rejection
Confidence: 3: I am familiar with this topic and its relevant literature.
