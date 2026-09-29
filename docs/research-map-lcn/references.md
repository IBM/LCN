# Literature and source notes

Companion to [the research plan](research-plan.md). Reviewed 29 September 2026.
Entries distinguish primary full-text inspection from bibliographic/abstract
verification and material still requiring a deeper comparison.

## Direct predecessors

**[R1] Radu Marinescu, Junkyu Lee, Debarun Bhattacharjya, Fabio Cozman, and
Alexander Gray (2024). _Abductive Reasoning in Logical Credal Networks._ NeurIPS.**

[Proceedings page](https://papers.nips.cc/paper_files/paper/2024/hash/7d7020e945935214d756cd9a65c43170-Abstract-Conference.html)
· [local manuscript](../MAP_Inference_in_LCNs.pdf).

Primary text inspected. Section 3 defines maximin and maximax through the joint
explanation event P(y,e), describes assignment enumeration/LDS/SA, and introduces
approximate methods. Its concluding paragraph explicitly proposes depth-first
branch-and-bound and best-first search with new bounding heuristics. This is the
main baseline and the clearest statement of the gap to address.

**[R2] Radu Marinescu, Debarun Bhattacharjya, Junkyu Lee, Alexander Gray, and
Fabio Cozman (2023). _Credal Marginal MAP._ NeurIPS.**

[Proceedings page](https://papers.nips.cc/paper_files/paper/2023/hash/953390c834451505703c9da45de634d8-Abstract-Conference.html)
· [paper](https://papers.nips.cc/paper_files/paper/2023/file/953390c834451505703c9da45de634d8-Paper-Conference.pdf).

Primary text inspected, especially Definitions 1–3, Sections 4–5, and the
conclusion. It develops variable elimination over sets of functions, DFS,
credal mini-buckets, and local search. The detailed presentation emphasizes
maximax, and its mini-bucket bound is stated for maximax. Do not infer a correct
maximin search recursion by merely changing the pruning direction. Preserve
one compatible explanation when reconstructing a solution from sets of messages.

## AND/OR foundations and search algorithms

**[R3] Rina Dechter and Robert Mateescu (2007). _AND/OR search spaces for
graphical models._ Artificial Intelligence 171, 73–106.**

[DOI](https://doi.org/10.1016/j.artint.2006.11.003)
· [author-hosted manuscript](https://www.ics.uci.edu/~dechter/publications/r126.pdf).

Primary manuscript inspected for the search-space framework, decomposition,
pseudo-trees, and contexts. The author index labels the work 2006; the journal
metadata dates the volume to 2007. The plan uses the journal year. The foundational
width arguments apply to the represented factor problem; additional credal
constraints must appear in that representation before transferring a bound.

**[R4] Radu Marinescu and Rina Dechter (2009). _AND/OR Branch-and-Bound search
for combinatorial optimization in graphical models._ Artificial Intelligence
173, 1457–1491.**

[DOI](https://doi.org/10.1016/j.artint.2009.07.003)
· [author-hosted manuscript](https://www.ics.uci.edu/~dechter/publications/r151.pdf).

Primary text inspected. Use its depth-first tree search, static/dynamic variable
ordering, and mini-bucket heuristics as the baseline design. Its low-memory
search claims do not remove the memory costs of an LCN bound oracle.

**[R5] Radu Marinescu and Rina Dechter (2009). _Memory intensive AND/OR search
for combinatorial optimization in graphical models._ Artificial Intelligence
173, 1492–1524.**

[DOI](https://doi.org/10.1016/j.artint.2009.07.004)
· [author-hosted manuscript](https://www.ics.uci.edu/~dechter/publications/r153.pdf).

Primary text inspected, particularly context-based graph merging, cache-based
AOBB, best-first control, and assumptions on heuristic admissibility/monotonicity.
The LCN extension needs a new residual-equivalence argument when an ancestor's
probability model choices are not summarized by its world assignment.

**[R6] Radu Marinescu, Junkyu Lee, Rina Dechter, and Alexander Ihler (2018).
_AND/OR Search for Marginal MAP._ JAIR 63, 875–921.**

[DOI](https://doi.org/10.1613/jair.1.11265)
· [author-hosted paper](https://www.ics.uci.edu/~dechter/publications/r253.pdf).

Primary text inspected. Sections 3–4 describe weighted mini-buckets, constrained
pseudo-trees, AOBB, and best-first search. MAP variables form the upper portion
of a valid pseudo-tree; hidden-state summation and decision maximization cannot
be interchanged freely. The paper also explains the complementary anytime roles
of DFS and best-first search. Credal minimization adds another noncommuting
operator and an additional source of unresolved computation.

**[R7] Qi Lou, Rina Dechter, and Alexander Ihler (2018). _Anytime Anyspace
AND/OR Best-First Search for Bounding Marginal MAP._ AAAI.**

[Author-hosted paper](https://www.ics.uci.edu/~dechter/publications/r246.pdf).

Primary text inspected. The method brings external maximization and internal
summation into one best-first framework, selects useful refinements, and handles
limited memory. It motivates treating an unfinished LCN score computation as
search work rather than requiring every leaf oracle to finish immediately.

**[R8] Radu Marinescu, Akihiro Kishimoto, Adi Botea, Rina Dechter, and
Alexander Ihler (2019). _Anytime Recursive Best-First Search for Bounding
Marginal MAP._ AAAI.**

[DOI](https://doi.org/10.1609/aaai.v33i01.33017924)
· [author-hosted paper](https://www.ics.uci.edu/~dechter/publications/r255.pdf).

Primary text inspected. Provides the main recursive best-first comparison for
the memory-limited phase. A new LCN method should be compared with this design
and [R7], not positioned as the first anytime or bounded-memory AND/OR method.

## Heuristic bounds and credal semantics

**[R9] Rina Dechter and Irina Rish (2003). _Mini-buckets: A general scheme
for approximating inference._ Journal of the ACM.**

[DOI](https://doi.org/10.1145/636865.636866).

Bibliographic metadata checked; the relevant bound constructions were read
through [R6], Sections 3.1–3.4. Further reading is Qiang Liu and Alexander Ihler
(2011), _Bounding the partition function using Hölder's inequality_, and their
work on variational marginal MAP, cited in [R6]. These are prerequisites for a
weighted-mini-bucket implementation. Re-derive every inequality after adding
credal minimization, logical constraints, or message compression.

**[R10] Salem Benferhat, Amélie Levray, and Karim Tabia (2017). _Approximating
MAP Inference in Credal Networks Using Probability-Possibility Transformations._
ICTAI.**

[DOI](https://doi.org/10.1109/ICTAI.2017.00162).

Title, authors, venue/year and DOI checked through bibliographic metadata.
Detailed comparison and full-text verification remain work for the first phase.
Determine exactly which MAP objective and credal representation its approximation
targets before using it as a quality or bounding baseline.

**[R11] Fabio G. Cozman, Radu Marinescu, Junkyu Lee, Alexander Gray,
Ryan Riegel, and Debarun Bhattacharjya (2024). _Markov conditions and
factorization in logical credal networks._ IJAR 172, 109237.**

[DOI](https://doi.org/10.1016/j.ijar.2024.109237)
· [local revised manuscript](../LCN_IJAR_Revised.pdf).

Primary local text inspected, especially Section 5 on chain graphs, positivity,
internal component structure, and zero-probability complications. Distinguish
factorization of each precise distribution from independent specification of
the set of all such distributions. The latter is needed for scalar robust AND
backups and is a stronger requirement.

**[R12] Denis D. Mauá and Fabio G. Cozman (2020). _Thirty years of credal
networks: Specification, algorithms and complexity._ IJAR 126, 133–157.**

[DOI](https://doi.org/10.1016/j.ijar.2020.08.009)
· [local copy](../1-s2.0-S0888613X20302152-main%20(1).pdf).

Reviewed during the preceding sampling research and used here for semantic
context: strong extensions, separate specification, and the limitations of
credal inference. The original LCN set includes independence equalities and
need not be convex; replacing it with a convex strong extension is a semantic
choice, not merely a storage optimization.

## Related robust-optimization ideas

**[R13] Bo Zeng and Long Zhao (2013). _Solving two-stage robust optimization
problems using a column-and-constraint generation method._ Operations Research
Letters.**

[DOI](https://doi.org/10.1016/j.orl.2013.05.003).

Bibliographic identity checked. Full methodological comparison is follow-up work.
The plan uses scenario-master/adversary iteration as related prior art and derives
its LCN bound inequalities directly. It does not assume this paper's convergence
conditions cover the LCN's continuous, nonconvex feasible model set.

**[R14] Garud N. Iyengar (2005). _Robust Dynamic Programming._ Mathematics of
Operations Research.**

[DOI](https://doi.org/10.1287/moor.1040.0129).

Bibliographic record and abstract checked. The abstract explicitly identifies a
rectangularity property that permits robust counterparts of dynamic-programming
results. This motivates the distinction between probabilistic independence and
independent admissible model choices. Its dynamic decision setting differs from
static LCN MAP; a new decomposition theorem must be established here.

## Scope and follow-up search

The current review used the local MAP/factorization papers, the author-hosted
AND/OR publication collection, official NeurIPS proceedings, and Crossref
metadata. OpenAlex also identified the probability–possibility comparison;
some broader queries were rate-limited. No exhaustive novelty claim is made.

Before writing a paper, complete a citation matrix recording for each method:
model class, joint/posterior objective, maximin/maximax order, representation of
uncertainty, decomposition assumptions, search space, admissible heuristic,
anytime guarantee, memory management, and implementation availability. Search
forward from [R1], [R2], [R6], and [R13], including work through the submission
date. Check implementations and supplements for the precise handling of model
witnesses and assignment reconstruction.

Keep reproducibility records for the selected paper versions and software
revisions. The author-hosted PDFs may differ in pagination from final journal
versions; section and algorithm references are more reliable than page numbers.
