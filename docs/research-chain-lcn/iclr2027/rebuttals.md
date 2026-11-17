# Rebuttals — Submission 39159

## General response (all reviewers)

We thank the reviewers for their careful reading. The concerns about soundness were justified: several statements in the submission were wrong or not proven. We corrected them and reran every experiment. Changes are in **blue** in the revised PDF. The appendix was rewritten throughout, and sections with no counterpart in the submission are tagged *(new)*.

**Summary of the main changes**

1. **Restructuring.** Credal networks now come before LCNs (§2.1). CJT, the main contribution, is now §3. The credal-network route is §4. CVE, IVBP and CDVE are presented explicitly as adaptations of existing algorithms: §4.3 summarizes them, with Table 1 and Prop. 3. Their full descriptions, pseudocode and proofs are in Apps. F–H.
2. **Corrected theory.**
   - **The strong extension.** We now distinguish the product set $\mathcal P_{SE}$ from its convex hull $\mathcal D_{SE}$. Prop. 1 states $\mathcal D_{LCN}\subseteq\mathcal P_{SE}\subseteq\mathcal D_{SE}$, and Prop. 2 states $\mathcal P_{SE}=\mathcal D_{LCN}$. Both sets have the same bounds (Lemma 3).
   - **CVE pruning.** CVE now uses exact convex-hull pruning instead of dominance pruning (Thm. 5).
   - **IVBP.** IVBP was rebuilt as a credal sum-product algorithm, with a full exactness proof on polytrees (Thm. 7).
   - **CJT.** CJT's hypotheses H1–H4 are now checked mechanically for every instance (Thm. 1).
3. **Corrected experiments (§5, App. I).**
   - **Reference.** The reference is now CJT solved by the global solver SCIP. An endpoint is scored only if SCIP certifies it *and* H1–H4 hold.
   - **Benchmarks.** The experiments cover 740 instances. 480 are family-realizable LCNs with up to 100 atoms. The other 260 are *almost-realizable* LCNs, built by adding a controlled number of non-realizable sentences.
   - **Removed variants.** CVE$_{cm}$/CVE$_{cp}$ were removed.
   - **Realistic instances.** The 14 realistic LCNs (`real`) were rerun with the corrected code and scored the same way (§5 "Realistic LCNs"; App. I.8; Tables 2, 20, 21). They are 10 LCNs on classical Bayesian-network structures and the 4 application LCNs that the certified reference can handle. The submission's numbers came from the code before these corrections and were replaced.

---

## Reviewer A2AT

**Q1. CVE's dominance pruning does not preserve the extrema of normalized posteriors (old Thm. 2).**
You are right, and the rule was replaced. CVE now reduces every query-free potential to the pairs whose normalized functions are **vertices of the convex hull** of the set (§4.3; Alg. 2, App. F.5). The query ratio $q/(p+q)$ is linear-fractional in each function, so its extrema are attained at vertices (Lemma 3). Pareto pruning is applied only at the final step, where it is exact. A covering lemma (Lemma 12) shows that every reduction keeps, for every way the rest of the elimination can weight a pair, a pair at least as good for each bound. This gives exactness for $\mathcal D_{SE}$ with or without evidence (Thm. 5; proof in App. F.8). Empirically, CVE matches every certified endpoint on all 480 family-realizable instances (MAE $<10^{-5}$; §5).

**Q2. The IBP updates are non-standard, and the exactness proof (old Thm. 4) is missing.**
IVBP was rebuilt and renamed (IBP already denotes Iterative BP). It now runs sum-product on the credal network's factor graph. Each message is the set of **hull vertices of its normalized message vectors**; for binary nodes this is the factor-graph form of 2U (§4.3; updates in App. G.2, Alg. 3).

Every update is projective (linear, then normalized) in each input separately, so it maps the hull of a set into the hull of its image (Lemma 15). On a polytree the inputs of each update depend on disjoint sets of rows, so they vary independently. The hulls therefore propagate without loss, and every vertex is attained. Thm. 7 (proof in App. G.4) proves exactness on polytrees, including compound nodes and evidence. Empirically, IVBP is exact on every `tree`/`polytree` instance (§5). On loopy networks we state explicitly that it carries no guarantee.

**Q3. $\mathcal D_{SE}$ is a convex hull and may violate the LCN's independences.**
Agreed; the submission conflated the two sets. We now separate them:
- the product set $\mathcal P_{SE}$ (Eq. 1), whose members do factorize;
- its hull $\mathcal D_{SE}$ (Eq. 2), which may contain mixtures (§2.1).

The set-level results concern $\mathcal P_{SE}$:
- $\mathcal D_{LCN}\subseteq\mathcal P_{SE}$ for factorizing LCNs (Prop. 1);
- $\mathcal P_{SE}=\mathcal D_{LCN}$ under explicit hypotheses (Prop. 2): consistency, local feasibility, factorization, the product-Markov property, and full family-realizability.

The bounds transfer to $\mathcal D_{SE}$ because both sets have the same lower and upper probabilities, for marginals and conditionals (Lemma 3). App. E.4 gives counterexamples showing that each hypothesis is needed.

**Q4. How often do CJT's assumptions hold, and are the solver descriptions inconsistent?**
H1–H4 are checked for each instance rather than assumed (§3; App. B.2–B.3):
- H1 by a connectivity test;
- H2 and H4 by host search;
- H3 by construction, unless an assertion has to be relaxed, in which case the result is flagged.

They held on **all 740 instances**, through the ordered construction (Prop. 4), with no relaxed assertion (§5). The solver description is now the same in the main text and the appendix:
- Every endpoint is solved by SCIP. It is reported as certified if it is proven optimal or its primal–dual gap is $\le 10^{-4}$. Otherwise its dual bound is reported, which is a sound outer bound (§3; App. B.3).
- ipopt appears only as a separately labelled variant (CJT$_{\text{ipopt}}$). It is shown to fall into local optima on DAGs (§5).

**W1. The experiments rely on IPOPT, so CJT is not certified.**
The reference is now CJT solved by SCIP, and only certified endpoints are scored (§5; App. I.2). SCIP certifies all 17,700 endpoints on the `tree`, `polytree` and `ktree` sets, 98.6% / 81.6% of the endpoints on small / large DAGs, and 95.6–99.7% on the almost-realizable sets. The exact $2^n$ method certifies all endpoints only up to $n\approx 6$ (App. I.3).

**W2. Reliance on chain-graph structure and family-realizability, and the generalization of the approximations.**
CJT's exactness does **not** require family-realizability: it needs only H1–H4, which are checked per instance. Family-realizability matters only for the compiled route. Rather than restricting the benchmarks to family-realizable LCNs, we now measure what happens outside that class (§5; Fig. 3c; App. I.6):
- Adding $m$ non-realizable sentences makes the compiled bounds looser, but they stay **outer**: CVE contains the reference on 99.9% of atoms.
- The mean endpoint error grows from 0 at $m=0$ to 0.14 at $m=8$.
- CJT stays exact throughout.

**Realistic instances** (§5 "Realistic LCNs"; App. I.8). The rerun on the 14 realistic LCNs confirms your observation and explains it:
- Except `asia`, they all contain non-realizable sentences: 1–28 per scored instance, mostly non-root marginals.
- On `asia`, the only fully realizable instance, CVE and CDVE are exact.
- Elsewhere the compiled methods are outer on every atom, but their intervals are 1.1–8× as wide as the certified reference (3.5× on average for CDVE).
- ARIEL is indeed closer (MAE 0.011), but it has no guarantee. On `toy` its interval is narrower than the certified bounds (endpoint error 0.30), and it misses the reference interval on 2% of the atoms.

The compiled methods are therefore positioned as fast and exact on family-realizable LCNs, and sound but loose elsewhere. CJT is the general exact method, and it is what certifies the reference on these instances.

**W3. Limited novelty.**
We agree that CVE, IVBP and CDVE adapt existing techniques, and §4.3 now presents them that way. The main contributions are:
- **CJT** (§3), with mechanically checked exactness conditions (Thm. 1);
- the **characterization of when compilation is exact** (Prop. 2, the exact local sets for compound children in §4.1, and the counterexamples in App. E.4);
- **corrected guarantees** for the adapted algorithms under evidence and compound nodes (Table 1, Prop. 3).

---

## Reviewer c7e5

**W1. Are all CJT reference endpoints certified, and how are uncertified ones treated?**
Yes; only certified endpoints are used (§5 "Reference and metrics"; App. I.2). An endpoint enters the reference only if two conditions hold:
- SCIP proves it optimal, or its primal–dual gap is $\le 10^{-4}$;
- the structural conditions H1–H4 of Thm. 1 hold.

Uncertified endpoints are excluded from scoring, and CJT reports them by their dual bound, which is sound and outer. Certification rates are 100% on the `tree`/`polytree`/`ktree` sets, 98.6% / 81.6% on small / large DAGs, and 95.6–99.7% on the almost-realizable sets (§5).

**W2. A controlled analysis of the strong-extension relaxation.**
We ran it, following your design (§5 "Almost-realizable LCNs"; Fig. 3c; App. I.6).
- **Design.**
  - Family-realizable `tree`/`dag` bases ($n\in\{10,20\}$) receive $m\in\{0,1,2,4,8\}$ non-realizable sentences.
  - There are three kinds: non-root marginals (nr1), mixture conditionals (nr2) and cross-family sentences (nr3).
  - Tightness varies with $\delta\in\{0.05,0.1,0.3\}$.
  - A 60-instance ablation varies one kind at a time.
- **Isolating relaxation error.** CVE is exact for $\mathcal D_{SE}$, so the error of CVE against the certified LCN bounds *is* the relaxation error. The difference between IVBP/CDVE and CVE is the algorithmic error.
- **Findings.**
  - CVE, CVE$_\varepsilon$ and CDVE agree to within $2\times10^{-3}$, and IVBP agrees with them on trees. So the remaining error is the relaxation itself.
  - The relaxation is outer (99.9% of atoms).
  - It grows with $m$: the mean error is 0.03–0.04 at $m=1$ and 0.14 at $m=8$, and intervals become 1.3–2.3× wider.
  - It halves at $n=20$.
  - Non-root marginals are by far the most damaging kind (MAE 0.15, width ratio 3.9 at $m=4$, against 0.03–0.05 for the other kinds).
  - ARIEL stays within 0.013 MAE but has no guarantee.

**The realistic-instance numbers you cite** came from the uncorrected code. We reran the realistic LCNs and report the 14 that the certified reference covers (§5 "Realistic LCNs"; App. I.8; Tables 20–21). They behave like the almost-realizable sets:
- **Relaxation error, not algorithmic error.** CVE, CVE$_\varepsilon$ and CDVE return identical intervals, and IVBP is within 0.006 MAE of them.
- **Outer.** The intervals are outer on every atom.
- **Loose in proportion to non-realizability.** The width ratio is 1.1–2.3 on the Bayesian-network instances (at most 3 non-realizable sentences each) and 3.6–8.0 on the application LCNs (9–28 of 11–36 sentences non-realizable, mostly non-root marginals).
- **Exact where realizable.** On the one fully realizable instance (`asia`), CVE and CDVE are exact.
- **Certification and consistency.** The certified reference covers 86% of the endpoints (10 of 14 instances fully; none on `alarm`, where SCIP times out). We kept the application LCNs that the reference can certify. Of the 12 others, SCIP times out on nine judicial-power models and *proves* three chains inconsistent (CJT program infeasible with H1–H4 holding, Thm. 1); App. I.8 reports both.

**W3 / Q3. Old Prop. 2 / Prop. 4 conflate $\mathcal D_{SE}$ with the factorizing set.**
Correct, and fixed. We now distinguish the product set $\mathcal P_{SE}$ from its hull $\mathcal D_{SE}$ (§2.1, Eqs. 1–2). The claimed equality is now $\mathcal P_{SE}=\mathcal D_{LCN}$ (Prop. 2; proof in App. E.3); no member of the hull is treated as a product. The inclusion is $\mathcal D_{LCN}\subseteq\mathcal P_{SE}\subseteq\mathcal D_{SE}$ (Prop. 1; App. E.2). Bounds transfer to the hull because a linear-fractional objective attains its extrema at extreme points, which lie in $\mathcal P_{SE}$ (Lemma 3). App. E.4 shows the hypotheses of Prop. 2 are necessary.

**Q1. CDVE's inner bounds and CVE$_{cm}$.**
CDVE is inner for $\mathcal D_{SE}$ (every visited point is a member with $P(e)>0$). It is inner for $\mathcal D_{LCN}$ only when the two sets have the same bounds, e.g. under Prop. 2. Otherwise its relation to the LCN interval is not determined; this is now stated in Prop. 3 and Table 1 (§4.3). CVE$_{cm}$/CVE$_{cp}$ carried no rigorous guarantee and were **removed**. The guaranteed approximation is now the FPTAS CVE$_\varepsilon$ of Mauá et al. (2012). It returns inner bounds within a factor $1+\varepsilon$ of CVE's (Prop. 7; covering proof, Lemma 12).

**Q2. Old Eqs. (5)–(7) give the invalid interval [0.9, 0.5] on a tree with evidence.**
You found a real bug. The interval-combination rule of old Eq. (7) was replaced.
- **Why the bug cannot recur.** Messages are now sets of normalized vectors, stored as hull vertices. The belief is the hull of the normalized elementwise products of the incoming vertices, so its lower and upper bounds are a minimum and a maximum over one set, and an inverted interval cannot arise (App. G.1–G.2).
- **Your example.** We ran the corrected code on this tree: $P(A)=0.5$, $P(B\mid A)=0.9$, $P(B\mid\neg A)=0.1$, with evidence $B=1$. The prior message $[0.5,0.5]$ and the λ-message $[0.9,0.9]$ combine into the single belief vertex $0.5\cdot0.9/(0.5\cdot0.9+0.5\cdot0.1)=0.9$. IVBP returns the exact $[0.9,0.9]$, identical to CVE.
- **Exactness.** Thm. 7 proves exactness on polytrees with evidence. App. G.3 is a worked example.

**Formulation size vs. solving time.** Agreed. §3 now states after Thm. 2 that the bound concerns the formulation, not the solving time: global optimization of the nonconvex program is not polynomial in its size, but the program is exponential in $w$ rather than $n$.

---

## Reviewer JNVR

**No formal algorithms or proofs in the main text.**
Formal pseudocode is given for every method: Alg. 1 (CJT, App. B), Alg. 2 (CVE, App. F.5), Alg. 3 (IVBP, App. G.2) and Alg. 4 (CDVE, App. H). Full proofs are in Apps. C.2 (CJT), E (compilation), F.8, G.4 and H.2. Within the 10-page limit, the main text gives the key results with proof sketches:
- the CJT program and Thm. 1, with its two-inclusion proof sketch (§3);
- Prop. 2 and its proof idea (§4.2);
- Prop. 3, with the covering and projectivity arguments (§4.3).

App. A.1 adds a dependency map of all results.

**Incrementality; CJT reads like classical junction-tree propagation.**
CVE, IVBP and CDVE are adaptations, and §4.3 now presents them as such. CJT shares the junction-tree *structure* with classical JTP but differs in what it computes.
- **Calibration as constraints.** An LCN defines a nonconvex *set* of distributions, so CJT imposes calibration as constraints of one nonconvex program rather than computing it by message passing.
- **Discharging the independence assertions.** CJT must discharge the LCN's independence assertions on the tree, which has no JTP counterpart. It uses separator entailment, exact reduction of an assertion to the part still connected, or an ordered tree whose node-wise rows imply every assertion (Prop. 4; App. B.2).
- **Proving exactness.** CJT must prove that the result is exact. The reconstruction (Lemma 6) handles zero separator marginals without positivity, and H1–H4 are checked per instance (Thm. 1).
- **Conditional queries.** These need a Charnes–Cooper reformulation and certification by a global solver (§3; App. B.3).

On the name: following "the junction tree algorithm", we use CJT for the algorithm; §3 introduces it as "an exact algorithm".

**Treewidth capped at 3; does the method scale beyond it?**
The `ktree` sets have treewidth *exactly* 3, and the DAG sets $\le 3$. CJT's largest clusters have 2–8 atoms (§5). These sets were designed to separate the dependence on $n$ from the $2^n$ joint, and they show that CJT certifies exact bounds at $n=100$, where the joint formulation fails beyond $n\approx 6$. The dependence on treewidth is the one stated by Thm. 2: $O(n\,2^{w+1})$ variables. We agree that a treewidth sweep is informative, and we will add one to App. I: $k$-trees with $k\in\{2,\dots,6\}$ at fixed $n$, which our generator already produces.

**Relation to arithmetic circuits (Darwiche, 2009).**
Arithmetic circuits (ACs) compile the network polynomial of a *precise* network and exploit determinism and context-specific independence to evaluate it beyond the treewidth bound. LCN inference is different: it optimizes over a nonconvex set of distributions. An AC can therefore at best represent the *objective* inside the optimizer, not replace the optimization. The approaches are complementary, and we do not claim CJT is better.
- **CN route.** The local coefficients that CDVE computes for a family are exactly the partial derivatives of the network polynomial with respect to that family's parameters. One AC backward pass returns all of them, so an AC is a drop-in replacement for CDVE's variable-elimination step that would exploit determinism.
- **CJT.** A sentence with bound 0 or 1 forces cluster-state variables to zero, which presolve can remove. A compiled, determinism-aware encoding of the cluster marginals is a natural extension.

This discussion is now in the related work (§6).

**Writing and organization.**
- **Section order.** Credal networks are now defined first (§2.1), before LCNs (§2.2).
- **Abstract.** The abstract now states concrete results (certification rates, scale, conditions) instead of generic claims.
- **Lead-ins.** Each result is now introduced: Prop. 1 by the question of whether compilation loses any LCN member; Thm. 1 by its hypotheses and how they are checked; Thm. 2 after the proof sketch.
- **CJT complexity.** It is now a numbered theorem (Thm. 2, size), with B.4 giving the details.
- **Experiments.** §5 was rewritten.

**Acyclicity.**
LCNs allow a sentence such as $P(\neg(C\oplus D)\mid A)$ to tie $C$ and $D$ symmetrically. The structure then has an undirected chain component rather than a DAG edge, and this is not expressible as a Bayesian network over atoms. Our methods treat such components as compound nodes, with exact local sets (§4.1). Larger cyclic dependencies become larger components and are handled identically.

**Notation.**
- $\ell_s,u_s$, $\mathrm{atoms}(s)$ and $\tau_s$ are defined with the sentence forms (§2.2).
- The meaning of $(X\perp\!\!\!\perp Y\mid Z)$ is defined inline (§2.2; Def. 4).
- Clusters $C$ and $q_C$ are defined in §3.
- The **notation drift** is removed:
  - $\mathbf X$ always denotes the atoms, and the credal-network nodes are $\mathbf V$.
  - The graphs are distinguished as $G_P$ (primal), $G$ (chain graph) and $D$ (component DAG).
  - So are the parent sets: $\mathrm{pa}_P$, $\mathrm{pa}_G$ and $\mathrm{Pa}(v)/\mathrm{pa}_v$.
- App. A.2 tabulates all symbols.
- Formula nodes are now written $v_\varphi$, $v_\psi$, as you suggested (§2.2; App. A.3, proof of Lemma 4).

**Naming, citations and references.**
- Interval BP was renamed **IVBP** to avoid the clash with Iterative BP.
- Variable elimination now also cites Zhang & Poole (1996) (§4.3; App. F). The running-intersection property now cites Beeri et al. (1983) and Lauritzen & Spiegelhalter (1988) (App. B.1).
- The bibliography was harmonized: full author names, and the same fields and venue format for every journal and conference entry. Erroneous metadata was corrected.
