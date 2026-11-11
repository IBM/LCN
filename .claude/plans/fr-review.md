# Plan — review of §4.2 (family-realizable sentences) and its appendix (Sec. "The LCN set versus the strong extension")

Scope:
- Main text: §4.1 tail (definitions of exact / outer / inner, Prop. `prop:inclusion`), §4.2
  (`def:realizable`, `prop:suff`, the closing paragraph).
- Appendix: `sec:prelim-gaps` (eq. `eq:se-app`), `def:realizable-app`, `lem:invariance`, Example
  `ex:fr5`, `sec:incl` ("factorizing"), `prop:incl`, `lem:noGapA`, `prop:suff-app`.
- Downstream results that cite them: `thm:cve`, `thm:ibp` (compound claim), the CDVE inner claim,
  `lem:hull`.

## 1. Findings (verified numerically; test files in /tmp/fr/)

### F1 (critical). The compiled local sets are interval boxes, so `prop:suff` / `lem:noGapA` fail for compound families
- **What the code computes.** The compiler (`credal_network._solve_family`) stores, per state of
  the child block and per parent configuration, a lower and an upper bound (`lobo`/`upbo`). The
  vertices are those of box ∩ simplex. The main text describes exactly this ("computing lower and
  upper bounds on the conditional probabilities of ch_v given π").
- **What the proofs assume.** The appendix proof of `prop:incl` defines K_{v,π} as "the set of rows
  attainable by some joint satisfying exactly those in-scope sentences", i.e. the exact local
  projection.
- **Why it matters.** For a single-atom child the projection is an interval, so box = projection.
  For a compound child it is a polytope, and the box is strictly larger.
- **Counterexample** `/tmp/fr/r2_compound.lcn`: A → {C, D}; every sentence is R1 or R2, so the LCN
  is fully realizable per Def. 4.2.
  - The compiled row A=1 allows P(C⊙D) ∈ [0.8, 1] and P(C) ∈ [0.1, 0.4], wider than the
    sentences' [0.9, 1] and [0.2, 0.3].
  - Row A=0 allows P(C⊙D) ∈ [0, 1] and P(C) ∈ [0, 1] against [0.4, 0.6] and [0.5, 0.7].
  - Result: CVE (exact for 𝒟_SE) gives **P(C) ∈ [0.03, 0.82]**, whereas certified exact inference
    over 𝒟_LCN gives **[0.32, 0.58]**. So 𝒫_SE ≠ 𝒟_LCN, and the bounds differ.
- **Consequence.** `lem:noGapA`'s claim K_{v,π} = K^true_{v,π} is false for compound children, and
  so is `prop:suff`. The exactness-w.r.t.-𝒟_LCN parts of `thm:cve` and `thm:ibp` ("for binary
  and multi-valued (compound) variables alike") are false for compound nodes; they hold
  w.r.t. 𝒟_SE.
- `prop:incl` (outer) stays true: the box contains the projection.

### F2 (critical). The LMC step of `lem:noGapA` / `prop:suff-app` is wrong for chain components with more than one atom
- **The claim.** "q is a product of the form (eq:factor); factorization implies the global Markov
  property of the chain graph without positivity, so q satisfies every LMC assertion."
- **Why it fails.** Lauritzen's converse needs the *two-level* chain-graph factorization: each
  P(τ | pa τ) must itself factorize over the moral graph of τ ∪ pa(τ). The strong extension's
  free rows impose only the first level.
- **Counterexample 1** `/tmp/fr/partial_parent.lcn`: U → C, E → D, component {C—D}.
  - The LMC contains (C ⟂ E | U, D) and (D ⟂ U | E, C).
  - These tie the rows of the family P(C, D | E, U) *across* parent configurations, which the
    product of independent rows does not enforce.
- **Counterexample 2, fully realizable (R1 only)** `/tmp/fr/root_path.lcn`: a root component
  C—D—E (path), with LMC (C ⟂ E | D).
  - This is a nonlinear constraint *inside a single row*, so the true local set is not even
    convex.
  - The compiled vertices violate it by up to 0.25. Hence 𝒫_SE ⊄ 𝒟_LCN even under full
    realizability. Here the bounds coincide only because both are vacuous.
- **The "factorizing" definition** (`sec:incl`) and its positivity paragraph address
  𝒟_LCN ⊆ 𝒫_SE only; they do not repair the converse.

### F3 (major). The "if and only if" of Def. 4.2 / `def:realizable-app` and `lem:invariance` is false
- **What `lem:invariance` covers.** Only (i) Type-1 sentences and (ii) Type-2 sentences with
  atoms(ψ) ⊆ pa_v, and only for pa_v ≠ ∅. It misses:
  - (a) Type-2 sentences whose ψ mentions child atoms, e.g. P(C | D) on a root compound block, or
    P(C | D ∧ π*). These are functions of one row, a ratio of two linear forms, so they are
    realizable but neither R1 nor R2 (verified: `/tmp/fr/root_block.lcn`).
  - (b) R2 requires atoms(φ) ⊆ ch_v, but P(C ∧ U | U) = P(C | U) is realizable.
  - (c) Trivial φ / ψ (tautology, unsatisfiable): (i) says invariance "fails whenever φ is
    non-trivial", but the proof picks T(φ_{π1}|π1) = 1 and T(φ_{π2}|π2) = 0. That needs φ_π
    non-trivial for *both* chosen configurations, which is a gap when φ depends on π (e.g.
    φ = C ∨ U).
- **Consequence.** The sentence "the two forms exhaust the P(pa_v)-invariant sentences, so
  realizability can be decided syntactically" is false as stated.
- **Correct exact characterization (to prove).** s is realizable in v iff atoms(s) ⊆ ch_v ∪ pa_v
  and either
  - Type-1 with pa_v = ∅, or φ's truth value independent of the parents; or
  - Type-2 whose ψ-models (over ch_v ∪ pa_v) all lie in one parent configuration π*, i.e.
    ψ ⊨ (pa_v = π*), or ψ is unsatisfiable;
  - modulo trivial sentences.

  R1/R2 (what the generator emits) are the special case atoms(φ) ⊆ ch_v, atoms(ψ) ⊆ pa_v.

### F4 (minor). Statements that are out of date or imprecise
- §4.2 closing: "three algorithms that compute, respectively, exact, outer and inner bounds for
  𝒟_SE … By Prop. inclusion the first two are outer bounds for 𝒟_LCN" is out of date. IVBP is
  exact on polytrees and has no guarantee on loopy graphs.
- **The gaps paragraph** lists Gap A (per-family isolation) and Gap B (free product). It misses the
  box widening of F1 (call it Gap A′) and the intra-family LMC of F2.
- **`prop:suff`:** "every conditional query with positive denominator" should say
  P̲(e) > 0 (positive on all of 𝒟_SE, as `lem:hull` needs).
- **K^true_{v,π}** is undefined when P(pa_v = π) = 0 in every member. The `lem:noGapA` proof
  waves this ("the row is immaterial"); define it as the rows of factorizations of members,
  arbitrary on null configurations.
- **`ex:fr5`:** the measured-bounds table must be re-run with the current engines (CJT changed
  since).
- **Main text vs appendix:** the main text uses `prop:inclusion`, `prop:suff`, the appendix
  `prop:incl`, `prop:suff-app` (two copies). Keep them consistent after the fixes.

### F5 (side finding, code). Parser: `; False` means τ = True
- `model.py:1124` sets `tau = bool(line[pos+1:].strip())`, so any non-empty flag, `False`
  included, yields τ = True.
- No shipped `.lcn` file uses `; False` (checked), so no result is affected. It is still a latent
  semantics bug.

### What is *not* affected
- Every shipped benchmark set (`*_small_fr`, `*_large_fr`) is DAG-structured: all 48 sampled
  instances have one atom per node.
- For single-atom families the box equals the projection (F1 vanishes), every component is a
  single atom (F2 vanishes), and "factorizing" holds without positivity.
- So the experimental claims on the benchmark sets survive. The theory over-claims for compound
  nodes (alarm, smokers, `new3`, the `ivbp_compound_*` examples).

## 2. Fixes

### Paper, minimal-correct option (recommended)
1. **Restate `prop:suff` / `prop:suff-app` and `lem:noGapA`** for **DAG-structured** LCNs (all
   chain components single atoms), plus consistency and full realizability.
   - Then factorizing holds without positivity, box = projection, and the first-level product is
     the DAG factorization, so the global Markov property follows.
   - Optional generalization in a remark: chain components whose family moral graph is complete
     (no intra-family LMC), with local sets computed as exact polytopes. See the code option
     below.
2. **Rewrite `lem:noGapA`'s proof:**
   - box = projection for single-atom rows (the projection of a polyhedron under a linear-fractional
     map with positive denominator is an interval);
   - the precise K^true definition;
   - the LMC step via the DAG chain rule.
3. **Definition 4.2:** keep the semantic definition. Replace "iff it has one of two forms" with
   "in particular, it is realizable if it has form (R1) or (R2)". Add `lem:invariance` restated as
   the exact characterization (F3), with a corrected proof that handles π-dependent φ and trivial
   sentences. R1/R2 remain the generator's forms (note that they are sufficient, not necessary).
4. **`prop:inclusion` / `prop:incl`:**
   - say K_{v,π} is the interval box of the local projection (what the code computes); the proof's
     "contains the projection" then reads "contains its box";
   - keep the factorizing hypothesis and the positivity remark.
5. **Gap paragraph:** add the box widening for compound blocks and the intra-family LMC as reasons
   compound families are only outer.
6. **Downstream statements:**
   - `thm:cve`, `thm:ibp` and the CDVE text: "exact (inner) w.r.t. 𝒟_SE; w.r.t. 𝒟_LCN under the
     hypotheses of `prop:suff` (DAG-structured, fully realizable)";
   - `thm:ibp` drops "multi-valued (compound) variables alike" for the 𝒟_LCN claim but keeps it
     for 𝒟_SE;
   - fix the stale "exact, outer and inner" sentence.
7. **Rerun `ex:fr5`'s table** (exact_scip, cjt_scip).
8. **Add a counterexample paragraph** in the appendix: the r2_compound instance (F1) and the root
   path (F2), showing the DAG restriction is necessary for the current compilation.

### Code, optional (would let the theorem cover compound families)
- **C1.** Compile compound families' local sets as **exact polytopes**: the row-wise constraints of
  realizable sentences are linear in the row, so the vertices come from an H→V conversion per row
  (dims ≤ 2^|block|; LRS / cdd). This needs a `.cn` format extension (store H-rows or vertices) and
  cache invalidation. It still does not fix F2 (intra-family LMC), which would additionally
  require "family moral graph complete" as a hypothesis or nonconvex rows.
- **C2.** Fix the τ parser (F5): parse `True` / `False` explicitly and reject other values.

## 3. Verification
- Keep the three counterexamples as regression files `examples/fr_counter_{box,lmc_path}.lcn`,
  plus a test asserting that the DAG-structured realizable instances satisfy CVE = exact_scip
  (`tests/test_realizability.py`). Use the shipped `tree`/`polytree`/`dag_small_fr` sets: certified
  equality of all endpoints on a sample, with and without evidence (P̲(e) > 0 cases).
- Test the corrected characterization (F3) against brute-force invariance: random small families,
  random sentences, numeric check of P(pa_v)-invariance vs the syntactic test.
- Recompile the paper; check every reference to `prop:suff`, `lem:noGapA`, `thm:cve`, `thm:ibp`.

## 4. Decisions needed
1. **Fix scope.** Restrict the sufficiency theorem (and the downstream 𝒟_LCN exactness claims) to
   DAG-structured LCNs (recommended; matches every benchmark), or also implement C1 and state a
   compound-family version (complete family moral graphs + exact polytopes)?
2. **Definition.** Replace the false "iff" with an exact characterization lemma (recommended), or
   simply weaken it to "if (R1) or (R2)"?
3. **Parser.** Fix the τ parsing bug (C2) as part of this pass?

## Decisions (2026-11-11)
1. **Fix scope:** restrict `prop:suff` / `lem:noGapA` and the 𝒟_LCN-exactness parts of `thm:cve`,
   `thm:ibp` (and the CDVE inner-for-𝒟_LCN claim) to **DAG-structured** LCNs. Compound families
   are exact (inner) w.r.t. 𝒟_SE and outer for 𝒟_LCN. Paper-only fix, plus the counterexamples as
   regression files and tests. No compiler extension.
2. **Definition:** keep the semantic definition. R1 / R2 become *sufficient* forms (the
   generator's). Replace `lem:invariance` by an exact characterization with a corrected proof
   (child atoms in ψ, parent atoms in φ, π-dependent / trivial sentences).
3. **Parser:** fix the τ flag parsing (`True` / `False` explicit, other values rejected) and add a
   test.
