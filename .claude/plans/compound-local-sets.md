# Design — exact local credal sets for compound families

Goal: a representation of K(ch_v | pa_v = π) for a compound child (a multi-atom chain component)
such that the strong-extension bounds equal the 𝒟_LCN bounds whenever the sentences are
family-realizable.

## Findings (prototypes in /tmp/cmp/)

### Two cases, by the structure of the family
Let τ = ch_v (the component's atoms) and pa_v its parents. Take the moral graph of the family
τ ∪ pa_v (the second level of the LWF chain-graph factorization).

- **Case A — complete family graph:** τ is a complete component and every parent is adjacent to
  every atom of τ.
  - The chain-graph Markov property imposes **no constraint inside the family**: the second-level
    factorization is trivial.
  - The realizable sentences constrain each row **linearly** and **separately** (`eq:rowwise`). So
    the true family set is a product over π of polytopes
    K*_{v,π} = {r ∈ Δ(2^|τ|) : L^±_{s,π}(r) ≥ 0 for every s realizable at (v, π)}.
- **Case B — incomplete family graph.** The LMC contains assertions inside the family, e.g.
  (C ⟂ E | D) in a root path C — D — E, or (C ⟂ E | U, D) / (D ⟂ U | C, E) when U → C only and
  E → D only.
  - Inside one row these are **nonlinear**: the true row set is a curved, non-convex surface.
  - Across rows they **tie** the rows of different parent configurations together.

### Case A: exact polytopes give exact bounds (verified)
Replace the per-state interval box by the **exact projection of the local program onto the row**:

    K_{v,π} = { y(·, π) : y ≥ 0, A y ≤ b t, E y = 1 }

This is the Charnes–Cooper system of the local LP; a linear projection of a polyhedron is a
polytope.
- **How:** compute its vertices with the convex-hull method and an LP oracle (HiGHS):
  1. maximize coordinate and random directions;
  2. build the hull with qhull in the affine hull of the points;
  3. for every facet, maximize its normal; add any point beyond it;
  4. stop when no facet moves (finite: a polytope has finitely many facets).
- **It is the tightest local set.** It includes every in-scope sentence, realizable or not, so it is
  never looser than the box, which equals it for single-atom children.
- **Result on `examples/fr_counter_box.lcn`** (fully realizable, case A):
  | method | P(C) | P(D) |
  |---|---|---|
  | box | [0.03, 0.82] | [0.03, 0.82] |
  | exact polytope rows | **[0.32, 0.58]** | **[0.03, 0.82]** |
  | 𝒟_LCN (SCIP certified) | [0.32, 0.58] | [0.03, 0.82] |

  Vertex counts per row: 11 → 12 and 8 → 6.

### Case B: no finite exact representation by a credal network (verified)
`/tmp/cmp/path_inform.lcn`: a root path C — D — E with R1 sentences on the two cliques; the LMC is
(C ⟂ E | D).
- With exact row polytopes and no evidence, the bounds equal 𝒟_LCN (C, D, E ∈ [0.40, 0.55],
  [0.45, 0.55], [0.40, 0.55]).
- With evidence C=1, **P(E | C=1)** comes out **[0.333, 1.0]** against 𝒟_LCN **[0.520, 0.918]**
  (box: [0.333, 1.0]). The intra-family assertion is what tightens it, and a free row cannot
  express it.
- For bounds alone it would suffice to use the convex hull of the true family set S_v. The query is
  linear-fractional in the family's table, so its extremes over conv(S_v) lie at extreme points,
  which belong to S_v.
- **But the hull is curved.** I ran the convex-hull method with a *global* SCIP oracle that
  includes the CI equality p(c,d,e) p(d) = p(c,d) p(d,e). From 29 points, 1274 of the 1650 facets
  were cut in the first round, and qhull then failed on near-degenerate geometry. conv(S_v) has
  infinitely many extreme points, so no finite vertex set represents it.
- CVE, IVBP and CDVE all need finite extreme-point sets. **For case B the CN route can only give
  sandwich bounds** (see §2.B). Exactness needs the clique-level, nonlinear program, i.e. CJT.

### Coverage
Compound families in `examples/`, classified by an intra-family LMC test (`/tmp/cmp/classify.py`):
- **Case A:** alarm C-D, fr_counter_box C-D, ivbp_compound_mid B-C, ivbp_compound_parents B-C and
  F-G, new3 X2-X3 and X6-X7, smokers F1-F2-F3 and S1-S2-S3.
- **Case B:** only fr_counter_lmc_path, path_inform and partial_parent.
- The shipped benchmark sets have no compound nodes at all.

## 2. Proposal

### A. Exact polytope rows for compound families (implement)
1. **Compiler** (`credal_network` / `local_credal_sets`).
   - **Single-atom children:** keep the box (it is exact: an interval).
   - **Compound children:** compute the exact row projection's vertices with the convex-hull method
     and the HiGHS oracle above; infeasible rows are flagged as today.
   - **Storage:** the vertices go into the `.cn` file next to the box intervals, with a format
     version bump. The box is kept for display and for any interval consumer; the vertices are
     authoritative.
2. **`CredalNetworkVertices`.** For compound families take the stored exact vertices instead of the
   LRS vertices of the box; LRS stays for single-atom families. CVE, IVBP and CDVE consume extreme
   points and need no change.
3. **`linear-tight`.** It adds in-scope LMC equalities, which are nonlinear. The exact projection
   applies to the LP part. Keep the current behaviour for `linear-tight` (box), or project the LP
   rows only: open question.
4. **Robustness.**
   - The facet loop is capped; on reaching the cap, keep the points found plus the box as a sound
     outer fallback, and record a flag.
   - qhull failures (degenerate geometry) fall back to the LP vertex test we already have
     (`pair_potentials._lp_hull_vertices`), or to the box.
5. **Theory** (paper §4.2 / appendix): generalize `prop:suff` from "DAG-structured" to **"every
   family's moral graph is complete"** (case A), with exact polytope rows.
   - Proof: the row-wise lemma applies to the exact polytope; the LMC step follows because a
     product over the component DAG with complete families satisfies the LWF Markov property (the
     second level is trivial).
   - DAG-structured is the special case where every block is one atom.
   - The `fr_counter_box` counterexample then becomes a positive example. Keep it as the
     "box vs polytope" illustration.
6. **Tests:**
   - exact rows on `fr_counter_box`, matching certified 𝒟_LCN;
   - exact rows contain every true row and equal the box for single-atom children;
   - compound examples (alarm, ivbp_compound_*), where CVE with polytope rows is never looser than
     with boxes and still outer vs SCIP/CJT;
   - `.cn` round trip; timing on smokers (8-state blocks, 23-vertex rows).

### B. Case B (incomplete family graph): options
- **B1 (recommended): no exactness claim; outer via the polytope.**
  - **Detection:** a structural test flags these families (`realizability_report` gains
    `complete_families`).
  - **Bounds:** they are outer, using the exact polytope, which is still tighter than the box.
  - **Paper:** exactness for 𝒟_LCN is stated for case A, and case B is pointed to CJT.
- **B2: sandwich bounds.** Inner bounds from sampled points of S_v, found with the global oracle;
  outer bounds from the polytope cut by the oracle's support hyperplanes. Their gap shrinks with
  more oracle calls but never closes in finitely many steps on a curved set. Costly (SCIP per
  facet) and only approximate.
- **B3: refinement.** If the component's own graph is chordal and its sentences condition on
  separators, split the block into a sub-DAG, which is then case A or DAG-structured. Sentences on
  clique joints (as in path_inform) make the refined LCN non-realizable, so this only covers part of
  case B.

## 3. Decisions needed
1. **Implement A?** Exact polytope rows for compound families, and generalize `prop:suff` to
   complete families. Recommended.
2. **Case B:** B1 (detect; outer only; CJT for exactness), B2 (sandwich) or B3 (refinement)?
   Recommended: B1.
3. **`linear-tight` with compound children:** keep the box, or project its LP part exactly?
   Recommended: project the LP part. The extra LMC equalities are exactly the case-B constraints
   and stay out of reach anyway.

## 4. Status (implemented)
- **A:** `row_polytopes.py`; compiler `row_vertices` (CN/VTX v3); `_apply_row_vertices`; tests in `tests/test_row_polytopes.py`; `fr_counter_box` now exact.
- **B1:** `realizability.lmc_factorized`, reported as `complete_families` (the "product-Markov" hypothesis).
- **Paper:** `prop:suff`, `prop:suff-app` and `lem:noGapA` are generalized (factorizing + product-Markov); the K_v definition is now the polytope; the counterexamples are updated; `examples/fr_counter_lmc_evidence.lcn` is added.
- **Finding:** the structural "complete family" test of §1 is too weak. smokers, new3 and ivbp_compound_mid are *not* product-Markov:
  - A child that depends on part of a compound parent forces linear **row ties** (equal rows at parent states agreeing on the used atoms). This is representable as a table-level polytope, but CVE/IVBP/CDVE pick rows independently.
  - smokers also has nonlinear intra-family CI.
- **Open follow-up:** tie-aware engines (shared row choice across tied parent states). This would be exact for the linear-tie sub-case.
