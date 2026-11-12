# Revision plan v2: proofs, paper ↔ code consistency, benchmarks

Scope: `iclr_2027.tex` (main text and appendix), the engines under `lcn/inference/marginal/cn/`, and the shipped `*_fr` / `nr_*` benchmarks.
Experiment results are out of scope; they will be rerun.
The rerun must wait for the Part 1 code fixes, because several of them change reported numbers.

The plan is based on three read-only audits (CJT; compilation and realizability; CVE/IVBP/CDVE).
The fourth audit (benchmarks) has not reported yet, so Part 6 lists its checks and the issues already known; it will be completed when that audit reports.
Line numbers refer to `iclr_2027.tex` as of commit `20f9de5`.

The audits found no soundness error in the main CVE, IVBP, CDVE and RIP arguments:
- the covering lemma and the FPTAS inner direction;
- the polytree hull argument, including the multi-parent box;
- Dinkelbach and barren pruning;
- `lem:rip` with zero separators, `lem:reduce` and `lem:invariance`.

They did find two code bugs that change results (Part 1, items 1.1 and 1.3), one false example claim, and many gaps in the proofs and in disclosure.

---

## Part 1 — Code fixes (before the rerun)

### Critical

- **1.1 The CJT augmented tree breaks running intersection (H1) and still reports `exact`.**
  - **Cause:** `_JTClusters.__init__` passes `exclude={query}` to `min_fill_order` (`cjt.py:204,217`). The query node never gets a bucket, so the buckets whose message is `{Q}` become separate roots: the result is a forest, with no separator tying the copies of Q together.
  - **Repro** (`/tmp/cjtaudit/forest2.lcn`): E→X→Q, Q→B, Q→C, evidence E=1. CJT reports hi P(Q|E)=0.87, certified, `h3`, structurally exact. The correct value is 0.2706, and a hand bound gives ≤ 0.444.
  - **Fix:**
    - Drop the exclusion; the query ∪ evidence clique already provides a host.
    - Add an explicit RIP check: for every atom, the clusters containing it must form a connected subtree. Set `exact=False` if the check fails.
    - Fix the misleading comment at `cjt.py:177-180,193`.
    - Add `forest2.lcn` to `examples/` as a regression test.
  - **Harness:** `analyze_results.py` selects reference endpoints by `certified` alone. It should require `certified ∧ cjt_exact`, so that endpoints from a broken structure or a relaxed H3 can never become the reference.
- **1.2 The CJT Charnes–Cooper program is unbounded.** y and r have no upper bounds (`cjt.py:575-577`), and SCIP times out even at n=5.
  - Add the valid bounds r ≥ 1 (since r = Σy ≥ ⟨A_e,y⟩ = 1) and y_{C,i} ≤ r.
- **1.3 IVBP when P̄(e)=0.** A λ-message whose corners all have zero mass is replaced by the neutral message (`ivbp.py:297-302`), so other atoms get informative bounds. Repro: A→B with P(B|·)=0 and evidence B=1 gives A∈[0.4,0.6], while CVE and CDVE give [0,1].
  - Return an empty message, and report [0,1] for every atom whenever any belief is empty.

### Major

- **1.4 CVE bucket products grow to C^k.** `_combine` (`cve.py:240-246,263-265`) does not reduce between the pairwise products inside a bucket. This contradicts the C² in `thm:cve-cplx`.
  - Preferred fix: apply `prune_query_free` after each pairwise product. This is sound by `lem:cover`(2).
  - Otherwise, redefine C in the theorem.
- **1.5 The structure check accepts structures that are not chain graphs.** `LCN.simplify_structure_graph` (`model.py:695-715`) contracts each undirected component and silently drops the directed edges inside it. Example: A→B together with A–C–B.
  - Either reject such structures (and warn), or state every result for the compiled block DAG and describe the contraction in the paper (see 2.6).
- **1.6 Undisclosed or unsafe fallbacks in compilation.**
  - Add the rows where `row_polytope_vertices` failed (box kept) to `realizability_report` as a field, e.g. `exact_local_sets`.
  - `LocalCredalSetSolver.locally_feasible` returns True when HiGHS is missing or raises (`local_credal_sets.py:232,249`). Make it raise or return None instead.
- **1.7 CJT smaller items.**
  - When `gap_tol>0`, the incumbent is reported as certified. Either always report the dual bound, or assert `gap_tol==0`.
  - Sentences with no atoms are silently skipped (`cjt.py:1298`). Check them for consistency or raise.
  - The d-separation safety net (`cjt.py:1120-1127`) must require Z to be a union of whole nodes.
  - Optional: contract every subsumed cluster, not only subsumed leaves. In the alarm example this gives 16 variables instead of 22, and it changes the example's counts.
- **1.8 Example file.** `examples/fr_counter_factorizing.lcn` is not fully realizable: s4 = P(C∨D) is on a non-root compound block. It is also not product-Markov.
  - Either replace s4 by an R2 sentence such as P(C∨D | Y) and rerun its numbers, or change the text (see 4.6).

### Docs

- CLAUDE.md and the docstrings at `cve.py:61-63` and `pair_potentials.py:274` say "one pair per α-class". The code keeps **two** pairs per class, which matches the paper; fix the docs.
- Fix the class-count formula comment to include the zero class.

### Tests to add

- The RIP-connectivity check, using `forest2`.
- IVBP with P̄(e)=0.
- The CVE pairwise-reduce gives identical results to the current code on the existing regression set.
- CJT with the new CC bounds gives identical certified values on `cjt_zero_evidence` and `cjt_dropped_lmc`.

---

## Part 2 — Foundations (Sec. 2, App. A): definitions every proof relies on

1. **Conditional independence semantics.** Define (X ⟂ Y | S) as the equalities `eq:lmc` for **all** (x,y,s), with configurations s of probability zero vacuous. The current equality quantifies over (y,z) only (l.185, 755-759).
   - Move the fact "the x=1 slice suffices" to `sec:nlp`, with its one-line derivation.
   - State the semi-graphoid axioms used later.
   - Cite factorization ⇒ global Markov without positivity (Lauritzen 1996, Prop. 3.8 / Thm 3.27).
2. **D_LCN in cleared form.**
   - Define D_LCN with the rows ℓP(ψ) ≤ P(φ∧ψ) ≤ uP(ψ) (l.185-190 and l.759-764).
   - Define g_s explicitly, as the ratio, used only in the realizability definition.
   - Define atoms(s), τ, and "attached to a family".
   - τ is never defined. Add it to the structure construction (l.158-160), since the counterexamples and `nr3` rely on τ = False.
3. **Conditional query semantics.**
   - Define lower and upper P(a|e) as **inf/sup over {p ∈ D : p(e) > 0}**, which is what every engine implements (regular conditioning).
   - Replace min/max by inf/sup throughout (`thm:cjt`, `thm:d5`, `eq:nlp-obj-ev`), and say when the bounds are attained.
   - Define the output as [0,1] when P̄(e) = 0.
4. **lem:hull, extended.**
   - (a) P_SE is compact: it is the continuous multilinear image of a product of polytopes.
   - (b) Conditional version: for b ≥ 0, a mixture with ⟨b,p⟩ > 0 is a μ-average of the ratios of its components with ⟨b,p_k⟩ > 0, where μ_k ∝ λ_k⟨b,p_k⟩. So the inf/sup over CH(S) ∩ {b>0} equal those over S ∩ {b>0}.
   - This removes the hypothesis "P̲(y) > 0" from `prop:suff`, `thm:cve` and `thm:ibp` (replace it by P̄(y) > 0).
   - Fix l.147: ext(D_SE) is a **subset** of the vertex products.
5. **DAG-structured lemma** (new). For a DAG structure:
   - the primal-graph parents and descendants used by LMC(LCN) (l.162, whose descendant definition is unusual) coincide with those of the DAG;
   - hence the LMC is the directed local Markov property;
   - hence the LCN is **factorizing** (no positivity needed) and **product-Markov**.

   This is asserted at l.251, l.2011-2013, l.2117 and l.2127-2131 but never proved, and `realizability.py:172` hard-codes it. State the lemma once and cite it everywhere.
6. **Compound components.** "Factorizing" cannot be checked for compound LCNs, and the code never tests it.
   - Either state the compound guarantee against the **closure of the strictly positive part** of D_LCN (contained in P_SE, since P_SE is closed),
   - or give a checkable sufficient condition (e.g. the LMC is trivial inside every compound component).
   - Also say how a structure that is not a chain graph is handled (see 1.5).
7. **Notation sweep.** Add a notation table to App. A and enforce one meaning per symbol:
   - G: CN DAG, primal graph, chain graph → use G_P, G, D.
   - pa: in the primal graph, the chain graph, pa_v as nodes and as atoms → separate them.
   - ch_v: variables vs child node → one meaning.
   - V_i: CN variables vs components → separate.
   - X/Y/Z: atom sets vs LMC blocks → separate.
   - T: chain components, row mass, tables, junction tree → separate.
   - D: DAG, neighbour cluster, atom → separate.
   - C: cluster vs atom → separate.
   - S: separator, conditioning set, compact set → separate.
   - A_a… overloaded → rename.
   - m, n: reused for different counts → fix.
   - π_ik vs π → one form.
   - b: budget vs indicator → separate.
   - e: evidence vs probability → separate.
   - \readoff: ratio vs CC objective → two names.
   - Use the macros at l.368.
   - Define Λ_o, Λ_m, Ordered and Triage, which appear undefined in Alg. 1.

---

## Part 3 — CJT (Sec. 3, App. B–C)

1. **One theorem statement, with identical wording in the main text and the appendix.** Given a tree T, define G*(T) from it, then H1–H4.
   - Remove the circularity "a junction tree of its chordal graph G*" (l.322-323 vs `def:d5` l.1517-1521).
   - Restore "and satisfies Λ_T" in the main-text H3.
   - Define "implied by Λ" semantically: every distribution satisfying Λ satisfies the row. Then state that the code certifies this syntactically (decomposition and symmetry of a single assertion with the same conditioning set).
   - Make DLCN ≠ ∅ an explicit assumption; otherwise CJT returns [0,1], which is not "exact".
   - The theorem is stated for inf/sup (Part 2.3).
2. **Helly lemma** (new, one line): every cluster is a clique of G*, and every clique of G* lies in a cluster. `lem:rip`, `lem:reduce` and `prop:ordered` use this silently (l.1521-1523, 1675).
3. **lem:rip.**
   - Handle an empty separator explicitly (q_S ≡ 1, κ = q_C), as the code does at `cjt.py:603`.
   - Define RIP for forests: the clusters containing an atom lie in one component and form a connected subtree. This is exactly the property item 1.1 violated.
   - Add one sentence each for "removing a leaf preserves RIP" and "C∖S meets no cluster of T′".
   - Mark uniqueness as a side remark (it is not used).
4. **lem:reduce.**
   - Require X, Y, Z to be disjoint.
   - G* must be the graph of the final tree, after widening.
   - Add the one-line argument that Z separates K from Y_out.
5. **prop:ordered.**
   - The soundness hypothesis must be "each row is entailed by a single LCN assertion", which is what `_lmc_entails` checks — not "follows from the local Markov property of D".
   - Prove the perfect-elimination fact: x's later neighbours are exactly N_x, by induction on the buckets.
   - State that in practice the ordered tree applies only to DAG-structured LCNs. Any compound atom has its sibling in its boundary, so the code rejects it. Then either drop the compound-encoding sentence (l.1023-1025) or label it as unreachable.
6. **Proof of thm:d5.** Add:
   - compactness of the feasible set (so the optimum exists);
   - P_p(e) > 0 in the read-off identity;
   - in the CC paragraph, the simplex row and the dropped bounds q ≤ 1;
   - the derivation r ≥ 1 > 0, matching the code fix 1.2.

   Also correct l.1509-1512 ("uses only `lem:locality` and `lem:rip`"): the proof also uses `lem:reduce`, `prop:ordered`, the semi-graphoid axioms and Lauritzen's F⇒G.
7. **Main-text omissions.**
   - l.295-296: state the conditions under which one row per node implies all assertions.
   - l.316: "all four are checked" — after fix 1.1, H1 is checked; say how.
   - l.299: "G* has the clusters" → "T has".
   - l.225: "parents {A,D}" → "boundary".
   - l.860: remove the jargon "credal-network directory".
8. **Size theorem.**
   - The rows per assertion are ≤ 2^{|S|}(2^{|Y|}−1) ≤ 2^w, not 2^{w+1}.
   - The number of imposed assertions is ≤ |Λ| for min-fill and ≤ n for the ordered tree, so the total is O(n 2^w) rows.
   - Under evidence, add one variable (r) and one row.
   - Correct the separator-count justification.
   - Recount the example if subsumed clusters are contracted (1.7).
9. **Disclose what the implementation actually does** (`sec:algo-host`, `sec:algo-solve`).
   - **Widening:** adds whole node sets, all pending scopes at once; stops at the first round over the budget; at most |Λ|+1 rounds. The budget b is in atoms (`max_cluster_atoms`, default 16).
   - **Relaxation:** when no consequence is hostable, the assertion is dropped. This is sound but outer; say so.
   - **Tree choice:** the ordered tree is skipped when over budget, and ties go to the ordered tree.
   - **Augmented route:** uses the min-fill triage only, not "the same triage" (l.1083).
   - **Evidence normalization row:** placed on `jt.host(e)`. This is equivalent to C_a by RIP; say so.
   - **"One system serves all marginals"** (l.1043-1046) holds only for atoms that satisfy H4.
   - **ipopt:** restarts only when the result is suspicious; feasible and acceptable terminations are accepted.
   - **Hand check** (l.1458-1464): add the attainability direction.

---

## Part 4 — Compilation and family-realizability (Sec. 4.1–4.2, App. D–E)

1. **Define the local set K_{v,π} before `prop:incl`.** It is currently defined inside the proof of `lem:noGapA`. The definition should state:
   - the Charnes–Cooper LP with ρ(π) > 0;
   - the in-scope sentences only;
   - **no LMC equalities** (none at all, not just the cross-family ones; fix l.409 and l.821);
   - "infeasible ⇒ simplex" as a sound **convention** (the projection itself is empty);
   - an interval for a single-atom child, and the exact row polytope for a compound child, with the box fallback disclosed.

   Also define K^true_v = {P_p(ch_v|π) : p ∈ D_LCN, p(π) > 0}.
2. **State the scope of what the paper covers:** `method="linear"`, `merge_budget=1`.
   - `linear-tight` with ipopt can give **inner** intervals and break `prop:incl`. Say so, or remove the option.
   - The polytope hull is inner by about 1e-7 (tolerance 1e-9 plus HiGHS tolerances), so the polytopes are "exact up to tolerance".
3. **prop:incl.**
   - Rows with P_p(π) > 0 lie in K because p's marginal is a feasible point. Rows with P_p(π) = 0 can be chosen arbitrarily, and K is non-empty. Say this.
   - Make "for every query" (l.2024) precise using Part 2.3.
4. **lem:noGapA.**
   - Local feasibility is used to rule out the simplex convention, not to guarantee non-emptiness. Say so.
   - Set Π_s = ∅ for constant Type-1 sentences.
   - Optional remark: local feasibility is stronger than necessary.
5. **prop:suff and prop:suff-app.**
   - Add the "exact local sets" hypothesis to the main-text statement (the appendix has it).
   - Use the conditional query in inf/sup form.
   - Add the marginalization lemma q(c,π) = q(π) P_{v,π}(c) for products along the DAG (marginalize onto the ancestral set).
   - Mention the vacuous case |Π_ψ| = 0.
   - Say that consistency is used only for constant Type-1 sentences.
   - Cite the Part 2.5 lemma for the DAG case.
6. **Counterexamples (`sec:fr-counter`).**
   - l.2154 says every counterexample is fully realizable; `fr_counter_factorizing` is not (see 1.8).
   - `fr_counter_lmc_path` gives [0,1] on every atom in both D_LCN and CVE. It shows only a **set-level** gap; the gap in bounds needs the `lmc_evidence` variant. Fix l.2146.
   - l.2174 "ranges freely over the box": the compiler now stores a 56-vertex polytope.
   - l.2172 uses "true local set" without defining it.
   - Cite one counterexample for the across-rows failure of (iii) (l.439).
   - Verified numbers to keep: `box` [0.03,0.82]→[0.32,0.58]; `lmc_evidence` [0.5196,0.9182] vs [0.333,1]; `factorizing` 0 vs 0.12; `lf`.
7. **lem:invariance.**
   - The definition's "same conditional table" should read "agree on every π that has positive probability under both".
   - Add the clause "P(ψ)>0 and P′(ψ)>0" to the main-text definition, as in the appendix.
   - Rename "u uniform", which clashes with the upper bound u.
   - The code (`realizable_in_family`) implements the lemma exactly; keep the sentence that says so.
8. **fr5 example.**
   - The witness at l.1997 is wrong. Use P(x2 | x3∧x4) = 0, which is a free row and makes ψ_{s3} unreachable.
   - Typo at l.1939 ("and example").
9. **Product-Markov test disclosure** (l.2148-2150): 3 random trials, tolerance 1e-9, returns None above 14 atoms, and short-circuits for DAGs (justified by the lemma in Part 2.5).
10. **Minor.**
    - l.373: D_SE → P_SE.
    - l.804: "moralizes" → "contracts".
    - l.815: "interval local set" → "interval or polytope".

---

## Part 5 — CVE, IVBP, CDVE (Sec. 4.3–4.5, App. F–H)

1. **thm:cve and thm:ibp.**
   - Restate them for P̄(y) > 0 as inf/sup over the members with P(y) > 0.
   - Move remarks 2592-2595 and 2924-2929 into the proofs, with the explicit vertex argument from Part 2.4(b).
   - Add the caveat "exact for the compiled local sets; outer for D_SE if the box fallback was used".
   - Add "exact up to the floating-point precision of the enumerated vertices", and list the thresholds:
     - ray dedup at 15 decimals;
     - SVD rank cut 1e-12·scale·√m;
     - qhull up to dimension 7 (dimensions 6–7 only with ≤ 5000 rows);
     - Clarkson's LP test only up to 1000 rows, and keep-all above that (sound);
     - LRS vertex accuracy about 4e-7;
     - IVBP merge at 12 decimals, and `_MAX_CORNERS = 200000` (raises).
2. **thm:cve-cplx.**
   - Make it consistent with fix 1.4.
   - Say that it is per query variable; all marginals cost a factor m more.
   - Fix the class-count formula to (2 − ⌊log_α t⌋)^{2|Ω|} with t taken after scaling to mass 1 (l.2504).
3. **Pruning.**
   - The code applies the Pareto prune only to the final scalar pairs (`cve.py:284`). Rewrite l.449-452 and `def:prune` to say this, and drop the claim that "dominance is preserved by products and sums".
   - Simplify `lem:cover`(1): dominance is a strict partial order on a finite set, so every dominated pair has a maximal dominator that survives.
   - Handle a dominator whose ratio is undefined (use the dominator from the other order).
   - Alternatively, drop Pareto pruning from the algorithm, since it saves nothing over a plain min/max.
4. **lem:cover(4).** Say that u∘q_τ is marginalized onto φ's scope and that each potential is consumed once, which is why the factors multiply to α^{2N}.
5. **lem:cvecond.** Note that the implementation rescales to max 1, so its functions are not literally conditional tables; the ratio is invariant to the rescaling.
6. **IVBP proofs.**
   - State `lem:proj` for arbitrary (infinite) sets.
   - Add the sandwich M_e ⊆ S_e ⊆ conv M_e (every computed vertex, including every box corner, is attained) to the induction.
   - In `thm:ibp-cplx`:
     - give the cost of the π-message and of the λ-message separately (C_V differs between them);
     - "D+1 sweeps reach the fixed point, plus one to confirm it"; the code caps at 2n+2;
     - state the loopy defaults τ = 1e-6 and T = 100;
     - say explicitly that C_V and M_V can be exponential, which is why exactness is consistent with Mauá et al. 2014;
     - say "coincides with 2U" for the output only, not step by step.
   - Fix the P̄(y)=0 sentence (l.2928-2931) and l.2730-2731 to match fix 1.3.
7. **CDVE.**
   - Rename "centroid" to "vertex average".
   - Random starts with P(y) = 0 are discarded; [0,1] is returned only when P̄(y) = 0.
   - State the defaults: R = 8, T = 50, Dinkelbach cap 100, strict-improvement tolerance 1e-12.
   - `thm:cdve-cplx`: add the factor 2 for the two senses and the per-start posterior VE. With vn and vd precomputed, each Dinkelbach iteration costs O(E_V).
8. **Examples.**
   - The alarm, chain and CDVE numbers are verified.
   - Alarm step 3: "sees φ_B alone" holds only up to rounding (two near-duplicates are kept).
   - Alarm with C=1: replace the non-sequitur "P̲(C=1)=0 ⇒ no information" by the actual cause: the vacuous A rows given B∨E, plus C-D unconstrained given A=0.
9. **Leftovers.**
   - l.2249 still says "three algorithms… sets of functions".
   - The CN background uses m variables but indexes D_1..D_n (l.2254-2256).
   - Sec. 5 and App. I still mention CVE_cm and CVE_cp, which were removed; these will be rewritten with the rerun.

---

## Part 6 — Benchmarks (`*_fr`, `nr_*`) — pending the benchmark audit

Known before that audit reports:
- **The `nr_*` benchmarks (`nr_tree`, `nr_dag`, `nr_dag_ablation`) are not described anywhere in the paper.** Add a subsection, with a definition, to the benchmark paragraph and the appendix. It should cover:
  - the kinds nr1, nr2 and nr3, matched to the two halves of `def:realizable`;
  - the round-robin assignment of kinds;
  - nested m;
  - bounds [v−δ, v+δ] around the base witness value (and any clipping);
  - why consistency is preserved by construction, and why the structure and the LMC are unchanged;
  - that `nr_tree` gets no nr2;
  - the CJT gate.
- **`sec-cjt-gate`** (l.3077-3097) claims that every instance carries a header recording the CJT gate (structural and certified), and that "condition (i) holds on all 556 instances". This has to be verified against the shipped headers. Otherwise regenerate with `--cjt-exact certified`, or weaken the claim.
- **`ktree_large_fr`** contains extra `.cn` files (105 files in all). Check that they are format version 3, or delete them so that the `compile` stage rebuilds them.
- **CJT reference on the `nr` sets:** the gate checks marginals without evidence. After fix 1.1, the evidence runs (augmented route) need `cjt_exact` in the reference selection (see 1.1).

Checks the benchmark audit performs (results to be merged here):
- inventory;
- headers;
- the topology claims;
- both halves of realizability, for every sentence;
- bounds, and ≤ 3 atoms per formula;
- consistency (product witness);
- the structural CJT gate;
- for the `nr` sets: the kinds, unchanged structure and LMC, nesting, m=0 equals the base, δ, and the round robin;
- a review of the generator code, and the reproducing commands for each set.

---

## Part 7 — Proof presentation standard (applies to every result)

For each lemma, proposition and theorem, the revised text will have:
1. **Statement:** every hypothesis named, with the same wording in the main text and the appendix.
2. **A one-paragraph proof idea** in the main text.
3. **The appendix proof**, written as numbered steps. Each step cites the definition or lemma it uses; there are no steps of the form "it is easy to see".
4. **The degenerate cases handled explicitly:**
   - zero-probability parent configurations, separators and evidence;
   - empty sets: separators, Π_ψ;
   - infeasible local programs;
   - an empty D_LCN.
5. **A sentence tying it to the code** where an engine implements it (function name), with any tolerance or fallback disclosed.

Add a dependency table of the results to App. A (which result uses which), so the reader can verify the chain:

```
lem:hull ─┬─ prop:incl ── thm:cve / thm:ibp (outer for D_LCN)
          └─ prop:suff ── thm:cve / thm:ibp (exact for D_LCN)
DAG lemma (new) ── prop:incl, prop:suff, prop:ordered
Helly (new) ── lem:rip, lem:reduce, prop:ordered ── thm:d5
```

---

## Order of work

1. Part 1 code fixes and their tests. Rerun only the regression examples cited in the paper (alarm, asia, `cjt_zero_evidence`, `cjt_dropped_lmc`, `fr_counter_*`, `forest2`, the IVBP and CVE examples). Update every number in the text that changes.
2. Merge the benchmark audit into Part 6, and regenerate or fix any benchmark sets it flags. Only then rerun the experiments.
3. Part 2 (foundations and notation). Every later section depends on it.
4. Part 3 (CJT), Part 4 (compilation), then Part 5 (CVE/IVBP/CDVE), using the Part 7 standard.
5. Consistency pass:
   - grep every claim of "exact", "outer", "inner" and "certified" in the main text, and check each against its theorem's hypotheses;
   - grep every default value (R, T, τ, budgets, tolerances), and check each against the code;
   - update CLAUDE.md where the code changed.
6. Final read of the main text against the 9-page limit. Move material to the appendix if the new definitions and lemmas push it over.

---

## Status (2026-11-12)

- **Part 1 (code): done.** Every item except the optional 1.7 cluster contraction, and 1.8, which was handled in the text instead (the example file is unchanged). Regression files added: `examples/cjt_forest_evidence.lcn`, `examples/ivbp_zero_evidence.lcn`. New tests: `tests/test_chain_graph_check.py`, `TestAugmentedTreeRIP`, `test_impossible_evidence_is_vacuous_everywhere`, `test_reference_requires_structural_exactness`. Full suite green.
  - **Deviation:** a directed edge *parallel* to an undirected edge is absorbed, not rejected; it occurs in `tests/test_structured_consistency.py`.
  - **Deviation:** `locally_feasible` warns and assumes the row is feasible instead of raising.
- **Parts 2–5 and 7 (paper): done.** Merged from four parallel revisions. The paper builds with no undefined references (74 pages).
  - New labels: `def:ci`, `def:cond`, `lem:semigraphoid`, `lem:dag`, `rem:posclosure`, `lem:helly`, `def:localset`, `lem:marg`, `lem:vertexprod`, `lem:cve-induction`, `tab:deps`, `tab:notation`.
- **Part 6 (benchmarks): deferred** by request. That includes the `nr_*` description, the `sec-cjt-gate` header claim (it should also mention the explicit RIP check now done in `generator.py`), and the `ktree_large_fr` `.cn` files.
- **Open items:**
  - Page budget: Experiments now starts on p.10 and Conclusion on p.12. Compress when Sec. 5 is rewritten after the rerun.
  - Confirm the Lauritzen (1996) numbers cited: Prop. 3.8 and Thm. 3.27.
  - Sec. 5 and App. I still mention CVE_cm; to be rewritten with the rerun.

## Part 6 — Benchmark audit results (2026-11-12)

**Solves.** No SCIP solves were run in this pass. Inputs:
- the stopped audit's per-file checks (`/tmp/lcn_audit/{fr,nr,rr,aw,certall}.jsonl`);
- a re-run of the structural CJT gate with the new running-intersection (H1) check, done without solving (`/tmp/bench_audit/gate.jsonl`);
- verification of the regenerated polytree sets (`/tmp/bench_audit/verify_poly.json`).

**Results that hold for all 740 files.**
- **Chain graphs:** every file is a genuine chain graph.
- **CJT structural gate:** passes everywhere, H1 included.
- **Realizability:** every sentence is R1/R2.
- **Bounds:** within [0,1]. The main-sentence widths match ε = 0.3, clipped at 0.01 and 1.
- **Consistency:** shown analytically, with no solver. The fully independent product of per-atom interval midpoints satisfies every sentence, and the LMC trivially. The random product-witness search had failed on 14 instances (11 `ktree_large`, 1 `ktree_small`, 2 `dag_large`); that was the search giving up.
- **Reproducibility:** every `_fr` set regenerates byte-for-byte from seed 42 with `--family-realizable --epsilon 0.3 --num-extras 2`.
- **Small/large overlap:** the n=10 instances of the small and large sets are distinct.
- **`nr_*` sets:**
  - every added sentence is non-realizable, of its claimed kind, in round-robin order;
  - `nr_tree` has no nr2;
  - the structure graph and the LMC are unchanged from the base;
  - m=0 equals the `_fr` base exactly, and the larger m are nested extensions.

**Problems found.**
1. **Polytree generator bug (fixed in code).** `_graph_polytree` forbade only two *directed* paths between a pair of nodes, which admits undirected cycles. 23/60 `polytree_small_fr` and 23/60 `polytree_large_fr` instances are not polytrees.
   - The generator now uses a union-find over the skeleton; regression test `tests/test_generator_polytree.py`.
   - Regenerated sets are in `/tmp/bench_audit/regen/polytree_{small,large}_fr`. All 120 have a forest skeleton, are fully realizable, DAG-structured, locally feasible and consistent, pass the gate, and have widths 2–3.
   - **Pending:** replace the shipped sets (and rerun the six tests that read them).
2. **Header claim is false for the `_fr` sets.** The `_fr` files have no header at all. The `nr_*` headers record "structurally exact … solves not certified at generation time". `sec-cjt-gate` must be rewritten.
3. **Certified claim (ii) is false.** In the 293 instances already run with SCIP at 60 s per endpoint, 54 have at least one uncertified endpoint (e.g. `dag_small` 4/60, `dag_large` 12/20 run, `nr_dag_ablation` 20/60). State certification as an output of the reference run, using the per-endpoint `certified ∧ structural` filter now in `analyze_results.py`.
4. **"Formulas span at most three atoms" is false for k-trees.** With k=3, every Type-2 conditions on the full 3-parent conjunction, so its sentences have 4 atoms. More generally, family-realizable sentences have k+1 atoms.
5. **Extras.** The two extra marginals restate root marginals, and often both land on the same root atom: always on trees (one root); 32–60 instances per set elsewhere. Describe them as "two extra interval bounds on root marginals".
6. **The `nr` sets are not described in the paper.** Facts to state:
   - bases: `dag_*_fr` and `tree_*_fr`, n ∈ {10, 20}, 10 instances each;
   - m ∈ {0, 1, 2, 4, 8} with δ = 0.1, plus an ablation with one kind per file, m = 4, δ ∈ {0.05, 0.3};
   - bounds [v−δ, v+δ] around the independent-product witness value, clipped to [0,1] in 98 of 260 files;
   - all instances are structurally CJT-exact, but none is certified at generation time.
7. **`ktree_large_fr` ships 45 compiled `.cn` caches** (v3, linear, ipopt), unlike every other set. They're valid, since k-trees have single-atom children only. Either drop them from git, or ship caches for every set via the compile stage.

**Done (2026-11-12):**
- Item 1: shipped `polytree_{small,large}_fr` replaced by the regenerated sets (112 files changed, 8 identical). The 6 test files that read them pass (153 tests).
- Items 2–5: paper fixed:
  - main-text Benchmarks paragraph rewritten;
  - `sec-cjt-gate` rewritten (structural gate verified on all 480 + 260 + 26 instances; certification is an output of the reference run, scored per endpoint);
  - new appendix subsection `sec-nr-bench` describing the almost-realizable sets.
- Item 7: the 45 `.cn` caches in `ktree_large_fr` were deleted (working-tree deletions, not staged). The 16 `.cn`/`.vtx` caches in `benchmarks/real` were deleted too, and `.gitignore` now ignores `benchmarks/**/*.cn` and `*.vtx`.
