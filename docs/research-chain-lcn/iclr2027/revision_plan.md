# Revision plan — ICLR 2027 submission 39159

Reviews: A2AT (1, conf 4), c7e5 (2, conf 3), JNVR (2, conf 3). See `reviews.md`.

## Diagnosis of the soundness claims

| # | Claim | Finding | Verdict |
|---|---|---|---|
| S1 | Convex hull in Eq. (4) breaks Prop. 2 `D_SE = D_LCN` (A2AT, c7e5 x2) | Main text defines `D_SE` as a convex hull; appendix `eq:se-app` drops the hull. `D_LCN` is nonconvex (bilinear LMC equalities), so it cannot equal a hull. | Real; repairable (bounds coincide even though sets do not). |
| S2 | CVE dominance prune does not preserve normalized-posterior extrema (A2AT) | `cve.py` prunes dominated-from-above functions, then takes both min and max of the normalized survivors. Unsound with evidence. Without evidence every intermediate function is plausibly a conditional table (sums to 1 over its eliminated children) so domination implies equality — to verify. | Real with evidence; probably fine without. |
| S3 | IBP exactness on trees unproved; c7e5 counterexample gives invalid `[0.9,0.5]` | Paper equations differ from code: code sends vacuous upward factor->parent messages (`ibp.py:207-231`), so no upward evidence updates; `np.maximum(lo,hi)` masks empty intersections. Appendix proof cites Pearl (not applicable to interval messages) and says "CVE". Ternary credal polytrees are NP-hard (Mauá et al. 2014), so multi-valued exactness cannot hold. | Real; defensible only for binary, no-evidence. |
| S4 | (not raised) Thm 2/4 hypotheses | Prop. 2 needs (A) exact local sets + (B) realizability; Thms 2/4 assume only (B). | Gap. |
| S5 | CJT certified exact? (all three) | Main text: "solved globally"; appendix: SCIP gap 0; experiments: `ipopt` backend (`cjt_l`) as reference — which `rem:ipoptinner` says is an inner bound. | Real inconsistency; all MAEs vs an uncertified reference. |
| S6 | (not raised) positivity | Inclusion proof applies chain-graph factorization to every `p in D_LCN`; line 225 says local/global Markov equivalence needs positivity; cleared form admits zeros. | Gap to close or state. |

## Phase 0 — resolve the open questions (before rewriting)
1. **S2 check**: CVE with vs without pruning, with and without evidence; prove the no-evidence conditional-table lemma or find a counterexample.
2. **S3 check**: encode c7e5's tree; run IBP & CVE. Run IBP on `tree_*_fr` / `polytree_*_fr` with random evidence vs certified reference.
3. **S5 check**: rerun CJT references with `cjt_g` (SCIP), record per-solve gaps; compare to the ipopt references.
4. Confirm the paper's experiment logs used no evidence.

## Phase 1 — repair the theory
- **S1**: define product set `P_SE`, `D_SE = CH(P_SE)`; restate Prop. 2 as `P_SE = D_LCN`; lemma: linear / linear-fractional (positive denominator) objectives have equal extrema over a set and its hull ⇒ bounds over `D_SE` = bounds over `D_LCN`.
- **S4**: prove (B)+consistency ⇒ (A), or add (A) to Thms 2/4.
- **S6**: components collapse to compound nodes, factorization is a DAG over components; DAG local Markov ⇒ factorization without positivity. Prove, or assume positivity.
- **S2**: restrict Thm 2 to evidence-free marginal queries, prove via the Phase-0 lemma. For evidence: separate min/max dominance + generalized Bayes rule / bisection (Mauá & Cozman), or declare out of scope. Full proof of pruning correctness.
- **S3**: rewrite Eqs. (5)–(7) to the actual algorithm (downward-only, explicit empty-intersection rule). Restrict Thm 4 to binary, singly-connected, family-realizable, no evidence; real proof (polytree parents independent, objective multilinear in parent marginals, binary credal sets are intervals ⇒ corner extrema; essentially evidence-free 2U). State outer/unreliable under evidence or compound nodes; cite NP-hardness. Optional: 2U-style upward messages (Fagiuoli & Zaffalon 1998).
- **Guarantees**: CDVE is inner w.r.t. `D_SE` only; CVE_cm (mean representative) has no guarantee — call it a heuristic (optional: pointwise-max representative gives an outer upper bound on the unnormalized table).
- **CJT**: numbered complexity theorem separating formulation size `O(n 2^{w+1})` from solve time; one consistent solver statement across main text and appendix.

## Phase 2 — experiments
- **E1** certified reference: SCIP for every reference; per-set certified fraction; exclude or separately report uncertified; small ipopt-vs-SCIP agreement table.
- **E2** hypothesis prevalence: per set, how often H2/H3 hold and how often CJT falls back to direct (incl. Supreme Court).
- **E3** controlled relaxation (c7e5): vary fraction/type of non-realizable sentences (non-root Type-1, multi-config psi) and interval width; plain CVE (exact on `D_SE`) vs certified CJT isolates relaxation error; IBP/CDVE show additional error.
- **E4** treewidth scaling (JNVR): `ktree`, k in {2..8}, fixed n; CJT/SCIP time and program size vs w.
- **E5** small evidence experiment, only if Phase 1 adds evidence support.

## Phase 3 — presentation
- Compact algorithm boxes for CVE, IBP, CJT; lead-in sentences and proof sketches for every theorem.
- Make room within 9 pages: compress Sec. 2.1 primal/structure text, shrink Fig. 2, move CDVE details to appendix, shorten Table 1.
- Reframe contribution: CJT + realizability theory as the core; CVE/IBP/CDVE as adaptations.
- Rename IBP (clash with Iterative BP), e.g. "IntBP".
- Related work: arithmetic circuits / determinism / credal SPNs (Darwiche 2009; Mauá et al. 2017); CJT is treewidth-bound, does not exploit determinism.
- Notation: one meaning for **X**, G, pa(.); define P, l, u, indep, atoms(s), C, "projection"; formula-node notation `v_psi`; period in Eq. (4).
- Citations: Zhang & Poole 1996 (VE); 1980s RIP refs (Beeri et al.; Lauritzen & Spiegelhalter); consistent bib formatting.
- Fig. 1 with a longer cycle; rewrite Sec. 5 prose; specific abstract; CredalJT/CJT and CVE_cp leftovers.

## Order and risk
Phase 0 first (outcomes fix theorem statements). S1, S2+S4, S3 and E1 decide A2AT; E3 is c7e5's main ask; algorithm boxes/proofs are JNVR's. Biggest risk: E1 — if SCIP fails to certify many references, the experiments must be reframed around certified subsets.

## Phase 0 results (2026-11-09)
Scripts and inputs in `phase0/` (`run_case.py`, `h3_triage.py`, `prune_check.py`, `c7e5_tree.lcn`).

**P0.4 — evidence.** All 4,235 records under `results/` have `evidence = {}`. Every reported
number is a no-evidence marginal, so restricting the CVE/IBP theorems to evidence-free
queries invalidates none of the experiments.

**P0.2 — IBP (c7e5's tree).** `A -> B`, P(A)=.5, P(B|A)=.9, P(B|!A)=.1, evidence B=1. Truth
P(A|B)=0.9. CVE, CJT and direct global: [0.9, 0.9]. **IBP: [0.5, 0.5]** — wrong, not merely
loose (excludes the truth); it also reports the evidence atom B as [0.5, 0.5]. Without evidence
all engines agree. Confirms S3: the evidence semantics are broken (no upward messages).
Theorem 4 must be restricted to no evidence.

**P0.1 — CVE pruning.** 10 tree + 10 polytree n=5 instances, pruned vs unpruned (`Potential.prune`
patched to identity; unpruned = exact strong extension by definition).
- No evidence: **0/20** differ (max 1e-7). Consistent with the conditional-table lemma.
- With evidence: **28/40** differ, by up to **0.90**; pruned intervals are *narrower* than
  exact — unsound. Confirms A2AT's S2 objection exactly. Fix = restrict Thm 2 to no
  evidence (proof via the lemma) and/or implement generalized-Bayes / separate min-max pruning.

**P0.3 — CJT reference (new, most serious).**
- *SCIP certification is not recorded.* `junction_nlp._solve_sense` (scip branch) calls
  `_read_gap` and discards it, and never checks the termination condition: a time-limited
  solve returns its incumbent and is logged `status: ok`. "ok" != certified.
- *ipopt vs SCIP agree* wherever both finished: ~11k endpoints, max |diff| 2.75e-5. So the
  ipopt-vs-SCIP question (S5) is minor.
- *But H3 fails on most instances*, so CJT itself is only an outer bound there. Triage of LMC
  assertions (`h3_triage.py`):

  | set | instances with a dropped LMC | with a lost hostable sub-independence |
  |---|---|---|
  | tree_small_fr | 40/60 | 0 |
  | polytree_small_fr | 48/60 | 21 |
  | ktree_small_fr | 20/60 | 0 |
  | dag_small_fr | 56/60 | 42 |
  | real | 9/10 | 7 |
  | junkyu | 7/16 | 7 |

- *Dropped LMCs bind.* With no evidence, CVE (exact on `D_SE` ⊇ `D_LCN`) should contain the
  reference, yet it is strictly *inside* CJT on 17 dag_small and 7 polytree_small endpoints (up to
  0.128). Worked case `dag_fr_tw3_n6_7`: roots x2, x4 ⇒ LMC `x2 ⟂ {x0,x1,x4}` (4 atoms > max cluster
  3, not separator-entailed) is dropped, losing `x2 ⟂ x4`. Hand calculation: with independence
  min P(x3) = .6737·.3494·.0965 = **0.0227** (= CVE); without it .0965·(.6737+.3494−1) = **0.0022**
  (= CJT). The code comment justifying the drop ("strong extension satisfies every LMC") does
  not apply to CJT, which does not optimize over the strong extension.
- Direct 2^n global solver on that n=6 instance did not finish in 600 s.

**Consequences for the plan.**
1. The paper's reference is an *outer* bound on most instances; MAEs against it are not errors
   against the truth. On dag/polytree the reported "error" of CVE/IBP partly measures CJT's own
   looseness.
2. Fix CJT before rerunning anything (new item, highest priority):
   a. host decomposed sub-assertions: for an unhostable `X ⟂ Y | Z`, impose every hostable
      `X ⟂ Y' | Z` with `Y' ⊆ Y` (sound — decomposition) — recovers the lost `x2 ⟂ x4`;
   b. or augment the elimination with LMC scopes when the width budget allows (exact, H3 holds);
   c. record per-solve SCIP gap + termination; mark uncertified results.
   Report H3 status per instance; claim exactness only where H3 holds.
3. E1 becomes: rerun references with the fixed CJT + certification, cross-check against direct
   global on small n.
4. CVE theorem: no-evidence version is empirically sound; IBP theorem: no-evidence only, plus
   an explicit statement that IBP is not valid under evidence.

## CJT fix + CJT-exact generator (2026-11-09)

**Code (`lcn/inference/marginal/cn/junction_nlp.py`).**
- *Unsound entailment test fixed.* `_rip_implied` accepted `X ⟂ Y | Z` whenever a tree separator
  `S ⊇ Z` split X from Y; that only enforces `X ⟂ Y | S`. Replaced by `separates(X, Y, Z)` —
  graph separation in G* by Z itself (the paper's own Definition). Affected 15 assertions / 14
  instances (polytree_small 6, dag_small 5, real 3).
- *Exact reduction* `reduce_assertion`: `X ⟂ Y | Z` ⇔ `X ⟂ Y∩K | Z` with K = reachable(X, G*∖Z)
  (global Markov + weak union + contraction; no positivity). Recovers e.g. `x2 ⟂ x4` from the
  dropped 4-atom assertion.
- *Min-fill triage* `_triage_lmc`: reduce → host → widen by the reduced scope to a fixed point
  within budget → relax to hostable pairwise consequences (sound, outer, H3 fails). No assertion
  is silently dropped any more.
- *Ordered-Markov tree* (`_JTClusters(order="topological")`, `_triage_ordered`): eliminate the
  node DAG in reverse topological order; one row `x ⟂ N_x∖pa(x) | pa(x)` per node implies the DAG
  factorization, hence every LMC assertion — H3 by construction, no widening. Each row is checked
  to be entailed by a single LCN assertion; every LCN assertion is checked to be a d-separation.
  The narrowest tree on which H3 holds is used.
- *Certification.* SCIP endpoints are certified iff proven optimal or |primal−dual| ≤ 1e-4;
  otherwise the **dual bound** is reported (sound). The old code returned the incumbent of an
  unfinished solve (unsound inner) and logged "ok". New attributes `h3`, `d5_exact`, `certified`,
  `exact`, `solve_status`; runner logs `cjt_h3`, `cjt_certified`, `cjt_exact`, `cjt_solves`.
- Symmetric duplicate assertions are imposed once.

**Effect.** H3 now holds on **all 556** benchmark instances (was failing on most), at mean width
growth ≤ 0.92 atoms per set (max +6 on one dag_large instance). Regression instance
`examples/cjt_dropped_lmc.lcn`: `min P(x3)` = 0.022727 (exact, = CVE) vs 0.0022 before; certified.
Known limitation: LMC assertions that split a compound node (e.g. `smokers`) disable the ordered
tree; min-fill widening then reaches the full joint (exact but no cheaper).

**Generator (`lcn/benchmarks/generator.py`).** `cjt_exact="structural"|"certified"` gate
(`cjt_exactness_report`), CLI `--cjt-exact/--cjt-max-width/--cjt-time-limit` on
`gen_benchmarks.py` and `gen_{trees,polytrees,ktrees,dags}.py`; report written as `.lcn` header.
Fresh candidates: 40/40 accepted structurally (dag-fr n=20/50/100, polytree-fr n=20), width ≤ 10.

**Tests.** `tests/test_credal_jt.py` +6 (separation, reduction, H3, regression value, ordered tree,
dual-bound fallback); new `tests/test_generator_cjt_exact.py` (10). Full suite green.

**Paper (appendix).** Rewrote Alg. `alg:cjt`, §`sec:algo-jt`, §`sec:algo-host` (two triages),
new Remark `rem:triagefix` (the correction, with counts) and Example `ex:dropped`, Remark
`rem:dual` (certification), Lemma `lem:reduce`, Proposition `prop:ordered` (with proofs),
restated H3 in `thm:d5` and main-text Thm. 6, updated the proof and Remark `rem:d5reading`(c);
new §`sec-cjt-gate`. Sec. 5 Benchmarks now states the gate.

**Still to do.**
1. Certified gate over the shipped sets (running: `phase0/certify_small.log`, then the large sets);
   if any instance fails certification, regenerate that set with `--cjt-exact certified` or soften
   the Sec. 5 sentence. **The Sec. 5 claim is not yet backed for the shipped sets.**
2. E1: rerun every reference with the fixed `cjt_g` and recompute all MAEs, Table 1, figures,
   and the appendix tables — every number in Sec. 5 was computed against the old reference.
3. `exact.py` `_OPTIMAL` counts `TerminationCondition.feasible` as confirmed — same certification
   bug in the direct engine (affects `exact_g` logs); not yet fixed.

## A1–A7 theory fixes (2026-11-09)

- **A1 (hull).** Main text: new Eq. `eq:pse` (product set P_SE), `eq:se` = CH(P_SE); Prop. 1 is now
  `D_LCN ⊆ P_SE ⊆ D_SE`; Prop. 2 is now `P_SE = D_LCN` + equal bounds. Appendix: eq:se-app renamed to
  P_SE, new Lemma `lem:hull` (bounds of linear-fractional queries invariant under convex hull), proofs
  restated for P_SE.
- **A2 (CVE).** Thm 2 restricted to consistent, factorizing, family-realizable LCNs and marginal queries
  WITHOUT evidence. New Lemma `lem:cvecond`: evidence-free potentials are conditional tables over a fixed
  split, so pruning removes only duplicates (any elimination order). Full proof of Thm 2. Phase-0 evidence
  result (28/40 cases differ, up to 0.90) cited as the reason for the restriction; generalized Bayes rule
  named as the fix not implemented. CVE_cm declared a heuristic.
- **A3 (IBP).** Eqs. 5–7 rewritten to the implementation: vacuous factor-to-parent messages, endpoint
  distributions as corners, empty-intersection rule, no evidence. Thm 4 restricted to binary polytrees,
  family-realizable, no evidence; real proof (independent parents in a polytree, multilinear child
  marginal, corner extrema, induction). Necessity of each hypothesis stated (c7e5 counterexample;
  NP-hardness for ternary, `maua2014jair` added to ref.bib). Appendix: algorithm box made evidence-free,
  §Evidence replaced by a scope statement, the evidence worked example removed.
  Empirical check: no-evidence IBP = CVE to 1e-7 on all 65 tree/polytree_small instances where both ran.
- **A4 (Gap A).** New Lemma `lem:noGapA`: consistency + full realizability ⇒ K_{v,π} = K^true_{v,π}. So
  hypothesis (A) is dropped from Prop. 2 / Thms 2, 4.
- **A5 (positivity).** "Factorizing" LCNs defined: DAG-structured (no positivity needed) or strictly
  positive D_LCN. Counterexample shape stated (atom-wise LMC of a 2-atom component needs intersection).
  Props 1, 2 and Thms 2, 4 assume factorizing; Thm 4's binary-polytree case is automatically factorizing.
  The CJT ordered tree is already sound here (`_lmc_entails` rejects compound-node rows).
- **A6 (guarantees).** CDVE: inner w.r.t. D_SE; w.r.t. D_LCN only when bounds coincide (main + appendix).
  CVE_cm: heuristic. Intro sentence before §3.4 qualified.
- **A7 (CJT solver).** Main text now states SCIP + dual-bound fallback; consistent with the appendix.
- Sec. 5 sentence on which engines are exact updated to the new theorem statements.

**Open after A1–A7.**
1. **Main text is now 10 pages** (conclusion starts on p.10). The Sec. 5 prose will be rewritten with the
   E1 numbers — cut there first; other candidates: Sec. 2.1 primal-graph text, Fig. 2.
2. **IBP worked example (appendix, "Query: P(B=1), No Evidence") is stale**: it uses box corners
   q=[1,1] (non-distributions, normalizer 4.0) — the old algorithm the code comment calls a bug.
   Regenerate its numbers from the current `ibp.py` trace.
3. Lemma `lem:cvecond` empirical audit (pruning removes only duplicates without evidence) running:
   `phase0/prune_audit.py`.
4. Sec. 5 line "Proposition 2 does not hold" for realistic instances: still correct (they are not
   realizable), unchanged.

**Lemma `lem:cvecond` audit (2026-11-10, `phase0/audit_*.log`).** 20 n=5 instances (10 tree, 10 polytree),
topological order: without evidence, 400 prune calls removed **0** non-duplicate functions; with one evidence
atom, 320 prune calls removed **25,833** non-duplicates. Confirms the lemma and the evidence restriction of
Thm 2. Not run: min-fill order, `real/` (the combined batch died without output; cause not investigated).

## CJT section review (2026-11-10)
Removed all notes about earlier versions/defects (Remark `rem:triagefix`; the "tree separator ⊋ S"
note in Def. `def:d5`; the "usual statement contains two defects" lead-in to Lemma `lem:rip`; the
pointer to earlier failures in Remark `rem:d5reading`; the "earlier version… removed" note in the IBP
scope subsection). Example `ex:dropped` kept as a neutral illustration of the reduction.
Proof fixes: Lemma `lem:rip` states that leaf removal preserves RIP; Def. `def:d5` describes T as the
bucket-elimination tree and G* precisely; Prop. `prop:ordered` now states its applicability condition
(every LMC assertion a node-level d-separation, no split compound node) and justifies the perfect
elimination order incl. pruned leaves; Thm. `thm:d5` states conditional queries exactly over
D_LCN^ε = {P(e) ≥ ε} with its own proof paragraph (main-text Thm. 1 matches); Remark `rem:d5reading`(b)
no longer claims unconditional soundness (it fails under evidence when min P(e) < ε); eq. width: w is the
width of T, not the treewidth; "both checks passed on every benchmark" corrected.
NOTE for the experiments rewrite: the 'Coverage and exclusions' appendix section still describes the
old (ipopt, uncorrected) reference — revise with E1.

## CJT implementation review (2026-11-10)
- **Bug: conditional queries.** The auxiliary-t + floor P(e) ≥ 1e-6 formulation is at SCIP's feasibility
  tolerance, so SCIP certified P(A|E) ∈ [0,1] on `examples/cjt_zero_evidence.lcn` (exact [0.2,0.4]).
  Replaced by the Charnes–Cooper scaling (exact, no floor). Type-1 rows had to be scaled by r (first
  attempt wrongly treated them as homogeneous: c7e5 then gave 0.45). Now scip and ipopt both return
  [0.2,0.4] and 0.9 (c7e5).
- **H4 failures** no longer go to the uncertified 2^n local fallback: a per-atom tree augmented by
  {a} ∪ atoms(e) is built (alarm, evidence C=1: B was [0,1], now [0.0645, 0.1241] sound dual bounds).
- Docstrings/module header rewritten to the current algorithm; no notes about earlier versions.
- **Verification** (`phase0/cjt_verify.py`): 48 random n=5 instances over 8 topologies (incl. chain and
  random with undirected components): (A) reconstruction check — solve with random objectives,
  reassemble the joint by Lemma rip, check every sentence and LMC assertion: max violation 1.1e-6
  (solver tolerance); the check detects 0.08–0.15 violations when the assertion rows are removed, so it
  is sensitive; (B) oracle: CJT vs direct 2^n global, certified endpoints only: max |diff| 2e-7. 0 bad.
  Conditional-query oracle: `phase0/cjt_verify_ev.py`.
- Tests: +4 (Charnes–Cooper model shape, zero-probability evidence ×2 solvers, posterior 0.9, augmented
  route). 198 tests pass.
- Paper: NLP objective, Thm. d5 (now exact over D_LCN^+ = {P(e) > 0}, no ε), its proof, Remarks
  rem:d5reading/rem:denfloor, main-text Thm. 1, the H4 paragraph and Alg. cjt updated to match.
- **Second bug found by the conditional-query oracle** (2026-11-10): the marginal-form LMC rows
  P(x,y) = P(x)P(y) are not homogeneous (degree 1 vs 2) and were not scaled, so the Charnes–Cooper
  program was infeasible (polytree-fr instance) or forced r = 1 (alarm | C=1 returned P(A) instead of
  P(A|C)). Fixed (r·⟨A_a,y⟩ = ⟨A_b,y⟩⟨A_c,y⟩). Also: SCIP `infeasible` returned ±1e20 sentinels that were
  marked certified — now never certified/returned, and values outside [0,1] are rejected.
  Hand-checked: alarm | C=1 is vacuous for all atoms (correct); P(x0 | x3=1) on the polytree instance =
  [0.4368, 0.99] = the bound of s4 directly. The earlier "[0.0645, 0.1241]" for B | C=1 was wrong.
  +2 tests. Conditional oracle rerun: `phase0/cjt_verify_ev2.log`.
- **Third/fourth/fifth bugs, in the direct engine (`exact.py`, the oracle):** `_OPTIMAL` counted
  `feasible` and `locallyOptimal` as confirmed (item B4); its division-with-floor conditional objective
  "confirmed" values like 8.8e15 and -0.014; a SCIP abort ("error in LP solver") raised through the
  engine. Fixed minimally: only optimal/globallyOptimal (or closed gap) confirm; out-of-[0,1] values are
  never confirmed; ApplicationError -> "unsolved". The direct engine still uses division + floor for
  conditional queries (not converted to Charnes–Cooper) — flagged, not done.
- **Conditional-query oracle, final (`phase0/cjt_verify_ev4.log`):** 11 random n=5 chain-graph
  instances (tree-fr, polytree-fr, ktree-fr, dag-fr, chain; random k=0), one random evidence atom each,
  32 endpoints certified by both engines: max |CJT − direct| = 1e-6. 0 disagreements. (The 12th
  instance is not a chain graph; CJT does not apply.) All 200 tests pass.

## Codebase cleanup (2026-11-10; plan: `.claude/plans/codebase-cleanup.md`)
- Removed `cn/ccte.py`, `cn/ijgp.py` (not correct) and every harness/README/CLAUDE.md reference;
  research notes kept.
- Renamed `cn/ibp.py` → `cn/ivbp.py` (class `IntervalBP`), `cn/junction_nlp.py` → `cn/cjt.py`,
  `tests/test_credal_jt.py` → `tests/test_cjt.py`; all in-repo importers and the three notebooks'
  import lines updated. The `phase0/` scratch scripts still import `junction_nlp` (not updated).
- Harness: algorithms are exactly `exact ariel cjt ivbp cve approxlp`; variants by flags (`--solver`,
  `--n-clusters/--cluster-rep/--epsilon`, `--n-iters`, `--cert-tol`), build solver `--cn-solver`,
  stages `--stage[s] compile|enumerate`; output `{algo}{variant_suffix}.jsonl`; records carry
  `variant` and (exact/SCIP) `exact_status`. `run_algorithm.py --verbosity 0` prints exactly one JSON
  document. `analyze_results.py --reference cjt` (default) compares certified endpoints only;
  containment tolerance 1e-4 (was 1e-9, which flagged exact agreement as not contained).
  `run.sh` rewritten to the new interface (trailing flags pass through).
- `results/` → `results_legacy/` (with README). New experiments write to `results/`.
- Tests: new `tests/test_harness.py` (27); 227 pass. Smoke: every algorithm on alarm; batch
  run → analyze → plot on 2 tree instances (CVE/IVBP/CDVE = CJT to 1e-6; CVE_cm off by ~0.05,
  containment 0.2).

## CVE revision (2026-11-10)
- CVE now uses pair potentials (Mauá & Cozman 2020) and is exact with evidence. `thm:cve` was updated
  (it no longer says "without evidence"), and its proof now goes through the odds-covering
  Lemma `lem:cover`.
- Clustering (CVE_cm / CVE_cp) was removed. The Mauá, de Campos & Zaffalon (2012) FPTAS is the
  approximation (CVE_ε, `prop:cve-fptas`: inner bounds within a factor 1+ε).
- Appendix CVE rewritten: potentials, algorithm and alarm example; the box-LP compound projection is
  gone.
- **Pending at the experiment rerun:** the CVE_cm rows of the results tables and the text at
  §6 (setup, line ~641: "potentials clustered to at most 10 functions"). Replace them with CVE_ε
  (e.g. ε = 0.1, run name `cve_e0.1`) or drop them.

## IBP revision (2026-11-10)
- IBP is now credal sum-product on the factor graph (π/λ messages, evidence factors), with messages
  stored as the hull vertices of the normalized message sets.
- `thm:ibp`: exact on polytree credal networks, binary or compound, with or without evidence.
  Proof via projective images of hulls (`lem:proj`).
- Appendix IBP section rewritten (new trace, with evidence).
- **Pending at the rerun:** the experiments still state that all queries are evidence-free. Evidence
  experiments for CVE / IBP / CJT are now possible. Some shipped `polytree_*_fr` instances compile
  to loopy credal networks (e.g. n20_1): check before claiming IBP exactness on that set.

## CDVE / ApproxLP revision (2026-11-11)
- Bugs fixed:
  - evidence on an atom of a compound node crashed (`KeyError`);
  - compound queries optimized only state 1 and then projected a non-attained per-state box onto the
    atoms (alarm D: [0.906, 1] vs exact [0, 1]; not even guaranteed inner).
- Method:
  - per-atom objective;
  - exact family step (one VE for the coefficients, then Dinkelbach over the separately specified
    rows), instead of one VE per vertex per row;
  - barren pruning;
  - escape from the centroid saddle;
  - `n_restarts` random vertex starts (default 8).
- §4.4 and the appendix were rewritten to match (algorithm, A→B→C trace with evidence, complexity
  proof). The new `thm:cdve-cplx` adds A (R+1) T factors.
- **Pending at the rerun:** the CDVE numbers in the tables are from the old code.

## §4.2 family-realizability review (2026-11-11)

Found (each verified numerically; regression files in `examples/fr_counter_*.lcn`):

| issue | effect |
|---|---|
| the compiled local sets are per-state interval boxes, wider than the true local set for compound blocks | CVE P(C) ∈ [0.03, 0.82] vs D_LCN [0.32, 0.58] on a fully realizable LCN |
| the free rows of multi-atom components violate intra-component LMC, e.g. (C ⟂ E \| D) | the vertices violate it by 0.25 |
| a missing local-feasibility hypothesis, even for DAGs | contradictory row sentences force P(U) = 0: exact [0, 0], strong extension [0, 1] |
| the "iff" of the definition / `lem:invariance` was false | e.g. P(C\|D) on a root block is realizable but neither R1 nor R2 |
| `ex:loopyexact` said n=2, k=3 | it is n=4, k=2 |
| §4.2 called IVBP "outer" | stale |

Fixed:
- `prop:suff` / `prop:suff-app`: consistent, **DAG-structured, locally feasible**, fully realizable.
- New proof via a row-wise form of each realizable sentence (`eq:rowwise`): `lem:noGapA` is now
  "row-wise enforcement", with no K^true conventions.
- The LMC step uses the directed factorization.
- `lem:invariance` is now the exact characterization, with its proof.
- `thm:cve` / `thm:ibp` are exact w.r.t. D_SE always, and w.r.t. D_LCN under the hypotheses of
  `prop:suff`; the CDVE inner claim is likewise.
- New appendix subsection `sec:fr-counter`; intro / related-work wording updated.
- Both examples' numbers re-verified; the conclusion is still on page 11.
- Code: `realizability.py` (exact test), the parser τ fix, `tests/test_realizability.py`,
  `tests/test_parser_tau.py`.
