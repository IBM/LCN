# Revision plan v3 — rewrite the experiments with the Nov-2026 rerun

Input: `results/` (SLURM array 1154751 and follow-ups), ev0 only, 6060/6060 records,
11 sets (8 `_fr` + `nr_tree`, `nr_dag`, `nr_dag_ablation`) × 9 runs
(`exact_scip`, `exact_ipopt`, `ariel`, `cjt_scip`, `cjt_ipopt`, `cve`, `cve_eps_0.01`, `ivbp`, `cdve`).
Budget per run: 3600 s, **8 GB** (the paper currently says 30 GB), 1 thread.
Exact (2^n) runs only for n ≤ 15. Reference: `cjt_scip`, certified ∧ structural endpoints only.

## 1. Findings (what the new numbers say)

### 1.1 The reference (CJT + SCIP)
| set | endpoints certified | instances fully certified | median / max time (s) |
|---|---|---|---|
| tree, polytree, ktree (small+large) | 100 % | 240/240 | 0.6–4.5 / ≤ 49 |
| dag_small | 98.6 % | 56/60 | 0.9 / 1141 |
| dag_large | 85.0 % | 18/60 (1 timeout at n=100) | 389 / 3304 |
| nr_tree | 99.7 % | 94/100 | 1.7 / 663 |
| nr_dag | 95.6 % | 55/100 | 54 / 1331 |
| nr_dag_ablation | 90.4 % | 40/60 | 9.7 / 3242 |

* Ordered triage used on **every** instance; nothing relaxed; max cluster 2–8 atoms.
* CJT certifies on trees/polytrees/k-trees up to n=100 in seconds; DAGs are the hard case
  (median 787 s at n=100), and certification drops with the number of nr sentences on DAGs
  (96 % at m=0 → 94 % at m=8; median time 19 s → 168 s).

### 1.2 Exact inference over the 2^n joint is not a usable baseline
* `exact_scip` certifies everything only at n=5 (tree, ktree); by n=7 most endpoints are
  `unsolved` on tree/polytree/dag (3600 s). ktree is the exception (confirmed up to n≈9).
* `exact_ipopt` times out (≥25/60 per small set, all of the large/nr ones) and is wrong when it
  finishes (MAE 0.13–0.20).
* So the old story "exact solves 15/60" becomes: "exact certifies only n ≤ 6; CJT certifies
  n = 100 trees in 11 s".

### 1.3 Family-realizable sets: theory confirmed
* **CVE / CVE_ε (ε=0.01)** equal the reference on every `_fr` set (MAE ≤ 1e-5, outer ≥ 0.998,
  inner = 1.000). This is Prop. `prop:suff` in action: P_SE = D_LCN. 3 "violations" in
  dag_large are numerical (≤ 2.2e-4, vs. the 1e-4 certification tolerance) — state this.
* **CDVE** (R=8 restarts): inner on 100 % of atoms, MAE ≤ 1e-4 everywhere, agrees with CVE to
  < 1e-4 on 99–100 % of endpoints, worst miss 0.085 (tree_large). Runs in 0.2–16 s.
  **This reverses the old text** ("CDVE 273–614 s, slower than ARIEL").
* **IVBP**: exact (0 error, flag `ivbp_exact`) on all tree/polytree instances (polytrees are
  now genuine forests — the old "18/60 loopy" caveat is gone) and on 16/60 dag_small. On loopy
  ktree/dag: MAE ≈ 1e-3, max 0.10, and **neither outer nor inner** (outer of CVE on only
  86–91 % of endpoints). Always converges, 2–4 sweeps. Fastest method everywhere (≤ 11 s at n=100).
* **ARIEL**: 0 error on trees; MAE 3e-4–4e-3 elsewhere but max 0.38; no guarantee (outer
  0.97–1.0, inner 0.80–0.97). 10–150 s — the slowest approximate method at scale.
* **CJT + ipopt**: fast, exact on tree/ktree/polytree_large, but local optima on DAGs
  (dag_small MAE 0.0095, max 0.83; dag_large n=20 MAE 0.018). Shows that SCIP, not the
  formulation, is what buys certification.
* **Scalability**: CVE runs out of 8 GB on k-trees (k=3) from n=10 (15/60 large solved;
  CVE_ε 36/60) and on dag n ≥ 50 (49/60). Credal-set sizes, not treewidth alone, limit CVE.
  CDVE/IVBP cover everything.

### 1.4 Almost-realizable sets: the compilation gap, measured
* All CN methods (CVE, CVE_ε, CDVE, IVBP-on-trees) give the **same** answer, so the error is
  entirely the compilation gap P_SE ⊋ D_LCN, not algorithmic error. It is **outer**
  (outer = 1.000 for CVE): consistent with `prop:incl` (D_LCN ⊆ P_SE for DAG-structured LCNs).
* Gap grows ~linearly with m: tree n=10 MAE 0 → 0.028 (m=1) → 0.072 (m=4) → 0.13–0.15 (m=8);
  width ratio 1.25 → 2.3. DAG n=10: 0.037 → 0.096 → 0.16. Larger n dilutes it (n=20 halves it).
* Ablation (dag n=10, m=4, single kind): `nr1` (non-root marginal) is by far the worst
  (MAE 0.14–0.16 at δ=0.05, width ×3.9); `nr2` (mixture) 0.02–0.03; `nr3` (cross-family)
  0.01–0.05. Tighter δ ⇒ larger gap (the LCN constrains more than the CN can express).
* **ARIEL is the most accurate engine here** (MAE ≤ 0.013, mostly ≤ 0.005), because it works
  on the LCN, not on the compiled CN — but it has no guarantee.
* CJT+ipopt degrades with m (local optima): up to 0.096 MAE on dag m=8.

### 1.5 Take-home messages for abstract / intro / conclusion
1. CJT+SCIP is a practical *certified* exact engine: 100 % certified on tree/polytree/k-tree up
   to n=100, ≥ 85 % of endpoints on unstructured DAGs, where 2^n exact inference stops at n≈6.
2. On family-realizable LCNs the CN route is exact and much cheaper: CVE matches the certified
   reference; CDVE gives inner bounds within 1e-4 in seconds; IVBP is exact on polytrees and
   within ~1e-3 on loopy ones, at sub-second to ~10 s cost.
3. When realizability fails, the gap is outer, grows with the number of non-realizable
   sentences and is dominated by non-root marginals; CJT remains exact, ARIEL is tight but
   unguaranteed. This is the practical argument for having both routes.

## 2. Decisions needed from the user
* **Realistic LCNs** (`benchmarks/real`, `benchmarks/junkyu`, Tab. `tab:real`,
  `outputs/results-{real,junkyu}.tex`) were **not** rerun; their numbers use removed
  algorithms (CVE_cm, old IVBP/CDVE) and the old ipopt reference. Options: (a) rerun them on
  SLURM now (2 sets × 9 runs), (b) drop them, (c) keep with a "pre-revision" caveat (not
  recommended).
* Only ev0 was run. The paper claims conditional-query exactness; either run an evidence
  grid (e.g. ev2/ev5 on the small sets) or state explicitly that experiments are marginal-only.
* Main-text figure: which set(s). Proposal: one 2×2 figure — (a) dag_large MAE, (b) dag_large
  time, (c) nr MAE vs m (tree & dag, n=10), (d) time vs n for all four large topologies
  (or a compact summary table instead of (d)).

## 3. Edits

### 3.1 Artefacts (regenerated, no hand-typed numbers)
* `analyze_results.py --group-by benchmark,size --latex` → `outputs/results-{tree,polytree,ktree,dag}-{small,large}.tex`
  (runs: cjt_ipopt, ariel, cve, cve_eps_0.01, ivbp, cdve, exact_scip/ipopt on small).
  Add columns `outer`, `inner`, `solved` (currently missing).
* New `outputs/results-nr.tex` (by type,size,nr_m) and `outputs/results-nr-ablation.tex`
  (by nr_kind,nr_delta); new `outputs/results-reference.tex` (certification table 1.1).
* `plot_results.py` → `figs/{set}_mae.pdf`, `figs/{set}_runtime.pdf` for the 8 sets, plus
  `figs/nr_mae_vs_m.pdf` (`--x nr_m`). Remove stale figs (CVE_cm curves).
* A script `phase0/make_exp_artifacts.py` that regenerates all of the above from `results/`.

### 3.2 Main text, §5 Experiments (rewrite)
* Setup: Python 3.12, ipopt 3.14.19 (MUMPS), SCIP 10.0.2, 3600 s / 8 GB / 1 thread, SLURM.
* Benchmarks paragraph: keep, update (polytrees are forests; nr sets: 260 instances).
* Algorithms: the nine runs from `runs.py`; drop CVE_cm; CVE_ε with ε=0.01; CDVE R=8.
* Reference & metrics: CJT+SCIP certified endpoints; MAE_l/MAE_u, outer/inner rates, width
  ratio, solved counts. Note that exact_* is a baseline, not the reference, and why.
* Results paragraphs: (1) reference/certification, (2) `_fr` sets (1.3), (3) almost-realizable
  (1.4), (4) [realistic, depending on decision].
* One compact table: per topology (large), MAE / outer / inner / time at n=100 for each run.

### 3.3 Appendix "Detailed Experimental Evaluation" (rewrite)
* `sec-cjt-gate`: replace the "293-instance sample, 54 uncertified" text by table 1.1.
* `sec-nr-bench`: keep the construction; add the result tables (by m; ablation).
* `sec-coverage`: rewrite from the solved counts (CVE memory-outs on ktree/dag, exact_* coverage,
  cjt_ipopt timeouts, CJT one timeout); delete the "bookkeeping gaps" paragraph.
* Random LCNs: new figures/captions (polytree caption: no loopy instances anymore; ktree
  caption: CVE memory-out; dag caption: CDVE 6 s not 600 s).
* New subsection: guarantee checks (violations: 0 on `_fr` except 3 numerical; expected
  violations on nr; loopy IVBP not outer).
* New subsection: CJT backend comparison (SCIP vs ipopt) and exact-2^n baseline.

### 3.4 Abstract, Introduction, Conclusion
* Abstract: replace the last sentence with 2–3 quantitative findings (1.5).
* Intro: add a short "Findings" sentence at the end of the contributions paragraph; replace
  "against CJT as the exact reference" by "against certified CJT endpoints".
* Conclusion: add the empirical summary (1.5) and the next step it suggests (compiling
  non-root marginals / tighter local sets; scaling CVE on k-trees).
* Reproducibility statement: mention the run registry and the regeneration script.

### 3.5 Consistency checks before finishing
* grep for `CVE_{cm}`, `CVE$_{cp}`, `IBP`, `30\,GB`, `ApproxLP`, `TODO(rerun)` → none left.
* Every number in text traced to an artefact; build the PDF (latexmk, light) and check page
  budget (main text ≤ 9 pages + refs).
