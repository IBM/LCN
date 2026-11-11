# Plan — evaluation harness and results analysis: nine runs, one naming scheme

## 0. Current state (what is inconsistent)

**Algorithm names.** `run_algorithm.py` takes `--algorithm {exact, ariel, cjt, ivbp, cve, approxlp}`;
variants are flags, and the file / run name gets a suffix only for *non-default* flags. Concretely:
- **The default solver is implicit:** `exact` and `cjt` mean SCIP, `exact_ipopt` and `cjt_ipopt` the
  ipopt runs. You can't tell the solver from the name of the default run.
- **The same method has three names:** `approxlp` (CLI, files, records `approxlp_*`), `ApproxLP`
  (class / module), CDVE (paper, LaTeX label).
- **The paper writes IBP** (26 occurrences, incl. `thm:ibp`); code, CLI and LaTeX tables write
  IVBP / `ivbp`.

**Engine APIs.** They name the solver differently: `ExactInference.run(solver="global"|"local")`
vs `CredalJT.run(solver="scip"|"ipopt")`, and the harness translates between them.

**Analysis.**
- `_EXACT_ENGINES` excludes **every** exact/cjt run that is not the reference, so the ipopt runs
  (local, uncertified) are never scored, although they are approximations.
- The metrics assume the scored method is an *outer* approximation ("contain" = outer). There is
  no inner-containment check for CVE_ε / CDVE, no check of a method's own guarantee (CVE exact,
  IVBP exact on polytrees), and no evidence grouping.
- Evidence is a single global `--evidence` JSON for the whole benchmark. It is meaningless across
  instances with different atoms, so in practice every run is evidence-free.

**Display names** are hard-coded three times: `analyze_results._LATEX_ALGO_NAMES`,
`plot_results.ALGO_STYLE`, `run.sh`.

## 1. The nine runs (single source of truth)

A `RUNS` registry in `experiments/runs.py`, imported by `run_algorithm.py`, `run_experiment.py`,
`analyze_results.py`, `plot_results.py` and the tests. Each run has a canonical **run name** (used
for the file, the record, the CSV and the plots), a display label, the engine call, and its
**guarantee**:

| run name     | algorithm | flags                    | label (paper / tables)     | guarantee (vs 𝒟_LCN, under the paper's hypotheses) |
|--------------|-----------|--------------------------|----------------------------|-----------------------------------------------------|
| `exact_scip` | exact     | `--solver scip`          | Exact (SCIP)               | exact where certified (per endpoint)               |
| `exact_ipopt`| exact     | `--solver ipopt`         | Exact (ipopt)              | none (local NLP)                                   |
| `ariel`      | ariel     | (ipopt)                  | ARIEL                      | none                                               |
| `cjt_scip`   | cjt       | `--solver scip`          | CJT (SCIP)                 | exact where certified (H1–H4 + SCIP cert)          |
| `cjt_ipopt`  | cjt       | `--solver ipopt`         | CJT (ipopt)                | none (local NLP)                                   |
| `cve`        | cve       |                          | CVE                        | exact vs 𝒟_SE (= 𝒟_LCN if family-realizable)      |
| `cve_eps_{E}` | cve       | `--epsilon E`            | CVE$_\varepsilon$ (ε = E)  | inner vs 𝒟_SE, within factor 1+E                   |
| `ivbp`       | ivbp      |                          | IVBP                       | exact vs 𝒟_SE on polytrees (`ivbp_exact`), else none |
| `cdve`       | cdve      | `--n-restarts R`         | CDVE                       | inner vs 𝒟_SE                                      |

Naming rules:
- **The solver is always explicit** for the two NLP engines (`exact_scip`, not `exact`). This
  removes the "implicit default" ambiguity.
- **ε is part of the name** (`cve_eps_0.1`). Default for CVE_ε: E = 0.1, the value the paper will
  report.
- **Tuning flags do not create new algorithms.** `--n-iters` (ariel, ivbp, cdve) and `--n-restarts`
  (cdve) keep their defaults (10 / 100 / 50, 8). A non-default value is recorded in the record's
  `params` and appended as a suffix (`cdve_r0`, `ariel_it20`), so files never collide. The
  analysis groups by full run name.
- **`approxlp` is renamed `cdve` everywhere:**
  - CLI `--algorithm cdve`;
  - module `lcn/inference/marginal/cn/cdve.py`, class `CDVE` (the docstring keeps "follows
    ApproxLP (Antonucci et al. 2015)");
  - record fields `cdve_sweeps` / `cdve_eliminations`, test file `tests/test_cdve.py`.
  - This is a clean break, as for ibp→ivbp: no alias.
- **CLI algorithm choices** become exactly `exact ariel cjt cve ivbp cdve`. `--solver` is
  **required** for exact and cjt: no hidden default, and argparse errors if it is missing. A
  `--run NAME` shortcut accepts any registry name (`--run cjt_ipopt`, `--run cve_eps_0.1`).
- **Engine API aligned:** `ExactInference.run(solver="scip"|"ipopt")`. `"global"`/`"local"` are
  removed and callers updated (tests, notebooks only if they use it; checked).
- **Record schema:** every record carries
  - `run` (the canonical name), `algorithm`, `params` (all flags, defaults included; replaces
    `variant`), `guarantee`, `evidence` and `status`;
  - and its engine flags: `exact_status`, `cjt_*`, `cve_alpha` etc., `ivbp_exact` / `ivbp_converged`,
    `cdve_sweeps`.
  - Old field names (`variant`, `approxlp_*`) are not read; old logs live in `results_legacy/`.

## 2. Evidence that is meaningful per instance

`run_experiment.py --evidence-atoms K [--evidence-seed S]`:
- **Per instance**, draw K atoms and values with a seed derived from (instance basename, S, K).
  Every run of that instance gets the **same** evidence.
- The sample is drawn **from the reference distribution's support**: values are kept only if
  P̲(e) > 0 under a quick check. The check is cheap: CVE's / IVBP's `empty` flag on the compiled
  network, or a centroid evaluation. The checks that P̲ > 0 holds are part of the paper's theorem
  hypotheses for CVE / IVBP.
- **Output path:** `results/{benchmark}/ev{K}/{run}.jsonl` (`ev0` = no evidence), so evidence
  levels never mix.
- **Kept:** `--evidence JSON` for single-instance runs in `run_algorithm.py`.

## 3. Run orchestration
- `run_experiment.py --runs NAME...` (registry names) or `--runs all`. `--algorithms` is kept only
  as an alias that expands each algorithm to its default run(s); recommended to drop it.
  `--stages compile enumerate` is unchanged.
- **`--exact-threshold` (max n for `exact_*`)** stays. Applied to both exact runs; CJT unrestricted.
- **`run.sh <bench> <run> <factorization> [time] [flags]`:** `run` is a registry name. The help
  text lists the nine runs.
- **Per-solve time limits:** the harness splits the budget across endpoint solves
  (`per_solve_limit`). Keep this for `exact_*` and `cjt_*`.

## 4. Analysis (`analyze_results.py`)
- **Reference:** `--reference` (default `cjt_scip`), only **certified** endpoints. `exact_scip` can
  be the reference on small sets.
- **Scored rows:** every run except the reference. The other certified engine is reported as a
  *cross-check* row ("agreement on endpoints both certify"), not hidden. The ipopt runs are scored
  as approximations (they are local, no guarantee).
- **Common instance set:** scores use the instances where the reference has status ok. Per run it
  reports **solved** (ok / timeout / memory / error counts), so failures are visible instead of
  silently dropped.
- **Metrics:**
  - kept: MAE / max of the lower and upper endpoints, width ratio, times, induced width;
  - **outer rate** (contains the reference) and **inner rate** (contained in it), both with the
    certification tolerance;
  - **guarantee check:** for each record, does the result satisfy the run's declared guarantee
    against the reference? This uses the registry `guarantee` field and the record's own flags. CVE
    and IVBP-with-`ivbp_exact` must match; CVE_ε and CDVE must be inner; CVE_ε must also be within
    1+ε of CVE when the `cve` run is present. Violations are counted and listed per instance/atom.
    A violation means a bug or a broken hypothesis (e.g. a non-family-realizable instance), and the
    analysis prints it.
- **Grouping:** `--group-by {benchmark, size, evidence}` and combinations (e.g. `size,evidence`).
- **LaTeX / CSV:**
  - labels come from the registry;
  - the column order is the registry order;
  - `--runs` restricts and orders the rows;
  - the old `_LATEX_ALGO_NAMES` / `_LATEX_EXCLUDE` are removed.

## 5. Plots (`plot_results.py`)
- Style (marker, colour, label) comes from the registry. The default plotted runs are
  `ariel ivbp cve cve_eps_0.1 cdve` against the reference, plus `--runs` to choose.
- Optional evidence panel: error vs number of evidence atoms.

## 6. Tests
- **`tests/test_harness.py` rewritten around the registry:**
  - registry names ↔ CLI ↔ file names round-trip;
  - `--solver` required for exact/cjt;
  - removed names rejected (`approxlp`, `exact` without solver, `ibp`, `cve_cm`, …);
  - tuning suffixes; record schema;
  - deterministic per-instance evidence;
  - analysis: certified-only reference, scored ipopt runs, inner/outer rates, guarantee-violation
    detection on synthetic records, solved counts, grouping by evidence.
- **`tests/test_cdve.py`** (renamed), updated imports everywhere.
- **Smoke test:** `run_experiment.py --runs all` on 2 tiny instances × ev0/ev1 with a short time
  limit, then `analyze_results.py`, end to end in a temp dir.

## 7. Docs and paper
- **CLAUDE.md, README:** run table, CLI, file layout, record schema.
- **Paper, consistent method names (text only; the experiments section waits for the reruns):**
  - IBP → **IVBP** in all prose, algorithm captions and theorem names (labels may stay `thm:ibp`).
  - CDVE stays CDVE (it "follows ApproxLP").
  - CVE$_\varepsilon$ for the FPTAS.
  - Exact (SCIP) / Exact (ipopt), CJT (SCIP) / CJT (ipopt) where the solver matters.
- **revision_plan.md:** the rerun checklist becomes "nine runs × benchmarks × ev{0, 1, 3}".

## 8. Order of work
1. `experiments/runs.py` registry, plus the rename approxlp → cdve (module, class, tests,
   callers), plus `ExactInference(solver="scip"|"ipopt")`.
2. `run_algorithm.py` (CLI, record schema), `run_experiment.py` (`--runs`, per-instance evidence,
   `ev{K}/` layout), `run.sh`.
3. `analyze_results.py` (reference, scored set, inner/outer, guarantee checks, solved counts,
   grouping), `plot_results.py`.
4. Tests and the smoke test; docs; paper naming pass.

## Open questions
1. **Always-explicit solver names** (`exact_scip`, `cjt_scip`), or keep the bare `exact` / `cjt`
   for the SCIP default? Recommended: explicit.
2. **Module / class rename** approxlp.py / `ApproxLP` → cdve.py / `CDVE` (code matches the paper),
   or keep the module name and only rename the run? Recommended: full rename.
3. **Paper:** rename IBP → IVBP throughout? Recommended: yes (the code, harness and your request
   use IVBP).
4. **CVE_ε value(s)** to run by default: only ε = 0.1, or also 0.01? Recommended: 0.1 (one row; ε
   is a flag, so more can be added).
5. **Evidence levels** for the reruns: K ∈ {0, 1, 3}? Recommended: {0, 1, 3}.

## Decisions (2026-11-11)
1. The solver is always explicit: `exact_scip`, `exact_ipopt`, `cjt_scip`, `cjt_ipopt`;
   `--solver` is required.
2. Full rename to CDVE: CLI / run / records `cdve`, module `cn/cdve.py`, class `CDVE`,
   `tests/test_cdve.py`. Clean break.
3. Paper: IBP → IVBP everywhere (prose, captions, theorem names; labels may stay).
4. Rerun grid: `cve_eps_0.1` only; per-instance evidence K ∈ {0, 1, 3} → `results/<bench>/ev{K}/`.

5. CVE_ε run naming: base name `cve_eps`, with the ε value appended after an underscore, e.g.
   `cve_eps_0.1` for ε = 0.1 (file `cve_eps_0.1.jsonl`, record `run: "cve_eps_0.1"`,
   label CVE$_\varepsilon$ (ε = 0.1)). The value is formatted with `%g` (0.1, 0.01, 1e-3 →
   `cve_eps_0.001`).
   - `--run cve_eps_0.1` and `--algorithm cve --epsilon 0.1` are equivalent.
   - `--run cve_eps` without a value is rejected.

## Implementation status (2026-11-11)

**Done:**
- `experiments/runs.py`: registry, `parse_run` / `run_name` / `guarantee` / `label` / `style` /
  `order_key`.
- **Renames:**
  - `approxlp.py`/`ApproxLP` → `cdve.py`/`CDVE`, `tests/test_cdve.py`;
  - `ExactInference(solver="scip"|"ipopt")` (the `"global"`/`"local"` values are gone; callers
    updated in cjt, the generator, tests, README).
- **`run_algorithm.py`:**
  - `--run NAME` or `--algorithm` + flags; `--solver` required for exact/cjt (CLI error and
    `ValueError` in `run_single`);
  - records carry `run` / `algorithm` / `params` / `guarantee` (no `variant`), plus `cdve_*` fields.
- **`run_experiment.py`:** `--runs … | all`, per-instance deterministic evidence
  (`sample_evidence`, `--evidence-atoms`, `--evidence-seed`), layout `results/<bench>/ev<K>/<run>.jsonl`.
- **`analyze_results.py` rewritten:**
  - certified-only reference (default `cjt_scip`); every other run scored;
  - evidence atoms excluded;
  - MAE / std / max, outer / inner rates, width ratio, guarantee violations (with a list and a CSV),
    solved counts;
  - `--group-by` any of benchmark, evidence, type, size, instance; CSV / LaTeX / violations outputs.
- **`plot_results.py` rewritten:** single `--csv`, `--runs`, `--benchmark`, `--evidence`,
  `--by-evidence`, styles from the registry.
- **`run.sh`:** takes a run name, or `all`.
- **Tests:** `tests/test_harness.py` (53, incl. an end-to-end smoke test: 6 runs × ev0/ev1 →
  analysis → plots).
- Full suite: 341 passed.
- **Docs and paper:**
  - CLAUDE.md, README updated;
  - paper IBP → IVBP (28 occurrences; labels unchanged); compiles at 56 pages.

**Deviations from the plan:**
- **No P̲(e) > 0 filter on the sampled evidence.** It is not cheaply computable (minimizing P(e)
  over the strong extension is itself a credal problem). Evidence is drawn uniformly, and every
  engine applies regular conditioning. Documented in CLAUDE.md and the `run_experiment` docstring.
- **`--algorithms` dropped** from `run_experiment.py` (replaced by `--runs`); `--evidence JSON`
  is kept only in `run_algorithm.py`.

**Observed in the smoke run** (2 tiny trees; not a result):
- `exact_ipopt` is far from exact without evidence (MAE ≈ 0.15–0.23);
- ARIEL errs up to 1.0 with evidence;
- `cjt_scip` with evidence on alarm needs the time-limit split (a SCIP endpoint hits its limit);
- all guarantee checks hold.
