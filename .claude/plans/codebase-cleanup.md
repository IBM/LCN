# Codebase cleanup plan — remove CCTE/IJGP, rename IBP→IVBP and junction_nlp→cjt, slim the harness

Decisions (confirmed): variants via **flags**, not names; old `results/` **archived**; CCTE/IJGP removal is
**code only** (research notes kept); renames are a **clean break** (no shims), all in-repo callers updated.

## 1. Remove CCTE and IJGP (code only)
- `git rm lcn/inference/marginal/cn/ccte.py lcn/inference/marginal/cn/ijgp.py`.
  Both are leaves: the only importer is `experiments/run_algorithm.py`.
- Remove their harness branches (`ccte`, `ccte_e`, `ccte_cp`, `ccte_cm`, `ijgp`, `ijgp_e`, `ijgp_cp`,
  `ijgp_cm`) and their CLI flags (`--ibound`, the ijgp-only help text, `ccte_e` epsilon help).
- Docstring/comment mentions to update (no behaviour change): `credal_network.py:284`,
  `vertices.py:22`, `cve.py:113` ("same approximation CredalCTE offers"), `cve.py:220`
  ("Same LP as CredalCTE._solve_singleton_lp" → describe the LP directly), `run_experiment.py` docstring
  and `APPROX_ALGORITHMS`, `analyze_results.py` `_ALGORITHMS`.
- `README.md`, `CLAUDE.md`: drop `ccte.py`/`ijgp.py` from the module lists.
- Kept (historical): `docs/research-chain-lcn/scratchpad/{credal_cte,ccte_exactness,properties,
  tighter_approximation}.tex`, `.claude/plans/ccte.md`, research-plan docs.
- Check: `potentials.Potential.cluster_prune` and `epsilon_prune` stay — CVE uses them.

## 2. Rename `ibp.py` → `ivbp.py`
- `git mv lcn/inference/marginal/cn/ibp.py lcn/inference/marginal/cn/ivbp.py`.
- Class: keep `IntervalBP` (already the correct name for interval belief propagation; renaming it adds
  churn for no gain). Log prefix `[IBP]` → `[IVBP]`.
- Callers: `experiments/run_algorithm.py` (import + branch). No tests import it today.
- Comment in `ibp.py` itself points to `docs/ibp_exactness.tex`: leave the doc filename, update nothing
  else in docs.

## 3. Rename `junction_nlp.py` → `cjt.py`
- `git mv lcn/inference/marginal/cn/junction_nlp.py lcn/inference/marginal/cn/cjt.py`. Class
  `CredalJT` unchanged.
- In-repo importers to update:
  - `lcn/inference/utils/structured_consistency.py` (deferred import + docstring),
  - `lcn/benchmarks/generator.py` (`cjt_exactness_report`),
  - `lcn/inference/marginal/cn/cve.py` (docstring),
  - `experiments/run_algorithm.py`,
  - `tests/test_credal_jt.py` (≈15 `from ...junction_nlp import`), `tests/test_generator_cjt_exact.py`
    (via generator only — check),
  - `notebooks/demo_{alarm,chain_graph,smokers}.ipynb`: **only** the import line (these notebooks have
    uncommitted user edits — touch nothing else),
  - `CLAUDE.md`, `README.md`.
- Not updated (scratch/history): `docs/research-chain-lcn/iclr2027/phase0/*.py`, `.claude/plans/*`,
  `docs/research-*/research-plan.md`. Note in `revision_plan.md` that phase0 scripts import the old
  name.
- Internal "[D5]" log prefixes → "[CJT]" (cosmetic, same change).

## 4. Harness: six algorithms + variant flags
`experiments/run_algorithm.py`:
- `ALGORITHMS = ["exact", "ariel", "cjt", "ivbp", "cve", "approxlp"]`; keep the two pipeline steps
  `compile` and `enumerate` as **build stages**, not algorithms (they write the `.cn`/`.vtx` caches the
  credal-network methods load) — exposed as `--stage compile|enumerate` (or kept as hidden names;
  decide during implementation, default: keep them under a separate `STAGES` list accepted by
  `--algorithm` for backward compatibility of the batch scripts? → **No**: separate `--stage` flag).
- Variant flags (each recorded in the result record as `variant`):
  | algorithm | flag(s) | default | replaces |
  |---|---|---|---|
  | `exact` | `--solver scip\|ipopt` | `scip` | `exact_g` / `exact_l` |
  | `cjt` | `--solver scip\|ipopt`, `--cjt-time-limit`, `--cert-tol` | `scip` | `cjt`/`cjt_g` / `cjt_l` |
  | `cve` | `--n-clusters N` (0 = exact), `--cluster-rep plub\|mean`, `--epsilon` | `0`, `mean` | `cve`, `cve_cm`, `cve_cp`, `cve_e` |
  | `ivbp` | `--n-iters`, `--threshold` | 100, 1e-6 | `ibp` |
  | `approxlp` | `--n-iters` | 50 | `approxlp` |
  | `ariel` | `--n-iters` | 10 | `ariel` |
  `cve_d4` (cross-family coupling) is dropped; `--solver` for the **compile** step (per-family LPs) is
  renamed `--cn-solver` to avoid clashing with the engine solver.
- Output file naming: `{algo}{suffix}.jsonl` where the suffix encodes non-default variants
  (`cve_c10mean.jsonl`, `exact_ipopt.jsonl`, `cjt_ipopt.jsonl`), so different variants never overwrite
  one another. Record fields: `algorithm`, `variant` (dict of the flags), plus the existing CJT fields
  (`cjt_h3`, `cjt_certified`, `cjt_exact`, `cjt_solves`).
- Exact engine status: record per-endpoint `confirmed/unconfirmed/unsolved` (already available as
  `ExactInference.status`) so analysis can filter to certified endpoints.

`experiments/run_experiment.py`: same `--algorithms` choices and flags, passthrough to `run_single`;
drop `ccte`/`ijgp` examples from the docstring; `APPROX_ALGORITHMS = ["ariel", "ivbp", "cve",
"approxlp"]`.

`experiments/analyze_results.py`:
- `_ALGORITHMS` → the six names (+ variant-suffixed files discovered by glob).
- Reference selection: `--reference cjt` (default; **certified endpoints only**, using
  `cjt_solves[...].certified`), `--reference exact` (certified endpoints only). Replaces the current
  `cjt_any` (prefers the uncertified ipopt backend) and `exact` (prefers exact_g then exact_l).
- Labels: `ivbp` → "IVBP", `approxlp` → "CDVE", `cve` with clusters → "CVE$_{cm}$".

`experiments/plot_results.py`: style table keyed by the new names; `DEFAULT_ALGORITHMS = ["cve",
"cve_c10mean", "ariel", "ivbp", "approxlp"]`.

## 5. Old results
- `git mv results results_legacy` (untouched; old names, old CJT reference). `analyze_results.py`
  does not read `results_legacy/` by default. Add `results_legacy/README.md`: generated before the CJT
  triage/certification fixes; do not use as a reference.

## 6. Tests and verification
- Update imports in `tests/test_credal_jt.py`; rename it `tests/test_cjt.py` for consistency.
- New `tests/test_harness.py` (fast, no solver-heavy runs): `ALGORITHMS` is exactly the six names;
  removed names are rejected by argparse; variant flags map to the right engine calls (monkeypatched
  engines); output-file suffixes are distinct per variant; `--reference cjt` drops uncertified endpoints.
- Smoke run: `run_algorithm.py --algorithm {each}` on `examples/alarm.lcn`, plus `run_experiment.py`
  on `benchmarks/tree_small_fr` restricted to n=5 (1–2 instances), then `analyze_results.py` and
  `plot_results.py` on that output.
- Full `uv run pytest tests/`; `uvx ruff check --select F,E9` on all touched files; `grep -rn
  "junction_nlp\|ccte\|ijgp\|cn\.ibp\b"` over `lcn/ experiments/ tests/ notebooks/ README.md
  CLAUDE.md` must come back empty.

## 7. Docs
- `CLAUDE.md`: module list (`cjt.py`, `ivbp.py`; no ccte/ijgp), harness section (six algorithms,
  variant flags, `--stage`, `results_legacy/`), the CJT notes now under `cjt.py`.
- `README.md`: same module/algorithm list.
- Paper: no change needed for this cleanup (it already says CJT/IVBP? — **check**: the paper says
  "IBP"; renaming the algorithm in the paper to IVBP was the JNVR review item; flag, don't change here).

## Order
1 → 2 → 3 (each followed by `pytest` and the grep check) → 4 → 5 → 6 → 7. One commit per step if
committing is requested.

## Risks
- `run_experiment.py`/batch scripts or the user's shell history may call old names (`cjt_l`, `exact_g`,
  `cve_cm`); argparse will reject them with the valid choices listed — intended, since the clean break
  was chosen.
- The `.cn`/`.vtx` caches are keyed by `(method, merge_budget, solver)`; renaming `--solver` →
  `--cn-solver` must keep writing the same key so existing caches stay valid.
