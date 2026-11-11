# Legacy results (do not use as a reference)

These JSONL logs were produced by the experiment harness **before** the CJT and
harness revisions of November 2026, under the old algorithm names (`ibp`,
`cjt_l`, `cjt_g`, `cve_cm`, `exact_g`, `ccte*`, …).

- The CJT records predate the LMC-triage fix: on most instances an LMC
  assertion was dropped, so `cjt`/`cjt_l` bounds are outer, not exact, and must
  not be used as the reference.
- SCIP endpoints were recorded without certification; `exact_g` counted
  `feasible` terminations as confirmed.
- CCTE and IJGP have since been removed from the code base.

`experiments/analyze_results.py` reads `results/` by default and never this
directory. Rerun the experiments with the current harness instead.
