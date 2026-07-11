# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LCN (Logical Credal Networks) is a Python library for probabilistic logic that combines propositional logic with imprecise probability bounds. An LCN program specifies constraints like `0.3 <= P(A | B) <= 0.5` over propositional formulas, defining a set of probability distributions. The library provides exact and approximate inference algorithms for marginal queries and MAP/MMAP explanations.

## Build & Development Commands

The package is built with the **hatchling** backend and is installable with **uv** (recommended) or plain `pip` (requires Python 3.10+; 3.12 recommended).

```bash
# Recommended: uv (creates the venv and installs from uv.lock)
uv sync
uv run pytest tests/                 # run the test suite
uv run python lcn/inference/marginal/exact.py   # run a module demo

# Alternative: pip in a conda env
pip install -e .
pytest tests/

# Single test / specific test
pytest tests/test_mixed_graph.py
pytest tests/test_mixed_graph.py::TestClassName::test_method -v
```

**External dependency:** The `ipopt` nonlinear solver is required at runtime. Install via `brew install ipopt` (macOS) or build from source via `coinbrew` (Linux). The local install is `ipopt 3.14.19` with **MUMPS only** (no HSL/pardiso/spral), so `linear_solver` is not a tuning lever.

**Optional solver:** `scip` (`brew install scip`, local install 10.0.2) is an optional *global* solver used **only** by the verifier `lcn/inference/marginal/verify_scip.py`, which Pyomo drives via its AMPL/NL interface (`SolverFactory('scip')`). It is not needed by the core inference engines. Being a spatial branch-and-bound global solver on a nonconvex bilinear problem, it certifies some marginals fast (e.g. alarm C/D) but may hit the per-solve time limit on harder atoms — the verifier reports `CERTIFIED`/`PARTIAL`/`UNSOLVED` with the optimality gap.

**Lint:** `uvx ruff check <path>` (ruff is not a declared dependency; run via `uvx`).

## Architecture

### Core Model (`lcn/core/`)
- `model.py` — `LCN` (main model class; load via `l = LCN(); l.from_lcn("file.lcn")`), `Sentence` (Type1 `P(φ)` / Type2 `P(φ|ψ)` with lower/upper bounds), `Formula`, `Atom`. The LCN builds a **primal graph**, **structure graph**, and derives **independence assumptions** via the Local Markov Condition (`local_markov_condition()`).
- `parser.py` — parses propositional formulas. Operators: `and`/`&`, `or`/`|`, `xor`/`^`, `nand`/`/`, `not`/`!`. Uses `json_schema.py`.
- `independencies.py` — `Independencies` / `IndependenceAssertion` (conditional independence statements; `event1=X`, `event2=Y`, `event3=S` for `X ⟂ Y | S`).
- `mixed_graph.py` — `MixedGraph` (directed + bidirected edges) for deriving Markov conditions.

### Inference (`lcn/inference/`)

All inference algorithms use **Pyomo** to formulate optimization problems solved by `ipopt`. The marginal NLPs are **nonconvex** (conditional-probability ratios + quadratic Markov equalities), so ipopt is a *local* solver — see the gotchas in the [[ipopt-config-shared]] and [[exact-not-a-reliable-oracle]] memories.

**Marginal inference** (`lcn/inference/marginal/`):
- `exact.py` — exact marginal inference over the full 2^n joint; `ExactInference.run(..., mode="exact"|"fast")`.
- `ariel.py` — ARIEL message-passing (approximate) over a factor graph; also has a factor-graph + independence-assumption analysis (`ArielInference.analyze()`).
- `sccp.py`, `sccp_ariel.py` — SCC factor-graph propagation (see [[sccp-algorithms]]).
- `cn/` — the **credal-network family** (compile to a credal net, enumerate extreme points): `cve.py` (Credal Variable Elimination — the hub), `factorization.py`, `ibp.py` (Interval BP), `ccte.py` (Credal Cluster Tree Elimination), `approxlp.py` (ApproxLP), `ijgp.py` (Iterative Join-Graph Propagation). `cve.py` requires the optional `pyagrum` dependency.
  - `credal_network.py` — `CredalNetwork` (the compiled credal net: chain-graph factorization + **interval** local credal sets P(child|parents), pyAgrum-free); `from_lcn(..., n_jobs=K)` computes the per-family interval solves in parallel (each family is one `ProcessPoolExecutor` task, order-preserving). `save_cn`/`load_cn` serialize it to a portable JSON `.cn` (intervals only, plus a `compile_time` and `(method,merge_budget,solver)` provenance header; extreme points are re-derived on demand by `CredalNetworkVertices`). `cn_metadata(path)` reads just that header for cache checks.
  - `compile_cn.py` — the **compiler CLI**: compiles a `.lcn` into its credal network and writes a `.cn` alongside the input (same basename). `python -m lcn.inference.marginal.cn.compile_cn foo.lcn [--n-jobs K] [--method linear|linear-tight] [--solver ipopt|scip] [--merge-budget N] [-o OUT.cn]`.
  - **`.cn` cache** — `CredalNetworkVertices.from_lcn(..., lcn_file=<path>, cache=True)` transparently loads a compiled `.cn` next to the `.lcn` instead of re-solving, **iff** its recorded `(method, merge_budget, solver)` match the requested build (`n_jobs`/`compile_time` are not part of the key); on a hit the reported `build_time` reuses the stored `compile_time` plus local vertex-enumeration. Disabled for CredalJT (`solve_families=False` carries no cacheable intervals). In the experiment runner this is the **`compile` algorithm** (`experiments/run_algorithm.py`, batch via `run_experiment.py`): it writes a `.cn` per instance recording `compile_time`; all other credal-net algorithms then auto-load it. Both runners surface `--n-jobs` (parallel per-family solves) and `--no-cache` (force recompute).
  - **`.vtx` cache (LRS vertices)** — the second, higher layer: `CredalNetworkVertices` also serializes the enumerated extreme points to a JSON `.vtx` (same basename: `foo.lcn`→`foo.cn`→`foo.vtx`) via `save_vtx`/`vtx_metadata`/`load_extreme_points`, recording `enumeration_time` (LRS wall-clock) and carrying `compile_time` through, keyed by the same `(method, merge_budget, solver)`. `from_lcn` checks the `.vtx` **before** the `.cn`: on a match it skips BOTH the per-family solves and LRS (loads vertices; rebuilds only the cheap chain-graph *structure* for `cnv.cn`/`bn_min`, recompiling the `.cn` structure-only if absent so a `.vtx` is usable standalone), and reports `build_time = compile_time + enumeration_time`. The LRS enumeration itself parallelizes over `--n-jobs` (`_build` shards each local credal set to a worker via a single-node `gum.CredalNet`; config-string keys are canonicalized to the full-net parent order so parallel output is bit-identical to serial — see `docs/lrs.tex`). Under parallel mode `cnv.credal_net` is `None` (no engine reads it). In the runner this is the **`enumerate` algorithm**; engines surface `loaded_vertices_from_cache`/`enumeration_time`.

**MAP inference** (`lcn/inference/map/`):
- `exact_map.py` — exact MAP/MMAP via DFS / Limited Discrepancy Search / Simulated Annealing. Scores configurations with `solve_exact_model` (defined here; query-bound NLP).
- `approx_map.py` — approximate MAP/MMAP via AMAP (uses `ariel.ArielInference`), approximate LDS, or approximate SA.

**Shared utilities** (`lcn/inference/utils/`):
- `common.py` — the shared toolbox: `make_ipopt(debug, mode)`, `check_consistency`, `make_conjunction`, `eval_indicator`/`dot` (Pyomo indicator/expression helpers), `build_truth_table`/`conjunction_indicator`/`lmc_constraint_groups`/`lmc_constraint_groups_vec` (LMC constraint construction), `find_feasible_points`/`optimize_marginal_slsqp` (SLSQP fallback), `check_consistency_product_witness`.
- `factor_graph.py` — `FactorGraph`, `VariableNode`, `FactorNode`, `FactorGraphEdge`.
- `convert_uai_to_lcn.py` — UAI → LCN conversion.

There is **no** `lcn/inference/legacy/` package — it was removed; the active engines are the ones above.

### Generator (`lcn/benchmarks/`)
- `generator.py` — `Generator` class for random LCN instances (chain/dag/polytree/tree/ktree/random topologies). `generate(..., consistency_mode="product"|"full")` rejection-samples consistent instances; `"product"` (default) uses a fast, sound product-distribution witness check that guarantees consistency for n ≤ 10.
- **`ktree` topology** — the maximal graph of treewidth exactly `k` (Arnborg–Proskurowski): start with a `(k+1)`-clique, then attach each new vertex to an existing `k`-clique. Rendered as a DAG (`_graph_ktree`) so every atom conditions on the **full conjunction** of its `k` clique-parents (via `_build_lcn(full_parents=True)`); the moralized/atom graph then has treewidth exactly `k`, letting inference scaling be studied against treewidth rather than raw `n`. The `k` parameter is a `generate(..., k=…)` arg and the `--k` CLI flag; requires `n ≥ k + 1`; `k=1` is a random rooted tree. **`ktree-fr`** is the same topology with extra marginals restricted to root atoms, so it has no non-family-realizable *sentence* — **but** for `k ≥ 2` a k-tree is not singly-connected (loopy after moralization), so unlike `tree-fr`/`polytree-fr` this does **not** make Credal VE / Interval BP exact (a *structural* non-realizability the extras placement can't fix; use CredalJT / `ExactInference(solver="global")`). CLI: `--family-realizable` on `gen_ktrees.py` or `--types ktree-fr` on `gen_benchmarks.py`.
- **`dag` topology** — a random DAG with **user-tunable in-degree** (`--max-parents`, default 2, no upper cap; `_graph_dag` guarantees DAG-ness by drawing parents only from earlier vertices in a random ordering). Unlike `ktree` (treewidth *exactly* `k` by construction), the DAG is **unstructured** and its moralized chain-graph treewidth is bounded *above* by `max_treewidth` (default 4, `--max-treewidth` CLI) via **rejection sampling**: candidates whose `moralized_treewidth(scopes)` (module-level min-fill helper, same algorithm as the engines' induced-width) exceeds the cap are resampled — this bound holds regardless of `max_parents` (higher fan-in just rejects more candidates). Accepted scopes still go through the standard product-witness consistency check, so a `dag` instance is both treewidth-bounded and consistent.
- `gen_benchmarks.py` (`--max-treewidth` for dag), `gen_chains.py`, `gen_polytrees.py`, `gen_trees.py`, `gen_ktrees.py` (`--k` treewidth), `gen_dags.py` (`--max-treewidth`, ≤2 parents), `gen_random.py`, `gen_easy.py` — benchmark-set generation scripts.

### LCN File Format
Programs are defined in `.lcn` files (see `examples/`). Syntax:
```
label: lb <= P(formula) <= ub
label: lb <= P(formula | formula) <= ub
```
Lines starting with `#` are comments. Each sentence requires a unique label.

## Key Patterns

- Inference algorithms follow a consistent pattern: instantiate with an `LCN` object, then call `run()` with algorithm-specific parameters.
- **LMC independence constraints** for `(X ⟂ Y | S)` must use the *joint* factorization `P(x,y,z)·P(z) = P(x,z)·P(y,z)` over **all** configurations of the Y block (built by `lmc_constraint_groups`/`lmc_constraint_groups_vec` in `utils/common.py`). A per-element/pairwise decomposition under-constrains the model when `|Y| ≥ 2` and gives wrong (too-loose) bounds. The vectorized variant is bit-identical but avoids the `2^|Y|·2^n` `Formula.evaluate` cost.
- **Solver robustness:** ipopt alone is unreliable on the dense joint-LMC system. `check_consistency` and `ExactInference`/`solve_exact_model` back ipopt with a two-phase SLSQP fallback (`find_feasible_points` → `optimize_marginal_slsqp`). Use `make_ipopt(mode="fast")` only for feasibility-style solves whose result is re-verified.
- The `docs/` directory contains LaTeX algorithm descriptions (`.tex`) and corresponding PDFs — consult these for mathematical details of specific algorithms.
- The `examples/` directory contains `.lcn` files (asia, alarm, cancer, smokers, new/new2/new3, factual); larger benchmark sets live under `benchmarks/{chain,dag,polytree,...}/`.
