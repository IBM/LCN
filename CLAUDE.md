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

**MAP inference** (`lcn/inference/map/`):
- `exact_map.py` — exact MAP/MMAP via DFS / Limited Discrepancy Search / Simulated Annealing. Scores configurations with `solve_exact_model` (defined here; query-bound NLP).
- `approx_map.py` — approximate MAP/MMAP via AMAP (uses `ariel.ArielInference`), approximate LDS, or approximate SA.

**Shared utilities** (`lcn/inference/utils/`):
- `common.py` — the shared toolbox: `make_ipopt(debug, mode)`, `check_consistency`, `make_conjunction`, `eval_indicator`/`dot` (Pyomo indicator/expression helpers), `build_truth_table`/`conjunction_indicator`/`lmc_constraint_groups`/`lmc_constraint_groups_vec` (LMC constraint construction), `find_feasible_points`/`optimize_marginal_slsqp` (SLSQP fallback), `check_consistency_product_witness`.
- `factor_graph.py` — `FactorGraph`, `VariableNode`, `FactorNode`, `FactorGraphEdge`.
- `convert_uai_to_lcn.py` — UAI → LCN conversion.

There is **no** `lcn/inference/legacy/` package — it was removed; the active engines are the ones above.

### Generator (`lcn/benchmarks/`)
- `generator.py` — `Generator` class for random LCN instances (chain/dag/polytree/random topologies). `generate(..., consistency_mode="product"|"full")` rejection-samples consistent instances; `"product"` (default) uses a fast, sound product-distribution witness check that guarantees consistency for n ≤ 10.
- `gen_benchmarks.py`, `gen_chains.py`, `gen_polytrees.py`, `gen_random.py` — benchmark-set generation scripts.

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
