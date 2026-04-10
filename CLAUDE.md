# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LCN (Logical Credal Networks) is a Python library for probabilistic logic that combines propositional logic with imprecise probability bounds. An LCN program specifies constraints like `0.3 <= P(A | B) <= 0.5` over propositional formulas, defining a set of probability distributions. The library provides exact and approximate inference algorithms for marginal queries and MAP/MMAP explanations.

## Build & Development Commands

```bash
# Install in development mode (requires conda env with Python 3.10+)
pip install -e .

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_mixed_graph.py

# Run a specific test
pytest tests/test_mixed_graph.py::TestClassName::test_method -v
```

**External dependency:** The `ipopt` nonlinear solver is required at runtime. Install via `brew install ipopt` (macOS) or build from source via `coinbrew` (Linux).

## Architecture

### Core Model (`lcn/model.py`)
- `LCN` — the main model class. Load from `.lcn` files via `l = LCN(); l.from_lcn("file.lcn")`.
- `Sentence` — a probability-labeled sentence (Type1: `P(φ)`, Type2: `P(φ|ψ)`), each with lower/upper bounds.
- `Formula` — a parsed propositional logic formula over `Atom`s.
- The LCN builds a **primal graph**, **structure graph**, and derives **independence assumptions** via the Local Markov Condition (LMC).

### Parser (`lcn/parser.py`)
Parses propositional logic formulas. Operators: `and`/`&`, `or`/`|`, `xor`/`^`, `nand`/`/`, `not`/`!`. Uses `lcn/json_schema.py` for schema-based parsing.

### Independence (`lcn/independencies.py`, `lcn/mixed_graph.py`)
- `Independencies` / `IndependenceAssertion` — represents conditional independence statements derived from the LCN's graphical structure.
- `MixedGraph` — graph with both directed and bidirected edges, used for deriving Markov conditions.

### Inference (`lcn/inference/`)

All inference algorithms use **Pyomo** to formulate optimization problems solved by `ipopt`.

**Marginal inference** (`lcn/inference/marginal/`):
- `exact.py` — exact marginal inference via nonlinear optimization over the full joint
- `ariel.py` — ARIEL message-passing (approximate), operates on a factor graph
- `cve.py` — Credal Variable Elimination
- `ccte.py` — Credal Cluster Tree Elimination
- `ibp.py` — Interval Belief Propagation
- `approxlp.py` — ApproxLP approximate inference

**MAP inference**:
- `exact_map.py` — exact MAP/MMAP via DFS, Limited Discrepancy Search, or Simulated Annealing
- `approx_map.py` — approximate MAP/MMAP via AMAP (ARIEL-based), approximate LDS, or approximate SA

**Supporting modules**:
- `factor_graph.py` — `FactorGraph`, `VariableNode`, `FactorNode`, `FactorGraphEdge` used by message-passing algorithms
- `factorization.py` — factorization-based marginal inference
- `utils.py` — shared utilities: consistency checking (`check_consistency`), conjunction building, etc.
- `lrs.py`, `fraclp.py`, `fraclp2.py` — additional solver utilities

### Generator (`lcn/generator.py`)
`Generator` class for creating random LCN instances with configurable graph topologies (chains, DAGs, polytrees). Used for benchmarking and testing.

### LCN File Format
Programs are defined in `.lcn` files (see `examples/`). Syntax:
```
label: lb <= P(formula) <= ub
label: lb <= P(formula | formula) <= ub
```
Lines starting with `#` are comments. Each sentence requires a unique label.

## Key Patterns

- Inference algorithms follow a consistent pattern: instantiate with an `LCN` object, then call `run()` with algorithm-specific parameters.
- The `docs/` directory contains LaTeX algorithm descriptions (`.tex`) and corresponding PDFs — consult these for mathematical details of specific algorithms.
- The `examples/` directory contains `.lcn` benchmark files (asia, alarm, cancer, chains, DAGs, polytrees).
