# Plan: Create `Generator` class in `lcn/generator2.py`

## Context
The existing `lcn/generator.py` has standalone functions (`make_lcn_random1`, `make_lcn_dag`, `make_lcn_polytree`, etc.) that share significant duplicated code. Each function:
1. Creates a graph structure (scopes)
2. Generates formulas using `make_formula` (limited to `and`/`or` with positive/negative literals)
3. Generates random probability bounds `[val-ε, val+ε]`
4. Builds an LCN with `Sentence` objects
5. Optionally checks consistency

### Limitations of the existing generator:
- `phi` is always a single atomic literal (`x{child}` or `!x{child}`)
- `psi` uses only `and`/`or` connectors, not `xor`, `not`, or nested formulas
- No support for multi-variable phi formulas (e.g. `(A or B)`)
- No chain graph topology
- No `max_vars_per_sentence` parameter
- Formula generation doesn't use `not`, `xor`, or `nand`/`nor`
- Consistency checking is by flag but doesn't retry on inconsistency

## Design

### New file: `lcn/generator2.py`

### Class: `Generator`

```python
class Generator:
    def __init__(self, seed: int = 42)
    def generate(
        self,
        num_vars: int,
        graph_type: str,        # "random", "dag", "polytree", "chain"
        num_instances: int = 1,
        max_vars_per_sentence: int = 3,
        num_extras: int = 0,
        epsilon: float = 0.3,
        max_retries: int = 100,
        verbosity: int = 1
    ) -> List[LCN]
```

### Internal methods:

**`_make_graph(num_vars, graph_type)`** → `List[List[int]]` (scopes)
Generates the graph structure. Returns a list of scopes where each scope is `[parent_ids..., child_id]`.

- `"random"`: Random graph (can have cycles). Chain of variables + random extra edges.
- `"dag"`: Random DAG. Variables ordered, each child picks parents from higher-ordered vars only.
- `"polytree"`: Random polytree. Start from chain, swap edges while maintaining tree property.
- `"chain"`: Chain graph (DAG where each node has at most 1 parent). Simple linear chain with random ordering.

**`_make_random_formula(variables, max_vars, rng)`** → `str`
Generate a random propositional logic formula over a subset of the given variables using connectors: `and`, `or`, `not`, `xor`. Recursively builds formulas:
- Base case: pick a variable, optionally negate it (`!x`)
- Recursive case: pick a binary connector (`and`, `or`, `xor`), generate left and right sub-formulas
- Depth-limited to avoid overly complex formulas
- Uses parentheses for proper precedence

**`_make_sentence(scope, rng, epsilon, max_vars)`** → `Sentence`
Given a scope `[parents..., child]`:
- For Type 1 (no parents): generate phi formula from child variable(s)
- For Type 2 (with parents): generate phi from child atoms, psi from parent atoms
- Generate random bounds: `val = rng.uniform()`, `lo = max(0, val-ε)`, `up = min(1, val+ε)`
- The phi formula can involve multiple atoms (up to max_vars_per_sentence) using random connectors

**`_build_lcn(scopes, num_vars, rng, epsilon, max_vars, extras)`** → `LCN`
Build the LCN from scopes, create sentences, add extra P(x) sentences.

**`_check_and_build(lcn)`** → `bool`
Build primal graph, structure graph, LMC. Run consistency check. Return True if consistent.

### Formula generation strategy:
- Pick `k` variables from the available set (1 ≤ k ≤ min(max_vars, len(available)))
- For k=1: return atom or negated atom
- For k=2: pick connector from {and, or, xor}, optionally negate each atom
- For k≥3: recursively split into sub-formulas joined by random connectors
- Wrap in parentheses as needed

### Consistency handling:
- Generate an LCN instance
- Check consistency using `check_consistency(lcn)`
- If inconsistent, retry with new random structure/bounds (up to max_retries)
- Only return consistent instances

### `__main__` block:
```python
gen = Generator(seed=42)
for graph_type in ["random", "dag", "polytree", "chain"]:
    instances = gen.generate(num_vars=5, graph_type=graph_type, num_instances=3)
    for i, lcn in enumerate(instances):
        print(f"{graph_type} instance {i}: {len(lcn.sentences)} sentences")
        print(lcn)
```

## Key differences from existing generator.py:
1. **Class-based** instead of standalone functions
2. **Richer formulas**: uses `not`, `xor` in addition to `and`, `or`; supports multi-variable phi
3. **`max_vars_per_sentence`** parameter controls formula complexity
4. **Chain graph** topology added
5. **Consistency retry loop**: generates until consistent (up to max_retries)
6. **Single `generate()` entry point** instead of 4 separate functions
7. **Unified formula builder** that can create nested formulas with all standard connectors
