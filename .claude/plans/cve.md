# Plan: Create `lcn/inference/marginal/cve.py` — CredalVE class

## Context
Use the existing `Factorization` class to obtain local probability intervals from an LCN's chain graph decomposition, then assemble a pyAgrum `CredalNet` and use its LRS-based `intervalToCredal()` to enumerate extreme points of each credal set.

## Key Dependencies
- `lcn.inference.factorization.Factorization` — builds local factors with `[lobo, upbo]` intervals
- `lcn.model.LCN` — the input model
- `pyAgrum` — `BayesNet`, `CredalNet`, `LabelizedVariable`

## Design

### Class: `CredalVE`

```python
class CredalVE:
    def __init__(self, lcn: LCN)
    def build(self, verbosity=1) -> dict   # returns extreme_points dict
```

### `build()` method — step by step:

**Step 1: Run the factorization**
- Call the LCN's structure graph methods if not already built (build_primal_graph, build_structure_graph, is_chain_graph, process_chain_graph)
- Create `Factorization(lcn)` and call `.build()` to get the list of factors
- Each factor is a dict keyed by interpretation index, with entries: `{interpretation, scope, child, parents, lobo, upbo}`

**Step 2: Organize factor intervals into CPT format**
- For each factor, identify the child variable(s) and parent variable(s)
- For compound nodes (e.g. "A-B"), the child is multi-variable — BUT pyagrum requires single-variable nodes. For compound children, we'll need to handle them as a single variable with `2^k` states (where k = number of atoms in the compound)
- Group the factor entries by parent configuration to get: for each parent config, the lower/upper bounds on each child state
- Store as: `{node_name: {parent_config_index: (lower_list, upper_list)}}`

**Step 3: Build two pyAgrum BayesNets (bn_min, bn_max)**
- For each node in the simplified structure graph:
  - If single variable: add `LabelizedVariable(name, name, 2)` (binary)
  - If compound "A-B-...": add `LabelizedVariable(name, name, 2^k)` with k = number of atoms
- Add arcs from each parent node to child node (matching the simplified structure graph edges)
- Fill `bn_min.cpt(node)` with the lower bounds and `bn_max.cpt(node)` with the upper bounds
- The CPT filling order must match pyagrum's internal ordering (iterate over parent configs in pyagrum's Instantiation order)

**Step 4: Create CredalNet and run LRS vertex enumeration**
- `cn = gum.CredalNet(bn_min, bn_max)`
- `cn.intervalToCredal()` — this runs LRS internally to compute all extreme points

**Step 5: Extract and store extreme points**
- Parse `str(cn)` to extract the vertices for each node and each parent configuration
- The format is:
  ```
  NodeName:Labelized({0|1|...})
  <parent_config> : [[v1_0, v1_1, ...], [v2_0, v2_1, ...], ...]
  ```
- Store in `self.extreme_points`: `{node_name: {parent_config_str: list_of_vertex_lists}}`
- Also store `self.credal_net` for downstream inference

### Helper: `_parse_credal_net_vertices(cn) -> dict`
Parses `str(cn)` to extract vertices per node per parent config. Returns:
```python
{
    "A": {"<>": [[0.3, 0.7], [0.5, 0.5]]},
    "B": {"<A:0>": [[0.2, 0.8], [0.4, 0.6]], "<A:1>": [[0.3, 0.7], [0.5, 0.5]]}
}
```

### CPT ordering concern
pyAgrum orders parent instantiations in a specific way (based on `Instantiation` objects). The factorization iterates over `itertools.product([0,1], repeat=len(scope))` with scope = [child_vars, parent_vars]. We need to map from the factorization's ordering to pyAgrum's ordering when filling CPTs.

For each factor, group entries by parent config (fix parent bits, vary child bits):
- For a given parent config: collect `(child_val, lobo, upbo)` pairs
- Fill `bn_min.cpt(node)[parent_instantiation_slice]` with the lower bounds
- Fill `bn_max.cpt(node)[parent_instantiation_slice]` with the upper bounds

### `__main__` block
Load `examples/alarm.lcn`, build the CredalVE, print factors and extreme points.

## File Structure
```
lcn/inference/marginal/cve.py
├── _parse_credal_net_vertices(cn) -> dict
├── class CredalVE
│   ├── __init__(lcn)
│   ├── build(verbosity) -> dict
│   ├── self.factorization   — Factorization instance
│   ├── self.credal_net      — pyAgrum CredalNet
│   ├── self.extreme_points  — parsed vertices
│   ├── self.bn_min          — lower BayesNet
│   └── self.bn_max          — upper BayesNet
└── __main__ block
```
