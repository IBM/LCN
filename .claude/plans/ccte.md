# Plan: Create `CredalCTE` class in `lcn/inference/marginal/ccte.py`

## Background: Bucket Tree Elimination

Standard bucket/variable elimination (as in CredalVE) computes bounds for a **single query variable** — it picks an elimination order, processes buckets from first to last, and extracts the marginal from the final bucket. To query a different variable, the entire elimination must be re-run with a different ordering.

**Bucket tree elimination** (also called cluster/junction tree propagation) extends this to compute **all marginals** in a single two-pass algorithm:

1. **Build the bucket tree**: Given an elimination ordering, each variable gets a bucket. When a bucket eliminates a variable and produces a message, that message goes to a specific "parent" bucket — the next bucket in the ordering that shares variables with the message's scope. This parent-child relationship forms a tree (the bucket tree).

2. **Upward pass (collect-to-root)**: Process buckets in elimination order (leaves to root). Each bucket combines its local potentials and incoming messages from children, marginalizes out its variable, and sends the result as a message to its parent bucket. This is identical to standard VE — it computes the marginal at the root.

3. **Downward pass (distribute-from-root)**: Process buckets in reverse elimination order (root to leaves). Each bucket receives a message from its parent containing "global" information (from the rest of the network), combines it with its local potentials and messages from other children, and produces a message sent to each child. After this pass, every bucket has enough information to compute its local marginal.

4. **Extract all marginals**: At each bucket, combine the upward messages from children, the downward message from parent, and local potentials. Marginalize to get the marginal for that bucket's variable.

## Relationship to CredalVE

- CredalVE.run() does **only the upward pass** — it's standard bucket elimination for a single query
- CredalCTE adds:
  - Building the bucket tree structure (tracking parent-child relationships between buckets)
  - The downward pass (sending messages from root back to leaves)
  - Marginal extraction at every bucket (not just the root)
- The `Potential` class from cve.py is reused unchanged (combine, marginalize, prune, epsilon_prune)

## Key differences from standard (precise) cluster tree:

In credal networks, potentials contain **sets of functions** (extreme points). The operations are:
- **Combine**: Cartesian product of function sets × pointwise multiply
- **Marginalize**: Sum out per function
- **Prune**: Remove dominated functions (to control cardinality)
- **Epsilon-prune**: Remove epsilon-dominated functions (relaxed pruning for FPTAS)
- Messages are **Potential** objects, not single functions

## Design

### Class: `CredalCTE`

```python
class CredalCTE:
    def __init__(self, cve: CredalVE)  # takes a built CredalVE instance
    def run(self, evidence: dict = {},
            elim_heuristic: str = "topological",
            epsilon: float = None,
            verbosity: int = 1) -> Dict[str, Tuple[np.ndarray, np.ndarray]]
    # Returns {var_name: (lower_bounds, upper_bounds)} for ALL variables
```

### Internal structure

**Bucket tree construction** (`_build_bucket_tree`):
- Input: potentials, elimination ordering
- For each variable Xi in the ordering:
  - Collect potentials mentioning Xi into bucket[Xi]
  - Determine parent bucket: the next variable in ordering that shares scope with the combined bucket
- Output: tree structure (parent pointers) and initial bucket assignments

**Upward pass** (`_collect`):
- Process buckets in elimination order
- At each bucket Xi:
  1. Combine all potentials in the bucket (local + messages from children)
  2. Marginalize out Xi → upward message λ_Xi
  3. Prune the message (exact or epsilon, depending on mode)
  4. Send λ_Xi to parent bucket
  5. Store the **combined potential before marginalization** (needed for downward pass)

**Downward pass** (`_distribute`):
- Process buckets in reverse elimination order (root first)
- At each bucket Xi:
  1. Receive message π_Xi from parent (for root, π = None or trivial)
  2. For each child Xj:
     - Combine: parent message π_Xi × local potentials × all child messages except Xj's
     - Marginalize out Xi → downward message π_Xj
     - Prune (exact or epsilon)
     - Send π_Xj to child Xj

**Extract marginals** (`_extract_marginals`):
- At each bucket Xi:
  1. Combine: parent message π_Xi × all child messages λ_Xj × local potentials
  2. This gives the full potential over Xi's scope
  3. Marginalize everything except Xi
  4. Normalize each function, take min/max for bounds

### Evidence handling
Same as CredalVE: add indicator potentials for evidence variables. Evidence variables are still eliminated (their marginals will be trivial).

### Epsilon Approximation for Bucket Tree Elimination

The epsilon approximation from CredalVE (Mauá et al. 2012) extends naturally to the two-pass bucket tree:

**Definition**: A function f is ε-dominated by g if g(y) ≥ f(y) - ε for all y. The ε-prune operation removes ε-dominated functions from a potential.

**Where epsilon-pruning is applied** (6 points — 3 in each pass):

1. **Upward pass** — at each bucket Xi after combining and marginalizing:
   - The upward message λ_Xi is epsilon-pruned before being sent to the parent
   - This controls the cardinality of upward messages (same as CredalVE.run_approx)

2. **Downward pass** — at each bucket Xi when computing messages to children:
   - The downward message π_Xj is epsilon-pruned before being sent to each child
   - This controls the cardinality of downward messages (NEW — not in CredalVE)

3. **Marginal extraction** — at each bucket before extracting bounds:
   - The combined potential (parent msg × child msgs × local) is epsilon-pruned
   - This reduces cost of the final marginalization step

**Impact on bounds:**
- Exact pruning (ε=0): The two-pass algorithm produces the same bounds as running CredalVE separately for each query variable. Bounds are tight (exact).
- Epsilon pruning (ε>0): Both upward and downward messages may lose some extreme functions, so the final bounds at each bucket may be wider (outer approximation). The error accumulates through both passes, so the effective error at any bucket is bounded by O(ε × depth), where depth is the bucket's distance from the root in the bucket tree.
- In practice, using the same ε for all pruning steps works well. For tighter control, one could use different ε values for upward vs. downward passes.

**FPTAS property:**
For credal networks of bounded treewidth w* and bounded variable cardinality k, the ε-approximate CTE runs in time polynomial in the input size, 1/ε, and the number of variables n. This extends the FPTAS result of Mauá et al. (2012) from single-query VE to all-marginals CTE.

### Method signature

```python
def run(self, evidence={}, elim_heuristic="topological", epsilon=None, verbosity=1):
    """
    Compute all marginals using bucket tree elimination.

    Args:
        evidence: {var: value} dict
        elim_heuristic: "topological" or "min-fill"
        epsilon: None for exact, >0 for epsilon-approximate
        verbosity: 0=silent, 1=summary, 2=detailed

    Returns:
        Dict mapping var_name -> (lower_bounds_array, upper_bounds_array)
    """
```

### `__main__` block
```python
cve = CredalVE(lcn=l)
cve.build(verbosity=0)
cte = CredalCTE(cve=cve)

# Exact all-marginals
all_marginals = cte.run(evidence={"B": 0, "E": 0})
for var, (lo, hi) in all_marginals.items():
    print(f"P({var}=1) in [{lo[1]:.6f}, {hi[1]:.6f}]")

# Epsilon-approximate all-marginals
all_marginals_approx = cte.run(evidence={"B": 0, "E": 0}, epsilon=0.01)
for var, (lo, hi) in all_marginals_approx.items():
    print(f"P({var}=1) in [{lo[1]:.6f}, {hi[1]:.6f}]")
```

## File structure
```
lcn/inference/marginal/ccte.py
├── class CredalCTE
│   ├── __init__(cve)
│   ├── run(evidence, elim_heuristic, epsilon, verbosity) -> dict
│   ├── _build_potentials(evidence) -> list of Potential
│   ├── _build_bucket_tree(potentials, elim_order) -> (buckets, parent, children)
│   ├── _collect(buckets, elim_order, parent, children, prune_fn) -> (up_msgs, combined)
│   ├── _distribute(elim_order, parent, children, up_msgs, combined, prune_fn) -> down_msgs
│   └── _extract_marginals(elim_order, cards, up_msgs, down_msgs, local_potentials, prune_fn) -> dict
└── __main__ block
```
