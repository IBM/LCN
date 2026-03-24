# Plan: Implement ApproxLP in `lcn/inference/marginal/approxlp.py`

## Algorithm Overview (from Mauá & Cozman 2020, Section 5.2.3)

ApproxLP is a **linearization-based** approximate inference algorithm for credal networks. It reformulates the marginal inference problem as a multilinear program (MLP) and then solves it via iterative linearization:

1. The marginal inference P(Z=z|Y=y) for a credal network is expressed as:
   ```
   min/max f'_m(z)
   s.t.  f'_m(Z) = t · f_m(Z)          (Charnes-Cooper normalization)
         sum_z' f'_m(z') = 1
         f_i(sep_i) = sum_{x_i} prod{f in bucket_i}   (VE constraints)
         p(Xi|πi) ∈ K(Xi|πi)            (credal set constraints)
   ```

2. This is a multilinear program because the VE constraints involve products of the optimization variables p(Xi|πi) and the intermediate functions f_i.

3. **ApproxLP linearizes this** by fixing all local distributions except one at a time:
   - Initialize all p(Xi|πi) to some feasible point (e.g., center of each credal set)
   - At each iteration, pick one variable Xi
   - Fix all distributions except p(Xi|πi)
   - The resulting program becomes **linear** in p(Xi|πi)
   - Solve the LP to get the optimal p(Xi|πi)
   - Replace the incumbent and repeat
   - Stop when no improvement or max iterations reached

4. This is essentially **coordinate descent on the multilinear objective**, where each step is an LP.

## Simplified Practical Implementation

For our credal network (with extreme points from LRS), the implementation simplifies:

Instead of solving LPs over the continuous credal set constraints, we can **enumerate extreme points** for each variable and pick the one that optimizes the objective. This is because:
- Each credal set K(Xi|πi) is a polytope whose extreme points we already have
- The LP optimum over a polytope is always at a vertex
- So "solve LP" = "try each extreme point and pick the best one"

### Algorithm: ApproxLP for Credal Networks

```
Input: credal network with extreme points, query Z, evidence Y=y
Output: [P_lower(Z=z|y), P_upper(Z=z|y)]

1. Initialize: for each node Xi and parent config πi, pick the center
   (average) of the extreme points as the current distribution p(Xi|πi).

2. Run standard VE with these fixed distributions to get the initial
   P(Z=z|y) = numerator / denominator.

3. Iterate for max_iters:
   a. For each variable Xi (in some order):
      - For each parent config πi:
        - For each extreme point v of K(Xi|πi):
          - Temporarily set p(Xi|πi) = v
          - Re-run VE (or incrementally update) to get new P(Z=z|y)
          - If this improves the objective (lower for min, higher for max):
            keep this vertex; otherwise revert.
      - If no improvement was made for Xi, continue to next variable.
   b. If no improvement was made for any variable, stop (converged).

4. Return the final P(Z=z|y).

Run the above twice: once with sense=min, once with sense=max.
```

### Optimization: Incremental VE Update

A full VE re-run for each extreme point is expensive. The key insight from Antonucci et al. (2013, 2015) is that when only one local distribution changes, most of the VE computation stays the same. Specifically:

- The VE computation can be decomposed into a tree of intermediate factors
- Changing p(Xi|πi) only affects the factors that involve Xi
- We can recompute only the affected path in the elimination tree

For simplicity, the initial implementation will re-run full VE each time (correct but slower). An incremental update can be added as an optimization later.

## Design

### Class: `ApproxLP`

```python
class ApproxLP:
    def __init__(self, cve: CredalVE)
    def run(self, query: str, evidence: dict = {},
            n_iters: int = 50, verbosity: int = 1)
```

### Internal methods:

**`_build_factors()`** → same as IBP, builds factor graph from extreme points

**`_init_distributions(factors, cards)`** → initialize each p(Xi|πi) to center of credal set (average of extreme points)

**`_run_ve_with_fixed(distributions, query, evidence, cards, bn, sense)`** → run VE with fixed point distributions (single BN), return P(Z=z|y)

**`_coordinate_descent(factors, query, evidence, cards, bn, sense, n_iters)`** → the main ApproxLP loop: iterate over variables, try each extreme point, keep improvements

### Following existing patterns:
- Takes `CredalVE` instance (same as IntervalBP and CredalCTE)
- Uses `self.cve.extreme_points` and `self.cve.bn_min` for structure
- Stores results in `self.lower_bound`, `self.upper_bound`, `self.lower_bounds`, `self.upper_bounds`

### `__main__` block
```python
cve = CredalVE(lcn=l)
cve.build(verbosity=0)
alp = ApproxLP(cve=cve)
alp.run(query="B", evidence={}, verbosity=1)
alp.run(query="A", evidence={"B": 0, "E": 0}, verbosity=1)
```

## File structure
```
lcn/inference/marginal/approxlp.py
├── class ApproxLP
│   ├── __init__(cve)
│   ├── run(query, evidence, n_iters, verbosity)
│   ├── _build_factors() -> (cards, factors)
│   ├── _init_distributions(factors, cards) -> dict
│   ├── _eval_objective(distributions, query, evidence, cards, factors, sense) -> float
│   └── _coordinate_descent(factors, query, evidence, cards, sense, n_iters) -> float
└── __main__ block
```

## Key differences from IntervalBP:
- **IntervalBP**: propagates interval messages, explores extreme points via corner-point enumeration at each factor
- **ApproxLP**: fixes all distributions to specific points, runs exact inference, then optimizes one distribution at a time via coordinate descent over extreme points
- ApproxLP produces **inner** bounds (may not reach the true extremes), IntervalBP produces **outer** bounds (may be too wide)
