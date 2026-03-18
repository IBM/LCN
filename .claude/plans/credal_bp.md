# Plan: Belief Propagation for Credal Networks in CredalVE

## Literature Analysis

### 2U (exact, binary, polytrees only)
- Propagates interval-valued messages on polytree DAGs with binary variables
- Messages are scalar intervals `[l, u]` representing bounds on likelihood ratios
- Exploits the binary structure to avoid enumerating vertices: the min/max of products of intervals can be computed in closed form
- **Limitation**: binary variables only, polytrees only

### L2U (approximate, binary, any topology)
- Loopy version of 2U: iterates interval message-passing on general graphs until convergence
- Same message format as 2U (scalar intervals)
- Convergence guaranteed (lower bounds monotonically increase, upper bounds decrease)
- **Limitation**: binary variables only

### GL2U (approximate, multi-valued via binarization)
- Handles multi-valued variables by binarizing them: a k-valued variable is replaced by a cluster of ⌈log2(k)⌉ binary variables connected in a chain
- Runs L2U on the expanded binary network
- **Limitation**: binarization introduces auxiliary structure, can be lossy, awkward for our use case

### ApproxLP (Antonucci et al. 2015)
- LP-based loopy propagation that works natively with multi-valued variables
- Each message is an **interval vector**: for variable X with k states, the message from factor f to variable X is a pair of vectors `(lower[k], upper[k])` bounding P(X=x) from f's perspective
- At each variable node: tighten bounds by intersecting incoming messages (take componentwise max of lowers, min of uppers)
- At each factor node: solve a local LP to compute the tightest bounds on each neighbor variable given the factor's constraints and the incoming messages from other neighbors
- **Natively supports multi-valued variables**

### Modified BP for LCNs (Qian, Marinescu et al. 2021 — this codebase!)
- Same structure as ApproxLP but adapted for LCN factor graphs
- Variable-to-factor messages: interval tightening (Eqs 20-21 in the LCN paper)
- Factor-to-variable messages: solve a local NLP (since LCN constraints can be nonlinear due to conditional probabilities)
- Already implemented in `lcn/inference/approx_marginal.py`

## Proposed Algorithm: `run_bp`

A generalized interval-propagation BP that works directly on the credal network's DAG structure (from `self.bn_min`) with multi-valued variables. This is essentially **ApproxLP adapted to our credal network** built by `build()`.

### Message Format
For variable X with cardinality k:
- Message is a pair `(lower: np.ndarray[k], upper: np.ndarray[k])` bounding P(X=x) from the sender's perspective
- Initialized to `(zeros(k), ones(k))` — vacuous

### Graph Structure
Use the DAG from `self.bn_min`. Each variable X has:
- Parents `Pa(X)` connected by arcs in the DAG
- Children `Ch(X)`
- A local credal set `K(X|Pa(X))` given by `self.extreme_points[X]`

We use a **factor graph** derived from the DAG: one factor per variable X containing its CPT credal set `K(X|Pa(X))`, connected to X and all of Pa(X).

### Message Updates

**Variable-to-factor** (X → f):
For each state x of X:
```
l_{X→f}(x) = max over all neighboring factors f' ≠ f of l_{f'→X}(x)
u_{X→f}(x) = min over all neighboring factors f' ≠ f of u_{f'→X}(x)
```

**Factor-to-variable** (f → X), where f is the factor for variable Y with CPT K(Y|Pa(Y)):
Solve a local LP/optimization over the extreme points of K(Y|Pa(Y)):
- For each extreme point vertex `v` of the local credal set and each parent config:
  - Combine the vertex with the incoming interval messages from other neighbors
  - Compute the marginal bounds on X
- Take the tightest (inner) bounds across all combinations

Concretely, for the factor associated with node Y:
- The factor's scope is {Y} ∪ Pa(Y)
- Given incoming messages from all scope variables except X, for each vertex of K(Y|Pa(Y)):
  - Enumerate extreme combinations of the incoming intervals (corner points)
  - Compute the joint, marginalize to X
  - Track min/max for each state of X

### Evidence Handling
For evidence X=e:
- Set the message bounds for X to indicator: `lower[e] = upper[e] = 1.0`, all others `= 0.0`
- These bounds are never updated (clamped)

### Convergence
- Iterate until the maximum change in any message is below a threshold, or max iterations reached
- Lower bounds monotonically increase, upper bounds monotonically decrease (guaranteed)

### Marginal Extraction
After convergence, for each variable X:
```
lower(X=x) = max over all factors f of l_{f→X}(x)
upper(X=x) = min over all factors f of u_{f→X}(x)
```
Then normalize: `P(X=x) ∈ [l(x)/sum(u), u(x)/sum(l)]` (using Bayes bounds).

### Variational Extension
Add an optional **mean-field variational** mode:
- Instead of propagating intervals, maintain a factored distribution `q(X) = ∏ q_i(X_i)` that approximates the lower/upper bounds
- At each iteration, update each `q_i` by minimizing/maximizing the expected log-probability over the credal set's extreme points
- This produces tighter inner bounds (complementing the outer bounds from interval BP)
- Implemented as a separate mode: `method="variational"` in `run_bp()`

## Method Signatures

```python
def run_bp(self, query: str, evidence: dict = {},
           n_iters: int = 100, threshold: float = 1e-6,
           method: str = "interval",  # "interval" or "variational"
           verbosity: int = 1):
    """
    Belief propagation for credal networks.

    Args:
        query: query variable name
        evidence: {var: value} dict
        n_iters: max iterations
        threshold: convergence threshold
        method: "interval" for interval BP (outer approx),
                "variational" for mean-field variational (inner approx)
        verbosity: verbosity level
    """
```

## Implementation Steps

### Step 1: Factor graph construction from credal network DAG
Build a factor graph from `self.bn_min`:
- One factor per variable (containing its credal set)
- Factor scope = {variable} ∪ parents
- Store the extreme points for each factor (from `self.extreme_points`)

### Step 2: Message data structures
- `msg_var_to_factor[var][factor]` = `(lower, upper)` arrays
- `msg_factor_to_var[factor][var]` = `(lower, upper)` arrays
- Initialize all to `(0, 1)` vectors; clamp evidence variables

### Step 3: Iterative message passing
For each iteration:
1. Update all variable-to-factor messages (tightening)
2. Update all factor-to-variable messages (local optimization over extreme points)
3. Check convergence (max delta)

### Step 4: Factor-to-variable message computation
For factor f (associated with node Y, scope = {Y} ∪ Pa(Y)), computing message to neighbor X:
- Collect incoming messages from all scope variables except X
- For each extreme point vertex of K(Y|Pa(Y)):
  - For each combination of extreme bounds from incoming messages:
    - Build a local joint distribution
    - Marginalize to X
    - Update running min/max

### Step 5: Variational extension
Mean-field approach:
- Maintain `q[var]` = probability distribution for each variable
- Initialize to uniform
- At each iteration, for each variable X:
  - For each extreme point of the credal sets involving X:
    - Compute expected energy under current q for other variables
    - Update q[X] proportionally
  - Normalize q[X]
- Track min/max across iterations for bounds

### Step 6: Extract and store results
Same as `run()`: `self.lower_bound`, `self.upper_bound`, `self.lower_bounds`, `self.upper_bounds`

## File Changes
- Add `run_bp()` method to `CredalVE` class in `cve.py`
- Update `__main__` block to demonstrate BP inference
