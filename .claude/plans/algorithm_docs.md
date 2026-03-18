# Plan: LaTeX Documentation for Credal Inference Algorithms

## Output
Four LaTeX files in `docs/`:
1. `docs/credal_ve.tex` — Credal Variable Elimination (exact)
2. `docs/credal_ve_epsilon.tex` — Epsilon-approximate Credal Variable Elimination
3. `docs/interval_bp.tex` — Interval Belief Propagation
4. `docs/variational_credal.tex` — Mean-Field Variational Inference for Credal Networks

## Structure of Each Document
Each LaTeX file follows the same template:

1. **Title and abstract** (1 paragraph summary)
2. **Preliminaries** — credal networks, strong extension, extreme points, notation
3. **Algorithm description** — detailed prose explanation of each step
4. **Pseudocode** — formal `algorithm` environment with `algorithmic` package
5. **Running example** — step-by-step trace on the alarm.lcn network (5 variables: A, B, C, D, E)
6. **Complexity analysis** — time/space complexity
7. **References**

## Content Details per Document

### 1. `credal_ve.tex` — Credal Variable Elimination

**Preliminaries:**
- Credal network definition: DAG G, local credal sets K(Xi|Pa(Xi))
- Strong extension: convex hull of all products of local extreme points (Theorem 2 from Mauá & Cozman 2020)
- Potential: finite set of non-negative functions over joint domain
- Operations: product (Cartesian × pointwise multiply), sum-marginal (sum out per function), pruning (remove dominated)

**Algorithm:**
- Input: credal network extreme points, query variable Z, evidence Y=y
- Step 1: Create potentials from extreme points (one potential per node)
- Step 2: Add evidence indicator potentials
- Step 3: Compute elimination order (topological or min-fill heuristic)
- Step 4: Bucket elimination — for each variable Xi in order: collect, combine, marginalize, prune
- Step 5: Combine remaining potentials
- Step 6: Normalize and extract lower/upper bounds

**Pseudocode:**
```
Algorithm: CredalVE(extreme_points, Z, y)
Input: extreme_points per node, query Z, evidence y
Output: [P_lower(Z=z|Y=y), P_upper(Z=z|Y=y)]
1. For each node Xi: create potential φ_Xi from ext K(Xi|Pa(Xi))
2. For each evidence variable Y_j=y_j: add indicator potential δ_{y_j}
3. Compute elimination order o = (X_1, ..., X_{n-1}) excluding Z
4. For each X_i in o:
   a. Γ_i = {φ : X_i in scope(φ)}
   b. ψ = product of all φ in Γ_i
   c. ψ' = marginalize X_i from ψ
   d. ψ'' = prune(ψ')
   e. Replace Γ_i with {ψ''} in the potential set
5. Combine all remaining potentials into φ_final(Z)
6. For each f in φ_final.functions:
   prob = f / sum(f)
   lower[z] = min(lower[z], prob[z])
   upper[z] = max(upper[z], prob[z])
7. Return [lower, upper]
```

**Running example:** Alarm network with B, E, A, C-D.
- Show the extreme points for each node
- Trace through elimination of E, then A, then C-D
- Show how potentials combine and shrink
- Final bounds for P(B=1) = [0.10, 0.20]

### 2. `credal_ve_epsilon.tex` — Epsilon-Approximate CVE

**Key difference from exact:** Replace exact pruning with epsilon-pruning.

**Epsilon-dominance definition:**
Function f is ε-dominated by g if g(y) ≥ f(y) - ε for all y.

**Pseudocode:** Same as CVE but step 4d uses epsilon_prune(ε) instead of prune().

**Running example:** Same alarm network, show how with ε=0.1:
- More functions get pruned at each step
- Final bounds may be wider but computation faster
- Compare timing and bound quality for ε ∈ {0.001, 0.01, 0.1}

**FPTAS property:** Error bounded by ε, polynomial in input and 1/ε for bounded treewidth.

### 3. `interval_bp.tex` — Interval Belief Propagation

**Preliminaries:**
- Factor graph from credal network DAG
- Messages as interval vectors: for variable X with k states, message is (lower[k], upper[k])

**Algorithm:**
- Input: factor graph with extreme points, query Z, evidence y, n_iters, threshold
- Step 1: Initialize all messages to vacuous [0, 1]^k
- Step 2: Clamp evidence to indicator intervals
- Step 3: Iterate until convergence:
  - Variable→Factor: tighten by intersecting incoming messages
  - Factor→Variable: enumerate vertex × corner-point combos, compute normalized marginals, track min/max
- Step 4: Extract marginal bounds from final messages

**Pseudocode:**
```
Algorithm: IntervalBP(factors, Z, y, n_iters, threshold)
Initialize msg_v2f, msg_f2v to ([0,...,0], [1,...,1])
Clamp evidence variables
For iter = 1, ..., n_iters:
  delta = 0
  // Variable-to-factor
  For each variable v, each neighbor factor f:
    l_{v→f}(x) = max_{f'≠f} l_{f'→v}(x)
    u_{v→f}(x) = min_{f'≠f} u_{f'→v}(x)
  // Factor-to-variable
  For each factor f, each neighbor variable v:
    For each vertex combo vc, corner combo cc:
      Compute marginal m_v from factor using vc and cc
      l_{f→v} = min(l_{f→v}, m_v / sum(m_v))
      u_{f→v} = max(u_{f→v}, m_v / sum(m_v))
  If delta < threshold: break
Return bounds from intersecting all f→Z messages
```

**Running example:** Alarm network, trace 2-3 iterations showing message evolution.

### 4. `variational_credal.tex` — Mean-Field Variational for Credal Networks

**Preliminaries:**
- Mean-field approximation: q(X) = ∏ q_i(X_i)
- Coordinate ascent: update each q_i to minimize KL divergence
- For credal networks: optimize over extreme point selections

**Algorithm:**
- For each combination of extreme points (one per local credal set):
  - Fix the extreme points → precise BN
  - Run standard mean-field coordinate ascent until convergence
  - Record q[query]
- Track min/max of q[query] across all explored combinations

**Pseudocode:**
```
Algorithm: VariationalCredal(factors, Z, y, n_iters, threshold)
lower = [1,...,1], upper = [0,...,0]
For each vertex combination vc (sampled or exhaustive):
  Initialize q[v] = uniform for all v
  Clamp q[evidence] = indicator
  For iter = 1, ..., n_iters:
    For each variable v (not evidence):
      log_q[v] = 0
      For each factor f involving v:
        log_q[v] += E_{q\v}[log P(f | vc)]
      q[v] = normalize(exp(log_q[v]))
    If converged: break
  lower = min(lower, q[Z])
  upper = max(upper, q[Z])
Return [lower, upper]
```

**Running example:** Alarm network, show 2-3 vertex combinations and how q[B] evolves.

## LaTeX Packages Used
- `algorithm2e` or `algorithmicx` for pseudocode
- `amsmath`, `amssymb` for math
- `booktabs` for tables
- `tikz` for DAG diagrams (optional)
- Standalone `article` class documents

## Running Example Data
From alarm.lcn:
- Variables: B (binary), E (binary), A (binary), C-D (4-valued compound)
- Sentences: s1-s5
- Extreme points from the CredalVE build (already computed)
- VE result: P(B=1) ∈ [0.10, 0.20], P(A=1|B=0,E=0) ∈ [0.00, 0.044]
