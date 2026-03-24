# Plan: LaTeX documentation for ApproxLP

## Output
`docs/approxlp.tex`

## Structure (following same outline as other docs)

### 1. Title/Abstract
ApproxLP: Approximate Credal Network Inference via Iterative Linearization

### 2. Preliminaries
- Multilinear programming formulation of credal inference (from the survey Sec 5.2.1)
- The linearization idea: fix all distributions except one → LP → iterate
- Since credal sets have known extreme points, LP over polytope = vertex selection

### 3. Algorithm Description
- Initialize distributions to center of credal sets
- Coordinate descent: for each variable, each parent config, each extreme point:
  - Temporarily set distribution to that vertex
  - Evaluate P(Z|y) via standard VE with fixed precise BN
  - If objective improves, keep the vertex
- Run twice: sense=min, sense=max
- Return inner bounds

### 4. Detailed Pseudocode
Algorithm 1: ApproxLP
- Input: extreme points, query Z, evidence y, max_iters
- Output: [P_lower(Z=z|y), P_upper(Z=z|y)]
- Show initialization, outer loop, inner loop, VE evaluation, convergence check

### 5. Factor Graph Figure (same TikZ as other docs)

### 6. Credal Factor Tables (same as other docs)

### 7. Detailed Running Example: alarm.lcn

**Query 1: P(B=1), no evidence**
- Show initial distributions (centers):
  P(B) = [0.85, 0.15], P(E) = [0.925, 0.075], etc.
- Initial eval: P(B=1) = 0.15
- Iteration 0 (min): try B v1=[0.8,0.2]→P(B=1)=0.2 (worse), try B v2=[0.9,0.1]→P(B=1)=0.1 (better, keep)
  Try E v1, v2: no change. Try A vertices: no change (B is a root node so only f_B matters).
- Converged: lower=0.1
- Same for max: upper=0.2
- Result: P(B=1) ∈ [0.1, 0.2]

**Query 2: P(A=1|B=0,E=0)**
- With evidence B=0, E=0, only parent config (0,0) matters for A
- Initial: center P(A|(0,0)) = [0.9778, 0.0222]
- Try v1=[0.9556, 0.0444] → P(A=1)=0.0444 (max keeps)
- Try v2=[1.0, 0.0] → P(A=1)=0.0 (min keeps)
- Result: [0.0, 0.0444]

### 8. Comparison with other methods
Table: VE, epsilon-VE, IBP, Variational, ApproxLP
- Bounds quality, speed, bound type (inner vs outer)

### 9. Properties
- Inner bounds (guaranteed to be within true interval)
- Convergence: monotonic improvement at each step, guaranteed to converge
- Speed: each iteration is standard VE on precise BN (very fast)
- Limitation: may get stuck at local optimum (coordinate descent, not global)

### 10. Complexity
- Per iteration: n_factors × max_vertices × VE cost
- VE cost per precise BN: O(n k^w*) where w* is treewidth
- Total: O(n_iters × n × V_max × n × k^w*)

### 11. References
