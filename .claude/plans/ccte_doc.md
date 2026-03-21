# Plan: LaTeX documentation for CredalCTE

## Output
`docs/credal_cte.tex` — detailed description of Credal Cluster Tree Elimination

## Structure

### 1. Title/Abstract
Credal Cluster Tree Elimination: compute ALL marginals in one two-pass execution.

### 2. Preliminaries
- Bucket elimination recap (single query, upward pass only)
- Bucket tree: how the elimination order induces a tree
- Extension to two passes: collect (upward) + distribute (downward) = all marginals

### 3. Bucket Tree Construction
- Given elimination order, assign potentials to buckets
- Parent determined by scope propagation: after eliminating Xi, message scope = union of bucket scopes minus Xi, parent = next variable in ordering in that scope
- Example: alarm.lcn with order [E, B, A, C-D]

### 4. TikZ figure of bucket tree
Tree: E → B → A → C-D (root)
Show:
- Each bucket with its local potentials
- Upward messages (λ) along edges
- Downward messages (π) along edges

### 5. Upward Pass (Collect)
- Process E, B, A, C-D in order
- At each bucket: combine local + children msgs, marginalize, prune
- Detailed trace:
  - Bucket E: combine f_A (16 funcs) × f_E (2 funcs) = 32 funcs → marginalize E → λ_E over {A,B} (32 funcs)
  - Bucket B: combine f_B (2 funcs) × λ_E (32 funcs) = 64 → marginalize B → λ_B over {A} (59 funcs after pruning)
  - Bucket A: combine f_CD (44 funcs) × λ_B (59 funcs) → marginalize A → λ_A over {C-D} (2584 funcs)
  - Bucket C-D: root, λ_{C-D} over {} (scalar)

### 6. Downward Pass (Distribute)
- Process C-D, A, B, E in reverse order
- At root C-D: send to A combining local (none) → π_A = trivial (1 func)
- At A: send to B combining π_A + local f_CD → π_B over {C-D}
- At B: send to E combining π_B + local f_B → π_E over {C-D}

### 7. Marginal Extraction
- At each bucket: combine local + children up msgs + parent down msg → marginalize to single var → normalize → bounds
- Show results for all 4 variables

### 8. Epsilon Approximation
- epsilon_prune at all 3 stages (upward, downward, extraction)
- Comparison table: exact vs epsilon for the alarm network

### 9. References
