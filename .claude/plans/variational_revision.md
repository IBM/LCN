# Plan: Revise docs/variational_credal.tex with detailed alarm network example

## Changes

### 1. Add tikz for factor graph (same as interval_bp.tex)

### 2. Add new Section 3: "Vertex Combinations in Detail"
- Formal definition matching interval_bp.tex but focused on how vc selects a precise BN
- A global vc selects one vertex per factor per parent config → a complete precise BN
- For alarm: total = 2 × 2 × 2⁴ × (4×11) = 2816 global combinations
- Each vc fixes P(B), P(E), P(A|B,E), P(CD|A) → standard BN

### 3. Replace Section 4 (Running Example) with detailed trace

**Structure:**
1. Factor graph figure + complete credal factor tables (reuse from interval_bp.tex)
2. Query: P(B=1), no evidence
3. **Vertex Combo 1 (all v1):**
   - Show the selected CPTs
   - Initialize q = uniform
   - Iteration 0: update q(B), q(E), q(A), q(CD) step by step
     - For q(B): show Case 1 (f_B contribution) and Case 2 (f_A contribution where B is parent)
     - Show the log_q computation, exponentiation, normalization
     - Result: q(B) = [1.000, 0.000], q(E) = [1.000, 0.000], q(A) = [0.000, 1.000]
   - Iteration 1: show how q updates shift
     - q(B) = [0.151, 0.849], q(E) = [0.849, 0.151]
   - Converged q(B) ≈ [0.238, 0.762]
   - Record: lower(B=1) = 0.762, upper(B=1) = 0.762

4. **Vertex Combo 2 (all v2):**
   - Show the selected CPTs (all P(A|·) = [1.0, 0.0])
   - Converges immediately: q(B) = [0.9, 0.1], q(E) = [0.95, 0.05], q(A) = [0.0, 1.0]
   - Record: update lower(B=1) = min(0.762, 0.1) = 0.1, upper unchanged

5. **Bound tracking across combos:**
   - Show how min/max accumulate across the 2816 (or sampled 500) combos
   - Final bounds for P(B=1)

6. Comparison with exact VE and IBP
