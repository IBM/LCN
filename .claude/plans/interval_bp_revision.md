# Plan: Revise docs/interval_bp.tex with detailed example

## Overview
Replace the current running example section (Section 5) with a comprehensive step-by-step trace of IBP on the alarm.lcn network. Also add detailed explanations of vertex combinations (vc) and corner-point combinations (cc) with concrete notation.

## Changes

### 1. Add tikz package for factor graph figure
Add `\usepackage{tikz}` and `\usetikzlibrary{positioning,shapes}` to preamble.

### 2. Expand Section 3 (Factor-to-Variable) — rewrite vc and cc definitions
Before the running example, add a new subsection that formally defines:
- **Vertex Combination (vc)**: A vertex combination selects one extreme point per parent configuration. If factor f_i has parent configs π₁,...,πₘ with v₁,...,vₘ vertices respectively, then there are v₁×...×vₘ vertex combinations. Each vc = (j₁,...,jₘ) selects vertex j_k from config π_k. This gives a specific conditional distribution P_vc(X_i|π_k) for each parent config.
- **Corner-Point Combination (cc)**: For each "other" variable w (not the target), the incoming message is an interval [l_w, u_w]. A corner-point picks either the lower or upper bound for w. With |O| other variables, there are 2^|O| combinations. Each cc selects a specific bound vector q_w for each other variable w.
- The factor-to-variable message scans over ALL (vc, cc) pairs, computing the normalized marginal for the target variable under each, and takes the componentwise min/max.

### 3. Replace Section 5 (Running Example) with detailed alarm network trace

**Structure:**
1. Factor graph figure (tikz)
2. Table of all credal factors with all extreme points
3. Query: P(B=1), no evidence
4. Detailed message trace for iterations 0 and 1:
   - f_B → B: show vc enumeration (2 vertices, no other vars → 2 marginals)
   - f_E → E: same pattern
   - f_A → B: show vc enumeration (2⁴=16 combos across 4 parent configs), cc enumeration (2²=4 combos for A and E), compute each marginal step
   - f_A → A: similar detail
   - f_A → E: similar
   - f_CD → CD: show with 4+11=15 vertices
   - f_CD → A: show reverse message
5. Variable-to-factor tightening
6. Final marginal extraction
7. Comparison with VE

### Factor graph figure
```
[B] --- [f_B]
[B] --- [f_A] --- [A] --- [f_CD] --- [CD]
[E] --- [f_A]
[E] --- [f_E]
```
Square nodes for factors, circles for variables. Labels inside.

### Detailed f_B → B computation
- Scope: {B}, no other vars, no parents
- Vertex combination: vc ∈ {1, 2}
  - vc=1: P(B) = [0.80, 0.20] → normalized p = [0.80, 0.20]
  - vc=2: P(B) = [0.90, 0.10] → normalized p = [0.90, 0.10]
- No corner-point combinations (no other vars)
- Result: l = min([0.80,0.20], [0.90,0.10]) = [0.80, 0.10]
         u = max([0.80,0.20], [0.90,0.10]) = [0.90, 0.20]

### Detailed f_A → B computation (key example)
- Factor f_A: node=A, scope={A,B,E}, parents={B,E}
- Target: B, other vars O = {A, E}
- Vertex combinations: 2 vertices for each of 4 parent configs → 2⁴=16 vc
- Corner-point combinations: 2 other vars → 2²=4 cc
- For each (vc, cc): compute m(B=0), m(B=1), normalize
- Show 2-3 specific (vc,cc) computations in full detail
- Final bounds

### Detailed variable-to-factor tightening
- B → f_A: l_{B→f_A}(x) = max over f'≠f_A of l_{f'→B}(x) = l_{f_B→B}(x)
- Show that messages from f_B tighten the bounds for B sent to f_A
