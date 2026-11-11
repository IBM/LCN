# Plan — Interval Belief Propagation with evidence: credal sum-product on the factor graph

## 0. Where the current IVBP stands (`lcn/inference/marginal/cn/ivbp.py`)

- **Not sum-product.**
  - Variable→factor messages *intersect* the incoming intervals (max of lower, min of upper bounds).
    They do not multiply them.
  - Factor→parent messages are fixed to the vacuous [0, 1], so there is no diagnostic flow at all.
  - Evidence is clamped on the observed variable's outgoing messages only. It therefore reaches
    descendants but never ancestors: A→B with P(B|A) = .9/.1 and B=1 gives P(A=1) = 0.5 (exact:
    0.9). The paper states this limitation (`thm:ibp` "without evidence", Appendix "Scope: no
    evidence").
- **Cost.** A factor→child message enumerates every **vertex combination** of the CPT (the product
  of the per-row vertex counts) times every corner combination of the parent intervals. That is
  exponential in the number of parent configurations, whereas the rows are separable (§2.3).
- **Compound nodes.** These are multi-valued (e.g. alarm `C-D`). Per-atom bounds are obtained by an
  ipopt LP over the per-state box, run through Pyomo for every atom. This is slow and outer.
- **`method="variational"`.** A mean-field method over ≤ 500 sampled vertex combinations, unused by
  the harness and the paper.

## 1. Target: the classical sum-product algorithm, lifted to credal sets

Factor graph: one variable node per credal-network node V (cardinality 2^|atoms(V)|), one family
factor f_V(V, pa(V)) holding the local credal sets K(V | π), and one **evidence factor** e_V(V)
per node with observed atoms (the 0/1 indicator of the observed atom values over V's states).

Precise sum-product (Kschischang, Frey & Loeliger 2001; Pearl 1988), for a fixed choice P of local
conditionals:

    μ_{v→f}(x) = ∏_{g ∈ N(v)∖f} μ_{g→v}(x)
    μ_{f→v}(x) = Σ_{x_{N(f)∖v}} φ_f(x, x_{N(f)∖v}) ∏_{w ∈ N(f)∖v} μ_{w→f}(x_w)
    b_v(x) ∝ ∏_{g ∈ N(v)} μ_{g→v}(x)        (b_v = P(V | e) on a tree)

**Credal lift.** As P ranges over the strong extension, every message ranges over a *set* of
non-negative vectors. Messages only matter up to scale, so we propagate the set of **normalized**
messages, \hat μ = μ / Σ μ ∈ Δ(D_v), and represent it by its per-state interval [l, u] (the box
∩ simplex polytope). For binary variables this is exactly one interval [a⁻, a⁺] on μ(1)/Σμ.
This is the factor-graph form of Fagiuoli & Zaffalon's 2U, with π-messages (factor→child) and
λ-messages (factor→parent) unified into one rule.

## 2. Message computations

All three updates below are exact over the incoming sets because each output state's normalized value
is **linear-fractional (quasi-linear) in each incoming message and in each CPT row separately**. A
quasi-linear function attains its extrema over a product of polytopes at a vertex of each factor
(fix all blocks but one, optimize that one, and iterate).

### 2.1 Variable → factor (product)
- **Binary:**
  - g(a, b) = ab / (ab + (1−a)(1−b)) is non-decreasing in each argument.
  - The product interval is therefore [g(a⁻, b⁻), g(a⁺, b⁺)], folded over the incoming messages. This
    is closed form, with no enumeration.
  - Corners with 0/0 (a = 1, b = 0: contradictory evidence for that choice) are skipped. This is
    regular conditioning; if every corner is skipped, the message is "empty" (P(e) = 0).
- **Multi-valued:**
  - pairwise product of polytopes: enumerate the vertices of each box∩simplex, form normalized
    products, and take per-state min/max (the box of the result);
  - folding pairwise and re-boxing after each step is a sound outer approximation (inclusion is
    preserved by products).

### 2.2 Factor → variable (sum-product)
For target v ∈ scope(f_V) and each **corner** of the incoming messages (a vertex of each incoming
box∩simplex; for binary, an endpoint), compute min and max over the CPT rows of

    \hat μ_{f→v}(x) = m(x) / Σ_{x'} m(x'),   m(x) = Σ_{π, x_V consistent with v = x} P(x_V | π) w(π, x_V),

with w the product of the corner messages. For fixed corners, m(x) and Σ m are **linear and
separable in the rows** P(· | π).
- **Toward the child** (v = V): Σ m is constant (the parent corners are distributions), so this is
  a linear objective, and per-row min/max over the row's vertices is exact (one pass, no vertex
  combinations).
- **Toward a parent** (v = U_i): this is a ratio of separable linear forms. It is solved exactly by
  **Dinkelbach's** parametric method: max Σ_π ⟨α_π − t β_π, P(·|π)⟩ is separable per row (max over
  that row's vertices); iterate t ← ratio until a fixed point. This terminates in finitely many
  steps because the vertex sets are finite (typically 2–3 iterations).
  - For binary V the optimum is closed form: rows with u_i = 1 maximize, the others minimize
    P(1|π) according to the sign of λ(1) − λ(0). This is 2U's λ-message.

This replaces the vertex-combination enumeration: the cost is O(corners × rows × vertices per row),
instead of O(corners × ∏_π vertices).

The **evidence factor** sends the point message \hat δ (its normalized indicator) to its variable.
An observed atom of a compound node gives the indicator over the states with that atom value.

### 2.3 Beliefs and read-off
- b_v = the normalized product of all incoming factor messages, computed by §2.1 including e_V.
- **Binary:** [b⁻, b⁺] is P(V=1 | e).
- **Compound:** each atom's bound is the min/max of Σ_{s ∈ T_a} \hat b(s), taken directly over the
  vertex products of the last pairwise product. This is tighter than the per-state box followed by an
  LP, and it is numpy, not ipopt. The Pyomo LP code goes away.

## 3. Schedule and guarantees
- **Polytree factor graph** (the credal network's DAG is singly connected; the evidence factors are
  leaves):
  - **two-pass schedule** from a root: collect, then distribute;
  - every message is computed once from final inputs, so there is no iteration and no threshold;
  - all marginals come out in one run.
  - **Exact** for binary variables, with or without evidence (P̲(e) > 0): this is 2U's theorem.
    - Proof sketch: in a polytree the inputs of every update depend on disjoint parameter subsets,
      so they range independently over their (connected) attainable sets. The output is
      quasi-linear in each, so extremes sit at their endpoints; induction along the schedule gives
      exactness.
  - **Outer** (sound) for multi-valued variables, from the box∩simplex relaxation. Exact
    polytree inference is NP-hard for ternary variables (Mauá et al. 2014).
- **Loopy factor graph:** iterate (flooding, as L2U does) until max |Δ| < threshold or `n_iters`
  is reached. There is no guarantee in either direction; report `converged` and `iterations`.
- **Flags on the engine:** `self.exact` (polytree ∧ all binary), `self.singly_connected`,
  `self.converged`, `self.iterations`, `self.empty` (P(e) = 0 for every choice, so answers are
  vacuous).
- **No-evidence regression:** on binary polytrees the old and new IVBP must agree. Both are exact
  there: without evidence every λ-message is the neutral point 1/2, because each row sums to 1.

## 4. API / harness
- `IntervalBP.run(evidence=None, n_iters=100, threshold=1e-6, verbosity=1)`. It returns the same
  dict as today: compound-state bounds plus per-atom `(P(=0), P(=1))` vectors, and sets
  `marginals` and `singleton_marginals`.
- **Remove `method="variational"`** (`_run_variational`, mean-field over ≤ 500 sampled
  combinations), which nothing uses. This is a clean break, per the earlier cleanup policy: **confirm**.
- **`run_algorithm.py`:** drop `method="interval"` from the call; record `ivbp_exact`,
  `ivbp_converged` and `ivbp_iterations`.
- **`analyze_results.py`:** none needed. IVBP on loopy sets is still scored as an approximation.

## 5. Verification
1. **Exactness vs brute force:** reuse `_bruteforce` from `tests/test_cve.py` (enumerate every
   vertex product, condition, then min/max) on binary trees and polytrees with n = 5 (6 instances),
   with 0, 1 and 2 evidence atoms. Expect a match to 1e-9.
2. **Exactness vs CVE** (exact under evidence) on `tree_small_fr` and `polytree_small_fr`,
   n = 5–10, with 1–3 random evidence atoms.
3. **The diagnostic example:** A→B, B=1 gives P(A=1) = 0.9 (today: 0.5).
4. **Regression without evidence:** new vs old IVBP on binary polytrees (equal to 1e-9) and on loopy
   sets (report the differences; no guarantee either way).
5. **Compound node (alarm):** the IVBP interval contains the CVE interval, with and without evidence
   (outer).
6. **Zero-probability evidence corner cases** (a vertex P(B|A) = 0) match brute force.
7. **Loopy:** dag and ktree n = 10 converge; report the iteration counts and the error vs CVE/CJT.
8. **Timing:** old vs new on polytree n = 50/100 and dag n = 20 (expected faster: separable rows
   instead of vertex-combination enumeration).

## 6. Paper (method text only; experiments untouched until the reruns)
- **§4.4 IBP (main text):**
  - rewrite as credal sum-product: the three update rules, normalized interval messages, the
    two-pass schedule on polytrees and iteration on loopy graphs;
  - `thm:ibp` becomes "binary polytree, with or without evidence (P̲(e) > 0): exact", citing
    \citep{2u};
  - drop "downward only, not a method for queries with evidence";
  - `thm:ibp-cplx` becomes O(messages × corners × rows × vertices per row).
- **Appendix `sec-full-ibp`:**
  - replace "Scope: no evidence", vertex/corner combinations and Algorithm `alg:ibp` with the new
    rules and pseudocode;
  - rerun the alarm trace, adding an evidence case (the A→B diagnostic example now gives 0.9);
  - new exactness proof (quasi-linearity plus disjoint parameter sets on polytrees);
  - keep the "each hypothesis is used" paragraph, minus the evidence clause.
- **Intro (line ~128):** "IBP, exact on family-realizable binary polytrees **with or without
  evidence**".
- **Experiments-section mentions of "without evidence"** (line ~654) stay until the reruns; flag
  them in `revision_plan.md`.

## 7. Order of work
1. New message primitives (binary closed forms, multi-valued box∩simplex vertices, Dinkelbach),
   with unit tests against brute force on single factors.
2. The factor graph with evidence factors, the polytree detection and two-pass schedule, the loopy
   flooding schedule, and the read-off.
3. Remove the variational method and the Pyomo LP; update the harness call.
4. Tests §5; then the docs (CLAUDE.md, README) and the paper §6.

## Implementation status (2026-11-10)

Change from the plan (multi-valued variables). Messages are **sets of hull vertices** of the
normalized message vectors, not per-state boxes.
- Every update is projective in each input, and projective maps send convex hulls to convex hulls,
  so the vertex sets are lossless.
- Result: **exact on polytrees for compound variables as well** (the box version would only have been
  outer).
- λ-message: per corner, m(u) separates by row ⇒ a box ⇒ hull of the normalized box corners.
- π-message: per corner, a Minkowski sum of the weighted row-vertex sets. No vertex-combination
  enumeration.
- Binary messages are 2-vertex intervals (2U).

Done:
- `ivbp.py` rewritten: credal sum-product, evidence factors, alternating sweeps until no change
  (polytree, certifies the fixed point) or threshold/`n_iters` (loopy).
- Flags `exact`, `singly_connected`, `converged`, `iterations`. Compound atoms are read off the
  belief vertices.
- `method="variational"` and the Pyomo singleton LP were removed. The harness records
  `ivbp_exact`, `ivbp_converged`, `ivbp_iterations`.
- Tests: `tests/test_ivbp.py` (29), plus new examples `examples/ivbp_compound_{mid,parents}.lcn`.

Verification:

| case | reference | result |
|---|---|---|
| binary trees / polytrees n=5, 0–2 evidence atoms | brute force | ≤ 1e-12 |
| alarm (compound leaf C-D), evidence on C, D and elsewhere | brute force | ≤ 1e-12 |
| compound middle `B-C` (12-vertex rows), evidence above / below / on it | brute force | ≤ 1e-12 |
| two compound parents of one child, diagnostic evidence | CVE | ≤ 1e-12 |
| tree / polytree n=10–100 with evidence | CVE | ≤ 1e-12, ≤ 4 sweeps (diameter up to 18) |
| A→B diagnostic | — | 0.9 (was 0.5) |
| loopy, no evidence | old IVBP | equal to the digit |
| loopy, with evidence | — | converges in 3–11 sweeps, no guarantee (e.g. ktree n10: 5e-2 vs CVE) |

- Speed: ktree n8 0.02 s vs 0.38 s for the old code.
- Full suite: 297 passed.

Findings:
- `polytree_fr_n20_1` (and possibly other `polytree_*_fr` instances) compiles to a **loopy** credal
  network (cycle x19-x18-x15-x5-x2), so IVBP reports `exact = False` there. This is a generator /
  compilation question to check separately.
- `smokers.lcn`: the `linear` compile gives vacuous C|S rows (the sentences are not local under the
  S1-S2-S3 compound), so every bound is [0,1]. That is the compiled network, not IVBP. Same for D|B-C
  and E|B-C in `ivbp_compound_mid.lcn`.

Paper:
- §4.4 rewritten (credal sum-product, `eq:sum-product`). `thm:ibp`: polytree credal network, binary
  or compound, with or without evidence, P̲(e) > 0. `thm:ibp-cplx` restated; at most D+1 sweeps.
- Intro sentence updated; bib `kschischang2001factor` added.
- Appendix: factor graph, credal messages, update rules, Algorithm `alg:ibp`, the A→B→C trace (with
  C=1 evidence, checked against enumeration) and the compound example; the old 480-line trace is gone.
- New Lemma `lem:proj` and the exactness proof.
- Compiles: 56 pages, conclusion still on p. 11.
- Left for the reruns: experiments text line ~607 ("All queries are marginals without evidence, the
  setting of Theorems cve and ibp").

## Theorem / proof re-check (2026-11-10)

Fixes to `thm:ibp` and its proof:
- **Missing hypothesis.** Added "factorizing". `prop:suff-app` needs it, and with compound nodes the
  structure is no longer a DAG (the old binary theorem got it from the DAG structure).
- **Lemma `lem:proj` generalized** from a single linear map to multilinear maps over a product of
  hulls, with the one-argument-at-a-time induction written out.
- **Proof rewritten with explicit steps:**
  - parameter blocks per directed edge, the factor's own rows included; disjointness at factors and
    at variables (needs separate specification);
  - scale invariance: each update is linear in each input, so normalized outputs depend only on
    normalized inputs;
  - the zero-mass argument: P_θ(e) = <μ_{f→V}, μ_{V→f}> on a tree;
  - induction on message depth;
  - the full-box argument for the λ-message, with every corner attained by a real row choice, so no
    zero corner under P̲(e) > 0;
  - the belief step;
  - the fixed point and its certification.
- **Removed an unproven claim:** "if only P̄(e) > 0 the result is regular conditioning" is now
  "not claimed" (the empirical check on a zero-vertex instance matched, but that is no proof).
- **Complexity formula corrected:** O(Σ_V C_V (R_V e_V d + |pa(V)| 2^d)).
- **Empirical checks:**
  - a P̲(e) = 0 instance (zero vertices) matches brute force and CVE;
  - a 13-node zigzag collider polytree is exact vs CVE in 2–3 sweeps;
  - the maximum factor-graph diameter on the benchmarks is 36, and the sweep cap max(n_iters, 2n+2)
    is ≥ D+1 on every polytree benchmark.
