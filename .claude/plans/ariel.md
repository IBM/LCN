# Plan: Create `lcn/inference/marginal/ariel.py`

## Context
The existing `lcn/inference/approx_marginal.py` implements belief-propagation-style approximate inference over an LCN's factor graph. The `solve_factor_subproblem` function builds a local NLP for each factor-to-variable message. The new `ariel.py` will follow the same overall BP algorithm but improve the NLP construction using the same patterns established in `exact.py` (`_eval_indicator`, `_dot`, numpy vectors), add proper independence constraints from the LCN's Local Markov Condition, and detect when lower > upper bounds (infeasibility).

## Key Files
- **Source of logic:** `lcn/inference/approx_marginal.py` (solve_factor_subproblem, Message, Marginal, ApproximateInference)
- **Helpers to reuse:** `lcn/inference/marginal/exact.py` (`_eval_indicator`, `_dot`)
- **Factor graph:** `lcn/inference/factor_graph.py` (FactorGraph, FactorNode, VariableNode, FactorGraphEdge)
- **Model:** `lcn/model.py` (LCN, Sentence, SentenceType, Formula)
- **Independencies:** `lcn/independencies.py` (Independencies, IndependenceAssertion)
- **Target:** `lcn/inference/marginal/ariel.py` (new file)

## Implementation

### Step 1: Import shared helpers from exact.py

Import `_eval_indicator` and `_dot` from `lcn.inference.marginal.exact` to avoid code duplication. Also import factor graph classes, model classes, utils.

### Step 2: Rewrite `solve_factor_subproblem` as `_solve_local_nlp`

Same signature as the original (takes VariableNode `n`, FactorNode `f`, neighbors list, incoming messages dict, sense). Improvements:

1. **Precompute interpretations once** as list of dicts (same as exact.py pattern)
2. **Use `_eval_indicator` + `_dot`** for all constraint/objective construction instead of manual `for j in range(N)` loops
3. **Sentence constraints** — same Type1/Type2 handling as exact.py but scoped to the factor's variables
4. **Incoming message constraints** — for each neighboring variable `m`, use `_eval_indicator` on `Formula(m)` then constrain with Lagrange relaxation variables (same `model.v` approach as original)
5. **Independence constraints** — use the LCN's independencies (passed as parameter) filtered to variables in the factor's scope. For each applicable assertion, build the same `P(x,S,T)*P(S) = P(x,S)*P(S,T)` constraints using `_eval_indicator`/`_dot`. This replaces the buggy pairwise independence in the original (line 130 uses `A1` twice instead of `A1` and `A2`).
6. **Objective** — same penalty-based objective as original, built via `_dot`

### Step 3: Implement `Message` class (same as original)

Keep the same `Message` class with `update_variable_to_factor` and `update_factor_to_variable` methods. The factor-to-variable update calls `_solve_local_nlp` instead of `solve_factor_subproblem`.

### Step 4: Implement `Marginal` class (same as original)

Same `Marginal` class with `variable`, `lower_bound`, `upper_bound` and `update` method.

### Step 5: Implement `ApproximateInference` class with bound checking

Same BP loop as original (`run` method) with one addition:

- **After collecting marginals**, iterate over all variables and check if `lower_bound > upper_bound`. If so, print a warning: `"WARNING: variable {name} has lb={lb:.4f} > ub={ub:.4f} (infeasible)"` and set a `self.feasible = False` flag. Otherwise `self.feasible = True`.
- Also check per-iteration: after each factor-to-variable message update, if `msg.lower_bound > msg.upper_bound`, log a warning at verbosity > 0.

### Step 6: `__main__` block

Same as `approx_marginal.py`: load `examples/asia.lcn`, check consistency, run approximate inference with default params.

## File Structure

```
lcn/inference/marginal/ariel.py
├── _solve_local_nlp(n, f, neighbors, incoming, independencies, sense, debug) -> Tuple
├── class Message
│   ├── __init__(edge, type)
│   ├── set_lower_bound / set_upper_bound / set_bounds
│   ├── update_variable_to_factor(fg, factor_messages)
│   └── update_factor_to_variable(fg, variable_messages, independencies, debug)
├── class Marginal
│   ├── __init__(variable, lower_bound, upper_bound)
│   └── update(incoming_messages)
├── class ApproximateInference
│   ├── __init__(lcn)
│   └── run(n_iters, threshold, debug, evidence, verbosity)
└── __main__ block
```

## Bug Fixes from Original
1. **Line 130 bug:** `val2 = sum(A1[i]*model.p[i] ...)` should use `A2` not `A1` — fixed by using proper `_eval_indicator` for each formula
2. **Independence constraints:** Original uses naive pairwise independence between all neighbor pairs. New version uses the LCN's actual independence assertions filtered to the factor's scope.
3. **Bound inversion detection:** Original silently allows lb > ub. New version flags this.
