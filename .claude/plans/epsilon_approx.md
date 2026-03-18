# Plan: Epsilon-approximate Credal Variable Elimination

## Source
Mauá & Cozman (2020), "Thirty years of credal networks", Section 5.2, page 144:
> "By relaxing the maximality criteria to allow elements to be slightly dominated by some other element, Mauá et al. showed that the same algorithm can be adapted to provide a provably good approximation algorithm, that is, a procedure that outputs a solution whose error is at most a given value ε > 0."

## What changes

### 1. Add `epsilon_prune(epsilon)` method to `Potential`
The exact `prune()` removes function `f` if there exists `g` with `g(y) ≥ f(y)` for all `y` (strict domination). The epsilon variant relaxes this: remove `f` if there exists `g` with `g(y) ≥ f(y) - ε` for all `y`. This means functions that are "almost dominated" are also removed, controlling the potential cardinality growth at the cost of bounded approximation error.

### 2. Add `run_approx()` method to `CredalVE`
Same structure as `run()` — builds potentials, creates elimination order, does bucket elimination — but calls `epsilon_prune(epsilon)` instead of `prune()` at each elimination step. Takes the same arguments as `run()` plus `epsilon: float`.

### 3. Add `epsilon` argument to `run()`
When `epsilon` is provided (not None), `run()` delegates to `run_approx()`. When `epsilon=None` (default), exact pruning is used as before.

## Changes to cve.py

### `Potential.epsilon_prune(epsilon)`
```python
def epsilon_prune(self, epsilon: float) -> 'Potential':
    # Remove f_i if there exists f_j such that
    # f_j(y) >= f_i(y) - epsilon for all y
    # (i.e., f_i is epsilon-dominated by f_j)
```

### `run_approx(query, evidence, epsilon, elim_heuristic, verbosity)`
Separate function that mirrors `run()` but uses `epsilon_prune()`. This is a standalone method that operates on the same `self.extreme_points`, `self.bn_min` data built by `build()`.

### `run()` modified signature
```python
def run(self, query, evidence={}, elim_heuristic="topological",
        epsilon=None, verbosity=1):
    if epsilon is not None:
        return self.run_approx(query, evidence, epsilon,
                               elim_heuristic, verbosity)
    # ... existing exact code ...
```

### `__main__` block
Add demonstration of epsilon approximation with different epsilon values.
