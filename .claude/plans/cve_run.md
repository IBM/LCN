# Plan: Implement `run()` method in `CredalVE` for marginal inference

## Algorithm Source
The Algebraic Variable Elimination algorithm from Mauá & Cozman (2020), "Thirty years of credal networks", Algorithm 2 (page 144), and the simplified CVE from Marinescu et al. (NeurIPS 2023), Algorithm 1.

## Overview
For marginal inference (no MAP variables — all variables except the query are sum variables), CVE simplifies to:
1. Initialize potentials from the local credal set extreme points
2. Eliminate variables one-by-one (by summation) in a chosen order
3. Extract lower/upper bounds from the final potential

## Key Data Structures

### Potential
A potential `φ(Y)` over variables `Y` is a **set of non-negative real-valued functions** on `Y`. Since our variables are discrete with known domains, each function is stored as a numpy array indexed by the joint configurations of `Y`.

For the **simplified marginal case** (no MAP, just computing P(query | evidence)), we use the simpler formulation from the NeurIPS paper (Definition 3):
- A potential `φ(Y)` = set of functions `{p₁, p₂, ...}` where each `pₖ` maps joint configurations of `Y` to non-negative reals
- **Product**: `φ(Y) · ψ(Z) = {p · q : p ∈ φ, q ∈ ψ}` — pointwise multiplication over joint domain `Y ∪ Z`
- **Sum-marginal**: `Σ_Z φ(Y) = {Σ_Z p(Y) : p ∈ φ}` — sum out `Z` from each function
- **Pruning (max)**: `max φ(Y)` = keep only non-dominated elements under componentwise `≥`

### For lower/upper bound computation
Following Algorithm 2 from the paper:
- Each potential element is a **pair** `(p, q)` where `p` computes `P(Z≠z, Y=y)` and `q` computes `P(Z=z, Y=y)`
- The query potential `φ'_Z = {(1-δ_z, δ_z)}` splits the computation
- **Upper bound**: `max{q/(p+q) : (p,q) ∈ ψ_final}`
- **Lower bound**: either use `min` pruning, or compute `1 - P_upper(Z≠z|Y=y)`

Given the complexity of the pair-based approach, and since for our use case the extreme points are already available, I'll implement the **simpler enumeration approach**: enumerate all combinations of extreme points (one per local credal set per parent config), run standard VE for each combination, and take min/max. This is exact for separately-specified credal networks (Theorem 4 from the paper).

## Simpler Approach: Enumerate-and-Eliminate

For small networks (which is our target), the number of extreme point combinations is manageable.

**Algorithm:**
1. For each node, get its extreme point CPTs (from `self.extreme_points`)
2. Generate all combinations of extreme point selections (one vertex per node per parent config)
3. For each combination, assemble a standard BN (precise probabilities), compute P(query | evidence) using standard VE
4. Track min and max across all combinations

**But this is exponential in the number of credal sets.** A better approach:

## Implemented Approach: Bucket-based VE with potentials

Follow Algorithm 1 from the NeurIPS paper, simplified for pure marginal inference (all variables are sum variables, no MAP variables):

### Step 1: Initialize potentials
For each variable `Xi`:
- `φ_Xi` = set of functions from `ext(K(Xi|Pa(Xi)))` — each extreme point of the local credal set defines one function over `{Xi} ∪ Pa(Xi)`

For evidence variable `Y=y`:
- Add indicator potential `φ'_Y = {δ_y}` — a single function that is 1 when `Y=y`, 0 otherwise

### Step 2: Create elimination ordering
Eliminate all variables except the query `Z`. Use a simple ordering: reverse topological order of the DAG, excluding `Z`, placing `Z` last.

### Step 3: Bucket elimination
For each variable `Xi` in the elimination order:
1. Collect all potentials `Γ_Xi` that have `Xi` in their scope
2. Combine them: `ψ = Π{φ ∈ Γ_Xi}` (pointwise product of all potential sets — Cartesian product of function sets, then pointwise multiply)
3. Marginalize: `ψ' = Σ_Xi ψ` (sum out `Xi` from each function)
4. Prune: remove dominated elements (optional, for efficiency)
5. Place `ψ'` back into the appropriate bucket

### Step 4: Extract bounds
After all variables except `Z` are eliminated, combine remaining potentials to get final potential over `Z`.
- `P_lower(Z=z|Y=y) = min{f(z)/Σ_z' f(z') : f ∈ final_potential}`
- `P_upper(Z=z|Y=y) = max{f(z)/Σ_z' f(z') : f ∈ final_potential}`

## Implementation Details

### Potential representation
Each function in a potential is a numpy array. The potential's scope (set of variable names) determines the array shape.

```python
class Potential:
    scope: List[str]          # variable names
    cards: List[int]          # cardinality of each variable
    functions: List[np.ndarray]  # each array has shape = tuple(cards)
```

### Product of two potentials
`φ(Y) · ψ(Z)`:
- New scope = `Y ∪ Z`
- For each pair `(p ∈ φ, q ∈ ψ)`: broadcast-multiply `p` and `q` over the joint domain
- Result has `|φ| * |ψ|` functions

### Sum-marginal
`Σ_Xi φ(Y)`:
- New scope = `Y \ {Xi}`
- For each `p ∈ φ`: sum `p` along the axis corresponding to `Xi`

### Pruning
`max φ(Y)`: remove function `p` if there exists `q ∈ φ` such that `q(y) ≥ p(y)` for all `y` (componentwise dominance).

### Evidence
For each evidence variable `Y_j = y_j`, create a potential with a single indicator function over `{Y_j}` that is 1 at `y_j` and 0 elsewhere.

### Elimination ordering
Use min-degree heuristic or reverse topological order. For simplicity, use reverse topological order of the DAG (from `bn_min`).

## Method Signature

```python
def run(self, query: str, evidence: dict = {}, verbosity: int = 1):
    """
    Compute lower and upper bounds on P(query_var=1 | evidence).

    Args:
        query: str - name of the query variable (node in the credal network)
        evidence: dict - {variable_name: value} for observed variables
        verbosity: int - verbosity level

    Sets:
        self.lower_bound: float
        self.upper_bound: float
    """
```

## File Changes
- Only modify `lcn/inference/marginal/cve.py`: add `Potential` class and `run()` method to `CredalVE`
- Update `__main__` block to demonstrate `run()`
