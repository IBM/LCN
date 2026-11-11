# coding=utf-8
# Copyright 2023-present the International Business Machines.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Pair potentials for credal variable elimination (Maua & Cozman, "Thirty years
# of credal networks", IJAR 126 (2020), Sec. 5.2, Algorithm 2 and eqs. 7-8).
#
# A potential phi(Y) is a finite set of PAIRS (p, q) of nonnegative functions of
# Y; each pair is one choice of extreme points of the local credal sets. For a
# query Z = z the query potential is {(1 - delta_z, delta_z)}: at the end p sums
# the mass of Z != z and q the mass of Z = z (together with any evidence), so
# P(Z = z | e) = q / (p + q) for that choice of extreme points.
#
# Storage is dense: all pairs of a potential are two arrays P, Q of shape
# (m, *cards(scope)), so every operation (product, sum-out, evidence masking,
# pruning) is a vectorized numpy call rather than a Python loop over pairs.

from typing import Dict, List, Sequence

import numpy as np

# Pairs whose p and q both vanish carry no information (they assign zero
# probability to the evidence) and are dropped, which is regular conditioning.
# The test is EXACT (== 0): evidence indicators produce exact zeros, whereas a
# positive threshold also discards legitimately tiny P(e) (60 evidence atoms on
# tree_fr_n100_1 gave P(e) ~ 1e-32 and made every answer vacuous). Underflow is
# avoided instead by rescaling query-free pairs (``normalize``).
_ZERO_TOL = 0.0
# Rows compared per numpy call in the dominance test (bounds memory use).
_BLOCK_ELEMS = 2_000_000


def _row_view(X: np.ndarray) -> np.ndarray:
    """Each row of a 2-D array as one opaque scalar (a void view of its
    bytes), so whole rows can be hashed / compared in a single pass."""
    X = np.ascontiguousarray(X)
    if X.dtype.kind == "f":
        X = X + 0.0                       # canonical +0.0 for -0.0
    return X.view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))).ravel()


def _first_unique_rows(X: np.ndarray) -> np.ndarray:
    """Sorted indices of the first occurrence of each distinct row of X.
    Hash-based (pandas factorize when available; pandas is an optional
    extra, the fallback sorts the row bytes as one key): O(m d), unlike
    ``np.unique(axis=0)``, whose lexicographic row sort dominated CVE's run
    time (10.4 of 13 s on ktree_fr_k3_n10_1 with evidence)."""
    inv = _row_classes(X)
    first = np.full(inv.max() + 1, X.shape[0], dtype=np.int64)
    np.minimum.at(first, inv, np.arange(X.shape[0]))
    return np.sort(first)


def _row_classes(X: np.ndarray) -> np.ndarray:
    """Class id (0..k-1) of every row of X; equal rows share an id."""
    if X.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    if X.shape[1] == 0:
        return np.zeros(X.shape[0], dtype=np.int64)
    v = _row_view(X)
    try:
        import pandas as pd
    except ImportError:                    # pandas is an optional extra
        _, inv = np.unique(v, return_inverse=True)
        return inv.reshape(-1).astype(np.int64)
    codes, _ = pd.factorize(np.array([bytes(r) for r in v], dtype=object))
    return np.asarray(codes, dtype=np.int64)


class PairPotential:
    """A set of (p, q) function pairs over ``scope``, stored as two arrays of
    shape ``(m, *[cards[v] for v in scope])``."""

    __slots__ = ("scope", "cards", "P", "Q")

    def __init__(self, scope: Sequence[str], cards: Dict[str, int],
                 P: np.ndarray, Q: np.ndarray):
        self.scope = list(scope)
        self.cards = cards
        self.P = P
        self.Q = Q

    # ------------------------------------------------------------------
    @classmethod
    def from_functions(cls, scope, cards, functions: np.ndarray):
        """A CPT potential {(f, f)}: both components equal (p = q)."""
        F = np.asarray(functions, dtype=float)
        return cls(scope, cards, F, F.copy())

    @property
    def m(self) -> int:
        return self.P.shape[0]

    def __len__(self):
        return self.m

    # ------------------------------------------------------------------
    def _aligned(self, new_scope):
        """P, Q reshaped/transposed so their trailing axes follow ``new_scope``
        (size-1 axes for variables of new_scope not in self.scope)."""
        perm = [self.scope.index(v) for v in new_scope if v in self.scope]
        shape = [self.m] + [self.cards[v] if v in self.scope else 1
                            for v in new_scope]
        axes = [0] + [1 + i for i in perm]
        P = np.transpose(self.P, axes).reshape(shape)
        Q = np.transpose(self.Q, axes).reshape(shape)
        return P, Q

    def product(self, other: "PairPotential") -> "PairPotential":
        """{(p r, q s) : (p, q) in self, (r, s) in other} over the union scope.
        One broadcast per component: (m1, 1, ...) x (1, m2, ...) -> (m1 m2, ...)."""
        new_scope = list(self.scope) + [v for v in other.scope
                                        if v not in self.scope]
        cards = dict(self.cards)
        cards.update(other.cards)
        P1, Q1 = self._aligned(new_scope)
        P2, Q2 = other._aligned(new_scope)
        dims = tuple(cards[v] for v in new_scope)
        m = self.m * other.m
        # (m1, 1, ...) * (1, m2, ...) broadcasts every size-1 axis to full size.
        P = (P1[:, None] * P2[None, :]).reshape((m,) + dims)
        Q = (Q1[:, None] * Q2[None, :]).reshape((m,) + dims)
        return PairPotential(new_scope, cards, P, Q)

    def sum_out(self, var: str) -> "PairPotential":
        """Sum-marginal of both components over ``var``."""
        if var not in self.scope:
            return self
        ax = 1 + self.scope.index(var)
        return PairPotential([v for v in self.scope if v != var], self.cards,
                             self.P.sum(axis=ax), self.Q.sum(axis=ax))

    def mask(self, var: str, keep: np.ndarray) -> "PairPotential":
        """Multiply both components by the 0/1 indicator ``keep`` over ``var``'s
        states (evidence on an atom of a, possibly compound, node)."""
        ax = 1 + self.scope.index(var)
        shape = [1] * self.P.ndim
        shape[ax] = len(keep)
        k = np.asarray(keep, dtype=float).reshape(shape)
        return PairPotential(self.scope, self.cards, self.P * k, self.Q * k)

    def query(self, var: str, target: np.ndarray) -> "PairPotential":
        """Multiply in the query potential {(1 - t, t)} over ``var``, with ``t``
        the 0/1 indicator of the query event over ``var``'s states."""
        ax = 1 + self.scope.index(var)
        shape = [1] * self.P.ndim
        shape[ax] = len(target)
        t = np.asarray(target, dtype=float).reshape(shape)
        return PairPotential(self.scope, self.cards, self.P * (1.0 - t),
                             self.Q * t)

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------
    def _rows(self):
        return (self.P.reshape(self.m, -1), self.Q.reshape(self.m, -1))

    def _keep(self, idx: np.ndarray) -> "PairPotential":
        return PairPotential(self.scope, self.cards, self.P[idx], self.Q[idx])

    def drop_zero_and_duplicates(self) -> "PairPotential":
        """Remove pairs with p = q = 0 and exact duplicate pairs."""
        if self.m == 0:
            return self
        P, Q = self._rows()
        nz = (P.max(axis=1) > _ZERO_TOL) | (Q.max(axis=1) > _ZERO_TOL)
        idx = np.flatnonzero(nz)
        if idx.size == 0:
            return self._keep(idx)
        PQ = np.concatenate([P[idx], Q[idx]], axis=1)
        first = _first_unique_rows(PQ)
        if first.size == idx.size:
            return self if idx.size == self.m else self._keep(idx)
        return self._keep(idx[first])

    def prune(self, sense: str = "both") -> "PairPotential":
        """
        Pareto pruning of eqs. (7)-(8), vectorized (exact).

        For ``sense="max"`` a pair (p, q) is removed when another pair (r, s)
        satisfies r <= p and s >= q on every coordinate (the max of
        q / (p + q) can never be attained by it). ``"min"`` uses the reverse
        order. ``"both"`` removes a pair only when it is dominated in BOTH orders
        -- the union of the two fronts is kept, so one elimination yields both
        the lower and the upper bound exactly (dominance in either order is
        preserved by products with a common nonnegative pair and by sums).
        """
        pot = self.drop_zero_and_duplicates()
        m = pot.m
        if m <= 1:
            return pot
        P, Q = pot._rows()
        d = P.shape[1]
        if sense in ("max", "both"):
            dom_max = _dominated(P, Q, d)
        if sense in ("min", "both"):
            dom_min = _dominated(Q, P, d)
        if sense == "max":
            keep = ~dom_max
        elif sense == "min":
            keep = ~dom_min
        else:
            keep = ~(dom_max & dom_min)
        return pot._keep(np.flatnonzero(keep))

    def normalize(self) -> "PairPotential":
        """Scale every pair of a query-free potential (p == q) to max |f| = 1.

        Sound because every downstream use of a query-free pair is invariant to
        scaling it by a positive constant: products and sums carry the factor
        into the final pair (c p, c q), whose ratio q / (p + q) is unchanged,
        and the hull prune works on rays. It keeps long products of
        probabilities (many evidence atoms) inside floating-point range."""
        if self.m == 0:
            return self
        P, _ = self._rows()
        mx = P.max(axis=1)
        mx[mx == 0] = 1.0
        shape = (self.m,) + (1,) * (self.P.ndim - 1)
        F = self.P / mx.reshape(shape)
        return PairPotential(self.scope, self.cards, F, F.copy())

    def is_query_free(self) -> bool:
        """True iff every pair has p == q, i.e. the query potential has not yet
        been multiplied into this potential."""
        return self.P is self.Q or np.array_equal(self.P, self.Q)

    def prune_query_free(self) -> "PairPotential":
        """Pruning for a potential that does not contain the query (p == q).

        Pairs are rescaled (``normalize``), zero rows and exact duplicates are
        removed (rescaled copies now coincide), then every function that is
        not a VERTEX of the convex hull of the potential's functions. This is
        exact: once the other potentials' choices are fixed, the final ratio
        q / (p + q) is linear-fractional (with a positive denominator) in this
        potential's function f, because f enters p and q linearly. A
        linear-fractional function attains its extrema over conv{f_1..f_m} at a
        vertex, so non-vertex functions can never be needed for either bound;
        being homogeneous of degree 0, it is also invariant to rescaling f, so
        the test is done on the rows normalized to sum 1 (extreme rays).
        (Pareto dominance, the pair order, removes nothing here: for p == q a
        pair is dominated only by an identical one.)
        """
        pot = self.normalize().drop_zero_and_duplicates()
        if pot.m <= 2:
            return pot
        P, _ = pot._rows()
        # q / (p + q) is homogeneous of degree 0 in f, so only the RAY of f
        # matters: normalize rows to sum 1 (drops one dimension; positive
        # multiples become duplicates) and keep the extreme rays.
        N = P / P.sum(axis=1, keepdims=True)
        _, first = np.unique(np.round(N, 15), axis=0, return_index=True)
        first = np.sort(first)
        keep = first[hull_vertices(N[first])]
        if keep.size == pot.m:
            return pot
        return PairPotential(pot.scope, pot.cards, pot.P[keep],
                             pot.P[keep].copy())

    def quotient(self, alpha: float) -> "PairPotential":
        """
        The alpha-quotient Psi/alpha of Maua, de Campos & Zaffalon (IJAR 2012,
        Sec. 2.4-2.5): one pair per alpha-equivalence class.

        Two pairs are alpha-equivalent when, on every coordinate of p and of q,
        both entries are zero or both are positive with the same
        floor(log_alpha(.)). Rows are first scaled to sum(p + q) = 1 (the
        posterior ratio q / (p + q) is invariant to scaling a pair by a positive
        constant, so rescaled copies collapse), then keyed on the integer grid.

        Guarantee: every removed pair (p, q) has a kept (r, s) in its class with
        r <= alpha p and s >= q / alpha coordinatewise (after the scaling), so
        any downstream odds <a, q> / <b, p> is preserved up to a factor alpha^2
        in either direction (see CredalVE.run for how the factors compose).

        Representatives are REAL pairs (no synthesis), so the final bounds are
        inner bounds. Per class the member with the largest and the one with
        the smallest share sum(q) / sum(p + q) are kept (one row if they
        coincide, e.g. for a query-free potential where p == q), favouring the
        upper and the lower bound respectively.
        """
        m = self.m
        if m <= 1 or alpha <= 1.0:
            return self
        P, Q = self._rows()
        tot = P.sum(axis=1) + Q.sum(axis=1)
        ok = tot > _ZERO_TOL
        if not ok.all():
            idx = np.flatnonzero(ok)
            if idx.size == 0:
                return self._keep(idx)
            return self._keep(idx).quotient(alpha)
        X = np.concatenate([P, Q], axis=1) / tot[:, None]
        with np.errstate(divide="ignore"):
            L = np.floor(np.log(X) / np.log(alpha))
        # zeros (log = -inf) get a sentinel below every finite key
        L[~np.isfinite(L)] = np.iinfo(np.int64).min
        K = L.astype(np.int64)
        inv = _row_classes(K)
        if inv.max() + 1 == m:
            return self
        share = Q.sum(axis=1) / tot
        order = np.lexsort((share, inv))        # by class, then by share
        cls = inv[order]
        first = np.flatnonzero(np.r_[True, cls[1:] != cls[:-1]])
        last = np.r_[first[1:] - 1, m - 1]
        lo_i, hi_i = order[first], order[last]
        two = share[hi_i] > share[lo_i] + 1e-12  # a second, distinct member
        keep = np.unique(np.concatenate([lo_i, hi_i[two]]))
        return self._keep(keep)

    # ------------------------------------------------------------------
    def bounds(self) -> (float, float):
        """(min, max) of q / (p + q) over the pairs of a scalar potential
        (empty scope). ``(0.0, 1.0)`` if every pair is zero (the evidence has
        probability zero for every choice of extreme points)."""
        p = self.P.reshape(self.m)
        q = self.Q.reshape(self.m)
        tot = p + q
        ok = tot > _ZERO_TOL
        if not ok.any():
            return 0.0, 1.0
        r = q[ok] / tot[ok]
        return float(r.min()), float(r.max())


# qhull is used up to this affine dimension; above it, the LP vertex test.
_QHULL_MAX_DIM = 7
_HULL_RANK_TOL = 1e-12
# A point is declared interior only if its convex-combination residual is
# below this (relative) tolerance; otherwise it is kept, which is always sound.
_LP_RESID_TOL = 1e-10
# The LP test costs about (m + h) LPs of h columns (h = #vertices); above this
# many rows it costs more than the products it saves (measured: 69 s on an
# 18,400-row potential of dag_fr_tw3_n10_1 whose unpruned downstream takes
# 0.1 s), so larger potentials keep every row (sound).
_LP_MAX_ROWS = 1000
# qhull above affine dimension 5 is slow on large sets and often fails on the
# nearly flat ones that occur here (3.4 s for a failing 18,400-row d=7 call).
_QHULL_BIG_DIM = 5
_QHULL_MAX_ROWS = 5000


def hull_vertices(X: np.ndarray) -> np.ndarray:
    """Sorted indices of the rows of ``X`` that are vertices of conv(rows).

    The rows are first projected onto their affine hull (SVD), so the
    lower-dimensional sets that arise in practice (conditional tables sum to a
    constant; a binary single-variable potential is a segment) are handled
    exactly: affine dimension 0 keeps one row, 1 keeps the two endpoints,
    2.._QHULL_MAX_DIM uses qhull, and higher dimensions (or a qhull failure,
    e.g. on nearly flat sets) the LP vertex test (:func:`_lp_hull_vertices`)
    for at most _LP_MAX_ROWS rows. Any failure keeps every row (always sound)."""
    m = X.shape[0]
    allidx = np.arange(m)
    if m <= 2:
        return allidx
    C = X - X.mean(axis=0)
    scale = max(1.0, float(np.abs(X).max()))
    _, sv, Vt = np.linalg.svd(C, full_matrices=False)
    r = int((sv > _HULL_RANK_TOL * scale * max(1.0, np.sqrt(m))).sum())
    if r == 0:
        return allidx[:1]
    Y = C @ Vt[:r].T
    if r == 1:
        y = Y[:, 0]
        return np.unique([int(np.argmin(y)), int(np.argmax(y))])
    if m <= r + 1:
        return allidx
    if r <= _QHULL_MAX_DIM and (r <= _QHULL_BIG_DIM or m <= _QHULL_MAX_ROWS):
        try:
            from scipy.spatial import ConvexHull
            return np.sort(ConvexHull(Y).vertices)
        except Exception:
            pass
    if m > _LP_MAX_ROWS:
        return allidx
    try:
        return _lp_hull_vertices(Y)
    except Exception:
        return allidx


def _lp_hull_vertices(Y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Indices of a superset of the vertices of conv(rows of Y) (Y of shape
    (m, r)), by Clarkson's output-sensitive LP algorithm.

    A working set V of rows, all certified vertices, is grown from the extreme
    rows of random directions. Each remaining row i is tested against conv(V)
    with one feasibility LP (y_i = sum_j l_j y_j, l >= 0, sum l = 1; size |V|,
    not m). If it is inside, it is not a vertex and is dropped. Otherwise the
    separation LP gives a direction a with a.y_i > max_V a.y; the row k
    maximizing a.y over ALL rows (lexicographic tie-break, which picks a
    vertex of that face) is a new vertex outside conv(V); add it and re-test
    i. Every LP has at most h = #vertices columns, so the cost is about
    (m + h) LPs of size h.

    Soundness: a row is dropped only when the LP finds a convex combination
    of OTHER rows reproducing it to within _LP_RESID_TOL (checked on the
    returned weights); any doubt keeps it. Keeping extra rows never changes the
    bounds, it only costs time."""
    from scipy.optimize import linprog
    m, r = Y.shape
    scale = max(1.0, float(np.abs(Y).max()))
    tol = 1e-12 * scale
    rng = np.random.default_rng(seed)

    def argmax_vertex(a):
        val = Y @ a
        top = np.flatnonzero(val >= val.max() - tol)
        if top.size == 1:
            return int(top[0])
        sub = Y[top]
        return int(top[np.lexsort(sub.T[::-1])[-1]])   # lexicographic max

    inV = np.zeros(m, dtype=bool)
    for a in rng.standard_normal((2 * r, r)):
        inV[argmax_vertex(a)] = True
        inV[argmax_vertex(-a)] = True

    def in_hull(i, S):
        A = np.vstack([Y[S].T, np.ones(S.size)])
        bvec = np.concatenate([Y[i], [1.0]])
        res = linprog(np.zeros(S.size), A_eq=A, b_eq=bvec, bounds=(0, None),
                      method="highs")
        return res.status == 0 and \
            float(np.abs(Y[S].T @ res.x - Y[i]).max()) <= _LP_RESID_TOL * scale

    def separate(i, S):
        # max t  s.t.  a.y_j - b <= 0 (j in S),  a.y_i - b >= t,  -1 <= a <= 1
        k = S.size
        c = np.zeros(r + 2)
        c[-1] = -1.0
        A = np.zeros((k + 1, r + 2))
        A[:k, :r] = Y[S]
        A[:k, r] = -1.0
        A[k, :r] = -Y[i]
        A[k, r] = 1.0
        A[k, r + 1] = 1.0
        res = linprog(c, A_ub=A, b_ub=np.zeros(k + 1),
                      bounds=[(-1, 1)] * r + [(None, None), (None, 1.0)],
                      method="highs")
        if res.status != 0 or -res.fun <= tol:
            return None
        return res.x[:r]

    order = np.argsort(np.linalg.norm(Y - Y.mean(axis=0), axis=1))
    keep = inV.copy()
    for i in order:
        if inV[i]:
            continue
        while True:
            S = np.flatnonzero(inV)
            if in_hull(i, S):
                break                       # interior (not a vertex): drop
            a = separate(i, S)
            if a is None:                   # numerically on the boundary
                keep[i] = True
                break
            k = argmax_vertex(a)
            if inV[k] or k == i:
                inV[i] = keep[i] = True     # i itself is extreme
                break
            inV[k] = keep[k] = True         # new vertex; re-test i
    return np.flatnonzero(keep)


def _dominated(A: np.ndarray, B: np.ndarray, d: int) -> np.ndarray:
    """dom[i] = exists j != i with A[j] <= A[i] and B[j] >= B[i] on every
    coordinate, and (A[j], B[j]) != (A[i], B[i]).

    Rows are first sorted by S = sum(A) - sum(B): a row can only be dominated by
    a row with S no larger (dominance implies it), so candidate dominators of
    row i are restricted to rows sorted before it or tied with it. The test is
    done in blocks of rows, one broadcast per block."""
    m = A.shape[0]
    S = A.sum(axis=1) - B.sum(axis=1)
    order = np.argsort(S, kind="stable")
    As, Bs, Ss = A[order], B[order], S[order]
    dom_sorted = np.zeros(m, dtype=bool)
    block = max(1, int(_BLOCK_ELEMS // max(1, m * d)))
    for start in range(0, m, block):
        stop = min(m, start + block)
        # Candidate dominators: rows with S <= S of the block's last row.
        lim = np.searchsorted(Ss, Ss[stop - 1] + 1e-12, side="right")
        Aj, Bj = As[:lim], Bs[:lim]                     # (k, d)
        Ai, Bi = As[start:stop], Bs[start:stop]         # (b, d)
        le = np.all(Aj[None, :, :] <= Ai[:, None, :], axis=2)
        ge = np.all(Bj[None, :, :] >= Bi[:, None, :], axis=2)
        cand = le & ge                                  # (b, k)
        # exclude self and exact duplicates (removed earlier, but be safe)
        same = np.all(Aj[None] == Ai[:, None], axis=2) & \
            np.all(Bj[None] == Bi[:, None], axis=2)
        cand &= ~same
        dom_sorted[start:stop] = cand.any(axis=1)
    dom = np.empty(m, dtype=bool)
    dom[order] = dom_sorted
    return dom


def build_cpt_pairs(child: str, parents: List[str], cards: Dict[str, int],
                    vertices_by_config: Dict[tuple, list]) -> PairPotential:
    """CPT potential {(f, f)} over [child] + parents, one f per combination of
    one extreme point per parent configuration, built with one fancy-index
    assignment per parent configuration (no per-entry Python loop)."""
    scope = [child] + list(parents)
    kc = cards[child]
    if parents:
        import itertools
        configs = list(itertools.product(*[range(cards[p]) for p in parents]))
    else:
        configs = [()]
    vert_lists = []
    for pc in configs:
        vs = vertices_by_config.get(pc)
        if not vs:
            # A missing local credal set is not "uniform": substituting any
            # distribution would silently change the bounds.
            raise ValueError(f"No extreme points for {child} given "
                             f"{dict(zip(parents, pc))}.")
        vert_lists.append(np.asarray(vs, dtype=float))   # (n_pc, kc)
    counts = [len(v) for v in vert_lists]
    m = int(np.prod(counts))
    # combo index -> chosen vertex per configuration (mixed radix)
    idx = np.indices(counts).reshape(len(counts), -1).T if counts else \
        np.zeros((1, 0), dtype=int)
    shape = tuple(cards[v] for v in scope)
    F = np.zeros((m,) + shape)
    for c, pc in enumerate(configs):
        sel = (slice(None), slice(None)) + tuple(pc)
        F[sel] = vert_lists[c][idx[:, c]]                 # (m, kc)
    return PairPotential.from_functions(scope, cards, F)
