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

# Exact local credal sets of compound families.
#
# For a family v with a multi-atom child block and a parent configuration pi,
# the local program ranges over joints rho on the family scope that satisfy
# the in-scope sentences (linear rows A rho <= b) with rho(pa_v = pi) > 0. Its
# set of conditional rows rho(. , pi) / rho(pi) is, by the Charnes-Cooper
# substitution y = rho / rho(pi), the LINEAR projection onto the row
# coordinates of the polyhedron
#     Y = { (y, t) : y >= 0, t >= 0, A y - b t <= 0, E y = 1 },
# hence a polytope (it lies in the simplex). The interval box of its per-state
# bounds -- what the compiler stores for every family -- is exact for a
# single-atom child (an interval) but in general strictly larger for a
# compound child. This module computes the polytope's vertices exactly with
# the convex-hull method: an LP oracle (HiGHS) maximizes a direction over Y;
# starting from coordinate and random directions, the hull of the points found
# is refined by maximizing every facet normal, until no facet moves (finite for
# a polytope), and the affine hull is completed by probing its orthogonal
# complement. On any failure the caller falls back to the (outer) box.

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import linprog

_TOL = 1e-9
_ROUND = 12


class RowPolytope:
    """The projection of the Charnes-Cooper polyhedron of one local program
    onto the row coordinates ``idx`` (child states of the parent
    configuration, MSB first)."""

    def __init__(self, A: np.ndarray, b: np.ndarray, E: Optional[np.ndarray],
                 idx, n: int):
        self.k = len(idx)
        self.idx = list(idx)
        self.n = n
        if E is None:
            # Root family: the variables are the joint p itself (the simplex
            # rows are part of A p <= b).
            self.A_ub, self.b_ub = A, b
            self.A_eq = self.b_eq = None
            self.bounds = [(0.0, 1.0)] * n
            self.nvar = n
        else:
            self.A_ub = np.hstack([A, -b[:, None]])
            self.b_ub = np.zeros(len(b))
            self.A_eq = np.concatenate([np.asarray(E, float), [0.0]])[None, :]
            self.b_eq = np.array([1.0])
            self.bounds = [(0.0, None)] * (n + 1)
            self.nvar = n + 1
        self.n_lp = 0

    def support(self, d: np.ndarray) -> Optional[np.ndarray]:
        """A row maximizing <d, row>, or None if the program is infeasible or
        the LP fails."""
        c = np.zeros(self.nvar)
        c[self.idx] = -np.asarray(d, float)
        self.n_lp += 1
        res = linprog(c, A_ub=self.A_ub, b_ub=self.b_ub, A_eq=self.A_eq,
                      b_eq=self.b_eq, bounds=self.bounds, method="highs")
        if res.status != 0 or res.x is None:
            return None
        return np.asarray(res.x[self.idx], float)


def _affine_basis(P: np.ndarray, tol: float = 1e-9):
    C = P - P.mean(axis=0)
    _, sv, Vt = np.linalg.svd(C, full_matrices=True)
    r = int((sv > tol * max(1.0, sv.max() if sv.size else 1.0)).sum())
    return Vt[:r], Vt[r:]


def row_polytope_vertices(rp: RowPolytope, max_rounds: int = 60,
                          seed: int = 0) -> Tuple[Optional[np.ndarray], bool]:
    """Vertices of the row polytope. Returns (vertices, exact): ``exact`` is
    False when the round cap was reached or a numerical step failed, in which
    case ``vertices`` is None and the caller must use an outer set."""
    from scipy.spatial import ConvexHull
    k = rp.k
    rng = np.random.default_rng(seed)
    dirs = np.vstack([np.eye(k), -np.eye(k), rng.standard_normal((3 * k, k))])
    pts = []
    for d in dirs:
        p = rp.support(d)
        if p is None:
            return None, False
        pts.append(p)
    P = np.unique(np.round(np.array(pts), _ROUND), axis=0)
    # rows lie in {sum = 1}: the complement of the affine hull is probed
    # within the hyperplane sum = 0
    for _ in range(max_rounds):
        B, Q = _affine_basis(P)
        center = P.mean(axis=0)
        # 1. complete the affine hull: any direction orthogonal to it with a
        #    non-zero extent adds a point
        grew = False
        for q in Q:
            q = q - q.mean()                       # stay in sum = 0
            if np.linalg.norm(q) < 1e-12:
                continue
            for s in (1.0, -1.0):
                p = rp.support(s * q)
                if p is None:
                    return None, False
                if abs((p - center) @ q) > _TOL:
                    P = np.vstack([P, p])
                    grew = True
        if grew:
            P = np.unique(np.round(P, _ROUND), axis=0)
            continue
        r = B.shape[0]
        if r == 0:
            return P[:1], True
        if r == 1:
            lo, hi = rp.support(-B[0]), rp.support(B[0])
            if lo is None or hi is None:
                return None, False
            return np.unique(np.round(np.vstack([lo, hi]), _ROUND), axis=0), \
                True
        Y = (P - center) @ B.T
        if len(P) <= r:
            return None, False
        try:
            hull = ConvexHull(Y)
        except Exception:
            return None, False
        # 2. refine: maximize every facet's outward normal
        new = []
        for eq in hull.equations:                  # n . y + off <= 0 inside
            n, off = eq[:-1], eq[-1]
            p = rp.support(B.T @ n)
            if p is None:
                return None, False
            if n @ ((p - center) @ B.T) + off > _TOL:
                new.append(p)
        if not new:
            return P[np.sort(hull.vertices)], True
        P = np.unique(np.round(np.vstack([P[hull.vertices], np.array(new)]),
                               _ROUND), axis=0)
    return None, False
