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

# Interval Belief Propagation (IVBP): the sum-product algorithm on the factor
# graph of a credal network, lifted to credal sets.
#
# For one fixed choice of local conditionals, sum-product on the factor graph
# sends
#     mu_{v->f}(x) = prod_{g in N(v) \ f} mu_{g->v}(x)
#     mu_{f->v}(x) = sum_{x_{N(f) \ v}} phi_f(x, ...) prod_{w in N(f) \ v} mu_{w->f}
# and the belief b_v ~ prod_{g in N(v)} mu_{g->v} is P(V | e) on a tree. As the
# local conditionals range over the strong extension every message ranges over
# a SET of vectors. Messages only matter up to scale, so a message is stored as
# the finite set of VERTICES of the convex hull of its normalized vectors (a
# "credal message"); for a binary variable this is an interval, and the scheme
# is the factor-graph form of 2U (Fagiuoli & Zaffalon, AIJ 1998): factor->child
# messages are 2U's pi-messages, factor->parent messages its lambda-messages.
#
# Each update is a projective map (linear, then normalize) of each input, and a
# projective map with a positive denominator maps convex hulls onto convex
# hulls, so propagating hull vertices loses nothing. On a polytree the inputs
# of every update depend on disjoint parameters and vary independently, so the
# two-pass schedule computes the EXACT strong-extension posteriors, for binary
# and multi-valued (compound) variables alike, with or without evidence. On a
# loopy graph the same updates are iterated (as in loopy BP / L2U), without
# guarantee.

import itertools
import time
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

# Local
from lcn.core.model import LCN
from lcn.inference.marginal.cn.pair_potentials import hull_vertices
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.utils.common import check_consistency

# A credal message is an (m, k) array of normalized vectors (rows sum to 1).
_ROUND = 12          # decimals used to merge numerically identical vertices
_MAX_CORNERS = 200_000   # guard on the corner enumeration of one update


# ----------------------------------------------------------------------
# Credal-message primitives
# ----------------------------------------------------------------------

def _normalize_rows(M: np.ndarray) -> np.ndarray:
    """Rows scaled to sum 1; all-zero rows (a contradiction for that choice,
    probability zero) are dropped -- regular conditioning."""
    s = M.sum(axis=1)
    ok = s > 0
    return M[ok] / s[ok, None]


def _reduce(M: np.ndarray) -> np.ndarray:
    """Hull vertices of a set of normalized vectors (duplicates merged)."""
    if M.shape[0] <= 1:
        return M
    M = np.unique(np.round(M, _ROUND), axis=0)
    if M.shape[0] <= 2:
        return M
    if M.shape[1] == 2:                       # binary: an interval
        return M[[int(np.argmin(M[:, 1])), int(np.argmax(M[:, 1]))]]
    return M[hull_vertices(M)]


def _point(v: np.ndarray) -> np.ndarray:
    return _normalize_rows(np.asarray(v, dtype=float)[None, :])


def _vacuous(k: int) -> np.ndarray:
    """The neutral message: the single all-ones (uniform) vector. As a factor
    in a product it changes nothing."""
    return np.full((1, k), 1.0 / k)


def _product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """{normalize(a * b)}: the elementwise product of two credal messages over
    the same variable. Exact: for fixed b, a -> normalize(a * b) is projective,
    so the image of conv(A) is conv of the images of A's vertices."""
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.zeros((0, A.shape[1]))
    P = (A[:, None, :] * B[None, :, :]).reshape(-1, A.shape[1])
    return _reduce(_normalize_rows(P))


def _product_all(msgs: List[np.ndarray], k: int) -> np.ndarray:
    out = _vacuous(k)
    for M in msgs:
        out = _product(out, M)
        if out.shape[0] == 0:
            break
    return out


def _minkowski_sum(sets: List[np.ndarray]) -> np.ndarray:
    """Vertices of conv(S_1 + ... + S_r) for finite sets S_j (unnormalized),
    summed one at a time with hull reduction after each step (the vertices of
    a Minkowski sum are sums of vertices)."""
    acc = sets[0]
    for S in sets[1:]:
        acc = (acc[:, None, :] + S[None, :, :]).reshape(-1, acc.shape[1])
        if acc.shape[0] > 2:
            acc = np.unique(np.round(acc, _ROUND), axis=0)
            if acc.shape[0] > 2:
                acc = acc[hull_vertices(acc)]
    return acc


class IntervalBP:
    """
    Interval (credal) belief propagation on the factor graph of a credal
    network: sum-product with credal messages (see the module comment).

    Exact on polytrees (the credal network's DAG is singly connected), for
    binary and multi-valued variables, with or without evidence (provided
    every distribution of the strong extension gives the evidence positive
    probability). Iterated on loopy graphs without guarantee.
    """

    def __init__(self, cnv: CredalNetworkVertices):
        """
        Args:
            cnv: CredalNetworkVertices
                A built CredalNetworkVertices (extreme points enumerated).
        """
        assert cnv.extreme_points is not None, \
            "CredalNetworkVertices must be built before passing to IVBP."
        assert cnv.bn_min is not None
        self.cnv = cnv
        self.marginals = None
        self.singleton_marginals = None

    # ------------------------------------------------------------------
    # Factor graph
    # ------------------------------------------------------------------

    def _build(self):
        """Variables (cards, atoms), family factors with their per-row vertex
        sets, and the parent / child structure."""
        bn = self.cnv.bn_min
        self.cards = {bn.variable(n).name(): bn.variable(n).domainSize()
                      for n in bn.nodes()}
        self.nodes = [bn.variable(n).name() for n in bn.topologicalOrder()]
        self.parents, self.rows = {}, {}
        for node in self.nodes:
            nid = bn.idFromName(node)
            par = [bn.variable(p).name() for p in sorted(bn.parents(nid))]
            self.parents[node] = par
            configs = list(itertools.product(
                *[range(self.cards[p]) for p in par])) if par else [()]
            by_cfg = {}
            for cfg, verts in self.cnv.extreme_points[node].items():
                if cfg == "<>":
                    by_cfg[()] = verts
                    continue
                vals = dict(x.split(":") for x in cfg[1:-1].split("|"))
                by_cfg[tuple(int(vals[p]) for p in par)] = verts
            rows = []
            for pc in configs:
                if pc not in by_cfg or not by_cfg[pc]:
                    raise ValueError(f"No extreme points for {node} given "
                                     f"{dict(zip(par, pc))}.")
                rows.append(np.asarray(by_cfg[pc], dtype=float))
            self.rows[node] = (configs, rows)
        self.children = {v: [] for v in self.nodes}
        for v in self.nodes:
            for p in self.parents[v]:
                self.children[p].append(v)
        und = nx.Graph()
        und.add_nodes_from(self.nodes)
        und.add_edges_from((p, v) for v in self.nodes for p in self.parents[v])
        self.singly_connected = nx.is_forest(und)

    def _evidence_messages(self, evidence):
        """{node -> indicator credal message} for nodes with observed atoms
        (the 0/1 indicator over the node's states of the observed atom values,
        MSB-first state packing of compound nodes)."""
        na = self.cnv.cn.node_atoms
        out = {}
        for node in self.nodes:
            atoms = na[node]
            k = self.cards[node]
            ind = np.ones(k)
            hit = False
            for i, a in enumerate(atoms):
                if a in evidence:
                    hit = True
                    bit = len(atoms) - 1 - i
                    ind *= np.array([((s >> bit) & 1) == int(evidence[a])
                                     for s in range(k)], dtype=float)
            if hit:
                if ind.sum() == 0:
                    raise ValueError(f"Contradictory evidence on {node}.")
                out[node] = _point(ind)
        return out

    # ------------------------------------------------------------------
    # Factor -> variable messages (the family factor f_V of node V)
    # ------------------------------------------------------------------

    def _corners(self, sets):
        """All combinations of one vertex per credal message."""
        n = 1
        for S in sets:
            n *= S.shape[0]
        if n > _MAX_CORNERS:
            raise MemoryError(f"IVBP: {n} corner combinations in one update "
                              f"(cap {_MAX_CORNERS}).")
        return itertools.product(*sets)

    def _factor_to_child(self, node, parent_msgs):
        """mu_{f_V -> V}: for each corner (one vertex q_j per parent message),
        sum_pi (prod_j q_j(pi_j)) P(V | pi), with every row P(V | pi) ranging
        over its vertices: a Minkowski sum of scaled vertex sets. Each corner's
        weights sum to one, so the result is already normalized."""
        configs, rows = self.rows[node]
        par = self.parents[node]
        if not par:
            return _reduce(rows[0])
        out = []
        for corner in self._corners([parent_msgs[p] for p in par]):
            w = np.array([np.prod([corner[j][pc[j]] for j in range(len(par))])
                          for pc in configs])
            terms = [w[c] * rows[c] for c in range(len(configs)) if w[c] > 0]
            if not terms:
                continue
            out.append(_minkowski_sum(terms))
        if not out:
            return np.zeros((0, self.cards[node]))
        return _reduce(_normalize_rows(np.vstack(out)))

    def _factor_to_parent(self, node, target, parent_msgs, child_msg):
        """mu_{f_V -> U}: for each corner (one vertex of the child message
        lambda and of every other parent's message),
            m(u) = sum_{pi : pi_U = u} w(pi) <lambda, P(V | pi)>.
        Each coordinate m(u) depends on its own rows only, so as the rows range
        over their vertices m ranges over a BOX [m_lo, m_hi]; the normalized
        image of a box is the hull of its normalized corners."""
        configs, rows = self.rows[node]
        par = self.parents[node]
        ti = par.index(target)
        others = [p for p in par if p != target]
        oi = [par.index(p) for p in others]
        ku = self.cards[target]
        out = []
        for corner in self._corners([child_msg] +
                                    [parent_msgs[p] for p in others]):
            lam, oq = corner[0], corner[1:]
            lo = np.zeros(ku)
            hi = np.zeros(ku)
            for c, pc in enumerate(configs):
                w = 1.0
                for j, idx in enumerate(oi):
                    w *= oq[j][pc[idx]]
                if w == 0:
                    continue
                vals = rows[c] @ lam                    # one per vertex
                lo[pc[ti]] += w * vals.min()
                hi[pc[ti]] += w * vals.max()
            free = np.flatnonzero(hi > lo)
            if free.size == 0:
                out.append(lo[None, :])
                continue
            if 2 ** free.size > _MAX_CORNERS:
                raise MemoryError("IVBP: too many box corners.")
            for bits in itertools.product((0, 1), repeat=free.size):
                m = lo.copy()
                m[free[np.array(bits, dtype=bool)]] = \
                    hi[free[np.array(bits, dtype=bool)]]
                out.append(m[None, :])
        res = _normalize_rows(np.vstack(out)) if out else np.zeros((0, ku))
        if res.shape[0] == 0:
            # lambda is zero on every reachable child state: this parent
            # message carries no information (the evidence is impossible
            # below this edge for every choice) -- treat as neutral.
            return _vacuous(ku)
        return _reduce(res)

    # ------------------------------------------------------------------
    # Message passing
    # ------------------------------------------------------------------

    def _var_to_factor(self, v, exclude, f2v, ev):
        """Product of every message into v except the one from ``exclude``:
        f2v holds family-factor messages keyed (factor_node, v); ev the
        evidence messages."""
        msgs = [M for (f, w), M in f2v.items() if w == v and f != exclude]
        if v in ev:
            msgs.append(ev[v])
        return _product_all(msgs, self.cards[v])

    def _update_factor(self, node, f2v, ev):
        """Recompute every message out of the family factor f_node."""
        par = self.parents[node]
        new = {}
        pm = {p: self._var_to_factor(p, node, f2v, ev) for p in par}
        new[(node, node)] = self._factor_to_child(node, pm)
        if par:
            cm = self._var_to_factor(node, node, f2v, ev)
            for p in par:
                new[(node, p)] = self._factor_to_parent(node, p, pm, cm)
        return new

    def run(self, evidence: dict = None, n_iters: int = 100,
            threshold: float = 1e-6, verbosity: int = 1
            ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Credal sum-product. Computes bounds on P(V = s | e) for every node V and
        state s, and on P(a = 1 | e) for every non-evidence atom a.

        Args:
            evidence: {atom_name: 0/1} for observed atoms.
            n_iters: iteration cap on loopy graphs (polytrees need none).
            threshold: convergence threshold on loopy graphs (max change of a
                message's per-state bounds).
            verbosity: 0 silent, 1 summary, 2 per-iteration detail.

        Returns:
            {name -> (lower, upper)}: per-state bounds for every node, and the
            2-vectors [P(=0), P(=1)] for every non-evidence atom of a compound
            node. Also sets self.marginals, self.singleton_marginals ({atom ->
            (lo, hi)} for P(atom=1), every non-evidence atom), self.exact,
            self.singly_connected, self.converged, self.iterations,
            self.run_time.
        """
        evidence = {} if evidence is None else dict(evidence)
        t0 = time.perf_counter()
        self._build()
        na = self.cnv.cn.node_atoms
        all_atoms = {a for ats in na.values() for a in ats}
        for a in evidence:
            assert a in all_atoms, f"Evidence variable '{a}' not found."
        ev = self._evidence_messages(evidence)

        # Initial messages: neutral everywhere.
        f2v = {}
        for v in self.nodes:
            f2v[(v, v)] = _vacuous(self.cards[v])
            for p in self.parents[v]:
                f2v[(v, p)] = _vacuous(self.cards[p])

        if verbosity > 0:
            print(f"[IVBP] {len(self.nodes)} nodes, evidence={evidence}, "
                  f"{'polytree' if self.singly_connected else 'loopy'}")

        # Sweeps alternate forward (topological) and reverse order. On a
        # polytree every message depends only on the messages of its own
        # subtree, so the sweeps reach the unique fixed point in a number of
        # sweeps bounded by the diameter; we stop when a sweep changes nothing
        # (threshold 0 up to rounding), which certifies the fixed point. On a
        # loopy graph the same sweeps run until the change is below
        # ``threshold`` or ``n_iters`` sweeps.
        tol = 1e-12 if self.singly_connected else threshold
        cap = max(n_iters, 2 * len(self.nodes) + 2) \
            if self.singly_connected else n_iters
        self.converged, self.iterations = False, 0
        for it in range(cap):
            delta = 0.0
            order = self.nodes if it % 2 == 0 else list(reversed(self.nodes))
            for node in order:
                new = self._update_factor(node, f2v, ev)
                for key, M in new.items():
                    delta = max(delta, _set_change(f2v[key], M))
                f2v.update(new)
            self.iterations = it + 1
            if verbosity > 1:
                print(f"  sweep {it + 1}: max change {delta:.3e}")
            if delta <= tol:
                self.converged = True
                break

        # Beliefs and read-off.
        self.beliefs = {}
        self.marginals = {}
        self.singleton_marginals = {}
        self.empty = False
        for v in self.nodes:
            b = self._var_to_factor(v, None, f2v, ev)
            self.beliefs[v] = b
            k = self.cards[v]
            if b.shape[0] == 0:
                self.empty = True
                lo, hi = np.zeros(k), np.ones(k)
            else:
                lo, hi = b.min(axis=0), b.max(axis=0)
            self.marginals[v] = (lo, hi)
            for i, a in enumerate(na[v]):
                if a in evidence:
                    continue
                if b.shape[0] == 0:
                    self.singleton_marginals[a] = (0.0, 1.0)
                    continue
                bit = len(na[v]) - 1 - i
                T = np.array([(s >> bit) & 1 for s in range(k)], dtype=float)
                pa = b @ T          # linear in b: extremes at belief vertices
                self.singleton_marginals[a] = (float(pa.min()),
                                               float(pa.max()))

        self.exact = self.singly_connected and self.converged
        self.run_time = time.perf_counter() - t0
        if verbosity > 0:
            self._print()

        results = dict(self.marginals)
        for a, (lo, hi) in self.singleton_marginals.items():
            results[a] = (np.array([1.0 - hi, lo]), np.array([1.0 - lo, hi]))
        return results

    def _print(self):
        tag = "exact (polytree)" if self.exact else \
            f"loopy, {'converged' if self.converged else 'NOT converged'} " \
            f"after {self.iterations} iterations"
        print(f"[IVBP] {tag}; {self.run_time:.4f} s")
        for a in sorted(self.singleton_marginals):
            lo, hi = self.singleton_marginals[a]
            print(f"  P({a}=1): [{lo:.6f}, {hi:.6f}]")


def _set_change(A: np.ndarray, B: np.ndarray) -> float:
    """Change between two credal messages: the Hausdorff distance (max norm)
    between their vertex sets -- zero iff the hulls coincide (both are hull
    vertex sets, merged at _ROUND decimals)."""
    if A.shape[0] == 0 or B.shape[0] == 0:
        return 0.0 if A.shape[0] == B.shape[0] else 1.0
    D = np.abs(A[:, None, :] - B[None, :, :]).max(axis=2)
    return float(max(D.min(axis=1).max(), D.min(axis=0).max()))


if __name__ == "__main__":
    file_name = "examples/alarm.lcn"
    lcn_model = LCN()
    lcn_model.from_lcn(file_name=file_name)
    check_consistency(lcn_model)
    cnv = CredalNetworkVertices.from_lcn(lcn_model, method="linear",
                                         verbosity=0)
    ibp = IntervalBP(cnv=cnv)
    print("\n=== IVBP (no evidence) ===")
    ibp.run(evidence={}, verbosity=1)
    print("\n=== IVBP (C = 1) ===")
    ibp.run(evidence={"C": 1}, verbosity=1)
