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

# CDVE (Coordinate Descent with Variable Elimination): inner bounds for credal networks by coordinate descent over
# the local credal sets (Antonucci, de Campos, Huber & Zaffalon, "Approximate
# credal network updating by linear programming", 2015).
#
# For an atom a of node Q and evidence e, the objective is
#     P(a | e) = sum_{s |= a} P(Q = s, e) / sum_s P(Q = s, e),
# a ratio of multilinear functions of the local conditionals. With every
# family but one fixed, both P(a, e) and P(e) are LINEAR in the rows of that
# family V:  P(., e) = sum_{x, pi} P(x | pi) T(x, pi, Q), where T is the product
# of all other factors marginalized onto {V} u pa(V) u {Q} (one variable
# elimination). The family step maximizes (minimizes) the ratio
#     sum_pi <n_pi, r_pi> / sum_pi <d_pi, r_pi>
# over the product of the rows' credal sets EXACTLY, by Dinkelbach's method:
# for a given t every row independently takes the vertex maximizing
# <n_pi - t d_pi, v>. This follows ApproxLP's local linear(-fractional) program over
# one variable's (separately specified) credal set. Sweeps over the families
# repeat until none improves. Every point visited is a member of the strong
# extension with P(e) > 0, so the results are INNER bounds.

import itertools
import time
from typing import Dict, Tuple

import numpy as np

# Local
from lcn.core.model import LCN
from lcn.inference.marginal.cn.potentials import min_fill_order
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.utils.common import check_consistency

_DINKELBACH_TOL = 1e-15


def _expand(arr, src, tgt, cards):
    """``arr`` over scope ``src`` reshaped (transposed + size-1 axes) so that
    its axes follow ``tgt``."""
    idx = {v: i for i, v in enumerate(src)}
    a = np.transpose(arr, [idx[v] for v in tgt if v in idx]) if src else arr
    return a.reshape([cards[v] if v in idx else 1 for v in tgt])


def _combine(s1, a1, s2, a2, cards):
    scope = list(s1) + [v for v in s2 if v not in s1]
    return scope, _expand(a1, s1, scope, cards) * _expand(a2, s2, scope, cards)


class CDVE:
    """
    CDVE, following ApproxLP (Antonucci et al. 2015): coordinate descent over the local credal sets
    of a credal network, one family at a time, each step solved exactly.
    Inner bounds on P(atom = 1 | evidence) for every non-evidence atom, with
    or without evidence, for binary and compound (multi-valued) nodes.
    """

    def __init__(self, cnv: CredalNetworkVertices):
        assert cnv.extreme_points is not None, \
            "CredalNetworkVertices must be built before passing to CDVE."
        assert cnv.bn_min is not None
        self.cnv = cnv
        self.marginals = None
        self.singleton_marginals = None

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    def _build(self):
        bn = self.cnv.bn_min
        self.cards = {bn.variable(n).name(): bn.variable(n).domainSize()
                      for n in bn.nodes()}
        self.nodes = [bn.variable(n).name() for n in bn.topologicalOrder()]
        self.parents, self.configs, self.verts = {}, {}, {}
        for node in self.nodes:
            nid = bn.idFromName(node)
            par = [bn.variable(p).name() for p in sorted(bn.parents(nid))]
            self.parents[node] = par
            cfgs = list(itertools.product(
                *[range(self.cards[p]) for p in par])) if par else [()]
            by_cfg = {}
            for cfg, vs in self.cnv.extreme_points[node].items():
                if cfg == "<>":
                    by_cfg[()] = vs
                    continue
                vals = dict(x.split(":") for x in cfg[1:-1].split("|"))
                by_cfg[tuple(int(vals[p]) for p in par)] = vs
            rows = []
            for pc in cfgs:
                if pc not in by_cfg or not by_cfg[pc]:
                    raise ValueError(f"No extreme points for {node} given "
                                     f"{dict(zip(par, pc))}.")
                rows.append(np.asarray(by_cfg[pc], dtype=float))
            self.configs[node] = cfgs
            self.verts[node] = rows
        self._order_cache = {}

    def _ancestral(self, seeds):
        """Seeds plus all their ancestors (families outside this set are
        barren: they sum out to one and do not affect P(Q, e))."""
        out, stack = set(), list(seeds)
        while stack:
            v = stack.pop()
            if v not in out:
                out.add(v)
                stack.extend(self.parents[v])
        return out

    def _indicator(self, node, atom_index, value):
        atoms = self.cnv.cn.node_atoms[node]
        bit = len(atoms) - 1 - atom_index
        return np.array([((s >> bit) & 1) == value
                         for s in range(self.cards[node])], dtype=float)

    def _evidence_masks(self, evidence):
        na = self.cnv.cn.node_atoms
        masks = {}
        for node in self.nodes:
            m = np.ones(self.cards[node])
            hit = False
            for i, a in enumerate(na[node]):
                if a in evidence:
                    hit = True
                    m = m * self._indicator(node, i, int(evidence[a]))
            if hit:
                if m.sum() == 0:
                    raise ValueError(f"Contradictory evidence on {node}.")
                masks[node] = m
        return masks

    # ------------------------------------------------------------------
    # Variable elimination
    # ------------------------------------------------------------------

    def _cpt(self, node, rows):
        """CPT array over [node] + parents from one row per configuration."""
        k = self.cards[node]
        arr = np.empty((k,) + tuple(self.cards[p] for p in self.parents[node]))
        for pc, r in zip(self.configs[node], rows):
            arr[(slice(None),) + pc] = r
        return arr

    def _eliminate(self, factors, keep, key):
        """Sum the product of ``factors`` onto the variables ``keep`` (in that
        order), eliminating along a cached min-fill order."""
        order = self._order_cache.get(key)
        if order is None:
            order = min_fill_order([s for s, _ in factors], exclude=set(keep))
            self._order_cache[key] = order
        pots = list(factors)
        for var in order:
            bucket = [p for p in pots if var in p[0]]
            if not bucket:
                continue
            pots = [p for p in pots if var not in p[0]]
            s, a = bucket[0]
            for s2, a2 in bucket[1:]:
                s, a = _combine(s, a, s2, a2, self.cards)
            ax = s.index(var)
            pots.append(([v for v in s if v != var], a.sum(axis=ax)))
        s, a = [], np.array(1.0)
        for s2, a2 in pots:
            s, a = _combine(s, a, s2, a2, self.cards)
        full = tuple(self.cards[v] for v in keep)
        return np.broadcast_to(_expand(a, s, keep, self.cards), full)

    def _factors(self, cur, relevant, masks, exclude=None):
        fs = [([v] + self.parents[v], self._cpt(v, cur[v]))
              for v in relevant if v != exclude]
        fs += [([v], m) for v, m in masks.items() if v in relevant]
        return fs

    def _family_coefficients(self, node, query, target, cur, relevant, masks):
        """n, d arrays over [node] + parents: P(target, e) = <n, CPT_node>
        and P(e) = <d, CPT_node> with every other family fixed."""
        fam = [node] + self.parents[node]
        keep = fam if query in fam else fam + [query]
        T = self._eliminate(self._factors(cur, relevant, masks, exclude=node),
                            keep, ("fam", node, query))
        if query in fam:
            ax = fam.index(query)
            shape = [1] * len(fam)
            shape[ax] = len(target)
            return T * target.reshape(shape), np.asarray(T)
        return T @ target, T.sum(axis=-1)

    def _posterior(self, query, cur, relevant, masks):
        P = self._eliminate(self._factors(cur, relevant, masks), [query],
                            ("post", query))
        s = P.sum()
        return P / s if s > 0 else None

    # ------------------------------------------------------------------
    # One run: coordinate descent for one atom and one sense
    # ------------------------------------------------------------------

    def _family_step(self, node, n, d, rows, sense):
        """Exact optimization of sum_pi <n_pi, r_pi> / sum_pi <d_pi, r_pi> over
        the product of the rows' vertex sets (Dinkelbach). Returns the new
        rows and the ratio, or None if no strict improvement."""
        cfgs = self.configs[node]
        V = self.verts[node]
        vn = [V[c] @ n[(slice(None),) + pc] for c, pc in enumerate(cfgs)]
        vd = [V[c] @ d[(slice(None),) + pc] for c, pc in enumerate(cfgs)]
        N0 = sum(float(r @ n[(slice(None),) + pc]) for r, pc in zip(rows, cfgs))
        D0 = sum(float(r @ d[(slice(None),) + pc]) for r, pc in zip(rows, cfgs))
        if D0 <= 0:
            return None
        t0 = t = N0 / D0
        sgn = 1.0 if sense == "max" else -1.0
        pick = None
        for _ in range(100):
            idx = [int(np.argmax(sgn * (a - t * b))) for a, b in zip(vn, vd)]
            F = sum(sgn * (a[i] - t * b[i]) for a, b, i in zip(vn, vd, idx))
            if F <= _DINKELBACH_TOL * max(1.0, D0):
                break
            N = sum(a[i] for a, i in zip(vn, idx))
            D = sum(b[i] for b, i in zip(vd, idx))
            t, pick = N / D, idx          # F > 0 implies D > 0 (n <= d)
        if pick is not None and sgn * (t - t0) > 1e-12 * max(1.0, abs(t0)):
            return [V[c][i].copy() for c, i in enumerate(pick)], t
        # No strict improvement. If some row is still not a vertex (the
        # centre initialization), move every row to a best vertex at the
        # current ratio: the objective does not get worse (F(t) >= 0 for the
        # argmax), and the move leaves the saddle point that the symmetric
        # centre creates (e.g. an uninformative child row hides a parent).
        # Each row becomes a vertex at most once, so this cannot cycle.
        if all(any(np.array_equal(r, v) for v in V[c])
               for c, r in enumerate(rows)):
            return None
        idx = [int(np.argmax(sgn * (a - t0 * b))) for a, b in zip(vn, vd)]
        N = sum(a[i] for a, i in zip(vn, idx))
        D = sum(b[i] for b, i in zip(vd, idx))
        if D <= 0 or sgn * (N / D - t0) < -1e-12 * max(1.0, abs(t0)):
            return None
        return [V[c][i].copy() for c, i in enumerate(idx)], N / D

    def _start(self, restart):
        """Start point: restart 0 is the centre of every local credal set
        (ApproxLP's initialization); restart j > 0 picks one vertex per row
        uniformly at random (deterministic seed)."""
        if restart == 0:
            return {v: [r.mean(axis=0) for r in self.verts[v]]
                    for v in self.nodes}
        rng = np.random.default_rng(restart)
        return {v: [r[rng.integers(len(r))].copy() for r in self.verts[v]]
                for v in self.nodes}

    def _descend(self, query, target, sense, relevant, masks, n_iters,
                 restart=0):
        cur = self._start(restart)
        fams = [v for v in self.nodes if v in relevant
                and any(len(r) > 1 for r in self.verts[v])]
        value = None
        sweeps = 0
        for sweeps in range(1, n_iters + 1):
            improved = False
            for node in fams:
                n, d = self._family_coefficients(node, query, target, cur,
                                                 relevant, masks)
                self.n_eliminations += 1
                step = self._family_step(node, n, d, cur[node], sense)
                if step is not None:
                    cur[node], value = step
                    improved = True
            if not improved:
                break
        post = self._posterior(query, cur, relevant, masks)
        if post is None:            # this start gives the evidence prob. 0
            return None, None, sweeps
        val = float(post @ target)
        return val, post, sweeps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, evidence: dict = None, n_iters: int = 50,
            n_restarts: int = 8,
            verbosity: int = 1) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Inner bounds on P(atom = 1 | evidence) for every non-evidence atom.

        Args:
            evidence: {atom_name: 0/1} for observed atoms (atoms of compound
                nodes included).
            n_iters: cap on the number of sweeps over the families, per atom,
                sense and start.
            n_restarts: extra starts from random vertex choices (seeds 1..),
                besides the centre start; the best result over the starts is
                kept. Every start ends at a member of the strong extension, so
                the bounds stay inner.
            verbosity: 0 silent, 1 summary, 2 per-atom detail.

        Returns:
            {name -> (lower, upper)}: for every node the per-state range of
            the posteriors P(node | e) at the optima found (attained values),
            and [P(=0), P(=1)] for every non-evidence atom of a compound node.
            Also sets self.marginals, self.singleton_marginals ({atom -> (lo,
            hi)} for P(atom = 1)), self.sweeps (max sweeps used),
            self.n_eliminations, self.run_time.
        """
        evidence = {} if evidence is None else dict(evidence)
        t0 = time.perf_counter()
        self._build()
        na = self.cnv.cn.node_atoms
        all_atoms = {a for ats in na.values() for a in ats}
        for a in evidence:
            assert a in all_atoms, f"Evidence variable '{a}' not found."
        masks = self._evidence_masks(evidence)
        self.n_eliminations = 0
        self.sweeps = 0

        self.singleton_marginals = {}
        self.marginals = {}
        for node in self.nodes:
            k = self.cards[node]
            relevant = self._ancestral({node} | set(masks))
            lo_s, hi_s = np.ones(k), np.zeros(k)
            seen = False
            for i, atom in enumerate(na[node]):
                if atom in evidence:
                    continue
                target = self._indicator(node, i, 1)
                bounds = []
                for sense in ("min", "max"):
                    best = None
                    for restart in range(n_restarts + 1):
                        val, post, sw = self._descend(
                            node, target, sense, relevant, masks, n_iters,
                            restart)
                        self.sweeps = max(self.sweeps, sw)
                        if val is None:
                            continue
                        lo_s = np.minimum(lo_s, post)
                        hi_s = np.maximum(hi_s, post)
                        seen = True
                        if best is None or (val < best if sense == "min"
                                            else val > best):
                            best = val
                    if best is None:     # P(e) = 0 at every start
                        best = 0.0 if sense == "min" else 1.0
                    bounds.append(best)
                self.singleton_marginals[atom] = (bounds[0], bounds[1])
                if verbosity > 1:
                    print(f"  P({atom}=1 | e) in [{bounds[0]:.6f}, "
                          f"{bounds[1]:.6f}]")
            if not seen:
                # every atom observed (or degenerate): report the posterior at
                # the centers
                cur = {v: [r.mean(axis=0) for r in self.verts[v]]
                       for v in self.nodes}
                post = self._posterior(node, cur, relevant, masks)
                lo_s, hi_s = (post, post.copy()) if post is not None else \
                    (np.zeros(k), np.ones(k))
            self.marginals[node] = (lo_s, hi_s)

        self.run_time = time.perf_counter() - t0
        if verbosity > 0:
            print(f"[CDVE] {len(self.singleton_marginals)} atoms, "
                  f"evidence={evidence}; max sweeps {self.sweeps}, "
                  f"{self.n_eliminations} eliminations, "
                  f"{self.run_time:.4f} s")
            for a in sorted(self.singleton_marginals):
                lo, hi = self.singleton_marginals[a]
                print(f"  P({a}=1): [{lo:.6f}, {hi:.6f}]")

        results = dict(self.marginals)
        for a, (lo, hi) in self.singleton_marginals.items():
            results[a] = (np.array([1.0 - hi, lo]), np.array([1.0 - lo, hi]))
        return results


if __name__ == "__main__":
    file_name = "examples/alarm.lcn"
    lcn_model = LCN()
    lcn_model.from_lcn(file_name=file_name)
    check_consistency(lcn_model)
    cnv = CredalNetworkVertices.from_lcn(lcn_model, method="linear",
                                         verbosity=0)
    alp = CDVE(cnv=cnv)
    print("\n=== CDVE (no evidence) ===")
    alp.run(evidence={}, verbosity=1)
    print("\n=== CDVE (E = 1) ===")
    alp.run(evidence={"E": 1}, verbosity=1)
