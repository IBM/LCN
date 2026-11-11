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

# Tests for CredalVE with pair potentials (Maua & Cozman, IJAR 2020, Alg. 2).
#
# The reference is the DEFINITION of the strong-extension bound: enumerate
# every combination of one extreme point per local credal set (per node and
# parent configuration), form the joint, condition on the evidence, and take the
# min / max of P(atom = 1 | e). CredalVE must reproduce it exactly, with and
# without evidence, including atoms of compound nodes.

import contextlib
import io
import itertools
import os
import random

import numpy as np
import pytest

from lcn.core.model import LCN
from lcn.inference.marginal.cn.cve import CredalVE
from lcn.inference.marginal.cn.pair_potentials import (
    PairPotential, _dominated)
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _cnv(path):
    lcn = LCN()
    _quiet(lcn.from_lcn, file_name=path)
    return _quiet(CredalNetworkVertices.from_lcn, lcn, method="linear",
                  verbosity=0)


def _bruteforce(cnv, evidence):
    """min/max P(a = 1 | e) over every product of local extreme points."""
    bn = cnv.bn_min
    names = [bn.variable(n).name() for n in bn.topologicalOrder()]
    cards = {v: bn.variable(bn.idFromName(v)).domainSize() for v in names}
    parents = {v: [bn.variable(p).name()
                   for p in sorted(bn.parents(bn.idFromName(v)))]
               for v in names}
    na = cnv.cn.node_atoms
    rows = []
    for v in names:
        for cfg, verts in cnv.extreme_points[v].items():
            if cfg == "<>":
                pc = ()
            else:
                d = dict(x.split(":") for x in cfg[1:-1].split("|"))
                pc = tuple(int(d[p]) for p in parents[v])
            rows.append((v, pc, verts))
    states = list(itertools.product(*[range(cards[v]) for v in names]))
    atoms = {a: (v, i) for v in names for i, a in enumerate(na[v])}

    def bit(v, s, i):
        return (s >> (len(na[v]) - 1 - i)) & 1

    ev_mask = np.array([all(bit(atoms[a][0], dict(zip(names, st))[atoms[a][0]],
                                atoms[a][1]) == val
                            for a, val in evidence.items()) for st in states])
    hit = {a: np.array([bit(v, dict(zip(names, st))[v], i) == 1
                        for st in states])
           for a, (v, i) in atoms.items() if a not in evidence}
    best = {a: [1.0, 0.0] for a in hit}
    for choice in itertools.product(*[range(len(r[2])) for r in rows]):
        cpt = {(r[0], r[1]): r[2][c] for r, c in zip(rows, choice)}
        joint = np.empty(len(states))
        for k, st in enumerate(states):
            asg = dict(zip(names, st))
            pr = 1.0
            for v in names:
                pr *= cpt[(v, tuple(asg[p] for p in parents[v]))][asg[v]]
            joint[k] = pr
        pe = joint[ev_mask].sum()
        if pe <= 1e-15:
            continue
        for a, h in hit.items():
            r = joint[ev_mask & h].sum() / pe
            best[a][0] = min(best[a][0], r)
            best[a][1] = max(best[a][1], r)
    return best


_SMALL = [os.path.join(_ROOT, "benchmarks", "tree_small_fr",
                       f"tree_fr_n5_{i}.lcn") for i in (1, 2, 3)] + \
         [os.path.join(_ROOT, "benchmarks", "polytree_small_fr",
                       f"polytree_fr_n5_{i}.lcn") for i in (3, 4)] + \
         [os.path.join(_ROOT, "examples", "alarm.lcn")]   # compound node C-D


@pytest.mark.parametrize("path", _SMALL)
@pytest.mark.parametrize("with_evidence", [False, True])
def test_matches_strong_extension_definition(path, with_evidence):
    cnv = _cnv(path)
    atoms = sorted(cnv.lcn.atoms)
    rng = random.Random(path)
    ev = {rng.choice(atoms): rng.randint(0, 1)} if with_evidence else {}
    res = _quiet(CredalVE(cnv).run, evidence=ev, verbosity=0)
    ref = _bruteforce(cnv, ev)
    assert set(res) == set(ref)
    for a, (lo, hi) in ref.items():
        assert res[a][0][1] == pytest.approx(lo, abs=1e-9), (a, "lower")
        assert res[a][1][1] == pytest.approx(hi, abs=1e-9), (a, "upper")


def test_evidence_on_a_descendant(tmp_path):
    # A -> B, P(A)=.5, P(B|A)=.9, P(B|!A)=.1: P(A | B=1) = 0.9 exactly.
    f = tmp_path / "ab.lcn"
    f.write_text("s1: 0.5 <= P(A) <= 0.5\n"
                 "s2: 0.9 <= P(B | A) <= 0.9\n"
                 "s3: 0.1 <= P(B | !A) <= 0.1\n")
    res = _quiet(CredalVE(_cnv(str(f))).run, evidence={"B": 1}, verbosity=0)
    assert res["A"][0][1] == pytest.approx(0.9)
    assert res["A"][1][1] == pytest.approx(0.9)


def test_both_elimination_orders_agree():
    cnv = _cnv(os.path.join(_ROOT, "examples", "alarm.lcn"))
    a = _quiet(CredalVE(cnv).run, evidence={"E": 1}, elim_heuristic="min-fill",
               verbosity=0)
    b = _quiet(CredalVE(cnv).run, evidence={"E": 1},
               elim_heuristic="topological", verbosity=0)
    for k in a:
        assert np.allclose(a[k][0], b[k][0]) and np.allclose(a[k][1], b[k][1])


def test_removed_options_are_rejected():
    cnv = _cnv(os.path.join(_ROOT, "examples", "alarm.lcn"))
    for kw in ({"coupling": "cross-family"}, {"n_clusters": 10},
               {"cluster_representative": "mean"}):
        with pytest.raises(TypeError):
            CredalVE(cnv).run(evidence={}, verbosity=0, **kw)


# ----------------------------------------------------------------------
# Pair-potential primitives
# ----------------------------------------------------------------------

def test_product_and_sum_out():
    rng = np.random.default_rng(0)
    cards = {"A": 2, "B": 3, "C": 2}
    a = PairPotential(["A", "B"], cards, rng.random((3, 2, 3)),
                      rng.random((3, 2, 3)))
    b = PairPotential(["B", "C"], cards, rng.random((4, 3, 2)),
                      rng.random((4, 3, 2)))
    c = a.product(b)
    assert c.scope == ["A", "B", "C"] and c.P.shape == (12, 2, 3, 2)
    for i, j in itertools.product(range(3), range(4)):
        k = i * 4 + j
        exp_p = a.P[i][:, :, None] * b.P[j][None, :, :]
        exp_q = a.Q[i][:, :, None] * b.Q[j][None, :, :]
        assert np.allclose(c.P[k], exp_p) and np.allclose(c.Q[k], exp_q)
    s = c.sum_out("B")
    assert s.scope == ["A", "C"] and np.allclose(s.P, c.P.sum(axis=2))


def test_dominance_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(100):
        m, d = int(rng.integers(2, 25)), int(rng.integers(1, 5))
        A = rng.integers(0, 4, (m, d)).astype(float)
        B = rng.integers(0, 4, (m, d)).astype(float)
        brute = np.array([
            any(j != i and np.all(A[j] <= A[i]) and np.all(B[j] >= B[i])
                and not (np.all(A[j] == A[i]) and np.all(B[j] == B[i]))
                for j in range(m))
            for i in range(m)])
        assert np.array_equal(brute, _dominated(A, B, d))


def test_prune_both_preserves_both_extremes():
    rng = np.random.default_rng(2)
    for _ in range(200):
        m = int(rng.integers(2, 40))
        pot = PairPotential([], {}, rng.random(m), rng.random(m))
        assert np.allclose(pot.bounds(), pot.prune("both").bounds())


def test_hull_prune_keeps_every_linear_fractional_extremum():
    # Query-free prune keeps only convex-hull vertices; any ratio
    # <q, f> / <p + q, f> (linear-fractional in f) must keep its min and max.
    from lcn.inference.marginal.cn.pair_potentials import hull_vertices
    rng = np.random.default_rng(3)
    for _ in range(200):
        m, d = int(rng.integers(3, 60)), int(rng.integers(1, 6))
        F = rng.random((m, d))
        if rng.random() < 0.5:                       # conditional tables
            F /= F.sum(axis=1, keepdims=True)
        keep = hull_vertices(F)
        for _ in range(5):
            a, b = rng.random(d), rng.random(d) + 0.1
            r = (F @ a) / (F @ b)
            assert np.isclose(r[keep].max(), r.max())
            assert np.isclose(r[keep].min(), r.min())


def test_lp_hull_vertices_superset_of_vertices_and_sound():
    # The LP test must keep every true vertex (soundness of the prune) and
    # every dropped row must lie in the hull of the kept ones.
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull
    from lcn.inference.marginal.cn.pair_potentials import _lp_hull_vertices
    rng = np.random.default_rng(4)
    for _ in range(20):
        r, m = int(rng.integers(2, 5)), int(rng.integers(10, 80))
        Y = rng.standard_normal((m, r))
        keep = _lp_hull_vertices(Y)
        assert set(ConvexHull(Y).vertices) <= set(keep)
        for i in set(range(m)) - set(keep):
            A = np.vstack([Y[keep].T, np.ones(len(keep))])
            res = linprog(np.zeros(len(keep)), A_eq=A,
                          b_eq=np.r_[Y[i], 1.0], bounds=(0, None),
                          method="highs")
            assert res.status == 0


def test_zero_probability_vertices_are_conditioned_away(tmp_path):
    # P(B | A) has the vertex 0, so some vertex products give B=1 probability
    # zero; they must be dropped (regular conditioning), not read as 0/0.
    # A -> B: P(A) in [.2, .6], P(B | A) in [0, .5], P(B | !A) = .3.
    f = tmp_path / "zero.lcn"
    f.write_text("s1: 0.2 <= P(A) <= 0.6\n"
                 "s2: 0.0 <= P(B | A) <= 0.5\n"
                 "s3: 0.3 <= P(B | !A) <= 0.3\n")
    cnv = _cnv(str(f))
    res = _quiet(CredalVE(cnv).run, evidence={"B": 1}, verbosity=0)
    ref = _bruteforce(cnv, {"B": 1})
    assert res["A"][0][1] == pytest.approx(ref["A"][0], abs=1e-9)
    assert res["A"][1][1] == pytest.approx(ref["A"][1], abs=1e-9)
    # P(A | B) = .6*.5 / (.6*.5 + .4*.3) = 0.714285...; lower 0 (P(B|A)=0)
    assert res["A"][0][1] == pytest.approx(0.0, abs=1e-12)
    assert res["A"][1][1] == pytest.approx(0.3 / 0.42, abs=1e-6)


def test_many_evidence_atoms_do_not_underflow():
    # 60 observed atoms on a 100-node tree: P(e) ~ 1e-32. A positive zero
    # threshold used to discard every pair and return [0, 1] everywhere.
    path = os.path.join(_ROOT, "benchmarks", "tree_large_fr",
                        "tree_fr_n100_1.lcn")
    cnv = _cnv(path)
    atoms = sorted(cnv.lcn.atoms)
    ev = {a: 1 for a in atoms[:60]}
    res = _quiet(CredalVE(cnv).run, evidence=ev, verbosity=0)
    informative = [a for a, (lo, hi) in res.items()
                   if not (lo[1] == 0.0 and hi[1] == 1.0)]
    assert len(informative) >= 20
