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

# Tests for the FPTAS of CredalVE (Maua, de Campos & Zaffalon, IJAR 2012):
# alpha-quotient primitive, its covering guarantee, and the end-to-end
# (1 + epsilon) inner-bound guarantee against exact CVE.

import contextlib
import io
import os
import random

import numpy as np
import pytest

from lcn.core.model import LCN
from lcn.inference.marginal.cn.cve import CredalVE
from lcn.inference.marginal.cn.pair_potentials import PairPotential
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOL = 1e-12


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _cnv(path):
    lcn = LCN()
    _quiet(lcn.from_lcn, file_name=path)
    return _quiet(CredalNetworkVertices.from_lcn, lcn, method="linear",
                  verbosity=0), lcn


def _rand_pot(rng, m, d, zeros=True):
    P = rng.random((m, d))
    Q = rng.random((m, d))
    if zeros:
        P[rng.random((m, d)) < 0.1] = 0.0
        Q[rng.random((m, d)) < 0.1] = 0.0
    P[:, 0] += 1e-3                     # keep every pair nonzero
    return PairPotential(["X"], {"X": d}, P, Q)


# ----------------------------------------------------------------------
# alpha-quotient
# ----------------------------------------------------------------------

def test_quotient_keeps_real_pairs_and_shrinks():
    rng = np.random.default_rng(0)
    pot = _rand_pot(rng, 400, 1)          # 2 coordinates: classes are dense
    q = pot.quotient(1.5)
    assert q.m < pot.m
    rows = {tuple(np.r_[p, s]) for p, s in zip(pot.P, pot.Q)}
    assert all(tuple(np.r_[p, s]) in rows for p, s in zip(q.P, q.Q))


def test_quotient_alpha_one_is_identity():
    rng = np.random.default_rng(1)
    pot = _rand_pot(rng, 50, 2)
    assert pot.quotient(1.0) is pot


@pytest.mark.parametrize("alpha", [1.01, 1.2, 2.0])
def test_quotient_covers_every_odds_within_alpha_squared(alpha):
    # For every weight pair (a, b) >= 0 the max / min of <a,q>/<b,p> over the
    # kept set is within a factor alpha^2 of the max / min over the full set.
    rng = np.random.default_rng(2)
    for _ in range(30):
        m, d = int(rng.integers(5, 300)), int(rng.integers(1, 4))
        pot = _rand_pot(rng, m, d)
        q = pot.quotient(alpha)
        for _ in range(10):
            a, b = rng.random(d), rng.random(d) + 0.05
            full = (pot.Q @ a) / (pot.P @ b)
            kept = (q.Q @ a) / (q.P @ b)
            assert kept.max() * alpha ** 2 >= full.max() * (1 - _TOL)
            assert kept.min() <= full.min() * alpha ** 2 * (1 + _TOL)
            assert kept.max() <= full.max() * (1 + _TOL)     # inner
            assert kept.min() >= full.min() * (1 - _TOL)


def test_quotient_merges_rescaled_copies():
    # q/(p+q) is scale invariant, so c*(p, q) is one class with (p, q)
    # (up to floating-point ties on a grid boundary).
    rng = np.random.default_rng(3)
    P, Q = rng.random((10, 4)), rng.random((10, 4))
    pot = PairPotential(["X"], {"X": 4}, np.r_[P, 3 * P], np.r_[Q, 3 * Q])
    assert pot.quotient(1.05).m <= 11


def test_composition_error_stays_within_budget():
    # Random chain: N quotients interleaved with products and sum-outs; the
    # final ratio bounds stay within alpha^(2N) of the exact ones.
    rng = np.random.default_rng(4)
    cards = {"A": 2, "B": 2, "C": 2}
    for _ in range(20):
        def pot(scope, m):
            shape = (m,) + tuple(cards[v] for v in scope)
            P = rng.random(shape) + 1e-3
            return PairPotential(scope, cards, P, rng.random(shape) + 1e-3)
        f1, f2, f3 = pot(["A", "B"], 15), pot(["B", "C"], 15), pot(["C"], 10)
        alpha = 1.1
        ex = f1.product(f2).sum_out("B").product(f3).sum_out("A").sum_out("C")
        ap = f1.product(f2).quotient(alpha).sum_out("B")
        ap = ap.product(f3).quotient(alpha).sum_out("A").sum_out("C")
        ap = ap.quotient(alpha)
        n = 3
        lo, hi = ex.bounds()
        alo, ahi = ap.bounds()
        assert hi / alpha ** (2 * n) - _TOL <= ahi <= hi + _TOL
        assert lo - _TOL <= alo <= lo * alpha ** (2 * n) + _TOL


# ----------------------------------------------------------------------
# End to end: inner bounds within (1 + eps)
# ----------------------------------------------------------------------

_CASES = [os.path.join(_ROOT, "benchmarks", "tree_small_fr", f)
          for f in ("tree_fr_n5_1.lcn", "tree_fr_n8_1.lcn")] + \
         [os.path.join(_ROOT, "benchmarks", "polytree_small_fr", f)
          for f in ("polytree_fr_n5_3.lcn", "polytree_fr_n7_1.lcn")] + \
         [os.path.join(_ROOT, "benchmarks", "dag_small_fr",
                       "dag_fr_tw3_n7_1.lcn"),
          os.path.join(_ROOT, "benchmarks", "ktree_small_fr",
                       "ktree_fr_k3_n6_1.lcn"),
          os.path.join(_ROOT, "examples", "alarm.lcn")]


@pytest.mark.parametrize("path", _CASES)
@pytest.mark.parametrize("with_evidence", [False, True])
def test_fptas_inner_bounds_within_one_plus_eps(path, with_evidence):
    cnv, lcn = _cnv(path)
    rng = random.Random(path)
    ev = {rng.choice(sorted(lcn.atoms)): rng.randint(0, 1)} \
        if with_evidence else {}
    exact = _quiet(CredalVE(cnv).run, evidence=ev, verbosity=0)
    for eps in (1.0, 0.1, 0.01):
        eng = CredalVE(cnv)
        apx = _quiet(eng.run, evidence=ev, epsilon=eps, verbosity=0)
        assert eng.alpha > 1.0 and eng.n_quotient_ops >= 1
        for a in exact:
            lo, hi = exact[a][0][1], exact[a][1][1]
            alo, ahi = apx[a][0][1], apx[a][1][1]
            assert lo - _TOL <= alo <= (1 + eps) * lo + _TOL, (a, eps)
            assert hi / (1 + eps) - _TOL <= ahi <= hi + _TOL, (a, eps)


def test_epsilon_zero_and_none_are_exact():
    cnv, _ = _cnv(os.path.join(_ROOT, "examples", "alarm.lcn"))
    a = _quiet(CredalVE(cnv).run, evidence={"E": 1}, verbosity=0)
    b = _quiet(CredalVE(cnv).run, evidence={"E": 1}, epsilon=0.0, verbosity=0)
    for k in a:
        assert np.array_equal(a[k][0], b[k][0])
        assert np.array_equal(a[k][1], b[k][1])
    eng = CredalVE(cnv)
    _quiet(eng.run, evidence={}, epsilon=None, verbosity=0)
    assert eng.alpha == 1.0 and eng.guarantee == "exact"
