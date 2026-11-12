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

# Tests for IntervalBP: credal sum-product on the factor graph. On polytrees it
# must reproduce the strong-extension definition (enumerate every product of
# local extreme points, condition, take min/max) with and without evidence,
# for binary and compound (multi-valued) variables.

import contextlib
import io
import os
import random

import numpy as np
import pytest

from lcn.inference.marginal.cn.cve import CredalVE
from lcn.inference.marginal.cn.ivbp import (
    IntervalBP, _minkowski_sum, _product)
from test_cve import _bruteforce, _cnv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _p(*parts):
    return os.path.join(_ROOT, *parts)


_POLY = [_p("benchmarks", "tree_small_fr", f"tree_fr_n5_{i}.lcn")
         for i in (1, 2)] + \
        [_p("benchmarks", "polytree_small_fr", f"polytree_fr_n5_{i}.lcn")
         for i in (1, 3)] + \
        [_p("examples", "alarm.lcn"),                    # compound leaf C-D
         _p("examples", "ivbp_compound_mid.lcn")]        # compound middle B-C


@pytest.mark.parametrize("path", _POLY)
@pytest.mark.parametrize("n_ev", [0, 1, 2])
def test_polytree_matches_strong_extension(path, n_ev):
    cnv = _cnv(path)
    atoms = sorted(cnv.lcn.atoms)
    rng = random.Random(f"{path}{n_ev}")
    ev = {a: rng.randint(0, 1) for a in rng.sample(atoms, n_ev)}
    eng = IntervalBP(cnv)
    res = _quiet(eng.run, evidence=ev, verbosity=0)
    assert eng.singly_connected and eng.exact and eng.converged
    ref = _bruteforce(cnv, ev)
    assert set(ref) <= set(res)
    for a, (lo, hi) in ref.items():
        assert res[a][0][1] == pytest.approx(lo, abs=1e-9), (a, "lower")
        assert res[a][1][1] == pytest.approx(hi, abs=1e-9), (a, "upper")


@pytest.mark.parametrize("ev", [{}, {"H": 1}, {"F": 1, "H": 1},
                                {"C": 0}, {"D": 0}])
def test_compound_parents_match_cve(ev):
    # Two 4-state compound parents of one child: lambda messages INTO compound
    # variables; CVE (exact under evidence) is the reference.
    cnv = _cnv(_p("examples", "ivbp_compound_parents.lcn"))
    res = _quiet(IntervalBP(cnv).run, evidence=ev, verbosity=0)
    ref = _quiet(CredalVE(cnv).run, evidence=ev, verbosity=0)
    for a in ref:
        assert res[a][0][1] == pytest.approx(ref[a][0][1], abs=1e-9), a
        assert res[a][1][1] == pytest.approx(ref[a][1][1], abs=1e-9), a


def test_diagnostic_evidence():
    # A -> B, P(A)=.5, P(B|A)=.9, P(B|!A)=.1, B=1: P(A=1 | B=1) = 0.9 (the
    # previous downward-only IVBP returned 0.5).
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "ab.lcn")
        with open(f, "w") as fh:
            fh.write("s1: 0.5 <= P(A) <= 0.5\n"
                     "s2: 0.9 <= P(B | A) <= 0.9\n"
                     "s3: 0.1 <= P(B | !A) <= 0.1\n")
        cnv = _cnv(f)
    res = _quiet(IntervalBP(cnv).run, evidence={"B": 1}, verbosity=0)
    assert res["A"][0][1] == pytest.approx(0.9)
    assert res["A"][1][1] == pytest.approx(0.9)


def test_larger_polytree_matches_cve_with_evidence():
    cnv = _cnv(_p("benchmarks", "polytree_small_fr", "polytree_fr_n10_1.lcn"))
    ev = {"x1": 1, "x9": 1, "x6": 0}
    eng = IntervalBP(cnv)
    res = _quiet(eng.run, evidence=ev, verbosity=0)
    ref = _quiet(CredalVE(cnv).run, evidence=ev, verbosity=0)
    assert eng.exact
    for a in ref:
        assert res[a][0][1] == pytest.approx(ref[a][0][1], abs=1e-9)
        assert res[a][1][1] == pytest.approx(ref[a][1][1], abs=1e-9)


def test_loopy_graph_reports_no_guarantee():
    cnv = _cnv(_p("benchmarks", "dag_small_fr", "dag_fr_tw3_n10_1.lcn"))
    eng = IntervalBP(cnv)
    _quiet(eng.run, evidence={}, verbosity=0)
    assert not eng.singly_connected and not eng.exact
    assert eng.converged and eng.iterations >= 1


def test_removed_variational_method():
    cnv = _cnv(_p("examples", "alarm.lcn"))
    with pytest.raises(TypeError):
        IntervalBP(cnv).run(evidence={}, method="variational", verbosity=0)


# ----------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------

def test_product_of_hulls_is_exact():
    # normalize(a * b) over conv(A) x conv(B): extremes of every linear
    # functional are attained at products of vertices.
    rng = np.random.default_rng(0)
    for _ in range(50):
        k = int(rng.integers(2, 5))
        A = rng.dirichlet(np.ones(k), int(rng.integers(1, 6)))
        B = rng.dirichlet(np.ones(k), int(rng.integers(1, 6)))
        R = _product(A, B)
        la = rng.dirichlet(np.ones(len(A)), 200)
        lb = rng.dirichlet(np.ones(len(B)), 200)
        S = (la @ A) * (lb @ B)
        S /= S.sum(1, keepdims=True)
        for _ in range(5):
            c = rng.standard_normal(k)
            assert (S @ c).max() <= (R @ c).max() + 1e-9
            assert (S @ c).min() >= (R @ c).min() - 1e-9


def test_minkowski_sum_vertices():
    rng = np.random.default_rng(1)
    S1, S2 = rng.random((5, 3)), rng.random((4, 3))
    M = _minkowski_sum([S1, S2])
    full = (S1[:, None] + S2[None]).reshape(-1, 3)
    for _ in range(20):
        c = rng.standard_normal(3)
        assert np.isclose((M @ c).max(), (full @ c).max())


def test_impossible_evidence_is_vacuous_everywhere():
    """Upper P(e) = 0 (B is impossible, evidence B = 1): no member gives the
    evidence positive probability, so every atom is [0, 1], as for CVE and
    CDVE. The all-zero lambda message used to be replaced by the neutral
    message, which gave A = [0.4, 0.6] and C = [0.32, 0.48]."""
    cnv = _cnv(_p("examples", "ivbp_zero_evidence.lcn"))
    bp = IntervalBP(cnv)
    _quiet(bp.run, evidence={"B": 1}, verbosity=0)
    assert bp.empty and not bp.exact
    for a in ("A", "C"):
        assert bp.singleton_marginals[a] == (0.0, 1.0)
    ve = CredalVE(cnv)
    _quiet(ve.run, evidence={"B": 1}, verbosity=0)
    for a in ("A", "C"):
        assert tuple(ve.singleton_marginals[a]) == pytest.approx((0.0, 1.0))
