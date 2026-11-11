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

# Tests for ApproxLP (CDVE): coordinate descent over the local credal sets with
# an exact family step. The bounds must be INNER (inside the exact
# strong-extension interval, which CVE / brute force give), with and without
# evidence, for binary and compound atoms.

import contextlib
import io
import os
import random
import tempfile

import numpy as np
import pytest

from lcn.inference.marginal.cn.approxlp import ApproxLP
from lcn.inference.marginal.cn.cve import CredalVE
from test_cve import _bruteforce, _cnv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _p(*parts):
    return os.path.join(_ROOT, *parts)


_CASES = [_p("benchmarks", "tree_small_fr", "tree_fr_n5_1.lcn"),
          _p("benchmarks", "polytree_small_fr", "polytree_fr_n5_3.lcn"),
          _p("benchmarks", "dag_small_fr", "dag_fr_tw3_n6_1.lcn"),
          _p("examples", "alarm.lcn"),                    # compound C-D
          _p("examples", "ivbp_compound_mid.lcn")]        # compound B-C


@pytest.mark.parametrize("path", _CASES)
@pytest.mark.parametrize("n_ev", [0, 1, 2])
def test_inner_and_attained_vs_strong_extension(path, n_ev):
    cnv = _cnv(path)
    atoms = sorted(cnv.lcn.atoms)
    rng = random.Random(f"{path}{n_ev}")
    ev = {a: rng.randint(0, 1) for a in rng.sample(atoms, n_ev)}
    res = _quiet(ApproxLP(cnv).run, evidence=ev, verbosity=0)
    ref = _bruteforce(cnv, ev)
    for a, (lo, hi) in ref.items():
        alo, ahi = res[a][0][1], res[a][1][1]
        assert lo - 1e-9 <= alo <= ahi + 1e-12 <= hi + 2e-9, (a, lo, alo, ahi, hi)


def test_compound_atoms_and_compound_evidence():
    # alarm's C-D is a 4-state node: atoms are optimized directly (exact
    # here), and evidence on one of its atoms is accepted.
    cnv = _cnv(_p("examples", "alarm.lcn"))
    res = _quiet(ApproxLP(cnv).run, evidence={}, verbosity=0)
    assert res["D"][0][1] == pytest.approx(0.0, abs=1e-9)
    assert res["D"][1][1] == pytest.approx(1.0, abs=1e-9)
    res = _quiet(ApproxLP(cnv).run, evidence={"C": 1}, verbosity=0)
    assert "C" not in res and "D" in res


def test_diagnostic_evidence():
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "ab.lcn")
        with open(f, "w") as fh:
            fh.write("s1: 0.5 <= P(A) <= 0.5\n"
                     "s2: 0.9 <= P(B | A) <= 0.9\n"
                     "s3: 0.1 <= P(B | !A) <= 0.1\n")
        cnv = _cnv(f)
    res = _quiet(ApproxLP(cnv).run, evidence={"B": 1}, verbosity=0)
    assert res["A"][0][1] == pytest.approx(0.9)
    assert res["A"][1][1] == pytest.approx(0.9)


def test_family_step_is_exact_over_the_family():
    # The Dinkelbach family step must reach the optimum of the family's
    # linear-fractional program over the product of its rows' vertex sets.
    cnv = _cnv(_p("benchmarks", "dag_small_fr", "dag_fr_tw3_n6_1.lcn"))
    alp = ApproxLP(cnv)
    alp._build()
    rng = np.random.default_rng(0)
    import itertools
    for node in alp.nodes:
        V = alp.verts[node]
        if len(V) > 6:
            continue
        shape = (alp.cards[node],) + tuple(alp.cards[p]
                                            for p in alp.parents[node])
        for _ in range(5):
            d = rng.random(shape) + 0.05
            n = d * rng.random(shape)
            rows = [r.mean(axis=0) for r in V]
            for sense in ("min", "max"):
                vals = []
                for pick in itertools.product(*[range(len(r)) for r in V]):
                    N = sum(V[c][i] @ n[(slice(None),) + pc] for c, (i, pc)
                            in enumerate(zip(pick, alp.configs[node])))
                    D = sum(V[c][i] @ d[(slice(None),) + pc] for c, (i, pc)
                            in enumerate(zip(pick, alp.configs[node])))
                    vals.append(N / D)
                best = min(vals) if sense == "min" else max(vals)
                step = alp._family_step(node, n, d, rows, sense)
                got = step[1] if step is not None else None
                if got is not None:
                    assert got == pytest.approx(best, rel=1e-9)


def test_restarts_never_hurt():
    cnv = _cnv(_p("benchmarks", "polytree_small_fr", "polytree_fr_n7_1.lcn"))
    ev = {"x3": 1}
    r0 = _quiet(ApproxLP(cnv).run, evidence=ev, n_restarts=0, verbosity=0)
    r8 = _quiet(ApproxLP(cnv).run, evidence=ev, n_restarts=8, verbosity=0)
    ref = _quiet(CredalVE(cnv).run, evidence=ev, verbosity=0)
    for a in ref:
        assert r8[a][0][1] <= r0[a][0][1] + 1e-12
        assert r8[a][1][1] >= r0[a][1][1] - 1e-12
        assert r8[a][0][1] >= ref[a][0][1] - 1e-9
        assert r8[a][1][1] <= ref[a][1][1] + 1e-9


def test_coupling_option_removed():
    cnv = _cnv(_p("examples", "alarm.lcn"))
    with pytest.raises(TypeError):
        ApproxLP(cnv).run(evidence={}, coupling="cross-family", verbosity=0)
