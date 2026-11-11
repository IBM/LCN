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

# Exact local credal sets of compound families (row_polytopes): the
# convex-hull method vs brute-force vertex enumeration, the relation to the
# interval box, the compiled network's use of them, and the effect on bounds.

import contextlib
import io
import itertools
import os
import warnings

import numpy as np
import pytest

from lcn.core.model import LCN
from lcn.inference.marginal.cn.credal_network import CredalNetwork
from lcn.inference.marginal.cn.cve import CredalVE
from lcn.inference.marginal.cn.row_polytopes import (
    RowPolytope, row_polytope_vertices)
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.marginal.exact import ExactInference

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _cnv(path):
    lcn = LCN()
    _quiet(lcn.from_lcn, file_name=path)
    return _quiet(CredalNetworkVertices.from_lcn, lcn, method="linear",
                  verbosity=0)


def _brute_vertices(G, h, k):
    """Vertices of {r >= 0, sum r = 1, G r <= h} by active-set enumeration."""
    rows = [(-np.eye(k)[i], 0.0) for i in range(k)] + list(zip(G, h))
    out = []
    for act in itertools.combinations(range(len(rows)), k - 1):
        M = np.vstack([np.ones(k)] + [rows[i][0] for i in act])
        rhs = np.array([1.0] + [rows[i][1] for i in act])
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        r = np.linalg.solve(M, rhs)
        if (r >= -1e-9).all() and (G @ r <= h + 1e-9).all():
            out.append(np.round(r, 9))
    return np.unique(np.array(out), axis=0) if out else np.zeros((0, k))


def test_hull_method_matches_brute_force_vertices():
    # Root-family polytopes: p in the simplex with random linear sentence rows.
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(40):
        k = int(rng.choice([3, 4]))
        G = rng.normal(size=(int(rng.integers(1, 4)), k))
        x0 = rng.dirichlet(np.ones(k))
        h = G @ x0 + rng.uniform(0.02, 0.3, size=len(G))   # x0 interior-ish
        A = np.vstack([-np.ones(k), np.ones(k), G])
        b = np.concatenate([[-1.0, 1.0], h])
        V, exact = row_polytope_vertices(RowPolytope(A, b, None, range(k), k))
        assert exact
        ref = _brute_vertices(G, h, k)
        got = np.unique(np.round(V, 9), axis=0)
        assert len(got) == len(ref)
        for v in ref:
            assert np.min(np.abs(got - v).max(axis=1)) < 1e-7
        checked += 1
    assert checked == 40


def test_polytope_inside_box_with_the_same_state_bounds():
    for f in ("fr_counter_box.lcn", "alarm.lcn", "ivbp_compound_mid.lcn"):
        cnv = _cnv(os.path.join(_ROOT, "examples", f))
        cn = cnv.cn
        assert cn.row_vertices, f
        for factor in cn.factors:
            e0 = next(iter(factor.values()))
            node = e0["child"]
            if node not in cn.row_vertices:
                continue
            k = len(e0["child_lst"])
            for key, verts in cn.row_vertices[node].items():
                V = np.array(verts)
                assert np.allclose(V.sum(axis=1), 1.0, atol=1e-9)
                want = dict(x.rsplit(":", 1) for x in key.split("|")) \
                    if key else {}
                for e in factor.values():
                    it = list(e["interpretation"])
                    par = dict(zip(e["scope"][k:], it[k:]))
                    if any(int(want[p]) != v for p, v in par.items()):
                        continue
                    s = int("".join(map(str, it[:k])), 2)
                    # inside the box, and the box is the polytope's bounds
                    assert V[:, s].min() >= e["lobo"] - 1e-7
                    assert V[:, s].max() <= e["upbo"] + 1e-7
                    assert V[:, s].min() == pytest.approx(e["lobo"], abs=1e-6)
                    assert V[:, s].max() == pytest.approx(e["upbo"], abs=1e-6)


def test_single_atom_children_keep_the_box():
    cnv = _cnv(os.path.join(_ROOT, "benchmarks", "tree_small_fr",
                            "tree_fr_n5_1.lcn"))
    assert cnv.cn.row_vertices == {}


def test_box_counterexample_now_matches_d_lcn():
    # Fully realizable, compound child, complete family graph: with exact
    # polytope rows the strong-extension bounds equal the LCN bounds.
    cnv = _cnv(os.path.join(_ROOT, "examples", "fr_counter_box.lcn"))
    cve = _quiet(CredalVE(cnv).run, evidence={}, verbosity=0)
    ei = ExactInference(lcn=cnv.lcn)
    ex = _quiet(ei.run, evidence={}, solver="scip", verbosity=0,
                time_limit=60, progress_bar=False)
    for a in cve:
        st = ei.status[a]
        if st["min"][2] == "confirmed":
            assert cve[a][0][1] == pytest.approx(ex[a][0][1], abs=1e-4), a
        if st["max"][2] == "confirmed":
            assert cve[a][1][1] == pytest.approx(ex[a][1][1], abs=1e-4), a
    assert cve["C"][0][1] == pytest.approx(0.32, abs=1e-4)
    assert cve["C"][1][1] == pytest.approx(0.58, abs=1e-4)


def test_compound_parent_keys_are_mapped():
    # smokers: S1-S2-S3 | F1-F2-F3 (8-state child, 8-state compound parent).
    cnv = _cnv(os.path.join(_ROOT, "examples", "smokers.lcn"))
    rv = cnv.cn.row_vertices["S1-S2-S3"]
    assert len(rv) == 8
    for cfg, verts in cnv.extreme_points["S1-S2-S3"].items():
        state = int(cfg[1:-1].rsplit(":", 1)[1])
        bits = [(state >> (2 - i)) & 1 for i in range(3)]
        key = "|".join(f"F{i + 1}:{b}" for i, b in enumerate(bits))
        assert verts == rv[key]


def test_row_vertices_survive_the_cn_round_trip(tmp_path):
    lcn = LCN()
    _quiet(lcn.from_lcn, file_name=os.path.join(_ROOT, "examples",
                                                "fr_counter_box.lcn"))
    cn = _quiet(CredalNetwork.from_lcn, lcn, verbosity=0)
    path = str(tmp_path / "box.cn")
    cn.save_cn(path, method="linear", merge_budget=1, solver="ipopt")
    back = CredalNetwork.load_cn(path, lcn)
    assert back.row_vertices == cn.row_vertices
    assert CredalNetwork.cn_metadata(path)["version"] == 3
