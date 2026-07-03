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

# Tests for CredalVE's CCTE-style potential clustering / representative
# approximation (n_clusters, cluster_representative), which reuse
# Potential.cluster_prune. Two properties:
#   (1) n_clusters=0 (default) is a no-op: identical marginals to plain CVE.
#   (2) n_clusters>0 runs with either representative and returns valid,
#       [0,1]-contained bounds (a deliberate approximation, not necessarily
#       exact -- so we only assert validity/containment, not tightness).

import contextlib
import io
import os

import numpy as np
import pytest

from lcn.core.model import LCN
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.marginal.cn.cve import CredalVE

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_BITING = os.path.join(_ROOT, "examples", "d4_biting.lcn")


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


@pytest.fixture(scope="module")
def cnv():
    lcn = LCN()
    _quiet(lcn.from_lcn, file_name=_BITING)
    return _quiet(CredalNetworkVertices.from_lcn, lcn,
                  method="linear", verbosity=0)


def _run(cnv, **kw):
    return _quiet(CredalVE(cnv=cnv).run, evidence={},
                  elim_heuristic="min-fill", coupling="off", verbosity=0, **kw)


def test_n_clusters_zero_is_noop(cnv):
    """n_clusters=0 must reproduce plain CVE exactly (default path unchanged)."""
    base = _run(cnv)
    clustered_off = _run(cnv, n_clusters=0, cluster_representative="plub")
    assert set(base) == set(clustered_off)
    for a in base:
        assert np.allclose(base[a][0], clustered_off[a][0], atol=1e-9)
        assert np.allclose(base[a][1], clustered_off[a][1], atol=1e-9)


@pytest.mark.parametrize("rep", ["plub", "mean"])
def test_clustering_runs_and_bounds_valid(cnv, rep):
    """n_clusters>0 with either representative runs and yields valid bounds."""
    res = _run(cnv, n_clusters=2, cluster_representative=rep)
    assert res, "clustered CVE produced no marginals"
    for a, (lo, hi) in res.items():
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        assert np.all(lo <= hi + 1e-9), f"{a}: lo>hi"
        assert np.all(lo >= -1e-9) and np.all(hi <= 1 + 1e-9), f"{a}: out of [0,1]"


def test_bad_representative_rejected(cnv):
    """The cluster_representative guard rejects an unknown value."""
    with pytest.raises(AssertionError):
        _run(cnv, n_clusters=2, cluster_representative="bogus")


def test_clustering_composes_with_epsilon(cnv):
    """Clustering + epsilon-approximate pruning together run and stay valid."""
    res = _run(cnv, n_clusters=2, cluster_representative="mean", epsilon=0.05)
    assert res
    for a, (lo, hi) in res.items():
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        assert np.all(lo <= hi + 1e-9)
        assert np.all(lo >= -1e-9) and np.all(hi <= 1 + 1e-9)
