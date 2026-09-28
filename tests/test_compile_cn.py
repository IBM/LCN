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

# Tests for the credal-network compiler: CredalNetwork.save_cn/load_cn and the
# compile_cn CLI helper. Three properties:
#   (1) save_cn -> load_cn round-trips: reloaded factors/structure are
#       bit-identical to the freshly built CredalNetwork (interpretations are
#       tuples again, node_card ints, bounds unchanged).
#   (2) compile_lcn_to_cn writes a .cn alongside the input by default and its
#       contents match a direct build.
#   (3) n_jobs>1 produces output identical to the serial n_jobs=1 build.

import contextlib
import io
import os

import pytest

from lcn.core.model import LCN
from lcn.inference.marginal.cn.compile_cn import compile_lcn_to_cn
from lcn.inference.marginal.cn.credal_network import CredalNetwork
from lcn.inference.marginal.cn.vertices import (
    CredalNetworkVertices, _cache_matches,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_CANCER = os.path.join(_ROOT, "examples", "cancer.lcn")


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _factor_bounds(cn):
    """Structural + numeric fingerprint of a CredalNetwork's factors."""
    out = []
    for factor in cn.factors:
        entries = []
        for i in sorted(factor.keys()):
            e = factor[i]
            entries.append((
                e["child"], tuple(e["scope"]), tuple(e["interpretation"]),
                round(e["lobo"], 9), round(e["upbo"], 9)))
        out.append(entries)
    return out


@pytest.fixture(scope="module")
def prebuilt(tmp_path_factory):
    """Compile cancer.lcn ONCE for the whole module (the per-family ipopt solves
    are the expensive step). Returns a copy of the .lcn in a shared tmp dir, a
    compiled .cn next to it, the serial-build CredalNetwork, and its factor-bound
    fingerprint. Tests reuse this instead of re-solving, which turns most of them
    into (fast) cache loads + reads."""
    d = tmp_path_factory.mktemp("compile_cn")
    src = d / "cancer.lcn"
    src.write_text(open(_CANCER).read())

    lcn = LCN()
    _quiet(lcn.from_lcn, file_name=str(src))
    cn = _quiet(CredalNetwork.from_lcn, lcn, method="linear",
                n_jobs=1, verbosity=0)
    cn_path = str(d / "cancer.cn")
    cn.save_cn(cn_path, method="linear", merge_budget=1, solver="ipopt",
               compile_time=1.0, n_jobs=1)
    return {
        "dir": d,
        "src": str(src),
        "cn_path": cn_path,
        "lcn": lcn,
        "cn": cn,
        "bounds": _factor_bounds(cn),
    }


def test_save_load_round_trip(prebuilt):
    """load_cn(save_cn(cn)) reproduces the built CredalNetwork exactly."""
    reloaded = CredalNetwork.load_cn(prebuilt["cn_path"], prebuilt["lcn"])
    cn = prebuilt["cn"]

    assert reloaded.nodes == cn.nodes
    assert reloaded.node_atoms == cn.node_atoms
    assert reloaded.node_card == cn.node_card
    # node_card values must be ints (JSON preserves this; guard the cast).
    assert all(isinstance(v, int) for v in reloaded.node_card.values())
    # interpretations must be tuples again, not JSON lists.
    for factor in reloaded.factors:
        for entry in factor.values():
            assert isinstance(entry["interpretation"], tuple)
    assert _factor_bounds(reloaded) == prebuilt["bounds"]


def test_compile_cli_default_output(prebuilt):
    """compile_lcn_to_cn writes a .cn next to the input and matches the build."""
    # The prebuilt fixture already compiled via CredalNetwork.from_lcn; here we
    # drive the CLI helper on a separate copy and confirm the file is written
    # next to the input and its factors match the reference build.
    src = prebuilt["dir"] / "cli_cancer.lcn"
    src.write_text(open(_CANCER).read())

    out = _quiet(compile_lcn_to_cn, str(src), n_jobs=1, verbosity=0)
    assert out == str(prebuilt["dir"] / "cli_cancer.cn")
    assert os.path.exists(out)

    reloaded = CredalNetwork.load_cn(out, prebuilt["lcn"])
    assert _factor_bounds(reloaded) == prebuilt["bounds"]


def test_parallel_matches_serial(prebuilt):
    """n_jobs>1 must give identical factors to the serial build."""
    # One parallel compile, compared against the module's serial build (already
    # paid by the fixture) -- so this costs a single extra compile, not two.
    parallel = _quiet(CredalNetwork.from_lcn, prebuilt["lcn"], method="linear",
                      n_jobs=4, verbosity=0)
    assert parallel.nodes == prebuilt["cn"].nodes
    assert _factor_bounds(parallel) == prebuilt["bounds"]


def test_parallel_matches_serial_linear_tight(prebuilt):
    """n_jobs>1 must give identical factors to the serial build on the
    "linear-tight" path too -- this exercises the shared-LCN Local Markov
    Condition (pre-computed before the thread pool) and the bilinear solves,
    the paths most sensitive to a threading race."""
    lcn = prebuilt["lcn"]
    serial = _quiet(CredalNetwork.from_lcn, lcn, method="linear-tight",
                    n_jobs=1, verbosity=0)
    parallel = _quiet(CredalNetwork.from_lcn, lcn, method="linear-tight",
                      n_jobs=4, verbosity=0)
    assert parallel.nodes == serial.nodes
    assert _factor_bounds(parallel) == _factor_bounds(serial)


def test_compile_time_round_trips(prebuilt, tmp_path):
    """compile_time survives save->load and is readable via cn_metadata."""
    cn = prebuilt["cn"]
    out = str(tmp_path / "cancer.cn")
    cn.save_cn(out, method="linear", merge_budget=1, solver="ipopt",
               compile_time=12.5, n_jobs=4)

    reloaded = CredalNetwork.load_cn(out, prebuilt["lcn"])
    assert reloaded.compile_time == 12.5

    meta = CredalNetwork.cn_metadata(out)
    assert meta is not None
    assert meta["compile_time"] == 12.5
    assert meta["method"] == "linear"
    assert meta["merge_budget"] == 1
    assert meta["solver"] == "ipopt"
    assert meta["n_jobs"] == 4


def test_cn_metadata_none_on_bad_input(tmp_path):
    """cn_metadata returns None for a missing file and non-.cn JSON."""
    assert CredalNetwork.cn_metadata(str(tmp_path / "nope.cn")) is None

    not_a_cn = tmp_path / "other.cn"
    not_a_cn.write_text('{"format": "something-else", "version": 1}')
    assert CredalNetwork.cn_metadata(str(not_a_cn)) is None

    not_json = tmp_path / "garbage.cn"
    not_json.write_text("this is not json")
    assert CredalNetwork.cn_metadata(str(not_json)) is None


def test_vertices_cache_hit_and_param_mismatch(prebuilt):
    """CredalNetworkVertices.from_lcn loads the prebuilt .cn (and its
    compile_time), and ignores a .cn compiled with different params."""
    src = prebuilt["src"]
    lcn = prebuilt["lcn"]

    # (1) Matching build -> cache hit, compile_time surfaced (no re-solve: the
    # .cn from the fixture is loaded, only the fast LRS enumeration runs).
    hit = _quiet(CredalNetworkVertices.from_lcn, lcn, method="linear",
                 merge_budget=1, solver="ipopt",
                 lcn_file=src, cache=True, verbosity=0)
    assert hit.loaded_from_cache is True
    assert hit.compile_time is not None
    # Cached intervals must match the reference build's factors.
    assert _factor_bounds(hit.cn) == prebuilt["bounds"]

    # (2) Different params -> the .cn must NOT match. Assert the cache DECISION
    # directly on the header (via _cache_matches); a full from_lcn(merge_budget=3)
    # would fall through to an expensive merged compile just to observe the miss.
    cnmeta = CredalNetwork.cn_metadata(prebuilt["cn_path"])
    assert cnmeta is not None
    assert _cache_matches(cnmeta, "linear", 1, "ipopt") is True
    assert _cache_matches(cnmeta, "linear", 3, "ipopt") is False
    assert _cache_matches(cnmeta, "linear-tight", 1, "ipopt") is False
    assert _cache_matches(cnmeta, "linear", 1, "scip") is False

    # (3) cache=False -> must recompute even when a matching .cn exists. Same
    # params (merge_budget=1) keeps the fallback compile cheap.
    forced = _quiet(CredalNetworkVertices.from_lcn, lcn, method="linear",
                    merge_budget=1, solver="ipopt",
                    lcn_file=src, cache=False, verbosity=0)
    assert forced.loaded_from_cache is False
