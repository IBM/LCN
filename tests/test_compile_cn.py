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

from lcn.core.model import LCN
from lcn.inference.marginal.cn.compile_cn import compile_lcn_to_cn
from lcn.inference.marginal.cn.credal_network import CredalNetwork
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_CANCER = os.path.join(_ROOT, "examples", "cancer.lcn")


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _build(n_jobs=1):
    lcn = LCN()
    _quiet(lcn.from_lcn, file_name=_CANCER)
    cn = _quiet(CredalNetwork.from_lcn, lcn, method="linear",
                n_jobs=n_jobs, verbosity=0)
    return lcn, cn


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


def test_save_load_round_trip(tmp_path):
    """load_cn(save_cn(cn)) reproduces the built CredalNetwork exactly."""
    lcn, cn = _build()
    out = str(tmp_path / "cancer.cn")
    cn.save_cn(out, method="linear", merge_budget=1, solver="ipopt")

    reloaded = CredalNetwork.load_cn(out, lcn)

    assert reloaded.nodes == cn.nodes
    assert reloaded.node_atoms == cn.node_atoms
    assert reloaded.node_card == cn.node_card
    # node_card values must be ints (JSON preserves this; guard the cast).
    assert all(isinstance(v, int) for v in reloaded.node_card.values())
    # interpretations must be tuples again, not JSON lists.
    for factor in reloaded.factors:
        for entry in factor.values():
            assert isinstance(entry["interpretation"], tuple)
    assert _factor_bounds(reloaded) == _factor_bounds(cn)


def test_compile_cli_default_output(tmp_path):
    """compile_lcn_to_cn writes a .cn next to the input and matches a build."""
    # Copy the example into tmp so the default-output path lands in tmp_path.
    src = tmp_path / "cancer.lcn"
    src.write_text(open(_CANCER).read())

    out = _quiet(compile_lcn_to_cn, str(src), n_jobs=1, verbosity=0)
    assert out == str(tmp_path / "cancer.cn")
    assert os.path.exists(out)

    lcn, cn = _build()
    reloaded = CredalNetwork.load_cn(out, lcn)
    assert _factor_bounds(reloaded) == _factor_bounds(cn)


def test_parallel_matches_serial():
    """n_jobs>1 must give identical factors to the serial build."""
    _, serial = _build(n_jobs=1)
    _, parallel = _build(n_jobs=4)
    assert parallel.nodes == serial.nodes
    assert _factor_bounds(parallel) == _factor_bounds(serial)


def test_compile_time_round_trips(tmp_path):
    """compile_time survives save->load and is readable via cn_metadata."""
    lcn, cn = _build()
    out = str(tmp_path / "cancer.cn")
    cn.save_cn(out, method="linear", merge_budget=1, solver="ipopt",
               compile_time=12.5, n_jobs=4)

    reloaded = CredalNetwork.load_cn(out, lcn)
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


def test_vertices_cache_hit_and_param_mismatch(tmp_path):
    """CredalNetworkVertices.from_lcn loads a matching .cn (and its
    compile_time), and ignores a .cn compiled with different params."""
    # Compile once, alongside a copy of the .lcn so the default cache path lands
    # in tmp_path (foo.lcn -> foo.cn).
    src = tmp_path / "cancer.lcn"
    src.write_text(open(_CANCER).read())
    _quiet(compile_lcn_to_cn, str(src), n_jobs=1, verbosity=0)

    lcn = LCN()
    _quiet(lcn.from_lcn, file_name=str(src))

    # (1) Matching build -> cache hit, compile_time surfaced.
    hit = _quiet(CredalNetworkVertices.from_lcn, lcn, method="linear",
                 merge_budget=1, solver="ipopt",
                 lcn_file=str(src), cache=True, verbosity=0)
    assert hit.loaded_from_cache is True
    assert hit.compile_time is not None

    # Cached build must yield the same interval factors as a fresh compile.
    fresh = _quiet(CredalNetworkVertices.from_lcn, lcn, method="linear",
                   merge_budget=1, solver="ipopt", cache=False, verbosity=0)
    assert _factor_bounds(hit.cn) == _factor_bounds(fresh.cn)

    # (2) Different merge_budget -> must NOT load the cache.
    miss = _quiet(CredalNetworkVertices.from_lcn, lcn, method="linear",
                  merge_budget=3, solver="ipopt",
                  lcn_file=str(src), cache=True, verbosity=0)
    assert miss.loaded_from_cache is False

    # (3) cache=False -> must recompute even when a matching .cn exists.
    forced = _quiet(CredalNetworkVertices.from_lcn, lcn, method="linear",
                    merge_budget=1, solver="ipopt",
                    lcn_file=str(src), cache=False, verbosity=0)
    assert forced.loaded_from_cache is False
