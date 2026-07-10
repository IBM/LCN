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

# Tests for the enumerate/.vtx layer: parallel LRS enumeration, save_vtx/
# vtx_metadata/load_extreme_points, and the transparent .vtx cache in
# CredalNetworkVertices.from_lcn. Properties:
#   (1) parallel (n_jobs>1) enumeration yields extreme points IDENTICAL to the
#       serial monolithic enumeration (the key correctness guarantee);
#   (2) save_vtx -> vtx_metadata/load_extreme_points round-trips the vertices,
#       enumeration_time, and compile_time;
#   (3) vtx_metadata returns None for missing / non-.vtx JSON;
#   (4) from_lcn(lcn_file=<matching .vtx>) loads the cache (loaded_vertices_from_
#       cache True, reuses enumeration_time), yields the same CVE marginals as a
#       fresh build, and a (method,merge_budget,solver) mismatch does NOT load.

import contextlib
import io
import os

import numpy as np

from lcn.core.model import LCN
from lcn.inference.marginal.cn.compile_cn import compile_lcn_to_cn
from lcn.inference.marginal.cn.vertices import CredalNetworkVertices
from lcn.inference.marginal.cn.cve import CredalVE

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# alarm has a compound (cardinality-4) node C-D, so it exercises the
# multi-vertex / multi-parent enumeration path, yet is small enough to be fast.
_ALARM = os.path.join(_ROOT, "examples", "alarm.lcn")


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _cnv(lcn_file, **kw):
    lcn = LCN()
    _quiet(lcn.from_lcn, file_name=lcn_file)
    return lcn, _quiet(CredalNetworkVertices.from_lcn, lcn, method="linear",
                       verbosity=0, **kw)


def test_parallel_enumeration_matches_serial():
    """n_jobs>1 must yield extreme points identical to the serial enumeration."""
    _, serial = _cnv(_ALARM, cache=False)
    _, parallel = _cnv(_ALARM, n_jobs=4, cache=False)
    assert parallel.extreme_points == serial.extreme_points
    # Parallel mode does not build the monolithic credal_net (no consumer reads
    # it); the interval BNs are still present for structure.
    assert parallel.credal_net is None
    assert parallel.bn_min is not None
    assert parallel.enumeration_time is not None


def test_save_vtx_round_trip(tmp_path):
    """save_vtx -> vtx_metadata/load_extreme_points reproduces the vertices."""
    _, cnv = _cnv(_ALARM, cache=False)
    out = str(tmp_path / "alarm.vtx")
    cnv.save_vtx(out, method="linear", merge_budget=1, solver="ipopt",
                 enumeration_time=1.5, compile_time=2.5, n_jobs=4)

    meta = CredalNetworkVertices.vtx_metadata(out)
    assert meta is not None
    assert meta["method"] == "linear"
    assert meta["merge_budget"] == 1
    assert meta["solver"] == "ipopt"
    assert meta["enumeration_time"] == 1.5
    assert meta["compile_time"] == 2.5
    assert meta["n_jobs"] == 4

    ep, meta2 = CredalNetworkVertices.load_extreme_points(out)
    assert ep == cnv.extreme_points
    assert meta2["enumeration_time"] == 1.5


def test_vtx_metadata_none_on_bad_input(tmp_path):
    """vtx_metadata returns None for a missing file and non-.vtx JSON."""
    assert CredalNetworkVertices.vtx_metadata(str(tmp_path / "nope.vtx")) is None

    other = tmp_path / "other.vtx"
    other.write_text('{"format": "something-else", "version": 1}')
    assert CredalNetworkVertices.vtx_metadata(str(other)) is None

    garbage = tmp_path / "garbage.vtx"
    garbage.write_text("not json")
    assert CredalNetworkVertices.vtx_metadata(str(garbage)) is None


def test_vtx_cache_hit_mismatch_and_marginals(tmp_path):
    """A matching .vtx is loaded (timings reused, marginals unchanged); a
    param-mismatch .vtx is ignored."""
    # Compile + enumerate once, alongside a copy of the .lcn so the default
    # cache paths (foo.cn / foo.vtx) land in tmp_path.
    src = tmp_path / "alarm.lcn"
    src.write_text(open(_ALARM).read())
    _quiet(compile_lcn_to_cn, str(src), n_jobs=1, verbosity=0)

    lcn = LCN()
    _quiet(lcn.from_lcn, file_name=str(src))
    # Produce the .vtx via a fresh enumeration, then persist it.
    fresh = _quiet(CredalNetworkVertices.from_lcn, lcn, method="linear",
                   merge_budget=1, solver="ipopt", cache=False, verbosity=0)
    vtx_path = str(tmp_path / "alarm.vtx")
    fresh.save_vtx(vtx_path, method="linear", merge_budget=1, solver="ipopt",
                   enumeration_time=fresh.enumeration_time, compile_time=3.0,
                   n_jobs=1)

    # (1) Matching build -> vertex cache hit; timings reused.
    hit = _quiet(CredalNetworkVertices.from_lcn, lcn, method="linear",
                 merge_budget=1, solver="ipopt",
                 lcn_file=str(src), cache=True, verbosity=0)
    assert hit.loaded_vertices_from_cache is True
    assert hit.compile_time == 3.0
    assert hit.enumeration_time == fresh.enumeration_time
    assert hit.extreme_points == fresh.extreme_points

    # Marginals from the cached vertices match a fresh (no-cache) build.
    rc = _quiet(CredalVE(cnv=hit).run, evidence={}, verbosity=0)
    rf = _quiet(CredalVE(cnv=fresh).run, evidence={}, verbosity=0)
    assert set(rc) == set(rf)
    for k in rc:
        assert np.allclose(rc[k][0], rf[k][0])
        assert np.allclose(rc[k][1], rf[k][1])

    # (2) Different merge_budget -> must NOT load the .vtx.
    miss = _quiet(CredalNetworkVertices.from_lcn, lcn, method="linear",
                  merge_budget=3, solver="ipopt",
                  lcn_file=str(src), cache=True, verbosity=0)
    assert miss.loaded_vertices_from_cache is False

    # (3) cache=False -> recompute even when a matching .vtx exists.
    forced = _quiet(CredalNetworkVertices.from_lcn, lcn, method="linear",
                    merge_budget=1, solver="ipopt",
                    lcn_file=str(src), cache=False, verbosity=0)
    assert forced.loaded_vertices_from_cache is False
