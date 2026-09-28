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

# Tests for the structure-exploiting junction-tree consistency checker
# (lcn.inference.utils.structured_consistency) and its generator wiring.
#
# Properties:
#   (1) A genuinely inconsistent bounded-treewidth instance is proven
#       INCONSISTENT (the original junkyu n24 clinic-referral model, reconstructed
#       from the checked-in dropped-component fix by re-adding the three
#       incompatible component sentences: independent reviewers make P(all four)
#       a product below the component lower bound). The checked-in base file is
#       that fix and is CONSISTENT.
#   (2) A tiny contradictory chain is INCONSISTENT.
#   (3) Completeness: it certifies CONSISTENT an instance the conservative
#       product witness false-rejects (examples/cancer.lcn).
#   (4) The generator's consistency_mode="structured" gate accepts genuinely
#       consistent instances.
#
# INCONSISTENT verdicts need the global solver (SCIP); those assertions skip if
# SCIP is unavailable. CONSISTENT verdicts are sound under ipopt too.

import contextlib
import io
import os

import pytest

from lcn.core.model import LCN
from lcn.benchmarks.generator import Generator
from lcn.inference.utils.common import check_consistency_product_witness
from lcn.inference.utils.structured_consistency import (
    check_consistency_structured, is_consistent_structured,
    CONSISTENT, INCONSISTENT,
)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_JUNKYU = os.path.join(_REPO, "benchmarks", "junkyu")
_EXAMPLES = os.path.join(_REPO, "examples")

# The base n24 clinic-referral instance is the CONSISTENT (dropped-component)
# fix: its four per-stage reviewers are mutually independent in the chain graph,
# so each stage's four-way conjunction probability is the product of the four
# reviewer marginals -- which the priors bound well below what the original
# component_j sentences demanded, making the ORIGINAL inconsistent. The fix
# dropped the three incompatible component sentences (component_1/2/4). We
# reconstruct the original inconsistent instance in-repo by re-adding them, so
# the INCONSISTENT-detection test does not depend on a since-removed file.
_N24_BASE = os.path.join(_JUNKYU, "benchmark_scaling_directions_n24_2.lcn")
_N24_DROPPED_COMPONENTS = (
    "component_1: 0.104 <= P(Review_1_1 and Review_1_2 and Review_1_3 "
    "and Review_1_4) <= 0.282\n"
    "component_2: 0.092 <= P(Review_2_1 and Review_2_2 and Review_2_3 "
    "and Review_2_4) <= 0.188\n"
    "component_4: 0.038 <= P(Review_4_1 and Review_4_2 and Review_4_3 "
    "and Review_4_4) <= 0.158\n"
)


def _scip_available():
    try:
        from lcn.inference.marginal.exact import make_scip
        make_scip(time_limit=5.0)
        return True
    except Exception:
        return False


def _load(path):
    # Fail loudly on a missing benchmark rather than silently loading an empty
    # LCN (from_lcn returns a 0-sentence model for a nonexistent path, which
    # would make every downstream verdict meaningless).
    assert os.path.exists(path), f"benchmark file not found: {path}"
    lcn = LCN()
    with contextlib.redirect_stdout(io.StringIO()):
        lcn.from_lcn(path)
    assert len(lcn.sentences) > 0, f"no sentences parsed from {path}"
    return lcn


def _load_text(text, tmp_path, name="inst.lcn"):
    p = tmp_path / name
    p.write_text(text)
    return _load(str(p))


requires_scip = pytest.mark.skipif(
    not _scip_available(), reason="SCIP solver not available")


@requires_scip
def test_known_inconsistent_junkyu_n24(tmp_path):
    """The ORIGINAL n24 clinic-referral instance (base fix + the three dropped
    component sentences re-added) is provably inconsistent."""
    text = open(_N24_BASE).read() + "\n" + _N24_DROPPED_COMPONENTS
    lcn = _load_text(text, tmp_path, "n24_original.lcn")
    r = check_consistency_structured(lcn, solver="scip", time_limit=300)
    assert r.status == INCONSISTENT
    assert r.treewidth is not None and r.treewidth <= 6


def test_fixed_junkyu_n24_consistent():
    """The base n24 instance is the dropped-component fix and is consistent
    (an ipopt incumbent is a sound CONSISTENT certificate, so no SCIP needed)."""
    r = check_consistency_structured(_load(_N24_BASE), solver="ipopt",
                                     time_limit=120)
    assert r.status == CONSISTENT


@requires_scip
def test_contradictory_chain_inconsistent(tmp_path):
    """A tiny chain whose bounds cannot be jointly satisfied is INCONSISTENT."""
    p = tmp_path / "inc.lcn"
    p.write_text(
        "s1: 0.9 <= P(a) <= 1.0 ; True\n"
        "s2: 0.9 <= P(b | a) <= 1.0 ; True\n"
        "s3: 0.0 <= P(a and b) <= 0.05 ; True\n"
    )
    r = check_consistency_structured(_load(str(p)), solver="scip", time_limit=60)
    assert r.status == INCONSISTENT


def test_completeness_beats_product_witness():
    """Structured accepts a consistent instance the product witness rejects."""
    lcn = _load(os.path.join(_EXAMPLES, "cancer.lcn"))
    # Product witness is conservative: it false-rejects this non-product model.
    assert check_consistency_product_witness(lcn, restarts=40) is False
    # Structured is complete: it certifies consistency (ipopt incumbent is sound).
    assert is_consistent_structured(lcn, solver="ipopt", time_limit=120) is True


def test_generator_structured_mode_accepts_consistent():
    """generate(consistency_mode='structured') yields genuinely consistent
    instances."""
    g = Generator(seed=7)
    with contextlib.redirect_stdout(io.StringIO()):
        inst = g.generate(num_vars=7, graph_type="polytree", num_instances=1,
                          consistency_mode="structured", verbosity=0)
    assert len(inst) == 1
    assert is_consistent_structured(inst[0], solver="ipopt", time_limit=120) is True
