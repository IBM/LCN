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

# Invariant tests for the revised DAG generator. Properties:
#   (1) moralized_treewidth is correct on hand-built graphs (chain -> 1,
#       k-tree seed -> k, 2-parent collider -> 2, root -> 0);
#   (2) every generated "dag" instance's built chain-graph family treewidth is
#       <= the requested cap (tested at cap 3 and 4);
#   (3) generated "dag" families respect the user-set max_parents and the
#       oriented families admit a topological order (it is a DAG);
#   (4) max_parents is fully tunable -- values > 2 are accepted (no hard cap),
#       yielding higher fan-in while staying acyclic and treewidth-bounded;
#   (5) every generated "dag" instance is consistent -- i.e. the same
#       product-witness consistency check used for trees/polytrees/k-trees is
#       actually applied to dag (it passes on each accepted instance).
#
# Properties (2)-(5) are parametrized over BOTH dag variants: the unstructured
# "dag" (treewidth met by rejection sampling) and the family-realizable "dag-fr"
# (treewidth met BY CONSTRUCTION via a partial k-tree). Covering "dag-fr" here is
# what pins the cap for it: the rejection block in generate() is keyed on the
# graph_type, so a "dag-fr" left out of that condition would silently bypass the
# cap entirely. Property (6) below covers the reason "dag-fr" exists -- plain
# "dag" cannot reach large n at all, because its acceptance rate collapses
# (<1% at n=50, 0% at n>=30 for 3 parents).

import contextlib
import io

import pytest

from lcn.benchmarks.generator import Generator, moralized_treewidth
from lcn.inference.marginal.cn.factorization import ChainGraphFactorization
from lcn.inference.utils.common import (
    check_consistency_product_witness_scoped,
)


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _built_family_scopes(lcn):
    """Flattened atom scopes of the LCN's chain-graph families -- exactly the
    graph the inference engines eliminate over (mirrors credal_network build)."""
    if lcn.primal_graph is None:
        lcn.build_primal_graph()
    if lcn.structure_graph is None:
        lcn.build_structure_graph()
    if lcn.simplified_structure_graph is None:
        lcn.simplify_structure_graph()
    if lcn.families is None:
        lcn.process_chain_graph()
    factors = ChainGraphFactorization(lcn).build(verbosity=0)
    return [list(f["scope"]) for f in factors]


def test_moralized_treewidth_known_cases():
    """The width helper matches hand-computed values."""
    # chain 0-1-2-3-4 as families [child, parent]
    assert moralized_treewidth([[0], [0, 1], [1, 2], [2, 3], [3, 4]]) == 1
    # k=2 tree seed: v2 conditions on v0,v1 -> moral clique {0,1,2}
    assert moralized_treewidth([[0], [0, 1], [0, 1, 2], [1, 2, 3]]) == 2
    # 2-parent collider v2 | v0, v1 -> clique {0,1,2}
    assert moralized_treewidth([[0], [1], [0, 1, 2]]) == 2
    # single root
    assert moralized_treewidth([[0]]) == 0


@pytest.mark.parametrize("graph_type", ["dag", "dag-fr"])
@pytest.mark.parametrize("cap", [3, 4])
def test_dag_treewidth_bounded(cap, graph_type):
    """Every generated dag/dag-fr instance has built family treewidth <= cap."""
    gen = Generator(seed=13)
    insts = _quiet(gen.generate, num_vars=12, graph_type=graph_type,
                   num_instances=5, max_treewidth=cap, verbosity=0)
    assert len(insts) == 5
    for lcn in insts:
        scopes = _built_family_scopes(lcn)
        tw = moralized_treewidth(scopes)
        assert tw <= cap, f"treewidth {tw} exceeds cap {cap}"


def _assert_is_dag(lcn):
    """The family child<-parents edges must admit a topological order (Kahn)."""
    from collections import defaultdict
    children = [f["child"] for f in lcn.families]
    edges = [(p, f["child"]) for f in lcn.families for p in f["parents"]]
    indeg = defaultdict(int)
    adj = defaultdict(list)
    nodes = set(children)
    for p, c in edges:
        nodes.add(p)
        adj[p].append(c)
        indeg[c] += 1
    queue = [n for n in nodes if indeg[n] == 0]
    seen = 0
    while queue:
        u = queue.pop()
        seen += 1
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    assert seen == len(nodes), "family orientation contains a cycle"


@pytest.mark.parametrize("graph_type", ["dag", "dag-fr"])
def test_dag_is_acyclic_and_respects_max_parents(graph_type):
    """Generated dag/dag-fr families respect the (default) max_parents bound and
    the orientation is acyclic."""
    gen = Generator(seed=21)
    insts = _quiet(gen.generate, num_vars=12, graph_type=graph_type,
                   num_instances=5, max_parents=2, max_treewidth=4, verbosity=0)
    for lcn in insts:
        if lcn.families is None:
            _built_family_scopes(lcn)  # populates lcn.families
        for fam in lcn.families:
            assert len(fam["parents"]) <= 2, \
                f"{fam['child']} has {len(fam['parents'])} parents"
        _assert_is_dag(lcn)


@pytest.mark.parametrize("graph_type", ["dag", "dag-fr"])
def test_dag_max_parents_is_tunable_above_two(graph_type):
    """max_parents > 2 is accepted (no cap): the generated DAG allows higher
    fan-in yet stays acyclic and treewidth-bounded."""
    gen = Generator(seed=99)
    # Larger n and a looser treewidth cap so 3-parent families can appear and
    # still pass the rejection filter.
    insts = _quiet(gen.generate, num_vars=15, graph_type=graph_type,
                   num_instances=5, max_parents=3, max_treewidth=4, verbosity=0)
    assert len(insts) == 5
    max_seen = 0
    for lcn in insts:
        if lcn.families is None:
            _built_family_scopes(lcn)
        for fam in lcn.families:
            # no hard 2-parent cap anymore, only the user-set max_parents=3
            assert len(fam["parents"]) <= 3
            max_seen = max(max_seen, len(fam["parents"]))
        _assert_is_dag(lcn)
        # treewidth cap still enforced despite the higher fan-in
        assert moralized_treewidth(_built_family_scopes(lcn)) <= 4
    # at least one family should exercise 3 parents (else the test is vacuous)
    assert max_seen == 3


@pytest.mark.parametrize("graph_type", ["dag", "dag-fr"])
def test_dag_instances_are_consistent(graph_type):
    """The product-witness consistency check (same as tree/polytree/ktree) is
    applied to both dag variants: every accepted instance passes it."""
    gen = Generator(seed=34)
    insts = _quiet(gen.generate, num_vars=12, graph_type=graph_type,
                   num_instances=5, max_treewidth=4, verbosity=0)
    for lcn in insts:
        assert _quiet(check_consistency_product_witness_scoped, lcn,
                      restarts=40) is True


@pytest.mark.parametrize("num_vars", [30, 50])
def test_dag_fr_scales_to_large_n_within_cap(num_vars):
    """The reason "dag-fr" exists: it must actually GENERATE at sizes where the
    unstructured "dag" cannot.

    "dag" meets the treewidth cap only by resampling, and its acceptance rate
    collapses as n grows (measured over 500 samples at max_parents=2, cap 3: 17%
    at n=30, 0.6% at n=50), so large bounded-width DAGs are effectively
    ungeneratable that way. "dag-fr" draws parents from a k-clique of an
    underlying partial k-tree, so the cap holds by construction at any n. Assert
    we get the full requested count AND every instance respects the cap -- a
    constructive bug that broke the invariant would otherwise be hidden by the
    rejection safety net silently dropping instances.
    """
    cap = 3
    gen = Generator(seed=5)
    insts = _quiet(gen.generate, num_vars=num_vars, graph_type="dag-fr",
                   num_instances=3, max_parents=2, max_treewidth=cap,
                   num_extras=2, verbosity=0)
    assert len(insts) == 3, (
        f"dag-fr n={num_vars}: got {len(insts)}/3 instances -- the constructive "
        f"treewidth bound should make these cheap to generate")
    for lcn in insts:
        tw = moralized_treewidth(_built_family_scopes(lcn))
        assert tw <= cap, f"n={num_vars}: treewidth {tw} exceeds cap {cap}"
        _assert_is_dag(lcn)
