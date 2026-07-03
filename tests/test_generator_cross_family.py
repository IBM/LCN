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

# Invariant tests for the generator's tree/polytree families. Two distinct
# properties, at two levels of strength:
#
# 1. No ATOM-SCOPE cross-family sentence (test_no_cross_family_sentences).
#    A sentence P(phi)/P(phi|psi) is atom-scope cross-family when its atom set
#    fits inside no single family scope ({child} union {parents}). The generator
#    never produces one for tree/polytree: it builds phi from the child atom and
#    psi from the parent atoms, and emits no multi-variable component sentence
#    (that path is chain-only). This holds for BOTH the default "tree"/"polytree"
#    and the family-realizable "tree-fr"/"polytree-fr" classes.
#
# 2. FAMILY-REALIZABILITY (test_fr_classes_are_family_realizable) -- the STRONGER
#    property that governs exactness of the strong-extension engines (Credal VE,
#    Interval BP). A Type-1 marginal P(phi) whose formula mentions a NON-root
#    child is atom-scope family-local yet still an effective cross-family
#    constraint (it couples the child's conditional to its parents' marginal),
#    so it is NOT family-realizable and makes CVE/IBP loose even on a tree (see
#    docs/strong_extension_exactness.tex). The DEFAULT tree/polytree classes may
#    contain such a sentence (num_extras draws from all atoms), so they are NOT
#    guaranteed family-realizable. The "-fr" classes place extras on root atoms
#    only and ARE guaranteed family-realizable; this test pins that.
#
# Cross-family *LMC assertions* are structural and expected, and are allowed
# throughout.

import os

import pytest

from lcn.core.model import LCN
from lcn.benchmarks.generator import Generator
from lcn.inference.marginal.cn.factorization import ChainGraphFactorization
from lcn.core.model import SentenceType


_EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


def _family_scopes(lcn):
    """Return the list of family scopes (as sets of atom names) for the LCN's
    chain-graph factorization -- the same notion the inference engines use.

    Mirrors the CredalNetwork.from_lcn build sequence (credal_network.py):
    structure graph -> simplified structure graph -> chain-graph families.
    """
    if lcn.primal_graph is None:
        lcn.build_primal_graph()
    if lcn.structure_graph is None:
        lcn.build_structure_graph()
    if lcn.simplified_structure_graph is None:
        lcn.simplify_structure_graph()
    if lcn.families is None:
        lcn.process_chain_graph()
    if lcn.independencies is None:
        lcn.local_markov_condition()
    factors = ChainGraphFactorization(lcn).build(verbosity=0)
    return [set(f["scope"]) for f in factors]


def _cross_family_sentences(lcn):
    """Return the labels of SENTENCES whose atom scope fits in no single family
    scope. Uses the same predicate as CouplingConstraints.from_lcn:
    cross-family iff the sentence's atom set is a subset of no family scope."""
    scopes = _family_scopes(lcn)

    def is_cross_family(atom_set):
        return not any(atom_set.issubset(fs) for fs in scopes)

    offenders = []
    for sid, s in lcn.sentences.items():
        atoms = set(s.get_atoms().keys())
        if atoms and is_cross_family(atoms):
            offenders.append((sid, s.type, sorted(atoms)))
    return offenders


@pytest.mark.parametrize("graph_type", ["tree", "polytree"])
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_no_cross_family_sentences(graph_type, seed):
    """Generated tree/polytree instances must have zero cross-family sentences."""
    gen = Generator(seed=seed)
    instances = gen.generate(
        num_vars=8,
        graph_type=graph_type,
        num_instances=3,
        max_vars_per_sentence=3,
        num_extras=2,
        epsilon=0.3,
        verbosity=0,
    )
    assert instances, f"generator produced no {graph_type} instances (seed={seed})"

    for k, lcn in enumerate(instances):
        offenders = _cross_family_sentences(lcn)
        assert not offenders, (
            f"{graph_type} instance {k} (seed={seed}) has cross-family "
            f"sentence(s): {offenders}")


def test_predicate_detects_a_real_cross_family_sentence():
    """Sanity guard: the predicate is not vacuous -- on d4_biting (whose BC
    sentence P(B and C) is cross-family by construction) it MUST flag exactly
    that sentence. Otherwise the tests above could pass for the wrong reason."""
    path = os.path.join(_EXAMPLES_DIR, "d4_biting.lcn")
    lcn = LCN()
    lcn.from_lcn(file_name=path)

    offenders = _cross_family_sentences(lcn)
    offender_atoms = [atoms for (_sid, _typ, atoms) in offenders]
    assert ["B", "C"] in offender_atoms, (
        f"expected the cross-family sentence P(B and C) to be detected; "
        f"found offenders: {offenders}")
    # And every flagged sentence is a genuine sentence (not an LMC assertion).
    for _sid, typ, _atoms in offenders:
        assert typ in (SentenceType.Type1, SentenceType.Type2)


def _root_atoms(lcn):
    """Root atoms: the children of families with no parents (whose family factor
    is the plain marginal P(x))."""
    if lcn.families is None:
        _family_scopes(lcn)  # runs the build sequence that populates families
    roots = set()
    for fam in lcn.families:
        if not fam["parents"]:
            child = fam["child"]
            roots |= set(child.split("-") if "-" in child else [child])
    return roots


def _non_realizable_sentences(lcn):
    """Return labels of NON-family-realizable sentences: a Type-1 sentence whose
    formula mentions a non-root atom (a marginal / joint that couples a family's
    conditional to its parents' marginal). Type-2 conditionals and Type-1 on
    roots are family-realizable. See docs/strong_extension_exactness.tex."""
    roots = _root_atoms(lcn)
    offenders = []
    for sid, s in lcn.sentences.items():
        if s.type == SentenceType.Type1:
            atoms = set(s.get_atoms().keys())
            if atoms and not atoms.issubset(roots):
                offenders.append((sid, sorted(atoms)))
    return offenders


@pytest.mark.parametrize("graph_type", ["tree-fr", "polytree-fr"])
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_fr_classes_are_family_realizable(graph_type, seed):
    """The family-realizable classes must contain no non-realizable sentence:
    every Type-1 marginal is on a root atom, so the strong-extension engines
    (Credal VE, Interval BP) are exact on them."""
    gen = Generator(seed=seed)
    instances = gen.generate(
        num_vars=8,
        graph_type=graph_type,
        num_instances=3,
        max_vars_per_sentence=3,
        num_extras=3,
        epsilon=0.3,
        verbosity=0,
    )
    assert instances, f"generator produced no {graph_type} instances (seed={seed})"
    for k, lcn in enumerate(instances):
        offenders = _non_realizable_sentences(lcn)
        assert not offenders, (
            f"{graph_type} instance {k} (seed={seed}) has non-realizable "
            f"Type-1 sentence(s) on non-root atoms: {offenders}")
        # And it is of course still atom-scope cross-family free.
        assert not _cross_family_sentences(lcn)
