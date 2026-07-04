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

# Cross-family coupling constraints for scheme D4 (region-restricted credal
# network) -- see docs/tighter_approximation.tex.
#
# The cn/ inference algorithms (CVE, CCTE, ApproxLP) optimize over the STRONG
# EXTENSION of the credal network: every (node, parent-config) slot chooses a
# local extreme point independently, so the joint is a free product of those
# vertices. Schemes D1-D3 tighten the LOCAL credal sets (per-family looseness);
# they cannot express a constraint that spans more than one family. Those
# CROSS-FAMILY constraints -- an LCN sentence or an LMC independence assertion
# whose atom scope is a subset of NO single family scope -- are exactly what the
# free product drops.
#
# This module builds, from an LCN and its chain-graph factorization, the set of
# cross-family constraints as feasibility predicates over a single materialized
# joint "function" (a numpy array over a node scope, as used by `Potential`).
# D4 then forbids any vertex combination whose assembled joint violates one.
#
# The residual math is deliberately IDENTICAL to the LMC factorization verifier
# (verify_lmc_factorization.py::_build_residuals) and to ExactInference's joint
# encoding (exact.py), both of which go through `lmc_constraint_groups_vec`. So
# a function D4 rejects is exactly a joint the verifier would report as an LMC
# violation -- the gap meter and the enforcement share one definition.

from typing import Dict, List

import numpy as np

from lcn.core.model import LCN, SentenceType
from lcn.inference.utils.common import (
    build_truth_table, eval_indicator,
    lmc_constraint_groups_vec,
)


def warn_conditional_coupling(evidence, tag, verbosity):
    """
    Emit the D4 conditional-query soundness warning.

    The vertex-enumeration credal VE / CTE / ApproxLP bound a conditional query
    P(q | e) by a min/max of per-vertex ratios. That ratio's extremum need not
    lie at a vertex of the unconditioned credal set, so for conditional queries
    these methods are NOT guaranteed to be outer bounds even with coupling off;
    the D4 filter can make the interval visibly too tight (e.g. collapse it to a
    near-point). Use the CredalJT engine (the junction-tree exact NLP, scheme
    D5) for trustworthy conditional bounds. This is a no-op without evidence.
    """
    if evidence and verbosity > 0:
        print(f"[{tag}] WARNING: coupling='cross-family' (D4) with evidence "
              f"{dict(evidence)} -- D4 is unsound for CONDITIONAL queries "
              f"(the bound may be too tight, not a valid outer bound). Use the "
              f"CredalJT engine for an exact conditional bound.")


def _node_state_to_atom_bits(state: int, atoms: List[str]) -> Dict[str, int]:
    """
    Decode a (possibly compound) node state integer into its per-atom 0/1 bits.

    Mirrors cn/vertices.py and verify_lmc_factorization.py::_node_state: the
    atoms of a node are packed MSB-first, so for node "C-D" with atoms [C, D]
    the state is (C << 1) | D. This is the inverse, recovering {C: .., D: ..}.
    """
    bits = {}
    for a in reversed(atoms):
        bits[a] = state & 1
        state >>= 1
    return bits


class _Constraint:
    """One cross-family constraint, with its atom set and a residual check.

    The residual checker is built LAZILY: constructing it materializes a
    2^|atoms| truth table plus dense indicator arrays (via _make_sentence_checker
    / _make_lmc_checker), which is prohibitive for a wide cross-family constraint
    (e.g. a chain/tree LMC assertion spanning ~20 atoms -> 2^20 rows). Callers
    that only need the constraint's atom / node set -- notably the CredalJT (D5)
    engine via ``constraint_node_sets`` -- never invoke ``feasible`` and so never
    pay that cost. The table is built on the first ``feasible`` call and memoized.
    """

    def __init__(self, kind: str, atoms: List[str], checker_factory, descr: str):
        # kind: "lmc" | "type1" | "type2"; atoms: sorted atom names it touches.
        # checker_factory: zero-arg callable returning the residual-check closure
        # (deferred so the 2^|atoms| tables are not built until actually needed).
        self.kind = kind
        self.atoms = list(atoms)
        self.atom_set = set(atoms)
        self._checker_factory = checker_factory
        self._checker = None
        self.descr = descr

    def feasible(self, joint_flat: np.ndarray) -> bool:
        """`joint_flat`: normalized length-2^|atoms| joint over self.atoms in
        the SAME order as self.atoms (MSB-first truth-table order).

        Builds (and memoizes) the residual checker on first use -- this is where
        the 2^|atoms| truth table is materialized, so a caller that never calls
        ``feasible`` pays nothing."""
        if self._checker is None:
            self._checker = self._checker_factory()
        return self._checker(joint_flat)


class CouplingConstraints:
    """
    The cross-family constraints of an LCN's chain-graph factorization, exposed
    as feasibility predicates over a single joint `function`.

    Build with :meth:`from_lcn`. An empty instance (``len(cc) == 0``) means the
    factorization has no cross-family residual -- every constraint already fits
    inside one family (handled by D1) -- so callers can skip all D4 work.
    """

    def __init__(self, constraints: List[_Constraint], tol: float):
        self._constraints = constraints
        self.tol = tol

    def __len__(self):
        return len(self._constraints)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_lcn(cls, lcn: LCN, factors: List[Dict],
                 tol: float = 1e-7) -> "CouplingConstraints":
        """
        Args:
            lcn: the source LCN (must have independencies computed, or they are
                computed here lazily via local_markov_condition()).
            factors: the SYMBOLIC chain-graph factor descriptors, each with a
                "scope" key (e.g. cnv.cn.factorization.factors, or the list
                returned by ChainGraphFactorization.build()). Using the BUILT
                factors means a D2 merge_budget is respected automatically:
                merged super-families have larger scopes, so fewer constraints
                are cross-family.
            tol: absolute tolerance for the (bilinear) LMC residual and the
                sentence bound checks.
        """
        if lcn.primal_graph is None:
            lcn.build_primal_graph()
        if lcn.independencies is None:
            lcn.local_markov_condition()

        family_scopes = [set(f["scope"]) for f in factors]

        def is_cross_family(atom_set: set) -> bool:
            # Cross-family iff it fits inside NO single family scope.
            return not any(atom_set.issubset(fs) for fs in family_scopes)

        constraints: List[_Constraint] = []

        # --- Cross-family LCN sentences -----------------------------------
        # Pass a checker FACTORY (thunk), not a built checker: the 2^|atoms|
        # tables are materialized only on the first feasible() call, so a
        # caller that only reads atoms/node sets (e.g. CredalJT's
        # constraint_node_sets) never allocates them.
        for sid, s in lcn.sentences.items():
            atoms = sorted(s.get_atoms().keys())
            if not atoms or not is_cross_family(set(atoms)):
                continue
            kind = "type1" if s.type == SentenceType.Type1 else "type2"
            factory = (lambda s=s, atoms=atoms:
                       cls._make_sentence_checker(s, atoms, tol))
            constraints.append(_Constraint(kind, atoms, factory, f"sentence {sid}"))

        # --- Cross-family LMC assertions ----------------------------------
        for indep in lcn.independencies.get_assertions():
            atoms = sorted(indep.all_vars)
            if not atoms or not is_cross_family(set(indep.all_vars)):
                continue
            factory = (lambda indep=indep, atoms=atoms:
                       cls._make_lmc_checker(indep, atoms, tol))
            constraints.append(_Constraint("lmc", atoms, factory, str(indep)))

        return cls(constraints, tol)

    @staticmethod
    def _make_sentence_checker(sentence, atoms: List[str], tol: float):
        """Linear (Type1) / ratio (Type2) bound check on the joint over `atoms`.

        Indicators are built over a local truth table whose column order is
        `atoms`, so they align with the flat joint passed to the checker.
        """
        table_interps = [dict(zip(atoms, row))
                         for row in build_truth_table(len(atoms))]
        lo = sentence.get_lower_bound()
        hi = sentence.get_upper_bound()
        if sentence.type == SentenceType.Type1:
            A = eval_indicator(sentence.phi_formula, table_interps)

            def check(p, A=A, lo=lo, hi=hi):
                v = float(A @ p)
                return (v >= lo - tol) and (v <= hi + tol)
            return check
        else:
            Aqr = eval_indicator(sentence.phi_and_psi_formula, table_interps)
            Ar = eval_indicator(sentence.psi_formula, table_interps)

            def check(p, Aqr=Aqr, Ar=Ar, lo=lo, hi=hi):
                den = float(Ar @ p)
                num = float(Aqr @ p)
                if den <= tol:
                    # Conditioning event has ~zero mass: the conditional is
                    # undefined, so the sentence imposes nothing here.
                    return True
                r = num / den
                return (r >= lo - tol) and (r <= hi + tol)
            return check

    @staticmethod
    def _make_lmc_checker(indep, atoms: List[str], tol: float):
        """Bilinear LMC residual check, identical math to _build_residuals.

        Groups are built over a local truth table with column order `atoms`.
        """
        table = build_truth_table(len(atoms))
        col_of = {v: i for i, v in enumerate(atoms)}
        groups = lmc_constraint_groups_vec(indep, table, col_of)

        def check(p, groups=groups):
            for group in groups:
                if group[0] == 'conditional':
                    _, Aa, Ab, Ac, Ad = group
                    resid = float(Aa @ p) * float(Ab @ p) \
                        - float(Ac @ p) * float(Ad @ p)
                else:  # 'marginal'
                    _, Aa, Ab, Ac = group
                    resid = float(Aa @ p) - float(Ab @ p) * float(Ac @ p)
                if abs(resid) > tol:
                    return False
            return True
        return check

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def constraint_node_sets(self, node_atoms: Dict[str, List[str]],
                             kinds=None) -> List[List[str]]:
        """
        For each constraint, the set of NODE names whose atoms it touches. Used
        to augment the elimination order so the constrained nodes co-occur in a
        common bucket (otherwise the constraint scope is never assembled and the
        check stays inert). `node_atoms` maps node name -> its atom list.

        ``kinds`` optionally restricts to constraints of the given kinds
        (subset of {"type1","type2","lmc"}). D4 (cve.py) passes None (all);
        D5/CredalJT passes ("type1","type2") so that only cross-family
        *sentences* force atoms together -- LMC assertions are structural and
        are handled by the junction tree's running-intersection separators, so
        augmenting with them would needlessly inflate the treewidth.
        """
        node_of_atom = {}
        for node, atoms in node_atoms.items():
            for a in atoms:
                node_of_atom[a] = node
        out = []
        for c in self._constraints:
            if kinds is not None and c.kind not in kinds:
                continue
            nodes = sorted({node_of_atom[a] for a in c.atoms
                            if a in node_of_atom})
            if nodes:
                out.append(nodes)
        return out

    def is_feasible(self, function: np.ndarray, scope: List[str],
                    node_atoms: Dict[str, List[str]],
                    cards: Dict[str, int]) -> bool:
        """
        True if `function` (a `Potential` function over node `scope`) violates
        NO checkable cross-family constraint.

        Checkability gate: a constraint is only evaluated when every one of its
        atoms is carried by some node in `scope`. Until then the predicate
        returns True ("not yet checkable, don't prune"), so filtering after any
        elimination step is sound (it can only remove genuinely-infeasible
        functions) and a no-op at intermediate scopes.
        """
        if not self._constraints:
            return True

        # Atoms available in this scope, and the per-row atom-bit decoding.
        scope_atoms = set()
        for node in scope:
            scope_atoms.update(node_atoms[node])

        applicable = [c for c in self._constraints
                      if c.atom_set.issubset(scope_atoms)]
        if not applicable:
            return True

        # Normalize the function to a joint distribution over its scope.
        total = float(function.sum())
        if total <= 0:
            return True  # degenerate; nothing to check
        norm = function / total

        # For each applicable constraint, marginalize `norm` down to the
        # constraint's atoms (in c.atoms order) and check the residual.
        for c in applicable:
            p = self._marginal_over_atoms(norm, scope, node_atoms, c.atoms)
            if not c.feasible(p):
                return False
        return True

    @staticmethod
    def _marginal_over_atoms(norm: np.ndarray, scope: List[str],
                             node_atoms: Dict[str, List[str]],
                             want_atoms: List[str]) -> np.ndarray:
        """
        Marginalize a normalized joint `norm` (multi-axis array over `scope`,
        axis i = node scope[i], index packed MSB-first over node_atoms[scope[i]])
        down to a flat length-2^|want_atoms| joint whose row order is the MSB-
        first truth table over `want_atoms` (matching the indicator tables built
        in the checkers).
        """
        want = list(want_atoms)
        n_want = len(want)
        want_index = {a: i for i, a in enumerate(want)}
        out = np.zeros(2 ** n_want, dtype=float)

        # Walk every cell of the multi-axis function once, accumulate its mass
        # into the want-atoms row it projects to.
        for idx in np.ndindex(*norm.shape):
            mass = norm[idx]
            if mass == 0.0:
                continue
            # Decode each node's state to atom bits; collect the wanted atoms,
            # then pack the want-row MSB-first over `want`.
            want_bits = {}
            for pos, node in enumerate(scope):
                bits = _node_state_to_atom_bits(idx[pos], node_atoms[node])
                for a, b in bits.items():
                    if a in want_index:
                        want_bits[a] = b
            row = 0
            for a in want:
                row = (row << 1) | want_bits[a]
            out[row] += mass
        return out
