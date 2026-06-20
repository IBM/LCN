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

# SCC Propagation with an ARIEL local solver (v2 of SCC-FGP).
#
# This is a second version of the SCC Factor-Graph Propagation algorithm
# implemented in `sccp.py`. The global structure is identical: the LCN's
# structure graph is decomposed into Strongly Connected Components (SCCs), each
# cycle is contracted into a super-node, and two-pass belief propagation runs on
# the acyclic condensation factor graph. The ONLY difference is how a non-trivial
# (cyclic) super-node is solved:
#
#   * sccp.py (v1)        : one EXACT joint NLP over the super-node's 2^|scope|
#                           worlds, solved by ipopt.
#   * sccp_ariel.py (v2)  : the ARIEL approximate scheme is run on the
#                           super-node's local sub-LCN (loopy interval belief
#                           propagation over the super-node's own internal
#                           factor graph).
#
# Trivial (singleton) super-nodes keep the cheap exact LP/NLP solve, since ARIEL
# would only add looseness there. This realizes the "hybrid" relationship from
# docs/scc_factor_graph.tex (Section "A hybrid relationship"): the SCC scheme
# isolates the irreducible cyclic cores and delegates them to ARIEL.

import time
import numpy as np
from pyomo.environ import SolverFactory
from typing import Dict, List, Tuple

# Local
from lcn.core.model import LCN, SentenceType, Sentence
from lcn.inference.utils.common import check_consistency, make_ipopt
from lcn.inference.marginal.ariel import ArielInference

# Reuse the SCC decomposition, super-node model, exact local solve and message
# machinery from the v1 implementation (sccp.py). Only the cyclic super-node
# solve is overridden here, so everything else is imported verbatim.
from lcn.inference.marginal.sccp import (
    _build_condensation,
    _classify_sentences,
    SuperNode,
    _build_supernode_cache,
    _solve_supernode,
    SCCMessage,
    format_condensation_dag,
    format_factorization,
)


# -----------------------------------------------------------------------
# ARIEL local solve for a cyclic super-node
# -----------------------------------------------------------------------

def _build_supernode_lcn(
        node: SuperNode,
        incoming: Dict[str, Tuple[float, float]]
) -> LCN:
    """
    Build the local sub-LCN of a super-node for the ARIEL solve.

    The sub-LCN contains the super-node's own assigned sentences (its internal
    constraints plus the incident linking constraints) and one Type-1 prior
    sentence per incoming interface message, encoding the belief about that
    interface atom as P(atom) in [lo, hi]. ARIEL then runs loopy interval belief
    propagation over the factor graph of this sub-LCN.

    Args:
        node: SuperNode
            The super-node whose local problem we are building.
        incoming: dict
            Mapping interface atom name -> (lower, upper) message interval. These
            are the messages arriving from the OTHER super-nodes; the destination
            atom of an outgoing message is excluded by the caller (BP rule).

    Returns:
        A fully-built LCN (primal graph, structure graph and Local Markov
        Condition computed), ready to hand to ArielInference.
    """
    subl = LCN()

    # Own constraints (internal + incident linking sentences)
    for _, s in node.sentences.items():
        subl.add_sentence(s)

    # Incoming interface messages as Type-1 priors P(atom) in [lo, hi]
    for atom, (lo, hi) in incoming.items():
        # Clamp to [0, 1] for numerical safety; a vacuous [0, 1] message simply
        # adds a trivial constraint.
        lo = max(0.0, min(1.0, lo))
        hi = max(lo, min(1.0, hi))
        subl.add_sentence(
            Sentence(label=f"m_{atom}", phi=atom, lower=lo, upper=hi)
        )

    # Build the graphs and the Local Markov Condition (mirrors LCN.from_lcn)
    subl.build_primal_graph()
    subl.build_structure_graph()
    subl.local_markov_condition()
    return subl


def _solve_supernode_ariel(
        node: SuperNode,
        incoming: Dict[str, Tuple[float, float]],
        n_iters: int,
        threshold: float,
        debug: bool = False
) -> Tuple[Dict[str, Tuple[float, float]], bool]:
    """
    Solve a cyclic super-node approximately using ARIEL on its local sub-LCN.

    A single ARIEL run yields lower/upper marginal bounds for EVERY atom in the
    super-node (interface and internal), so the caller can read off any target
    atom's marginal without re-solving.

    Args:
        node: SuperNode
            The (cyclic) super-node to solve.
        incoming: dict
            Incoming interface messages (atom -> (lo, hi)).
        n_iters: int
            Maximum number of ARIEL iterations.
        threshold: float
            ARIEL convergence threshold.
        debug: bool
            Debugging flag.

    Returns:
        A tuple (marginals, feasible) where marginals maps each atom name to its
        (lower, upper) bound on P(atom = 1).
    """
    subl = _build_supernode_lcn(node, incoming)
    algo = ArielInference(lcn=subl)
    res = algo.run(
        n_iters=n_iters,
        threshold=threshold,
        evidence={},
        debug=debug,
        verbosity=0
    )
    marginals = {atom: (lohi[0][1], lohi[1][1]) for atom, lohi in res.items()}
    return marginals, bool(algo.feasible)


# -----------------------------------------------------------------------
# SCC Propagation with ARIEL local solver
# -----------------------------------------------------------------------

class SCCPArielInference:
    """
    SCC Propagation for LCNs with an ARIEL local solver (v2).

    Same two-pass belief propagation over the acyclic condensation factor graph
    as `SCCFactorGraphInference` (sccp.py), but each cyclic super-node's
    factor-to-variable message is computed by running ARIEL on the super-node's
    local sub-LCN, rather than by solving one exact joint NLP. Trivial singleton
    super-nodes keep the exact ipopt LP solve.

    See docs/scc_factor_graph.tex (Section "A hybrid relationship").
    """

    def __init__(
            self,
            lcn: LCN
    ):
        self.lcn = lcn
        self.marginals = None
        self.feasible = None
        # Internal structures (populated in run())
        self.super_nodes = {}        # scc_id -> SuperNode
        self.caches = {}             # scc_id -> indicator cache (trivial nodes)
        self.scc_of_atom = {}
        self.cond_dag = None
        self.topo_order = []
        self.shared_atoms = set()    # message-carrying interface atoms
        self.f2v = {}                # (scc_id, atom) -> SCCMessage
        self.atom_to_sccs = {}       # shared atom -> list of scc ids
        # ARIEL knobs (set in run())
        self.n_iters = 10
        self.threshold = 1e-6

    # -- Setup --------------------------------------------------------------

    def _is_cyclic(self, scc_id: int) -> bool:
        """A super-node is cyclic (non-trivial) if it owns more than one atom."""
        return len(self.super_nodes[scc_id].owned) > 1

    def _setup(self, evidence: dict, verbosity: int):
        """Build the condensation, super-node factors, caches and adjacency.

        Identical to SCCFactorGraphInference._setup. Indicator caches are built
        for ALL super-nodes; trivial nodes use them for the exact solve, and
        cyclic nodes use them only for evidence/marginal bookkeeping (their solve
        goes through ARIEL on a rebuilt sub-LCN).
        """
        independencies = self.lcn.independencies

        # Stage 1: SCC decomposition
        (self.scc_of_atom, super_node_atoms,
         self.cond_dag, self.topo_order) = _build_condensation(self.lcn)
        internal_sents, linking_sents = _classify_sentences(
            self.lcn, self.scc_of_atom)

        # Stage 2: build the super-node factors (linking sentence owned by child)
        for scc_id, atoms in super_node_atoms.items():
            sn = SuperNode(scc_id=scc_id, owned=atoms)
            for sid in internal_sents.get(scc_id, []):
                sn.add_sentence(self.lcn.sentences[sid])
            for sid, owner in linking_sents:
                if owner == scc_id:
                    sn.add_sentence(self.lcn.sentences[sid])
            self.super_nodes[scc_id] = sn

        # Shared (interface) atoms from final scope overlap
        scope_count = {}
        for sn in self.super_nodes.values():
            for a in sn.scope:
                scope_count[a] = scope_count.get(a, 0) + 1
        self.shared_atoms = {a for a, c in scope_count.items() if c > 1}

        # Evidence: clamp atom to point mass, dropping any conflicting unary prior
        for var, val in evidence.items():
            phi = f"{var}" if val == 1 else f"!{var}"
            for scc_id, sn in self.super_nodes.items():
                if var not in sn.scope:
                    continue
                kept = {}
                for sid, s in sn.sentences.items():
                    is_unary_on_var = (set(s.atoms.keys()) == {var}
                                       and s.type == SentenceType.Type1)
                    if not is_unary_on_var:
                        kept[sid] = s
                sn.sentences = kept
                ev = Sentence(label=f"ev_{var}", phi=phi, lower=1.0, upper=1.0)
                sn.add_sentence(ev)

        # Mark shared (interface) atoms now that scopes are final
        for sn in self.super_nodes.values():
            sn.set_shared_atoms(self.shared_atoms)

        # Indicator caches (used by the exact solve for trivial nodes)
        for scc_id, sn in self.super_nodes.items():
            self.caches[scc_id] = _build_supernode_cache(sn, independencies)

        # Interface adjacency and message slots
        for scc_id, sn in self.super_nodes.items():
            for atom in sn.interface_atoms:
                self.atom_to_sccs.setdefault(atom, []).append(scc_id)
                self.f2v[(scc_id, atom)] = SCCMessage(atom, scc_id)

    # -- Message helpers ----------------------------------------------------

    def _v2f(self, atom: str, exclude_scc: int) -> Tuple[float, float]:
        """
        Variable-to-factor message: the belief about `atom` assembled from all of
        its neighbouring factors EXCEPT `exclude_scc` (interval intersection).
        """
        lo, hi = 0.0, 1.0
        for other in self.atom_to_sccs.get(atom, []):
            if other == exclude_scc:
                continue
            msg = self.f2v[(other, atom)]
            lo = max(lo, msg.lower_bound)
            hi = min(hi, msg.upper_bound)
        return lo, max(lo, hi)

    def _incoming_to_factor(self, scc_id: int, exclude_atom: str) -> Dict[str, Tuple[float, float]]:
        """
        Gather incoming variable-to-factor messages for super-node `scc_id`, one
        per interface atom, EXCLUDING `exclude_atom` (the destination, BP rule).
        """
        incoming = {}
        for atom in self.super_nodes[scc_id].interface_atoms:
            if atom == exclude_atom:
                continue
            incoming[atom] = self._v2f(atom, exclude_scc=scc_id)
        return incoming

    def _emit_factor_message(self, scc_id: int, atom: str, solver, debug: bool):
        """
        Compute and store the factor-to-variable message S{scc_id} -> atom.

        Dispatch by super-node type:
          * cyclic   -> run ARIEL on the super-node's sub-LCN and read off the
                        target atom's marginal;
          * trivial  -> exact ipopt min/max P(atom=1) (as in v1).
        """
        msg = self.f2v[(scc_id, atom)]
        incoming = self._incoming_to_factor(scc_id, exclude_atom=atom)

        if self._is_cyclic(scc_id):
            marg, feasible = _solve_supernode_ariel(
                self.super_nodes[scc_id], incoming,
                self.n_iters, self.threshold, debug)
            if not feasible:
                self.feasible = False
            lo, hi = marg.get(atom, (msg.lower_bound, msg.upper_bound))
            lo = max(lo, 0.0)
            hi = min(hi, 1.0)
        else:
            cache = self.caches[scc_id]
            lo_val, feas_lo = _solve_supernode(cache, incoming, atom, 'min', solver, debug)
            hi_val, feas_hi = _solve_supernode(cache, incoming, atom, 'max', solver, debug)
            if not feas_lo or not feas_hi:
                self.feasible = False
            lo = max(lo_val, 0.0) if (feas_lo and lo_val is not None) else msg.lower_bound
            hi = min(hi_val, 1.0) if (feas_hi and hi_val is not None) else msg.upper_bound

        msg.set_bounds(lo, hi)
        if debug:
            tag = "ariel" if self._is_cyclic(scc_id) else "exact"
            print(f"  emit  {msg}  ({tag})")

    def _edge_atoms(self, scc_id: int, neighbours: List[int]) -> set:
        """
        Interface variables on the edges between super-node `scc_id` and the
        given neighbours: the shared atoms present in both interface sets.
        """
        result = set()
        my_iface = self.super_nodes[scc_id].interface_atoms
        for nb in neighbours:
            nb_iface = self.super_nodes[nb].interface_atoms
            result |= (my_iface & nb_iface)
        return result

    # -- Main entry point ---------------------------------------------------

    def run(
            self,
            evidence: dict = {},
            n_iters: int = 10,
            threshold: float = 0.000001,
            debug: bool = False,
            verbosity: int = 2
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Run SCC Propagation (ARIEL local solver) to compute marginals for ALL
        atoms.

        Args:
            evidence: dict
                {variable_name: value} for observed variables.
            n_iters: int
                Maximum ARIEL iterations per cyclic super-node solve (default 10).
            threshold: float
                ARIEL convergence threshold (default 1e-6).
            debug: bool
                If True, show solver output and per-message traces.
            verbosity: int
                Verbosity level (0 is silent).

        Returns:
            Dict mapping each atom name to (lower_bounds, upper_bounds) numpy
            arrays (index 0 is P(atom=0) bounds, index 1 is P(atom=1) bounds);
            same format as ExactInference/ArielInference/SCCFactorGraphInference.
        """
        assert self.lcn is not None, "Make sure the LCN model exists."
        assert self.lcn.independencies is not None, "Make sure the LMC is applied."

        self.n_iters = n_iters
        self.threshold = threshold
        t_start = time.time()
        self.feasible = True

        if verbosity > 0:
            print(f"[SCCP-ARIEL] Computing all marginals")
            print(f"[SCCP-ARIEL] Evidence: {evidence}")

        # Build the condensation factor graph and caches
        self._setup(evidence, verbosity)

        if verbosity > 0:
            n_cyclic = sum(1 for sid in self.super_nodes if self._is_cyclic(sid))
            print(f"[SCCP-ARIEL] {len(self.super_nodes)} super-nodes "
                  f"({n_cyclic} cyclic -> ARIEL)")
            print(format_condensation_dag(
                self.super_nodes, self.cond_dag, self.topo_order))
            print(format_factorization(self.super_nodes, self.topo_order))

        # Shared, correctly-configured ipopt solver for trivial super-nodes
        solver = make_ipopt(debug=debug)

        # --- Pass 1: Collect (forward, topological order) ---
        if verbosity > 0:
            print(f"[SCCP-ARIEL] Collect pass (forward)...")
        for scc_id in self.topo_order:
            children = list(self.cond_dag.successors(scc_id))
            for atom in sorted(self._edge_atoms(scc_id, children)):
                self._emit_factor_message(scc_id, atom, solver, debug)

        # --- Pass 2: Distribute (backward, reverse topological order) ---
        if verbosity > 0:
            print(f"[SCCP-ARIEL] Distribute pass (backward)...")
        for scc_id in reversed(self.topo_order):
            parents = list(self.cond_dag.predecessors(scc_id))
            for atom in sorted(self._edge_atoms(scc_id, parents)):
                self._emit_factor_message(scc_id, atom, solver, debug)

        # --- Extract marginals ---
        self.marginals = self._extract_marginals(evidence, solver, debug)

        t_end = time.time()

        if verbosity > 0:
            print(f"[SCCP-ARIEL] Singleton variable marginals:")
            for atom_name in sorted(self.marginals):
                lo, hi = self.marginals[atom_name]
                for val in range(len(lo)):
                    print(f"  P({atom_name}={val}): [{lo[val]:.6f}, {hi[val]:.6f}]")
            print(f"[SCCP-ARIEL] Feasible: {self.feasible}")
            print(f"[SCCP-ARIEL] Time elapsed: {t_end - t_start:.4f} sec")

        return self.marginals

    def _extract_marginals(
            self,
            evidence: dict,
            solver,
            debug: bool
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute final marginals. Interface atoms are obtained by intersecting all
        incoming factor-to-variable messages. Internal (hidden) atoms of a cyclic
        super-node are read from a single final ARIEL run constrained by all the
        super-node's final incoming messages; internal atoms of a trivial node
        (none, by definition) would fall back to the exact NLP.
        """
        results = {}
        evidence_set = set(evidence.keys())

        for scc_id, sn in self.super_nodes.items():
            incoming_all = {
                atom: self._v2f(atom, exclude_scc=scc_id)
                for atom in sn.interface_atoms
            }

            # For a cyclic super-node, one final ARIEL run gives every owned
            # atom's marginal at once.
            ariel_marg = None
            if self._is_cyclic(scc_id):
                ariel_marg, feasible = _solve_supernode_ariel(
                    sn, incoming_all, self.n_iters, self.threshold, debug)
                if not feasible:
                    self.feasible = False

            for atom in sn.owned:
                if atom in evidence_set:
                    ev_val = evidence[atom]
                    lo_arr = np.zeros(2)
                    hi_arr = np.zeros(2)
                    lo_arr[ev_val] = 1.0
                    hi_arr[ev_val] = 1.0
                    results[atom] = (lo_arr, hi_arr)
                    continue

                if atom in sn.interface_atoms:
                    # Interface atom: intersect all incoming factor messages
                    lo, hi = 0.0, 1.0
                    for other in self.atom_to_sccs.get(atom, []):
                        msg = self.f2v[(other, atom)]
                        lo = max(lo, msg.lower_bound)
                        hi = min(hi, msg.upper_bound)
                    lo_1, hi_1 = lo, max(lo, hi)
                elif ariel_marg is not None:
                    # Internal atom of a cyclic super-node (e.g. B, D): read the
                    # ARIEL marginal from the final run.
                    lo_1, hi_1 = ariel_marg.get(atom, (0.0, 1.0))
                    lo_1 = max(lo_1, 0.0)
                    hi_1 = max(lo_1, min(hi_1, 1.0))
                else:
                    # Internal atom of a trivial super-node: exact NLP fallback.
                    cache = self.caches[scc_id]
                    lo_val, feas_lo = _solve_supernode(
                        cache, incoming_all, atom, 'min', solver, debug)
                    hi_val, feas_hi = _solve_supernode(
                        cache, incoming_all, atom, 'max', solver, debug)
                    if not feas_lo or not feas_hi:
                        self.feasible = False
                    lo_1 = max(lo_val, 0.0) if (feas_lo and lo_val is not None) else 0.0
                    hi_1 = min(hi_val, 1.0) if (feas_hi and hi_val is not None) else 1.0
                    hi_1 = max(lo_1, hi_1)

                if lo_1 > hi_1:
                    self.feasible = False
                lo_arr = np.array([1.0 - hi_1, lo_1])
                hi_arr = np.array([1.0 - lo_1, hi_1])
                results[atom] = (lo_arr, hi_arr)

        return results


if __name__ == "__main__":

    def print_singleton_marginals(results):
        """Print only singleton variable marginals from the results."""
        print("  Singleton variable marginals:")
        for var in sorted(results):
            lo, hi = results[var]
            for val in range(len(lo)):
                print(f"    P({var}={val}): [{lo[val]:.6f}, {hi[val]:.6f}]")

    # Load the LCN (the 7-variable example from docs/scc_factor_graph.tex)
    file_name = "examples/new2.lcn"
    l = LCN()
    l.from_lcn(file_name=file_name)
    print(l)

    # Check consistency
    ok = check_consistency(l)
    print("CONSISTENT" if ok else "INCONSISTENT")

    # Run SCC Propagation with the ARIEL local solver (no evidence)
    print("\n=== SCCPArielInference (no evidence) ===")
    algo = SCCPArielInference(lcn=l)
    results = algo.run(evidence={}, n_iters=20, threshold=1e-7, debug=False, verbosity=2)
    print_singleton_marginals(results)
