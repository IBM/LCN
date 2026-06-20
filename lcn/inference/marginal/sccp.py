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

# SCC Factor-Graph Propagation (SCC-FGP): a hybrid approximate marginal
# inference algorithm for LCNs. The LCN's structure graph is decomposed into
# its Strongly Connected Components (SCCs); each cycle is contracted into a
# super-node, yielding an acyclic condensation graph (DAG). Two-pass belief
# propagation (collect + distribute) then runs exactly on the resulting factor
# graph, where factor nodes are super-nodes, variable nodes are the interface
# atoms shared between super-nodes, and each factor-to-variable message is a
# local non-linear program solved by ipopt.
#
# See docs/scc_factor_graph.tex for the full algorithm description, the
# 7-variable worked example, and the side-by-side comparison with ARIEL.

import itertools
import time
import networkx as nx
import numpy as np
from pyomo.environ import *
from typing import Dict, List, Tuple

# Local
from lcn.core.model import LCN, SentenceType, Formula, Sentence
from lcn.core.independencies import Independencies
from lcn.inference.utils.common import check_consistency, make_conjunction
from lcn.inference.marginal.exact import _eval_indicator, _dot, _solve_with_objective


# -----------------------------------------------------------------------
# Stage 1: SCC decomposition of the structure graph
# -----------------------------------------------------------------------

def _build_condensation(lcn: LCN) -> Tuple[Dict[str, int], Dict[int, List[str]], "nx.DiGraph", List[int]]:
    """
    Decompose the LCN's structure graph into Strongly Connected Components
    (SCCs) and build the condensation graph (a DAG of super-nodes).

    The structure graph is a MixedGraph. We orient it as a directed graph by
    keeping directed edges as-is and treating each undirected edge {u, v} as
    the symmetric pair u -> v and v -> u (so atoms tied by an undirected edge
    land in the same SCC). Tarjan's algorithm (via networkx.condensation)
    contracts every cycle into a single super-node.

    Args:
        lcn: LCN
            The input LCN model (its structure graph must be built).

    Returns:
        A tuple (scc_of_atom, super_nodes, cond_dag, topo_order) where:
        - scc_of_atom: dict mapping each atom name to its super-node id (int).
        - super_nodes: dict mapping super-node id to a sorted list of atoms.
        - cond_dag: nx.DiGraph over super-node ids (the condensation DAG).
        - topo_order: list of super-node ids in topological order.
    """
    assert lcn.structure_graph is not None, "The structure graph must be built."
    sg = lcn.structure_graph

    # Orient the mixed structure graph as a directed graph
    G = nx.DiGraph()
    G.add_nodes_from(lcn.atoms.keys())
    for u, v in sg._directed.edges():
        G.add_edge(u, v)
    for u, v in sg._undirected.edges():
        G.add_edge(u, v)
        G.add_edge(v, u)

    # Tarjan's algorithm: contract SCCs into a condensation DAG
    cond_dag = nx.condensation(G)

    scc_of_atom = {}
    super_nodes = {}
    for sid in cond_dag.nodes:
        members = sorted(cond_dag.nodes[sid]["members"])
        super_nodes[sid] = members
        for atom in members:
            scc_of_atom[atom] = sid

    topo_order = list(nx.topological_sort(cond_dag))
    return scc_of_atom, super_nodes, cond_dag, topo_order


def _classify_sentences(
        lcn: LCN,
        scc_of_atom: Dict[str, int]
) -> Tuple[Dict[int, List[str]], List[Tuple[str, frozenset]]]:
    """
    Classify each LCN sentence as internal to a single super-node, or linking
    two super-nodes.

    A sentence is internal if all its atoms belong to one super-node. It is
    linking if its atoms span more than one super-node (e.g. a conditional
    P(B|A) where A and B are in different SCCs). Following
    docs/scc_factor_graph.tex, a linking constraint P(phi | psi) is OWNED by the
    child super-node --- the one containing the conditioned atoms phi. The
    parent super-node (containing the conditioning atoms psi) merely supplies a
    message about the interface variable psi; it does not own the constraint.
    The owner's joint scope therefore gains the parent-side psi atom (as an
    interface variable) but the parent's scope does NOT gain the child's phi.

    Args:
        lcn: LCN
            The input LCN model.
        scc_of_atom: dict
            Mapping from atom name to super-node id.

    Returns:
        A tuple (internal_sentences, linking_sentences) where:
        - internal_sentences: dict super-node id -> list of sentence labels.
        - linking_sentences: list of (sentence label, owner_scc_id) where the
          owner is the child super-node containing the conditioned (phi) atoms.
    """
    internal_sentences = {}
    linking_sentences = []
    for sid, s in lcn.sentences.items():
        sccs = sorted({scc_of_atom[a] for a in s.atoms.keys()})
        if len(sccs) == 1:
            internal_sentences.setdefault(sccs[0], []).append(sid)
        else:
            # Linking sentence: owner = child SCC containing the phi atoms.
            # (Type-1 linking sentences are rare; assign to any touched SCC.)
            if s.type == SentenceType.Type2:
                phi_atoms = list(s.phi_formula.atoms.values())
                owner = scc_of_atom[phi_atoms[0]]
            else:
                owner = sccs[0]
            linking_sentences.append((sid, owner))
    return internal_sentences, linking_sentences


def _compute_shared_atoms(
        lcn: LCN,
        linking_sentences: List[Tuple[str, frozenset]]
) -> set:
    """
    Compute the set of interface variables (message-carrying atoms) of the
    condensation factor graph.

    Following docs/scc_factor_graph.tex, the interface variable of a linking
    constraint P(phi | psi) is its conditioning (psi) atom --- the parent-side
    atom whose belief is propagated to the child super-node. The conditioned
    (phi) atom remains internal to its (child) super-node. For a Type-1 linking
    sentence (no psi) every atom is treated as an interface variable.

    Example (running 7-variable LCN): the linking sentences c_XA, c_AB, c_YB,
    c_CE, c_nCE have conditioning atoms X, A, Y, C, C, so the interface
    variables are {X, A, Y, C} --- exactly the doc's variable nodes. B and D
    stay internal to S_BCD.

    Args:
        lcn: LCN
            The input LCN model.
        linking_sentences: list
            The linking sentences from _classify_sentences.

    Returns:
        A set of interface (message-carrying) atom names.
    """
    shared = set()
    for sid, _ in linking_sentences:
        s = lcn.sentences[sid]
        if s.type == SentenceType.Type2:
            shared.update(s.psi_formula.atoms.values())
        else:
            shared.update(s.atoms.keys())
    return shared


# -----------------------------------------------------------------------
# Stage 2: Super-node factor model and indicator cache
# -----------------------------------------------------------------------

class SuperNode:
    """
    A super-node (factor node) in the condensation factor graph.

    The super-node owns the atoms of one SCC. Its assigned sentences are the
    internal sentences (scope within the SCC) plus every incident linking
    sentence (one endpoint in this SCC). The factor's full joint scope is the
    union of the atoms over all assigned sentences --- this includes the SCC's
    own atoms AND the neighbour-side atoms referenced by linking sentences
    (e.g. S_BCD's scope is {B,C,D, A,Y} because c_AB references A and c_YB
    references Y).

    For the running example (docs/scc_factor_graph.tex):
        S_BCD: owned={B,C,D}, scope={A,B,C,D,Y},
               interface(shared, owned)={B,C}, external(shared, not owned)={A,Y}.
    """

    def __init__(
            self,
            scc_id: int,
            owned: List[str]
    ):
        self.scc_id = scc_id
        self.owned = list(owned)        # atoms whose SCC is this one
        self.sentences = {}             # label -> Sentence
        self.scope = list(owned)        # recomputed as sentences are added
        self._scope_set = set(owned)

    def add_sentence(self, s: Sentence) -> None:
        self.sentences[s.label] = s
        for a in s.atoms.keys():
            if a not in self._scope_set:
                self._scope_set.add(a)
        self.scope = sorted(self._scope_set)

    def set_shared_atoms(self, shared: set) -> None:
        """
        Record which scope atoms are shared (message-carrying) interface atoms.

        - interface_atoms: shared atoms in scope (those that carry messages to
          or from this factor), whether owned by this SCC or by a neighbour.
        - external_atoms: shared atoms in scope NOT owned by this SCC (incoming
          parent/sibling beliefs).
        - internal_atoms: owned atoms that are NOT shared (hidden; their final
          marginal needs an extra local NLP).
        """
        self.interface_atoms = set(self.scope) & shared
        self.external_atoms = self.interface_atoms - set(self.owned)
        self.internal_atoms = [a for a in self.owned if a not in shared]

    def __str__(self):
        output = f"SuperNode S{self.scc_id}\n"
        output += f"  owned: {self.owned}\n"
        output += f"  scope: {self.scope}\n"
        output += f"  interface: {sorted(getattr(self, 'interface_atoms', set()))}\n"
        output += f"  sentences: {list(self.sentences.keys())}\n"
        return output


def _build_supernode_cache(
        node: SuperNode,
        independencies: Independencies
) -> dict:
    """
    Pre-compute and cache the interpretations and indicator vectors for a
    super-node factor over its full joint scope. Mirrors ariel._build_factor_cache
    but indexes every atom in the joint scope (internal + interface) so that we
    can both add message constraints and optimize the marginal of any atom.

    Returns a dict with:
        - 'N': number of interpretations (2^|scope|)
        - 'sentence_indicators': dict sid -> (type, lo, hi, indicators)
        - 'variable_indicators': dict atom -> indicator array (over the scope)
        - 'independence_groups': list of cached LMC constraint indicators
    """
    vars_list = node.scope
    items_tuples = list(itertools.product([0, 1], repeat=len(vars_list)))
    interpretations = [dict(zip(vars_list, t)) for t in items_tuples]
    N = len(interpretations)

    # Cache sentence indicators (internal + linking sentences of this factor)
    sentence_indicators = {}
    for sid, s in node.sentences.items():
        if s.type == SentenceType.Type1:
            A = _eval_indicator(s.phi_formula, interpretations)
            sentence_indicators[sid] = ('type1', s.get_lower_bound(),
                                        s.get_upper_bound(), A)
        else:
            Aqr = _eval_indicator(s.phi_and_psi_formula, interpretations)
            Ar = _eval_indicator(s.psi_formula, interpretations)
            sentence_indicators[sid] = ('type2', s.get_lower_bound(),
                                        s.get_upper_bound(), Aqr, Ar)

    # Cache variable indicators for every atom in the joint scope
    variable_indicators = {}
    for v in vars_list:
        variable_indicators[v] = _eval_indicator(
            Formula(label=v, formula=v), interpretations)

    # Cache independence (LMC) constraint indicators local to the joint scope
    scope_set = set(vars_list)
    independence_groups = []
    for indep in independencies.get_assertions():
        X, T, S = list(indep.event1), list(indep.event2), list(indep.event3)
        all_vars = set(X) | set(T) | set(S)
        if not all_vars.issubset(scope_set):
            continue
        configs_S = [()] if len(S) == 0 else list(
            itertools.product([0, 1], repeat=len(S)))
        if len(S) > 0:
            for t in T:
                x = X[0]
                literals = {x: 1, t: 1}
                for cfg in configs_S:
                    literals.update(dict(zip(S, list(cfg))))
                    Fa = make_conjunction(variables=X + S + [t], literals=literals)
                    Fb = make_conjunction(variables=S, literals=literals)
                    Fc = make_conjunction(variables=X + S, literals=literals)
                    Fd = make_conjunction(variables=S + [t], literals=literals)
                    Aa = _eval_indicator(Fa, interpretations)
                    Ab = _eval_indicator(Fb, interpretations)
                    Ac = _eval_indicator(Fc, interpretations)
                    Ad = _eval_indicator(Fd, interpretations)
                    independence_groups.append(('conditional', Aa, Ab, Ac, Ad))
        else:
            for t in T:
                x = X[0]
                literals = {x: 1, t: 1}
                Fa = make_conjunction(variables=X + [t], literals=literals)
                Fb = make_conjunction(variables=X, literals=literals)
                Fc = make_conjunction(variables=[t], literals=literals)
                Aa = _eval_indicator(Fa, interpretations)
                Ab = _eval_indicator(Fb, interpretations)
                Ac = _eval_indicator(Fc, interpretations)
                independence_groups.append(('marginal', Aa, Ab, Ac))

    return {
        'N': N,
        'sentence_indicators': sentence_indicators,
        'variable_indicators': variable_indicators,
        'independence_groups': independence_groups,
    }


# -----------------------------------------------------------------------
# Stage 3: Local NLP solve over a super-node's joint space
# -----------------------------------------------------------------------

def _solve_supernode(
        cache: dict,
        incoming: Dict[str, Tuple[float, float]],
        target_atom: str,
        sense: str,
        solver,
        debug: bool = False
) -> Tuple[float, bool]:
    """
    Build and solve the local NLP of a super-node factor over its joint space.
    Computes min/max P(target_atom = 1) subject to the factor's internal,
    linking and LMC constraints, plus the incoming interface-message bounds.

    Args:
        cache: dict
            Pre-computed indicators for the super-node (from _build_supernode_cache).
        incoming: dict
            Mapping interface atom name -> (lower, upper) message interval. These
            are the messages arriving from the OTHER super-nodes (the destination
            atom's own incoming message must be excluded by the caller).
        target_atom: str
            The atom whose marginal P(target_atom = 1) is optimized.
        sense: str
            Either 'min' or 'max'.
        solver: SolverFactory
            Reusable ipopt solver instance.
        debug: bool
            Debugging flag.

    Returns:
        A tuple (objective_value, feasible).
    """
    assert sense in ("min", "max")

    N = cache['N']
    sentence_indicators = cache['sentence_indicators']
    variable_indicators = cache['variable_indicators']
    independence_groups = cache['independence_groups']

    # Create the Pyomo model and the joint distribution variables
    model = ConcreteModel()
    model.ITEMS = Set(initialize=range(N))
    model.p = Var(model.ITEMS, within=NonNegativeReals)
    model.constr = ConstraintList()

    # Probability distribution constraint
    model.constr.add(sum(model.p[i] for i in model.ITEMS) == 1.0)

    # Sentence constraints (internal + linking), using cached indicators
    for sid, indicators in sentence_indicators.items():
        if indicators[0] == 'type1':
            _, lobo, upbo, A = indicators
            expr = _dot(A, model, model.ITEMS)
            model.constr.add(expr >= lobo)
            model.constr.add(expr <= upbo)
        else:
            _, lobo, upbo, Aqr, Ar = indicators
            expr_qr = _dot(Aqr, model, model.ITEMS)
            expr_r = _dot(Ar, model, model.ITEMS)
            model.constr.add(expr_qr >= lobo * expr_r)
            model.constr.add(expr_qr <= upbo * expr_r)

    # Incoming interface-message constraints (hard bounds; the condensation
    # graph is acyclic so there is no double-counting to relax with slack).
    for atom, (lo, hi) in incoming.items():
        if atom not in variable_indicators:
            continue
        expr = _dot(variable_indicators[atom], model, model.ITEMS)
        model.constr.add(expr >= lo)
        model.constr.add(expr <= hi)

    # Independence (LMC) constraints local to the joint scope
    for group in independence_groups:
        if group[0] == 'conditional':
            _, Aa, Ab, Ac, Ad = group
            val1 = _dot(Aa, model, model.ITEMS) * _dot(Ab, model, model.ITEMS)
            val2 = _dot(Ac, model, model.ITEMS) * _dot(Ad, model, model.ITEMS)
            model.constr.add(val1 - val2 == 0.0)
        else:
            _, Aa, Ab, Ac = group
            val1 = _dot(Aa, model, model.ITEMS)
            val2 = _dot(Ab, model, model.ITEMS) * _dot(Ac, model, model.ITEMS)
            model.constr.add(val1 - val2 == 0.0)

    # Objective: min/max P(target_atom = 1)
    A_obj = variable_indicators[target_atom]
    obj_expr = _dot(A_obj, model, model.ITEMS)
    return _solve_with_objective(model, obj_expr, sense, solver, debug)


# -----------------------------------------------------------------------
# Messages
# -----------------------------------------------------------------------

class SCCMessage:
    """
    A message on an edge between an interface variable node and a super-node
    factor in the condensation factor graph. A message is a credal set over a
    single (Boolean) interface atom, represented by an interval [lower, upper]
    bounding P(atom = 1).
    """

    def __init__(self, atom: str, scc_id: int):
        self.atom = atom            # the interface atom this message is about
        self.scc_id = scc_id        # the super-node at the factor end
        self.lower_bound = 0.0
        self.upper_bound = 1.0

    def set_bounds(self, lo: float, hi: float):
        self.lower_bound = lo
        self.upper_bound = max(lo, hi)  # enforce lower <= upper

    def __str__(self):
        return f"S{self.scc_id}<->{self.atom}: [{self.lower_bound:.4f}, {self.upper_bound:.4f}]"


# -----------------------------------------------------------------------
# SCC Factor-Graph Propagation inference
# -----------------------------------------------------------------------

class SCCFactorGraphInference:
    """
    SCC Factor-Graph Propagation (SCC-FGP) for LCNs.

    Decomposes the LCN structure graph into SCCs, builds the acyclic
    condensation factor graph, and runs two-pass belief propagation (collect +
    distribute) over it. Each factor-to-variable message is computed by an
    exact local NLP (ipopt) over the super-node's joint space.

    See docs/scc_factor_graph.tex for details.
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
        self.caches = {}             # scc_id -> indicator cache
        self.scc_of_atom = {}
        self.cond_dag = None
        self.topo_order = []
        self.shared_atoms = set()    # message-carrying interface atoms
        # Factor-to-variable messages: (scc_id, atom) -> SCCMessage
        self.f2v = {}
        # Adjacency: shared atom -> list of scc ids whose scope contains it
        self.atom_to_sccs = {}

    # -- Setup --------------------------------------------------------------

    def _setup(self, evidence: dict, verbosity: int):
        """Build the condensation, super-node factors, caches and adjacency."""
        independencies = self.lcn.independencies

        # Stage 1: SCC decomposition
        (self.scc_of_atom, super_node_atoms,
         self.cond_dag, self.topo_order) = _build_condensation(self.lcn)
        internal_sents, linking_sents = _classify_sentences(
            self.lcn, self.scc_of_atom)

        # Stage 2: build the super-node factors. A linking sentence is owned by
        # its child super-node; add_sentence pulls ALL of the sentence's atoms
        # into the owner's joint scope (so the indicators are correct even for
        # multi-atom formulas such as P(B or C | S), where the conditioned
        # formula itself spans several SCCs).
        for scc_id, atoms in super_node_atoms.items():
            node = SuperNode(scc_id=scc_id, owned=atoms)
            for sid in internal_sents.get(scc_id, []):
                node.add_sentence(self.lcn.sentences[sid])
            for sid, owner in linking_sents:
                if owner == scc_id:
                    node.add_sentence(self.lcn.sentences[sid])
            self.super_nodes[scc_id] = node

        # Interface (shared) atoms are derived from the FINAL scopes: an atom is
        # shared (message-carrying) iff it appears in more than one super-node's
        # joint scope. This is robust to linking sentences whose formulas span
        # several SCCs, and guarantees each cross-SCC coupling is mediated by a
        # message (so the result stays a valid outer approximation).
        scope_count = {}
        for node in self.super_nodes.values():
            for a in node.scope:
                scope_count[a] = scope_count.get(a, 0) + 1
        self.shared_atoms = {a for a, c in scope_count.items() if c > 1}

        # Evidence: add a point-mass unary sentence P(atom)=val into every
        # super-node whose joint scope contains the evidence atom (mirrors
        # FactorGraph.add_evidence semantics).
        # We clamp the evidence atom to a point mass P(atom)=val. To avoid a
        # contradiction with a prior bound on that atom (e.g. evidence X=1 vs a
        # prior c_X: P(X) in [0.6,0.8]), we first DROP any existing unary
        # sentence on that atom from the super-node, then add the point mass ---
        # mirroring FactorGraph.add_evidence, so the observation overrides the
        # prior (the extracted marginals are then conditional on the evidence).
        for var, val in evidence.items():
            phi = f"{var}" if val == 1 else f"!{var}"
            for scc_id, node in self.super_nodes.items():
                if var not in node.scope:
                    continue
                kept = {}
                for sid, s in node.sentences.items():
                    is_unary_on_var = (set(s.atoms.keys()) == {var}
                                       and s.type == SentenceType.Type1)
                    if not is_unary_on_var:
                        kept[sid] = s
                node.sentences = kept
                ev = Sentence(label=f"ev_{var}", phi=phi, lower=1.0, upper=1.0)
                node.add_sentence(ev)

        # Mark shared (interface) atoms on each factor now that scopes are final
        for node in self.super_nodes.values():
            node.set_shared_atoms(self.shared_atoms)

        # Build the caches (once per super-node)
        for scc_id, node in self.super_nodes.items():
            self.caches[scc_id] = _build_supernode_cache(node, independencies)

        # Interface adjacency: a shared atom attaches to every super-node whose
        # joint scope contains it. The message slot (scc_id, atom) holds the
        # factor-to-variable message emitted by that super-node about that atom.
        for scc_id, node in self.super_nodes.items():
            for atom in node.interface_atoms:
                self.atom_to_sccs.setdefault(atom, []).append(scc_id)
                self.f2v[(scc_id, atom)] = SCCMessage(atom, scc_id)

        if verbosity > 1:
            print(f"[SCCFactorGraph] Super-nodes ({len(self.super_nodes)}):")
            for scc_id in self.topo_order:
                print("  " + str(self.super_nodes[scc_id]).replace("\n", "\n  "))

    # -- Message helpers ----------------------------------------------------

    def _v2f(self, atom: str, exclude_scc: int) -> Tuple[float, float]:
        """
        Variable-to-factor message: the belief about `atom` assembled from all
        of its neighbouring factors EXCEPT `exclude_scc`. For intervals this is
        the intersection [max of lowers, min of uppers]. Returns the vacuous
        interval [0, 1] when there is no other neighbour.
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
        Gather the incoming variable-to-factor messages for super-node `scc_id`,
        one per interface atom, EXCLUDING `exclude_atom` (the destination of the
        outgoing message, per the belief-propagation rule).
        """
        incoming = {}
        for atom in self.super_nodes[scc_id].interface_atoms:
            if atom == exclude_atom:
                continue
            incoming[atom] = self._v2f(atom, exclude_scc=scc_id)
        return incoming

    def _emit_factor_message(self, scc_id: int, atom: str, solver, debug: bool):
        """
        Compute and store the factor-to-variable message S{scc_id} -> atom by
        solving the local NLP (min/max P(atom=1)) with all OTHER interface
        messages as constraints.
        """
        cache = self.caches[scc_id]
        incoming = self._incoming_to_factor(scc_id, exclude_atom=atom)
        lo_val, feas_lo = _solve_supernode(cache, incoming, atom, 'min', solver, debug)
        hi_val, feas_hi = _solve_supernode(cache, incoming, atom, 'max', solver, debug)

        msg = self.f2v[(scc_id, atom)]
        if not feas_lo or not feas_hi:
            self.feasible = False
        lo = max(lo_val, 0.0) if (feas_lo and lo_val is not None) else msg.lower_bound
        hi = min(hi_val, 1.0) if (feas_hi and hi_val is not None) else msg.upper_bound
        msg.set_bounds(lo, hi)
        if debug:
            print(f"  emit  {msg}")

    # -- Main entry point ---------------------------------------------------

    def run(
            self,
            evidence: dict = {},
            debug: bool = False,
            verbosity: int = 2
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Run SCC Factor-Graph Propagation to compute marginals for ALL atoms.

        Args:
            evidence: dict
                {variable_name: value} for observed variables.
            debug: bool
                If True, show solver output and per-message traces.
            verbosity: int
                Verbosity level (0 is silent).

        Returns:
            Dict mapping each atom name to (lower_bounds, upper_bounds) numpy
            arrays, where index 0 is the P(atom=0) bounds and index 1 the
            P(atom=1) bounds (same format as ExactInference/ArielInference).
        """
        assert self.lcn is not None, "Make sure the LCN model exists."
        assert self.lcn.independencies is not None, "Make sure the LMC is applied."

        t_start = time.time()
        self.feasible = True
        evidence_set = set(evidence.keys())

        if verbosity > 0:
            print(f"[SCCFactorGraph] Computing all marginals")
            print(f"[SCCFactorGraph] Evidence: {evidence}")

        # Build the condensation factor graph and caches
        self._setup(evidence, verbosity)

        if verbosity > 0:
            n_cyclic = sum(1 for n in self.super_nodes.values() if len(n.owned) > 1)
            print(f"[SCCFactorGraph] {len(self.super_nodes)} super-nodes "
                  f"({n_cyclic} cyclic), topological order: {self.topo_order}")

        # Shared solver instance
        solver = SolverFactory('ipopt')
        if not debug:
            solver.options['print_level'] = 0

        # --- Pass 1: Collect (forward, topological order) ---
        if verbosity > 0:
            print(f"[SCCFactorGraph] Collect pass (forward)...")
        for scc_id in self.topo_order:
            children = list(self.cond_dag.successors(scc_id))
            child_atoms = self._edge_atoms(scc_id, children)
            for atom in sorted(child_atoms):
                self._emit_factor_message(scc_id, atom, solver, debug)

        # --- Pass 2: Distribute (backward, reverse topological order) ---
        if verbosity > 0:
            print(f"[SCCFactorGraph] Distribute pass (backward)...")
        for scc_id in reversed(self.topo_order):
            parents = list(self.cond_dag.predecessors(scc_id))
            parent_atoms = self._edge_atoms(scc_id, parents)
            for atom in sorted(parent_atoms):
                self._emit_factor_message(scc_id, atom, solver, debug)

        # --- Stage 4: Extract marginals ---
        self.marginals = self._extract_marginals(evidence, solver, debug)

        t_end = time.time()

        if verbosity > 0:
            print(f"[SCCFactorGraph] Singleton variable marginals:")
            for atom_name in sorted(self.marginals):
                lo, hi = self.marginals[atom_name]
                for val in range(len(lo)):
                    print(f"  P({atom_name}={val}): [{lo[val]:.6f}, {hi[val]:.6f}]")
            print(f"[SCCFactorGraph] Feasible: {self.feasible}")
            print(f"[SCCFactorGraph] Time elapsed: {t_end - t_start:.4f} sec")

        return self.marginals

    def _edge_atoms(self, scc_id: int, neighbours: List[int]) -> set:
        """
        The interface variables on the edges between super-node `scc_id` and the
        given neighbouring super-nodes --- i.e. the shared atoms present in both
        scopes. A factor sends a message about each such variable to the
        neighbour on that edge, in BOTH propagation directions (e.g. on the
        S_BCD--S_E edge the variable is C: S_BCD sends mu about C in the collect
        pass, S_E sends mu about C back in the distribute pass).
        """
        result = set()
        my_iface = self.super_nodes[scc_id].interface_atoms
        for nb in neighbours:
            nb_iface = self.super_nodes[nb].interface_atoms
            result |= (my_iface & nb_iface)
        return result

    def _extract_marginals(
            self,
            evidence: dict,
            solver,
            debug: bool
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute final marginals: interface atoms by intersecting all incoming
        factor-to-variable messages; internal (hidden) atoms by one final local
        NLP per super-node using all final incoming messages.
        """
        results = {}
        evidence_set = set(evidence.keys())

        for scc_id, node in self.super_nodes.items():
            # Final incoming messages for this super-node (all interface atoms)
            incoming_all = {
                atom: self._v2f(atom, exclude_scc=scc_id)
                for atom in node.interface_atoms
            }
            cache = self.caches[scc_id]

            for atom in node.owned:
                if atom in evidence_set:
                    ev_val = evidence[atom]
                    lo_arr = np.zeros(2)
                    hi_arr = np.zeros(2)
                    lo_arr[ev_val] = 1.0
                    hi_arr[ev_val] = 1.0
                    results[atom] = (lo_arr, hi_arr)
                    continue

                if atom in node.interface_atoms:
                    # Interface atom: intersect all incoming factor messages
                    lo, hi = 0.0, 1.0
                    for other in self.atom_to_sccs.get(atom, []):
                        msg = self.f2v[(other, atom)]
                        lo = max(lo, msg.lower_bound)
                        hi = min(hi, msg.upper_bound)
                    lo_1, hi_1 = lo, max(lo, hi)
                else:
                    # Internal atom: one final NLP constrained by all incoming
                    # messages (Eq. final-internal in the doc).
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

    # Load the LCN (the 7-variable example)
    file_name = "examples/new3.lcn"
    l = LCN()
    l.from_lcn(file_name=file_name)
    print(l)

    # Check consistency
    ok = check_consistency(l)
    print("CONSISTENT" if ok else "INCONSISTENT")

    # Run SCC Factor-Graph Propagation (no evidence)
    print("\n=== SCCFactorGraphInference (no evidence) ===")
    algo = SCCFactorGraphInference(lcn=l)
    results = algo.run(evidence={}, debug=False, verbosity=2)
    print_singleton_marginals(results)
