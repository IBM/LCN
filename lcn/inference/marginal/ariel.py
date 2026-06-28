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

# ARIEL: Approximate marginal inference for LCNs
# Based on: Marinescu et al. Approximate Inference in LCNs. IJCAI-2023.

import itertools
import time
import numpy as np
from pyomo.environ import (
    ConcreteModel,
    ConstraintList,
    NonNegativeReals,
    Objective,
    Set,
    SolverStatus,
    TerminationCondition,
    Var,
    maximize,
    minimize,
)
from typing import Dict, List, Tuple

# Local
from lcn.core.model import LCN, SentenceType, Formula
from lcn.core.independencies import Independencies, IndependenceAssertion
from lcn.inference.utils.factor_graph import FactorGraph, FactorNode, VariableNode, FactorGraphEdge
from lcn.inference.utils.common import (
    check_consistency, make_ipopt, lmc_constraint_groups_vec, build_truth_table
)
from lcn.inference.utils.common import eval_indicator, dot
from lcn.inference.marginal.sccp import _wrap_items


def format_factor_box(f: FactorNode, width: int = 56) -> str:
    """
    Render a single factor node as a titled ASCII box listing its scope (the
    boundary variables), and the LCN sentences it groups. The visual style
    mirrors ``format_supernode_box`` in lcn.inference.marginal.sccp.

    Args:
        f: FactorNode
            The factor node to render.
        width: int
            Target inner width of the box in characters.

    Returns:
        A multi-line string containing the box.
    """
    label = f.get_label()
    scope = sorted(f.scope)
    sentences = sorted(f.sentences.keys())

    rows = [
        ("scope", scope),
        ("sentences", sentences),
    ]
    pad = max(len(h) for h, _ in rows)          # heading column width
    avail = width - pad - 3                      # room left for the values
    body_lines = []
    for heading, items in rows:
        wrapped = _wrap_items([str(i) for i in items], avail)
        for k, chunk in enumerate(wrapped):
            head = heading if k == 0 else ""
            body_lines.append(f"{head:<{pad}} : {chunk}")

    inner = max([len(line) for line in body_lines] + [len(label) + 4])
    inner = max(inner, width)
    top = f"+-- {label} " + "-" * (inner - len(label) - 3) + "+"
    bot = "+" + "-" * (inner + 1) + "+"
    out = [top]
    for line in body_lines:
        out.append(f"| {line:<{inner}}|")
    out.append(bot)
    return "\n".join(out)


def format_factor_graph(fg: FactorGraph) -> str:
    """
    Render the factor graph being processed by ARIEL in a user-friendly form:
    a header with node counts, the variable nodes, an aligned variable--factor
    edge list, and a detail box per factor node (in sorted label order).

    Args:
        fg: FactorGraph
            The factor graph to render.

    Returns:
        A multi-line string ready to print.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Factor Graph")
    lines.append("=" * 60)
    lines.append(f"# variable nodes: {len(fg.variable_nodes)}")
    lines.append(f"# factor nodes  : {len(fg.factor_nodes)}")
    lines.append("Variables: " + ", ".join(sorted(fg.variable_nodes.keys())))
    lines.append("")

    # Aligned edge list (variable -- factor), ordered by factor then variable.
    edges = sorted(
        ((e.factor_node.get_label(), e.variable_node.get_name()) for e in fg.edges),
        key=lambda fv: (fv[0], fv[1]),
    )
    lines.append("Edges (variable -- factor):")
    if edges:
        wvar = max(len(v) for _, v in edges)
        for fid, v in edges:
            lines.append(f"  {v:<{wvar}} -- {fid}")
    else:
        lines.append("  (none)")
    lines.append("")

    # Detail box per factor node.
    for fid in sorted(fg.factor_nodes.keys()):
        lines.append(format_factor_box(fg.factor_nodes[fid]))
    return "\n".join(lines)


def _build_factor_cache(f: FactorNode, independencies: Independencies) -> dict:
    """
    Pre-compute and cache all interpretations and indicator vectors for
    a factor node. Called once per factor during initialization.

    Returns a dict with:
        - 'vars_list': sorted list of variable names in scope
        - 'interpretations': list of assignment dicts
        - 'N': number of interpretations
        - 'sentence_indicators': dict sid -> (type, indicators)
        - 'variable_indicators': dict var_name -> indicator array
        - 'independence_groups': list of (Aa, Ab, Ac, Ad) tuples
    """
    vars_list = sorted(f.scope)
    items_tuples = list(itertools.product([0, 1], repeat=len(vars_list)))
    interpretations = [dict(zip(vars_list, t)) for t in items_tuples]
    N = len(interpretations)

    # Cache sentence indicators
    sentence_indicators = {}
    for sid, s in f.sentences.items():
        if s.type == SentenceType.Type1:
            A = eval_indicator(s.phi_formula, interpretations)
            sentence_indicators[sid] = ('type1', s.get_lower_bound(),
                                        s.get_upper_bound(), A)
        else:
            Aqr = eval_indicator(s.phi_and_psi_formula, interpretations)
            Ar = eval_indicator(s.psi_formula, interpretations)
            sentence_indicators[sid] = ('type2', s.get_lower_bound(),
                                        s.get_upper_bound(), Aqr, Ar)

    # Cache variable indicators (for each variable in scope)
    variable_indicators = {}
    for v in vars_list:
        variable_indicators[v] = eval_indicator(
            Formula(label=v, formula=v), interpretations)

    # Cache independence constraint indicators. Each in-scope LMC assertion is
    # expanded with the correct joint encoding over all Y configurations, built
    # with the vectorized helper (numpy column masks over a precomputed truth
    # table; bit-identical to the Formula path). The consumer (_solve_local_nlp)
    # applies each group as a single quadratic equality.
    table = build_truth_table(len(vars_list))
    col_of = {v: i for i, v in enumerate(vars_list)}

    scope_set = set(f.scope)
    independence_groups = []
    for indep in independencies.get_assertions():
        all_vars = set(indep.event1) | set(indep.event2) | set(indep.event3)
        if not all_vars.issubset(scope_set):
            continue
        independence_groups.extend(lmc_constraint_groups_vec(indep, table, col_of))

    return {
        'vars_list': vars_list,
        'interpretations': interpretations,
        'N': N,
        'sentence_indicators': sentence_indicators,
        'variable_indicators': variable_indicators,
        'independence_groups': independence_groups,
    }


def _solve_local_nlp(
        n: VariableNode,
        f: FactorNode,
        neighbors: List[str],
        incoming: Dict,
        cache: dict,
        solver,
        sense: str = "min",
        debug: bool = False
) -> Tuple:
    """
    Create and solve the local non-linear program corresponding to the
    factor-to-variable message (f -> n).

    Args:
        n: VariableNode
            The target variable node in the factor graph.
        f: FactorNode
            The source factor node in the factor graph.
        neighbors: List[str]
            The list of neighboring variable node names, other than `n`.
        incoming: Dict
            The dict of incoming messages to `f`, other than the one for `n`.
        cache: dict
            Pre-computed interpretations and indicator vectors for this factor.
        solver: SolverFactory
            Reusable ipopt solver instance.
        sense: str
            The objective sense, either `min` or `max`.
        debug: bool
            A flag indicating debugging mode.

    Returns:
        A Tuple containing the objective value and a boolean flag indicating
        a feasible or an infeasible solution.
    """
    assert sense in ["min", "max"]

    N = cache['N']
    sentence_indicators = cache['sentence_indicators']
    variable_indicators = cache['variable_indicators']
    independence_groups = cache['independence_groups']

    # Create the Pyomo model and variables
    model = ConcreteModel()
    model.ITEMS = Set(initialize=range(N))
    multipliers_index = {k: v for k, v in enumerate(neighbors)}
    model.AUX = Set(initialize=multipliers_index.keys())
    model.p = Var(model.ITEMS, within=NonNegativeReals)
    model.v = Var(model.AUX, within=NonNegativeReals)
    model.constr = ConstraintList()

    # Probability distribution constraint
    model.constr.add(sum(model.p[i] for i in model.ITEMS) == 1.0)

    # Sentence constraints (using cached indicators)
    for sid, indicators in sentence_indicators.items():
        if indicators[0] == 'type1':
            _, lobo, upbo, A = indicators
            expr = dot(A, model, model.ITEMS)
            model.constr.add(expr >= lobo)
            model.constr.add(expr <= upbo)
        else:
            _, lobo, upbo, Aqr, Ar = indicators
            expr_qr = dot(Aqr, model, model.ITEMS)
            expr_r = dot(Ar, model, model.ITEMS)
            model.constr.add(expr_qr >= lobo * expr_r)
            model.constr.add(expr_qr <= upbo * expr_r)

    # Incoming variable-to-factor message constraints (with Lagrange relaxation)
    for m in neighbors:
        A = variable_indicators[m]
        expr = dot(A, model, model.ITEMS)
        msg = incoming[m]
        slack = sum(model.v[j] for j in model.AUX)
        model.constr.add(expr + slack >= msg.lower_bound)
        model.constr.add(expr - slack <= msg.upper_bound)

    # Independence constraints (using cached indicators)
    for group in independence_groups:
        if group[0] == 'conditional':
            _, Aa, Ab, Ac, Ad = group
            val1 = dot(Aa, model, model.ITEMS) * dot(Ab, model, model.ITEMS)
            val2 = dot(Ac, model, model.ITEMS) * dot(Ad, model, model.ITEMS)
            model.constr.add(val1 - val2 == 0.0)
        else:
            _, Aa, Ab, Ac = group
            val1 = dot(Aa, model, model.ITEMS)
            val2 = dot(Ab, model, model.ITEMS) * dot(Ac, model, model.ITEMS)
            model.constr.add(val1 - val2 == 0.0)

    # Objective: P(n) with penalty on slack variables
    A_obj = variable_indicators[n.name]
    penalty = 1000.0
    if sense == 'min':
        obj = dot(A_obj, model, model.ITEMS) + penalty * sum(model.v[j] for j in model.AUX)
        model.objective = Objective(expr=obj, sense=minimize)
    else:
        obj = dot(A_obj, model, model.ITEMS) - penalty * sum(model.v[j] for j in model.AUX)
        model.objective = Objective(expr=obj, sense=maximize)

    try:
        results = solver.solve(model, load_solutions=True, tee=debug)
        if (results.solver.status == SolverStatus.ok) and \
            (results.solver.termination_condition == TerminationCondition.optimal):
            objective = sum(float(A_obj[i]) * model.p[i].value for i in model.ITEMS)
            feasible = True
        elif (results.solver.termination_condition == TerminationCondition.infeasible):
            objective = sum(float(A_obj[i]) * model.p[i].value for i in model.ITEMS)
            feasible = False
        else:
            if debug:
                print(f"Solver status: {results.solver.status}")
            objective = None
            feasible = False

    except Exception as e:
        if debug:
            print(f"Exception during ipopt: {str(e)}")
        objective = None
        feasible = False

    return objective, feasible


class Message:
    """
    The messages passed along the edges of the factor graph.
    """
    def __init__(
            self,
            edge: FactorGraphEdge,
            type: str
    ):
        self.lower_bound = 0.0
        self.upper_bound = 1.0
        self.edge = edge
        self.type = type

    def set_lower_bound(self, lowbo: float):
        self.lower_bound = lowbo

    def set_upper_bound(self, upbo: float):
        self.upper_bound = upbo

    def set_bounds(self, lowbo: float, upbo: float):
        self.lower_bound = lowbo
        self.upper_bound = upbo

    def __str__(self):
        if self.type == "variable_to_factor":
            output = f"{self.edge.variable_node.get_name()}-->"
            output += f"{self.edge.factor_node.get_label()}: "
            output += f"[{self.lower_bound}, {self.upper_bound}]"
            return output
        elif self.type == "factor_to_variable":
            output = f"{self.edge.factor_node.get_label()}-->"
            output += f"{self.edge.variable_node.get_name()}: "
            output += f"[{self.lower_bound}, {self.upper_bound}]"
            return output

    def update_variable_to_factor(
            self,
            fg: FactorGraph,
            factor_messages: Dict,
    ):
        """
        Update the variable-to-factor message (variable v -> factor f).
        """
        assert self.type == "variable_to_factor"

        nid = self.edge.variable_node.get_name()
        fid = self.edge.factor_node.get_label()
        neighbors = []
        for nf in fg.variable_node_neighbors[nid]:
            if nf.label != fid:
                neighbors.append(nf)

        for nf in neighbors:
            msg = factor_messages[nf.label]
            self.lower_bound = max(self.lower_bound, msg.lower_bound)
            self.upper_bound = min(self.upper_bound, msg.upper_bound)

        # Enforce lower_bound <= upper_bound
        self.upper_bound = max(self.lower_bound, self.upper_bound)

    def update_factor_to_variable(
            self,
            fg: FactorGraph,
            variable_messages: Dict,
            cache: dict,
            solver,
            debug: bool = False
    ):
        """
        Update the factor-to-variable message (factor f -> variable n).
        """
        assert self.type == "factor_to_variable"

        nid = self.edge.variable_node.get_name()
        fid = self.edge.factor_node.get_label()
        neighbors = []
        for m in fg.factor_node_neighbors[fid]:
            if m.name != nid:
                neighbors.append(m.name)

        lower_bound, feasible_lb = _solve_local_nlp(
            self.edge.variable_node,
            self.edge.factor_node,
            neighbors,
            variable_messages,
            cache,
            solver,
            'min',
            debug
        )
        upper_bound, feasible_ub = _solve_local_nlp(
            self.edge.variable_node,
            self.edge.factor_node,
            neighbors,
            variable_messages,
            cache,
            solver,
            'max',
            debug
        )

        self.lower_bound = max(lower_bound, 0.0) if feasible_lb else self.lower_bound
        self.upper_bound = min(upper_bound, 1.0) if feasible_ub else self.upper_bound

        # Enforce lower_bound <= upper_bound
        self.upper_bound = max(self.lower_bound, self.upper_bound)


class Marginal:
    """
    Represents the marginal probability bounds of a variable.
    """

    def __init__(
            self,
            variable: VariableNode,
            lower_bound: float = 0.0,
            upper_bound: float = 1.0
    ) -> None:
        self.variable = variable
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def update(self, incoming_messages: List[Message]):
        for msg in incoming_messages:
            self.lower_bound = max(self.lower_bound, msg.lower_bound)
            self.upper_bound = min(self.upper_bound, msg.upper_bound)
        # Enforce lower_bound <= upper_bound
        self.upper_bound = max(self.lower_bound, self.upper_bound)


class ArielInference:
    """
    ARIEL Inference for LCNs. Implements the belief propagation style
    algorithm described in [Marinescu et al. Approximate Inference in LCNs. IJCAI-2023].
    """

    def __init__(
            self,
            lcn: LCN
    ):
        self.fg = None
        self.lcn = lcn
        self.variable_to_factor_messages = []
        self.factor_to_variable_messages = []
        self.incoming_to_variable = {}
        self.incoming_to_factor = {}
        self.feasible = None

    def run(
            self,
            n_iters: int = 10,
            threshold: float = 0.000001,
            debug: bool = False,
            evidence: dict = {},
            verbosity: int = 1
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Run the ARIEL inference algorithm for computing the marginals.

        Args:
            n_iters: int
                The number of iterations (default is 10).
            threshold: float
                The threshold used to decide the convergence of the algorithm.
            debug: bool
                The flag indicating debugging mode (default is False).
            evidence: dict
                The optional evidence given as input.
            verbosity: int
                The verbosity level (default 1).

        Returns:
            Dict mapping variable name to (lower_bounds, upper_bounds)
            numpy arrays.
        """

        self.evidence = evidence
        self.threshold = threshold
        t_start = time.time()

        # Get the independencies from the Local Markov Condition
        assert self.lcn.independencies is not None, "Make sure the LMC is applied."
        independencies = self.lcn.independencies

        # Create the factor graph
        assert self.fg is None
        self.fg = FactorGraph(lcn=self.lcn)
        if debug:
            print("Factor graph")
            print(format_factor_graph(self.fg))
        self.fg.add_evidence(evidence)
        if debug:
            print("Factor graph with evidence")
            print(format_factor_graph(self.fg))

        # Pre-compute caches for each factor node (once)
        factor_caches = {}
        for fid, f in self.fg.factor_nodes.items():
            factor_caches[fid] = _build_factor_cache(f, independencies)

        # Create a shared, correctly-configured ipopt solver instance
        solver = make_ipopt(debug=debug)

        # Initialize the messages
        for e in self.fg.edges:
            self.variable_to_factor_messages.append(
                Message(edge=e, type="variable_to_factor")
            )
            self.factor_to_variable_messages.append(
                Message(edge=e, type="factor_to_variable")
            )

        # Setup the incoming messages hash tables
        for nid, n in self.fg.variable_nodes.items():
            neighbors = self.fg.variable_node_neighbors[nid]
            incoming = {}
            for msg in self.factor_to_variable_messages:
                if msg.edge.factor_node in neighbors and msg.edge.variable_node == n:
                    incoming[msg.edge.factor_node.label] = msg
            self.incoming_to_variable[nid] = incoming
        for fid, f in self.fg.factor_nodes.items():
            neighbors = self.fg.factor_node_neighbors[fid]
            incoming = {}
            for msg in self.variable_to_factor_messages:
                if msg.edge.variable_node in neighbors and msg.edge.factor_node.label == fid:
                    incoming[msg.edge.variable_node.name] = msg
            self.incoming_to_factor[fid] = incoming

        if debug:
            print(self.fg)
            print(f"Initial variable_to_factor messages ({len(self.variable_to_factor_messages)}):")
            for msg in self.variable_to_factor_messages:
                print(msg)
            print(f"Initial factor_to_variable_messages ({len(self.factor_to_variable_messages)}):")
            for msg in self.factor_to_variable_messages:
                print(msg)

        # Iterative message passing
        if verbosity > 0:
            print("[ArielInference] Running marginal inference...")
        for iter in range(n_iters):
            if verbosity > 0:
                print(f"Iteration {iter} ...")
            t_iter_start = time.time()
            delta = 0.0

            # Update variable-to-factor messages (v->f)
            if verbosity > 1:
                print("### Variable to factor messages ###")
            for msg in self.variable_to_factor_messages:
                nid = msg.edge.variable_node.name
                fid = msg.edge.factor_node.label
                if debug:
                    print(f"Processing variable_to_factor message: {nid}-->{fid}: [{msg.lower_bound}, {msg.upper_bound}]")

                lobo, upbo = msg.lower_bound, msg.upper_bound
                factor_messages = self.incoming_to_variable[nid]
                msg.update_variable_to_factor(self.fg, factor_messages)
                delta += (abs(msg.lower_bound - lobo) + abs(msg.upper_bound - upbo))
                if debug:
                    print(f"Updated variable_to_factor message: {nid}-->{fid}: [{msg.lower_bound}, {msg.upper_bound}]")

            # Update factor-to-variable messages (f->v)
            if verbosity > 1:
                print("### Factor to variable messages ###")
            for msg in self.factor_to_variable_messages:
                nid = msg.edge.variable_node.name
                fid = msg.edge.factor_node.label
                if debug:
                    print(f"Processing factor_to_variable message: {fid}-->{nid}: [{msg.lower_bound}, {msg.upper_bound}]")

                lobo, upbo = msg.lower_bound, msg.upper_bound
                variable_messages = self.incoming_to_factor[fid]
                cache = factor_caches[fid]
                msg.update_factor_to_variable(
                    self.fg, variable_messages, cache, solver, debug)
                delta += (abs(msg.lower_bound - lobo) + abs(msg.upper_bound - upbo))

                if debug:
                    print(f"Updated factor_to_variable message: {fid}-->{nid}: [{msg.lower_bound:.4f}, {msg.upper_bound:.4f}]")

            # Early stopping condition
            delta /= float(2. * len(self.fg.edges))
            if verbosity > 0:
                print(f"After iteration {iter} average change in messages is {delta:.6f}")
                print(f"Elapsed time per iteration: {time.time() - t_iter_start:.4f} sec")
            if self.threshold is not None and delta <= self.threshold:
                if verbosity > 0:
                    print(f"Converged after {iter} iterations with delta={delta:.6f}")
                break

        # Collect marginals and check for bound inversions
        self.marginals = {}
        self.feasible = True
        for nid, n in self.fg.variable_nodes.items():
            marg = Marginal(n)
            factor_messages = self.incoming_to_variable[nid]
            for _, msg in factor_messages.items():
                marg.lower_bound = max(marg.lower_bound, msg.lower_bound)
                marg.upper_bound = min(marg.upper_bound, msg.upper_bound)
            # Enforce lower_bound <= upper_bound
            marg.upper_bound = max(marg.lower_bound, marg.upper_bound)

            if marg.lower_bound > marg.upper_bound:
                self.feasible = False
            self.marginals[nid] = marg

        t_end = time.time()

        # Build return dict in standard format: atom -> (lo_array, hi_array)
        results = {}
        if verbosity > 0:
            print("[ArielInference] Singleton variable marginals:")
        for nid in sorted(self.marginals):
            marg = self.marginals[nid]
            lo_1 = marg.lower_bound
            hi_1 = marg.upper_bound
            lo_arr = np.array([1.0 - hi_1, lo_1])
            hi_arr = np.array([1.0 - lo_1, hi_1])
            results[nid] = (lo_arr, hi_arr)
            if verbosity > 0:
                print(f"  P({nid}=0): [{lo_arr[0]:.6f}, {hi_arr[0]:.6f}]")
                print(f"  P({nid}=1): [{lo_arr[1]:.6f}, {hi_arr[1]:.6f}]")

        if verbosity > 0:
            print(f"[ArielInference] Feasible: {self.feasible}")
            print(f"[ArielInference] Time elapsed: {t_end - t_start:.4f} sec")

        return results

    def _message_independencies(self) -> Tuple[Independencies, Dict]:
        """
        Compute the independence assumptions that ARIEL *adds* because of the
        factor-to-variable messages.

        When ARIEL computes the message ``f -> n`` (see ``_solve_local_nlp``),
        every other boundary variable ``m`` in ``scope(f) \\ {n}`` enters the
        local NLP only through its singleton marginal interval (the incoming
        v->f message); there is no joint term coupling those variables. ARIEL
        therefore implicitly assumes the boundary variables of the factor
        (other than the target ``n``) are mutually -- i.e. pairwise, marginally
        -- independent. This method materializes those assumptions.

        Returns:
            A tuple (added, breakdown) where:
              - added: an Independencies holding the deduplicated pairwise
                marginal assertions (a |= b) over all factors/targets. Symmetric
                pairs (a |= b) == (b |= a) are deduped by IndependenceAssertion.
              - breakdown: dict fid -> {target -> [(a, b), ...]} recording which
                message introduced each pairwise assumption (informational; the
                same pair typically recurs across many messages).
        """
        assert self.fg is not None, "Run the inference first (build the factor graph)."

        added = Independencies()
        breakdown = {}
        for fid, f in self.fg.factor_nodes.items():
            scope = sorted(f.scope)
            per_target = {}
            for n in scope:
                others = [m for m in scope if m != n]
                pairs = []
                for a, b in itertools.combinations(others, 2):
                    pairs.append((a, b))
                    assertion = IndependenceAssertion(a, b)
                    if not added.contains(assertion):
                        added.add_assertions(assertion)
                if pairs:
                    per_target[n] = pairs
            if per_target:
                breakdown[fid] = per_target

        return added, breakdown

    def _contrast_with_lmc(self, added: Independencies) -> Dict:
        """
        Three-way contrast between the independencies ARIEL adds via the
        factor-to-variable messages and the Local Markov Condition (LMC)
        independencies of the original LCN.

        Args:
            added: Independencies
                The pairwise message-independencies from _message_independencies.

        Returns:
            A dict with three Independencies objects:
              - 'enforced_from_lmc': LMC assertions whose full variable set fits
                inside some single factor scope -- i.e. actually added to a local
                NLP (mirrors the all_vars.issubset(scope) gate in
                _build_factor_cache).
              - 'dropped_from_lmc': LMC assertions NOT contained in any single
                factor scope, hence never enforced anywhere.
              - 'added_not_in_lmc': message-independencies not present in the LCN's
                LMC independencies.
        """
        assert self.fg is not None, "Run the inference first (build the factor graph)."

        lmc = self.lcn.independencies
        scopes = [set(f.scope) for f in self.fg.factor_nodes.values()]

        enforced = Independencies()
        dropped = Independencies()
        for indep in lmc.get_assertions():
            if any(indep.all_vars.issubset(sc) for sc in scopes):
                enforced.add_assertions(indep)
            else:
                dropped.add_assertions(indep)

        added_not_in_lmc = Independencies()
        for indep in added.get_assertions():
            if not lmc.contains(indep):
                added_not_in_lmc.add_assertions(indep)

        return {
            "enforced_from_lmc": enforced,
            "dropped_from_lmc": dropped,
            "added_not_in_lmc": added_not_in_lmc,
        }

    def analyze(self, verbosity: int = 1) -> Dict:
        """
        Post-hoc analysis of the independence assumptions made by ARIEL,
        contrasted with the Local Markov Condition (LMC) of the original LCN.
        Call this after ``run()`` (which builds the factor graph and performs
        the message passing).

        Prints, in order:
          1. the factor graph being processed (as readable boxes + edge list);
          2. the LMC independencies of the original LCN;
          3. the independencies ARIEL adds via factor-to-variable messages;
          4. a three-way contrast (LMC enforced locally / LMC dropped / added by
             ARIEL and not in the LMC);
          5. a post-hoc summary with category counts and the final marginals.

        Args:
            verbosity: int
                0 = silent (return the structured result only); 1 = print the
                sections; 2 = also print the per-message breakdown.

        Returns:
            A dict with keys: 'added' (Independencies), 'breakdown' (dict),
            'enforced_from_lmc', 'dropped_from_lmc', 'added_not_in_lmc'
            (Independencies). Returned so the analysis is testable without
            parsing stdout.
        """
        assert self.fg is not None, \
            "No factor graph: call run() before analyze()."

        added, breakdown = self._message_independencies()
        contrast = self._contrast_with_lmc(added)
        lmc = self.lcn.independencies

        def _emit(title, indeps):
            assertions = indeps.get_assertions()
            print(f"{title} ({len(assertions)}):")
            if assertions:
                for a in assertions:
                    print(f"  {a}")
            else:
                print("  (none)")
            print("")

        if verbosity > 0:
            print(format_factor_graph(self.fg))
            print("")
            print("=" * 60)
            print("Independence analysis")
            print("=" * 60)

            _emit("Local Markov Condition (original LCN)", lmc)
            _emit("Added by ARIEL factor-to-variable messages "
                  "(pairwise among factor boundary variables)", added)

            if verbosity > 1 and breakdown:
                print("Per-message breakdown (factor -> target : assumed pairs):")
                for fid in sorted(breakdown):
                    for target in sorted(breakdown[fid]):
                        pairs = breakdown[fid][target]
                        pretty = ", ".join(f"({a} |= {b})" for a, b in pairs)
                        print(f"  {fid} -> {target} : {pretty}")
                print("")

            print("-" * 60)
            print("Three-way contrast with the LMC")
            print("-" * 60)
            _emit("[1] LMC enforced locally (scope fits in a factor)",
                  contrast["enforced_from_lmc"])
            _emit("[2] LMC dropped (scope spans factors, never enforced)",
                  contrast["dropped_from_lmc"])
            _emit("[3] Added by ARIEL, not in the LMC",
                  contrast["added_not_in_lmc"])

            print("-" * 60)
            print("Post-hoc summary")
            print("-" * 60)
            print(f"  LMC independencies               : "
                  f"{len(lmc.get_assertions())}")
            print(f"  ... enforced locally             : "
                  f"{len(contrast['enforced_from_lmc'].get_assertions())}")
            print(f"  ... dropped (multi-factor scope) : "
                  f"{len(contrast['dropped_from_lmc'].get_assertions())}")
            print(f"  Added by ARIEL messages (total)  : "
                  f"{len(added.get_assertions())}")
            print(f"  ... not in the LMC               : "
                  f"{len(contrast['added_not_in_lmc'].get_assertions())}")
            if self.marginals is not None:
                print(f"  Final marginals (feasible={self.feasible}):")
                for nid in sorted(self.marginals):
                    marg = self.marginals[nid]
                    print(f"    P({nid}=1): "
                          f"[{marg.lower_bound:.6f}, {marg.upper_bound:.6f}]")

        return {
            "added": added,
            "breakdown": breakdown,
            "enforced_from_lmc": contrast["enforced_from_lmc"],
            "dropped_from_lmc": contrast["dropped_from_lmc"],
            "added_not_in_lmc": contrast["added_not_in_lmc"],
        }


if __name__ == "__main__":

    def print_singleton_marginals(results):
        """Print only singleton variable marginals from the results."""
        print("  Singleton variable marginals:")
        for var in sorted(results):
            lo, hi = results[var]
            for val in range(len(lo)):
                print(f"    P({var}={val}): [{lo[val]:.6f}, {hi[val]:.6f}]")

    # Load the LCN. We use asia.lcn here because it has a factor with three
    # boundary variables, so the factor-to-variable messages introduce a
    # non-trivial pairwise independence assumption (see analyze() below).
    file_name = "examples/chain.lcn"
    lcn_model = LCN()
    lcn_model.from_lcn(file_name=file_name)
    print(lcn_model)

    # Check consistency
    ok = check_consistency(lcn_model)
    if ok:
        print("CONSISTENT")
    else:
        print("INCONSISTENT")

    # Run ARIEL marginal inference
    print("\n=== ArielInference (no evidence) ===")
    algo = ArielInference(lcn=lcn_model)
    results = algo.run(n_iters=10, threshold=0.000001, debug=False)
    print_singleton_marginals(results)

    # Analyze the independence assumptions made by ARIEL and contrast them
    # with the Local Markov Condition of the original LCN.
    print("\n=== Independence analysis ===")
    algo.analyze(verbosity=2)
