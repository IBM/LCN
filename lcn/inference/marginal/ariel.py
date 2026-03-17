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

# Approximate marginal inference for LCNs (improved version)

import itertools
import time
import numpy as np
from pyomo.environ import *
from typing import Dict, List, Tuple

# Local
from lcn.model import LCN, SentenceType, Formula, Sentence
from lcn.independencies import Independencies
from lcn.inference.factor_graph import FactorGraph, FactorNode, VariableNode, FactorGraphEdge
from lcn.inference.utils import check_consistency, make_conjunction
from lcn.inference.marginal.exact import _eval_indicator, _dot


def _solve_local_nlp(
        n: VariableNode,
        f: FactorNode,
        neighbors: List[str],
        incoming: Dict,
        independencies: Independencies,
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
        independencies: Independencies
            The independence assertions from the LCN's Local Markov Condition.
        sense: str
            The objective sense, either `min` or `max`.
        debug: bool
            A flag indicating debugging mode.

    Returns:
        A Tuple containing the objective value and a boolean flag indicating
        a feasible or an infeasible solution.
    """

    assert sense in ["min", "max"]

    # Precompute interpretations over the factor's scope
    vars_list = sorted(f.scope)
    items_tuples = list(itertools.product([0, 1], repeat=len(vars_list)))
    interpretations = [dict(zip(vars_list, t)) for t in items_tuples]
    N = len(interpretations)

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

    # Sentence constraints
    for _, s in f.sentences.items():
        lobo = s.get_lower_bound()
        upbo = s.get_upper_bound()
        if s.type == SentenceType.Type1:  # P(q)
            A = _eval_indicator(s.phi_formula, interpretations)
            expr = _dot(A, model, model.ITEMS)
            model.constr.add(expr >= lobo)
            model.constr.add(expr <= upbo)
        else:  # P(q|r)
            Aqr = _eval_indicator(s.phi_and_psi_formula, interpretations)
            Ar = _eval_indicator(s.psi_formula, interpretations)
            expr_qr = _dot(Aqr, model, model.ITEMS)
            expr_r = _dot(Ar, model, model.ITEMS)
            model.constr.add(expr_qr >= lobo * expr_r)
            model.constr.add(expr_qr <= upbo * expr_r)

    # Incoming variable-to-factor message constraints (with Lagrange relaxation)
    for m in neighbors:
        A = _eval_indicator(Formula(label=m, formula=m), interpretations)
        expr = _dot(A, model, model.ITEMS)
        msg = incoming[m]
        slack = sum(model.v[j] for j in model.AUX)
        model.constr.add(expr + slack >= msg.lower_bound)
        model.constr.add(expr - slack <= msg.upper_bound)

    # Independence constraints from the LCN's Local Markov Condition
    scope_set = set(f.scope)
    for indep in independencies.get_assertions():
        X, T, S = list(indep.event1), list(indep.event2), list(indep.event3)
        # Only add constraints if all involved variables are in this factor's scope
        all_vars = set(X) | set(T) | set(S)
        if not all_vars.issubset(scope_set):
            continue

        configs_S = [()] if len(S) == 0 else list(itertools.product([0, 1], repeat=len(S)))
        if len(S) > 0:
            for t in T:
                x = X[0]
                literals = {x: 1, t: 1}
                for s in configs_S:
                    literals.update(dict(zip(S, list(s))))
                    Fa = make_conjunction(variables=X + S + [t], literals=literals)
                    Fb = make_conjunction(variables=S, literals=literals)
                    Fc = make_conjunction(variables=X + S, literals=literals)
                    Fd = make_conjunction(variables=S + [t], literals=literals)
                    Aa = _eval_indicator(Fa, interpretations)
                    Ab = _eval_indicator(Fb, interpretations)
                    Ac = _eval_indicator(Fc, interpretations)
                    Ad = _eval_indicator(Fd, interpretations)
                    val1 = _dot(Aa, model, model.ITEMS) * _dot(Ab, model, model.ITEMS)
                    val2 = _dot(Ac, model, model.ITEMS) * _dot(Ad, model, model.ITEMS)
                    model.constr.add(val1 - val2 == 0.0)
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
                val1 = _dot(Aa, model, model.ITEMS)
                val2 = _dot(Ab, model, model.ITEMS) * _dot(Ac, model, model.ITEMS)
                model.constr.add(val1 - val2 == 0.0)

    # Objective: P(n) with penalty on slack variables
    A_obj = _eval_indicator(Formula(label=n.name, formula=n.name), interpretations)
    penalty = 1000.0
    if sense == 'min':
        obj = _dot(A_obj, model, model.ITEMS) + penalty * sum(model.v[j] for j in model.AUX)
        model.objective = Objective(expr=obj, sense=minimize)
    else:
        obj = _dot(A_obj, model, model.ITEMS) - penalty * sum(model.v[j] for j in model.AUX)
        model.objective = Objective(expr=obj, sense=maximize)

    try:
        opt = SolverFactory('ipopt')
        tee_flag = True if debug else False
        results = opt.solve(model, load_solutions=True, tee=tee_flag)
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

    def update_factor_to_variable(
            self,
            fg: FactorGraph,
            variable_messages: Dict,
            independencies: Independencies,
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
            independencies,
            'min',
            debug
        )
        upper_bound, feasible_ub = _solve_local_nlp(
            self.edge.variable_node,
            self.edge.factor_node,
            neighbors,
            variable_messages,
            independencies,
            'max',
            debug
        )

        self.lower_bound = max(lower_bound, 0.0) if feasible_lb else self.lower_bound
        self.upper_bound = min(upper_bound, 1.0) if feasible_ub else self.upper_bound


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


class ApproximateInference:
    """
    Approximate Inference for LCNs. Implements the belief propagation style
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
    ):
        """
        Run the approximate inference algorithm for computing the marginals.

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
            print(self.fg)
        self.fg.add_evidence(evidence)
        if debug:
            print("Factor graph with evidence")
            print(self.fg)

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
            print(f"[ApproximateInference] Running marginal inference...")
        for iter in range(n_iters):
            if verbosity > 0:
                print(f"Iteration {iter} ...")
            t_iter_start = time.time()
            delta = 0.0

            # Update variable-to-factor messages (v->f)
            if verbosity > 0:
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
            if verbosity > 0:
                print("### Factor to variable messages ###")
            for msg in self.factor_to_variable_messages:
                nid = msg.edge.variable_node.name
                fid = msg.edge.factor_node.label
                if debug:
                    print(f"Processing factor_to_variable message: {fid}-->{nid}: [{msg.lower_bound}, {msg.upper_bound}]")

                lobo, upbo = msg.lower_bound, msg.upper_bound
                variable_messages = self.incoming_to_factor[fid]
                msg.update_factor_to_variable(self.fg, variable_messages, independencies, debug)
                delta += (abs(msg.lower_bound - lobo) + abs(msg.upper_bound - upbo))

                # Check for bound inversion in messages
                if msg.lower_bound > msg.upper_bound and verbosity > 0:
                    print(f"WARNING: message {fid}-->{nid} has lb={msg.lower_bound:.4f} > ub={msg.upper_bound:.4f}")

                if debug:
                    print(f"Updated factor_to_variable message: {fid}-->{nid}: [{msg.lower_bound:.4f}, {msg.upper_bound:.4f}]")

            # Early stopping condition
            delta /= float(2. * len(self.fg.edges))
            if verbosity > 0:
                print(f"After iteration {iter} average change in messages is {delta}")
                print(f"Elapsed time per iteration: {time.time() - t_iter_start} sec")
            if self.threshold is not None and delta <= self.threshold:
                if verbosity > 0:
                    print(f"Converged after {iter} iterations with delta={delta}")
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
            self.marginals[nid] = marg

            if marg.lower_bound > marg.upper_bound:
                self.feasible = False
                if verbosity > 0:
                    print(f"WARNING: variable {nid} has lb={marg.lower_bound:.4f} > ub={marg.upper_bound:.4f} (infeasible)")

        t_end = time.time()

        if verbosity > 0:
            print(f"[ApproximateInference] Marginals:")
            for nid, _ in self.fg.variable_nodes.items():
                marg = self.marginals[nid]
                print(f"{nid}: [{marg.lower_bound:.4f}, {marg.upper_bound:.4f}]")
            print(f"[ApproximateInference] Feasible: {self.feasible}")
            print(f"[ApproximateInference] Time elapsed: {t_end - t_start} sec")


if __name__ == "__main__":

    # Load the LCN
    file_name = "examples/alarm.lcn"
    l = LCN()
    l.from_lcn(file_name=file_name)
    print(l)

    # Check consistency
    ok = check_consistency(l)
    if ok:
        print("CONSISTENT")
    else:
        print("INCONSISTENT")

    # Run approximate marginal inference
    algo = ApproximateInference(lcn=l)
    algo.run(n_iters=10, threshold=0.000001, debug=False)
