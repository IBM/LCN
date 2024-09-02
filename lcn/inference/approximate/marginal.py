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

# Exact and Approximate marginal inference algorithms for LCNs

import itertools
import time
from pyomo.environ import *
from typing import Dict, List, Tuple

# Local
from lcn.model import LCN, SentenceType, Formula
from lcn.inference.factor_graph import FactorGraph, FactorNode, VariableNode, FactorGraphEdge
from lcn.independencies import Independencies
from lcn.inference.utils import check_consistency, make_conjunction

infinity = float('inf')

def solve_factor_subproblem(
        n: VariableNode, 
        f: FactorNode, 
        neighbors: List[str], 
        incoming: Dict, 
        sense: str = "min", 
        debug=False
) -> Tuple:
    """
    Create and solve the non-linear program corresponding to the message
    (f->n). We use the ipopt solver and create the program on the fly.

    Args:
        n: VariableNode
            The target variable node in the factor graph.
        f: FactorNode
            The source factor node in the factor graph.
        neighbors: List[str]
            The list of neighboring variable node names, other than `n`.
        incoming: Dict
            The dict of incoming messages to `f`, othern than the one for `n`.
        sense: str
            The objective sense, either `min` or `max`.
        debug: bool
            A flag indicating debugging mode (True or False).

    Returns:
        A Tuple containig the objective value and a boolean flag idicating
        a feasible or an infeasible solution.
    """

    assert sense in ["min", "max"]

    # Get all interpretations of the factor's scope
    vars = list(f.scope)
    items = list(itertools.product([0, 1], repeat=len(vars)))
    index = {k:v for k, v in enumerate(items)}
    multipliers = {k:v for k, v in enumerate(neighbors)}
    N = len(items)

    # Create the model and variables
    model = ConcreteModel()
    model.ITEMS = Set(initialize=index.keys())
    model.AUX = Set(initialize=multipliers.keys())
    model.p = Var(model.ITEMS, within=NonNegativeReals)
    model.v = Var(model.AUX, within=NonNegativeReals)
    model.constr = ConstraintList()

    # Create the constraints for the factor's sentences
    model.constr.add(sum(model.p[i] for i in model.ITEMS) == 1.0)
    for _, s in f.sentences.items():
        if s.type == SentenceType.Type1: # P(phi)
            A = [0] * N
            lobo = s.get_lower_bound()
            upbo = s.get_upper_bound()
            for j in range(N): # loop over all interpretations
                config = dict(zip(vars, index[j]))
                A[j] = 1 if s.phi_formula.evaluate(table=config) == True else 0
            model.constr.add(sum(A[i]*model.p[i] for i in model.ITEMS) >= lobo)
            model.constr.add(sum(A[i]*model.p[i] for i in model.ITEMS) <= upbo)
        else: # Type 2 sentence P(phi | psi)
            Aqr = [0] * N
            Ar = [0] * N
            lobo = s.get_lower_bound()
            upbo = s.get_upper_bound()
            for j in range(N):
                config = dict(zip(vars, index[j]))
                Aqr[j] = 1 if s.phi_and_psi_formula.evaluate(table=config) == True else 0
                Ar[j] = 1 if s.psi_formula.evaluate(table=config) == True else 0
            val = sum(Ar[i]*model.p[i] for i in model.ITEMS)
            model.constr.add(sum(Aqr[i]*model.p[i] for i in model.ITEMS) >= lobo*val)
            model.constr.add(sum(Aqr[i]*model.p[i] for i in model.ITEMS) <= upbo*val)
    
    # Create the constraints correponding to the incoming variable_to_factor messages
    # We relax these constraints using Lagrange multipliers (model.v[j])
    for m in neighbors:
        A = [0] * N
        temp = Formula(label=m, formula=m)
        for j in range(N):
            config = dict(zip(vars, index[j]))
            A[j] = 1 if temp.evaluate(table=config) == True else 0
        msg = incoming[m] # message
        model.constr.add(sum(A[i]*model.p[i] for i in model.ITEMS) + sum(model.v[j] for j in model.AUX) >= msg.lower_bound)
        model.constr.add(sum(A[i]*model.p[i] for i in model.ITEMS) - sum(model.v[j] for j in model.AUX) <= msg.upper_bound)

    # Create the independence constraints (if any)
    for pair in itertools.combinations(neighbors, 2):
        v1 = Formula(label=pair[0], formula=pair[0])
        v2 = Formula(label=pair[1], formula=pair[1])
        v12 = Formula(label=pair[0] + pair[1], formula=f"{pair[0]} and {pair[1]}")
        A1 = [0] * N
        A2 = [0] * N
        A12 = [0] * N
        for j in range(N):
            config = dict(zip(vars, index[j]))
            A1[j] = 1 if v1.evaluate(table=config) == True else 0
            A2[j] = 1 if v2.evaluate(table=config) == True else 0
            A12[j] = 1 if v12.evaluate(table=config) == True else 0
        val1 = sum(A1[i]*model.p[i] for i in model.ITEMS)
        val2 = sum(A1[i]*model.p[i] for i in model.ITEMS)
        val12 = sum(A12[i]*model.p[i] for i in model.ITEMS)
        model.constr.add(val12 == val1 * val2)

    # Create the objective
    A = [0] * N
    for j in range(N):
        config = dict(zip(vars, index[j]))
        temp = Formula(label=n.name, formula=n.name)
        A[j] = 1 if temp.evaluate(table=config) == True else 0

    penalty = 1000.0
    if sense == 'min':
        obj = sum(A[i]*model.p[i] for i in model.ITEMS) + penalty * (sum(model.v[j] for j in model.AUX))
        model.objective = Objective(expr=obj, sense=minimize)
    else:
        obj = sum(A[i]*model.p[i] for i in model.ITEMS) - penalty * (sum(model.v[j] for j in model.AUX))
        model.objective = Objective(expr=obj, sense=maximize)

    try:
        # Solve the non-linear model
        opt = SolverFactory('ipopt')
        tee_flag = True if debug else False
        results = opt.solve(model, load_solutions=True, tee=tee_flag)
        if (results.solver.status == SolverStatus.ok) and \
            (results.solver.termination_condition == TerminationCondition.optimal):
            objective = sum(A[i]*model.p[i].value for i in model.ITEMS)
            feasible = True
        elif (results.solver.termination_condition == TerminationCondition.infeasible):
            objective = sum(A[i]*model.p[i].value for i in model.ITEMS)
            feasible = False
        else:
            print(f"Solver status: {results.solver.status}")
            objective = None
            feasible = False

        #print(f"Objective ({sense}): {objective} feasible: {feasible}")
    except Exception as e:
        print(f"Exception during ipopt: {str(e)}")
        objective = None
        feasible = False
    
    return objective, feasible

class Message:
    """
    The messages passed along the edges of the factor graph (approximate inference).
    """
    def __init__(
            self, 
            edge: FactorGraphEdge, 
            type: str
    ):
        """
        Create the initial message.

        Args:
            edge: FactorGraphEdge
                The corresponding edge.
            type: str
                The message type: variable_to_factor or factor_to_variable
        """
        self.lower_bound = 0.0
        self.upper_bound = 1.0
        self.edge = edge
        self.type = type

    def set_lower_bound(
            self, 
            lowbo: float
    ):
        """
        Set the lower probability bound.

        Args:
            lowbo: float
                The new lower probability bound.
        """
        self.lower_bound = lowbo

    def set_upper_bound(
            self, 
            upbo: float
    ):
        """
        Set the upper probability bound.

        Args:
            upbo: float
                The new upper probability bound
        """
        self.upper_bound = upbo

    def set_bounds(
            self, 
            lowbo: float, 
            upbo: float
    ):
        """
        Update the lower and upper probability bounds.

        Args:
            lowbo: float
                The new lower probability bound.
            upbo: float
                The new upper probability bound.
        """
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
        Update the variable-to-factor message (i.e., variable v -> factor f).

        Args:
            fg: FactorGraph
                The underlying factor graph.
            factor_messages: dict
                The incoming factor to variable messages.
        """
        
        assert(self.type == "variable_to_factor")

        # Get the neighboring factor nodes except for 'f'
        nid = self.edge.variable_node.get_name()

        fid = self.edge.factor_node.get_label()
        neighbors = []
        for nf in fg.variable_node_neighbors[nid]:
            if nf.label != fid:
                neighbors.append(nf)

        # Compute the min/max of the incoming messages
        for nf in neighbors:
            msg = factor_messages[nf.label]
            self.lower_bound = max(self.lower_bound, msg.lower_bound)
            self.upper_bound = min(self.upper_bound, msg.upper_bound)
    # --

    def update_factor_to_variable(
            self, 
            fg: FactorGraph, 
            variable_messages: Dict, 
            debug=False
    ):
        """
        Update the factor-to-variable message (i.e., factor f -> node n).

        Args:
            fg: FactorGraph
                The underlying factor graph.
            variable_messages: dict
                The incoming variable to factor messages.
        """

        assert(self.type == "factor_to_variable")

        # Get the neighboring variable nodes except for 'n'
        nid = self.edge.variable_node.get_name()
        fid = self.edge.factor_node.get_label()
        neighbors = [] # list of neighboring variable node names
        for m in fg.factor_node_neighbors[fid]:
            if m.name != nid:
                neighbors.append(m.name)

        # Solve the non-linear programs corresponding to (f->n)
        lower_bound, feasible_lower_bound = solve_factor_subproblem(
            self.edge.variable_node, 
            self.edge.factor_node, 
            neighbors, 
            variable_messages, 
            'min', 
            debug
        )
        upper_bound, feasible_upper_bound = solve_factor_subproblem(
            self.edge.variable_node, 
            self.edge.factor_node, 
            neighbors, 
            variable_messages, 
            'max', 
            debug
        )
        
        # assert (feasible_lower_bound is True and feasible_upper_bound is True)

        self.lower_bound = max(lower_bound, 0.0) if feasible_lower_bound else self.lower_bound
        self.upper_bound = min(upper_bound, 1.0) if feasible_upper_bound else self.upper_bound
# --        

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
        """
        Constructor of the marginal.

        Args:
            variable: VariableNode
                The variable in the LCN model.
            lower_bound: float
                The lower probability bound.
            upper_bound: float
                The upper probability bound.
        """
        self.variable = variable
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def update(
            self, 
            incoming_messages: List[Message]
    ):
        """
        Update the marginal bounds given a list of incoming messages.

        Args:
            incoming_messages: List[Message]
                A list of incoming messages.
        """

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
        """
        Constructor for the approximate inference solver.

        Args:
            lcn: LCN
                The input LCN model.
        """
        
        self.fg = None
        self.lcn = lcn
        self.variable_to_factor_messages = []
        self.factor_to_variable_messages = []
        self.incoming_to_variable = {}
        self.incoming_to_factor = {}

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

        # Create the factor graph
        assert(self.fg is None)
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
                Message(
                    edge=e, 
                    type="variable_to_factor"
                )
            )
            self.factor_to_variable_messages.append(
                Message(
                    edge=e, 
                    type="factor_to_variable"
                )
            )

        # Setup the incoming messages hash tables (dict)
        for nid, n in self.fg.variable_nodes.items():
            neighbors = self.fg.variable_node_neighbors[nid] # list of factors connected to n
            incoming = {}
            for msg in self.factor_to_variable_messages:
                if msg.edge.factor_node in neighbors and msg.edge.variable_node == n:
                    incoming[msg.edge.factor_node.label] = msg
            self.incoming_to_variable[nid] = incoming
        for fid, f in self.fg.factor_nodes.items():
            neighbors = self.fg.factor_node_neighbors[fid] # list of nodes connected to f
            incoming = {}
            for msg in self.variable_to_factor_messages:
                if msg.edge.variable_node in neighbors and msg.edge.factor_node.label == fid:
                    incoming[msg.edge.variable_node.name] = msg
            self.incoming_to_factor[fid] = incoming

        # Print if debug mode
        if debug:
            print(self.fg)
            print(f"Initial variable_to_factor messages ({len(self.variable_to_factor_messages)}):")
            for msg in self.variable_to_factor_messages:
                print(msg)
            print(f"Initial factor_to_variable_messages ({len(self.factor_to_variable_messages)}):")
            for msg in self.factor_to_variable_messages:
                print(msg)
            print("Incoming messages for variable nodes:")
            for nid, _ in self.fg.variable_nodes.items():
                incoming = self.incoming_to_variable[nid]
                print(f"  variable {nid}: {incoming}")
            print("Incoming messages for factor nodes:")
            for fid, _ in self.fg.factor_nodes.items():
                incoming = self.incoming_to_factor[fid]
                print(f"  factor {fid}: {incoming}")

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
                    output = f"Processing variable_to_factor message: {nid}-->{fid}:"
                    output += f" [{msg.lower_bound}, {msg.upper_bound}]"
                    print(output)

                lobo, upbo = msg.lower_bound, msg.upper_bound
                factor_messages = self.incoming_to_variable[nid]
                msg.update_variable_to_factor(self.fg, factor_messages)
                delta += (abs(msg.lower_bound - lobo) + abs(msg.upper_bound - upbo))    
                if debug:
                    output = f"Updated variable_to_factor message: {nid}-->{fid}:"
                    output += f" [{msg.lower_bound}, {msg.upper_bound}]"
                    print(output)
            
            # Update factor-to-variable messages
            if verbosity > 0:
                print("### Factor to variable messages ###")
            for msg in self.factor_to_variable_messages:
                nid = msg.edge.variable_node.name
                fid = msg.edge.factor_node.label
                if debug:
                    output = f"Processing factor_to_variable message: {fid}-->{nid}:"
                    output += f" [{msg.lower_bound}, {msg.upper_bound}]"
                    print(output)
                
                lobo, upbo = msg.lower_bound, msg.upper_bound
                variable_messages = self.incoming_to_factor[fid]
                msg.update_factor_to_variable(self.fg, variable_messages, debug)
                delta += (abs(msg.lower_bound - lobo) + abs(msg.upper_bound - upbo))
                if debug:
                    output = f"Updated factor_to_variable message: {fid}-->{nid}:"
                    output += f" [{msg.lower_bound}, {msg.upper_bound}]"
                    print(output)

            # Early stopping condition: check for convergence
            delta /= float(2.*len(self.fg.edges))
            if verbosity > 0:
                print(f"After iteration {iter} average change in messages is {delta}")
                print(f"Elapsed time per iteration: {time.time() - t_iter_start} sec")
            if self.threshold is not None and delta <= self.threshold:
                if verbosity > 0:
                    print(f"Converged after {iter} iterations with delta={delta}")
                break

        # Collect marginals
        self.marginals = {}
        for nid, n in self.fg.variable_nodes.items():
            marg = Marginal(n)
            factor_messages = self.incoming_to_variable[nid]
            for _, msg in factor_messages.items():
                marg.lower_bound = max(marg.lower_bound, msg.lower_bound)
                marg.upper_bound = min(marg.upper_bound, msg.upper_bound)
            self.marginals[nid] = marg
        
        t_end = time.time()
        
        if verbosity > 0:
            print(f"[ApproximateInference] Marginals:")
            for nid, _ in self.fg.variable_nodes.items():
                marg = self.marginals[nid]
                print(f"{nid}: [{marg.lower_bound}, {marg.upper_bound}]")
            print(f"[ApproximateInference] Time elapsed: {t_end - t_start} sec")

if __name__ == "__main__":

    # Load the LCN
    file_name = "/home/radu/git/fm-factual/examples/lcn/alarm.lcn"
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




