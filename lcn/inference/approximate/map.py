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

# Approximate MAP inference algorithms for LCNs

import itertools
import time
import random
import numpy as np
from pyomo.environ import *
from typing import Dict, List, Tuple
from collections import deque

# Local
from lcn.model import LCN, SentenceType, Formula, Sentence
from lcn.inference.factor_graph import FactorGraph, FactorNode, VariableNode, FactorGraphEdge
from lcn.inference.utils import check_consistency
from lcn.inference.approximate.marginal import ApproximateInference
from lcn.inference.utils import make_init_config, select_neighbor, find_neighbors

infinity = float('inf')

def build_augmented_lcn(
        lcn: LCN, 
        assignment: Dict,
        debug: bool = False
) -> LCN:
    """
    Create the augmented LCN needed to evaluate a MAP assignment. For each 
    variable Yi we add an auxilliary binary variable Wi. Then, we add the
    following constraints:
        P(W1=1|Y1=y1) = 1 and P(W1=1|Y1!=y1) = 0
        P(Wj=1|Wj-1=1, Yj=yj) = 1 and P(Wj=1|Wj-1=1, Yj!=yj) = 0
        (we need to add constraints for all value assignments to the Wj vars).

    Args:
        lcn: LCN
            The original LCN.
        assignment: Dict
            The dict containing the complete MAP assignment (including evidence).

    Returns: tuple(LCN, str)
        The new augmented LCN instance and the last auxilliary variable as query.
    """

    # Create the new LCN instance
    aug_lcn = LCN()
    for _, sen in lcn.sentences.items():
        aug_lcn.add_sentence(sen)

    # Add auxiliary variables (W1, ..., Wn)
    k = 0
    org_vars = sorted(assignment.keys())
    aux_vars = [f"w{i + 1}" for i in range(len(org_vars))]
    for i in range(len(org_vars)):
        if i == 0:
            w_i = aux_vars[i]
            y_i = org_vars[i]
            if assignment[y_i] == 1:
                k = k + 1
                aug_lcn.add_sentence(Sentence(f"c{k}", f"{w_i}", f"{y_i}", 1.0, 1.0))
                k = k + 1
                aug_lcn.add_sentence(Sentence(f"c{k}", f"!{w_i}", f"{y_i}", 0.0, 0.0))
                k = k + 1
                aug_lcn.add_sentence(Sentence(f"c{k}", f"{w_i}", f"!{y_i}", 0.0, 0.0))
                k = k + 1
                aug_lcn.add_sentence(Sentence(f"c{k}", f"!{w_i}", f"!{y_i}", 1.0, 1.0))
            else:
                k = k + 1
                aug_lcn.add_sentence(Sentence(f"c{k}", f"{w_i}", f"!{y_i}", 1.0, 1.0))
                k = k + 1
                aug_lcn.add_sentence(Sentence(f"c{k}", f"!{w_i}", f"!{y_i}", 0.0, 0.0))
                k = k + 1
                aug_lcn.add_sentence(Sentence(f"c{k}", f"{w_i}", f"{y_i}", 0.0, 0.0))
                k = k + 1
                aug_lcn.add_sentence(Sentence(f"c{k}", f"!{w_i}", f"{y_i}", 1.0, 1.0))
        else:
            w_i = aux_vars[i]
            y_i = org_vars[i]
            w_j = aux_vars[i - 1]
            if assignment[y_i] == 1:
                k = k + 1
                aug_lcn.add_sentence(Sentence(f"c{k}", f"{w_i}", f"{w_j} and {y_i}", 1.0, 1.0))
                k = k + 1
                aug_lcn.add_sentence(Sentence(f"c{k}", f"!{w_i}", f"{w_j} and {y_i}", 0.0, 0.0))
                k = k + 1
                aug_lcn.add_sentence(Sentence(f"c{k}", f"{w_i}", f"{w_j} and !{y_i}", 0.0, 0.0))
                k = k + 1
                aug_lcn.add_sentence(Sentence(f"c{k}", f"!{w_i}", f"{w_j} and !{y_i}", 1.0, 1.0))
                k = k + 1
                aug_lcn.add_sentence(Sentence(f"c{k}", f"{w_i}", f"!{w_j} and {y_i}", 0.0, 0.0))
                k = k + 1
                aug_lcn.add_sentence(Sentence(f"c{k}", f"!{w_i}", f"!{w_j} and {y_i}", 1.0, 1.0))
                k = k + 1
                aug_lcn.add_sentence(Sentence(f"c{k}", f"{w_i}", f"!{w_j} and !{y_i}", 0.0, 0.0))
                k = k + 1
                aug_lcn.add_sentence(Sentence(f"c{k}", f"!{w_i}", f"!{w_j} and !{y_i}", 1.0, 1.0))
            else:
                k = k + 1
                aug_lcn.add_sentence(Sentence(f"c{k}", f"{w_i}", f"{w_j} and !{y_i}", 1.0, 1.0))
                k = k + 1
                aug_lcn.add_sentence(Sentence(f"c{k}", f"!{w_i}", f"{w_j} and !{y_i}", 0.0, 0.0))
                k = k + 1
                aug_lcn.add_sentence(Sentence(f"c{k}", f"{w_i}", f"{w_j} and {y_i}", 0.0, 0.0))
                k = k + 1
                aug_lcn.add_sentence(Sentence(f"c{k}", f"!{w_i}", f"{w_j} and {y_i}", 1.0, 1.0))
                k = k + 1
                aug_lcn.add_sentence(Sentence(f"c{k}", f"{w_i}", f"!{w_j} and !{y_i}", 0.0, 0.0))
                k = k + 1
                aug_lcn.add_sentence(Sentence(f"c{k}", f"!{w_i}", f"!{w_j} and !{y_i}", 1.0, 1.0))
                k = k + 1
                aug_lcn.add_sentence(Sentence(f"c{k}", f"{w_i}", f"!{w_j} and {y_i}", 0.0, 0.0))
                k = k + 1
                aug_lcn.add_sentence(Sentence(f"c{k}", f"!{w_i}", f"!{w_j} and {y_i}", 1.0, 1.0))

    if debug:
        print("Augmented LCN")
        print(aug_lcn)
        print(f"Aux query var: {aux_vars[-1]}")
        print(f"Aux variables: {aux_vars}")
        print(f"Org Variables: {org_vars}")
        print(f"Assignment: {assignment}")

    return aug_lcn, aux_vars[-1]

def eval_approx_map_assignment(
        lcn: LCN,
        assignment: Dict,
        map_task: str = "maximin",
        n_iters: int = 10,
        threshold: float = 0.000001,
        debug: bool = False
) -> Tuple:
    
    """
    Approximate evaluation of a variable assignment (MAP assignment).

    Args:
        lcn: LCN
            The original LCN instance.
        assignment: Dict
            A dict containing the MAP assignment (all vars including evidence).
        map_task: str
            The maximin or maximax MAP task.
        n_iters: int
            The number of iterations for the ARIEL solver.
        threshold: float
            The threshold for the ARIEL solver.
    
    Returns: tuple(float, float)
        The score of the MAP assignment and the time elapsed in seconds.
    """
    
    t_start = time.time()

    # Create the augmented LCN (with auxilliary variables and constraints).
    l, q = build_augmented_lcn(lcn, assignment, debug)

    # Create the evaluator
    ariel = ApproximateInference(lcn=l)
    ariel.run(n_iters=n_iters, threshold=threshold, verbosity=0)
    m = ariel.marginals[q]
    elapsed = time.time() - t_start

    # Get the score and time elapsed in seconds
    score = m.lower_bound if map_task == "maximin" else m.upper_bound
    return score, elapsed

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
    
class ApproximateMAPInference:
    """
    Approximate Inference for LCNs. Implements the belief propagation style
    algorithm described in [Marinescu et al. Approximate Inference in LCNs. IJCAI-2023].
    """

    def __init__(
            self, 
            lcn: LCN,
            method: str,
            eps: float = 0.,
            max_discrepancy: int = 1,
            num_iterations: int = 10,
            max_flips: int = 10,
            init_temperature: float = 100.,
            alpha: float = .01,
            seed: int = 42,
            debug: bool = False,
            ariel_iterations: int = 10,
            ariel_threshold: float = 0.000001,
            eval_iterations: int = 10,
            eval_threshold: float = 0.000001
    ):
        """
        Constructor for the approximate inference solver.

        Args:
            lcn: LCN
                The input LCN model.
            method: str
                The MAP inference type: [maximin, maximax, interval] 
            eps: float
                Epsilon value for epsilon-coverings (default 0.0).
            max_discrepancy: int
                The maximum discrepancy level used by the LDS solver.
            num_iterations: int
                Number of SA iterations.
            max_flips: int
                Maximumm number of flips per SA iteration.
            init_temperature: float
                Initial SA temperature.
            alpha: float
                SA temperature cooling schedule.
            seed: int
                Seed for the random number generator.
            debug: bool
                Flag indicating debugging mode.
            ariel_iterations: int
                The number of iterations for the ARIEL approximation.
            ariel_threshold: float
                The convergence threshold used by the ARIEL approximation.
            eval_iterations: int
                The number of iterations used by the approximate MAP evaluation.
            eval_threshold: float
                The threshold used by the approximate MAP evaluation.
        """
        
        self.lcn = lcn
        self.method = method
        self.epsilon = eps
        self.max_discrepancy = max_discrepancy
        self.num_iterations = num_iterations
        self.max_flips = max_flips
        self.init_temperature = init_temperature
        self.alpha = alpha
        self.seed = seed
        self.ariel_iterations = ariel_iterations
        self.ariel_threshold = ariel_threshold
        self.eval_iterations = eval_iterations
        self.eval_threshold = eval_threshold
        self.debug = debug

    def run(
            self,
            algo: str,
            time_limit: int = -1,
            evidence: dict = {},
            query: list = [],
            num_query: int = 0,
            map_init: str = "default"
    ):
        """
        Run an exact MAP inference algorithm.

        Args:
            algo: str
                Name of the exact MAP inference algorithm.
            time_limit: int
                The time limit in seconds (default -1 means no limit).
            evidence: dict
                A dict representing the evidence (observed variables).
            query: list
                A list of MAP variables (MMAP task)
            num_query: int
                The number of query variables to be selected randomly (MMAP task)
            map_init: str
                The initial MAP assignment initialization (for LDS/SA algorithms)
        """

        self.algo = algo
        self.time_limit = time_limit
        self.evidence = evidence
        self.query = query
        self.num_query = num_query
        self.map_init = map_init

        if self.algo == "ariel":
            self._run_ariel()
        elif self.algo == "alds":
            self._run_lds()
        elif self.algo == "asa":
            self._run_sa()
        else:
            raise NotImplementedError(f"Approx MAP algorithm {self.algo} is not implemented.")

    def _run_lds(self):
        """
        Limited Discrepancy Search with approximate MAP evaluations.
        """

        # Safety checks
        if self.method not in ["maximin", "maximax"]:
            raise NotImplementedError(f"LDS is not implemented for intervals yet.")

        # Get all variables.
        variables = self.lcn.get_variables()
        evidence_vars = [k for k, _ in self.evidence.items()]
        if len(self.query) > 0: # MMAP query
            map_vars = self.query
        else: # MAP query
            if self.num_query > 0:
                candidates = list(set(variables) - set(evidence_vars))
                num_cand = min(self.num_query, len(candidates))
                random.Random(4).shuffle(candidates)
                map_vars = candidates[:num_cand]
                self.query = map_vars
            else:
                map_vars = list(set(variables) - set(evidence_vars))
        map_domains = [2 for _ in map_vars] # all binary variables
        map_vals = [[0, 1] for _ in map_vars] # list of values for each variable
        task = "MMAP" if len(self.query) > 0 else "MAP"
        print(f"[ALDS] Searching over MAP variables: {map_vars}")
        print(f"[ALDS] Query: {task}")
        print(f"[ALDS] MAP method: {self.method}")
        print(f"[ALDS] MAP config evaluation: approximate")
        print(f"[ALDS] MAP init config: {self.map_init}")

        # Create a random MAP assignment and evaluate it.
        if self.map_init == "default":
            init_config = [0] * len(map_vars)
        else:
            self._run_ariel()
            init_config = [self.best_solution_config[var] for var in map_vars]

        # Evaluate the initial configuration (for solution quality)
        assignment = dict(zip(map_vars, init_config))
        assignment.update(self.evidence)
        init_score, eval_time = eval_approx_map_assignment(
            lcn=self.lcn, 
            assignment=assignment, 
            map_task=self.method,
            n_iters=self.eval_iterations,
            threshold=self.eval_threshold,
        )
        print(f"[ALDS] Initial MAP config: {init_config}")
        print(f"[ALDS] Initial MAP score: {init_score}")
        print(f"[ALDS] Initial MAP time: {eval_time}")

        # Start the timer
        start_time = time.time()

        # Initialize the search space
        stack = deque()
        root = (-1, [], self.max_discrepancy)
        stack.append(root)
        best_score = -np.infty
        best_config = None
        timeout = False

        def next_node():
            n = None
            if len(stack) > 0:
                n = stack.pop()
            return n
        #--
        def expand_node(n):
            i, a, k = n[0], n[1], n[2]
            if i >= len(map_vars) - 1:
                return True # leaf node (full configuration)
            else:
                d = map_domains[i + 1]
                for val in range(d):
                    if val != init_config[i + 1]:
                        ch = (i + 1, a + [val], k-1)
                    else:
                        ch = (i + 1, a + [val], k)
                    if ch[2] >= 0: # check if discrepancy ok
                        stack.append(ch)
                return False
        #--

        # Limited discrepancy search
        n = next_node()
        while n:
            if (expand_node(n)):
                interpretation = [map_vals[x][y] for x,y in enumerate(n[1])]
                assignment = dict(zip(map_vars, interpretation))
                assignment.update(self.evidence)
                score, eval_time = eval_approx_map_assignment(
                    lcn=self.lcn, 
                    assignment=assignment, 
                    map_task=self.method,
                    n_iters=self.eval_iterations,
                    threshold=self.eval_threshold,
                )

                print(f" interpretation: {interpretation} score: {score} time: {eval_time}")

                # Check for better solution
                if self.method == "maximin":
                    current_score = score
                    if current_score > best_score:
                        best_score = current_score
                        best_config = interpretation
                elif self.method == "maximax":
                    current_score = score
                    if current_score > best_score:
                        best_score = current_score
                        best_config = interpretation
                
                # Check for timeout
                if self.time_limit > 0 and time.time() - start_time >= self.time_limit:
                    print(f"[LDS] Search interrupted due to TIMEOUT.")
                    timeout = True
                    break
                
            n = next_node()

        # Stop timer and report solution
        elapsed = time.time() - start_time
        if not timeout:
            print(f"[ALDS] Search terminated successfully.")
        print(f"[ALDS] Time elapsed (sec): {elapsed}")
        print(f"[ALDS] Search timeout: {timeout}")
        if self.method == "maximin":
            solution = dict(zip(map_vars, best_config))
            solution.update(self.evidence)
            print(f"[ALDS] MAXIMIN-MAP score: {best_score}")
            print(f"[ALDS] MAXIMIN-MAP solution: {solution}")
            self.best_solution_value = best_score
            self.best_solution_config = solution
        elif self.method == "maximax":
            solution = dict(zip(map_vars, best_config))
            solution.update(self.evidence)
            print(f"[ALDS] MAXIMAX-MAP score: {best_score}")
            print(f"[ALDS] MAXIMAX-MAP solution: {solution}")
            self.best_solution_value = best_score
            self.best_solution_config = solution

    def _run_sa(self):
        """
        Simulated Annealing over the MAP variables. Each MAP assignment together
        with the evidence is evaluated using the approximate ARIEL scheme that is
        modified to compute the maxmin/maxmax value of the assignment.
        """

        # Safety checks
        if self.method not in ["maximin", "maximax"]:
            raise NotImplementedError(f"SA is not implemented for intervals yet.")

        # Get all variables.
        variables = self.lcn.get_variables()
        evidence_vars = [k for k, _ in self.evidence.items()]
        if len(self.query) > 0: # MMAP query
            map_vars = self.query
        else: # MAP query
            if self.num_query > 0:
                candidates = list(set(variables) - set(evidence_vars))
                num_cand = min(self.num_query, len(candidates))
                random.Random(4).shuffle(candidates)
                map_vars = candidates[:num_cand]
                self.query = map_vars
            else:
                map_vars = list(set(variables) - set(evidence_vars))
        task = "MMAP" if len(self.query) > 0 else "MAP"
        print(f"[ASA] Searching over MAP variables: {map_vars}")
        print(f"[ASA] Query: {task}")
        print(f"[ASA] MAP method: {self.method}")
        print(f"[ASA] Number of iterations: {self.num_iterations}")
        print(f"[ASA] Max flips per iteration: {self.max_flips}")
        print(f"[ASA] Initial temperature: {self.init_temperature}")
        print(f"[ASA] Cooling schedule: {self.alpha}")
        print(f"[ASA] MAP config evaluation: approximate")
        print(f"[ASA] MAP init config: {self.map_init}")

        # Initialize the cache and start the timer
        cache = {}
        np.random.seed(self.seed)
        best_score = -np.infty
        best_config = None
        num_flips = 0
        timeout = False
        
        # Create a random MAP assignment and evaluate it.
        if self.map_init == "default":
            best_config = make_init_config(map_vars)
        else:
            self._run_ariel()
            best_config = [self.best_solution_config[var] for var in map_vars]

        # Start the timer
        start_time = time.time()
        assignment = dict(zip(map_vars, best_config))
        assignment.update(self.evidence)
        best_score, eval_time = eval_approx_map_assignment(
            lcn=self.lcn, 
            assignment=assignment, 
            map_task=self.method,
            n_iters=self.eval_iterations,
            threshold=self.eval_threshold,
        )
        print(f"[ASA] Initial MAP config: {best_config}")
        print(f"[ASA] Initial MAP score: {best_score}")
        print(f"[ASA] Initial MAP time: {eval_time}")

        # Local search for a number of iterations
        for iter in range(self.num_iterations):
            T = self.init_temperature
            print(f"Iteration #{iter}")
            print(f"  - initial temperature: {T}")
            current_config = best_config
            current_score = best_score

            # Perform a maximumm number of flips per iteration
            for _ in range(self.max_flips):
                num_flips += 1

                # Select a random neighbor of the current configuration
                neighbors = find_neighbors(current_config)
                next_config = select_neighbor(neighbors)

                # Check for a cached score value
                key = tuple(next_config)
                if key in cache:
                    next_score = cache[key]
                else:
                    assignment = dict(zip(map_vars, next_config))
                    assignment.update(self.evidence)
                    next_score, eval_time = eval_approx_map_assignment(
                        lcn=self.lcn, 
                        assignment=assignment, 
                        map_task=self.method,
                        n_iters=self.eval_iterations,
                        threshold=self.eval_threshold,
                    )
                    cache[key] = next_score

                delta = np.log(next_score) - np.log(current_score)
                if delta > 0: # accept next config
                    current_score = next_score
                    current_config = next_config
                else:
                    p = np.random.random() # sample the uniform distribution
                    threshold = np.exp(delta/T)
                    if p < threshold: # move to worse config
                        current_score = next_score
                        current_config = next_config

                # Check for better solution
                if current_score > best_score:
                    best_score = current_score
                    best_config = current_config

                    print(f"  - found better interpretation: {best_config} [{best_score} {np.log(best_score)} {eval_time}]")
                
                # Adjust temperature
                T *= self.alpha

                # Check for timeout (during flips)
                elapsed = time.time()
                if self.time_limit > 0 and elapsed > self.time_limit:
                    print(f"[SA] Search terminated due to TIMEOUT.")
                    timeout = True
                    break
            
            # Check for timeout (during iteration)
            if timeout:
                break
        
        # Stop timer and report solution
        elapsed = time.time() - start_time
        if not timeout:
            print(f"[ASA] Search terminated successfully.")
        print(f"[ASA] Time elapsed (sec): {elapsed}")
        print(f"[ASA] Search timeout: {timeout}")
        if self.method == "maximin":
            solution = dict(zip(map_vars, best_config))
            solution.update(self.evidence)
            print(f"[ASA] MAXIMIN-MAP score: {best_score}")
            print(f"[ASA] MAXIMIN-MAP solution: {solution}")
            self.best_solution_value = best_score
            self.best_solution_config = solution
        elif self.method == "maximax":
            solution = dict(zip(map_vars, best_config))
            solution.update(self.evidence)
            print(f"[ASA] MAXIMAX-MAP score: {best_score}")
            print(f"[ASA] MAXIMAX-MAP solution: {solution}")
            self.best_solution_value = best_score
            self.best_solution_config = solution
    
    def _run_ariel(self):
        """
        Run the approximate inference algorithm for computing the marginals.
        """

        # Initialize
        self.fg = None
        self.variable_to_factor_messages = []
        self.factor_to_variable_messages = []
        self.incoming_to_variable = {}
        self.incoming_to_factor = {}

        # Start the timer
        t_start = time.time()

        # Create the factor graph
        assert(self.fg is None)
        self.fg = FactorGraph(lcn=self.lcn)
        if self.debug:
            print("Initial factor graph")
            print(self.fg)
        self.fg.add_evidence(self.evidence)
        if self.debug:
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

        # Get all variables.
        variables = self.lcn.get_variables()
        evidence_vars = [k for k, _ in self.evidence.items()]
        if len(self.query) > 0: # MMAP query
            map_vars = self.query
        else: # MAP query
            if self.num_query > 0:
                candidates = list(set(variables) - set(evidence_vars))
                num_cand = min(self.num_query, len(candidates))
                random.Random(4).shuffle(candidates)
                map_vars = candidates[:num_cand]
                self.query = map_vars
            else:
                map_vars = list(set(variables) - set(evidence_vars))
        task = "MMAP" if len(self.query) > 0 else "MAP"

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
        if self.debug:
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
        print(f"[ARIEL] Running marginal inference...")
        print(f"[ARIEL] MAP variables: {map_vars}")
        print(f"[ARIEL] Query: {task}")
        for iter in range(self.ariel_iterations):
            print(f"Iteration {iter} ...")
            t_iter_start = time.time()
            delta = 0.0

            # Update variable-to-factor messages (v->f)
            print("### Variable to factor messages ###")
            for msg in self.variable_to_factor_messages:
                nid = msg.edge.variable_node.name
                fid = msg.edge.factor_node.label
                if self.debug:
                    output = f"Processing variable_to_factor message: {nid}-->{fid}:"
                    output += f" [{msg.lower_bound}, {msg.upper_bound}]"
                    print(output)

                lobo, upbo = msg.lower_bound, msg.upper_bound
                factor_messages = self.incoming_to_variable[nid]
                msg.update_variable_to_factor(self.fg, factor_messages)
                delta += (abs(msg.lower_bound - lobo) + abs(msg.upper_bound - upbo))    
                if self.debug:
                    output = f"Updated variable_to_factor message: {nid}-->{fid}:"
                    output += f" [{msg.lower_bound}, {msg.upper_bound}]"
                    print(output)
            
            # Update factor-to-variable messages
            print("### Factor to variable messages ###")
            for msg in self.factor_to_variable_messages:
                nid = msg.edge.variable_node.name
                fid = msg.edge.factor_node.label
                if self.debug:
                    output = f"Processing factor_to_variable message: {fid}-->{nid}:"
                    output += f" [{msg.lower_bound}, {msg.upper_bound}]"
                    print(output)
                
                lobo, upbo = msg.lower_bound, msg.upper_bound
                variable_messages = self.incoming_to_factor[fid]
                msg.update_factor_to_variable(self.fg, variable_messages, self.debug)
                delta += (abs(msg.lower_bound - lobo) + abs(msg.upper_bound - upbo))
                if self.debug:
                    output = f"Updated factor_to_variable message: {fid}-->{nid}:"
                    output += f" [{msg.lower_bound}, {msg.upper_bound}]"
                    print(output)

            # Early stopping condition: check for convergence
            delta /= float(2.*len(self.fg.edges))
            print(f"[ARIEL] After iteration {iter} average change in messages is {delta}")
            print(f"[ARIEL] Elapsed time per iteration: {time.time() - t_iter_start} sec")
            if self.ariel_threshold is not None and delta <= self.ariel_threshold:
                print(f"[ARIEL] Converged after {iter} iterations with delta={delta}")
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
        
        print(f"[ARIEL] Marginals:")
        for nid, _ in self.fg.variable_nodes.items():
            marg = self.marginals[nid]
            print(f"{nid}: [{marg.lower_bound}, {marg.upper_bound}]")
        print(f"[ARIEL] Time elapsed (sec): {t_end - t_start}")

        # Get the MAP config (consistent with evidence)
        self.best_solution_value = 1.
        self.best_solution_config = {}
        if self.method == "maximin":
            # look at the lower bounds
            for nid, _ in self.fg.variable_nodes.items():
                if nid in self.evidence:
                    continue # skip evidence for now
                if len(self.query) > 0 and nid not in self.query:
                    continue # skip non-MAP vars if MMAP query
                marg = self.marginals[nid]
                lb = marg.lower_bound
                if lb > (1. - lb):
                    self.best_solution_config[nid] = 1
                    self.best_solution_value *= lb
                else:
                    self.best_solution_config[nid] = 0
                    self.best_solution_value *= (1. - lb)
            self.best_solution_config.update(self.evidence)
            print(f"[ARIEL] MAXIMIN-MAP score: {self.best_solution_value}")
            print(f"[ARIEL] MAXIMIN-MAP solution: {self.best_solution_config}")
        elif self.method == "maximax":
            # look at the upper bounds
            for nid, _ in self.fg.variable_nodes.items():
                if nid in self.evidence:
                    continue # skip evidence for now
                if len(self.query) > 0 and nid not in self.query:
                    continue # skip non-MAP vars if MMAP query
                marg = self.marginals[nid]
                ub = marg.upper_bound
                if ub > (1. - ub):
                    self.best_solution_config[nid] = 1
                    self.best_solution_value *= ub
                else:
                    self.best_solution_config[nid] = 0
                    self.best_solution_value *= (1. - ub)
            self.best_solution_config.update(self.evidence)
            print(f"[ARIEL] MAXIMAX-MAP score: {self.best_solution_value}")
            print(f"[ARIEL] MAXIMAX-MAP solution: {self.best_solution_config}")
        else:
            raise NotImplementedError(f"MAP method {self.method} is not implemented.")


if __name__ == "__main__":

    # Load the LCN
    file_name = "/home/radu/git/fm-factual/examples/lcn/factual.lcn"
    l = LCN()
    l.from_lcn(file_name=file_name)
    print(l)

    # Check consistency
    ok = check_consistency(l)
    if ok:
        print("CONSISTENT")
    else:
        print("INCONSISTENT")

    evidence = {} #{'S': 0, 'X': 1}
    query = ['A1', 'A2']
    # Run approximate marginal inference
    algo = ApproximateMAPInference(
        lcn=l, 
        method="maximax", 
        eps=0., 
        max_discrepancy=3, 
        ariel_threshold=0.000001, 
        ariel_iterations=5, 
        num_iterations=5, 
        max_flips=10,
        eval_iterations=5,
        eval_threshold=0.00001,
    )
    algo.run(algo="ariel", query=query, evidence=evidence)


