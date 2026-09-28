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

# Exact and Approximate MAP inference algorithms for LCNs

import time
import random
import itertools
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
    value,
)

from typing import Dict, List, Tuple
from collections import deque

# Local
from lcn.core.model import LCN, SentenceType, Formula
from lcn.core.independencies import Independencies
from lcn.inference.utils.common import eval_indicator, dot
from lcn.inference.utils.common import (
    make_conjunction, check_consistency, make_ipopt,
    build_truth_table, lmc_constraint_groups_vec,
    find_feasible_points, optimize_marginal_slsqp,
)
from lcn.inference.utils.common import make_init_config, select_neighbor, find_neighbors

# -----------------------------------------------------------------------
# Query-bound scorer used by the exact MAP search (evaluate_config below).
# Computes exact lower/upper bounds on P(query | evidence) for a single query
# formula via the full joint NLP; the SLSQP two-phase fallback backstops ipopt
# on the dense, nonconvex joint-LMC system.
# -----------------------------------------------------------------------

def solve_exact_model(
        lcn: LCN,
        query_formula: str,
        independencies: Independencies,
        evidence: dict = {},
        sense: str = 'min',
        debug: bool = False,
        verbosity: int = 1,
        max_iter: int = 10000,
        max_cpu_time: int = 7200,
        acceptable_tol: float = None,
        hessian_approximation: str = None,
        use_slsqp_fallback: bool = True
) -> Tuple:
    """
    Compute exact lower/upper bounds on the probability of the query formula
    by solving the corresponding non-linear constraint program (for the input
    LCN and independencies given by the Local Markov Condition).

    Args:
        lcn: LCN
            The input LCN model.
        query_formula: str
            A string representing the query formula.
        independencies: Independencies
            The independencies given by the Local Markov Condition.
        evidence: dict
            A dict containing the observed evidence variables.
        sense: str
            The sense of the optimization problem. It is either `min` or `max`.
        debug: bool
            A flag indicating the debugging mode.
        verbosity: int
            Verbosity level (0 is silent).
        max_iter: int
            Maximum number of iterations used by the ipopt solver (default 10000).
        max_cpu_time: int
            Maximum CPU time in seconds used by the ipopt solver (default 7200 sec).
        acceptable_tol: float
            Acceptable tolerance value used by the ipopt solver (default 0.00001).
        hessian_approximation: str
            The Hessian approximation used by the ipopt solver (default 'limited-memory').
        use_slsqp_fallback: bool
            When True (default) a two-phase SLSQP fallback backstops ipopt on
            suspicious/failed solves (no-evidence case only), giving reliable
            bounds on the dense nonconvex joint-LMC system. Set False to use
            ipopt alone -- faster, useful for benchmarking/ablation, but the
            bound may be missing or looser on harder instances.

    Returns:
        A tuple representing the objective value and a flag indicating its optimality.
    """

    # Step 1: Precompute interpretation table and indicator vectors
    vars_list = [k for k, _ in lcn.atoms.items()]
    items_tuples = list(itertools.product([0, 1], repeat=len(vars_list)))
    interpretations = [dict(zip(vars_list, t)) for t in items_tuples]
    N = len(interpretations)

    # Create the Pyomo model and variables
    model = ConcreteModel()
    model.ITEMS = Set(initialize=range(N))
    model.p = Var(model.ITEMS, within=NonNegativeReals, bounds=(0.0, 1.0))
    model.constr = ConstraintList()

    # Constraint residual callables (over a numpy solution vector), mirroring
    # the Pyomo constraints, for the SLSQP robustness fallback below.
    checks = [("eq", lambda p: float(p.sum()) - 1.0)]

    # Probability distribution constraint
    model.constr.add(sum(model.p[i] for i in model.ITEMS) == 1.0)

    # Step 2: Build sentence constraints using precomputed indicators
    for sid, s in lcn.sentences.items():
        lobo = s.get_lower_bound()
        upbo = s.get_upper_bound()
        if s.type == SentenceType.Type1:  # P(q)
            A = eval_indicator(s.phi_formula, interpretations)
            expr = dot(A, model, model.ITEMS)
            model.constr.add(expr >= lobo)
            model.constr.add(expr <= upbo)
            checks.append(("ineq", lambda p, A=A, lobo=lobo: float(A @ p) - lobo))
            checks.append(("ineq", lambda p, A=A, upbo=upbo: upbo - float(A @ p)))
        else:  # P(q|r)
            Aqr = eval_indicator(s.phi_and_psi_formula, interpretations)
            Ar = eval_indicator(s.psi_formula, interpretations)
            expr_qr = dot(Aqr, model, model.ITEMS)
            expr_r = dot(Ar, model, model.ITEMS)
            model.constr.add(expr_qr >= lobo * expr_r)
            model.constr.add(expr_qr <= upbo * expr_r)
            checks.append(("ineq", lambda p, Aqr=Aqr, Ar=Ar, lobo=lobo: float(Aqr @ p) - lobo * float(Ar @ p)))
            checks.append(("ineq", lambda p, Aqr=Aqr, Ar=Ar, upbo=upbo: upbo * float(Ar @ p) - float(Aqr @ p)))

    # Step 3: Build independence constraints. Joint encoding over all Y
    # configurations, built with the vectorized helper (see
    # lmc_constraint_groups_vec).
    table = build_truth_table(len(vars_list))
    col_of = {v: i for i, v in enumerate(vars_list)}

    for indep in independencies.get_assertions():
        if verbosity > 0:
            print(f"adding constraints for independence: {indep}")
        for group in lmc_constraint_groups_vec(indep, table, col_of):
            if group[0] == 'conditional':
                _, Aa, Ab, Ac, Ad = group
                val1 = dot(Aa, model, model.ITEMS) * dot(Ab, model, model.ITEMS)
                val2 = dot(Ac, model, model.ITEMS) * dot(Ad, model, model.ITEMS)
                model.constr.add(val1 - val2 == 0.0)
                checks.append(("eq", lambda p, Aa=Aa, Ab=Ab, Ac=Ac, Ad=Ad:
                               float(Aa @ p) * float(Ab @ p) - float(Ac @ p) * float(Ad @ p)))
            else:
                _, Aa, Ab, Ac = group
                val1 = dot(Aa, model, model.ITEMS)
                val2 = dot(Ab, model, model.ITEMS) * dot(Ac, model, model.ITEMS)
                model.constr.add(val1 - val2 == 0.0)
                checks.append(("eq", lambda p, Aa=Aa, Ab=Ab, Ac=Ac:
                               float(Aa @ p) - float(Ab @ p) * float(Ac @ p)))

    # Step 4: Build objective (with evidence bug fix)
    obj_formula = Formula(label="obj", formula=query_formula)
    A_query = eval_indicator(obj_formula, interpretations)

    if len(evidence) == 0:
        obj = dot(A_query, model, model.ITEMS)
    else:
        ev = [k for k, _ in evidence.items()]
        Fe = make_conjunction(variables=ev, literals=evidence)
        E = eval_indicator(Fe, interpretations)
        AE = A_query * E  # element-wise numpy multiply (fixes == vs = bug)
        obj = dot(AE, model, model.ITEMS) / dot(E, model, model.ITEMS)

    if sense == 'min':
        model.objective = Objective(expr=obj, sense=minimize)
    else:
        model.objective = Objective(expr=obj, sense=maximize)

    # Solve the non-linear model with the shared, correctly-configured ipopt
    # setup; the explicit keyword arguments override the defaults for callers
    # that need to.
    try:
        opt = make_ipopt(debug=debug)
        opt.options['max_iter'] = max_iter
        opt.options['max_cpu_time'] = max_cpu_time
        if acceptable_tol is not None:
            opt.options['acceptable_tol'] = acceptable_tol
        if hessian_approximation is not None:
            opt.options['hessian_approximation'] = hessian_approximation
        tee_flag = True if debug else False
        results = opt.solve(model, tee=tee_flag)
        if (results.solver.status == SolverStatus.ok) and \
            (results.solver.termination_condition == TerminationCondition.optimal):
            if verbosity > 0:
                print(f"Solver status: {results.solver.status}")
            objective_value = value(model.objective)
            objective_optimal = True
        elif (results.solver.termination_condition == TerminationCondition.infeasible):
            if verbosity > 0:
                print(f"Solver status: {results.solver.status}")
            objective_value = value(model.objective)
            objective_optimal = False
        else:
            if verbosity > 0:
                print(f"Solver status: {results.solver.status}")
            objective_value = value(model.objective)
            objective_optimal = False

    except Exception as e:
        if verbosity > 0:
            print(f"Exception during ipopt: {str(e)}")
        objective_value = None
        objective_optimal = False

    # Two-phase SLSQP fallback: a single ipopt solve is unreliable on the dense
    # joint-LMC system and frequently returns None even when the bound exists.
    # For the no-evidence case the objective P(query) is linear in p, so we can
    # find feasible seeds (find_feasible_points) and optimize from them
    # (optimize_marginal_slsqp). The evidence case has a non-linear ratio
    # objective and stays ipopt-only.
    def _is_suspicious(v, opt):
        if v is None:
            return True
        if sense == 'max' and v >= 1.0 - 1e-6:
            return True
        if sense == 'min' and v <= 1e-6:
            return True
        return False

    if (use_slsqp_fallback and len(evidence) == 0
            and _is_suspicious(objective_value, objective_optimal)):
        seeds = find_feasible_points(N, checks, n_points=8, restarts=80)
        v, ok = optimize_marginal_slsqp(N, A_query, checks, sense, seeds) \
            if seeds else (None, False)
        if ok and v is not None:
            if objective_value is None or not objective_optimal:
                objective_value, objective_optimal = v, True
            elif sense == 'max':
                objective_value = max(objective_value, v)
            else:
                objective_value = min(objective_value, v)

    if verbosity > 0:
        print(f"[Ipopt] objective={objective_value}, optimal={objective_optimal}")
    return objective_value, objective_optimal


class IntervalSolution:
    def __init__(self, config: Dict, lower_bound: float, upper_bound: float):
        self.config = config
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    def dominates(self, sol, eps: float = 0.):
        return True if (1. + eps) * self.lower_bound >= sol.upper_bound else False
    def is_dominated(self, sol, eps: float = 0.):
        return True if (1. + eps) * sol.lower_bound >= self.upper_bound else False
    def __str__(self):
        output = f"{self.config}: [{self.lower_bound}, {self.upper_bound}]"
        return output

def evaluate_config(lcn: LCN, variables: List, interpretation: List, evidence: Dict):
    # Create a configuration of the variables
    config = dict(zip(variables, interpretation))
    config.update(evidence)

    # Build the conjunction query over all assigned variables (evidence literals
    # are baked into the conjunction, so no separate evidence is passed below).
    all_variables = [v for v, _ in config.items()]
    q = make_conjunction(all_variables, config)
    q = f"({q})"

    # Score P(q) bounds with the exact marginal engine: min for the lower bound,
    # max for the upper bound.
    lb, _ = solve_exact_model(lcn, q, lcn.independencies, evidence={},
                              sense="min", verbosity=0)
    ub, _ = solve_exact_model(lcn, q, lcn.independencies, evidence={},
                              sense="max", verbosity=0)
    return (lb, ub)

class ExactMAPInference:
    """
    Exact MAP inference algorithms for LCNs.
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
            seed: int = 42
    ):
        """
        Constructor for the exact MAP solver.

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
        """

        self.algo = algo
        self.time_limit = time_limit
        self.evidence = evidence
        self.query = query
        self.num_query = num_query
        self.map_init = map_init

        # The exact marginal engine needs the LCN's structure and LMC
        # independencies; build them once before searching.
        if self.lcn.independencies is None:
            self.lcn.build_primal_graph()
            self.lcn.build_structure_graph()
            self.lcn.local_markov_condition()

        if self.algo == "dfs":
            self._run_dfs()
        elif self.algo == "lds":
            self._run_lds()
        elif self.algo == "sa":
            self._run_sa()
        else:
            raise NotImplementedError(f"Exat MAP algorithm {algo} is not implemented.")
        
    def _run_dfs(self):
        """
        Run Depth-First Search (DFS) over the MAP variables and evaluate each
        assignment using the exact inference algorithm from [Marinescu et al, NeurIPS 2022].

        """

        # Get all variables
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
        print(f"[DFS] Searching over MAP variables: {map_vars}")
        print(f"[DFS] Query: {task}")
        print(f"[DFS] MAP method: {self.method}")

        # Initialize the search space
        stack = deque()
        root = (-1, [])
        stack.append(root)
        timeout = False
        start_time = time.time()
        best_score = -np.inf
        best_config = None
        best_frontier = []

        print("[DFS] Start search...")
        while len(stack) > 0:
            n = stack.pop()
            i, a = n[0], n[1]
            if i >= len(map_vars) - 1: # new solution found
                interpretation = a
                score = evaluate_config(self.lcn, map_vars, interpretation, self.evidence)
                print(f" interpretation: {interpretation} bounds: [{score}]")

                # Check for better solution
                if self.method == "maximin":
                    current_score = score[0] # lower bound
                    if current_score > best_score:
                        best_score = current_score
                        best_config = interpretation
                elif self.method == "maximax":
                    current_score = score[1] # upper bound
                    if current_score > best_score:
                        best_score = current_score
                        best_config = interpretation
                elif self.method == "interval":
                    solution = dict(zip(map_vars, interpretation))
                    solution.update(self.evidence)
                    new_sol = IntervalSolution(solution, score[0], score[1])
                    temp = []
                    for sol in best_frontier:
                        if not new_sol.dominates(sol, self.epsilon):
                            temp.append(sol)
                    temp.append(new_sol)
                    best_frontier = temp
            else: # expand node
               for val in range(2):
                    succ = (i + 1, a + [val])
                    stack.append(succ)
            
            # Check for timeout
            if self.time_limit > 0 and time.time() - start_time >= self.time_limit:
                print("[DFS] Search interrupted due to TIMEOUT.")
                timeout = True
                break
        
        # Stop timer and report solution
        elapsed = time.time() - start_time
        if not timeout:
            print("[DFS] Search terminated successfully.")
        print(f"[DFS] Time elapsed (sec): {elapsed}")
        print(f"[DFS] Search timeout: {timeout}")
        if self.method == "maximin":
            solution = dict(zip(map_vars, best_config))
            solution.update(self.evidence)
            print(f"[DFS] MAXIMIN-MAP score: {best_score}")
            print(f"[DFS] MAXIMIN-MAP solution: {solution}")
            self.best_solution_value = best_score
            self.best_solution_config = solution
        elif self.method == "maximax":
            solution = dict(zip(map_vars, best_config))
            solution.update(self.evidence)
            print(f"[DFS] MAXIMAX-MAP score: {best_score}")
            print(f"[DFS] MAXIMAX-MAP solution: {solution}")
            self.best_solution_value = best_score
            self.best_solution_config = solution
        elif self.method == "interval":
            print(f"[DFS] INTERVAL-MAP frontier: {len(best_frontier)}")
            for sol in best_frontier:
                print(sol)
            self.best_frontier = best_frontier

    def _run_lds(self):
        """
        Run Limited Discrepancy Search (LDS) over the MAP variables and
        evaluate each assignment using the exact inference algorithm from
        [Marinescu et al, NeurIPS 2022].
        """

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
        init_config = [0] * len(map_vars)
        task = "MMAP" if len(self.query) > 0 else "MAP"
        print(f"[LDS] Searching over MAP variables: {map_vars}")
        print(f"[LDS] Query: {task}")
        print(f"[LDS] MAP method: {self.method}")

        # Initialize the search space
        stack = deque()
        root = (-1, [], self.max_discrepancy)
        stack.append(root)
        best_score = -np.inf
        best_config = None
        best_frontier = []

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

        # Start the timer
        start_time = time.time()
        timeout = False

        # Limited discrepancy search
        n = next_node()
        while n:
            if (expand_node(n)):
                interpretation = [map_vals[x][y] for x,y in enumerate(n[1])]
                score = evaluate_config(self.lcn, map_vars, interpretation, self.evidence)

                print(f" interpretation: {interpretation} bounds: [{score}]")

                # Check for better solution
                if self.method == "maximin":
                    current_score = score[0] # lower bound
                    if current_score > best_score:
                        best_score = current_score
                        best_config = interpretation
                elif self.method == "maximax":
                    current_score = score[1] # upper bound
                    if current_score > best_score:
                        best_score = current_score
                        best_config = interpretation
                elif self.method == "interval":
                    solution = dict(zip(map_vars, interpretation))
                    solution.update(self.evidence)
                    new_sol = IntervalSolution(solution, score[0], score[1])
                    temp = []
                    for sol in best_frontier:
                        if not new_sol.dominates(sol, self.epsilon):
                            temp.append(sol)
                    temp.append(new_sol)
                    best_frontier = temp
                
                # Check for timeout
                if self.time_limit > 0 and time.time() - start_time >= self.time_limit:
                    print("[LDS] Search interrupted due to TIMEOUT.")
                    timeout = True
                    break
                
            n = next_node()

        # Stop timer and report solution
        elapsed = time.time() - start_time
        if not timeout:
            print("[LDS] Search terminated successfully.")
        print(f"[LDS] Time elapsed (sec): {elapsed}")
        print(f"[LDS] Search timeout: {timeout}")
        if self.method == "maximin":
            solution = dict(zip(map_vars, best_config))
            solution.update(self.evidence)
            print(f"[LDS] MAXIMIN-MAP score: {best_score}")
            print(f"[LDS] MAXIMIN-MAP solution: {solution}")
            self.best_solution_value = best_score
            self.best_solution_config = solution
        elif self.method == "maximax":
            solution = dict(zip(map_vars, best_config))
            solution.update(self.evidence)
            print(f"[LDS] MAXIMAX-MAP score: {best_score}")
            print(f"[LDS] MAXIMAX-MAP solution: {solution}")
            self.best_solution_value = best_score
            self.best_solution_config = solution
        elif self.method == "interval":
            print(f"[LDS] INTERVAL-MAP frontier: {len(best_frontier)}")
            for sol in best_frontier:
                print(sol)
            self.best_frontier = best_frontier
    
    def _run_sa(self):
        """
        Run Simulated Annealing (SA) over the MAP variables and evaluate each
        assignment using the exact inference from [Marinescu et al, NeurIPS 2022].
        """

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
        print(f"[SA] Searching over MAP variables: {map_vars}")
        print(f"[SA] Query: {task}")
        print(f"[SA] MAP method: {self.method}")
        print(f"[SA] Number of iterations: {self.num_iterations}")
        print(f"[SA] Max flips per iteration: {self.max_flips}")
        print(f"[SA] Initial temperature: {self.init_temperature}")
        print(f"[SA] Cooling schedule: {self.alpha}")
        print("[SA] MAP config evaluation: exact")

        # Initialize the cache and start the timer
        cache = {}
        timeout = False
        start_time = time.time()
        np.random.seed(self.seed)
        best_score = -np.inf
        best_config = None
        best_frontier = []
        num_flips = 0

        # Create a random MAP assignment and evaluate it.
        current_config = make_init_config(map_vars)
        score = evaluate_config(self.lcn, map_vars, current_config, self.evidence)
        best_config = current_config
        if self.method == "maximin":
            best_score = score[0] # lower bound
        elif self.method == "maximax":
            best_score = score[1] # upper bound
        elif self.method == "interval":
            raise NotImplementedError("SA is not implemented for intervals yet.")
        
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
                    score = evaluate_config(self.lcn, map_vars, next_config, self.evidence)
                    next_score = score[0] if self.method == "maximin" else score[1]
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

                    print(f"  - found better interpretation: {best_config} [{np.log(best_score)}]")
                
                # Adjust temperature
                T *= self.alpha

                # Check for timeout (during flips)
                elapsed = time.time()
                if self.time_limit > 0 and elapsed > self.time_limit:
                    print("[SA] Search terminated due to TIMEOUT.")
                    timeout = True
                    break
            
            # Check for timeout (during iteration)
            if timeout:
                break
        
        # Stop timer and report solution
        elapsed = time.time() - start_time
        if not timeout:
            print("[SA] Search terminated successfully.")
        print(f"[SA] Time elapsed (sec): {elapsed}")
        print(f"[SA] Search timeout: {timeout}")
        if self.method == "maximin":
            solution = dict(zip(map_vars, best_config))
            solution.update(self.evidence)
            print(f"[SA] MAXIMIN-MAP score: {best_score}")
            print(f"[SA] MAXIMIN-MAP solution: {solution}")
            self.best_solution_value = best_score
            self.best_solution_config = solution
        elif self.method == "maximax":
            solution = dict(zip(map_vars, best_config))
            solution.update(self.evidence)
            print(f"[SA] MAXIMAX-MAP score: {best_score}")
            print(f"[SA] MAXIMAX-MAP solution: {solution}")
            self.best_solution_value = best_score
            self.best_solution_config = solution
        elif self.method == "interval":
            print(f"[SA] INTERVAL-MAP frontier: {len(best_frontier)}")
            for sol in best_frontier:
                print(sol)
            self.best_frontier = best_frontier
            

if __name__ == "__main__":

    # Load the LCN
    file_name = "examples/asia.lcn"
    lcn_model = LCN()
    lcn_model.from_lcn(file_name=file_name)
    print(lcn_model)

    # Check consistency
    ok = check_consistency(lcn=lcn_model)
    if ok:
        print("CONSISTENT")
    else:
        print("INCONSISTENT")

    evidence = {'D': 1, 'X': 0, 'S': 1}
    query = ['B', 'C']

    # Run exact MAP inference
    algo = ExactMAPInference(
        lcn=lcn_model,
        method="maximin", 
        eps=0., 
        max_discrepancy=3, 
        num_iterations=5, 
        max_flips=10
    )
    algo.run(algo="dfs", query=query, evidence=evidence)

    print("Done.")

