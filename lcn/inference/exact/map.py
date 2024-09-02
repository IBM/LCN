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
import numpy as np

from pyomo.environ import *
from typing import Dict, List
from collections import deque

# Local
from lcn.model import LCN
from lcn.inference.exact.marginal import ExactInferece
from lcn.inference.utils import make_conjunction, check_consistency
from lcn.inference.utils import make_init_config, select_neighbor, find_neighbors

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

def evaluate_config(evaluator, variables: List, interpretation: List, evidence: Dict):
    # Create a configuration of the variables
    config = dict(zip(variables, interpretation))
    config.update(evidence)

    # Evaluate current assignment
    all_variables = [v for v, _ in config.items()]
    q = make_conjunction(all_variables, config)
    q = f"({q})"
    evaluator.run(q, verbosity=0)
    lb = evaluator.lower_bound
    ub = evaluator.upper_bound
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

        # Create the evaluator
        evaluator = ExactInferece(self.lcn)

        # Initialize the search space
        stack = deque()
        root = (-1, [])
        stack.append(root)
        timeout = False
        start_time = time.time()
        best_score = -np.infty
        best_config = None
        best_frontier = []

        print(f"[DFS] Start search...")
        while len(stack) > 0:
            n = stack.pop()
            i, a = n[0], n[1]
            if i >= len(map_vars) - 1: # new solution found
                interpretation = a
                score = evaluate_config(evaluator, map_vars, interpretation, self.evidence)
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
                print(f"[DFS] Search interrupted due to TIMEOUT.")
                timeout = True
                break
        
        # Stop timer and report solution
        elapsed = time.time() - start_time
        if not timeout:
            print(f"[DFS] Search terminated successfully.")
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

        # Create the evaluator
        evaluator = ExactInferece(self.lcn)

        # Initialize the search space
        stack = deque()
        root = (-1, [], self.max_discrepancy)
        stack.append(root)
        best_score = -np.infty
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
                score = evaluate_config(evaluator, map_vars, interpretation, self.evidence)

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
                    print(f"[LDS] Search interrupted due to TIMEOUT.")
                    timeout = True
                    break
                
            n = next_node()

        # Stop timer and report solution
        elapsed = time.time() - start_time
        if not timeout:
            print(f"[LDS] Search terminated successfully.")
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
        print(f"[SA] MAP config evaluation: exact")

        # Create the evaluator
        evaluator = ExactInferece(self.lcn)

        # Initialize the cache and start the timer
        cache = {}
        timeout = False
        start_time = time.time()
        np.random.seed(self.seed)
        best_score = -np.infty
        best_config = None
        best_frontier = []
        num_flips = 0

        # Create a random MAP assignment and evaluate it.
        current_config = make_init_config(map_vars)
        score = evaluate_config(evaluator, map_vars, current_config, self.evidence)
        best_config = current_config
        if self.method == "maximin":
            best_score = score[0] # lower bound
        elif self.method == "maximax":
            best_score = score[1] # upper bound
        elif self.method == "interval":
            raise NotImplementedError(f"SA is not implemented for intervals yet.")
        
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
                    score = evaluate_config(evaluator, map_vars, next_config, self.evidence)
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
                    print(f"[SA] Search terminated due to TIMEOUT.")
                    timeout = True
                    break
            
            # Check for timeout (during iteration)
            if timeout:
                break
        
        # Stop timer and report solution
        elapsed = time.time() - start_time
        if not timeout:
            print(f"[SA] Search terminated successfully.")
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
    file_name = "/home/radu/git/fm-factual/examples/lcn/factual2.lcn"
    l = LCN()
    l.from_lcn(file_name=file_name)
    print(l)

    # Check consistency
    ok = check_consistency(lcn=l)
    if ok:
        print("CONSISTENT")
    else:
        print("INCONSISTENT")

    evidence = {}
    query = ['A1']

    # Run exact MAP inference
    algo = ExactMAPInference(
        lcn=l, 
        method="maximax", 
        eps=0., 
        max_discrepancy=3, 
        num_iterations=5, 
        max_flips=10
    )
    algo.run(algo="dfs", query=query, evidence=evidence)

    print(f"Done.")

