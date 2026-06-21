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

import time
import random
import numpy as np
from typing import Dict, Tuple
from collections import deque

# Local
from lcn.core.model import LCN, Sentence
from lcn.inference.marginal.ariel import ArielInference
from lcn.inference.utils.common import check_consistency
from lcn.inference.utils.common import make_init_config, select_neighbor, find_neighbors

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
    aug, q = build_augmented_lcn(lcn, assignment, debug)

    # ArielInference needs the LCN's structure graph and LMC independencies.
    aug.build_primal_graph()
    aug.build_structure_graph()
    aug.local_markov_condition()

    # Run ARIEL on the augmented LCN and read the bounds on P(q = 1), where q is
    # the final auxiliary variable. The result maps each atom to (lo_arr, hi_arr)
    # with index 1 being the P(atom = 1) bounds.
    ariel = ArielInference(lcn=aug)
    res = ariel.run(evidence={}, n_iters=n_iters, threshold=threshold,
                    verbosity=0)
    lo_q, hi_q = res[q][0][1], res[q][1][1]
    elapsed = time.time() - t_start

    # Get the score and time elapsed in seconds
    score = lo_q if map_task == "maximin" else hi_q
    return score, elapsed

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
            raise NotImplementedError("LDS is not implemented for intervals yet.")

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
        print("[ALDS] MAP config evaluation: approximate")
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
        best_score = -np.inf
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
                    print("[LDS] Search interrupted due to TIMEOUT.")
                    timeout = True
                    break
                
            n = next_node()

        # Stop timer and report solution
        elapsed = time.time() - start_time
        if not timeout:
            print("[ALDS] Search terminated successfully.")
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
            raise NotImplementedError("SA is not implemented for intervals yet.")

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
        print("[ASA] MAP config evaluation: approximate")
        print(f"[ASA] MAP init config: {self.map_init}")

        # Initialize the cache and start the timer
        cache = {}
        np.random.seed(self.seed)
        best_score = -np.inf
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
                    print("[SA] Search terminated due to TIMEOUT.")
                    timeout = True
                    break
            
            # Check for timeout (during iteration)
            if timeout:
                break
        
        # Stop timer and report solution
        elapsed = time.time() - start_time
        if not timeout:
            print("[ASA] Search terminated successfully.")
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
        Run the AMAP scheme: compute approximate marginals with the ARIEL
        message-passing engine (lcn.inference.marginal.ariel.ArielInference),
        then greedily pick the per-variable MAP assignment from the bounds.
        """

        # Start the timer
        t_start = time.time()

        # Get all variables and select the MAP/MMAP variables.
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

        # ArielInference needs the LCN's structure graph and LMC independencies.
        if self.lcn.independencies is None:
            self.lcn.build_primal_graph()
            self.lcn.build_structure_graph()
            self.lcn.local_markov_condition()

        # Run ARIEL marginal inference. The result maps each atom to
        # (lo_arr, hi_arr); index 1 holds the P(atom = 1) bounds.
        print("[ARIEL] Running marginal inference...")
        print(f"[ARIEL] MAP variables: {map_vars}")
        print(f"[ARIEL] Query: {task}")
        ariel = ArielInference(lcn=self.lcn)
        res = ariel.run(evidence=self.evidence,
                        n_iters=self.ariel_iterations,
                        threshold=self.ariel_threshold,
                        verbosity=1 if self.debug else 0)
        # marginals[nid] = (lower P(nid=1), upper P(nid=1))
        self.marginals = {nid: (lo[1], hi[1]) for nid, (lo, hi) in res.items()}
        t_end = time.time()

        print("[ARIEL] Marginals:")
        for nid in sorted(self.marginals):
            lo, hi = self.marginals[nid]
            print(f"{nid}: [{lo}, {hi}]")
        print(f"[ARIEL] Time elapsed (sec): {t_end - t_start}")

        # Get the MAP config (consistent with evidence)
        self.best_solution_value = 1.
        self.best_solution_config = {}
        if self.method == "maximin":
            # look at the lower bounds
            for nid in sorted(self.marginals):
                if nid in self.evidence:
                    continue # skip evidence for now
                if len(self.query) > 0 and nid not in self.query:
                    continue # skip non-MAP vars if MMAP query
                lb = self.marginals[nid][0]
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
            for nid in sorted(self.marginals):
                if nid in self.evidence:
                    continue # skip evidence for now
                if len(self.query) > 0 and nid not in self.query:
                    continue # skip non-MAP vars if MMAP query
                ub = self.marginals[nid][1]
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
    file_name = "examples/asia.lcn"
    lcn_model = LCN()
    lcn_model.from_lcn(file_name=file_name)
    print(lcn_model)

    # Check consistency
    ok = check_consistency(lcn_model)
    if ok:
        print("CONSISTENT")
    else:
        print("INCONSISTENT")

    evidence = {'D': 1, 'X': 0, 'S': 1}
    query = ['B', 'C']

    # Run approximate marginal inference
    algo = ApproximateMAPInference(
        lcn=lcn_model,
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


