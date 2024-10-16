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
from typing import Tuple

# Local
from lcn.model import LCN, SentenceType, Formula
from lcn.independencies import Independencies
from lcn.inference.utils import make_conjunction, check_consistency

infinity = float('inf')

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
        hessian_approximation: str = None
) -> Tuple:
    """
    Compute exact lower/upper bounds on the probabability of the query formula
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

    Returns:
        A tuple representing the objective value and a flag indicating its optimality.
    """
    
    # Create the interpretations
    vars = [k for k, _ in lcn.atoms.items()]
    items = list(itertools.product([0, 1], repeat=len(vars)))
    index = {k:v for k, v in enumerate(items)}
    N = len(items)

    # Create the model and variables
    model = ConcreteModel()
    model.ITEMS = Set(initialize=index.keys())
    model.p = Var(model.ITEMS, within=NonNegativeReals)
    model.constr = ConstraintList()

    # Create the constraint ensuring a probability distribution over interpretations
    model.constr.add(sum(model.p[i] for i in model.ITEMS) == 1.0)

    # Create the constraints for the sentences
    for sid, s in lcn.sentences.items():
        if s.type == SentenceType.Type1: # Type 1 sentence P(phi)
            A = [0] * N
            lobo = s.get_lower_bound()
            upbo = s.get_upper_bound()
            for j in range(N):
                config = dict(zip(vars, index[j]))
                A[j] = 1 if s.phi_formula.evaluate(table=config) == True else 0
            model.constr.add(sum(A[i]*model.p[i] for i in model.ITEMS) >= lobo)
            model.constr.add(sum(A[i]*model.p[i] for i in model.ITEMS) <= upbo)
        else: # Type 2 sentence: P(phi|psi)
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

    # Constraints corresponding to the independence assumptions
    # Atom x is conditionaly independent of non-parents non-descendants (T) 
    # given its parents (S) in the primal graph of the LCN
    # Namely, we consider independence assertions [X, Y=T, Z=S]
    # i.e., P(x|S,T) = P(x|S)
    #   P(x,S,T)P(S) = P(x,S)P(S,T)
    # Here, independencies are coming from LCN's Local Markov Condition
    for indep in independencies.get_assertions():
        X, T, S = list(indep.event1), list(indep.event2), list(indep.event3)
        if verbosity > 0:
            print(f"adding constraints for independence: {indep}")
        configs_S = [()] if len(S) == 0 else list(itertools.product([0, 1], repeat=len(S)))     
        if len(S) > 0:
            for t in T:
                # add constraints P(x|t,S) = P(x|S)
                x = X[0]
                literals = {x:1, t:1}
                # print(f"adding constraint: {x} _||_ {t} | {S}")
                for s in configs_S:
                    literals.update(dict(zip(S, list(s))))
                    Aa = [0] * N
                    Ab = [0] * N
                    Ac = [0] * N
                    Ad = [0] * N
                    Fa = make_conjunction(variables=X+S+[t], literals=literals)
                    Fb = make_conjunction(variables=S, literals=literals)
                    Fc = make_conjunction(variables=X+S, literals=literals)
                    Fd = make_conjunction(variables=S+[t], literals=literals)
                    for j in range(N):
                        interpretation = dict(zip(vars, index[j]))
                        Aa[j] = 1 if Fa.evaluate(table=interpretation) else 0
                        Ab[j] = 1 if Fb.evaluate(table=interpretation) else 0
                        Ac[j] = 1 if Fc.evaluate(table=interpretation) else 0
                        Ad[j] = 1 if Fd.evaluate(table=interpretation) else 0
                    val1 = sum(Aa[i]*model.p[i] for i in model.ITEMS) * sum(Ab[i]*model.p[i] for i in model.ITEMS)
                    val2 = sum(Ac[i]*model.p[i] for i in model.ITEMS) * sum(Ad[i]*model.p[i] for i in model.ITEMS)
                    model.constr.add(val1 - val2 == 0.0)    
        else:
            # no parents so basically P(x,t) = P(x)P(t)
            for t in T:
                x = X[0]
                literals = {x: 1, t: 1}
                # print(f"adding constraint: {x} _||_ {t}")
                Aa = [0] * N
                Ab = [0] * N
                Ac = [0] * N
                Fa = make_conjunction(variables=X+[t], literals=literals)
                Fb = make_conjunction(variables=X, literals=literals)
                Fc = make_conjunction(variables=[t], literals=literals)
                for j in range(N):
                    interpretation = dict(zip(vars, index[j]))
                    Aa[j] = 1 if Fa.evaluate(table=interpretation) else 0
                    Ab[j] = 1 if Fb.evaluate(table=interpretation) else 0
                    Ac[j] = 1 if Fc.evaluate(table=interpretation) else 0
                val1 = sum(Aa[i]*model.p[i] for i in model.ITEMS)
                val2 = sum(Ab[i]*model.p[i] for i in model.ITEMS) * sum(Ac[i]*model.p[i] for i in model.ITEMS)
                model.constr.add(val1 - val2 == 0.0)    

    # Create the objective
    obj_formula = Formula(label="obj", formula=query_formula)
    A = [0] * N
    for j in range(N):
        config = dict(zip(vars, index[j]))
        A[j] = 1 if obj_formula.evaluate(table=config) == True else 0

    # Check if we have evidence
    if len(evidence) == 0:
        obj = sum(A[i]*model.p[i] for i in model.ITEMS)
        if sense == 'min':
            model.objective = Objective(expr=obj, sense=minimize)
        else:
            model.objective = Objective(expr=obj, sense=maximize)
    else:
        ev = [k for k, _ in evidence.items()]
        Fe = make_conjunction(variables=ev, literals=evidence)
        E = [0] * N
        AE = [0] * N
        for j in range(N):
            interpretation = dict(zip(vars, index[j]))
            E[j] = 1 if Fe.evaluate(table=interpretation) == True else 0
            if A[j] == 1 and E[j] == 1:
                AE[j] == 1
        obj1 = sum(AE[i]*model.p[i] for i in model.ITEMS)
        obj2 = sum(E[i]*model.p[i] for i in model.ITEMS)
        if sense == 'min':
            model.objective = Objective(expr=obj1/obj2, sense=minimize)
        else:
            model.objective = Objective(expr=obj1/obj2, sense=maximize)

    try:
        # Solve the non-linear model exactly
        opt = SolverFactory('ipopt')
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

    if verbosity > 0:
        print(f"[Ipopt] objective={objective_value}, optimal={objective_optimal}")
    return objective_value, objective_optimal
    
class ExactInferece:
    """
    The exact marginal inference algorithm for LCNs
    see [Marinescu et al. Logical Credal Networks. NeurIPS 2022]
    """

    def __init__(
            self, 
            lcn: LCN
    ):
        """
        Constructor of the Exact Inference solver.

        Args:
            lcn: LCN
                The input LCN model.
        """
        self.lcn = lcn
        self.lower_bound = None
        self.upper_bound = None
        self.feasible = None

    def run(
            self, 
            query_formula: str, 
            evidence: dict = {}, 
            debug: bool = False,
            verbosity: int = 1
    ):
        """
        Run the exact inference algorithm.

        Args:
            query_formula: str
                A string representing the query formula.
            evidence: dict
                A dictionary containing the observed evidence variables.
            debug: bool
                A flag indicating that ipopt is run in debugging mode.
            verbosity: int
                Verbosity level (0 is silent)
        """

        assert self.lcn is not None, "Make sure the LCN model exists."
        assert self.lcn.independencies is not None, "Make sure the LMC is applied."

        # Start the timer
        t_start = time.time()

        # Get the independencies from the Local Markov Condition
        if verbosity > 0:
            num_indep = len(self.lcn.independencies.get_assertions())
            print(f"[Local Markov Condition: {num_indep} independencies]")
            for indep in self.lcn.independencies.get_assertions():
                print(indep)

        lower_bound, feasible_lb = solve_exact_model(
            lcn=self.lcn, 
            query_formula=query_formula, 
            evidence=evidence,
            independencies=self.lcn.independencies, 
            sense='min', 
            debug=debug,
            verbosity=verbosity,
            acceptable_tol=0.000001,
            hessian_approximation="limited-memory"
        )
        upper_bound, feasible_ub = solve_exact_model(
            lcn=self.lcn, 
            query_formula=query_formula, 
            independencies=self.lcn.independencies, 
            evidence=evidence,
            sense='max', 
            debug=debug,
            verbosity=verbosity,
            acceptable_tol=0.000001,
            hessian_approximation="limited-memory"
        )

        t_end = time.time()
        self.lower_bound = .0 if not feasible_lb else max(abs(lower_bound), 0.0)
        self.upper_bound = .0 if not feasible_ub else min(abs(upper_bound), 1.0)
        self.feasible = feasible_lb and feasible_ub

        if verbosity > 0:
            print(f"[ExactInference] Result for {query_formula} is: [ {self.lower_bound}, {self.upper_bound} ]")        
            print(f"[ExactInference] Feasibility: lb={feasible_lb}, ub={feasible_ub}, all={self.feasible}")
            print(f"[ExactInference] Time elapsed: {t_end - t_start} sec")


if __name__ == "__main__":

    # Load the LCN
    file_name = "examples/asia.lcn"
    l = LCN()
    l.from_lcn(file_name=file_name)
    print(l)

    # Check consistency
    ok = check_consistency(l)
    if ok:
        print("CONSISTENT")
    else:
        print("INCONSISTENT")

    # Run exact marginal inference
    query = "(B and !C)"
    algo = ExactInferece(lcn=l)
    algo.run(query_formula=query, debug=False)




