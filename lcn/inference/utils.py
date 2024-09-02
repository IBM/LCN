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
import numpy as np
from pyomo.environ import *
from typing import List, Dict

# Local
from lcn.model import LCN, Formula, SentenceType

def make_init_config(vars: List):
    return [1 if np.random.random() > 0.5 else 0 for _ in vars]

def select_neighbor(elements: List):
    sel = np.random.randint(0, len(elements))
    return elements[sel]

def find_neighbors(config: List) -> List[List]:
    neighbors = []
    for pos in range(len(config)):
        neighbor = []
        for i, val in enumerate(config):
            if i != pos: # leave value unchanged
                neighbor.append(val)
            else: # flip the value at position pos
                neighbor.append(1 if val == 0 else 0)
        neighbors.append(neighbor)
    return neighbors

def make_conjunction(variables: List, literals: Dict) -> Formula:
    """
    Returns the conjunction of the input literals
    """
    
    assert len(variables) > 0, "Variables list cannot be empty."
    assert len(literals) > 0, "Literals dict cannot be empty."

    lits = []
    for x in variables:
        if literals[x] == 1:
            lits.append(x)
        else:
            lits.append(f"!{x}")
    conjunction_str = " and ".join(lits)
    return Formula(label="conjunction", formula=conjunction_str)

def check_consistency(lcn: LCN) -> bool:
    """
    Check if the LCN is consistent or not. An LCN is consistent
    if there exists a model (i.e., interpretation) that satisfies
    the LCN's sentences and the Local Markov Condition independencies.

    Args:
        lcn: LCN
            The input LCN model.
    
    Returns:
        `True` if the LCN is consistent and `False` otherwise.
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
    independencies = lcn.local_markov_condition()
    print(f"Local Markov Condition yields {len(independencies.get_assertions())} independencies.")
    for indep in independencies.get_assertions():
        X, T, S = list(indep.event1), list(indep.event2), list(indep.event3)
        print(f"independence: {indep}")
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
    model.objective = Objective(expr=1.0, sense=maximize)

    try:
        # Solve the non-linear model exactly
        opt = SolverFactory('ipopt')
        opt.options['max_iter'] = 100000
        opt.options['max_cpu_time'] = 7200
        # opt.options['acceptable_tol'] = 0.00001
        # opt.options['hessian_approximation'] = 'limited-memory'
        results = opt.solve(model, tee=False)
        if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
            print(f"Solver status: {results.solver.status}")
            objective_value = value(model.objective)
            objective_optimal = True
        elif (results.solver.termination_condition == TerminationCondition.infeasible):
            print(f"Solver status: {results.solver.status}")
            objective_value = value(model.objective)
            objective_optimal = False
        else:
            print(f"Solver status: {results.solver.status}")
            objective_value = value(model.objective)
            objective_optimal = False

    except Exception as e:
        print(f"Exception during ipopt: {str(e)}")
        objective_value = None
        objective_optimal = False

    print(f"[Ipopt] objective={objective_value}, optimal={objective_optimal}")
    return objective_optimal

