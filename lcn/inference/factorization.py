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
from pyomo.environ import (
    ConcreteModel, 
    Set, NonNegativeReals, 
    Var, ConstraintList, 
    Objective, minimize, maximize,
    SolverFactory,
    SolverStatus,
    value,
    TerminationCondition
)

# Local
from lcn.model import LCN, SentenceType, Formula
from lcn.inference.utils import make_conjunction, check_consistency

infinity = float('inf')
max_iter = 1000
acceptable_tol = 1e-9
max_cpu_time = 100
debug = True
verbosity = 1


class Factorization:
    def __init__(self, lcn: LCN):
        self.lcn = lcn
        self.factors = []

    def solve_submodel(self, scope, literals, child, parents, sentences, sense):
        items = list(itertools.product([0, 1], repeat=len(scope)))
        index = {k:v for k, v in enumerate(items)}            
        N = len(items)
    
        # Create the model and variables
        model = ConcreteModel()
        model.ITEMS = Set(initialize=index.keys())
        model.p = Var(model.ITEMS, within=NonNegativeReals)
        model.constr = ConstraintList()

        # Create the constraints for the factor's sentences
        model.constr.add(sum(model.p[i] for i in model.ITEMS) == 1.0)
        for sid in sentences:
            s = self.lcn.sentences.get(sid)
            if s.type == SentenceType.Type1: # P(phi)
                A = [0] * N
                lobo = s.get_lower_bound()
                upbo = s.get_upper_bound()
                for j in range(N): # loop over all interpretations
                    config = dict(zip(scope, index[j]))
                    A[j] = 1 if s.phi_formula.evaluate(table=config) == True else 0
                model.constr.add(sum(A[i]*model.p[i] for i in model.ITEMS) >= lobo)
                model.constr.add(sum(A[i]*model.p[i] for i in model.ITEMS) <= upbo)
            else: # Type 2 sentence P(phi | psi)
                Aqr = [0] * N
                Ar = [0] * N
                lobo = s.get_lower_bound()
                upbo = s.get_upper_bound()
                for j in range(N):
                    config = dict(zip(scope, index[j]))
                    Aqr[j] = 1 if s.phi_and_psi_formula.evaluate(table=config) == True else 0
                    Ar[j] = 1 if s.psi_formula.evaluate(table=config) == True else 0
                val = sum(Ar[i]*model.p[i] for i in model.ITEMS)
                model.constr.add(sum(Aqr[i]*model.p[i] for i in model.ITEMS) >= lobo*val)
                model.constr.add(sum(Aqr[i]*model.p[i] for i in model.ITEMS) <= upbo*val)
    
        # Create the objective
        Fq = make_conjunction(variables=scope, literals=literals)
        A = [0] * N
        for j in range(N):
            config = dict(zip(scope, index[j]))
            A[j] = 1 if Fq.evaluate(table=config) == True else 0

        # Check if we have a denominator
        if len(parents) == 0:
            obj = sum(A[i]*model.p[i] for i in model.ITEMS)
            if sense == 'min':
                model.objective = Objective(expr=obj, sense=minimize)
            else:
                model.objective = Objective(expr=obj, sense=maximize)
        else:
            Fe = make_conjunction(variables=parents, literals=literals)
            E = [0] * N
            AE = [0] * N
            for j in range(N):
                config = dict(zip(scope, index[j]))
                E[j] = 1 if Fe.evaluate(table=config) == True else 0
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
            # if hessian_approximation is not None:
            #     opt.options['hessian_approximation'] = hessian_approximation
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
        return objective_value

    def build(self):
        """
        Process the factorization
        """

        # Ensure that the LCN has been postprocessed (structure, etc.)
        assert self.lcn.structure_graph is not None
        assert self.lcn.simplified_structure_graph is not None
        assert self.lcn.families is not None

        # Process each family
        self.factors = []
        for family in self.lcn.families:
            print(f"Processing family: {family}")
            child = family["child"]
            parents = family["parents"]
            sentences = family["sentences"]
            vars = [child] if "-" not in child else child.split("-")
            for par in parents:
                if "-" in par:
                    vars.append(par.split("-"))
                else:
                    vars.append(par)

            print(f"Processing family: {child} <-- {parents}")
            print(f"Full scope: {vars}")

            # Iterate over all interpretations of the scope
            factor = {}
            interpretations = list(itertools.product([0, 1], repeat=len(vars)))
            for i, interpretation in enumerate(interpretations):
                literals = dict(zip(vars, interpretation))
                lobo = self.solve_submodel(vars, literals, child, parents, sentences, sense="min")
                upbo = self.solve_submodel(vars, literals, child, parents, sentences, sense="max")
                factor[i] = {
                    "interpretation": interpretation,
                    "scope": vars,
                    "child": child,
                    "parents": parents,
                    "lobo": lobo,
                    "upbo": upbo
                }
            
            self.factors.append(factor)

        return self.factors

