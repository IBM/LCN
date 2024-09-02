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

# Generate random LCNs

import os
import itertools
import argparse
import numpy as np
import networkx as nx
from lcn.model import LCN, Sentence, Atom
from lcn.inference.utils import check_consistency
from lcn.utils import set_seed

def make_formula(variables: list, interpretation: list, connector: str):
    lits = []
    for i, var in enumerate(variables):
        if interpretation[i] == 0:
            lits.append(f"!x{var}")
        else:
            lits.append(f"x{var}")
    formula = f" {connector} ".join(lits)
    return formula

def make_lcn_random1(
        num_nodes: int, 
        num_extras: int, 
        epsilon: float,
        debug: bool = False,
        check_consistency: bool = True
) -> LCN:
    """
    Generate a random LCN with Random structure having the following sentences:
        P(y|x) such that we get a random graph (with cycles)
        P(x) where x is selected randomly (num_extras)
        
        Args:
            num_nodes: int
                The number of variables in the LCN.
            num_extras: int
                The number of extra P(x) sentences to be added.
            epsilon: float
                The maximum gap between the lower and upper bounds per sentence.
        
        Returns: LCN
            An LCN instance.
    """
    
    n = num_nodes
    ordering = [i for i in range(n)]
    position = [i for i in range(n)]

    # Randomly, switch pairs of variables
    for i in range(n):
        j = np.random.randint(n)
        k = ordering[j]
        ordering[j] = ordering[i]
        ordering[i] = k
        position[ordering[j]] = j
        position[ordering[i]] = i
    
    # Create the graph structure
    scopes = [[ordering[0]]]
    for i in range(1, n):
        x = ordering[i - 1]
        y = ordering[i]
        scopes.append([x, y])

    count = 1
    while count <= num_extras:
        x = np.random.randint(n)
        y = np.random.randint(n)
        if x != y and [x, y] not in scopes:
            scopes.append([x, y])
            count += 1
    
    assert (len(scopes) == num_nodes + num_extras)
    if debug:
        print(f"Generated scopes:")
        for s in scopes:
            print(f"  {s}")

    # Create the LCN instance
    lcn = LCN()
    atoms = [Atom(f"x{i}") for i in range(n)]
    lcn.add_atoms(atoms)
    connectors = ["and", "or"]
    for sid, scope in enumerate(scopes):
        if len(scope) == 1:
            child = scope[0]
            phi = f"x{child}"
            psi = None
        else:
            child = scope[-1]
            parents = scope[:-1]
            phi = f"x{child}"
            configs = list(itertools.product([0, 1], repeat=len(parents)))
            conn = connectors[np.random.randint(len(connectors))]
            config = configs[np.random.randint(len(configs))]
            psi = make_formula(parents, config, conn)

        val = np.random.uniform()
        lobo = max(0., val - epsilon)
        upbo = min(1., val + epsilon)
        s = Sentence(
            label=f"s{sid}",
            phi=phi,
            psi=psi,
            lower=lobo,
            upper=upbo
        )
        lcn.add_sentence(s)

    if check_consistency:
        # print(f"Build the LCN's primal graph.")
        lcn.build_primal_graph()
        # print(f"Build the LCN's structure graph.")
        lcn.build_structure_graph()
        # print(f"Build the LCN's independence assumptions (LMC).")
        lcn.local_markov_condition()

    if debug:
        print(f"Generated LCN:")
        print(lcn)

    return lcn        
# --

def make_lcn_random2(
        num_nodes: int, 
        num_extras: int, 
        epsilon: float,
        debug: float = False,
        check_consistency: bool = True,
) -> LCN:
    """
    Generate a random LCN with Random structure having the following sentences:
        P(x|y,z) or P(x|y) such that we get a random graph (with cycles)
        P(x) where x is selected randomly (num_extras)
        
        In case of P(x|y,z), the formula containing y,z will be randomly sampled
        from an "AND" or an "OR", and similarly the literals y,z will be sampled
        randomly.

        Args:
            num_nodes: int
                The number of variables in the LCN.
            num_extras: int
                The number of extra P(x) sentences to be added.
            epsilon: float
                The maximum gap between the lower and upper bounds per sentence.
        
        Returns: LCN
            An LCN instance.
    """

    assert (num_nodes > 2)

    # Init the number of nodes, roots and parents
    n = num_nodes
    p = 2
    r = p
    c = n - r

    # Init the variable ordering
    ordering = [i for i in range(n)]
    position = [i for i in range(n)]

    # Randomly, switch pairs of variables
    for i in range(n):
        j = np.random.randint(n)
        k = ordering[j]
        ordering[j] = ordering[i]
        ordering[i] = k
        position[ordering[j]] = j
        position[ordering[i]] = i
    
    # Generate the scopes
    candidates = []
    scopes = [None] * n
    cpts = [False] * n
    count = 0
    while count < c:
        child = ordering[np.random.randint(n - p)]
        if cpts[child] is True:
            continue
        cpts[child] = True
        num_higher_vars = n - position[child] - 1
        num_parents = np.random.randint(p) + 1 # 2

        candidates.append(child)
        parents = set()
        i = 0
        while i < num_parents:
            parent = ordering[np.random.randint(n)]
            if child == parent:
                continue
            if parent not in parents:
                parents.add(parent)
                i += 1

        scope = [parent for parent in parents]
        scope.append(child)
        scopes[child] = scope
        count += 1
    
    # Add the roots
    for i in range(n):
        if cpts[i] is False:
            scopes[i] = [i]
    
    # Create extra knowledge: l <= P(x) <= u
    extras = set()
    num_extras = min(num_extras, len(candidates))
    count = 1
    while count <= num_extras:
        k = np.random.randint(len(candidates))
        x = candidates[k]
        if x not in extras:
            extras.add(x)
            scopes.append([x])
            count += 1

    if debug:
        print(f"Generated scopes:")
        for s in scopes:
            print(f"  {s}")

    # Create the LCN instance
    lcn = LCN()
    atoms = [Atom(f"x{i}") for i in range(n)]
    lcn.add_atoms(atoms)
    connectors = ["and", "or"]
    for sid, scope in enumerate(scopes):
        if len(scope) == 1:
            child = scope[0]
            phi = f"x{child}"
            psi = None
        else:
            child = scope[-1]
            parents = scope[:-1]
            phi = f"x{child}"
            configs = list(itertools.product([0, 1], repeat=len(parents)))
            conn = connectors[np.random.randint(len(connectors))]
            config = configs[np.random.randint(len(configs))]
            psi = make_formula(parents, config, conn)

        val = np.random.uniform()
        lobo = max(0., val - epsilon)
        upbo = min(1., val + epsilon)
        s = Sentence(
            label=f"s{sid}",
            phi=phi,
            psi=psi,
            lower=lobo,
            upper=upbo
        )
        lcn.add_sentence(s)

    if check_consistency:
        # print(f"Build the LCN's primal graph.")
        lcn.build_primal_graph()
        # print(f"Build the LCN's structure graph.")
        lcn.build_structure_graph()
        # print(f"Build the LCN's independence assumptions (LMC).")
        lcn.local_markov_condition()

    if debug:
        print(f"Generated LCN:")
        print(lcn)

    return lcn        

# --

def make_lcn_dag(
        num_nodes: int, 
        num_extras: int, 
        epsilon: float, 
        debug: bool = False,
        check_consistency: bool = True
) -> LCN:
    """
    Generate a random LCN with DAG structure having the following sentences:
        P(x|y,z) or P(x|y) such that we get a DAG
        P(x) where x is selected randomly (num_extras)
        
        In case of P(x|y,z), the formula containing y,z will be randomly sampled
        from an "AND" or an "OR", and similarly the literals y,z will be sampled
        randomly.

        Args:
            num_nodes: int
                The number of variables in the LCN.
            num_extras: int
                The number of extra P(x) sentences to be added.
            epsilon: float
                The maximum gap between the lower and upper bounds per sentence.
        
        Returns: LCN
            An LCN instance.
    """

    assert (num_nodes > 2)

    # Initialize the number of nodes, roots and parents
    n = num_nodes
    p = 2
    r = p
    c = n - r

    # Initialize the variable ordering
    ordering = [i for i in range(n)]
    position = [i for i in range(n)]

    # Randomly, switch pairs of variables
    for i in range(n):
        j = np.random.randint(n)
        k = ordering[j]
        ordering[j] = ordering[i]
        ordering[i] = k
        position[ordering[j]] = j
        position[ordering[i]] = i
    
    # Generate the scopes of the sentences
    candidates = []
    scopes = [None] * n
    cpts = [False] * n
    count = 0
    while count < c:
        child = ordering[np.random.randint(n - p)]
        if cpts[child] is True:
            continue
        cpts[child] = True
        num_higher_vars = n - position[child] - 1
        num_parents = 2 #np.random.randint(p) + 1 # select 1 or 2 parents
        if num_parents > num_higher_vars:
            num_parents = num_higher_vars

        candidates.append(child)
        parents = set()
        i = 0
        while i < num_parents:
            parent = ordering[position[child] + 1 + np.random.randint(num_higher_vars)]
            if child == parent:
                continue
            if parent not in parents:
                parents.add(parent)
                i += 1

        scope = [parent for parent in parents]
        scope.append(child)
        scopes[child] = scope
        count += 1
    
    # Add the roots
    for i in range(n):
        if cpts[i] is False:
            scopes[i] = [i]
    
    # Create extra knowledge: l <= P(x) <= u
    extras = set()
    num_extras = min(num_extras, len(candidates))
    count = 1
    while count <= num_extras:
        k = np.random.randint(len(candidates))
        x = candidates[k]
        if x not in extras:
            extras.add(x)
            scopes.append([x])
            count += 1

    if debug:
        print(f"Generated scopes:")
        for s in scopes:
            print(f"  {s}")

    # Create the LCN instance
    lcn = LCN()
    atoms = [Atom(f"x{i}") for i in range(n)]
    lcn.add_atoms(atoms)
    connectors = ["and", "or"]
    for sid, scope in enumerate(scopes):
        if len(scope) == 1:
            child = scope[0]
            phi = f"x{child}"
            psi = None
        else:
            child = scope[-1]
            parents = scope[:-1]
            phi = f"x{child}"
            configs = list(itertools.product([0, 1], repeat=len(parents)))
            conn = connectors[np.random.randint(len(connectors))]
            config = configs[np.random.randint(len(configs))]
            psi = make_formula(parents, config, conn)

        val = np.random.uniform()
        lobo = max(0., val - epsilon)
        upbo = min(1., val + epsilon)
        s = Sentence(
            label=f"s{sid}",
            phi=phi,
            psi=psi,
            lower=lobo,
            upper=upbo
        )
        lcn.add_sentence(s)

    if check_consistency:
        # print(f"Build the LCN's primal graph.")
        lcn.build_primal_graph()
        # print(f"Build the LCN's structure graph.")
        lcn.build_structure_graph()
        # print(f"Build the LCN's independence assumptions (LMC).")
        lcn.local_markov_condition()

    if debug:
        print(f"Generated LCN:")
        print(lcn)

    return lcn        

# --

def make_lcn_polytree(
        num_nodes: int, 
        num_extras: int, 
        epsilon: float,
        debug: bool = False,
        check_consistency: bool = True
) -> LCN:
    """
    Generate a random LCN with Polytree structure having the following sentences:
        P(x|y,z) or P(x|y) such that we get a Polytree
        P(x) where x is selected randomly (num_extras)
        
        In case of P(x|y,z), the formula containing y,z will be randomly sampled
        from an "AND" or an "OR", and similarly the literals y,z will be sampled
        randomly.

        Args:
            num_nodes: int
                The number of variables in the LCN.
            num_extras: int
                The number of extra P(x) sentences to be added.
            epsilon: float
                The maximum gap between the lower and upper bounds per sentence.
        
        Returns: LCN
            An LCN instance.
    """

    n = num_nodes

    ordering = [i for i in range(n)]
    position = [i for i in range(n)]

    # Randomly, switch pairs of variables
    for i in range(n):
        j = np.random.randint(n)
        k = ordering[j]
        ordering[j] = ordering[i]
        ordering[i] = k
        position[ordering[j]] = j

    # Create a ordered chain P(xi|xi-1)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(1, n):
        G.add_edge(ordering[i-1], ordering[i])

    iter = 1
    while iter < n:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i < j:
            u = ordering[i]
            v = ordering[j]
            edge = (u, v)# if np.random.uniform() <= 0.5 else (v, u)
            if G.has_edge(edge[0], edge[1]) is False:
                UG = nx.to_undirected(G)
                paths = list(nx.all_simple_paths(UG, u, v))
                assert(len(paths) == 1)
                k = paths[0][-2]
                G.remove_edge(k, v)
                G.add_edge(edge[0], edge[1])
                iter += 1
    
    scopes = []
    candidates = []
    for child in range(n):
        parents = list(G.predecessors(child))
        scope = parents + [child]
        if len(scope) > 1:
            candidates.append(child)
        scopes.append(scope)

    # Create extra knowledge: l <= P(x) <= u
    extras = set()
    num_extras = min(num_extras, len(candidates))
    count = 1
    while count <= num_extras:
        k = np.random.randint(len(candidates))
        x = candidates[k]
        if x not in extras:
            extras.add(x)
            scopes.append([x])
            count += 1

    if debug:
        print(f"Generated scopes:")
        for s in scopes:
            print(f"  {s}")

    # Create the LCN instance
    lcn = LCN()
    atoms = [Atom(f"x{i}") for i in range(n)]
    lcn.add_atoms(atoms)
    connectors = ["and", "or"]
    for sid, scope in enumerate(scopes):
        if len(scope) == 1:
            child = scope[0]
            phi = f"x{child}"
            psi = None
        else:
            child = scope[-1]
            parents = scope[:-1]
            phi = f"x{child}"
            configs = list(itertools.product([0, 1], repeat=len(parents)))
            conn = connectors[np.random.randint(len(connectors))]
            config = configs[np.random.randint(len(configs))]
            psi = make_formula(parents, config, conn)

        val = np.random.uniform()
        lobo = max(0., val - epsilon)
        upbo = min(1., val + epsilon)
        s = Sentence(
            label=f"s{sid}",
            phi=phi,
            psi=psi,
            lower=lobo,
            upper=upbo
        )
        lcn.add_sentence(s)

    if check_consistency:
        # print(f"Build the LCN's primal graph.")
        lcn.build_primal_graph()
        # print(f"Build the LCN's structure graph.")
        lcn.build_structure_graph()
        # print(f"Build the LCN's independence assumptions (LMC).")
        lcn.local_markov_condition()

    if debug:
        print(f"Generated LCN:")
        print(lcn)

    return lcn        

# --

def make_lcn_factuality(
        num_atoms: int, 
        num_contexts_per_atom: int, 
        epsilon: float,
        debug: bool = False,
        check_consistency: bool = False
) -> LCN:
    """
    Generate a random LCN encoding the factuality usecase. There are two types
    of sentences:
        P(Y|X)  - for 'entailment': X entails Y
        P(!Y|X) - for 'contradiction': X contradicts Y
        
        Args:
            num_atoms: int
                The number of atoms in the response.
            num_contexts_per_atom: int
                The number contexts per atom.
            epsilon: float
                The maximum gap between the lower and upper bounds per sentence.
        
        Returns: LCN
            An LCN instance.
    """

    # Set the number of atoms (response) and contexts
    n = num_atoms
    m = num_atoms * num_contexts_per_atom

    # Create the atoms and contexts
    fact_atoms = [f"A{i+1}" for i in range(n)]
    fact_contexts = [f"C{j+1}" for j in range(m)]

    # Assign contexts for each atom
    atom2contexts = {}
    for i, a in enumerate(fact_atoms):
        ctxts = [fact_contexts[i*num_contexts_per_atom + j] for j in range(num_contexts_per_atom)]
        atom2contexts[a] = ctxts

    # print(atom2contexts)

    relations = {}

    # Simulate entailment and contradiction for atom2contexts relations
    num_rels = 0
    for a in fact_atoms:
        atom_contexts = atom2contexts[a]
        # print(f"processing atom: {a} with contexts: {atom_contexts}")
        for c in atom_contexts:
            p = np.random.uniform()
            num_rels += 1
            r = f"R{num_rels}"
            t = "entail" if p < .5 else "contradict"
            relations[r] = {
                "type": t,
                "probability": np.random.uniform(),
                "source": c,
                "target": a
            }

    # Simulate entailment and contradiction for context2context relationships
    max_pairs = m * (m - 1) / 2
    num_pairs = 0
    count = int(.1 * max_pairs)
    visited = set()
    while num_pairs < count:
        num_rels += 1
        xi = np.random.randint(m)
        xj = np.random.randint(m)
        ci = fact_contexts[xi]
        cj = fact_contexts[xj]
        r_str = f"{ci}_{cj}"
        if xi != xj and r_str not in visited:
            visited.add(r_str)
            r = f"R{num_rels}"
            p = np.random.uniform()
            t = "entail" if p < .5 else "contradict"
            relations[r] = {
                "type": t,
                "probability": np.random.uniform(),
                "source": ci,
                "target": cj
            }
            num_pairs += 1

    # print(f"Generated relations: {len(relations)}")
    # print(relations)

    # Create the LCN instance
    lcn = LCN()
    atoms = [Atom(a) for a in fact_atoms]
    atoms.extend([Atom(c) for c in fact_contexts])
    lcn.add_atoms(atoms)
    for rid, rel in relations.items():
        if rel["type"] == "entail":
            phi = rel["target"]
            psi = rel["source"]
        elif rel["type"] == "contradict":
            phi = "!" + rel["target"]
            psi = rel["source"]

        val = rel["probability"]
        lobo = max(0., val - epsilon)
        upbo = min(1., val + epsilon)
        s = Sentence(
            label=rid,
            phi=phi,
            psi=psi,
            lower=lobo,
            upper=upbo
        )
        lcn.add_sentence(s)

    if check_consistency:
        # print(f"Build the LCN's primal graph.")
        lcn.build_primal_graph()
        # print(f"Build the LCN's structure graph.")
        lcn.build_structure_graph()
        # print(f"Build the LCN's independence assumptions (LMC).")
        lcn.local_markov_condition()

    if debug:
        print(f"Generated LCN:")
        print(lcn)

    return lcn        

# --


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LCN generator.")
    parser.add_argument(
        "-t",
        "--topology",
        help="LCN topology [dag, polytree, random1, random2, factuality].",
        type=str
    )

    parser.add_argument(
        "-s",
        "--samples",
        help="Number of problem instances to generate.",
        type=int,
        default=10
    )

    parser.add_argument(
        "-n",
        "--num_vars",
        help="Number of variables.",
        type=int,
        default=5
    )

    parser.add_argument(
        "--num_atoms",
        help="Number of atoms (factuality only).",
        type=int,
        default=5
    )

    parser.add_argument(
        "--num_contexts_per_atom",
        help="Number of contexts per atom (factuality only).",
        type=int,
        default=2
    )

    parser.add_argument(
        "-x",
        "--num_extras",
        help="Number of extra singleton sentences",
        type=int,
        default=2
    )

    parser.add_argument(
        "-e",
        "--epsilon",
        help="Epsilon value for sentence bounds",
        type=float,
        default=0.3
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        help="Output directory.",
        type=str,
        default=""
    )

    parser.add_argument(
        "--check_consistency",
        help="Flag indicating whether to check consistency or not",
        action="store_true"
    )


    args = parser.parse_args()
    assert (args.topology in ["dag", "polytree", "random1", "random2", "factuality"])

    set_seed(42)

    # Generate LCNs
    count = 1
    print(f"Generating {args.topology} LCNs: {args.samples} instances.")
    while count <= args.samples:
        num_vars = args.num_vars if args.topology != "factuality" else args.num_atoms
        filename = f"lcn_{args.topology}_n{num_vars}_{count}.lcn"
        filename = os.path.join(args.output_dir, filename)
        if args.topology == "random1":
            l = make_lcn_random1(
                num_nodes=args.num_vars,
                num_extras=args.num_extras,
                epsilon=args.epsilon,
                check_consistency=args.check_consistency
            )
        elif args.topology == "random2":
            l = make_lcn_random2(
                num_nodes=args.num_vars,
                num_extras=args.num_extras,
                epsilon=args.epsilon,
                check_consistency=args.check_consistency
            )
        elif args.topology == "dag":
            l = make_lcn_dag(
                num_nodes=args.num_vars,
                num_extras=args.num_extras,
                epsilon=args.epsilon,
                check_consistency=args.check_consistency
            )
        elif args.topology == "polytree":
            l = make_lcn_polytree(
                num_nodes=args.num_vars,
                num_extras=args.num_extras,
                epsilon=args.epsilon,
                check_consistency=args.check_consistency
            )
        elif args.topology == "factuality":
            l = make_lcn_factuality(
                num_atoms=args.num_atoms,
                num_contexts_per_atom=args.num_contexts_per_atom,
                epsilon=args.epsilon,
                check_consistency=args.check_consistency
            )

        if args.check_consistency:
            ok = check_consistency(l)
            if ok:
                print("CONSISTENT")
                l.save_lcn(filename)
                count += 1
        else:
            print(f"Saving {args.topology} instance {count}")
            l.save_lcn(filename)
            count += 1

    print("Done.")

    # l = make_lcn_dag(num_nodes=5, num_extras=2, epsilon=0.4, debug=True)
    # l = make_lcn_polytree(num_nodes=5, num_extras=2, epsilon=0.4, debug=True)
    # l = make_lcn_random2(num_nodes=5, num_extras=2, epsilon=0.4, debug=True)
    # l = make_lcn_random1(num_nodes=5, num_extras=2, epsilon=0.4, debug=True)
    
    # l.from_lcn(file_name=file_name)
    # print(l)

    # Check consistency
    # ok = check_consistency(l)
    # if ok:
    #     print("CONSISTENT")
    # else:
    #     print("INCONSISTENT")


