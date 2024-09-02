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

# The LCN model

import networkx as nx
from typing import List
from enum import Enum

# Local
from lcn.parser import parse_formula, evaluate_formula, validate_formula
from lcn.independencies import Independencies, IndependenceAssertion

class SentenceType(Enum):
    Type1 = 1
    Type2 = 2

class Formula:
    """
    A propositional logic formula over a set of variables/atoms. The following
    logical connectors are supported: not, and, or, xor, nand, nor. Please see
    the FormulaParser class for the actual symbols allowed by the parser.

    e.g., (A or !B and C) 
    """
    def __init__(
            self,
            label: str,
            formula: str
    ):
        """
        The propositional logic formula constructor.

        Args:
            formula: str
                A string representing the logic formula e.g., (A or !B and C)
        """
        
        self.label = label # unique identifier of the formula (is the atom if atomic)
        self.input_formula = formula # store the formula
        output, atoms = parse_formula(formula) # parse the formula
        if output is None or vars is None:
            raise ValueError(f"Malformed formula: {formula}")
        
        self.parsed_formula = output # parse tree of the formula
        self.atoms = atoms # a dict indexed by 'Vi' where i is the i-th variable
        if self.is_atomic(): # i.e., atomic formula
            self.label = self.atoms['V1']

    def evaluate(
            self, 
            table: dict
    ):
        """
        Evaluate the truth value of the formula given the truth values of its atoms.

        Args:
            table: dict
                A dict containing the truth values of the formula's atoms

        Returns:
            The truth value (True or False) of the formula
        """

        assert table is not None and len(table) > 0, \
            f"The truth values of the formula's atoms must be provided."
        
        return evaluate_formula(i=self.input_formula, table=table)

    def validate(self):
        """
        Check if the syntax of the formula is correct
        """
        return validate_formula(self.input_formula)        

    def __str__(self):
        return f"{self.input_formula}"
    
    def is_atomic(self):
        """
        Check if the formula is atomic or not. By definition, a formula is
        `atomic` if and only if it consists of a single positive literal.

        Returns:
            True if the formula is atomic, otherwise False.
        """
        if len(self.atoms) == 1 and \
            self.evaluate(table={self.atoms['V1']: 1}) == True:
            return True
        else:
            return False

class Atom:
    """
    An atomic formula i.e., proposition or grounded predicate
    """
    def __init__(
            self,
            name: str
    ):
        """
        Create an atomic formula.

        Args:
            name: str
                A unique identifier for the atom.
        """

        self.name = name
        self.formula = Formula(label=name, formula=name)

    def __str__(self):
        return self.name

class Sentence:
    """
    A sentence in the LCN model. There are two types of sentences:
      (1) P(phi)
      (2) P(phi | psi)
    """
    def __init__(
            self,
            label: str,
            phi: str,
            psi: str = None,
            lower: float = 0.0,
            upper: float = 1.0,
            tau: bool = True
    ):
        """
        Create a type (1) sentence of the form: lb <= P(phi) <= up 
        or a type (2) sentence of the form: lb <= P(phi | psi) <= ub. If the
        `psi` formula is None, then we assume a Type 1 sentence, otherwise we
        have Type 2 sentence.

        Args:
            label: str
                A unique identifier for the sentence
            phi: Formula
                The logical formula.
            psi : Formula
                The conditioned logical formula.
            lower: float
                The lower probability bound (default 0.0)
            upper: float
                The upper probability bound (default 1.0)
            tau: bool
                A flag indicating indendence of variables/atoms in `phi`
        """

        assert lower <= upper, \
            f"Invalid probability bounds: {lower} must be smaller than {upper}."

        self.label = label
        self.phi = phi # the `phi` string formula
        self.psi = psi # the `psi` string formula (default is None)
        self.phi_formula = Formula(label=self.label + "_phi", formula=phi)
        self.psi_formula = None
        if psi is not None:
            self.psi_formula = Formula(label=self.label + "_psi", formula=psi)
            self.phi_and_psi = f"({self.phi}) and ({self.psi})"
            self.phi_and_psi_formula = Formula(label=self.label + "_phi_psi", formula=self.phi_and_psi)
        self.lower_prob = lower
        self.upper_prob = upper
        self.tau = tau # default is True, i.e., variables in `phi` are independent
        self.type = SentenceType.Type1 if psi is None else SentenceType.Type2
        self.independencies = None

        # Update the atoms
        self.atoms = {}
        phi_atoms = self.phi_formula.atoms
        for _, v in phi_atoms.items():
            self.atoms[v] = Atom(v)

        psi_atoms = {} if self.psi_formula is None else self.psi_formula.atoms
        for _, v in psi_atoms.items():
            if v not in self.atoms:
                self.atoms[v] = Atom(v)

    def get_atoms(self) -> dict:
        """
        Returns a dict {name: Atom} containing the sentence's atoms.
        """
        return self.atoms
   
    def get_lower_bound(self):
        """
        Returns the lower probability bound.
        """
        return self.lower_prob
    
    def get_upper_bound(self):
        """
        Returns the upper probability bound.
        """
        return self.upper_prob
    
    def set_lower_bound(
            self, 
            lower: float
    ):
        """
        Set the lower probability bound.

        Args:
            lower: float
                The new lower probability bound.
        """
        assert (lower >= 0.0 and lower <= 1.0), \
            f"Invalid lower probability bound. Must be between 0 and 1."
        self.lower_prob = lower

    def set_upper_bound(
            self, 
            upper: float
    ):
        """
        Set the new upper probability bound.

        Args:
            upper: float
                The new upper probability bound.
        """
        assert (upper >= 0.0 and upper <= 1.0), \
            f"Invalid upper probability bound. Must be between 0 and 1."
        self.upper_prob = upper

    def validate(self):
        """
        Validate the LCN sentence, i.e., make sure it is syntactically correct.

        Returns:
            True if the syntax is correct, otherwise False.
        """

        if self.lower > self.upper:
            print(f"Sentence {self.name} is invalid: lower bound greater than upper bound.")
            return False
        
        if self.phi is not None:
            return self.phi.validate()
        
        if self.psi is not None:
            return self.psi.validate()
        
        return True

    def is_unary_positive(self):
        """
        Check if the sentence is defined on a single positive atom e.g., P(X).
        """
        if len(self.atoms) == 1 and "!" not in self.phi:
            return True
        else:
            return False
        
    def is_unary_negative(self):
        """
        Check if the sentence is defined on a single positive atom e.g., P(X).
        """
        if len(self.atoms) == 1 and "!" in self.phi:
            return True
        else:
            return False


    def __str__(self):
        if self.psi is None:
            output = f"{self.label}: "
            output += f"{self.lower_prob} <= "
            output += f"P({self.phi}) <= "
            output += f"{self.upper_prob}"
            if self.tau:
                output += " ; True"
            return output
        else:
            output = f"{self.label}: "
            output += f"{self.lower_prob} <= "
            output += f"P({self.phi} | {self.psi}) <= "
            output += f"{self.upper_prob}"
            if self.tau:
                output += " ; True"

            return output

class LCN:
    """
    The Logical Credal Network (LCN) model.
    """

    def __init__(self):
        """
        Create an empty LCN model
        """
        self.atoms = {}
        self.sentences = {}
        self.independencies = None
    
    def add_atom(
            self, 
            atom: Atom
    ):
        """
        Add a new atom to the LCN model.

        Args:
            atom: Atom
                The atom to be added to the model.
        """
        
        if atom.name not in self.atoms:
            self.atoms[atom.name] = atom

    def add_atoms(
            self,
            atoms: List[Atom]
    ):
        """
        Add a set of atoms to the model.

        Args:
            atoms: list
                A list of atoms.
        """
        for a in atoms:
            self.add_atom(a)

    def add_sentence(
            self, 
            sentence: Sentence
    ):
        """
        Add a new sentence to the LCN model.

        Args:
            sentence: Sentence
                The sentence to be added to the model.
        """

        if sentence.label not in self.sentences:
            self.sentences[sentence.label] = sentence
            for k, atom in sentence.atoms.items():
                if k not in self.atoms:
                    self.atoms[k] = atom

    def add_sentences(
            self, 
            sentences: List[Sentence]
    ):
        """
        Add a set of sentences to the model.

        Args:
            sentences: list
                A list of sentences.
        """
        for s in sentences:
            self.add_sentence(s)

    def get_variables(self) -> List:
        """
        Returns a list of variable names.
        """
        return [k for k, _ in self.atoms.items()]

    def __str__(self):
        output = "LCN\n"
        for label in sorted(self.sentences.keys()):
            output += str(self.sentences[label]) + "\n"
        return output

    def build_primal_graph(
            self, 
            formula_labels: bool = False
    ) -> nx.DiGraph:
        """
        Construct the primal graph of the LCN. The primal graph is a directed
        graph with variable nodes and formula nodes. The variable nodes
        correspond to the LCN's variables/atoms while the formula nodes
        correspond to the formulas involved in the LCN sentences. There are 
        directed edges between the nodes as follows:
        
        Type (1) sentences:
            - from the variable nodes to the corresponding formula node `phi` 

        Type (2) sentences:
            - from the variable nodes to the corresponding formula node `psi`
            - from formula node `psi` to formula node `phi`
            - from formula node `phi` to its corresponding variable nodes
            - [tau=False] from the variable nodes to the formula node `phi`

        Args:
            formula_labels: bool
                Flag for adding labels to formula nodes in the graph.

        Returns:
            An nx.DiGraph object containing the primal graph. 
        """
    
        G = nx.DiGraph()

        # Create the variable/atom nodes
        for vid in sorted(self.atoms.keys()):
            G.add_node(
                node_for_adding=vid,
                color="blue", 
                type="atom", 
                shape="ellipse",
                label=vid
            )

        # Create the non-atomic formula nodes
        for sid, sentence in self.sentences.items():
            if sentence.type == SentenceType.Type1: # Type 1 sentence
                if not sentence.phi_formula.is_atomic():
                    G.add_node(
                        node_for_adding=sentence.phi_formula.label, 
                        color="green", 
                        type="formula", 
                        shape="rectangle",
                        label=sentence.phi if formula_labels else ""
                    )
            else: # Type 2 sentence
                if not sentence.psi_formula.is_atomic():
                    G.add_node(
                        node_for_adding=sentence.psi_formula.label,
                        color="green",
                        type="formula",
                        shape="rectangle",
                        label=sentence.psi if formula_labels else ""
                    )
                if not sentence.phi_formula.is_atomic():
                    G.add_node(
                        node_for_adding=sentence.phi_formula.label,
                        color="green",
                        type="formula",
                        shape="rectangle",
                        label=sentence.phi if formula_labels else ""
                    )

        # Create the directed edges
        for sid, sentence in self.sentences.items():
            if sentence.type == SentenceType.Type1: # Type 1 sentence P(phi)
                if not sentence.phi_formula.is_atomic():
                    for _, v in sentence.phi_formula.atoms.items():
                        G.add_edge(
                            v,
                            sid,
                            color="black"
                        )
            else: # Type 2 sentence P(phi|psi)
                # edges from atoms to psi
                if not sentence.psi_formula.is_atomic():
                    for _, v in sentence.psi_formula.atoms.items():
                        G.add_edge(
                            v,
                            sentence.psi_formula.label,
                            color="black"
                        )
                # edges from psi to phi
                G.add_edge(
                    sentence.psi_formula.label,
                    sentence.phi_formula.label,
                    color="red"
                )

                # edges from phi to atoms                
                if not sentence.phi_formula.is_atomic():
                    for _, v in sentence.phi_formula.atoms.items():
                        G.add_edge(
                            sentence.phi_formula.label,
                            v,
                            color="blue"
                        )
                    if sentence.tau == False:
                        for k, v in sentence.phi_formula.atoms.items():
                            G.add_edge(
                                v,
                                sentence.phi_formula.label,
                                color="blue"
                            )

        self.primal_graph = G
        return self.primal_graph
    
    def build_structure_graph(self):
        """
        Construct the structure of the LCN. Specifically, the structure is a
        mixed graph defined over the atoms only and containing both directed as
        well as undirected edges.

        1. For each sentence P(phi), add undirecte edges between any pair of
        atoms in phi.

        2. For each sentence P(phi|psi), add directed edges from each atom in psi
        to each atom in phi.

        3. Replace bidirected edges between any pair of atoms by an undirected edge.

        For now, we skip step 3 and use two bidirected edges to represent the
        undirected edges from step 1. 
        """

        G = nx.DiGraph()

        # Create the variable/atom nodes
        for vid in sorted(self.atoms.keys()):
            G.add_node(
                node_for_adding=vid,
                color="blue", 
                type="atom", 
                shape="ellipse",
                label=vid
            )

        # Loop over the Type 1 sentences
        for _, sentence in self.sentences.items():
            for _, u in sentence.phi_formula.atoms.items():
                for _, v in sentence.phi_formula.atoms.items():
                    if u != v:
                        G.add_edge(
                            u,
                            v,
                            type="undirected",
                            color="blue"
                        )
            if sentence.type == SentenceType.Type2: # Type 1 sentence
                for _, u in sentence.psi_formula.atoms.items():
                    for _, v in sentence.phi_formula.atoms.items():
                        G.add_edge(
                            u,
                            v,
                            type="directed",
                            color="red"
                        )
        
        self.structure_graph = G
        return self.structure_graph
    
    def lcn_parents(
            self, 
            atom: str
    ) -> List:
        """
        Identify the parents of an atom in the primal graph. Given an atom X,
        another atom Y is a parent of X if there exists a directed path P from
        Y to X in the primal graph such that all intermediate nodes other than
        X and Y (if any) on P are formula nodes.

        Args:
            atom: str
                The input atom X.

        Returns:
            A list of atoms that are X's parents in the primal graph.
        """

        assert self.primal_graph is not None, f"The primal graph must exist."

        X = atom
        parents = []
        for candidate, _ in self.atoms.items():
            if candidate == X:
                continue # skip same node

            # Retrieve all paths from `parent` to `X`
            paths = nx.all_simple_paths(
                G=self.primal_graph,
                source=candidate,
                target=X
            )

            # Check if there exists a path between `candidate` and `X` that
            # satisfies the property: all intermediate nodes are `formula` nodes
            for path in paths:
                path_found = True
                for Y in path:
                    if Y == candidate or Y == X:
                        continue # skip the ends of the path
                    node_type = nx.get_node_attributes(self.primal_graph, "type")
                    if node_type[Y] != "formula":
                        path_found = False
                        break
                if path_found:
                    parents.append(candidate)
                    break

        return parents

    def lcn_descendants(
            self,
            atom: str,
            parents: List[str]
    ) -> List:
        """
        Identify the descendants of an atom in the primal graph. Given an atom
        X, another atom Y is a descendant of X if there exists a path P in the
        primal graph from X to Y such that none of the intermediate nodes on P
        other than X and Y is a parent of X.

        Args:
            atom: str
                The input atom X.
            parents: List[str]
                The list of X's parents in the primal graph (if any)
        
        Returns:
            A list of atoms that are X's descendants in the primal graph.
        """

        X = atom
        descendants = []
        for candidate, _ in self.atoms.items():
            if candidate == X:
                continue # skip same node

            # Retrieve all paths from `X` to `candidate`
            paths = nx.all_simple_paths(
                G=self.primal_graph,
                source=X,
                target=candidate
            )

            # Check if there exists a path between `X` and `candidate` that
            # satisfies the property: none of the intermediate nodes is X's parent
            for path in paths:
                path_found = True
                for Y in path:
                    if Y == X:
                        continue # skip the end of the path
                    if Y in parents:
                        path_found = False
                        break
                if path_found:
                    descendants.append(candidate)
                    break

        return descendants

    def lcn_non_parents_descendants(
            self,
            atom: str,
            all_atoms: List[str],
            parents: List[str],
            descendants: List[str]
    ) -> List[str]:
        """
        Identify the non-parents non-descendants of an atom X in the primal graph.

        Args:
            atom: str
                The input atom X.
            all_atoms: List[str]
                The list of all atoms in the primal graph.
            parents: List[str]
                The parents of X in the primal graph.
            descendants: List[str]
                The descendants of X in the primal graph.

        Returns:
            A list containing the non-parent non-descendant atoms of X.
        """

        npnd = set([atom])
        npnd = npnd.union(set(parents), set(descendants))
        return list(set(all_atoms).difference(npnd))

    def local_markov_condition(self) -> Independencies:
        """
        Identify the independecies given by the Local Markov Condition (LMC).
        The independencies are encode by an Independencies object which
        essentially contains all independence assertions (X, Y, Z) that are
        given by the Local Markov Condition: every atom X is independent of
        its non-descendant non-parent atoms Y given its parents Z. 

        Returns:
            A list of tuples (X, Y, Z) representing indendencies: (X || Y | Z),
            namely variables X and conditionally independent of variables Y,
            given the variables Z. If Z is the empty set, then variables X and
            Y are marginally independent.
        """
        
        assert self.primal_graph is not None, "The primal graph must exist."

        self.independencies = Independencies()
        all_atoms = sorted([a for a, _ in self.atoms.items()])

        for X in all_atoms:
            Z = self.lcn_parents(atom=X)
            D = self.lcn_descendants(atom=X, parents=Z)
            Y = self.lcn_non_parents_descendants(
                atom=X, 
                all_atoms=all_atoms, 
                parents=Z, 
                descendants=D
            )

            # Check for non-empty set Y
            if len(Y) > 0:
                assertion = IndependenceAssertion(X, Y, Z)
                if self.independencies.contains(assertion) == False:
                    self.independencies.add_assertions([X, Y, Z])
        
        return self.independencies

    def global_markov_condition(self) -> List:
        """
        Identify the independecies given by the Global Markov Condition (GMC).

        Returns:
            A list of tuples (X, Y, Z) representing indendencies: (X || Y | Z),
            namely variables X and conditionally independent of variables Y,
            given the variables Z. If Z is the empty set, then variables X and
            Y are marginally independent.
        """
        raise NotImplementedError(f"The Global Markov Condition is not available.")

    def from_uai(
            self, 
            file_name: str,
            lmc: bool = True
    ):
        """
        Create an LCN instance from a UAI file format.

        Args:
            file_name: str
                Full path to the input file.
            lmc: bool
                Apply the Local Markov Condition (default True).
        
        Returns:
            True if successful, and False otherwise.
        """

        def read_tokens(f):
            """
            Retrive the tokens in a given file.

            Args:
                f: File
                    A handle to an open file.
            """
            for line in f:
                for token in line.split():
                    yield token

        try:
            with open(file_name, 'r') as f:
                fi = read_tokens(f)
                fi.__next__()
                num_vars = int(fi.__next__())
                domains = [int(fi.__next__()) for _ in range(num_vars)]
                num_funs = int(fi.__next__())
                atom2var = {}
                var2atom = {}
                for i in range(num_vars):
                    atom = f"x{i}"
                    atom2var[atom] = i
                    var2atom[i] = atom

                # Scopes
                scopes = []
                for _ in range(num_funs):
                    arity = int(fi.__next__())
                    scopes.append([int(fi.__next__()) for _ in range(arity)])

                # Parse sentences
                sid = 1 
                sentences = []
                for i in range(num_funs):
                    num_tuples = int(fi.__next__())
                    scope = scopes[i]
                    for _ in range(num_tuples):
                        config = [int(fi.__next__()) for _ in range(len(scope))]
                        lobo = float(fi.__next__())
                        upbo = float(fi.__next__())

                        # build the phi formula
                        child = scope[-1]
                        parents = [] if len(scope) <= 1 else scope[:-1]
                        phi_str = f"{var2atom[child]}" if config[-1] == 1 \
                            else f"!{var2atom[child]}"
                        
                        # buid the psi formula (if any)
                        psi_str = None
                        if len(parents) > 0:
                            par_atoms = [
                                var2atom[par] if config[j] == 1 else f"!{var2atom[par]}" \
                                for j, par in enumerate(parents)
                            ]

                            psi_str = par_atoms[0] if len(par_atoms) == 1 \
                                else "(" + " and ".join(par_atoms) + ")"
                            
                        label = f"s{sid}"

                        sen = Sentence(
                            label=label,
                            phi=phi_str,
                            psi=psi_str,
                            lower=lobo,
                            upper=upbo
                        )
                        sentences.append(sen)
                        sid = sid + 1

                f.close()
                self.add_sentences(sentences)
                print(f"Parsed UAI format with {num_vars} variables and {num_funs} sentences.")
                print(f"Build the LCN's primal graph.")
                self.build_primal_graph()
                print(f"Build the LCN's structure graph.")
                self.build_structure_graph()
                if lmc:
                    print(f"Build the LCN's independence assumptions (LMC).")
                    self.local_markov_condition()
                return True
            
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            return False

    def save_lcn(
            self,
            file_name: str
    ):
        """
        Save the LCN to a file (using the .lcn file format).

        Args:
            file_name: str
                Full path to the output file.
        """
        with open(file_name, "w") as f:
            f.write(f"# Saved LCN instance\n\n")
            for sid, s in self.sentences.items():
                f.write(f"{str(s)}\n")
            f.close()


    def from_lcn(
            self, 
            file_name: str,
            lmc: bool = True
    ):
        """
        Create an LCN instance from a LCN file format.

        Args:
            file_name: str
                Full path to the input file.
            lmc: bool
                Apply the Local Markov Condition (default True)

        Returns:
            True if successful, and False otherwise.
        """

        def parse_sentence(line: str):
            """
            Parse the sentence.

            Args:
                line: str
                    A string containing the sentence
            
            Returns:
                A Sentence object containing the parsed sentence.
            """

            # Check the sentence label
            label = line.split()[0]
            if label[-1] != ':':
                err_str = f"Syntax error in line: {line}\n"
                err_str += f" sentence label {label} cannot contain space and must terminate with :"
                raise ValueError(err_str)
            
            # Check if the flag tau is present
            tau = False
            if ";" in line:
                pos = line.find(";")
                tau = bool(line[pos+1:].strip())
                line = line[:pos]

            # Get the lower and upper bounds
            line = line.replace(label, '')
            tokens = line.split("<=")
            lowbo = float(tokens[0].strip())
            upbo = float(tokens[2].strip())
            
            # Get the sentence and check its syntax
            sentence = tokens[1].strip()
            if not sentence.startswith("P(") or sentence[-1] != ')':
                raise ValueError(f"Syntax error: a sentence must be given as P(...)")
            count = sentence.count("|")
            if count == 0: # Type 1 sentence
                phi_str = sentence[2:-1].strip()
                phi_str = phi_str.strip()
                psi_str = None
            elif count == 1: # Type 2 sentence
                pos = sentence.find("|")
                phi_str = sentence[2:pos]
                phi_str = phi_str.strip()
                psi_str = sentence[pos+1:-1]
                psi_str = psi_str.strip()
            else:
                raise ValueError(f"Syntax error: symbol | can only occur at most one time.")

            return Sentence(
                label=label[:-1],
                phi=phi_str,
                psi=psi_str,
                lower=lowbo,
                upper=upbo,
                tau=tau
            )
        
        try:
            f = open(file_name, "r")
            sentences = []
            for line in f:
                if line.strip().startswith("#") or len(line.strip()) == 0:
                    continue # skip comments and empty lines

                sen = parse_sentence(line.strip())
                sentences.append(sen)

            f.close()
            self.add_sentences(sentences)
            print(f"Parsed LCN format with {len(sentences)} sentences.")
            print(f"Build the LCN's primal graph.")
            self.build_primal_graph()
            print(f"Build the LCN's structure graph.")
            self.build_structure_graph()
            if lmc:
                print(f"Build the LCN's independence assumptions (LMC).")
                self.local_markov_condition()

            return True
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            return False


if __name__ == "__main__":

    # Load an LCN from a file
    file_name = "/home/radu/git/fm-factual/examples/asia.lcn"
    l = LCN()
    l.from_lcn(file_name=file_name)

    # Print the LCN content
    print(l)

    # Build the parents and descendants for each atom
    atoms = sorted([a for a, _ in l.atoms.items()])
    parents = {}
    descendants = {}
    for X in atoms:
        p_list = l.lcn_parents(atom=X)
        d_list = l.lcn_descendants(atom=X, parents=p_list)
        parents[X] = p_list
        descendants[X] = d_list
    
    # Print the results
    print("[Parents and Descendants]")
    for X in atoms:
        print(f"    Parents of {X}: {parents[X]}")
        print(f"Descendants of {X}: {descendants[X]}")
    
    # Local Markov Condition
    print("[Local Markov Condition: independencies]")
    for indep in l.independencies.get_assertions():
        print(f"{indep}")

    print("Done.")
