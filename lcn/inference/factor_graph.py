# The entry-point script
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

# The Factor Graph associated with an LCN model

import networkx as nx
from typing import List, Dict

# Local
from lcn.model import Atom, Sentence, LCN


class VariableNode:
    """
    A `variable node` in the factor graph. The variable node is associate with
    an atom (or variable) in the LCN model.
    """
    def __init__(
            self, 
            a: Atom
    ):
        """
        Variable node constructor.

        Args:
            a: Atom
                The atom associated with the variable node.
        """
        self.name = a.name
        self.atom = a

    def get_name(self):
        """
        Returns the name associated with the node's atom.
        """
        return self.name

    def __str__(self):
        return str(self.atom)

class FactorNode:
    """
    A `factor node` in the factor graph. The factor node contains the set of 
    sentences that are defined over the same set of atoms i.e., scope.
    """
    def __init__(
            self, 
            label: str
    ) -> None:
        """
        FactorNode constructor.

        Args:
            label: str
                A unique label associated with the factor node 
        """
        self.label = label
        self.scope = set()  # scope of the factor i.e., set of atom names
        self.sentences = {} # a dict containing the sentences of the factor node

    def get_label(self):
        return self.label

    def add_sentence(
            self, 
            s: Sentence
    ) -> None:
        """
        Add a new sentence to the factor node.

        Args:
            s: Sentence
                An LCN sentence object.
        """
        self.sentences[s.label] = s
        temp = [aid for aid, _ in s.atoms.items()]
        self.scope.update(temp)
    
    def is_scope_equal(
            self, 
            s: dict
    ) -> bool:
        """
        Check if the current factor node includes the input scope.

        Args:
            s: dict
                A dictionary containing the atoms of the input scope.

        Returns:
            True if the factor node's scope contains the input scope, else False.
        """
        if len(self.scope) == len(s) and all(elem in self.scope for elem in s.keys()):
            return True
        else:
            return False 

    def __str__(self):
        output = f"FactorNode: {self.label}\n"
        output += f" scope: {self.scope}\n"
        output += f" sentences:\n"
        for _, s in self.sentences.items():
            output += "   " + str(s) + "\n"
        return output
    

class FactorGraphEdge:
    """
    An edge in the factor graph connecting a variable node to a factor node.
    """
    def __init__(
            self, 
            variable_node: VariableNode, 
            factor_node: FactorNode
    ) -> None:
        """
        Create the edge between a variable node and a factor node.

        Args:
            variable_node: VariableNode
                The input variable node.
            factor_node: FactorNode
                The input factor node.
        """
        self.variable_node = variable_node
        self.factor_node = factor_node
    
    def __str__(self):
        return str(self.variable_node) + f" -- {self.factor_node.label}"

class FactorGraph:
    """
    The factor graph of an LCN model. It is a bi-partite graph with variable 
    nodes (corresponding to the LCN's atoms) and factor nodes (corresponding
    to sets of LCN sentences defined over exactly the same set of atoms). A
    variable node is connected by an edge to a factor node if the corresponding
    atom is in the scope of a sentence in the factor node.
    """

    def __init__(
            self, 
            lcn: LCN
    ):
        """
        Create the factor graph of the input LCN model.

        Args:
            lcn: LCN
                The input LCN model.
        """

        self.lcn = lcn
        self.variable_nodes = {} # dict {str: VariableNode}
        self.factor_nodes = {} # dict {str: FactorNode}
        self.edges = []
        self.variable_node_neighbors = {} # dict {str: List[FactorNode]}
        self.factor_node_neighbors = {} # dict {str: VariableNode}

        atoms = lcn.atoms  # a dict of atoms {a: Atom}
        sentences = lcn.sentences # a dict of sentences {s: Sentence}

        # Create the variable nodes
        self._make_variable_nodes(
            atoms=[a for _, a in atoms.items()]
        )

        # Create the factor nodes
        self._make_factor_nodes(
            sentences=[s for _, s in sentences.items()]
        )

        # Create the edges
        self._make_edges()

    def _make_variable_nodes(
            self,
            atoms: List[Atom]
    ) -> None:
        """
        Create the variable nodes of the factor graph.

        Args:
            atoms: List[Atom]
                The input list of atoms.
        """
        self.variable_nodes = {a.name:VariableNode(a) for a in atoms}

    def _make_factor_nodes(
            self, 
            sentences: List[Sentence]
    ) -> None:
        """
        Create the factor nodes of the factor graph. Group the sentences from the
        list into subgroups such that all sentences in a subgroup span the
        same set of atoms/variables.

        Args:
            sentences: List[Sentence]
                The input list of sentences.
        """

        # Sort sentences in decreasing order of their scope sizes
        temp = sorted(sentences, key=lambda x: len(x.atoms), reverse=True)
        
        # Group sentences in corresponding factors so that all sentences in a
        # factor span the same atoms in their scopes.
        index = 1
        f = FactorNode(label=f"f{index}")
        f.add_sentence(temp.pop(0))
        self.factor_nodes[f.label] = f
        while len(temp) > 0:
            s = temp.pop(0)
            found = False
            for _, nf in self.factor_nodes.items():
                if nf.is_scope_equal(s.atoms):
                    nf.add_sentence(s)
                    found = True
                    break
            if not found:
                f = FactorNode(label=f"f{index}")
                index += 1
                f.add_sentence(s)
                self.factor_nodes[f.label] = f
    # --            

    def _make_edges(self):
        """
        Create the edges of factor graph by connecting each of the variable
        nodes to the corresponding factor nodes.
        """
        
        # Create the edges
        for _, fn in self.factor_nodes.items():
            for a in fn.scope:
                vn = self.variable_nodes[a]
                self.edges.append(FactorGraphEdge(variable_node=vn, factor_node=fn))

        # Create node adjacencies (i.e., neighbors of nodes are factors)
        for vid, vn in self.variable_nodes.items():
            self.variable_node_neighbors[vid] = []
            for e in self.edges:
                if e.variable_node.get_name() == vid:
                    self.variable_node_neighbors[vid].append(e.factor_node)
        
        # Create factor adjacencies (i.e., neighbors of factors are nodes)
        for fid, fn in self.factor_nodes.items():
            self.factor_node_neighbors[fid] = []
            for e in self.edges:
                if e.factor_node.get_label() == fid:
                    self.factor_node_neighbors[fid].append(e.variable_node)

    def build_graph(self):
        """
        Create an undirected graph representing the factor graph.
        """

        G = nx.Graph()

        # Create the variable nodes
        for vid in sorted(self.variable_nodes.keys()):
            G.add_node(
                node_for_adding=vid,
                color="blue", 
                shape="ellipse",
                label=vid
            )

        # Create the factor nodes
        for fid in sorted(self.factor_nodes.keys()):
            G.add_node(
                node_for_adding=fid,
                color="green", 
                shape="rectangle",
                label=fid
            )
            
        # Create the edges
        for e in self.edges:
            G.add_edge(
                u_of_edge=e.variable_node.get_name(),
                v_of_edge=e.factor_node.get_label(),
                color="black"
            )

        self.bipartite_graph = G
        return self.bipartite_graph

    def add_evidence(self, evidence: Dict):
        """
        Update the factor graph by adding additional sentences corresponding
        to the evidence, i.e., if X = 1, add constraint P(X) = 1 to a factor
        node that contains that variable. If there is a factor node that already 
        contains P(X) then replace the old constraint with the new one. Otherwise,
        just add the new constraint to the corresponding factor node. If no such
        factor node exists then create a new factor node to accomodate the
        new constraint.
        """

        num_ev = 0
        for var, val in evidence.items():
            num_ev += 1
            phi = f"!{var}" if val == 0 else f"{var}"
            label = f"ev{num_ev}"
            sen = Sentence(
                label=label,
                phi=phi,
                lower=1.,
                upper=1.
            )

            found = False
            for _, fn in self.factor_nodes.items():
                if len(fn.scope) == 1 and list(fn.scope)[0] == var: # found candidate factor
                    temp = {}
                    for sid, s in fn.sentences.items():
                        if (s.is_unary_positive() and sen.is_unary_positive()) or \
                            (s.is_unary_negative() and sen.is_unary_negative()):
                            continue
                        else:
                            temp[sid] = s

                    temp[label] = sen
                    fn.sentences = temp
                    found = True
                    break
                
            if not found: # no candidate factor found
                # create a new factor node
                fid = f"f{len(self.factor_nodes) + 1}"
                fn = FactorNode(label=fid)
                fn.add_sentence(sen)
                self.factor_nodes[fn.label] = fn
                
                # add the node to the graph and create relevant edges
                vn = self.variable_nodes[var]
                self.edges.append(FactorGraphEdge(variable_node=vn, factor_node=fn))

                # update the node neighbors lists
                if var not in self.variable_node_neighbors:
                    self.variable_node_neighbors[var] = []
                self.variable_node_neighbors[var].append(fn)
                if fid not in self.factor_node_neighbors:
                    self.factor_node_neighbors[fid] = []
                self.factor_node_neighbors[fid].append(vn)


    def __str__(self):
        output = "Factor Graph\n"
        output += f"# variable nodes: {len(self.variable_nodes)}\n"
        output += f"# factor nodes: {len(self.factor_nodes)}\n"
        output += f"variable nodes: {set(self.variable_nodes.keys())}\n"
        output += "factor nodes:\n"
        for _, fn in self.factor_nodes.items():
            output += str(fn) + "\n"
        output += "edges:\n"
        for e in self.edges:
            output += str(e) + "\n"
        return output

if __name__ == "__main__":

    file_name = "/home/radu/git/fm-factual/examples/lcn/alarm.lcn"
    l = LCN()
    l.from_lcn(file_name=file_name)
    print(l)

    fg = FactorGraph(lcn=l)
    print(fg)
    
    evidence = {'A': 1, 'E': 0}
    fg.add_evidence(evidence)
    print(fg)

    print("Done.")
