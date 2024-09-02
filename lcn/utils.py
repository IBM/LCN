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

# LCN utils

import numpy as np
import random

from networkx.classes.graph import Graph

# Set the random seed globally
def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)

# TODO: Revise the MixedGraph class to represent LCN structures

class MixedGraph(Graph):
    node_dict_factory = dict
    adjlist_dict_factory = dict
    edge_attr_dict_factory = dict
    def __init__(self):

        self.node_dict_factory = ndf = self.node_dict_factory
        self.adjlist_dict_factory = self.adjlist_dict_factory
        self.edge_attr_dict_factory = self.edge_attr_dict_factory
        self.graph = {}
        self.node = ndf()
        self.adj = ndf()
        self.pred = ndf()
        self.succ = ndf()
        self.edge=self.adj

    def add_node(self, n, **attr):
        if n not in self.node:
            self.adj[n] = self.adjlist_dict_factory()
            self.node[n] = attr

    def add_undirected_edge(self, u, v, attr_dict=None, **attr):
        if attr_dict is None:
            attr_dict=attr
        if u not in self.adj:
            self.adj[u] = self.adjlist_dict_factory()
            self.node[u] = {}
        if v not in self.adj:
            self.adj[v] = self.adjlist_dict_factory()
            self.node[v] = {}
        datadict = self.adj[u].get(v, self.edge_attr_dict_factory())
        datadict.update(attr_dict)
        self.adj[u][v] = datadict
        self.adj[v][u] = datadict

    def add_directed_edge(self, u, v, attr_dict=None, **attr):
        if attr_dict is None:
            attr_dict=attr
        if u not in self.succ:
            self.succ[u]= self.adjlist_dict_factory()
            self.pred[u]= self.adjlist_dict_factory()
            self.node[u] = {}
        if v not in self.succ:
            self.succ[v]= self.adjlist_dict_factory()
            self.pred[v]= self.adjlist_dict_factory()
            self.node[v] = {}
        datadict=self.succ[u].get(v,self.edge_attr_dict_factory())
        datadict.update(attr_dict)
        self.succ[u][v]=datadict
        self.pred[v][u]=datadict