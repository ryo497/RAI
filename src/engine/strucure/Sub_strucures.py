from typing import Any
import networkx as nx
from networkx import DiGraph
from visualization.graph import display_graph_info
from collections import defaultdict


class Gex:
    def __init__(self):
        self.substructures = set()
        self.node2parents = defaultdict(set)
        self.nodes = set()
    

    def node(self):
        return self.nodes


    def add_Pa_data(self,
                    sccs,
                    scc_graph,
                    substructures,
                    topological_order) -> Any:
        for idx, scc in enumerate(sccs):
            for node in scc:
                parent_s = set()
                for parent_idx in scc_graph[idx]:
                    parent_s |= sccs[parent_idx]
                self.node2parents[node] |= parent_s
                # print(f"Node: {node}, Parents: {parent_s}")
        if self.nodes == set():
            for idx in topological_order[1:]:
                self.nodes |= set(sccs[idx])
        self.substructures |= set(substructures)
        return

    def parents(self, node):
        return self.node2parents[node]



class Node():
    def __init__(self, name):
        self.name = name
        self.parents = -1
        self.children = []

    def add_parent(self, parent):
        self.parents.append(parent)

    def add_child(self, child):
        self.children.append(child)


class SubStructures(Node):
    def __init__(self, digraph, gex):
        super().__init__(SubStructures)
        self.G = digraph
        self.gex = gex
        # self.sub_structures = self.extract_sub_structures()


    def copy_from_orig(
            self,
            node_list : list,
            G : DiGraph):
        go = DiGraph()
        for node in node_list:
            go.add_node(node)
            for edge in G.successors(node):
                go.add_edge(node, edge)
            for edge in G.predecessors(node):
                go.add_edge(edge, node)
        return go
    

    def extract_subgraph(self, nodes, G):
        sub = nx.Graph()
        sub.add_nodes_from(nodes)
        edges_to_add = [(node, sub_node) for node in nodes for sub_node in nodes if node != sub_node and G.has_edge(node, sub_node)]
        sub.add_edges_from(edges_to_add)
        return sub
    
    def extract_subDigraph(self, nodes, G):
        sub = G.subgraph(nodes).copy()
        edges_to_add = [(node, sub_node) for node in nodes for sub_node in nodes if node != sub_node and G.has_edge(node, sub_node)]
        sub.add_edges_from(edges_to_add)
        return sub


    def extract_sub_structures(self):
        G = self.G
        sccs = list(nx.strongly_connected_components(G))
        scc_graph = nx.condensation(G)
        self.scc_graph = scc_graph
        # display_graph_info(scc_graph)
        topological_order = list(nx.topological_sort(scc_graph))
        topological_lowest_nodes = sccs[topological_order[0]]
        gc = self.extract_subgraph(topological_lowest_nodes, G)
        SubStructures = []
        for sccs_idx in topological_order[1:]:
            sub = self.extract_subgraph(sccs[sccs_idx], G)
            SubStructures.append(sub)
        # self.substrucures = SubStructures
        self.gex.add_Pa_data(sccs, scc_graph, SubStructures, topological_order)
        return gc, SubStructures, self.gex

    def parents(self, node):
        sub = self.get_structure(self.sub_structures, node)

    def get_structure(self, node):
        for sub in self.sub_structures:
            if node in sub:
                return sub
        return None
