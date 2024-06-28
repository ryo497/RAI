import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from visualization.graph import display_graph_info
from CI_testing.CMI import conditional_mutual_information
from engine.indicator.bayes_factor import bayes_factor
from CI_testing.g_testing import g2_test
from CI_testing.chi2 import chi2_test
import networkx as nx
from networkx import Graph, DiGraph
from engine.strucure.Sub_strucures import SubStructures
from itertools import combinations
from collections import defaultdict


class CI_testing():
    def __init__(self):
        self.result = defaultdict(set)
        pass


    def calc(self, X, Y, Z, data, test="bayes_factor", alpha=0.05):
        is_dependent = conditional_independence_test(X, Y, Z, data, test=test, alpha=alpha)
        self.save_result(self, is_dependent, X, Y)
        return is_dependent


    def save_result(self, is_dependent, X, Y, Z):
        if is_dependent:
            self.result[frozenset([X,Y])] |= set(Z)


    def get_condition_set(self, X, Y):
        return self.result[frozenset([X,Y])]


def rai_algorithm(Nz, gs, gex, gall, go, data):
    """_summary_

    Args:
        Nz (int): CIテストの字数
        gs (Graph): 入力無向グラフ
        gall[node] --> dict: 入力無向グラフのノードの隣接ノード {node: {metadata}}
        gex (set of Graph): _description_
        gall (Graph): _description_
        go (_type_): _description_
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    # gall = gs.copy()
    #  display_graph_info(gs)
    # for node in gs:
    #     print(len(gs[node]))
    print("A")
    if all(len(gs[node]) <= Nz for node in gs):
        go.add_nodes_from(gs.nodes(data=True))
        for node in gs.nodes:
            go.add_edges_from(gall.edges(node, data=True))
        return go, gs
    
    """_summary_
    1. For every node Y in Gstart and its parent X in Gex, if ∃S ⊂ {Pap(Y,Gstart) ∪
    Pa(Y,Gex)\X} and |S| = n such that X ⊥⊥ Y |S, then remove the edge between
    X and Y from Gall. とても大事なステップ
    2. Direct the edges in Gstart using orientation rules.
    """
    CI_Test = CI_testing()
    # display_graph_info(gs)
    print("B1")
    for node_y in gs.nodes:
        for node_x in gex.nodes:
            print("B1 processing")
            Z = gall.neighbors(node_x)
            if Z != set():
                Z = list(Z)
            if CI_Test.calc(node_x, node_y, Z, data, test="bayes_factor"):
                print("$#########")
                gall.remove_edge(node_y, node_x)
    # gs = apply_orientation_rules(gs, data, CI_Test)
    print("B2")
    for node_y in gs.nodes:
        neighbors = list(gs.neighbors(node_y))
        for node_x in neighbors:
            print("B2 processing")
            if Nz == 0:
                Z = []
                if CI_Test.calc(node_x, node_y, Z, data, test="bayes_factor"):
                    print("#########")
                    if gall.has_edge(node_x, node_y):
                        gall.remove_edge(node_x, node_y)
                    if gs.has_edge(node_x, node_y):
                        gs.remove_edge(node_x, node_y)
            else:
                set_Pa = gex.parents(node_y) | set(neighbors) - {node_x}
                num_Pa = len(set_Pa)
                if num_Pa >= Nz:
                    for Z in combinations(set_Pa, Nz):
                        if gs.has_edge(node_x, node_y) or gall.has_edge(node_x, node_y):
                            if conditional_independence_test(node_x, node_y, Z, data, test="bayes_factor"):
                                print("########")
                                if gall.has_edge(node_x, node_y):
                                    gall.remove_edge(node_x, node_y)
                                if gs.has_edge(node_x, node_y):
                                    gs.remove_edge(node_x, node_y)
    # display_graph_info(gs)
    gs = apply_orientation_rules(gs, data, CI_Test)
    gall = apply_orientation_rules(gall, data, CI_Test)
    print("C1")
    # display_graph_info(gs)
    # display_graph_info(gall)
    # display_graph_info(gs)
    # display_graph_info(gs)
    cls_sub_structures = SubStructures(gs, gex)
    # gc, sub_structures =
    gc, sub_stractures, gex = cls_sub_structures.extract_sub_structures()
    for sub in sub_stractures:
        print(f"substructures: {sub}")
        output, _ = rai_algorithm(Nz + 1, sub, gex, gall, go, data)
        print(f"substructures_rai {sub} done")
        for node in list(output.nodes):
            go.add_node(node)
            go.add_edges_from(gall.edges(node, data=True))
    # gex = gex | set(gex.sub_structures)
    # return rai_algorithm(Nz + 1, gc, gex, gall, data)
    # display_graph_info(gc)
    print("C2")
    output, _ = rai_algorithm(Nz + 1, gc, gex, gall, nx.DiGraph(), data)
    print("gc rai done")
    # display_graph_info(output)
    for node in output:
        go.add_node(node)
        go.add_edges_from(gall.edges(data=True))
    return go, gs

def conditional_independence_test(X, Y, Z, data, test="chi2", alpha=0.05):
    # ここに適切な条件付き独立性テストを実装
    if test == "chi2":
        chi2, p = chi2_test(X, Y, Z, data)
        return p > alpha
    elif test == "g2":
        g2, p = g2_test(X, Y, Z, data)
        return p > alpha
    elif test == "cmi":
        cmi = conditional_mutual_information(X,Y,Z, data)
        return cmi < alpha
    elif test == "bayes_factor":
        bf = bayes_factor(X,Y,Z, data)
        return bf < alpha

def find_substructures(graph):
    # ここに適切な部分構造探索を実装
    substructures = []
    visited = set()
    for node in graph:
        if node not in visited:
            subgraph = set()
            dfs(node, subgraph, visited, graph)
            substructures.append(subgraph)
    return substructures


def group_lowest_ropological_order_nodes(graph):
    print("group_lowest_ropological_order_nodes")
    if len(graph) == 1:
        return graph, []
    remove_cycles(graph)
    try:
        topological_order = list(nx.topological_sort(graph))
    except nx.NetworkXUnfeasible:
        print("The graph contains a cycle.")
        return None, []
    lowest_order_nodes = topological_order[:1]
    gc = graph.subgraph(lowest_order_nodes).copy()
    graph.remove_nodes_from(lowest_order_nodes)
    sub_structures = [graph.subgraph(sub).copy() for sub in nx.connected_components(graph.to_undirected())]
    complete_sub_structures = []
    for sub in sub_structures:
        complete_sub = nx.DiGraph(sub) # 無向グラフに変換
        for node in sub.nodes:
            complete_sub.add_edges_from([(node, sub_node) for sub_node in sub.nodes if node != sub_node])
        complete_sub_structures.append(complete_sub)
    # display_graph_info(graph)
    graph.add_nodes_from(lowest_order_nodes)

    return gc, complete_sub_structures


def remove_cycles(gs):
    try:
        cycles = list(nx.find_cycle(gs, orientation='ignore'))
        for cycle in cycles:
            gs.remove_edge(cycle[0], cycle[1])
    except nx.exception.NetworkXNoCycle:
        pass


def dfs(node, subgraph, visited, graph):
    visited.add(node)
    subgraph.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(neighbor, subgraph, visited, graph)


def common_neighbors(G, node_x, node_y):
    neighbors_x = set(G.neighbors(node_x))
    neighbors_y = set(G.neighbors(node_y))
    return neighbors_x & neighbors_y


def apply_orientation_rules(G, data, CI_Test):
    DG = nx.DiGraph()
    DG.add_nodes_from(G.nodes())
    # V-structureの検出と方向付け
    for (a, b), (a, c) in combinations(G.edges(), 2):
        if b != c and not G.has_edge(b, c) and nx.has_path(G, b, c):
            if a in CI_Test.result[frozenset([b,c])]:
                DG.add_edge(a, b)
                DG.add_edge(a, c)
    # display_graph_info(DG)

    # 残りの無向エッジの方向付け
    undirected_edges = [(u, v) for (u, v) in nx.DiGraph(G).edges() if not DG.has_edge(u, v) and not DG.has_edge(v, u)]
    for u, v in undirected_edges:
        if not DG.has_edge(u, v) and not DG.has_edge(v, u):
            if not nx.has_path(DG, v, u):
                DG.add_edge(u, v)
            else:
                DG.add_edge(v, u)
    # display_graph_info(DG)
    return DG
    # for edge in G.edges():
    #     if not DG.has_edge(edge[0], edge[1]) and not DG.has_edge(edge[1], edge[0]):
    #         if DG.in_degree(edge[0]) == 0:
    #             DG.add_edge(edge[1], edge[0])
    #         elif DG.in_degree(edge[1]) == 0:
    #             DG.add_edge(edge[0], edge[1])
    # return DG
