import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from visualization.graph import display_graph_info
from CI_testing.CMI import conditional_mutual_information
from engine.indicator.bayes_factor import bayes_factor
from CI_testing.g_testing import g2_test
from CI_testing.chi2 import chi2_test
import networkx as nx
from itertools import combinations

def rai_algorithm(Nz, gs, gex, gall, go, data):
    print("rai_algorithm")
    gall = gs.copy()
    if all(len(gall[node]) <= Nz for node in gs):
        go.add_nodes_from(gs.nodes)
        if len(gs.edges) != 0:
            go.add_edges_from(gs.edges(data=True))
        print(f"return {go}")
        return go
    for node_y in gs.nodes:
        for node_x in gex:
            print("conditional_independence_test_1")
            if conditional_independence_test(node_x, node_y, [], data, test="bayes_factor"):
                gall[node_y].remove(node_x)
    apply_orientation_rules(gs, data)
    for node_y in gs.nodes:
        neighbors = list(gs.neighbors(node_y))
        for node_x in neighbors:
            Z = common_neighbors(gs, node_x, node_y)
            if Z != set():
                Z = list(Z)
                if gs.has_edge(node_x, node_y) or gall.has_edge(node_x, node_y):
                    print("conditional_independence_test_2")
                    if conditional_independence_test(node_x, node_y, Z, data, test="bayes_factor"):
                        if gall.has_edge(node_x, node_y):
                            gall.remove_edge(node_x, node_y)
                        if gs.has_edge(node_x, node_y):
                            gs.remove_edge(node_x, node_y)
    apply_orientation_rules(gs, data)
    print("pre_group_lowest_ropological_order_nodes")
    # display_graph_info(gs)
    gc, sub_structures = group_lowest_ropological_order_nodes(gs)
    for sub in sub_structures:
        output = rai_algorithm(Nz + 1, sub, gex, gall, go, data)
        go.add_nodes_from(output.nodes(data=True))
        go.add_edges_from(output.edges(data=True))
    gex = gex | set(sub_structures)
    # return rai_algorithm(Nz + 1, gc, gex, gall, data)
    print("last")
    output = rai_algorithm(Nz + 1, gc, gex, gall, go, data)
    go.add_nodes_from(output.nodes(data=True))
    if output.edges:
        go.add_edges_from(output.edges(data=True))
    print(f"return {go}")
    return go

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


def apply_orientation_rules(gs, data, alpha=0.05):
    for (a, b), (a, c) in combinations(gs.edges(), 2):
        if b != c and not gs.has_edge(b, c) and nx.has_path(gs, b, c):
            gs.add_edge(a, b)
            gs.add_edge(a, c)
    return gs
    # # V-structureの検出と方向付け
    # for node in gs.nodes:
    #     predecessors = list(gs.predecessors(node))
    #     for i in range(len(predecessors)):
    #         for j in range(i + 1, len(predecessors)):
    #             if not gs.has_edge(predecessors[i], predecessors[j]) and not gs.has_edge(predecessors[j], predecessors[i]):
    #                 if not conditional_independence_test(data, predecessors[i], predecessors[j], [node], alpha, test="chi2"):
    #                     gs.add_edge(predecessors[i], node)
    #                     gs.add_edge(predecessors[j], node)

    # # 残りの無向エッジの方向付け
    # for edge in list(gs.edges):
    #     if not gs.has_edge(edge[1], edge[0]):
    #         if gs.in_degree(edge[0]) == 0:
    #             gs.add_edge(edge[1], edge[0])
    #         elif gs.in_degree(edge[1]) == 0:
    #             gs.add_edge(edge[0], edge[1])
