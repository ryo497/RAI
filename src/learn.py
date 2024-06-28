import pandas as pd
import numpy as np
import preprocess.utils as utils
import networkx as nx
from pgmpy.estimators import HillClimbSearch, K2Score
from engine.network.model import create_initial_structure
from visualization.graph import display_graph_info
from engine.strucure.Sub_strucures import Gex
from CI_testing import chi2, g_testing
from engine.indicator import CMI, bayes_factor
from engine.RAI.algorithm import rai_algorithm


def learn_bayesian_network(data):
    # 初期設定
    gs = create_initial_structure(data)
    gex = Gex()
    gall = gs.copy()
    go = nx.DiGraph()
    Nz = 0
    # RAIアルゴリズムの実行
    learned_structure, _ = rai_algorithm(Nz, gs, gex, gall, go, data)
    return learned_structure


def hamming_distance(G1, G2):
    """
    2つの有向グラフ G1 と G2 のハミング距離を計算する関数。
    """
    # グラフ G1 のエッジ集合
    edges_G1 = set(G1.edges())
    
    # グラフ G2 のエッジ集合
    edges_G2 = set(G2.edges())
    
    # それぞれのグラフに存在するエッジの対称差集合
    diff_edges = edges_G1.symmetric_difference(edges_G2)
    
    # ハミング距離は対称差集合のサイズ
    distance = len(diff_edges)
    
    return distance


# サンプルデータの読み込みと前処理
def main():
    data_type = "asia"
    data = utils.load_data(data_type)
    learned_structure = learn_bayesian_network(data)
    print(nx.is_directed_acyclic_graph(learned_structure))
    display_graph_info(learned_structure)
    network = HillClimbSearch(data)
    best_network = network.estimate()
    network = nx.DiGraph()
    node = best_network.nodes()
    edge = best_network.edges()
    network.add_nodes_from(node)
    network.add_edges_from(edge)
    distance = hamming_distance(learned_structure, network)
    print(distance)


if __name__ == "__main__":
    main()
