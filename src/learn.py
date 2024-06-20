import pandas as pd
import numpy as np
import preprocess.utils as utils
import networkx as nx
from engine.network.model import create_initial_structure
from visualization.graph import display_graph_info
from CI_testing import chi2, g_testing
from engine.indicator import CMI, bayes_factor
from engine.RAI.algorithm import rai_algorithm


def learn_bayesian_network(data):
    # 初期設定
    gs = create_initial_structure(data)
    gex = set()
    gall = gs.copy()
    go = nx.DiGraph()
    Nz = 0
    # RAIアルゴリズムの実行
    learned_structure = rai_algorithm(Nz, gs, gex, gall, go, data)
    return learned_structure

# サンプルデータの読み込みと前処理
def main():
    data_type = "asia"
    data = utils.load_data(data_type)
    learned_structure = learn_bayesian_network(data)
    display_graph_info(learned_structure)
    # model = BayesianModel()
    # for node, parents in learned_structure.items():
    #     model.add_edges_from([(parent, node) for parent in parents])
    # print(model)
    # hoge


if __name__ == "__main__":
    main()
