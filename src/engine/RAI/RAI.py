import networkx as nx
from networkx import DiGraph
from visualization.graph import display_graph_info
from collections import defaultdict
from engine.strucure.Sub_strucures import Gex
from engine.network.model import create_initial_structure
from engine.RAI.algorithm import rai_algorithm
from pgmpy.estimators import HillClimbSearch

class RAI:
    def __init__(self):
        self.gex = Gex()
        self.go = nx.DiGraph()
        self.Nz = 0

    def learn_bayesian_network(self):
        learned_structure, _ = rai_algorithm(self.Nz, self.gs, self.gex, self.gall, self.go, self.data)
        return learned_structure

    def estimate(self,data):
        self.data = data
        self.gs = create_initial_structure(data)
        self.gall = self.gs.copy()
        learned_structure = self.learn_bayesian_network()
        print(nx.is_directed_acyclic_graph(learned_structure))
        display_graph_info(learned_structure)
        network = HillClimbSearch(self.data)
        best_network = network.estimate()
        network = nx.DiGraph()
        node = best_network.nodes()
        edge = best_network.edges()
        network.add_nodes_from(node)
        network.add_edges_from(edge)
        # distance = hamming_distance(learned_structure, network)
        # print(distance)
