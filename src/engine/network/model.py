import networkx as nx

# def create_initial_structure(data):
#     nodes = data.columns.tolist()
#     gs = {node: set(nodes) - {node} for node in nodes}
#     return gs


def create_initial_structure(data):
    nodes = data.columns.tolist()
    gs = nx.Graph()
    gs.add_nodes_from(nodes)
    for noed in nodes:
        gs.add_edges_from([(node, noed) for node in nodes if node != noed])
    return gs
