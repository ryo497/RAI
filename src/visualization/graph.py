import matplotlib.pyplot as plt
import networkx as nx

# 有向グラフの描画
def display_graph_info(G):
    # ノードのリストを表示
    print("Nodes in the graph:")
    print(G.nodes())

    # エッジのリストを表示
    print("\nEdges in the graph:")
    print(G.edges())

    # 各ノードの隣接ノードを表示
    print("\nNeighbors of each node:")
    if nx.is_directed(G):
        for node in G.nodes():
            print(f"{node}: {list(G.successors(node))}")

    pos = nx.spring_layout(G)  # ノードの配置を計算
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=12, font_weight='bold', arrowstyle='-|>', arrowsize=15)
    plt.title("Directed Graph")
    plt.show()
