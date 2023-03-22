import networkx as nx
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt

dataset = KarateClub()
data = dataset[0]  # x=[34,34], edge_index=[2, 156]表示34个节点的source和target,156是边的个数, y=[34], train_mask=[34]表示有标签的数据的位置

def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False, node_color=color, cmap="Set2")
    plt.show()

def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap='Set2')
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()

if __name__ == '__main__':
    G = to_networkx(data, to_undirected=True)
    visualize_graph(G, color=data.y)


