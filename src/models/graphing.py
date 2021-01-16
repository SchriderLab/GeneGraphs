import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
import networkx as nx
import gmatch4py as gm # added

def graph_comparison(graph1, graph2):
    ged = gm.GraphEditDistance(1, 1, 1, 1)  # all edit costs are equal to 1
    result = ged.compare([graph1, graph2], None)
    return ged.distance(result)

def graph_visualization(graph1, graph2):
    pass
    # graphed1 = nx.spring_layout(graph1)
    # graphed2 = nx.spring_layout(graph2)
    # nx.draw(graphed1)
    # nx.draw(graphed2)
    # plt.show()

def z_plotting(data, type='pca'):
    arr = data['Z']
    if type == 'tsne':
        plotting = TSNE()
    else:
        plotting = PCA(n_components=2)
    arr_fitted = plotting.fit_transform(arr)
    if data['label'] == 0:
        plt.scatter(arr_fitted[:, 0], arr_fitted[:, 1], c='b')
    else:
        plt.scatter(arr_fitted[:, 0], arr_fitted[:, 1], marker='x', c='r')




def main():
    for i in range(0, 8):
        # file = '../../results_y/0000{}_000010.npz'.format(i + 1) # arbitrary slice of data
        for j in range(10, 40):
            file = '../../real_val_predict/0000{}_00000{}.npz'.format(j, i)
            data = np.load(file)
            z_plotting(data, type='tsne')
    blue = mpatches.Patch(color='blue', label='Model1')
    red = mpatches.Patch(color='red', label='Model2')
    plt.title("PCA Demographic Models")
    plt.legend(handles=[blue, red])
    plt.show()


if __name__ == "__main__":
    main()

