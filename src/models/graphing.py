import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
from mpl_toolkits import mplot3d
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


def z_plotting_on_the_fly(data, y, dims=2, num_classes=2, reduction='PCA'):

    y_plot = []
    cdict = {0: 'red', 1: 'blue', 2: 'green'}

    for elem in y:
        y_plot.extend([elem.item()]*99)
    if reduction == 'TSNE':
        plotting = TSNE(n_components=dims)
        reduction = 't-SNE' # just changing it for nicer look when we print later
    else:
        plotting = PCA(n_components=dims)
    arr_fitted = plotting.fit_transform(data)

    if dims == 3:
        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection="3d")

    i = 0
    for x in range(num_classes):
        class_slice = arr_fitted[i:i+495, :]
        y_slice = y_plot[i:i+495]
        if dims == 3:
            ax.scatter3D(class_slice[:, 0], class_slice[:, 1], class_slice[:, 2], c=cdict[x], label="Model {}".format(x+1))
        else:
            plt.scatter(class_slice[:, 0], class_slice[:, 1], c=cdict[x], label="Model {}".format(x+1))
        i += 495

    plt.title("{} Demographic Models".format(reduction))
    plt.legend()
    plt.show()


# deprecated
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

