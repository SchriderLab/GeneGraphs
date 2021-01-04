import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches


def pca_plotting(data):
    arr = data['Z']
    pca = PCA(n_components=2)
    arr_fitted = pca.fit_transform(arr)
    if data['label'] == 0:
        plt.scatter(arr_fitted[:, 0], arr_fitted[:, 1], c='b')
    else:
        plt.scatter(arr_fitted[:, 0], arr_fitted[:, 1], marker='x', c='r')


def main():
    for i in range(10, 60):
        # file = '../../results_y/0000{}_000010.npz'.format(i + 1) # arbitrary slice of data
        file = 'trashing_output/000000_0000{}.npz'.format(i)
        data = np.load(file)
        pca_plotting(data)
    blue = mpatches.Patch(color='blue', label='Model1')
    red = mpatches.Patch(color='red', label='Model2')
    plt.title("PCA Demographic Models")
    plt.legend(handles=[blue, red])
    plt.show()


if __name__ == "__main__":
    main()

