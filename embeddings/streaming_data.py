# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow import keras


class DataGenerator(Sequence):
    def __init__(self, samples, labels, padding, batch_size, n_classes, shuffle):
        "Initialization"
        self.samples = samples
        self.labels = labels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.padding = padding
        self.batch_size = batch_size

        # If # samples is less than batch size, set batch size to whatever
        # is available.
        if len(self.samples) < self.batch_size:
            self.batch_size = len(self.samples)

        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.samples) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of indices of samples
        sub_samples = [self.samples[k] for k in indexes]
        sub_y = [self.labels[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(sub_samples, sub_y)

        return X, y

    def pad_nparr(self, arr, pad_size):
        if arr.shape[0] < pad_size:
            _pad = np.zeros((pad_size, arr.shape[1]))
            _pad[: arr.shape[0], :] = arr
            return _pad
        elif arr.shape[0] > pad_size:
            arr = arr[:pad_size, :]
            return arr
        else:
            return arr

    def pad_3d_nparr(self, arr, pad_size):
        if arr.shape[0] < pad_size:
            _pad = np.zeros((pad_size, arr.shape[1], arr.shape[2]))
            _pad[: arr.shape[0], :, :] = arr
            return _pad
        else:
            return arr

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.samples))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, sub_sample, sub_labs):
        "Generates data containing batch_size samples"
        X_list = []
        for j in sub_sample:
            tree_list = []
            for tree_id in list(j.keys()):
                tree_list.append(
                    self.pad_nparr(np.array(j[tree_id]["embedding"]), self.padding)
                )
            X_list.append(self.pad_3d_nparr(np.stack(tree_list), self.padding))

        X = np.stack(X_list)

        y = [int(k) for k in sub_labs]

        return (
            X,
            keras.utils.to_categorical(y, num_classes=self.n_classes),
        )
