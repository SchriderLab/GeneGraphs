import os

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model, save_model

import plot_utils as pu
import streaming_data as sd


def load_data():

    """h5filelist = [
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/node_embeddings/10e4_test_infer_FINAL_multi_pulse_uni_BA.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/node_embeddings/10e4_test_infer_FINAL_multi_pulse_bi.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/node_embeddings/10e4_test_infer_FINAL_continuous_bi.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/node_embeddings/10e4_test_infer_FINAL_constant_2pop.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/node_embeddings/10e4_test_infer_FINAL_single_pulse_uni_AB.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/node_embeddings/10e4_test_infer_FINAL_single_pulse_uni_BA.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/node_embeddings/10e4_test_infer_FINAL_multi_pulse_uni_AB.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/node_embeddings/10e4_test_infer_FINAL_continuous_uni_BA.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/node_embeddings/10e4_test_infer_FINAL_single_pulse_bi.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/node_embeddings/10e4_test_infer_FINAL_continuous_uni_AB.hdf5",
    ]"""
    h5filelist = [
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/testout/hdf5files/testout_multi_pulse_uni_BA.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/testout/hdf5files/testout_multi_pulse_bi.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/testout/hdf5files/testout_continuous_bi.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/testout/hdf5files/testout_constant_2pop.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/testout/hdf5files/testout_single_pulse_uni_AB.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/testout/hdf5files/testout_single_pulse_uni_BA.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/testout/hdf5files/testout_multi_pulse_uni_AB.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/testout/hdf5files/testout_continuous_uni_BA.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/testout/hdf5files/testout_single_pulse_bi.hdf5",
        "/overflow/dschridelab/projects/GeneGraphs/embeddings/testout/hdf5files/testout_continuous_uni_AB.hdf5",
    ]

    samps = []
    labs = []
    lab_dict = {}
    for i in range(len(h5filelist)):
        raw_h5_data = h5py.File(h5filelist[i], "r")
        h5_data = raw_h5_data[list(raw_h5_data.keys())[0]]
        seqs = list(h5_data.keys())
        lab_dict[h5filelist[i].split("/")[-1].split(".")[0]] = i
        for j in seqs:
            samps.append(h5_data[j])
            labs.append(i)

    return lab_dict, samps, labs


def split_partitions(samps, labs):
    """
        Splits all data and labels into partitions for train/val/testing.

    Args:
        IDs (List): List of files used for training model
        labs (List): List of numeric labels for IDs

    Returns:
        Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]: Train/val/test splits of IDs and labs
    """
    X_train, X_val, y_train, y_val = train_test_split(
        samps, labs, stratify=labs, test_size=0.3
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, stratify=y_val, test_size=0.5
    )

    data_dict = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }

    return data_dict


def create_model(datadim=(100, 100, 128)):
    """
    Creates 1D convolutional model over tree sequences.

    Returns:
        Model: Keras compiled model.
    """
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        model_in = Input(datadim)
        h = Conv2D(64, 3, activation="relu", padding="same", name="conv1_1")(model_in)
        h = MaxPooling2D(pool_size=3, name="pool1", padding="same")(h)
        h = Dropout(0.15, name="drop1")(h)
        h = Conv2D(32, 3, activation="relu", padding="same", name="conv1_1")(model_in)
        h = MaxPooling2D(pool_size=3, name="pool1", padding="same")(h)
        h = Dropout(0.15, name="drop1")(h)
        h = Flatten(name="flatten1")(h)

        h = Dense(256, name="512dense", activation="relu")(h)
        h = Dropout(0.2, name="drop7")(h)

        h = Dense(128, name="last_dense", activation="relu")(h)
        h = Dropout(0.1, name="drop8")(h)
        output = Dense(10, name="out_dense", activation="softmax")(h)

        model = Model(inputs=[model_in], outputs=[output], name="Treenet")
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"],
        )

    return model


def fit_model(base_dir, model, train_gen, val_gen):
    """
    Fits a given model using training/validation data, plots history after done.

    Args:
        base_dir (str): Base directory where data is located, model will be saved here.
        model (Model): Compiled Keras model.
        train_gen: Training generator
        val_gen: Validation Generator

    Returns:
        Model: Fitted Keras model, ready to be used for accuracy characterization.
    """

    checkpoint = ModelCheckpoint(
        os.path.join(base_dir, "models", model.name),
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
    )

    earlystop = EarlyStopping(
        monitor="val_accuracy",
        min_delta=0.1,
        patience=10,
        verbose=1,
        mode="auto",
        restore_best_weights=True,
    )

    callbacks_list = [earlystop, checkpoint]

    history = model.fit(
        train_gen,
        epochs=40,
        verbose=1,
        callbacks=callbacks_list,
        validation_data=val_gen,
    )

    if not os.path.exists(os.path.join(base_dir, "images")):
        os.makedirs(os.path.join(base_dir, "images"))
    pu.plot_training(os.path.join(base_dir, "images"), history, model.name)

    # Won't checkpoint handle this?
    save_model(model, os.path.join(base_dir, "models", model.name))

    return model


def evaluate_model(
    model, test_gen, base_dir,
):
    """
    Evaluates model using confusion matrices and plots results.

    Args:
        model (Model): Fit Keras model.
        X_test (List[np.ndarray]): Testing data.
        Y_test (np.ndarray): Testing labels.
        base_dir (str): Base directory data is located in.
        time_series (bool): Whether data is time series or one sample per simulation.
    """

    pred = model.predict(test_gen)
    predictions = np.argmax(pred, axis=1)
    trues = test_gen.labels

    conf_mat = pu.print_confusion_matrix(trues, predictions)
    pu.plot_confusion_matrix(
        os.path.join(base_dir),
        conf_mat,
        [str(i) for i in range(10)],
        title=model.name,
        normalize=True,
    )
    pu.print_classification_report(trues, predictions)


def main():
    PADDING = 100
    N_CLASSES = 10

    lab_dict, samps, labs = load_data()

    data_dict = split_partitions(samps, labs)
    print(data_dict["X_train"][0])

    train_gen = sd.DataGenerator(
        data_dict["X_train"], data_dict["y_train"], PADDING, 32, N_CLASSES, True
    )
    val_gen = sd.DataGenerator(
        data_dict["X_val"], data_dict["y_val"], PADDING, 32, N_CLASSES, True
    )
    test_gen = sd.DataGenerator(
        data_dict["X_test"], data_dict["y_test"], PADDING, 32, N_CLASSES, False
    )

    model = create_model()
    model = fit_model(".", model, train_gen, val_gen)
    evaluate_model(model, test_gen, ".")


if __name__ == "__main__":
    main()
