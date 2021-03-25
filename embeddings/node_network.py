import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import _LRScheduler
import h5py
from tqdm import tqdm

import plot_utils as pu


class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier.
    https://www.kaggle.com/purplejester/a-simple-lstm-based-time-series-classifier"""

    def __init__(self, input_dim=100, hidden_dim=128, layer_dim=3, output_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        self.rnn.flatten_parameters()
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t for t in (h0, c0)]
        # fmt: on


class CNNClassifier(nn.Module):
    def __init__(self):
        """
        Args:
            num_classes (int): size of the output prediction vector
            num_nodes (int): padded size of height of tensor'
            embedding_dim (int): number of embedded feature vectors output by embedding program

        Input data is a tensor with dims (#trees, nodes, embedding dims), which corresponds to (depth, height, width).
        """
        # fmt: off
        super(CNNClassifier, self).__init__()

        #Ignore this, it's for 3D conv
        #move_size = (3, num_nodes, embedding_dim) #We only want to convolve over the spatial dimension? Right? Wrong? Maybe not, would sorting work?

        self.conv1 = nn.Sequential(
            nn.Conv2d(100, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            #nn.BatchNorm1d(64),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0))

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            #nn.BatchNorm1d(16),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=(1, 1), stride=1, padding=0))

        self.fcblock = nn.Sequential(
            nn.Linear(11904, 512), 
            nn.ReLU(), 
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),

            nn.Linear(512, 10))
        # nn.ReLU(),
        # nn.Dropout(p=0.5),

        # nn.Linear(512, 64),
        # nn.ReLU(),
        # nn.Dropout(p=0.5),

        # nn.Linear(64, 2))
        # fmt: on

    def forward(self, tree_seq_tensor):
        """The forward pass of the classifier

        Args:
            tree_seq (tensor): Tensor of a tree sequences, should be dims (#trees, nodes, embedding dims)
        Returns:
            the resulting tensor. tensor.shape should be (batch, num_classes).
        """
        x = self.conv1(tree_seq_tensor)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fcblock(x)

        return x


def load_data():
    h5filelist = [
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


def split_data(samps, labs):

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


def pad_nparr(arr, pad_size):
    if arr.shape[0] < pad_size:
        _pad = np.zeros((pad_size, arr.shape[1]))
        _pad[: arr.shape[0], :] = arr
        return _pad
    else:
        return arr


def pad_3d_nparr(arr, pad_size):
    if arr.shape[0] < pad_size:
        _pad = np.zeros((pad_size, arr.shape[1], arr.shape[2]))
        _pad[: arr.shape[0], :, :] = arr
        return _pad
    else:
        return arr


def train_model(
    epoch,
    model,
    batch_size,
    PADDING,
    data_dict,
    criterion,
    optimizer,
    device,
):
    train_acc = 0.0
    val_acc = 0.0

    train_loss = 0.0
    val_loss = 0.0

    # Training
    permutation = torch.randperm(len(data_dict["X_train"]))
    model.train()
    train_steps = range(0, len(data_dict["X_train"]), batch_size)
    for i in tqdm(train_steps, desc="Training minibatches"):
        tr_inds = permutation[i : i + batch_size]
        tr_X_list = []
        for j in [data_dict["X_train"][k] for k in list(tr_inds)]:
            tree_list = []
            for tree_id in list(j.keys()):
                tree_list.append(pad_nparr(np.array(j[tree_id]["embedding"]), PADDING))
            tr_X_list.append(pad_3d_nparr(np.stack(tree_list), PADDING))

        # print(tr_X_list[-1].shape)

        tr_X = torch.from_numpy(np.stack(tr_X_list)).to(device)

        # print(tr_X.shape)

        tr_y_list = []
        for j in [data_dict["y_train"][k] for k in list(tr_inds)]:
            tr_y_list.append(j)

        tr_y = torch.from_numpy(np.stack(tr_y_list)).to(device)

        optimizer.zero_grad()

        output_train = torch.softmax(model(tr_X.float()), 1)
        y_pred = np.argmax(output_train.detach().cpu().numpy(), axis=1)

        loss_train = criterion(output_train, tr_y)
        train_loss += loss_train.detach().item()

        acc = accuracy_score(y_pred, tr_y.detach().cpu().numpy())
        train_acc += acc

        # computing the updated weights of all the model parameters
        loss_train.backward()
        optimizer.step()

    with torch.no_grad():
        # Validation
        val_steps = range(0, len(data_dict["X_val"]), batch_size)
        permutation = torch.randperm(len(data_dict["X_val"]))
        model.eval()

        for i in tqdm(val_steps, desc="Validation minibatches"):
            val_inds = permutation[i : i + batch_size]
            val_X_list = []
            for j in [data_dict["X_val"][k] for k in list(val_inds)]:
                tree_list = []
                for tree_id in list(j.keys()):
                    tree_list.append(
                        pad_nparr(np.array(j[tree_id]["embedding"]), PADDING)
                    )
                val_X_list.append(pad_3d_nparr(np.stack(tree_list), PADDING))

            val_X = torch.from_numpy(np.stack(val_X_list)).to(device)

            val_y_list = []
            for j in [data_dict["y_val"][k] for k in list(val_inds)]:
                val_y_list.append(j)

            val_y = torch.from_numpy(np.stack(val_y_list)).to(device)

            output_val = torch.softmax(model(val_X.float()), 1)

            y_pred = np.argmax(output_val, axis=1)

            loss_val = criterion(output_val, val_y)
            val_loss += loss_val.detach().item()

            acc = accuracy_score(
                y_pred.detach().cpu().numpy(), val_y.detach().cpu().numpy()
            )
            val_acc += acc

            # computing the training and validation loss
            loss_val = criterion(output_val, val_y)
            val_loss += torch.mean(loss_val)

    # Average the losses/acc over entire epoch

    if epoch % 1 == 0:
        # printing the validation loss
        print(
            "\n",
            "Epoch : ",
            epoch + 1,
            " |\t",
            "train loss :",
            "{:10.2f}".format(train_loss / len(train_steps)),
            " |\t",
            "train acc:",
            "{:10.2f}".format(train_acc / len(train_steps)),
            " |\t",
            "val loss :",
            "{:10.2f}".format(val_loss / len(val_steps)),
            " |\t",
            "val acc:",
            "{:10.2f}".format(val_acc / len(val_steps)),
            "\n",
        )

    sys.stdout.flush()
    sys.stderr.flush()

    return (
        train_acc / len(data_dict["X_train"]),
        train_loss / len(data_dict["X_train"]),
        val_acc / len(data_dict["X_val"]),
        val_loss / len(data_dict["X_val"]),
    )


def plot_training(train_loss, val_loss, train_acc, val_acc):
    # plotting the training and validation loss
    plt.plot(train_loss, label="Training loss")
    plt.plot(val_loss, label="Validation loss")
    plt.legend()
    # plt.savefig("64_model_losses.png")

    # plotting the training and validation acc
    plt.plot(train_acc, label="Training acc")
    plt.plot(val_acc, label="Validation acc")
    plt.legend()
    plt.savefig("64_model_acc.png")


def test_model(model, PADDING, data_dict, modname, device):
    model.eval()
    test_X_list = []
    for j in data_dict["X_test"]:
        tree_list = []
        for tree_id in list(j.keys()):
            tree_list.append(pad_nparr(np.array(j[tree_id]["embedding"]), PADDING))
        test_X_list.append(pad_3d_nparr(np.stack(tree_list), PADDING))

    with torch.no_grad():
        test_out = model(
            torch.from_numpy(np.stack(test_X_list)).float().to(device)
        ).float()
    probs = list(torch.sigmoid(test_out).cpu().numpy())

    evaluate_model(probs, data_dict["y_test"], modname)


def evaluate_model(pred_probs, trues, modname):
    predictions = np.argmax(pred_probs, axis=1)

    # for i, j, k in zip(test_gen.list_IDs, predictions, trues):
    #    print("\t".join([str(i), str(j), str(k)]))
    """
    pred_dict = {
        "id": test_gen.list_IDs,
        "true": trues,
        "pred": predictions,
        "prob_hard": pred[:, 0],
        "prob_neut": pred[:, 1],
        "prob_soft": pred[:, 2],
    }

    pred_df = pd.DataFrame(pred_dict)

    pred_df.to_csv(
        os.path.join(base_dir, model.name + "_predictions.csv"),
        header=True,
        index=False,
    )
    """

    lablist = [str(i) for i in range(10)]

    conf_mat = pu.print_confusion_matrix(trues, predictions)
    pu.plot_confusion_matrix(
        os.path.join("."),  # , "images"),
        conf_mat,
        lablist,
        title=f"{modname}_10class_Graph_Embedding",
        normalize=True,
    )
    pu.print_classification_report(trues, predictions)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if sys.argv[1] == "cnn":
        model = CNNClassifier().float().to(device)
        modname = "cnn"
    elif sys.argv[1] == "lstm":
        model = LSTMClassifier().float().to(device)
        modname = "lstm"
    else:
        print("No model type given, opts are cnn or lstm\n")
        sys.exit(1)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    print(f"Using {torch.cuda.device_count()} GPUs.\n")

    PADDING = 100
    NUM_BATCHES = 20
    patience = 5

    lab_dict, samps, labs = load_data()
    print(set(labs))

    data_dict = split_data(samps, labs)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    training_stats = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}
    for epoch in tqdm(range(40), desc="Epochs"):
        epoch_stats = train_model(
            epoch,
            model,
            24,
            PADDING,
            data_dict,
            criterion,
            optimizer,
            device,
        )

        for stat, newstat in zip(list(training_stats.keys()), epoch_stats):
            training_stats[stat].append(newstat)

        if training_stats["val_acc"].index(max(training_stats["val_acc"])) <= (
            len(training_stats["val_acc"]) - patience
        ):
            print(
                "\nValidation accuracy not increasing within patience window, stopping.\n"
            )
            break

    try:
        plot_training(
            training_stats["train_loss"],
            training_stats["val_loss"],
            training_stats["train_acc"],
            training_stats["val_acc"],
        )
    except:
        print("Couldn't plot loss")

    print("Finished Training")
    PATH = f"./{modname}_10class_node_vec.pth"
    torch.save(model.state_dict(), PATH)

    test_model(model, PADDING, data_dict, modname, device)


if __name__ == "__main__":
    main()