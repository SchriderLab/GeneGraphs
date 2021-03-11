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

import plot_utils as pu


class CyclicLR(_LRScheduler):
    """
    https://www.kaggle.com/purplejester/a-simple-lstm-based-time-series-classifier

    """

    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]


def cosine(t_max, eta_min=0):
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min) * (1 + np.cos(np.pi * t / t_max)) / 2

    return scheduler


class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier.
    https://www.kaggle.com/purplejester/a-simple-lstm-based-time-series-classifier"""

    def __init__(self, input_dim=128, hidden_dim=256, layer_dim=3, output_dim=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]
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
            nn.Conv1d(128, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=0))

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=1, padding=0))

        self.drop_out = nn.Dropout()

        self.fcblock = nn.Sequential(
            nn.Linear(96, 1000), 
            nn.ReLU(), 
            nn.Linear(1000, 10))
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
        x = self.drop_out(x)
        x = torch.flatten(x, 1)
        x = self.fcblock(x)

        output = F.log_softmax(x, dim=1)

        return output


def load_data():
    seqfiles = glob("test/*csv")

    seqs = []
    labs = []
    seq_sizes = []

    for f in seqfiles:

        labs.append(int(f.split("/")[-1].split(".")[0].split(" ")[0]))

        rawdata = np.genfromtxt(f, delimiter=",")

        # Transpose to get into channels (embedding dims) x series (trees)
        seqs.append(rawdata.T)
        seq_sizes.append(rawdata.shape[0])

    # Need to pad up to biggest number of trees for convolution
    biggest_seq = np.max(seq_sizes)

    padded_seqs = []
    for seq in seqs:
        if seq.shape[1] < biggest_seq:
            _pad = np.zeros((seq.shape[0], biggest_seq))
            _pad[:, : seq.shape[1]] = seq
            padded_seqs.append(torch.from_numpy(_pad))
        else:
            padded_seqs.append(torch.from_numpy(seq))

    # print(padded_seqs, sorted(labs))
    return padded_seqs, labs


def split_data(treeseqs, labs):
    X_train, X_val, y_train, y_val = train_test_split(
        treeseqs, labs, stratify=labs, test_size=0.3
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
        title=f"{modname}_Graph_Embedding_1DCNN",
        normalize=True,
    )
    pu.print_classification_report(trues, predictions)


def train_model(
    epoch,
    model,
    data_dict,
    train_batch_inds,
    val_batch_inds,
    criterion,
    optimizer,
    device,
):
    model.train()

    train_acc = []
    val_acc = []

    train_loss = []
    val_loss = []

    train_total = 0
    val_total = 0

    train_correct = 0
    val_correct = 0

    for j, k in zip(range(len(train_batch_inds)), range(len(val_batch_inds))):
        tr_batch = torch.from_numpy(
            np.stack(
                data_dict["X_train"][train_batch_inds[j] : train_batch_inds[j] + 1]
            )
        ).to(device)
        tr_y = torch.from_numpy(
            np.stack(
                data_dict["y_train"][train_batch_inds[j] : train_batch_inds[j] + 1]
            )
        ).to(device)

        val_batch = torch.from_numpy(
            np.stack(data_dict["X_val"][val_batch_inds[k] : val_batch_inds[k] + 1])
        ).to(device)
        val_y = torch.from_numpy(
            np.stack(data_dict["X_train"][val_batch_inds[k] : val_batch_inds[k] + 1])
        ).to(device)

        # prediction for training set
        output_train = model(tr_batch.float())
        output_val = model(val_batch.float())

        # Training acc
        _scores, predictions = torch.max(output_train.data, 1)
        train_total += tr_y.size(0)
        train_correct += int(sum(predictions == tr_y))  # labels.size(0) returns int
        acc = round((train_correct / train_total) / 100, 2)
        train_acc.append(acc)

        # Val acc
        _scores, predictions = torch.max(output_val.data, 1)
        val_total += tr_y.size(0)
        val_correct += int(sum(predictions == tr_y))  # labels.size(0) returns int
        acc = round((val_correct / val_total) / 100, 2)
        val_acc.append(acc)

        # computing the training and validation loss
        loss_train = criterion(output_train, tr_y)
        loss_val = criterion(output_val, val_y)
        train_loss.append(loss_train)
        val_loss.append(loss_val)

        # computing the updated weights of all the model parameters
        loss_train.backward()
        optimizer.step()

    if epoch % 2 == 0:
        # printing the validation loss
        print(
            "Epoch : ",
            epoch + 1,
            "\t",
            "train loss :",
            train_loss[-1],
            "\t",
            "train acc:",
            train_acc[-1],
            "\t",
            "val loss :",
            val_loss[-1],
            "\t",
            "val acc:",
            val_acc[-1],
        )


def get_batch_inds(data_dict):
    train_batch_inds = list(
        range(
            0, len(data_dict["X_train"]), int(np.floor(len(data_dict["X_train"]) / 1))
        )
    )
    val_batch_inds = list(
        range(0, len(data_dict["X_val"]), int(np.floor(len(data_dict["X_val"]) / 1)))
    )

    return train_batch_inds, val_batch_inds


def test_model(model, data_dict, modname, device):
    with torch.no_grad():
        test_out = model(
            torch.from_numpy(np.stack(data_dict["X_test"])).float().to(device)
        ).float()

    probs = list(torch.exp(test_out).cpu().numpy())

    evaluate_model(probs, data_dict["y_test"], modname)


def plot_training(train_loss, val_loss, train_acc, val_acc):
    # plotting the training and validation loss
    plt.plot(train_loss, label="Training loss")
    plt.plot(val_loss, label="Validation loss")
    plt.legend()
    plt.savefig("model_losses.png")

    # plotting the training and validation acc
    plt.plot(train_acc, label="Training acc")
    plt.plot(val_acc, label="Validation acc")
    plt.legend()
    plt.savefig("model_acc.png")


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

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # print(model)

    seqs, labs = load_data()
    data_dict = split_data(seqs, labs)

    train_batch_inds, val_batch_inds = get_batch_inds(data_dict)

    sched = CyclicLR(
        optimizer, cosine(t_max=len(train_batch_inds) * 2, eta_min=1e-3 / 100)
    )

    for epoch in range(40):
        train_model(
            epoch,
            model,
            data_dict,
            train_batch_inds,
            val_batch_inds,
            criterion,
            optimizer,
            device,
        )

    print("Finished Training")
    PATH = f"./{modname}_graph_vec.pth"
    torch.save(model.state_dict(), PATH)

    test_model(model, data_dict, modname, device)


if __name__ == "__main__":
    main()
