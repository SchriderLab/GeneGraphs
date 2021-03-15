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
from tqdm import tqdm
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

    def __init__(self, input_dim=100, hidden_dim=128, layer_dim=1, output_dim=10):
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
            nn.Linear(768, 1000), 
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


def load_data(top_seqdir):
    seqdirs = glob(top_seqdir + "/*")

    samps = []
    labs = []
    lab_dict = {}
    for i in range(len(seqdirs)):
        lab_dict[seqdirs[i].split("/")[-1]] = i
        for j in list(glob(os.path.join(seqdirs[i], "*.csv"))):
            samps.append(j)
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


def get_batch_inds(data_dict, NUM_BATCHES):
    train_batch_inds = range(
        0,
        len(data_dict["X_train"]),
        int(np.floor(len(data_dict["X_train"]) / NUM_BATCHES)),
    )[:-1]

    val_batch_inds = list(
        range(
            0,
            len(data_dict["X_val"]),
            int(np.floor(len(data_dict["X_val"]) / NUM_BATCHES)),
        )[:-1]
    )

    return train_batch_inds, val_batch_inds


def pad_nparr(arr, pad_size):
    if arr.shape[1] < pad_size:
        _pad = np.zeros((arr.shape[0], pad_size))
        _pad[:, : arr.shape[1]] = arr
        return _pad
    else:
        return arr


def accuracy(true, pred):
    acc = (true == pred).float().detach().numpy()
    return float(100 * acc.sum() / len(acc))


def train_model(
    epoch,
    model,
    batch_size,
    PADDING,
    data_dict,
    criterion,
    optimizer,
    device,
    patience,
):
    # TODO Add some sort of shuffle to the file/lab generation
    model.train()

    train_labs = []
    train_preds = []

    val_labs = []
    val_preds = []

    train_acc = []
    val_acc = []

    train_loss = []
    val_loss = []

    # Training
    permutation = torch.randperm(len(data_dict["X_train"]))

    for i in tqdm(
        range(0, len(data_dict["X_train"]), batch_size),
        desc="Training minibatches",
    ):
        tr_inds = permutation[i : i + batch_size]
        tr_X_list = []
        for j in [data_dict["X_train"][k] for k in list(tr_inds)]:
            _arr = np.loadtxt(j, skiprows=1, delimiter=",", ndmin=2)
            _arr = np.delete(_arr, 0, axis=1)
            tr_X_list.append(pad_nparr(_arr.T, PADDING))

        tr_X = torch.from_numpy(np.stack(tr_X_list)).to(device)

        tr_y_list = []
        for j in [data_dict["y_train"][k] for k in list(tr_inds)]:
            tr_y_list.append(j)

        tr_y = torch.from_numpy(np.stack(tr_y_list)).to(device)

        optimizer.zero_grad()

        output_train = model(tr_X.float())

        # Training acc
        _scores, predictions = torch.max(output_train.data, 1)

        loss_train = criterion(output_train, tr_y)
        train_loss.append(torch.mean(loss_train))

        train_labs.append(tr_y.cpu())
        train_preds.append(predictions.cpu())

        # computing the updated weights of all the model parameters
        loss_train.backward()
        optimizer.step()

    acc = accuracy(torch.cat(train_labs), torch.cat(train_preds))
    train_acc.append(acc)

    # Validation
    permutation = torch.randperm(len(data_dict["X_val"]))

    for i in tqdm(
        range(0, len(data_dict["X_val"]), batch_size),
        desc="Validation minibatches",
    ):
        val_inds = permutation[i : i + batch_size]
        val_X_list = []
        for j in [data_dict["X_val"][k] for k in list(val_inds)]:
            _arr = np.loadtxt(j, skiprows=1, delimiter=",", ndmin=2)
            _arr = np.delete(_arr, 0, axis=1)
            val_X_list.append(pad_nparr(_arr.T, PADDING))

        val_X = torch.from_numpy(np.stack(val_X_list)).to(device)

        val_y_list = []
        for j in [data_dict["y_val"][k] for k in list(val_inds)]:
            val_y_list.append(j)

        val_y = torch.from_numpy(np.stack(val_y_list)).to(device)

        output_val = model(val_X.float())
        _scores, predictions = torch.max(output_val.data, 1)

        val_labs.append(val_y.cpu())
        val_preds.append(predictions.cpu())

        # computing the training and validation loss
        loss_val = criterion(output_val, val_y)
        val_loss.append(torch.mean(loss_val))

    acc = accuracy(torch.cat(val_labs), torch.cat(val_preds))
    val_acc.append(acc)

    if epoch % 1 == 0:
        # printing the validation loss
        print(
            "\n",
            "Epoch : ",
            epoch + 1,
            "\t",
            "train loss :",
            "{:10.2f}".format(train_loss[-1].item()),
            "\t",
            "train acc:",
            "{:10.2f}".format(train_acc[-1]),
            "\t",
            "val loss :",
            "{:10.2f}".format(val_loss[-1].item()),
            "\t",
            "val acc:",
            "{:10.2f}".format(val_acc[-1]),
            "\n",
        )

    if val_acc.index(max(val_acc)) <= (len(val_acc) - patience):
        print(
            "\nValidation accuracy not increasing within patience window, stopping.\n"
        )
        return True
    else:
        return False


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


def test_model(model, PADDING, data_dict, modname, device):

    test_X_list = []
    for j in data_dict["X_test"]:
        _arr = np.loadtxt(j, skiprows=1, delimiter=",", ndmin=2)
        _arr = np.delete(_arr, 0, axis=1)
        test_X_list.append(pad_nparr(_arr.T, PADDING))

    with torch.no_grad():
        test_out = model(
            torch.from_numpy(np.stack(test_X_list)).float().to(device)
        ).float()
    probs = list(torch.exp(test_out).cpu().numpy())

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
        title=f"{modname}_Graph_Embedding",
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
    NUM_BATCHES = 40

    lab_dict, samps, labs = load_data(
        "/overflow/dschridelab/users/wwbooker/GeneGraphs/embeddings/graph2vec/all_models_1/"
    )
    data_dict = split_data(samps, labs)

    train_batch_inds, val_batch_inds = get_batch_inds(data_dict, NUM_BATCHES)

    # sched = CyclicLR(
    #    optimizer, cosine(t_max=len(train_batch_inds) * 2, eta_min=1e-3 / 100)
    # )

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in tqdm(range(40), desc="Epochs"):
        earlystop = train_model(
            epoch,
            model,
            24,
            PADDING,
            data_dict,
            criterion,
            optimizer,
            device,
            5,
        )

        if earlystop:
            break
        else:
            continue

    print("Finished Training")
    PATH = f"./{modname}_graph_vec.pth"
    torch.save(model.state_dict(), PATH)

    test_model(model, PADDING, data_dict, modname, device)


if __name__ == "__main__":
    main()
