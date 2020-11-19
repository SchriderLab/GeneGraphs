import argparse
import torch
import torch.nn.funcational as F
import h5py
from data_on_the_fly_classes import DataGeneratorAE
from gcn import GCN

def parse_args():
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--odir", default = "None")

    parser.add_argument("--n_epochs", default = "5")
    parser.add_argument("--lr", default = "0.01")
    parser.add_argument("--weight_decay", default = "5e-4")

    parser.add_argument("--in_features", default = "4")
    parser.add_argument("--out_features", default = "16")

    args = parser.parse_args()

    return args

def main():

    args = parse_args()

    num_features = int(args.in_features)
    out_channels = int(args.out_features)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_features, out_channels)
    model.to(device)

    generator = DataGeneratorAE(h5py.File(args.ifile, 'r'))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    for epoch in range(int(args.n_epochs)):
        for j in range(len(generator)):
            batch = generator[j].to(device)

            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.nll_loss(out, batch.y)

            loss.backward()
            optimizer.step()

            print("Epoch: {}/{}, Step: {}/{}, Loss: {:.3f}".format(epoch+1,
                                                                   args.n_epochs, j, len(generator),
                                                                   loss.item()))
        generator.on_epoch_end()

    # need to change to test_data
    model.eval()
    # _, pred = model(data).max(dim=1)
    # correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    # acc = correct / int(data.test_mask.sum())
    # print('Accuracy: {:.4f}'.format(acc))

if __name__ == "__main__":
    main()