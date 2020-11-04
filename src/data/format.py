import os
import sys
import itertools

import os
import logging, argparse
import itertools

import platform
import h5py
import tskit

import numpy as np
import networkx as nx

from torch_geometric.utils.convert import from_networkx

def one_hot(i, n_populations = 3):
    ret = np.zeros(n_populations)

    ret[i] = 1.
    return ret

# make a dict mapping node id to feature vector
def make_node_dict(nodes, n_populations = 3):
    nodes = list(nodes)

    times = []

    for node in nodes:
        times.append(node.time)

    max_time = np.max(times)

    ret = dict()
    for node in nodes:
        # just time and one hot encoded population for now
        x = np.zeros(n_populations + 1)
        x[0] = node.time / max_time
        x[1:] = one_hot(node.population, n_populations)

        ret[node.id] = x

    return ret

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ofile", default = "None")
    parser.add_argument("--idir", default = "None")

    parser.add_argument("--odir", default = "None")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.mkdir(args.odir)
            logging.debug('root: made output directory {0}'.format(args.odir))
        else:
            os.system('rm -rf {0}'.format(os.path.join(args.odir, '*')))

    return args

def main():
    args = parse_args()

    ofile = h5py.File(args.ofile, 'w')
    tree_sequences = [os.path.join(args.idir, u) for u in os.listdir(args.idir) if u.split('.')[-1] == 'ts']
    index = 0

    for tree_sequence in tree_sequences:
        ts = tskit.load(tree_sequence)

        node_dict = make_node_dict(ts.nodes())

        X = []
        edge_index = []

        ts_list = ts.aslist()

        for tree in ts_list:
            G = nx.DiGraph(tree.as_dict_of_dicts())

            x = np.array([node_dict[u] for u in G.nodes()])

            data = from_networkx(G)
            ix = data.edge_index.detach().cpu().numpy()

            X.append(x)
            edge_index.append(ix)

        ofile.create_dataset('{0}/x'.format(index), data = np.array(X, dtype = np.float32), compression = 'lzf')
        ofile.create_dataset('{0}/edge_index'.format(index), data = np.array(edge_index, dtype = np.int64), compression = 'lzf')
        ofile.flush()

        index += 1

    ofile.close()

if __name__ == '__main__':
    main()