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


def one_hot(i, n_populations=3):
    ret = np.zeros(n_populations)

    ret[i] = 1.
    return ret


def one_hot_mutation(node_id, mutation_list, classes=2):
    label = np.zeros(classes)
    if node_id in mutation_list:
        label[0] = 1.
    else:
        label[1] = 1.
    return label


def make_node_dict(nodes, inf_nodes, mutation_list, real, mutations=True, n_populations=3):
    """Creates a dictionary mapping node IDs to their feature vectors
        Args:
            nodes (iterator): Iterator of nodes in tree sequence
            mutation_list (np array): Array of the node IDs containing mutations
            real (boolean): whether to record the inferred trees or real trees
            mutations (boolean): Whether or not to include OHE representation of whether or not the node
                                has a mutation
            n_populations (int): The number of populations
        Returns:
            dictionary(node IDs => feature vectors)
        """
    nodes = list(nodes)
    inf_nodes = list(inf_nodes)

    node_ids = [u.id for u in nodes]
    inf_node_ids = [u.id for u in inf_nodes]

    times = dict()

    # store all the inferred nodes time to normalize them (scale of 0 to 1)
    if real:
        for node in nodes:
            times[node.id] = node.time
    else:
        for node in inf_nodes:
            times[node.id] = node.time

    max_time = np.max(list(times.values()))

    ret = dict()
    for node in nodes:
        if node.id in times.keys():
            # just time and one hot encoded population for now
            x = np.zeros(n_populations + 1)
            x[0] = times[node.id] / max_time
            x[1:] = one_hot(node.population, n_populations)
            if mutations:
                mutation_label = one_hot_mutation(node.id, mutation_list)  # tree_dict
                x = np.append(x, mutation_label)

            ret[node.id] = x

    return ret


def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ofile", default="None")
    parser.add_argument("--idir", default="None")
    parser.add_argument("--models", default = "None")
    parser.add_argument("--odir", default="None")
    parser.add_argument('--real', action='store_true')


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
    real = args.real

    # create the output file
    ofile = h5py.File(args.ofile, 'w')

    # if we want to specify specific models to write to disk
    # or use all those that are in the input directory
    if args.models == "None":
        models = os.listdir(args.idir)
    else:
        models = args.models.split(',')

    for model in models:
        logging.info('root: working on model {0}'.format(model))
        idir = os.path.join(args.idir, model)

        # get the tree sequences and split by the real trees and inferred ones (we need the real ones to
        # grab some data that isn't stored in the inferred trees)
        tree_sequences = sorted([os.path.join(idir, u) for u in os.listdir(idir) if u.split('.')[-1] == 'ts'])
        real_trees = sorted([u for u in tree_sequences if not 'inferred' in u])
        inf_trees = sorted([u for u in tree_sequences if 'inferred' in u])

        tree_sequences = zip(real_trees, inf_trees)
        index = 0

        for real, inf in tree_sequences:
            ts = tskit.load(real)
            ts_inf = tskit.load(inf)

            # make a dictionary for the features of each node
            node_dict = make_node_dict(ts.nodes(), ts_inf.nodes(), ts.dump_tables().mutations.node, real)

            ts_list = []
            if real:
                ts_list = ts.aslist()
            else:
                ts_list = ts_inf.aslist()

            s_index = 0

            # for each tree store topology and node features as x and edge_index respectively
            for tree in ts_list:
                G = nx.DiGraph(tree.as_dict_of_dicts())
                x = np.array([node_dict[u] for u in G.nodes()])

                data = from_networkx(G)
                ix = data.edge_index.detach().cpu().numpy()

                ofile.create_dataset('{0}/{1}/{2}/x'.format(model, index, s_index), data = x,
                                     compression='lzf')
                ofile.create_dataset('{0}/{1}/{2}/edge_index'.format(model, index, s_index), data = ix,
                                     compression='lzf')

                s_index += 1

            ofile.flush()
            index += 1

    ofile.close()


if __name__ == '__main__':
    main()
