import os
import sys
import itertools

import os
import logging, argparse
import itertools

import platform
import tskit
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import pandas as pd
from networkx.drawing.nx_agraph import graphviz_layout
from networkx import optimize_graph_edit_distance

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


def make_node_dict(nodes, inf_nodes, mutation_list, mutations=True, n_populations=3):
    """Creates a dictionary mapping node IDs to their feature vectors
        Args:
            nodes (iterator): Iterator of nodes in tree sequence
            mutation_list (np array): Array of the node IDs containing mutations
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
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--p_graph", default = "0.005")
    parser.add_argument("--lambda", default = "0.5")

    parser.add_argument("--models", default = "None")

    parser.add_argument("--odir", default="inference_viz")
    parser.add_argument("--ofile", default = "inference_metrics.csv")

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

    # if we want to specify specific models to write to disk
    # or use all those that are in the input directory
    if args.models == "None":
        models = os.listdir(args.idir)
    else:
        models = args.models.split(',')

    result = dict()
    result['model'] = []
    result['kc_distance'] = []

    for model in models:
        logging.info('root: working on model {0}'.format(model))
        idir = os.path.join(args.idir, model)

        odir = os.path.join(args.odir, model)
        os.mkdir(odir)

        counter = 0

        # get the tree sequences and split by the real trees and inferred ones (we need the real ones to
        # grab some data that isn't stored in the inferred trees)
        tree_sequences = sorted([os.path.join(idir, u) for u in os.listdir(idir) if u.split('.')[-1] == 'ts'])
        real_trees = sorted([u for u in tree_sequences if not 'inferred' in u])
        inf_trees = sorted([u for u in tree_sequences if 'inferred' in u])

        tree_sequences = list(zip(real_trees, inf_trees))

        for ix in range(len(tree_sequences)):
            real, inf = tree_sequences[ix]

            ts = tskit.load(real)
            ts_inf = tskit.load(inf)

            # make a dictionary for the features of each node
            node_dict_inf = make_node_dict(ts.nodes(), ts_inf.nodes(), ts.dump_tables().mutations.node)
            node_dict = make_node_dict(ts.nodes(), ts.nodes(), ts.dump_tables().mutations.node)

            ts = ts.aslist(sample_lists = True)
            ts_inf = ts_inf.aslist(sample_lists = True)

            pos_inf = dict()
            for node in node_dict_inf.keys():
                pos_inf[node] = [np.random.uniform(-1, 1), node_dict_inf[node][0]]

            pos = dict()
            for node in node_dict.keys():
                pos[node] = [np.random.uniform(-1, 1), node_dict[node][0]]

            for j in range(len(ts_inf)):
                # get the inferred tree
                tree_inf = ts_inf[j]
                left, right = tree_inf.interval.left, tree_inf.interval.right

                # find its corresponding real tree
                for k in range(len(ts)):
                    tree = ts[j]
                    left_, right_ = tree_inf.interval.left, tree_inf.interval.right

                    if (left == left_) and (right == right_):
                        break

                # convert to nx DiGraph
                G1 = nx.DiGraph(tree.as_dict_of_dicts())
                G2 = nx.DiGraph(tree_inf.as_dict_of_dicts())

                if np.random.uniform() < float(args.p_graph):
                    pos = graphviz_layout(G1, prog='dot')
                    pos_inf = graphviz_layout(G2, prog='dot')

                    plt.rc('font', family='Arial', size=12)  # set to Arial/Helvetica
                    plt.rcParams.update({'figure.autolayout': True})
                    fig = plt.figure(figsize=(8, 8), dpi=100)

                    ax1 = plt.subplot(121)
                    ax1.set_title('real')
                    nx.draw(G1, pos, ax1)

                    ax2 = plt.subplot(122)
                    ax2.set_title('inferred')
                    nx.draw(G2, pos_inf, ax2)

                    plt.savefig(os.path.join(odir, '{0:06d}.png'.format(counter)), dpi = 50)
                    plt.close()

                    counter += 1

                result['model'].append(model)
                result['kc_distance'].append(tree.kc_distance(tree_inf))

    df = pd.DataFrame(result)
    df.to_csv(args.ofile, index = False)






if __name__ == '__main__':
    main()