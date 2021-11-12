# -*- coding: utf-8 -*-
import os
import argparse
import logging

import networkx as nx

import sys
import copy
import h5py
import numpy as np

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")

    parser.add_argument("--ofile", default = "None")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()
    
    idirs = [os.path.join(args.idir, u) for u in os.listdir(args.idir)]
    ofile = h5py.File(args.ofile, 'w')
    
    for idir in idirs:
        print('working on {}...'.format(idir))
        
        anc_files = [os.path.join(idir, u) for u in os.listdir(idir) if u.split('.')[-1] == 'anc']
        mut_files = [os.path.join(idir, u) for u in os.listdir(idir) if u.split('.')[-1] == 'mut']
        
        for ix in range(len(anc_files)):
            anc_file = open(anc_files[ix], 'r')
            
            lines = anc_file.readlines()[2:]
            
            for ij in range(len(lines)):
                line = lines[ij]
                
                nodes = []
                parents = []
                lengths = []
                n_mutations = []
                regions = []
                
                edges = []
                
                # new tree
                line = line.replace(':', ' ').replace('(', '').replace(')', '').replace('\n', '')
                line = line.split(' ')[:-1]
    
                start_snp = int(line[0])
                
                for j in range(2, len(line), 5):
                    nodes.append((j - 1) // 5)
                    
                    parents.append(int(line[j]))
                    lengths.append(float(line[j + 1]))
                    n_mutations.append(float(line[j + 2]))
                    regions.append((int(line[j + 3]), int(line[j + 4])))
                    
                    edges.append((parents[-1], nodes[-1]))
                    
                G = nx.DiGraph()
                G.add_edges_from(edges)
                
                current_day_nodes = []
                
                data = dict()
                
                # find the nodes which have no out degree
                for node in G.nodes():
                    d = G.out_degree(node)
                    
                    if d == 0:
                        current_day_nodes.append(node)
                        
                for node in current_day_nodes[:len(current_day_nodes) // 2]:
                    data[node] = [0., 1., 0., 0.]
                
                for node in current_day_nodes[len(current_day_nodes) // 2:]:
                    data[node] = [0., 0., 1., 0.]
                    
                t = 0.
                
                nodes = copy.copy(current_day_nodes)
                while len(data.keys()) < len(G.nodes()):
                    _ = []
                    for node in nodes:
                        for j in range(len(edges)):
                            if edges[j][-1] == node:
                                p = edges[j][0]
                                break
                        
                        data[p] = [data[node][0] + lengths[j], 0., 0., 1.]
                        _.append(p)
                        
                    nodes = copy.copy(_)
                    
                X = []
                for node in sorted(data.keys()):
                    X.append(data[node])
                    
                X = np.array(X)
                X[:,0] /= np.max(X[:,0])
                
                edges = np.array(edges).T
                
                ofile.create_dataset('{2}/{0}/{1}/x'.format(ix, ij, idir.split('/')[-1]), data = X)
                ofile.create_dataset('{2}/{0}/{1}/edge_index'.format(ix, ij, idir.split('/')[-1]), data = edges.astype(np.int32))
            
    ofile.close()
    
    # ${code_blocks}

if __name__ == '__main__':
    main()

