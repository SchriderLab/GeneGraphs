# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os, sys

import numpy as np
import argparse
import logging

from data_functions import writeTbsFile
import copy
import subprocess

# this function creates an array for writing to text that has the ms parameters
# from a CSV file produced via bootstrapped DADI runs
# the values expected in the CSV our given below
def parameters_df(df, ix, thetaOverRho, migTime, migProb, n):
    # estimated from ??
    u = 5.0e-9
    L = 10000
    
    # Isolation, migration model
    # estimated with DADI (default)
    # nu1 and nu2 (before)
    # nu1_0 and nu2_0 (after split)
    # migration rates (Nref_m12, Nref_m21)
    ll, aic, Nref, nu1_0, nu2_0, nu1, nu2, T, Nref_m12, Nref_m21 = df[ix]
    
    nu1_0 /= Nref
    nu2_0 /= Nref
    nu1 /= Nref
    nu2 /= Nref
    
    T /= (4*Nref / 15.)
    
    alpha1 = np.log(nu1/nu1_0)/T
    alpha2 = np.log(nu2/nu2_0)/T
    
    theta = 4 * Nref * u * L
    rho = theta / thetaOverRho
    
    migTime = migTime * T
    
    p = np.tile(np.array([theta, rho, nu1, nu2, alpha1, alpha2, 0, 0, T, T, migTime, 1 - migProb, migTime]), (n, 1)).astype(object)
    
    return p, ll, Nref

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--n_samples", default = "4000", help = "number of alignments to simulate per job")
    
    parser.add_argument("--ifile", default = "params.txt", help = "CSV file of bootstrapped demographic estimates. only applicable for the drosophila case; --model dros")
    
    parser.add_argument("--model", default = "a001", help = "model you'd like to simulate. current options our 'archie' and 'dros'")
    parser.add_argument("--direction", default = "ab", help = "directionality of migration. only applicable for the drosophila case; --model dros")
    parser.add_argument("--slurm", action = "store_true")
    
    parser.add_argument("--window_size", default = "1000", help = "size in base pairs of the region to simulate")
    
    parser.add_argument("--n_grid_points", default = "2")
    parser.add_argument("--trees", action = "store_true")

    parser.add_argument("--odir", default = "None", help = "directory for the ms output and logs to be written to")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.system('mkdir -p {}'.format(args.odir))
            logging.debug('root: made output directory {0}'.format(args.odir))
    # ${odir_del_block}

    return args

def main():
    args = parse_args()
    
    mu = 1e-4
    N = np.linspace(5000, 100000, int(args.n_grid_points))
    alpha0 = np.linspace(0., 1., int(args.n_grid_points))
    alpha1 = np.linspace(0., 1., int(args.n_grid_points))
    m12 = np.linspace(0.01, 0.5, int(args.n_grid_points))
    
    todo = list(zip(N, alpha0, alpha1, m12))
    
    ms_cmd = 'mkdir -p {6} && cd {6} && {7} 64 {0} -T -t {1} -I 2 32 32 -eg 0.0 1 {2} -eg 0.0 2 {3} -m 1 2 {4} | tee {5} && gzip mig.msOut'
    
    for ix in range(len(todo)):
        print('simulating for parameters: \n {}'.format(todo[ix]))
        
        n, a1, a2, m = todo[ix]
        
        odir = os.path.join(args.odir, '{0:05d}'.format(ix))
        
        cmd_ = ms_cmd.format(int(args.n_samples), 4*n*mu, a1, a2, m, 'mig.msOut', odir, os.path.join(os.getcwd(), 'msdir/ms'))
        print(cmd_)
        
        os.system(cmd_)
        
            
if __name__ == '__main__':
    main()


