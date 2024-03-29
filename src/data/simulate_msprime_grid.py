# -*- coding: utf-8 -*-
import os
import argparse
import logging

import msprime
import numpy as np
import itertools
import scipy.integrate as integrate

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

import copy
import random
import pickle

from io import StringIO
from skbio import read
from skbio.tree import TreeNode
import matplotlib.pyplot as plt


import sys

def nC2(n):
    return n * (n - 1) / 2

def f_a001_coal(alpha, N, t): 
    # alpha > 0. = growth rate
    # t = time in generations
    # N = effective population size at t = 0.
    # dt = time since last event
    # n = sample size (unused here)
    if N < 1:
        N = 1

    ret = 1 / (N * np.exp(-alpha * t)) * np.exp((1 - np.exp(alpha * t)) / (N * alpha))
    
    return ret
    
def F_a001_coal(alpha, N, t):
    if N < 1:
        N = 1
        
    ret = 1 - np.exp((1 - np.exp(alpha * t)) / (alpha * N))
    return ret

def f_a001_mig(a0, a1, m, N1, N2, t):
    # a0 > 0. = growth rate for pop0
    # a1 > 0. = growth rate for pop1
    # a0 != a1.
    # m migration fraction
    # t = time in generations
    # dt = time since last event
    # n = sample size (unused here)
    if N1 < 1:
        N1 = 1
    if N2 < 1:
        N2 = 1
    
    ret = m * np.exp(-a1 * t + a0 * t) * (N2 / N1) * np.exp(-1 * m * (N2 / N1) * (np.exp(t * (a0 - a1)) - 1) / (a0 - a1))
    return ret

def F_a001_mig(a0, a1, m, N1, N2, t):
    if N1 < 1:
        N1 = 1
    if N2 < 1:
        N2 = 1
    
    ret = 1 - np.exp(m * (N2 / N1) * (np.exp(t * (a0 - a1)) - 1) / (a1 - a0))
    
    return ret

def prob_a001_coal(i, a0, a1, m, t0, t1, N, n1, n2):
    t = t1 - t0
    
    if i == 0:
        if n2 > 1:
            return F_a001_coal(a0, N * np.exp(-a0 * t0), t) * f_a001_coal(a0, N * np.exp(-a0 * t0), t) * (1 - F_a001_coal(a0, N * np.exp(-a0 * t0), t)) ** (nC2(n1) - 1) * (1 - F_a001_coal(a1, N * np.exp(-a1 * t0), t)) ** (nC2(n2)) * (1 - F_a001_mig(a0, a1, m,  N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), t)) ** n1
        else:
            return F_a001_coal(a0, N * np.exp(-a0 * t0), t) * f_a001_coal(a0, N * np.exp(-a0 * t0), t) * (1 - F_a001_coal(a0, N * np.exp(-a0 * t0), t)) ** (nC2(n1) - 1) * (1 - F_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), t)) ** n1
    else:
        if n1 > 1:
            return F_a001_coal(a1, N * np.exp(-a0 * t0), t) * f_a001_coal(a1, N * np.exp(-a1 * t0), t) * (1 - F_a001_coal(a1, N * np.exp(-a1 * t0), t)) ** (nC2(n2) - 1) * (1 - F_a001_coal(a0, N * np.exp(-a0 * t0), t)) ** (nC2(n1)) * (1 - F_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), t)) ** n1
        elif n1 == 1:
            return F_a001_coal(a1, N * np.exp(-a0 * t0), t) * f_a001_coal(a1, N * np.exp(-a1 * t0), t) * (1 - F_a001_coal(a1, N * np.exp(-a1 * t0), t)) ** (nC2(n2) - 1) * (1 - F_a001_mig(a0, a1, m, N * np.exp(-a0 * t0), N * np.exp(-a1 * t0), t)) ** n1
        else:
            if n2 > 2:
                return F_a001_coal(a0, N * np.exp(-a0 * t0), t) * f_a001_coal(a1, N * np.exp(-a1 * t0), t) * (1 - F_a001_coal(a1, N * np.exp(-a1 * t0), t)) ** (nC2(n2) - 1)
            else:
                return f_a001_coal(a1, N * np.exp(-a1 * t0), t)
        
def prob_a001_mig(a0, a1, m, t0, t1, N, n1, n2):
    t = t1 - t0
    
    if n2 >= 2 and n1 >= 2:
        return F_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), t) * f_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), t) * (1 - F_a001_coal(a1, N * np.exp(-a1 * t0), t)) ** (nC2(n2)) * (1 - F_a001_coal(a0, N * np.exp(-a0 * t0), t)) ** (nC2(n1)) * (1 - F_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), t)) ** (n1 - 1)
    elif n2 >= 2 and n1 == 1:
        return F_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), t) * f_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), t) * (1 - F_a001_coal(a1, N * np.exp(-a1 * t0), t)) ** (nC2(n2))
    elif n2 == 1 and n1 == 1:
        return f_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), t)
    elif n2 == 1 and n1 >= 2:
        return F_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), t) * f_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), t) * (1 - F_a001_coal(a0, N * np.exp(-a0 * t0), t)) ** (nC2(n1)) * (1 - F_a001_mig(a0, a1, m, N * np.exp(-a0 * t0),  N * np.exp(-a1 * t0), t)) ** (n1 - 1)

def prob_a001_first(a0, a1, m, N, t, dt, n0, n1):
    # a0 > 0. = growth rate for pop0
    # a1 > 0. = growth rate for pop1
    # a0 != a1.
    # N = effective population size at t = 0.
    # m migration fraction
    # t = time in generations
    # dt = time since last event
    # n1 = sample size pop 0 (unused here)
    # n2 = sample size pop 1 (unused here)
    
    p_coal = integrate.quad(lambda v: prob_a001_coal(a1, N, t, v, n1) * (1 - integrate.quad(lambda u: prob_a001_mig(a0, a1, m, t, u, n0), 0, v)[0]), 0, dt)[0]
    
    p_mig = 1 - p_coal
    
    return p_coal, p_mig

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--n_grid_points", default = "8")
    parser.add_argument("--sample_sizes", default = "32,32")
    parser.add_argument("--n_replicates", default = "5000")
    
    parser.add_argument("--N", default = "None")
    parser.add_argument("--alpha0", default = "None")
    parser.add_argument("--alpha1", default = "None")
    parser.add_argument("--m", default = "m")

    parser.add_argument("--ofile", default = "None")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    # ${odir_del_block}

    return args

def main():
    args = parse_args()
    
    sample_sizes = tuple(map(int, args.sample_sizes.split(',')))
    s1, s2 = sample_sizes
    
    """
    mu = 1e-4
    N = np.linspace(10000, 20000, int(args.n_grid_points))
    alpha0 = np.linspace(0.01, 0.1, int(args.n_grid_points)) * 10e-2
    alpha1 = np.linspace(0.02, 0.09, int(args.n_grid_points)) * 10e-2
    m12 = np.linspace(0.05, 0.2, int(args.n_grid_points))
    
    todo = list(itertools.product(N, alpha0, alpha1, m12))
    """
    
    n = float(args.N)
    a1 = float(args.alpha0)
    a2 = float(args.alpha1)
    m = float(args.m)
    
    E = []
    A = []
    for j in range(int(args.n_replicates)):
        demography = msprime.Demography()
        demography.add_population(name="A", initial_size = n)
        demography.add_population(name="B", initial_size = n)
        
        demography.set_migration_rate("A", "B", m)
        
        demography.add_population_parameters_change(0., population = "A", growth_rate = a1)
        demography.add_population_parameters_change(0., population = "B", growth_rate = a2)
        
        s = msprime.sim_ancestry(samples = [msprime.SampleSet(s1, population = "A", ploidy = 1), msprime.SampleSet(s2, population = "B", ploidy = 1)], recombination_rate = 0., sequence_length = 1000, demography = demography, record_migrations = True)
        
        tables = s.dump_tables()
        tables.sort()
        
        t = list(s.aslist())[0]
        
        f = StringIO(t.as_newick())  
        t_ = read(f, format="newick", into=TreeNode)
        
        n_sample = s1 + s2
        
        ix = n_sample
        order = []
                
        for node in t_.levelorder():
            if node.name is None:
                node.name = ix
                ix += 1
                
            if type(node.name) == str:
                node.name = int(node.name.replace('n', ''))

            order.append(node.name)
            
        A_ = np.zeros((2*n_sample - 1, 2*n_sample - 1))
        for node in t_.levelorder():
            for u in node.children:
                A_[node.name, u.name] = u.length
                A_[u.name, node.name] = u.length
            
        pop1 = list(range(s1))
        pop2 = list(range(s1, s1 + s2))
        
        pop1 = [u for u in order if u in pop1]
        pop2 = [u for u in order if u in pop2]

        
        order = list(range(2 * n_sample - 1))
        order[:s1 + s2] = (pop1 + pop2)
        
        A_ = A_[np.ix_(order, order)]

        ages = tables.nodes.time
        pops = tables.nodes.population
        individuals = list(range(len(pops)))
        
        edges = np.array([tables.edges.parent, tables.edges.child], dtype = np.int32).T
        
        start = list(np.where(ages > 0)[0])[0]
        coals = []
        
        t = 0.
        # get the coalescence events
        for ij in range(start, len(individuals)):
            i = individuals[ij]
            t1 = ages[ij]
            pop = pops[ij]
            
            e = edges[np.where(edges[:,0] == i)[0]]
            c1 = e[0,1]
            c2 = e[1,1]
            
            coals.append((0, pop, t1))
            
            t = copy.copy(t1)

        migs = []
        # get the migration events
        
        time = tables.migrations.time
        node = tables.migrations.node
        
        for ij in range(len(time)):
            t = time[ij]
            i = node[ij]
            
            migs.append((1, i, t))
            
        events = sorted(migs + coals, key = lambda u: u[1])
        E.append(events)
        A.append(A_)
        
    np.savez_compressed(args.ofile, E = np.array(E, dtype = object), A = np.array(A), loc = np.array([n, a1, a2, m]))
        
    """
    pop_sizes = [s1, s2]
    
    logP = 0.
    
    t = 0.
    for e in events:
        t_ = e[1]
        
        if len(e) == 2:
            p = prob_a001_mig(a1, a2, m, t, t_, n, pop_sizes[0], pop_sizes[1])
            logP += np.log(p)

            if p < 0:
                print(p)
                print(a1, a2, m, t, t_, n, pop_sizes[0], pop_sizes[1])
            pop_sizes[0] -= 1
            pop_sizes[1] += 1
        elif e[0] == 0:
            p = prob_a001_coal(0, a1, a2, m, t, t_, n, pop_sizes[0], pop_sizes[1])
            logP += np.log(p)
  
            if p < 0:
                print(p)
                print(0, a1, a2, m, t, t_, n, pop_sizes[0], pop_sizes[1])
            pop_sizes[0] -= 1
        else:

            p = prob_a001_coal(1, a1, a2, m, t, t_, n, pop_sizes[0], pop_sizes[1])
            logP += np.log(p)

            if p < 0:
                print(p)
                print(1, a1, a2, m, t, t_, n, pop_sizes[0], pop_sizes[1])
            
            pop_sizes[1] -= 1
            
        t = copy.copy(t_)
    
    print(logP)
    """

    # ${code_blocks}

if __name__ == '__main__':
    main()

