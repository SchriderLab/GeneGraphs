# -*- coding: utf-8 -*-
import os
import argparse
import logging

import glob
import numpy as np

from data_functions import load_data

def write_ms(x, pos, ofile):
    print(pos)
    pos = (pos * 10000).astype(np.int32)
    
    ofile = open(ofile, 'w')
    ofile.write('ms {0} {1}\n'.format(x.shape[0], 1))
    ofile.write('blah\n\n')
    
    ofile.write('// blah blah blah\n')
    ofile.write('segsites: {}'.format(x.shape[1]) + '\n')
    ofile.write('positions: ' + ' '.join(list(map(str, list(pos)))) + '\n')
    
    for x_ in x:
        ofile.write(''.join(list(map(str, list(x_.astype(np.uint8))))) + '\n')
        
    ofile.write('\n')
    ofile.close()
    

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    
    parser.add_argument("--pop_sizes", default = "150,156")
    parser.add_argument("--L", default = "10000")
    parser.add_argument("--mu", default = "3.5e-9")
    parser.add_argument("--r", default = "1e-8")
    # ${args}

    parser.add_argument("--odir", default = "None")
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
    
    msFiles = sorted(glob.glob(os.path.join(args.idir, '*.txt')))
    ancFiles = [u.replace('txt', 'anc') for u in msFiles]
    
    rcmd = 'Rscript ms2haps.R {0} {1}'
    relate_cmd = 'relate/bin/Relate --mode All -m {0} -N {1} --haps {2} --sample {3} --map {4} --output {5}'
    
    pop_sizes = list(map(int, args.pop_sizes.split(',')))
    pop_size = sum(pop_sizes)
    
    L = float(args.L)  
    r = float(args.r)
    mu = float(args.mu)

    for ix in range(len(msFiles)):
        msFile = msFiles[ix]
        ancFile = ancFiles[ix]
        
        odir_ms = os.path.join(args.odir, msFile.split('/')[-2])
        
        # make a folder for the ms files to go into
        os.system('mkdir -p {}'.format(odir_ms))
        
        odir_relate = os.path.join(args.odir, msFile.split('/')[-2] + '_relate')
        
        # make a folder for the relate output files to go into
        os.system('mkdir -p {}'.format(odir_relate))
        
        x, y, positions = load_data(msFile, ancFile)
        
        for ij in range(len(x)):
            x_ = x[ij]
            y_ = y[ij]
            pos = positions[ij]
            
            
            logging.info('{},{}'.format(x_.shape, y_.shape))
            if x_.shape[0] != pop_size:
                logging.info('pop size error (x) in sim {0} in msFile {1}'.format(ix, msFile))
                continue
            elif y_.shape[0] != pop_size:
                logging.info('pop size error (y) in sim {0} in msFile {1}'.format(ix, msFile))
                continue
            elif x_.shape[1] != y_.shape[1]:
                logging.info('seg site mismatch error in sim {0} in msFile {1}'.format(ix, msFile))
                continue
            
            # write the ms file
            #x_ = x_[:300,:]
            #y_ = y_[:300,:]
            
            ms_file = os.path.join(odir_ms, '{0:05d}_{1:05d}.ms'.format(ix, ij))
            write_ms(x_, pos, ms_file)
            
            np.savez(os.path.join(odir_ms, '{0:05d}_{1:05d}.npz'.format(ix, ij)), x = x_, y = y_, positions = pos)
            
            cmd_ = rcmd.format(ms_file, ms_file.split('.')[0])
            os.system(cmd_)
            
            ofile = open(ms_file.split('.')[0] + '.map', 'w')
            ofile.write('pos COMBINED_rate Genetic_Map\n')
            ofile.write('0 {} 0\n'.format(r * L))
            ofile.write('{0} {1} {2}\n'.format(L, r * L, r * 10**8))
            ofile.close()
            
            ofile = open(ms_file.split('.')[0] + '.poplabels', 'w')
            ofile.write('sample population group sex\n')
            for k in range(1, 151):
                ofile.write('UNR{} POP POP 1\n'.format(k))
            ofile.close()
            
            #ofile = open(ifile.split('.')[0] + '.relate.anc', 'w')
            #ofile.write('>anc\n')
            #ofile.write(sum(['0' for k in range(n_sites)]) + '\n')
            #ofile.close()
            
            cmd_ = relate_cmd.format(mu, L, ms_file.split('.')[0] + '.haps', 
                                     ms_file.split('.')[0] + '.sample', ms_file.split('.')[0] + '.map', 
                                     ms_file.split('/')[-1].split('.')[0])
            os.system(cmd_)
            
            os.system('mv {0}* {1}'.format(ms_file.split('/')[-1].split('.')[0], odir_relate))
            
            
        
        
    

    # ${code_blocks}

if __name__ == '__main__':
    main()

