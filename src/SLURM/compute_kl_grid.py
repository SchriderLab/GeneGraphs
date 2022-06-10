# -*- coding: utf-8 -*-
import os
import argparse
import logging
import glob

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
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

    ifiles = os.listdir(args.idir)
    N = len(ifiles)
    
    cmd = 'sbatch -t 12:00:00 --mem=8G --wrap "python3 src/compute_kl_matrix.py --idir {0} --i {1} --ofile {2}"'
    
    for ix in range(N):
        cmd_ = cmd.format(args.idir, ix, os.path.join(args.odir, '{0:06d}.npz'.format(ix)))
        
        print(cmd_)
        os.system(cmd_)

if __name__ == '__main__':
    main()

