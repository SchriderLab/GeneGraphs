# -*- coding: utf-8 -*-
import os
import argparse
import logging

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "/proj/dschridelab/rrlove/ag1000g/data/ms/ms_modified/training/output/no_introgression/out_110421/out_110421_18_6")
    parser.add_argument("--mu", default = "5.0e-9") # mutation rate
    parser.add_argument("--L", default = "10000")
    parser.add_argument("--r", default = "1e-8")
    parser.add_argument("--pop_sizes", default = "20,14")

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
    
    cmd = 'sbatch -n 4 --mem 8G -t 2-00:00:00 -o {6} --wrap "module load gcc/11.2.0 && python3 src/data/relate_msmodified.py --idir {0} --odir {1} --L {2} --mu {3} --r {4} --pop_sizes {5}"'

    idirs = [os.path.join(args.idir, u) for u in os.listdir(args.idir) if not 'seedms' in u]
    for idir in idirs:
        cmd_ = cmd.format(idir, args.odir, args.L, args.mu, args.r, args.pop_sizes, idir + '.slurm.out')
        
        print(cmd_)
        os.system(cmd_)

    # ${code_blocks}

if __name__ == '__main__':
    main()

