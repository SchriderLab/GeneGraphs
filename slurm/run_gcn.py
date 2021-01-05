import os
import logging, argparse
import itertools

import platform

def parse_args():
    # Argument Parser
    parser.add_argument("--verbose", action="store_true", help="display messages")  
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--ifile_val", default="None")

    parser.add_argument("--idir", default="None")
    parser.add_argument("--odir", default = "None")

    parser.add_argument("--n_epochs", default = "5")
    parser.add_argument("--lr", default="0.01")
    parser.add_argument("--weight_decay", default="5e-4")

    parser.add_argument("--in_features", default = "6")
    parser.add_argument("--out_features", default = "16")

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

    return args

def main():
    args = parse_args()

    cmd = 'sbatch src/train_model.sh {0} {1} {2} {3} {4} {5} {6} {7} {8}'.format(
        args.ifile, args.ifile_val, args.idir, args.odir, args.n_epochs, args.lr, 
        args.weight_decay, args.in_features, args.out_features
    )

    print(cmd)
    os.system(cmd_)

if __name__ == '__main__':
    main()