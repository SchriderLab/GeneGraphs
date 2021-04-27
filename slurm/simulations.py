import os
import logging, argparse


# -------------------------------------------------------- #
#                   This file is deprecated                #
# -------------------------------------------------------- #


def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--odir", default="None")
    parser.add_argument("--models", default=None, nargs='+')
    parser.add_argument("--replicates", default="1000")
    parser.add_argument("--verbose", action="store_true")
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

    cmd = 'sbatch -p general -N 1 --mem 50g -n 1 -t 7-0:00:00 --mail-type=end --mail-user=onyen@email.unc.edu --wrap="python3 src/data/simulate_msprime.py --outdir {0}/{1} --id test --length 0.1 --model {2} --replicates {3}"'
    for model in args.models:
        cmd_ = cmd.format(args.odir, model, model, args.replicates)
        print(cmd_)
        os.system(cmd_)

if __name__ == "__main__":
    main()
