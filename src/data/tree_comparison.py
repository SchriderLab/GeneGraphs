import tskit
import argparse
import os
import random

# Remove this file


def parse_arguments():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--idir", default="None")
    parser.add_argument("--odir", default="None")
    parser.add_argument("--ofile", default="None")
    parser.add_argument("--models", default="None")

    args = parser.parse_args()

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.mkdir(args.odir)

    return args


def main():

    args = parse_arguments()
    path_to_file = os.path.join(args.odir, args.ofile)
    with open(path_to_file, "w") as file:

        if args.models == "None":
            models = os.listdir(args.idir)
        else:
            models = args.models.split(',')

        for model in models:
            idir = os.path.join(args.idir, model)
            tree_sequences = sorted([os.path.join(idir, u) for u in os.listdir(idir) if u.split('.')[-1] == 'ts'])
            real_trees = sorted([u for u in tree_sequences if not 'inferred' in u])
            inf_trees = sorted([u for u in tree_sequences if 'inferred' in u])

            tree_sequences = zip(real_trees, inf_trees)
            count = 1

            for real, inf in tree_sequences:
                ts = tskit.load(real)
                ts_inf = tskit.load(inf)

                ts_list = zip(ts.aslist(), ts_inf.aslist())
                for real_tree, tree in ts_list:
                    if random.choice(range(100)) > 97:
                        file.write("\n {0} real tree {1}: \n\n".format(model, count))
                        file.write(real_tree.draw(format='unicode'))
                        file.write("\n\n\n")
                        file.write("{0} inferred tree {1}: \n\n".format(model, count))
                        file.write(tree.draw(format='unicode'))
                        file.write("----------------------------------------"*5)
                        count += 1

                else:
                    continue

    file.close()

if __name__ == "__main__":
    main()
