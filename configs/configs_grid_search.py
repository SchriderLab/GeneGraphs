# still debugging/testing this script

import random
import argparse
import os


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_classes", default="10")
    parser.add_argument("--layer_types", default=["transformer_conv", "gat_conv", "gcn_conv"], nargs='+')
    parser.add_argument("--input_size", default="6")
    parser.add_argument("--search_size", default="10", help="how many configs to generate per layer type")
    parser.add_argument("--odir", default="configs/config_grid_search_output")

    args = parser.parse_args()
    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.mkdir(args.odir)
        else:
            os.system('rm -rf {0}'.format(os.path.join(args.odir, '*')))

    return args


def make_new_grid():
    return {
        "out_channels": random.randrange(1, 4),
        "big_out_channels": random.choice([0.5, 1., 1.5, 2.]), # if the model is deep we're reducing the size
        "depth": random.randint(1, 7),
        "num_heads": random.randint(1, 6),
        "normalize": random.choice([True, False]),
        "mlp_depth": random.randint(1, 5),
        "mlp_channels": random.choice([0.5, 1.0, 1.5, 2.0]),
        "batch_norm": random.choice([True, False]),
        "pooling_type": random.choice(["global_max_pool", "global_add_pool", "global_mean_pool", "None"]),
        "negative_slope": random.choice([0.1, 0.2]),
        "dropout": random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "lr": random.choice([5e-4, 1e-4, 5e-3, 1e-3, 1e-2]),
        "weight_decay": random.choice([1e-6, 5e-6, 1e-5, 5e-4])
    }


def main():

    args = parse_args()
    for layer_type in args.layer_types:
        grid_search = make_new_grid()
        for sampled_config in range(int(args.search_size)):
            in_channels = [int(args.input_size)]
            out_channels = []
            if layer_type == "transformer_conv":
                num_heads = grid_search["num_heads"]
            else:
                num_heads = None
            depth = grid_search["depth"]
            for i in range(depth):
                grid_search = make_new_grid()
                if depth > 5:
                    # if depth is big we make layer size smaller
                    out_channels.append(int(grid_search["big_out_channels"]*in_channels[-1]))
                else:
                    out_channels.append(grid_search["out_channels"]*in_channels[-1])
                if out_channels[-1] == 0:  # scrappy patch to edge case
                    out_channels[-1] == 10
                if i < depth - 1:
                    in_channels.append(out_channels[-1])

            in_channels = [str(x) for x in in_channels]
            in_channels = ",".join(in_channels)
            out_channels = [str(x) for x in out_channels]
            out_channels = ",".join(out_channels)

            mlp_depth = grid_search["mlp_depth"]
            mlp_channels = [int(out_channels.split(",")[-1])]
            for _ in range(mlp_depth):
                grid_search = make_new_grid()
                mlp_channels.append(int(grid_search["mlp_channels"]*mlp_channels[-1]))
                if mlp_channels[-1] > 400:
                    mlp_channels[-1] = 150 # reduce size if it is exploding
            mlp_channels.append(int(args.num_classes))

            mlp_channels = [str(x) for x in mlp_channels]
            mlp_channels = ",".join(mlp_channels)

            with open(os.path.join(args.odir, layer_type+"_config"+str(sampled_config+1)+".txt"), "w") as file:
                file.write("[encoder_params]\n")
                file.write("in_channels = " + in_channels + "\n")
                file.write("out_channels = " + out_channels + "\n")
                file.write("depth = " + str(depth) + "\n")
                file.write("layer_type = " + layer_type + "\n")
                file.write("num_heads = " + str(num_heads) + "\n\n")

                file.write("[" + layer_type.split("_")[0] + "_params]\n")
                if layer_type == "transformer_conv":
                    file.write("edge_dim = None\n")
                    file.write("normalize = " + str(grid_search["normalize"]) + "\n")
                    file.write("dropout = " + str(grid_search["dropout"]) + "\n\n")
                elif layer_type == "gcn_conv":
                    file.write("normalize = " + str(grid_search["normalize"]) + "\n\n")
                else:  # if layer_type == gat_conv
                    file.write("negative_slope = " + str(grid_search["negative_slope"]) + "\n")
                    file.write("dropout = " + str(grid_search["dropout"]) + "\n\n")

                file.write("[mlp_params]\n")
                file.write("channels = " + mlp_channels + "\n")
                file.write("batch_norm = " + str(grid_search["batch_norm"]) + "\n\n")

                file.write("[pooling_params]\n")
                file.write("pooling_type = " + str(grid_search["pooling_type"]) + "\n\n")

                file.write("[learning_params]\n")
                file.write("lr = " + str(grid_search["lr"]) + "\n")
                file.write("weight_decay = " + str(grid_search["weight_decay"]) + "\n")
                file.close()


if __name__ == "__main__":
    main()




