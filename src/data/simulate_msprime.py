import sys, os, argparse
import numpy as np
import tsinfer

from demographic_params import define_params, define_AJ_params
from demographic_models import sim_constant, \
    sim_growth, \
    sim_reduction, \
    sim_constant_2pop, \
    sim_single_pulse_uni_AB, \
    sim_single_pulse_uni_BA, \
    sim_single_pulse_bi, \
    sim_multi_pulse_uni_AB, \
    sim_multi_pulse_uni_BA, \
    sim_multi_pulse_bi, \
    sim_continuous_uni_AB, \
    sim_continuous_uni_BA, \
    sim_continuous_bi, \
    sim_AJ_substruc_pulse_presplit, \
    sim_AJ_substruc_pulses_postsplit, \
    sim_AJ_substruc


def creatDir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        if not os.path.isdir(dir):
            raise


def choose_model(model):
    if model == 'constant':
        model_func = sim_constant
    elif model == 'growth':
        model_func = sim_growth
    elif model == 'reduction':
        model_func = sim_reduction
    elif model == 'constant_2pop':
        model_func = sim_constant_2pop
    elif model == 'constant_2popAB':
        model_func = sim_constant_2popAB
    elif model == 'single_pulse_uni_AB':
        model_func = sim_single_pulse_uni_AB
    elif model == 'single_pulse_uni_BA':
        model_func = sim_single_pulse_uni_BA
    elif model == 'single_pulse_bi':
        model_func = sim_single_pulse_bi
    elif model == 'multi_pulse_uni_AB':
        model_func = sim_multi_pulse_uni_AB
    elif model == 'multi_pulse_uni_BA':
        model_func = sim_multi_pulse_uni_BA
    elif model == 'multi_pulse_bi':
        model_func = sim_multi_pulse_bi
    elif model == 'continuous_uni_AB':
        model_func = sim_continuous_uni_AB
    elif model == 'continuous_uni_BA':
        model_func = sim_continuous_uni_BA
    elif model == 'continuous_bi':
        model_func = sim_continuous_bi
    elif model == 'AJ_substruc_pulse_presplit':
        model_func = sim_AJ_substruc_pulse_presplit
    elif model == 'AJ_substruc_pulses_postsplit':
        model_func = sim_AJ_substruc_pulses_postsplit
    elif model == 'AJ_substruc':
        model_func = sim_AJ_substruc
    else:
        print('--model must be constant, growth, reduction, constant_2pop, constant_2popAB, constant_3pop, '
              'single_pulse_uni_AB, single_pulse_uni_BA, single_pulse_bi, multi_pulse_uni_AB, multi_pulse_uni_BA, '
              'multi_pulse_bi, continuous_uni_AB, continuous_uni_BA, continuous_bi, '
              'AJ_substruc_pulse_presplit, AJ_substruc_pulses_postsplit, AJ_substruc '
              '(or write a new msprime model function!)')
        sys.exit()
    return model_func


def parse_arguments():
    parser = argparse.ArgumentParser(description='Simulate with msprime')
    parser.add_argument("-d", "--outdir",
                        help="output directory to write npz simulation files to",
                        type=str, dest='out_dir', required=True)
    parser.add_argument("-p", "--paramdir",
                        help="output directory to write parameter files to. "
                             "If not specified, will print param files to --outdir instead",
                        type=str, dest='param_dir')
    parser.add_argument("-i", "--id",
                        help="simulation id", type=str, dest='sim_id',
                        required=True)
    parser.add_argument("-l", "--length",
                        help="locus length in Mb", type=float, dest="Lmb", required=True)
    parser.add_argument("-m", "--model",
                        help="constant, "
                             "growth, "
                             "reduction, "
                             "constant_2pop, "
                             "constant_2popAB, "
                             "constant_3pop, "
                             "single_pulse_uni_AB, "
                             "single_pulse_uni_BA, "
                             "single_pulse_bi, "
                             "multi_pulse_uni_AB, "
                             "multi_pulse_uni_BA, "
                             "multi_pulse_bi, "
                             "continuous_uni_AB, "
                             "continuous_uni_BA, "
                             "continuous_bi, "
                             "AJ_substruc_pulse_presplit, "
                             "AJ_substruc_pulses_postsplit, "
                             "AJ_substruc", type=str, dest="model", required=True)
    parser.add_argument("-n", "--replicates",
                        help="number of simulation replicates", type=int, dest="num_replicates", required=False)
    parser.add_argument("-ln", "--locus_reps",
                        help="number of locus replicates", type=int, dest="locus_replicates", required=False)
    parser.add_argument("-s", "--seed",
                        help="msprime simulation seed", type=int, dest="seed", required=False)
    parser.add_argument("-N0", "--N0",
                        help="Effective population size", type=float, dest="N0", required=False)
    parser.add_argument("-N1", "--N1",
                        help="Effective population size before growth", type=float, dest="N1", required=False)
    parser.add_argument("-NA", "--NA",
                        help="Effective population size of popA", type=float, dest="NA", required=False)
    parser.add_argument("-NB", "--NB",
                        help="Effective population size of popB", type=float, dest="NB", required=False)
    parser.add_argument("-t1", "--t1",
                        help="time of growth", type=float, dest="t1", required=False)
    parser.add_argument("-td", "--td",
                        help="time of divergence", type=float, dest="td", required=False)
    parser.add_argument("-u", "--mu",
                        help="mutation rate (e.g. 1.2e-8)", type=float, dest="u", required=False)
    parser.add_argument("--msout",
                        help="print ms-style output", action='store_true', required=False)
    parser.add_argument("--samples",
                        help="sample size", type=int, dest="sample_size", required=False)
    parser.add_argument("--inferred",
                        help="use inferred trees", action='store_true', dest="inferred", required=False)

    args = parser.parse_args()
    out_dir = args.out_dir
    if args.locus_replicates:
        locus_replicates = args.locus_replicates
    else:
        locus_replicates = 1
    if args.param_dir:
        param_dir = args.param_dir
    else:
        param_dir = out_dir
    sim_id = args.sim_id
    L = float(args.Lmb) * 1000000
    print('simulate {} model with mutations'.format(str(args.model)))
    model_func = choose_model(args.model)
    if args.num_replicates:
        num_replicates = args.num_replicates
    else:
        num_replicates = 1
    if args.inferred:
        inferred = args.inferred
    else:
        inferred = False

    fixed_params = {'seed': args.seed,
                    'sample_size': args.sample_size,
                    'N0': args.N0,
                    'N1': args.N1,
                    'NA': args.NA,
                    'NB': args.NB,
                    'NC': None,
                    't1': args.t1,
                    't2': None,
                    'u': args.u,
                    'td': args.td,
                    'tm': None,
                    'tm1': None,
                    'tm2': None,
                    'p_AB': None,
                    'p_BA': None,
                    'p_AB1': None,
                    'p_AB2': None,
                    'p_BA1': None,
                    'p_BA2': None,
                    'm_AB': None,
                    'm_BA': None,
                    'NE': None,
                    'NJ': None,
                    'NM': None,
                    'NEA': None,
                    'NWA': None,
                    'NAg': None,
                    'TAg': None,
                    'TAEW': None,
                    'Tm': None,
                    'TmE': None,
                    'TmW': None,
                    'TA': None,
                    'TMJ': None,
                    'TEM': None,
                    'm': None,
                    'mE': None,
                    'mW': None
                    }

    return out_dir, param_dir, sim_id, L, num_replicates, model_func, locus_replicates, fixed_params, args.msout, args.model, inferred


def write_param_file(params, param_file_path, j):
    head = list(params.keys())
    line = []
    for key, value in params.items():
        line.append(str(value))
    with open('{}_{}.params'.format(param_file_path, j), 'w') as param_file:
        param_file.write('\t'.join(head) + '\n')
        param_file.write('\t'.join(line))
    return


def get_sites(tree_sequence):
    p = np.array([])
    for site in tree_sequence.sites():
        p = np.append(p, [site.position])
    if len(p) > 0:
        positions = p / p[-1]
    else:
        return None
    return positions


def sim_locus(model_func, L, in_params, param_file_path, j, out_file_path, max_snps, locus_replicates, msout, inferred,
              out_dir):
    print('simulating 1 locus replicate')
    print('sim_locus')
    tree_replicates, params, y, label = model_func(L, in_params, locus_replicates)

    if msout:
        ms_file = '{}_{}.ms'.format(out_file_path, j)
        with open(ms_file, 'w') as f:
            f.write('ms {} {}\n'.format(in_params['sample_size'], locus_replicates))
            f.write('blah\n')

    for i, tree_sequence in enumerate(tree_replicates):
        positions = get_sites(tree_sequence)
        if positions is None:
            continue
        write_param_file(params, param_file_path, j)
        print('tree_sequence has a sequence length of {} Mb'.format(tree_sequence.sequence_length / 1000000))
        print('tree_sequence has {} samples'.format(tree_sequence.num_samples))
        n_snps = tree_sequence.num_sites
        print('tree_sequence has {} sites'.format(n_snps))
        if max_snps < n_snps:
            max_snps = n_snps

        tree_constant_np_matrix = np.transpose(tree_sequence.genotype_matrix())

        if msout:
            with open(ms_file, 'a') as f:
                f.write('\n//\t{}\n'.format(params))
                f.write('segsites: {}\n'.format(n_snps))
                f.write('positions: {}\n'.format(' '.join(str(e) for e in positions.tolist())))
                for indiv in tree_constant_np_matrix.tolist():
                    genotypes = ''.join(str(e) for e in indiv)
                    f.write('{}\n'.format(genotypes))

        with tsinfer.SampleData(
                path=os.path.join(out_dir, "inferred{0}.samples".format(j)),
                sequence_length=tree_sequence.sequence_length,
                num_flush_threads=2) as sample_data:
            # do we only want to iterate through the variants?
            # need to determine exactly how to encode our trees using this SampleData class
            for var in tree_sequence.variants():
                sample_data.add_site(var.site.position, var.genotypes, var.alleles)

            sample_data.finalise()

            inferred_ts = tsinfer.infer(sample_data)

    filename = '{}_{}.npz'.format(out_file_path, j)

    print('save tree_sequence to file {}\n'.format(filename))
    try:
        np.savez_compressed(filename, X=tree_constant_np_matrix, y=y, z=label, p=positions)
        tree_sequence.dump('{}_{:06d}.ts'.format(out_file_path, j))
        inferred_ts.dump('{}_{:06d}_inferred.ts'.format(out_file_path, j))
    except:
        pass

    return max_snps


def sim_locus_reps(model_func, L, in_params, param_file_path, j, out_file_path, max_snps, locus_replicates, msout):
    print('simulating {} locus replicates'.format(locus_replicates))
    print('sim_locus_reps')
    tree_matrices = []
    positions_list = []
    tree_replicates, params, y, label = model_func(L, in_params, locus_replicates)
    write_param_file(params, param_file_path, j)

    if msout:
        ms_file = '{}_{}.ms'.format(out_file_path, j)
        with open(ms_file, 'w') as f:
            f.write('ms {} {}\n'.format(in_params['sample_size'], locus_replicates))
            f.write('blah\n')

    for i, tree_sequence in enumerate(tree_replicates):
        positions_list.append(get_sites(tree_sequence))
        print('tree_sequence has a sequence length of {} Mb'.format(tree_sequence.sequence_length / 1000000))
        print('tree_sequence has {} samples'.format(tree_sequence.num_samples))
        n_snps = tree_sequence.num_sites
        print('tree_sequence has {} sites'.format(n_snps))
        if max_snps < n_snps:
            max_snps = n_snps

        tree_constant_np_matrix = np.transpose(tree_sequence.genotype_matrix())
        print(tree_constant_np_matrix)
        tree_matrices.append(tree_constant_np_matrix)

        if msout:
            with open(ms_file, 'a') as f:
                f.write('\n//\t{}\n'.format(params))
                f.write('segsites: {}\n'.format(n_snps))
                f.write('positions: {}\n'.format(' '.join(str(e) for e in positions_list[i].tolist())))
                for indiv in tree_constant_np_matrix.tolist():
                    genotypes = ''.join(str(e) for e in indiv)
                    f.write('{}\n'.format(genotypes))

    positions_reps = np.zeros((locus_replicates, max_snps))
    tree_np_matrices = np.zeros((locus_replicates, np.sum(in_params['sample_size']), max_snps))
    for i, tree_constant_np_matrix in enumerate(tree_matrices):
        n_snps = tree_constant_np_matrix.shape[1]
        tree_np_matrices[i, :, :n_snps] = tree_constant_np_matrix
        positions_reps[i, :n_snps] = positions_list[i]

    filename = '{}_{}.npz'.format(out_file_path, j)
    print('save tree_sequence to file {}\n'.format(filename))
    np.savez_compressed(filename, X=tree_np_matrices, y=y, z=label, p=positions_reps)
    return max_snps


def main():
    sys.stderr.write('starting simulations\n')

    out_dir, param_dir, sim_id, L, num_replicates, model_func, locus_replicates, fixed_params, msout, model, inferred = parse_arguments()

    creatDir(out_dir)
    creatDir(param_dir)
    print(num_replicates)
    if out_dir.endswith('/'):
        out_file_path = '{}{}'.format(out_dir, sim_id)
    else:
        out_file_path = '{}/{}'.format(out_dir, sim_id)
    if param_dir.endswith('/'):
        param_file_path = '{}{}'.format(param_dir, sim_id)
    else:
        param_file_path = '{}/{}'.format(param_dir, sim_id)

    max_snps = 0
    for j in range(num_replicates):
        if "AJ" in model:
            in_params, priors = define_AJ_params(fixed_params)
        else:
            in_params, priors = define_params(fixed_params)
        if locus_replicates > 1:
            max_snps = sim_locus_reps(model_func, L, in_params, param_file_path, j, out_file_path, max_snps,
                                      locus_replicates, msout)
        else:
            max_snps = sim_locus(model_func, L, in_params, param_file_path, j, out_file_path, max_snps,
                                 locus_replicates, msout, inferred, out_dir)  #

    snp_filename = '{}_maxsnps.txt'.format(out_file_path)
    with open(snp_filename, 'w') as f:
        f.write('{}\n'.format(max_snps))

    sys.stderr.write('Done!\n')


if __name__ == "__main__":
    main()
