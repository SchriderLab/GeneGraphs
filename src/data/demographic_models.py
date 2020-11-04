import random
import numpy as np
from collections import OrderedDict
import msprime


def selectVal(minVal, maxVal):
    size = maxVal-minVal
    return (random.random()*size)+minVal


def sim_constant(L, params, locus_replicates=1):
    if fixed_params['sample_size']:
        sample_size = fixed_params['sample_size']
    if fixed_params['u']:
        u = fixed_params['u']
    if not fixed_params['N0']:
        N0 = selectVal(1000, 40000)
    else:
        N0 = fixed_params['N0']
    if not fixed_params['seed']:
        seed = random.randint(1, 2**32-1)
    else:
        seed = fixed_params['seed']
#     print('N0: {}'.format(N0))
#     print('seed: {}'.format(seed))
    tree = msprime.simulate(sample_size=sample_size,
                            Ne=N0,
                            length=L,
                            recombination_rate=r,
                            mutation_rate=u,
                            random_seed=seed,
                            num_replicates=locus_replicates)
    params = OrderedDict()
    params['sample_size'] = sample_size
    params['length'] = L
    params['recomb_rate'] = r
    params['mutation_rate'] = u
    params['N0'] = N0
    params['seed'] = seed
    y = np.array([N0])
    label = np.array([0])
    return tree, params, y, label


def sim_growth(L, params, locus_replicates=1):
    if fixed_params['sample_size']:
        sample_size = fixed_params['sample_size']
    if fixed_params['u']:
        u = fixed_params['u']
    if not fixed_params['N0']:
        N0 = selectVal(1000, 40000)
    else:
        N0 = fixed_params['N0']
    if not fixed_params['N1']:
        N1 = N0 / selectVal(1, 10)
    else:
        N1 = fixed_params['N1']
    if not fixed_params['t1']:
        t1 = selectVal(100, 3499.99)
    else:
        t1 = fixed_params['t1']
    if not fixed_params['seed']:
        seed = random.randint(1, 2**32-1)
    else:
        seed = fixed_params['seed']

    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=sample_size, initial_size=N0)
    ]

    demographic_events = [
        msprime.PopulationParametersChange(
            time=t1, initial_size=N1, population_id=0
        )
    ]

    tree = msprime.simulate(population_configurations=population_configurations,
                            demographic_events=demographic_events,
                            length=L,
                            recombination_rate=r,
                            mutation_rate=u,
                            random_seed=seed,
                            num_replicates=locus_replicates)
    params = OrderedDict()
    params['sample_size'] = sample_size
    params['length'] = L
    params['recomb_rate'] = r
    params['mutation_rate'] = u
    params['N0'] = N0
    params['N1'] = N1
    params['t1'] = t1
    params['seed'] = seed
    y = np.array([N0, N1, t1])
    label = np.array([1])
    return tree, params, y, label


def sim_reduction(L, params, locus_replicates=1):
    if fixed_params['sample_size']:
        sample_size = fixed_params['sample_size']
    if fixed_params['u']:
        u = fixed_params['u']
    if not fixed_params['N1']:
        N1 = selectVal(1000, 40000)
    else:
        N1 = fixed_params['N1']
    if not fixed_params['N0']:
        N0 = N1 / selectVal(1, 10)
    else:
        N0 = fixed_params['N0']
    if not fixed_params['t1']:
        t1 = selectVal(100, 3499.99)
    else:
        t1 = fixed_params['t1']
    if not fixed_params['seed']:
        seed = random.randint(1, 2**32-1)
    else:
        seed = fixed_params['seed']

    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=sample_size, initial_size=N0)
    ]

    demographic_events = [
        msprime.PopulationParametersChange(
            time=t1, initial_size=N1, population_id=0
        )
    ]

    tree = msprime.simulate(population_configurations=population_configurations,
                            demographic_events=demographic_events,
                            length=L,
                            recombination_rate=r,
                            mutation_rate=u,
                            random_seed=seed,
                            num_replicates=locus_replicates)
    params = OrderedDict()
    params['sample_size'] = sample_size
    params['length'] = L
    params['recomb_rate'] = r
    params['mutation_rate'] = u
    params['N0'] = N0
    params['N1'] = N1
    params['t1'] = t1
    params['seed'] = seed
    y = np.array([N0, N1, t1])
    label = np.array([2])
    return tree, params, y, label


def sim_constant_2pop(L, params, locus_replicates=1):
    seed = params['seed']
    r = params['r']
    u = params['u']
    sample_size = params['sample_size']
    N0 = params['N0']
    NA = params['NA']
    NB = params['NB']
    td = params['td']

#     print('N0: {}'.format(N0))
#     print('NA: {}'.format(NA))
#     print('NB: {}'.format(NB))
#     print('td: {}'.format(td))
#     print('seed: {}'.format(seed))
    A, B = 1, 2
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N0),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size/2), initial_size=NA),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size/2), initial_size=NB)
    ]
    demographic_events = [
        msprime.MassMigration(
            time=td, source=A, destination=0, proportion=1.0),
        msprime.MassMigration(
            time=td, source=B, destination=0, proportion=1.0)
    ]
    tree = msprime.simulate(population_configurations=population_configurations,
                            demographic_events=demographic_events,
                            length=L,
                            recombination_rate=r,
                            mutation_rate=u,
                            random_seed=seed,
                            num_replicates=locus_replicates)
    params = OrderedDict()
    params['sample_size'] = sample_size
    params['length'] = L
    params['recomb_rate'] = r
    params['mutation_rate'] = u
    params['N0'] = N0
    params['NA'] = NA
    params['NB'] = NB
    params['td'] = td
    params['seed'] = seed
    y = np.array([N0, NA, NB, td])
    label = np.array([3])
    return tree, params, y, label


def sim_single_pulse_uni_AB(L, params, locus_replicates=1):
    seed = params['seed']
    r = params['r']
    u = params['u']
    sample_size = params['sample_size']
    N0 = params['N0']
    NA = params['NA']
    NB = params['NB']
    td = params['td']
    tm = params['tm']
    p_AB = params['p_AB']
    seed = params['seed']
#     print('N0: {}'.format(N0))
#     print('NA: {}'.format(NA))
#     print('NB: {}'.format(NB))
#     print('tm: {}'.format(tm))
#     print('td: {}'.format(td))
#     print('m_AB: {}'.format(m_AB))
#     print('seed: {}'.format(seed))
    A, B = 1, 2
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N0),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size/2), initial_size=NA),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size/2), initial_size=NB)
    ]
    demographic_events = [
        msprime.MassMigration(
            time=tm, source=A, destination=B, proportion=p_AB),
        msprime.MassMigration(
            time=td, source=A, destination=0, proportion=1.0),
        msprime.MassMigration(
            time=td, source=B, destination=0, proportion=1.0)
    ]
    tree = msprime.simulate(population_configurations=population_configurations,
                            demographic_events=demographic_events,
                            length=L,
                            recombination_rate=r,
                            mutation_rate=u,
                            random_seed=seed,
                            num_replicates=locus_replicates)
    params = OrderedDict()
    params['sample_size'] = sample_size
    params['length'] = L
    params['recomb_rate'] = r
    params['mutation_rate'] = u
    params['N0'] = N0
    params['NA'] = NA
    params['NB'] = NB
    params['td'] = td
    params['tm'] = tm
    params['p_AB'] = p_AB
    params['seed'] = seed
    y = np.array([NA, NB, td, tm, p_AB])
    label = np.array([5])
    return tree, params, y, label


def sim_single_pulse_uni_BA(L, params, locus_replicates=1):
    seed = params['seed']
    r = params['r']
    u = params['u']
    sample_size = params['sample_size']
    N0 = params['N0']
    NA = params['NA']
    NB = params['NB']
    td = params['td']
    tm = params['tm']
    p_BA = params['p_BA']
#     print('N0: {}'.format(N0))
#     print('NA: {}'.format(NA))
#     print('NB: {}'.format(NB))
#     print('tm: {}'.format(tm))
#     print('td: {}'.format(td))
#     print('m_BA: {}'.format(m_BA))
#     print('seed: {}'.format(seed))
    A, B = 1, 2
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N0),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size/2), initial_size=NA),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size/2), initial_size=NB)
    ]
    demographic_events = [
        msprime.MassMigration(
            time=tm, source=B, destination=A, proportion=p_BA),
        msprime.MassMigration(
            time=td, source=A, destination=0, proportion=1.0),
        msprime.MassMigration(
            time=td, source=B, destination=0, proportion=1.0)
    ]
    tree = msprime.simulate(population_configurations=population_configurations,
                            demographic_events=demographic_events,
                            length=L,
                            recombination_rate=r,
                            mutation_rate=u,
                            random_seed=seed,
                            num_replicates=locus_replicates)
    params = OrderedDict()
    params['sample_size'] = sample_size
    params['length'] = L
    params['recomb_rate'] = r
    params['mutation_rate'] = u
    params['N0'] = N0
    params['NA'] = NA
    params['NB'] = NB
    params['td'] = td
    params['tm'] = tm
    params['p_BA'] = p_BA
    params['seed'] = seed
    y = np.array([NA, NB, td, tm, p_BA])
    label = np.array([6])
    return tree, params, y, label


def sim_single_pulse_bi(L, params, locus_replicates=1):
    seed = params['seed']
    r = params['r']
    u = params['u']
    sample_size = params['sample_size']
    N0 = params['N0']
    NA = params['NA']
    NB = params['NB']
    td = params['td']
    tm = params['tm']
    p_AB = params['p_AB']
    p_BA = params['p_BA']
#     print('N0: {}'.format(N0))
#     print('NA: {}'.format(NA))
#     print('NB: {}'.format(NB))
#     print('tm: {}'.format(tm))
#     print('td: {}'.format(td))
#     print('m_AB: {}'.format(m_AB))
#     print('m_BA: {}'.format(m_BA))
#     print('seed: {}'.format(seed))
    A, B = 1, 2
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N0),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size/2), initial_size=NA),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size/2), initial_size=NB)
    ]
    demographic_events = [
        msprime.MassMigration(
            time=tm, source=B, destination=A, proportion=p_BA),
        msprime.MassMigration(
            time=tm, source=A, destination=B, proportion=p_AB),
        msprime.MassMigration(
            time=td, source=A, destination=0, proportion=1.0),
        msprime.MassMigration(
            time=td, source=B, destination=0, proportion=1.0)
    ]
    tree = msprime.simulate(population_configurations=population_configurations,
                            demographic_events=demographic_events,
                            length=L,
                            recombination_rate=r,
                            mutation_rate=u,
                            random_seed=seed,
                            num_replicates=locus_replicates)
    params = OrderedDict()
    params['sample_size'] = sample_size
    params['length'] = L
    params['recomb_rate'] = r
    params['mutation_rate'] = u
    params['N0'] = N0
    params['NA'] = NA
    params['NB'] = NB
    params['td'] = td
    params['tm'] = tm
    params['p_AB'] = p_AB
    params['p_BA'] = p_BA
    params['seed'] = seed
    y = np.array([N0, NA, NB, td, tm, p_AB, p_BA])
    label = np.array([7])
    return tree, params, y, label


def sim_multi_pulse_uni_AB(L, params, locus_replicates=1):
    seed = params['seed']
    r = params['r']
    u = params['u']
    sample_size = params['sample_size']
    N0 = params['N0']
    NA = params['NA']
    NB = params['NB']
    td = params['td']
    tm2 = params['tm2']
    tm1 = params['tm1']
    p_AB1 = params['p_AB1']
    p_AB2 = params['p_AB2']
#     print('N0: {}'.format(N0))
#     print('NA: {}'.format(NA))
#     print('NB: {}'.format(NB))
#     print('tm1: {}'.format(tm1))
#     print('tm2: {}'.format(tm2))
#     print('td: {}'.format(td))
#     print('m_AB1: {}'.format(m_AB1))
#     print('m_AB2: {}'.format(m_AB2))
#     print('seed: {}'.format(seed))
    A, B = 1, 2
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N0),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size/2), initial_size=NA),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size/2), initial_size=NB)
    ]
    demographic_events = [
        msprime.MassMigration(
            time=tm1, source=A, destination=B, proportion=p_AB1),
        msprime.MassMigration(
            time=tm2, source=A, destination=B, proportion=p_AB2),
        msprime.MassMigration(
            time=td, source=A, destination=0, proportion=1.0),
        msprime.MassMigration(
            time=td, source=B, destination=0, proportion=1.0)
    ]
    tree = msprime.simulate(population_configurations=population_configurations,
                            demographic_events=demographic_events,
                            length=L,
                            recombination_rate=r,
                            mutation_rate=u,
                            random_seed=seed,
                            num_replicates=locus_replicates)
    params = OrderedDict()
    params['sample_size'] = sample_size
    params['length'] = L
    params['recomb_rate'] = r
    params['mutation_rate'] = u
    params['N0'] = N0
    params['NA'] = NA
    params['NB'] = NB
    params['td'] = td
    params['tm1'] = tm1
    params['tm2'] = tm2
    params['p_AB1'] = p_AB1
    params['p_AB2'] = p_AB2
    params['seed'] = seed
    y = np.array([N0, NA, NB, td, tm1, tm2, p_AB1, p_AB2])
    label = np.array([8])
    return tree, params, y, label


def sim_multi_pulse_uni_BA(L, params, locus_replicates=1):
    seed = params['seed']
    r = params['r']
    u = params['u']
    sample_size = params['sample_size']
    N0 = params['N0']
    NA = params['NA']
    NB = params['NB']
    td = params['td']
    tm2 = params['tm2']
    tm1 = params['tm1']
    p_BA1 = params['p_BA1']
    p_BA2 = params['p_BA2']
#     print('N0: {}'.format(N0))
#     print('NA: {}'.format(NA))
#     print('NB: {}'.format(NB))
#     print('tm1: {}'.format(tm1))
#     print('tm2: {}'.format(tm2))
#     print('td: {}'.format(td))
#     print('m_BA1: {}'.format(m_BA1))
#     print('m_BA2: {}'.format(m_BA2))
#     print('seed: {}'.format(seed))
    A, B = 1, 2
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N0),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size/2), initial_size=NA),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size/2), initial_size=NB)
    ]
    demographic_events = [
        msprime.MassMigration(
            time=tm1, source=B, destination=A, proportion=p_BA1),
        msprime.MassMigration(
            time=tm2, source=B, destination=A, proportion=p_BA2),
        msprime.MassMigration(
            time=td, source=A, destination=0, proportion=1.0),
        msprime.MassMigration(
            time=td, source=B, destination=0, proportion=1.0)
    ]
    tree = msprime.simulate(population_configurations=population_configurations,
                            demographic_events=demographic_events,
                            length=L,
                            recombination_rate=r,
                            mutation_rate=u,
                            random_seed=seed,
                            num_replicates=locus_replicates)
    params = OrderedDict()
    params['sample_size'] = sample_size
    params['length'] = L
    params['recomb_rate'] = r
    params['mutation_rate'] = u
    params['N0'] = N0
    params['NA'] = NA
    params['NB'] = NB
    params['td'] = td
    params['tm1'] = tm1
    params['tm2'] = tm2
    params['p_AB1'] = p_BA1
    params['p_AB2'] = p_BA2
    params['seed'] = seed
    y = np.array([N0, NA, NB, td, tm1, tm2, p_BA1, p_BA2])
    label = np.array([9])
    return tree, params, y, label


def sim_multi_pulse_bi(L, params, locus_replicates=1):
    seed = params['seed']
    r = params['r']
    u = params['u']
    sample_size = params['sample_size']
    
    N0 = params['N0']
    NA = params['NA']
    NB = params['NB']
    td = params['td']
    tm2 = params['tm2']
    tm1 = params['tm1']
    p_AB = params['p_AB']
    p_BA = params['p_BA']
#     print('N0: {}'.format(N0))
#     print('NA: {}'.format(NA))
#     print('NB: {}'.format(NB))
#     print('tm1: {}'.format(tm1))
#     print('tm2: {}'.format(tm2))
#     print('td: {}'.format(td))
#     print('m_AB: {}'.format(m_AB))
#     print('m_BA: {}'.format(m_BA))
#     print('seed: {}'.format(seed))
    A, B = 1, 2
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N0),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size/2), initial_size=NA),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size/2), initial_size=NB)
    ]
    demographic_events = [
        msprime.MassMigration(
            time=tm1, source=A, destination=B, proportion=p_AB),
        msprime.MassMigration(
            time=tm2, source=B, destination=A, proportion=p_BA),
        msprime.MassMigration(
            time=td, source=A, destination=0, proportion=1.0),
        msprime.MassMigration(
            time=td, source=B, destination=0, proportion=1.0)
    ]
    tree = msprime.simulate(population_configurations=population_configurations,
                            demographic_events=demographic_events,
                            length=L,
                            recombination_rate=r,
                            mutation_rate=u,
                            random_seed=seed,
                            num_replicates=locus_replicates)
    params = OrderedDict()
    params['sample_size'] = sample_size
    params['length'] = L
    params['recomb_rate'] = r
    params['mutation_rate'] = u
    params['N0'] = N0
    params['NA'] = NA
    params['NB'] = NB
    params['td'] = td
    params['tm1'] = tm1
    params['tm2'] = tm2
    params['p_AB'] = p_AB
    params['p_BA'] = p_BA
    params['seed'] = seed
    y = np.array([N0, NA, NB, td, tm1, tm2, p_AB, p_BA])
    label = np.array([10])
    return tree, params, y, label


def sim_continuous_uni_AB(L, params, locus_replicates=1):
    seed = params['seed']
    sample_size = params['sample_size']
    u = params['u']
    r = params['r']
    N0 = params['N0']
    NA = params['NA']
    NB = params['NB']
    td = params['td']
    tm2 = params['tm2']
    tm1 = params['tm1']
    m_AB = params['m_AB']
#     print('N0: {}'.format(N0))
#     print('NA: {}'.format(NA))
#     print('NB: {}'.format(NB))
#     print('tm1: {}'.format(tm1))
#     print('tm2: {}'.format(tm2))
#     print('td: {}'.format(td))
#     print('m_AB: {}'.format(m_AB))
#     print('seed: {}'.format(seed))
    A, B = 1, 2
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N0),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size/2), initial_size=NA),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size/2), initial_size=NB)
    ]
    migration_matrix = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    demographic_events = [
        msprime.MigrationRateChange(
            time=tm1, rate=m_AB, matrix_index=(2,1)
        ),
        msprime.MigrationRateChange(
            time=tm2, rate=0
        ),
        msprime.MassMigration(
            time=td, source=A, destination=0, proportion=1.0),
        msprime.MassMigration(
            time=td, source=B, destination=0, proportion=1.0)
    ]
#     dd = msprime.DemographyDebugger(
#         population_configurations=population_configurations,
#         demographic_events=demographic_events)
#     dd.print_history()
    tree = msprime.simulate(population_configurations=population_configurations,
                            demographic_events=demographic_events,
                            length=L,
                            recombination_rate=r,
                            mutation_rate=u,
                            random_seed=seed,
                            num_replicates=locus_replicates)
    params = OrderedDict()
    params['sample_size'] = sample_size
    params['length'] = L
    params['recomb_rate'] = r
    params['mutation_rate'] = u
    params['N0'] = N0
    params['NA'] = NA
    params['NB'] = NB
    params['td'] = td
    params['tm1'] = tm1
    params['tm2'] = tm2
    params['m_AB'] = m_AB
    params['seed'] = seed
    y = np.array([N0, NA, NB, td, tm1, tm2, m_AB])
    label = np.array([11])
    return tree, params, y, label


def sim_continuous_uni_BA(L, params, locus_replicates=1):
    seed = params['seed']
    sample_size = params['sample_size']
    u = params['u']
    r = params['r']
    N0 = params['N0']
    NA = params['NA']
    NB = params['NB']
    td = params['td']
    tm2 = params['tm2']
    tm1 = params['tm1']
    m_BA = params['m_BA']
#     print('N0: {}'.format(N0))
#     print('NA: {}'.format(NA))
#     print('NB: {}'.format(NB))
#     print('tm1: {}'.format(tm1))
#     print('tm2: {}'.format(tm2))
#     print('td: {}'.format(td))
#     print('m_BA: {}'.format(m_BA))
#     print('seed: {}'.format(seed))
    A, B = 1, 2
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N0),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size/2), initial_size=NA),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size/2), initial_size=NB)
    ]
    migration_matrix = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    demographic_events = [
        msprime.MigrationRateChange(
            time=tm1, rate=m_BA, matrix_index=(1,2)
        ),
        msprime.MigrationRateChange(
            time=tm2, rate=0
        ),
        msprime.MassMigration(
            time=td, source=A, destination=0, proportion=1.0),
        msprime.MassMigration(
            time=td, source=B, destination=0, proportion=1.0)
    ]
    tree = msprime.simulate(population_configurations=population_configurations,
                            demographic_events=demographic_events,
                            length=L,
                            recombination_rate=r,
                            mutation_rate=u,
                            random_seed=seed,
                            num_replicates=locus_replicates)
    params = OrderedDict()
    params['sample_size'] = sample_size
    params['length'] = L
    params['recomb_rate'] = r
    params['mutation_rate'] = u
    params['N0'] = N0
    params['NA'] = NA
    params['NB'] = NB
    params['td'] = td
    params['tm1'] = tm1
    params['tm2'] = tm2
    params['m_BA'] = m_BA
    params['seed'] = seed
    y = np.array([N0, NA, NB, td, tm1, tm2, m_BA])
    label = np.array([12])
    return tree, params, y, label


def sim_continuous_bi(L, params, locus_replicates=1):
    seed = params['seed']
    sample_size = params['sample_size']
    u = params['u']
    r = params['r']
    N0 = params['N0']
    NA = params['NA']
    NB = params['NB']
    td = params['td']
    tm2 = params['tm2']
    tm1 = params['tm1']
    m_AB = params['m_AB']
    m_BA = params['m_BA']
#     print('N0: {}'.format(N0))
#     print('NA: {}'.format(NA))
#     print('NB: {}'.format(NB))
#     print('tm1: {}'.format(tm1))
#     print('tm2: {}'.format(tm2))
#     print('td: {}'.format(td))
#     print('m_AB: {}'.format(m_AB))
#     print('m_BA: {}'.format(m_BA))
#     print('seed: {}'.format(seed))
    A, B = 1, 2
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N0),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size/2), initial_size=NA),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size/2), initial_size=NB)
    ]
    migration_matrix = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    demographic_events = [
        msprime.MigrationRateChange(
            time=tm1, rate=m_BA, matrix_index=(1,2)
        ),
        msprime.MigrationRateChange(
            time=tm1, rate=m_AB, matrix_index=(2,1)
        ),
        msprime.MigrationRateChange(
            time=tm2, rate=0
        ),
        msprime.MassMigration(
            time=td, source=A, destination=0, proportion=1.0),
        msprime.MassMigration(
            time=td, source=B, destination=0, proportion=1.0)
    ]

    tree = msprime.simulate(population_configurations=population_configurations,
                            demographic_events=demographic_events,
                            length=L,
                            recombination_rate=r,
                            mutation_rate=u,
                            random_seed=seed,
                            num_replicates=locus_replicates)
    params = OrderedDict()
    params['sample_size'] = sample_size
    params['length'] = L
    params['recomb_rate'] = r
    params['mutation_rate'] = u
    params['N0'] = N0
    params['NA'] = NA
    params['NB'] = NB
    params['td'] = td
    params['tm1'] = tm1
    params['tm2'] = tm2
    params['m_AB'] = m_AB
    params['m_BA'] = m_BA
    params['seed'] = seed
    y = np.array([N0, NA, NB, td, tm1, tm2, m_AB, m_BA])
    label = np.array([13])
    return tree, params, y, label


def sim_AJ_substruc_pulse_presplit(L, params, locus_replicates=1):
    seed = params['seed']
    r = params['r']
    u = params['u']
    sample_size = params['sample_size']
    NE = params['NE']
    NJ = params['NJ']
    NM = params['NM']
    NEA = params['NEA']
    NWA = params['NWA']
    NAg = params['NAg']
    TAg = params['TAg']
    TAEW = params['TAEW']
    Tm = params['Tm']
    TA = params['TA']
    TMJ = params['TMJ']
    TEM = params['TEM']
    m = params['m']
    ME, JM, AJ, A = 0, 2, 4, 6
    E, M, J, WA, EA = 1, 3, 5, 7, 8
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=NE),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size[E]), initial_size=NE),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=NM),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size[M]), initial_size=NM),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=NJ),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size[J]), initial_size=NJ),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=NAg),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size[EA]), initial_size=NEA),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size[WA]), initial_size=NWA)
    ]
    demographic_events = [
        # instantaneous growth in EA and WA at the same time
        msprime.PopulationParametersChange(
            time=TAg, initial_size=NAg, population_id=WA
        ),
        msprime.PopulationParametersChange(
            time=TAg, initial_size=NAg, population_id=EA
        ),
        # EA and WA split from A
        msprime.MassMigration(
            time=TAEW, source=EA, destination=A, proportion=1.0),
        msprime.MassMigration(
            time=TAEW, source=WA, destination=A, proportion=1.0),
        # E geneflow into AJ
        msprime.MassMigration(
            time=Tm, source=A, destination=E, proportion=m),
        # A and J split from AJ ancestor
        msprime.MassMigration(
            time=TA, source=A, destination=AJ, proportion=1.0),
        msprime.MassMigration(
            time=TA, source=J, destination=AJ, proportion=1.0),
        # J and M split from JM ancestor
        msprime.MassMigration(
            time=TMJ, source=AJ, destination=JM, proportion=1.0),
        msprime.MassMigration(
            time=TMJ, source=M, destination=JM, proportion=1.0),
        # M and E split from ME ancestor
        msprime.MassMigration(
            time=TEM, source=JM, destination=ME, proportion=1.0),
        msprime.MassMigration(
            time=TEM, source=E, destination=ME, proportion=1.0)
    ]
    # dd = msprime.DemographyDebugger(
    #     population_configurations=population_configurations,
    #     demographic_events=demographic_events)
    # dd.print_history()

    tree = msprime.simulate(population_configurations=population_configurations,
                            demographic_events=demographic_events,
                            length=L,
                            recombination_rate=r,
                            mutation_rate=u,
                            random_seed=seed,
                            num_replicates=locus_replicates)

    params = OrderedDict()
    params['sample_size'] = sample_size
    params['length'] = L
    params['recomb_rate'] = r
    params['mutation_rate'] = u
    params['NE'] = NE
    params['NJ'] = NJ
    params['NM'] = NM
    params['NEA'] = NEA
    params['NWA'] = NWA
    params['NAg'] = NAg
    params['TAg'] = TAg
    params['TAEW'] = TAEW
    params['Tm'] = Tm
    params['TA'] = TA
    params['TMJ'] = TMJ
    params['TEM'] = TEM
    params['m'] = m
    params['seed'] = seed
    y = np.array([NE, NJ, NM, NEA, NWA, NAg, TAg, TAEW, Tm, TA, TMJ, TEM, m])
    label = np.array([14])
    return tree, params, y, label


def sim_AJ_substruc_pulses_postsplit(L, params, locus_replicates=1):
    seed = params['seed']
    r = params['r']
    u = params['u']
    sample_size = params['sample_size']
    NE = params['NE']
    NJ = params['NJ']
    NM = params['NM']
    NEA = params['NEA']
    NWA = params['NWA']
    NAg = params['NAg']
    TAg = params['TAg']
    TAEW = params['TAEW']
    TmW = params['TmW']
    TmE = params['TmE']
    TA = params['TA']
    TMJ = params['TMJ']
    TEM = params['TEM']
    mW = params['mW']
    mE = params['mE']
    ME, JM, AJ, A = 0, 2, 4, 6
    E, M, J, WA, EA = 1, 3, 5, 7, 8
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=NE),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size[E]), initial_size=NE),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=NM),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size[M]), initial_size=NM),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=NJ),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size[J]), initial_size=NJ),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=NAg),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size[EA]), initial_size=NEA),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size[WA]), initial_size=NWA)
    ]
    demographic_events = [
        # instantaneous growth in EA and WA at the same time
        msprime.PopulationParametersChange(
            time=TAg, initial_size=NAg, population_id=WA
        ),
        msprime.PopulationParametersChange(
            time=TAg, initial_size=NAg, population_id=EA
        ),
        # E geneflow into WA
        msprime.MassMigration(
            time=TmW, source=WA, destination=E, proportion=mW),
        # E geneflow into EA
        msprime.MassMigration(
            time=TmE, source=EA, destination=E, proportion=mE),
        # EA and WA split from A
        msprime.MassMigration(
            time=TAEW, source=EA, destination=A, proportion=1.0),
        msprime.MassMigration(
            time=TAEW, source=WA, destination=A, proportion=1.0),
        # A and J split from AJ ancestor
        msprime.MassMigration(
            time=TA, source=A, destination=AJ, proportion=1.0),
        msprime.MassMigration(
            time=TA, source=J, destination=AJ, proportion=1.0),
        # J and M split from JM ancestor
        msprime.MassMigration(
            time=TMJ, source=AJ, destination=JM, proportion=1.0),
        msprime.MassMigration(
            time=TMJ, source=M, destination=JM, proportion=1.0),
        # M and E split from ME ancestor
        msprime.MassMigration(
            time=TEM, source=JM, destination=ME, proportion=1.0),
        msprime.MassMigration(
            time=TEM, source=E, destination=ME, proportion=1.0)
    ]

    demographic_events_sorted = sorted(demographic_events, key=lambda x: x.time)
    # dd = msprime.DemographyDebugger(
    #     population_configurations=population_configurations,
    #     demographic_events=demographic_events_sorted)
    # dd.print_history()

    tree = msprime.simulate(population_configurations=population_configurations,
                            demographic_events=demographic_events_sorted,
                            length=L,
                            recombination_rate=r,
                            mutation_rate=u,
                            random_seed=seed,
                            num_replicates=locus_replicates)

    params = OrderedDict()
    params['sample_size'] = sample_size
    params['length'] = L
    params['recomb_rate'] = r
    params['mutation_rate'] = u
    params['NE'] = NE
    params['NJ'] = NJ
    params['NM'] = NM
    params['NEA'] = NEA
    params['NWA'] = NWA
    params['NAg'] = NAg
    params['TAg'] = TAg
    params['TAEW'] = TAEW
    params['TmE'] = TmE
    params['TmW'] = TmW
    params['TA'] = TA
    params['TMJ'] = TMJ
    params['TEM'] = TEM
    params['mE'] = mE
    params['mW'] = mW
    params['seed'] = seed
    y = np.array([NE, NJ, NM, NEA, NWA, NAg, TAg, TAEW, TmE, TmW, TA, TMJ, TEM, mE, mW])
    label = np.array([15])
    return tree, params, y, label


def sim_AJ_substruc(L, params, locus_replicates=1):
    seed = params['seed']
    r = params['r']
    u = params['u']
    sample_size = params['sample_size']
    NE = params['NE']
    NJ = params['NJ']
    NM = params['NM']
    NEA = params['NEA']
    NWA = params['NWA']
    NAg = params['NAg']
    TAg = params['TAg']
    TAEW = params['TAEW']
    TA = params['TA']
    TMJ = params['TMJ']
    TEM = params['TEM']
    ME, JM, AJ, A = 0, 2, 4, 6
    E, M, J, WA, EA = 1, 3, 5, 7, 8
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=NE),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size[E]), initial_size=NE),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=NM),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size[M]), initial_size=NM),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=NJ),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size[J]), initial_size=NJ),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=NAg),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size[EA]), initial_size=NEA),
        msprime.PopulationConfiguration(
            sample_size=int(sample_size[WA]), initial_size=NWA)
    ]
    demographic_events = [
        # instantaneous growth in EA and WA at the same time
        msprime.PopulationParametersChange(
            time=TAg, initial_size=NAg, population_id=WA
        ),
        msprime.PopulationParametersChange(
            time=TAg, initial_size=NAg, population_id=EA
        ),
        # EA and WA split from A
        msprime.MassMigration(
            time=TAEW, source=EA, destination=A, proportion=1.0),
        msprime.MassMigration(
            time=TAEW, source=WA, destination=A, proportion=1.0),
        # A and J split from AJ ancestor
        msprime.MassMigration(
            time=TA, source=A, destination=AJ, proportion=1.0),
        msprime.MassMigration(
            time=TA, source=J, destination=AJ, proportion=1.0),
        # J and M split from JM ancestor
        msprime.MassMigration(
            time=TMJ, source=AJ, destination=JM, proportion=1.0),
        msprime.MassMigration(
            time=TMJ, source=M, destination=JM, proportion=1.0),
        # M and E split from ME ancestor
        msprime.MassMigration(
            time=TEM, source=JM, destination=ME, proportion=1.0),
        msprime.MassMigration(
            time=TEM, source=E, destination=ME, proportion=1.0)
    ]
    # dd = msprime.DemographyDebugger(
    #     population_configurations=population_configurations,
    #     demographic_events=demographic_events)
    # dd.print_history()

    tree = msprime.simulate(population_configurations=population_configurations,
                            demographic_events=demographic_events,
                            length=L,
                            recombination_rate=r,
                            mutation_rate=u,
                            random_seed=seed,
                            num_replicates=locus_replicates)

    params = OrderedDict()
    params['sample_size'] = sample_size
    params['length'] = L
    params['recomb_rate'] = r
    params['mutation_rate'] = u
    params['NE'] = NE
    params['NJ'] = NJ
    params['NM'] = NM
    params['NEA'] = NEA
    params['NWA'] = NWA
    params['NAg'] = NAg
    params['TAg'] = TAg
    params['TAEW'] = TAEW
    params['TA'] = TA
    params['TMJ'] = TMJ
    params['TEM'] = TEM
    params['seed'] = seed
    y = np.array([NE, NJ, NM, NEA, NWA, NAg, TAg, TAEW, TA, TMJ, TEM])
    label = np.array([16])
    return tree, params, y, label