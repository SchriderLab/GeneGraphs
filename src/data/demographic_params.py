import random
import math


def define_params(fixed_params=None):
    priors = {}

    if fixed_params and fixed_params['seed']:
        seed = fixed_params['seed']
    else:
        seed = random.randint(1, 2 ** 32 - 1)

    if fixed_params and fixed_params['sample_size']:
        sample_size = fixed_params['sample_size']
    else:
        sample_size = 50

    if fixed_params and fixed_params['u']:
        u = fixed_params['u']
    else:
        u = 1.2e-8
    r = 1e-8 # change to 1e-11

    if fixed_params and fixed_params['N0']:
        N0 = fixed_params['N0']
    else:
        priors['N0'] = (1000, 40000)
        N0 = selectVal(priors['N0'][0], priors['N0'][1])

    if fixed_params and fixed_params['N1']:
        N1 = fixed_params['N1']
    else:
        priors['N1'] = (1, 10)
        N1 = N0 / selectVal(priors['N1'][0], priors['N1'][1])

    if fixed_params and fixed_params['NA']:
        NA = fixed_params['NA']
    else:
        priors['NA'] = (1000, 40000)
        NA = selectVal(priors['NA'][0], priors['NA'][1])

    if fixed_params and fixed_params['NB']:
        NB = fixed_params['NB']
    else:
        priors['NB'] = (1000, 40000)
        NB = selectVal(priors['NB'][0], priors['NB'][1])

    if fixed_params and fixed_params['t1']:
        t1 = fixed_params['t1']
    else:
        priors['t1'] = (100, 3499.99)
        t1 = selectVal(priors['t1'][0], priors['t1'][1])

    if fixed_params and fixed_params['td']:
        td = fixed_params['td']
    else:
        # original:
        priors['td'] = (100, 8000)
        # changed to the line below:
        # priors['td'] = (4000, 40000)
        td = selectVal(priors['td'][0], priors['td'][1])

    if fixed_params and fixed_params['tm']:
        tm = fixed_params['tm']
    else:
        # reverse comments for next 4 lines
        # priors['tm'] = (0, 1) # comment out for test
        # tm = selectVal(priors['tm'][0], priors['tm'][1]) * td # comment out for test
        priors['tm'] = 1 # temporarily change to priors['tm'] = (0, 1)
        tm = td # temporarily changed from tm = selectVal(priors['tm'][0], priors['tm'][1]) * td
        # tm = 0 # changed from above

    if fixed_params and fixed_params['tm2']:
        tm2 = fixed_params['tm2']
    else:
        # priors['tm2'] = (2, td)
        priors['tm2'] = (0, 1)
        tm2 = selectVal(priors['tm2'][0], priors['tm2'][1]) * td
        # tm2 = selectVal(priors['tm2'][0], priors['tm2'][1])

    if fixed_params and fixed_params['tm1']:
        tm1 = fixed_params['tm1']
    else:
        # priors['tm1'] = (0, tm2)
        priors['tm1'] = (0, 1)
        tm1 = selectVal(priors['tm1'][0], priors['tm1'][1]) * tm2
        # tm1 = selectVal(priors['tm1'][0], priors['tm1'][1])

    if fixed_params and fixed_params['p_AB']:
        p_AB = fixed_params['p_AB']
    else:
        priors['p_AB'] = (0.1, 0.9)
        p_AB = selectVal(priors['p_AB'][0], priors['p_AB'][1])

    if fixed_params and fixed_params['p_BA']:
        p_BA = fixed_params['p_BA']
    else:
        priors['p_BA'] = (0.1, 0.9)
        p_BA = selectVal(priors['p_BA'][0], priors['p_BA'][1])

    if fixed_params and fixed_params['p_AB1']:
        p_AB1 = fixed_params['p_AB1']
    else:
        priors['p_AB1'] = (0.1, 0.5)
        p_AB1 = selectVal(priors['p_AB1'][0], priors['p_AB1'][1])

    if fixed_params and fixed_params['p_AB2']:
        p_AB2 = fixed_params['p_AB2']
    else:
        priors['p_AB2'] = (0.1, 0.5)
        p_AB2 = selectVal(priors['p_AB2'][0], priors['p_AB2'][1])

    if fixed_params and fixed_params['p_BA1']:
        p_BA1 = fixed_params['p_BA1']
    else:
        priors['p_BA1'] = (0.1, 0.5)
        p_BA1 = selectVal(priors['p_BA1'][0], priors['p_BA1'][1])

    if fixed_params and fixed_params['p_BA2']:
        p_BA2 = fixed_params['p_BA2']
    else:
        priors['p_BA2'] = (0.1, 0.5)
        p_BA2 = selectVal(priors['p_BA2'][0], priors['p_BA2'][1])

    if fixed_params and fixed_params['m_AB']:
        m_AB = fixed_params['m_AB']
    else:
        priors['m_AB'] = (0, 1 / (tm2 - tm1))
        m_AB = selectVal(priors['m_AB'][0], priors['m_AB'][1])

    if fixed_params and fixed_params['m_BA']:
        m_BA = fixed_params['m_BA']
    else:
        priors['m_BA'] = (0, 1 / (tm2 - tm1))
        m_BA = selectVal(priors['m_BA'][0], priors['m_BA'][1])

    params = {
        'seed': seed,
        'sample_size': sample_size,
        'u': u,
        'r': r,
        'N0': N0,
        'N1': N1,
        'NA': NA,
        'NB': NB,
        't1': t1,
        'td': td,
        'tm': tm,
        'tm1': tm1,
        'tm2': tm2,
        'p_AB': p_AB,
        'p_BA': p_BA,
        'p_AB1': p_AB1,
        'p_AB2': p_AB2,
        'p_BA1': p_BA1,
        'p_BA2': p_BA2,
        'm_AB': m_AB,
        'm_BA': m_BA
    }
    return params, priors


def get_N_param(fixed_params, priors, N_name, min, max):
    if fixed_params and fixed_params[N_name]:
        param_value = fixed_params[N_name]
    else:
        priors[N_name] = (min, max)
        param_value = float(round(10 ** random.uniform(priors[N_name][0], priors[N_name][1])))
    return param_value


def get_N_param(fixed_params, priors, N_name, min, max):
    if fixed_params and fixed_params[N_name]:
        param_value = fixed_params[N_name]
    else:
        priors[N_name] = (min, max)
        param_value = float(round(10 ** random.uniform(priors[N_name][0], priors[N_name][1])))
    return param_value, priors


def get_m_param(fixed_params, priors, m_name, min, max):
    if fixed_params and fixed_params[m_name]:
        param_value = fixed_params[m_name]
    else:
        priors[m_name] = (min, max)
        param_value = random.uniform(priors[m_name][0], priors[m_name][1])
    return param_value, priors


def get_T_param(fixed_params, priors, T_name, min, max):
    if fixed_params and fixed_params[T_name]:
        param_value = fixed_params[T_name]
    else:
        priors[T_name] = (min, max)
        param_value = float(random.randint(priors[T_name][0], priors[T_name][1]))
    return param_value, priors


def define_AJ_params(fixed_params=None):
    priors = {}

    if fixed_params and fixed_params['seed']:
        seed = fixed_params['seed']
    else:
        seed = random.randint(1, 2 ** 32 - 1)

    if fixed_params and fixed_params['sample_size']:
        sample_size = fixed_params['sample_size']
    else:
        sample_size = [0, 24, 0, 28, 0, 28, 0, 38, 38]

    if fixed_params and fixed_params['u']:
        u = fixed_params['u']
    else:
        u = 1.2e-8
    r = 1e-8

    NE, priors = get_N_param(fixed_params, priors, 'NE', min=3, max=5)
    NWA, priors = get_N_param(fixed_params, priors, 'NWA', min=3, max=6.7)
    NEA, priors = get_N_param(fixed_params, priors, 'NEA', min=4, max=6.7)
    NAg, priors = get_N_param(fixed_params, priors, 'NAg', min=2, max=min(math.log10(NWA), math.log10(NEA)))
    NJ, priors = get_N_param(fixed_params, priors, 'NJ', min=3, max=6)
    NM, priors = get_N_param(fixed_params, priors, 'NM', min=3, max=6)
    m, priors = get_m_param(fixed_params, priors, 'm', min=0, max=1.0)
    mE, priors = get_m_param(fixed_params, priors, 'mE', min=0, max=1.0)
    mW, priors = get_m_param(fixed_params, priors, 'mW', min=0, max=1.0)
    TEM, priors = get_T_param(fixed_params, priors, 'TEM', min=400, max=1200)
    TA, priors = get_T_param(fixed_params, priors, 'TA', min=20, max=36)
    TMJ, priors = get_T_param(fixed_params, priors, 'TMJ', min=int(TA) + 1, max=int(TEM) - 1)
    TAEW, priors = get_T_param(fixed_params, priors, 'TAEW', min=2, max=int(TA) - 2)
    Tm, priors = get_T_param(fixed_params, priors, 'Tm', min=int(TAEW) + 1, max=int(TA) - 1)
    TmE, priors = get_T_param(fixed_params, priors, 'TmE', min=1, max=int(TAEW) - 1)
    TmW, priors = get_T_param(fixed_params, priors, 'TmW', min=1, max=int(TAEW) - 1)
    TAg, priors = get_T_param(fixed_params, priors, 'TAg', min=1, max=int(TAEW) - 1)

    params = {'seed': seed,
              'sample_size': sample_size,
              'u': u,
              'r': r,
              'NE': NE,
              'NJ': NJ,
              'NM': NM,
              'NEA': NEA,
              'NWA': NWA,
              'NAg': NAg,
              'TAg': TAg,
              'TAEW': TAEW,
              'Tm': Tm,
              'TmE': TmE,
              'TmW': TmW,
              'TA': TA,
              'TMJ': TMJ,
              'TEM': TEM,
              'm': m,
              'mE': mE,
              'mW': mW}
    return params, priors


def selectVal(minVal, maxVal):
    size = maxVal - minVal
    return (random.random() * size) + minVal
