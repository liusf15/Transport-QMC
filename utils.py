import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
MACHINE_EPSILON = np.finfo(np.float64).eps

def sample_gaussian(nsample, d, seed=0, sampler='rqmc'):
    if sampler == 'rqmc':
        soboleng = qmc.Sobol(d, scramble=True, seed=seed)
        X = ndtri(soboleng.random(nsample) * (1 - MACHINE_EPSILON) + .5 * MACHINE_EPSILON)
    elif sampler == 'mc':    
        if isinstance(seed, int):
            rng = np.random.default_rng(seed)
            X = rng.standard_normal((nsample, d))
        else:
            X = seed.standard_normal((nsample, d))
    else:
        raise NotImplementedError
    return X
