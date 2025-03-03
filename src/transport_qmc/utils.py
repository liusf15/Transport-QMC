import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

MACHINE_EPSILON = np.finfo(np.float32).eps

def sample_uniform(nsample, d, rng, sampler):
    if sampler == 'mc':
        U = rng.random((nsample, d))
    else:
        soboleng =qmc.Sobol(d, scramble=True, seed=rng)    
        U = soboleng.random(nsample) * (1 - MACHINE_EPSILON) + MACHINE_EPSILON * .5
    return U

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

def get_moments(samples, weights=None):
    if weights is None:
        weights = np.ones(samples.shape[0]) / samples.shape[0]
    else:
        weights = weights / np.sum(weights)
    moment_1 = np.sum(samples * weights[:, None], axis=0)
    moment_2 = np.sum(samples**2 * weights[:, None], axis=0)
    return moment_1, moment_2

