import numpy as np
from scipy.special import ndtri, stdtrit
from scipy.stats import qmc
from scipy.stats import multivariate_t, mvn
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

def sample_t(nsample, d, df, seed=0, sampler='rqmc', return_logp=False):
    if sampler == 'rqmc':
        soboleng = qmc.Sobol(d, scramble=True, seed=seed)
        X = stdtrit(df, (soboleng.random(nsample) * (1 - MACHINE_EPSILON) + .5 * MACHINE_EPSILON))
    elif sampler == 'mc':    
        if isinstance(seed, int):
            rng = np.random.default_rng(seed)
            X = rng.standard_t(df, size=(nsample, d))
        else:
            X = seed.standard_t(df, size=(nsample, d))
    else:
        raise NotImplementedError
    if return_logp:
        if df != np.Inf:
            logp = multivariate_t.logpdf(X, loc=np.zeros(d), shape=np.eye(d), df=df)
        else:
            logp = -0.5 * np.sum(X**2, axis=-1) - 0.5 * d * np.log(2 * np.pi)
        return X, logp
    return X

def get_moments(samples, weights=None):
    if weights is None:
        weights = np.ones(samples.shape[0]) / samples.shape[0]
    else:
        weights = weights / np.sum(weights)
    moment_1 = np.sum(samples * weights[:, None], axis=0)
    moment_2 = np.sum(samples**2 * weights[:, None], axis=0)
    return moment_1, moment_2

def get_effective_sample_size(weights):
    return np.sum(weights)**2 / np.sum(weights**2)

def get_mse(true_moments, est_moments):
    mse_1 = np.mean((true_moments[0] - est_moments[0])**2)
    mse_2 = np.mean((true_moments[1] - est_moments[1])**2)
    return mse_1, mse_2
