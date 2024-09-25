import numpy as np
from scipy.special import ndtri, stdtrit
from scipy.stats import qmc
from scipy.stats import multivariate_t
import matplotlib.pyplot as plt

MACHINE_EPSILON = np.finfo(np.float64).eps

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

def make_heatmap(target, i1=0, i2=1):
    grid_size = 200
    d = target.d
    zz = np.zeros(grid_size**2, d)
    xx, yy = np.meshgrid(np.linspace(-3, 3, grid_size), np.linspace(-3, 3, grid_size))
    zz[:, [i1, i2]] = np.concatenate([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)

    true_log_prob = target.log_prob(zz)
    true_prob = np.exp(true_log_prob.reshape(xx.shape))
    true_prob[np.isnan(true_prob)] = 0
    true_prob = true_prob / true_prob.sum() * (1 / (xx[1] - xx[0])**2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].pcolormesh(xx.numpy(), yy.numpy(), true_prob.data.numpy(), cmap='coolwarm', linewidth=0, rasterized=True)
    ax[0].set_title('Target')
    ax[0].set_xlabel('Dimension ' + str(i1))
    ax[0].set_ylabel('Dimension ' + str(i2))
    return fig, ax

