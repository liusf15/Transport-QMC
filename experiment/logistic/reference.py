import os
import numpy as np
import jax.numpy as jnp
import scipy
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from tqdm import trange
import jax
import argparse
import pickle
from qmc_flow.targets import BayesianLogisticRegression
from qmc_flow.utils import get_effective_sample_size, get_moments

MACHINE_EPS = np.finfo(np.float64).eps

def run(d, N, cov_x):
    rng = np.random.default_rng(0)
    if cov_x == 'equi':
        rho = 0.7
        cov_X = np.eye(d) * (1 - rho) + rho
        chol_x = np.linalg.cholesky(cov_X)
    elif cov_x == 'ar1':
        rho = 0.9
        cov_X = scipy.linalg.toeplitz(rho ** np.arange(d))
        chol_x = np.linalg.cholesky(cov_X)
    elif cov_x == 'low_rank':
        rho = 3
        D = np.ones(d) * .1
        D[:rho] = np.array([3., 2., 1.])
        orth = np.linalg.qr(rng.standard_normal((d, d)))[0]
        chol_x = orth.T @ np.diag(D)
    else:
        raise NotImplementedError

    X = rng.standard_normal((N, d)) @ chol_x.T
    beta = rng.random(d) * 2 - 1
    y = rng.binomial(1, 1 / (1 + np.exp(-jnp.dot(X, beta))))
    target = BayesianLogisticRegression(X, y, prior_scale=1.)

    def obj(mu):
        return np.sum(np.log(1 + np.exp(X @ mu))) - y.T @ (X @ mu) + np.dot(mu, mu) / 2
        
    res = minimize(obj, np.zeros(d))
    shift = res.x

    def obj_jax(mu):
        return jnp.sum(jnp.log(1 + jnp.exp(X @ mu))) - y.T @ (X @ mu) + jnp.dot(mu, mu) / 2

    jax.grad(obj_jax)(res.x)
    hess = jax.jacobian(jax.grad(obj_jax))(res.x)
    Sigma = np.linalg.inv(hess)
    chol = np.linalg.cholesky(Sigma)

    def simu(Z, shift, chol):
        log_q = -np.sum(Z**2, axis=1) / 2
        beta = Z @ chol.T + shift
        log_p = jax.vmap(target.log_prob)(beta)
        log_weights = log_p - log_q
        log_weights -= np.nanmean(log_weights)
        weights = np.exp(log_weights)
        moment_1, moment_2 = get_moments(beta, weights)
        ess = get_effective_sample_size(weights)
        return moment_1, moment_2, ess

    nrep = 1000
    n = 2**16
    moments_1 = np.zeros((nrep, d))
    moments_2 = np.zeros((nrep, d))
    ess = np.zeros(nrep)
    for i in trange(nrep):
        sobol_eng = qmc.Sobol(d, scramble=True, seed=i**2)
        Z = sobol_eng.random(n)
        Z = norm.ppf(Z * (1 - MACHINE_EPS) + .5 * MACHINE_EPS)
        moments_1[i], moments_2[i], ess[i] = simu(Z, shift, chol)

    moments_1.mean(0), moments_2.mean(0), ess.mean()
    moments_1.std(0) / np.sqrt(nrep), moments_2.std(0) / np.sqrt(nrep)
    results = {'moments_1': moments_1, 'moments_2': moments_2, 'ess': ess}
    filename = f'qmc_flow/experiment/logistic/moments_logistic_{cov_x}_d_{d}_N_{N}_rho_{rho}_n_{n}_rep_{nrep}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print("Saved to", filename) 

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--d', type=int, default=50)
    argparser.add_argument('--N', type=int, default=20)
    argparser.add_argument('--cov_x', type=str, default='equi', choices=['equi', 'ar1', 'low_rank'])

    args = argparser.parse_args()

    run(args.d, args.N, args.cov_x)
