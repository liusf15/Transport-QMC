import os
import argparse
import pandas as pd
import numpy as np
import jax.numpy as jnp
import scipy
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from tqdm import trange
import jax
import pickle
from qmc_flow.targets import BayesianLogisticRegression
from qmc_flow.utils import get_effective_sample_size, get_moments
from qmc_flow.models.tqmc_AS import TransportQMC_AS

MACHINE_EPSILON = np.finfo(np.float64).eps

def get_ref_moments(d, N, cov_x, rho):
    filename = f'qmc_flow/experiment/logistic/moments_logistic_{cov_x}_d_{d}_N_{N}_rho_{rho}.pkl'
    with open(filename, 'rb') as f:
        res = pickle.load(f)
    moments_1 = res['moments_1']
    moments_2 = res['moments_2']
    return np.mean(moments_1, axis=0), np.mean(moments_2, axis=0)

def get_best_params(path, d, N, cov_x, rho, nsample, num_composition, max_deg, max_iter):
    max_ess = -1
    for seed in range(10):
        filename = os.path.join(path, f'logistic_d_{d}_N_{N}_{cov_x}_{rho}_n_{nsample}_comp_{num_composition}_deg_{max_deg}_lbfgs_iter_{max_iter}_lr_1.0_{seed}.pkl')
        if not os.path.exists(filename):
            continue
        with open(filename, 'rb') as f:
            res = pickle.load(f)
        if res['ess'][-1] > max_ess:
            max_ess = res['ess'][-1]
            params = res['params']
            r = res['r']
            V = res['V']
    print(f'Best ESS: {max_ess}')
    return params, r, V


def run_experiment(d, N, cov_x, seed=0, ntrain=64, train_date='2024-09-22', num_composition=1, max_deg=3, max_iter=200, savepath='results'):
    
    rng = np.random.default_rng(0)
    if cov_x == 'equi':
        rho = 0.7
        cov_X = np.eye(d) * (1 - rho) + rho
    elif cov_x == 'ar1':
        rho = 0.9
        cov_X = scipy.linalg.toeplitz(rho ** np.arange(d))
    else:
        cov_X = np.eye(d)
    chol_X = np.linalg.cholesky(cov_X)
    X = rng.standard_normal((N, d)) @ chol_X.T
    beta = rng.random(d) * 2 - 1
    y = rng.binomial(1, 1 / (1 + np.exp(-jnp.dot(X, beta))))
    ref_moments_1, ref_moments_2 = get_ref_moments(d, N, cov_x, rho)

    target = BayesianLogisticRegression(X, y, prior_scale=1.)

    rootdir = '/mnt/ceph/users/sliu1/normalizing_flows/qmc_flow'
    path = os.path.join(rootdir, train_date, 'logistic')
    params, r, V = get_best_params(path, d, N, cov_x, rho, nsample=ntrain, num_composition=num_composition, max_deg=max_deg, max_iter=max_iter)
    model = TransportQMC_AS(d, r, V, target, base_transform='normal-icdf', nonlinearity='logit', num_composition=num_composition, max_deg=max_deg)


    def sample_uniform(nsample, d, rng, sampler):
        if sampler == 'mc':
            U = rng.random((nsample, d))
        else:
            soboleng =qmc.Sobol(d, scramble=True, seed=rng)    
            U = soboleng.random(nsample) * (1 - MACHINE_EPSILON) + MACHINE_EPSILON * .5
        return U

    @jax.jit
    def get_samples(U):
        X, log_det = jax.vmap(model.forward, in_axes=(None, 0))(params, U)
        proposal_log_densities = - log_det
        X = X @ model.V.T
        target_log_densities = jax.vmap(target.log_prob)(X)
        log_weights = target_log_densities - proposal_log_densities
        log_weights -= jnp.nanmean(log_weights)
        weights = jnp.exp(log_weights)
        return X, weights

    def get_moments(X, weights):
        if getattr(target, 'param_constrain', None):
            X = target.param_constrain(np.array(X, float))
        moments_1 = jnp.sum(X * weights[:, None], axis=0) / jnp.sum(weights)
        moments_2 = jnp.sum(X**2 * weights[:, None], axis=0) / jnp.sum(weights)
        return moments_1, moments_2

    def get_mse(moments_1, moments_2):
        mse_1 = (ref_moments_1 - moments_1)**2
        mse_2 = (ref_moments_2 - moments_2)**2
        return jnp.concat([mse_1, mse_2])
    
    m_list = np.arange(3, 16)
    rng = np.random.default_rng(2024+seed)
    mse = {}
    moments_1 = {}
    moments_2 = {}

    for m in m_list:
        print(m)
        # MC
        U = sample_uniform(2**m, d, rng, 'mc')
        X, weights = get_samples(U)
        moment_1, moment_2 = get_moments(X, weights)
        moments_1[('mc', m, seed)], moments_2[('mc', m, seed)] = moment_1, moment_2
        mse[('mc', m, seed)] = get_mse(moment_1, moment_2)

        # RQMC
        U = sample_uniform(2**m, d, rng, 'rqmc')
        X, weights = get_samples(U)
        moment_1, moment_2 = get_moments(X, weights)
        moments_1[('rqmc', m, seed)], moments_2[('rqmc', m, seed)] = moment_1, moment_2
        mse[('rqmc', m, seed)] = get_mse(moment_1, moment_2)
    
    filename = os.path.join(savepath, f'{cov_x}_d_{d}_N_{N}_ntrain_{ntrain}_comp_{num_composition}_deg_{max_deg}_iter_{max_iter}_seed_{seed}.pkl')
    mse = pd.DataFrame(mse).T.reset_index(names=['sampler', 'm', 'seed'])
    moments_1 = pd.DataFrame(moments_1).T.reset_index(names=['sampler', 'm', 'seed'])
    moments_2 = pd.DataFrame(moments_2).T.reset_index(names=['sampler', 'm', 'seed'])
    results = {'mse': mse, 'moments_1': moments_1, 'moments_2': moments_2}
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print('saved to', filename)
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--date', type=str, default='2024-09-22')
    argparser.add_argument('--d', type=int, default=50)
    argparser.add_argument('--N', type=int, default=20)
    argparser.add_argument('--cov_x', type=str, default='equi')
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--ntrain', type=int, default=64)
    argparser.add_argument('--num_composition', type=int, default=1)
    argparser.add_argument('--max_deg', type=int, default=3)
    argparser.add_argument('--max_iter', type=int, default=200)
    argparser.add_argument('--rootdir', type=str, default='experiment/results')

    args = argparser.parse_args()

    savepath = os.path.join(args.rootdir, args.date, 'logistic', 'mse')
    os.makedirs(savepath, exist_ok=True)
    run_experiment(args.d, args.N, args.cov_x, seed=args.seed, ntrain=args.ntrain, train_date='2024-09-22', num_composition=args.num_composition, max_deg=args.max_deg, max_iter=args.max_iter, savepath=savepath)
