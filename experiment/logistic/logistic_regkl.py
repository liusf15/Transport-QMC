import numpy as np
import scipy
import pandas as pd
import os
import argparse
import pickle
import time
from tqdm import trange
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp, ndtri
from scipy.stats import qmc, t
from qmc_flow.targets import BayesianLogisticRegression
from qmc_flow.models.tqmc_AS import TransportQMC_AS
from qmc_flow.train import lbfgs, lbfgs_annealed, sgd
from qmc_flow.utils import get_effective_sample_size, get_moments, sample_uniform, sample_gaussian

MACHINE_EPSILON = np.finfo(np.float64).eps

def get_ref_moments(d, N, cov_x, rho):
    filename = f'qmc_flow/experiment/logistic/moments_logistic_{cov_x}_d_{d}_N_{N}_rho_{rho}.pkl'
    with open(filename, 'rb') as f:
        res = pickle.load(f)
    moments_1 = res['moments_1']
    moments_2 = res['moments_2']
    return np.mean(moments_1, axis=0), np.mean(moments_2, axis=0)

def run_experiment(d, cov_x, N=20, nsample=64, num_composition=1, max_deg=3, annealed=False, optimizer='lbfgs', max_iter=50, lr=1., savepath='results'):
    
    ######################## generate data ########################
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
    ref_moments_1, ref_moments_2 = get_ref_moments(d, N, cov_x, rho)

    target = BayesianLogisticRegression(X, y, prior_scale=1.)

    ######################## find active subspace ########################
    M = 512
    key = jax.random.key(0)
    x = jax.random.normal(key, (M, d))
    G = jax.vmap(jax.grad(target.log_prob))(x) 
    G = G + x
    eigval, eigvec = np.linalg.eigh(G.T @ G / M)
    eigval = eigval[::-1]
    eigvec = eigvec[:, ::-1]

    var_explained = np.cumsum(eigval) / np.sum(eigval)
    r = np.where(var_explained > 0.99)[0][0] + 2 # rank
    r = rho
    print('Rank', r)
    V = eigvec[:]

    ######################## training ########################
    model_as = TransportQMC_AS(d, r, V, target, base_transform='normal-icdf', nonlinearity='logit', num_composition=num_composition, max_deg=max_deg)
    params = model_as.init_params()

    if annealed:
        loss_fn = jax.jit(model_as.reg_kl)
    else:
        loss_fn = jax.jit(model_as.reverse_kl)

    callback_fn = jax.jit(model_as.metrics)

    best_params = params
    max_val_ess = 0.
    min_val_loss = np.Inf

    for seed in range(10):
        print("Seed", seed)
        params = model_as.init_params()

        rng = np.random.default_rng(seed)
        soboleng = qmc.Sobol(d, seed=rng)
        U = soboleng.random(nsample)
        U = jnp.array(U * (1 - MACHINE_EPSILON) + .5 * MACHINE_EPSILON)    

        soboleng = qmc.Sobol(d, seed=rng)
        U_val = soboleng.random(nsample)
        U_val = jnp.array(U_val * (1 - MACHINE_EPSILON) + .5 * MACHINE_EPSILON)
        callback = lambda params: callback_fn(params, U_val)

        if annealed:
            loss = lambda params, lbd: loss_fn(params, lbd, U)    
            final_state, logs = lbfgs_annealed(loss, params, max_iter=max_iter, anneal_iter=max_iter/2, max_lbd=1., callback=callback, max_lr=lr)
        else:
            loss = lambda params: loss_fn(params, U)    
            final_state, logs = lbfgs(loss, params, max_iter=max_iter, callback=callback, max_lr=lr)
        
        print('final rkl', logs.rkl[-1], 'final chisq', logs.chisq[-1], 'reg_rkl', logs.reg_rkl[-1], 'final ESS', logs.ess[-1])

        if logs.reg_rkl[-1] < min_val_loss:
            min_val_loss = logs.reg_rkl[-1]
            best_params = final_state[0]
            max_val_ess = logs.ess[-1]
    print(min_val_loss, max_val_ess)
    ######################## rotate ########################
    # find the AS of the function z -> log w(T(z)), where z ~ N(0, I)
    # so that we can rotate the Gaussian samples by 
    def log_w(z):
        x, log_det = model_as.forward_from_normal(best_params, z)
        log_p = target.log_prob(model_as.V @ x)
        log_q = -.5 * jnp.sum(z**2) - .5 * d * jnp.log(2 * jnp.pi) - log_det
        return log_p - log_q
    Z = sample_gaussian(512, d, seed=0, sampler='rqmc')
    G_2 = jax.vmap(jax.grad(log_w))(Z)
    eigval, eigvec = np.linalg.eigh(G_2.T @ G_2 / M)
    eigval = eigval[::-1]
    rot = eigvec[:, ::-1]
    print(eigval, rot)

    ######################## testing ########################
    @jax.jit
    def get_samples_rot(U, rot=None):
        Z = ndtri(U)
        if rot is not None:
            Z = Z @ rot.T
        X, log_det = jax.vmap(model_as.forward_from_normal, in_axes=(None, 0))(best_params, Z)
        X = X @ model_as.V.T
        log_p = jax.vmap(target.log_prob)(X)
        log_q = -.5 * jnp.sum(Z**2, axis=1) - .5 * d * jnp.log(2 * jnp.pi) - log_det
        log_weights = log_p - log_q
        log_weights -= jnp.nanmean(log_weights)
        weights = jnp.exp(log_weights)
        return X, weights

    def get_mse(moments_1, moments_2):
        mse_1 = (ref_moments_1 - moments_1)**2
        mse_2 = (ref_moments_2 - moments_2)**2
        return jnp.concat([mse_1, mse_2])
    
    methods = ['mc', 'rqmc', 'rqmc_as']
    m_list = np.arange(3, 14)
    mse = {}
    mse_IS = {}
    moments_1 = {}
    moments_2 = {}
    moments_1_IS = {}
    moments_2_IS = {}
    nrep = 50
    for i in trange(nrep, desc='Testing'):
        for m in m_list:
            for sampler in methods:
                if sampler in ['mc', 'rqmc']:
                    U = sample_uniform(2**m, d, rng, sampler)
                    X, weights = get_samples_rot(U, rot=None)
                else:
                    U = sample_uniform(2**m, d, rng, 'rqmc')
                    X, weights = get_samples_rot(U, rot=rot)
                # IS
                moment_1, moment_2 = get_moments(X, weights)
                moments_1_IS[(sampler, m, i)], moments_2_IS[(sampler, m, i)] = moment_1, moment_2
                mse_IS[(sampler, m, i)] = get_mse(moment_1, moment_2)

                # no IS
                moment_1, moment_2 = get_moments(X, None)
                moments_1[(sampler, m, i)], moments_2[(sampler, m, i)] = moment_1, moment_2
                mse[(sampler, m, i)] = get_mse(moment_1, moment_2)
            
    model_params = {
        'r': r,
        'V': V,
        'rkl': logs.rkl,
        'fkl': logs.fkl,
        'chisq': logs.chisq,
        'ess': logs.ess,
        'best_params': best_params,
    }
    test_results = {
        'mse': mse, 
        'moments_1': moments_1, 
        'moments_2': moments_2, 
        'mse_IS': mse_IS, 
        'moments_1_IS': moments_1_IS, 
        'moments_2_IS': moments_2_IS
    }
    results = {'model_params': model_params, 'test_results': test_results}
    savepath = os.path.join(savepath, f'regkl_d_{d}_N_{N}_{cov_x}_{rho}_n_{nsample}_comp_{num_composition}_deg_{max_deg}_{optimizer}_iter_{max_iter}_lr_{lr}.pkl')
    with open(savepath, 'wb') as f:
        pickle.dump(results, f)
    print('saved to', savepath)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--date', type=str, default='2024-09-24')
    argparser.add_argument('--d', type=int, default='20')
    argparser.add_argument('--cov_x', type=str, default='equi')
    argparser.add_argument('--N', type=int, default=20)
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--m', type=int, default=6)
    argparser.add_argument('--num_composition', type=int, default=1)
    argparser.add_argument('--max_deg', type=int, default=3)
    argparser.add_argument('--annealed', action='store_true')
    argparser.add_argument('--optimizer', type=str, default='lbfgs')
    argparser.add_argument('--max_iter', type=int, default=100)
    argparser.add_argument('--lr', type=float, default=1.)
    argparser.add_argument('--rootdir', type=str, default='experiment/results')

    args = argparser.parse_args()
    nsample = 2**args.m
    savepath = os.path.join(args.rootdir, args.date, 'logistic')
    os.makedirs(savepath, exist_ok=True)
    run_experiment(args.d, args.cov_x, N=args.N, nsample=nsample, num_composition=args.num_composition, max_deg=args.max_deg, annealed=args.annealed, optimizer=args.optimizer, max_iter=args.max_iter, lr=args.lr, savepath=savepath)
