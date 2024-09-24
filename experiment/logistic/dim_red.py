import numpy as np
import scipy
import pandas as pd
import os
import argparse
import pickle
import time
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from scipy.stats import qmc, t
from qmc_flow.targets import BayesianLogisticRegression
from qmc_flow.models.tqmc_AS import TransportQMC_AS
from qmc_flow.train import lbfgs, sgd
from qmc_flow.utils import get_effective_sample_size, get_moments

MACHINE_EPSILON = np.finfo(np.float64).eps

def run_experiment(d, cov_x, N=20, seed=1, nsample=64, num_composition=1, max_deg=3, optimizer='lbfgs', max_iter=50, lr=1., savepath='results'):
    
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
    target = BayesianLogisticRegression(X, y, prior_scale=1.)

    M = 512
    # x = rng.standard_normal((M, d))
    key = jax.random.key(seed)
    x = jax.random.normal(key, (M, d))
    G = jax.vmap(jax.grad(target.log_prob))(x) 
    G = G + x
    eigval, eigvec = np.linalg.eigh(G.T @ G / M)
    eigval = eigval[::-1]
    eigvec = eigvec[:, ::-1]

    var_explained = np.cumsum(eigval) / np.sum(eigval)
    r = np.where(var_explained > 0.95)[0][0] + 1 # rank
    print('Rank', r)
    V = eigvec[:]

    model_as = TransportQMC_AS(d, r, V, target, base_transform='normal-icdf', nonlinearity='logit', num_composition=num_composition, max_deg=max_deg)
    params = model_as.init_params()

    rng = np.random.default_rng(seed)
    soboleng = qmc.Sobol(d, seed=rng)
    U = soboleng.random(nsample)
    U = jnp.array(U * (1 - MACHINE_EPSILON) + .5 * MACHINE_EPSILON)    
    loss_fn = jax.jit(lambda params: model_as.reverse_kl(params, U))

    soboleng = qmc.Sobol(d, seed=rng)
    U_val = soboleng.random(nsample)
    U_val = jnp.array(U_val * (1 - MACHINE_EPSILON) + .5 * MACHINE_EPSILON)
    callback = jax.jit(lambda params: model_as.metrics(params, U_val))

    start = time.time()
    if optimizer == 'lbfgs':
        final_state, logs = lbfgs(loss_fn, params, max_iter=max_iter, callback=callback, max_lr=lr)
    else:
        final_state, logs = sgd(loss_fn, params, max_iter=max_iter, lr=lr, callback=callback)
    end = time.time()
    print('Time elapsed', end - start)

    params = final_state[0]
    best_params = final_state[2]
    max_ess = final_state[3]
    print('final rkl', logs.rkl[-1])
    print('final ESS', logs.ess[-1])
    print('Max ESS', max_ess)

    ####################################### !! TODO !! #######################################
    # Mixture sampling 
    # optimize weight: E_q (p/q)^2, where q=alpha q_theta + (1-alpha) tilde q, where tilde q is the logistic distribution
    soboleng = qmc.Sobol(d, scramble=True, seed=rng)
    X = soboleng.random(2**10) * (1 - MACHINE_EPSILON) + MACHINE_EPSILON * .5
    X, log_det = jax.vmap(model_as.forward, in_axes=(None, 0))(best_params, X)
    log_q = - log_det
    X = X @ V.T
    log_p = jax.vmap(target.log_prob)(X)
    # log_l = jnp.sum(-X - 2 * jax.nn.softplus(-X), axis=1)
    
    @jax.jit
    def mixture_chisq(refine_params):
        loc = refine_params['loc']
        logscale = refine_params['logscale']
        alpha = refine_params['alpha']

        X_ = (X - loc) / jnp.exp(logscale)
        log_l = jnp.sum(jax.scipy.stats.t.logpdf(X_, df=5), axis=1) - jnp.sum(logscale)
        log_q_mix = logsumexp(jnp.array([log_q + jnp.log(1-alpha), log_l + jnp.log(alpha)]), axis=0) 
        log_weights = 2 * log_p - log_q - log_q_mix
        return logsumexp(log_weights) - jnp.log(len(log_weights))
    
    alpha = jnp.ones(1) * 0.1
    loc = jnp.zeros(d)
    logscale = jnp.zeros(d)
    refine_params = {'alpha': alpha, 'loc': loc, 'logscale': logscale}
    # loss_fn_refine = jax.jit(lambda refine_params: mixture_chisq(refine_params))
    refine_params, logs_2 = lbfgs(mixture_chisq, refine_params, max_iter=50)
    
    alpha = refine_params[0]['alpha'][0]
    loc = refine_params[0]['loc']
    logscale = refine_params[0]['logscale']

    # sample from mixture, 
    soboleng = qmc.Sobol(d+1, scramble=True, seed=rng)
    U_mixture = soboleng.random(2**10)
    idx = U_mixture[:, 0] < alpha # sample from l
    Z_1 = t.ppf(U_mixture[idx, 1:], df=5) * np.exp(logscale) + loc
    Z_2 = jax.vmap(model_as.forward, in_axes=(None, 0))(best_params, U_mixture[~idx, 1:])[0]
    Z_mixture = np.zeros_like(U_mixture[:, 1:])
    Z_mixture[idx] = Z_1
    Z_mixture[~idx] = Z_2
    # how to evaluate q_theta? Need inverse transform


    # samples, weights = model_as.sample(best_params, 2**16, seed)
    # get_moments(samples, weights)
    results = {
        'params': params,
        'r': r,
        'V': V,
        'time': end - start,
        'rkl': logs.rkl,
        'fkl': logs.fkl,
        'chisq': logs.chisq,
        'ess': logs.ess,
        'best_params': best_params,
        'max_ess': max_ess
    }
    savepath = os.path.join(savepath, f'logistic_d_{d}_N_{N}_{cov_x}_{rho}_n_{nsample}_comp_{num_composition}_deg_{max_deg}_{optimizer}_iter_{max_iter}_lr_{lr}_{seed}.pkl')
    with open(savepath, 'wb') as f:
        pickle.dump(results, f)
    print('saved to', savepath)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--date', type=str, default='2024-09-22')
    argparser.add_argument('--d', type=int, default='20')
    argparser.add_argument('--cov_x', type=str, default='equi')
    argparser.add_argument('--N', type=int, default=20)
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--m', type=int, default=6)
    argparser.add_argument('--num_composition', type=int, default=1)
    argparser.add_argument('--max_deg', type=int, default=3)
    argparser.add_argument('--optimizer', type=str, default='lbfgs')
    argparser.add_argument('--max_iter', type=int, default=100)
    argparser.add_argument('--lr', type=float, default=1.)
    argparser.add_argument('--rootdir', type=str, default='experiment/results')

    args = argparser.parse_args()
    nsample = 2**args.m
    savepath = os.path.join(args.rootdir, args.date, 'logistic')
    os.makedirs(savepath, exist_ok=True)
    run_experiment(args.d, args.cov_x, N=args.N, seed=args.seed, nsample=nsample, num_composition=args.num_composition, max_deg=args.max_deg, optimizer=args.optimizer, max_iter=args.max_iter, lr=args.lr, savepath=savepath)
