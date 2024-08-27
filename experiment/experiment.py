import numpy as np
import pandas as pd
import os
import argparse
import pickle
import time
import jax
import jax.numpy as jnp

from qmc_flow.targets import StanModel, Gaussian
from qmc_flow.models import CopulaModel
from qmc_flow.train import optimize, optimize_variance

def get_mse(true_moments, est_moments):
    mse_1 = np.mean((true_moments[0] - est_moments[0])**2)
    mse_2 = np.mean((true_moments[1] - est_moments[1])**2)
    return mse_1, mse_2

def run_experiment(name, max_deg, nsample, sampler, max_iter, seed, savepath):
    if name in ['arK', 'hmm', 'garch', 'arma', 'eight-schools', 'normal-mixture', 'glmm-poisson']:
        data_path = f"qmc_flow/stan_models/{name}.json"
        stan_path = f"qmc_flow/stan_models/{name}.stan"
        target = StanModel(stan_path, data_path)
    else:
        if name == 'gaussian':
            d = 10
            mean = jnp.zeros(d)
            # cov = jnp.array([[1., 0.5], [0.5, 1.]])
            cov = (jnp.ones((d, d)) * 0.5 + jnp.eye(d) * 0.5) * 2.
            # cov = jnp.eye(d) * 3.
            target = Gaussian(mean, cov)
    d = target.d
    

    nf = CopulaModel(d, target, max_deg=max_deg)
    params = nf.init_params()
    # mu, L, weights = nf.unpack_params(params)
    # params = nf.pack_params(mu, L, weights)
    # params = params.at[d:d+d].set(jnp.log(3) / 2 + 1.)
    
    div = 'rkl'
    print("Training", div) 
    start = time.time()
    # opt_jax = jax.jit(lambda params: optimize(nf, params, div, max_iter=max_iter, nsample=nsample, seed=seed, sampler=sampler, max_lr=1.))
    # params, logs = opt_jax(params)
    params, logs = optimize(nf, params, div, max_iter=max_iter, nsample=nsample, seed=seed, sampler=sampler, max_lr=.1)
    end = time.time()
    print("Time elapsed", end - start)

    # X = sample_gaussian(nsample, d, seed=seed, sampler=sampler)
    # Z_nf, log_det = nf.forward_and_logdet(params, X)
    # log_q = -0.5 * jnp.sum(X**2, axis=-1) - 0.5 * d * jnp.log(2 * jnp.pi)
    # proposal_log_densities = log_q - log_det
    # target_log_densities = jax.vmap(target.log_prob)(Z_nf)
    # log_weights = target_log_densities - proposal_log_densities
    # log_weights -= np.max(log_weights)
    # weights = np.exp(log_weights)
    constrained = ~(name == 'gaussian')
    samples, weights = nf.sample(nsample, params, constrained=constrained, seed=seed, sampler=sampler, return_weights=True)
    print('ESS', np.sum(weights)**2 / np.sum(weights**2))

    mu, L, weights = nf.unpack_params(params)
    params2 = nf.pack_params(mu, L*2, weights)
    # traininig chisquare divergence with a heavy-tailed starting point


    # print("Training Chi-squared divergence")
    # params2, logs2 = optimize(nf, params, 'chisq', max_iter=10, nsample=nsample, seed=seed, sampler=sampler, max_lr=0.1)


    # results_lbfgs = {'time': lbfgs_time, 'losses': losses_lbfgs['kl'], 'ESS': losses_lbfgs['ESS'], 'moments': losses_lbfgs['moments']}
    # results = {'lbfgs': results_lbfgs}
    savepath = os.path.join(savepath, f'copula_{div}_{sampler}_n_{nsample}_deg_{max_deg}_iter_{max_iter}_{seed}.pkl')
    with open(savepath, 'wb') as f:
        pickle.dump(logs, f)
    print('saved to', savepath)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--date', type=str, default='2024-08-16')
    argparser.add_argument('--model_name', type=str, default='arK')
    argparser.add_argument('--max_iter', type=int, default=100)
    argparser.add_argument('--max_deg', type=int, default=3)
    argparser.add_argument('--m', type=int, default=10)
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--sampler', type=str, default='mc')
    argparser.add_argument('--rootdir', type=str, default='experiment/results')
    args = argparser.parse_args()

    nsample = 2**args.m
    savepath = os.path.join(args.rootdir, args.date, args.model_name)
    os.makedirs(savepath, exist_ok=True)

    run_experiment(args.model_name, args.max_deg, nsample, args.sampler, args.max_iter, args.seed, savepath)
