import numpy as np
import pandas as pd
import os
import argparse
import pickle
import time
from scipy.stats import qmc
from scipy.special import ndtri
from scipy.optimize import minimize
import jax
import jax.numpy as jnp

from experiment.polynomial.targets import StanModel, Gaussian
from qmc_flow.models import PolynomialModel, CopulaModel
from qmc_flow.train import optimize, optimize_variance
from qmc_flow.utils import sample_gaussian

import bridgestan as bs

def get_mse(true_moments, est_moments):
    mse_1 = np.mean((true_moments[0] - est_moments[0])**2)
    mse_2 = np.mean((true_moments[1] - est_moments[1])**2)
    return mse_1, mse_2

def run_experiment(name, max_deg, nsample, sampler, max_iter, seed, savepath):
    if name in ['arK', 'hmm', 'garch', 'eight-schools']:
        data_path = f"qmc_flow/stan_models/{name}.json"
        stan_path = f"qmc_flow/stan_models/{name}.stan"
        target = StanModel(stan_path, data_path)
    else:
        if name == 'gaussian':
            d = 2
            mean = jnp.zeros(d)
            # cov = jnp.array([[1., 0.5], [0.5, 1.]])
            # cov = (jnp.ones((d, d)) * 0.5 + jnp.eye(d) * 0.5) * 2.
            cov = jnp.eye(d) * 3.
            target = Gaussian(mean, cov)
    d = target.d

    nf = CopulaModel(d, target, max_deg=max_deg)
    params = nf.init_params()
    params = params.at[d:d+d].set(jnp.log(3) / 2 + 1.)
    
    div = 'chisq'
    print("Training", div) 
    params, logs = optimize(nf, params, div, max_iter=max_iter, nsample=nsample, seed=seed, sampler=sampler, max_lr=.1)

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
