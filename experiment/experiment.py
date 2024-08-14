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
from experiment.polynomial.targets import arK
from qmc_flow.models import PolynomialModel

import bridgestan as bs

def get_mse(true_moments, est_moments):
    mse_1 = np.mean((true_moments[0] - est_moments[0])**2)
    mse_2 = np.mean((true_moments[1] - est_moments[1])**2)
    return mse_1, mse_2

def run_experiment(name, max_deg, nsample, method, max_iter, seed, savepath):
    datapath = f"qmc_flow/stan_models/{name}.json"
    stanpath = f"qmc_flow/stan_models/{name}.stan"

    # train normalizing flow
    target = arK(datapath)
    d = target.d
    bs_model = bs.StanModel(stanpath, datapath)
    nf = PolynomialModel(d, target, max_deg=max_deg)

    start = time.time()
    # params_lbfgs, losses_lbfgs = nf.optimize_lbfgs_scipy(max_iter=max_iter, nsample=nsample, seed=seed, sampler='rqmc')
    params_lbfgs, losses_lbfgs = nf.optimize(
        max_iter=max_iter, 
        max_backtracking=20,
        memory_size=10,
        max_lr=1.,
        nsample=nsample, 
        seed=seed, 
        sampler=method
        )
    lbfgs_time = time.time() - start
    print('time elapsed:', lbfgs_time)
    moments_lbfgs = losses_lbfgs['moments'][-1]

    # start = time.time()
    # params_adam, losses_adam = nf.optimize(max_iter=max_iter, lr=.1, nsample=nsample, seed=seed, algorithm='adam', sampler='rqmc')
    # adam_time = time.time() - start
    # print('time elapsed:', adam_time)
    # moments_adam = losses_adam['moments'][-1]
    # mse_adam = get_mse(ref_moments, moments_adam)

    results_lbfgs = {'time': lbfgs_time, 'losses': losses_lbfgs['kl'], 'ESS': losses_lbfgs['ESS'], 'moments': losses_lbfgs['moments']}
    results = {'lbfgs': results_lbfgs}
    # savepath = os.path.join(savepath, f'lbfgs_{method}_n_{nsample}_deg_{max_deg}_iter_{max_iter}_{seed}.pkl')
    # with open(savepath, 'wb') as f:
    #     pickle.dump(results, f)
    # print('saved to', savepath)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--date', type=str, default='2024-08-07')
    argparser.add_argument('--model_name', type=str, default='arK')
    argparser.add_argument('--max_iter', type=int, default=100)
    argparser.add_argument('--max_deg', type=int, default=3)
    argparser.add_argument('--m', type=int, default=10)
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--method', type=str, default='mc')
    argparser.add_argument('--rootdir', type=str, default='experiment/results')
    args = argparser.parse_args()

    nsample = 2**args.m
    savepath = os.path.join(args.rootdir, args.date, args.model_name)
    os.makedirs(savepath, exist_ok=True)

    run_experiment(args.model_name, args.max_deg, nsample, args.method, args.max_iter, args.seed, savepath)
