import numpy as np
import os
import argparse
import pickle
import time
import jax.numpy as jnp

from qmc_flow.targets import StanModel, Gaussian
from qmc_flow.models.copula import CopulaModel
from qmc_flow.train import optimize
from qmc_flow.utils import get_moments, get_effective_sample_size

# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)


def run_experiment(name, max_deg, nsample, sampler, max_iter, seed, savepath):
    df = 5
    if name in ['arK', 'hmm', 'garch', 'arma', 'eight-schools', 'normal-mixture', 'rosenbrock']:
        data_path = f"qmc_flow/stan_models/{name}.json"
        stan_path = f"qmc_flow/stan_models/{name}.stan"
        target = StanModel(stan_path, data_path)
    else:
        if name == 'gaussian':
            d = 10
            mean = jnp.zeros(d)
            cov = (jnp.ones((d, d)) * 0.5 + jnp.eye(d) * 0.5) * 2.
            target = Gaussian(mean, cov)
    d = target.d
    
    nf = CopulaModel(d, target, max_deg=max_deg)
    params = nf.init_params()
    
    div = 'rkl'
    print("Training", div) 
    start = time.time()
    params, logs = optimize(nf, params, div, df=df, max_iter=max_iter, nsample=nsample, seed=seed, sampler=sampler, max_lr=1.)
    end = time.time()
    print("Time elapsed", end - start)
    print('rkl, fkl, chisq') 
    print(np.array(logs).T[-1])

    nf_samples, weights = nf.sample(params, nsample, df=df, seed=seed, sampler=sampler)
    moment = get_moments(nf_samples, weights)
    ess = get_effective_sample_size(weights).item()
    print('ESS:', ess)

    results = {'moment': moment, 'ess': ess, 'logs': np.array(logs).T}
    
    savepath = os.path.join(savepath, f'betamix_{div}_{sampler}_n_{nsample}_deg_{max_deg}_iter_{max_iter}_{seed}.pkl')
    with open(savepath, 'wb') as f:
        pickle.dump(results, f)
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
