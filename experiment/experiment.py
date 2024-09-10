import numpy as np
import pandas as pd
import os
import argparse
import pickle
import time
import jax
import jax.numpy as jnp
from scipy.stats import qmc
from qmc_flow.targets import StanModel, Gaussian
from qmc_flow.models.tqmc import TransportQMC
from qmc_flow.train import lbfgs, sgd
from qmc_flow.utils import get_moments

MACHINE_EPSILON = np.finfo(np.float64).eps


def run_experiment(name='hmm', seed=1, nsample=64, num_composition=1, max_deg=3, optimizer='lbfgs', max_iter=50, lr=1e-3, savepath='results'):
    if name in ['arK', 'hmm', 'garch', 'arma', 'eight-schools', 'normal-mixture', 'rosenbrock', 'glmm-poisson', 'blr']:
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

    model = TransportQMC(d, target, base_transform='logit', num_composition=num_composition, max_deg=max_deg)
    params = model.init_params()
    
    soboleng = qmc.Sobol(d, seed=seed)
    U = soboleng.random(nsample)
    U = jnp.array(U * (1 - MACHINE_EPSILON) + .5 * MACHINE_EPSILON)
    model.reverse_kl(params, U)
    loss_fn = jax.jit(lambda params: model.reverse_kl(params, U))
    callback = jax.jit(lambda params: model.metrics(params, U))
    start = time.time()
    if optimizer == 'lbfgs':
        params, logs = lbfgs(loss_fn, params, max_iter=max_iter, callback=callback, max_lr=lr)
    else:
        params, logs = sgd(loss_fn, params, max_iter=max_iter, lr=lr, callback=callback)
    end = time.time()
    print('Time elapsed', end - start)
    print('rkl', logs.rkl[-1])
    print('ESS', logs.ess[-1])

    samples, weights = model.sample(params, nsample, seed=seed)
    moment_1, moment_2 = get_moments(samples, weights=weights)

    results = {
        'params': params,
        'time': end - start,
        'rkl': logs.rkl,
        'fkl': logs.fkl,
        'chisq': logs.chisq,
        'ess': logs.ess,
        'moment_1': moment_1,
        'moment_2': moment_2
    }
    savepath = os.path.join(savepath, f'tqmc_n_{nsample}_comp_{num_composition}_deg_{max_deg}_{optimizer}_iter_{max_iter}_lr_{lr}_{seed}.pkl')
    with open(savepath, 'wb') as f:
        pickle.dump(results, f)
    print('saved to', savepath)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--date', type=str, default='2024-09-07')
    argparser.add_argument('--name', type=str, default='hmm')
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--m', type=int, default=6)
    argparser.add_argument('--num_composition', type=int, default=1)
    argparser.add_argument('--max_deg', type=int, default=3)
    argparser.add_argument('--optimizer', type=str, default='lbfgs')
    argparser.add_argument('--max_iter', type=int, default=100)
    argparser.add_argument('--lr', type=float, default=1e-3)
    argparser.add_argument('--rootdir', type=str, default='experiment/results')

    args = argparser.parse_args()
    nsample = 2**args.m
    savepath = os.path.join(args.rootdir, args.date, args.name)
    os.makedirs(savepath, exist_ok=True)
    run_experiment(args.name, args.seed, nsample=nsample, num_composition=args.num_composition, max_deg=args.max_deg, optimizer=args.optimizer, max_iter=args.max_iter, lr=args.lr, savepath=savepath)
