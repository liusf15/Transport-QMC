import numpy as np
import pandas as pd
import os
import argparse
import pickle
import time
import jax
import jax.numpy as jnp
import jax_tqdm
import optax
from scipy.stats import qmc
from qmc_flow.targets import StanModel, Gaussian, BayesianLogisticRegression
from qmc_flow.models.tqmc import TransportQMC
from qmc_flow.train import lbfgs, sgd, lbfgs_annealed
from qmc_flow.utils import get_moments

MACHINE_EPSILON = np.finfo(np.float64).eps


def run_experiment(name='hmm', seed=1, nsample=64, num_composition=1, max_deg=3, annealed=False, optimizer='lbfgs', max_iter=50, lr=1., savepath='results'):
    if name == 'gaussian':
        d = 10
        mean = jnp.zeros(d)
        cov = (jnp.ones((d, d)) * 0.5 + jnp.eye(d) * 0.5) * 2.
        target = Gaussian(mean, cov)
    elif name == 'logistic':
        d = 10
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, d))
        beta = rng.random(d) * 2 - 1
        y = rng.binomial(1, 1 / (1 + np.exp(-jnp.dot(X, beta))))
        target = BayesianLogisticRegression(X, y, prior_scale=1.)
    else:
        data_path = f"qmc_flow/stan_models/{name}.json"
        stan_path = f"qmc_flow/stan_models/{name}.stan"
        target = StanModel(stan_path, data_path)
        
    d = target.d

    model = TransportQMC(d, target, base_transform='normal-icdf', nonlinearity='logit', num_composition=num_composition, max_deg=max_deg)
    params = model.init_params()
    
    soboleng = qmc.Sobol(d, seed=seed)
    U = soboleng.random(nsample)
    U = jnp.array(U * (1 - MACHINE_EPSILON) + .5 * MACHINE_EPSILON)
     
    U_val = soboleng.random(nsample)
    U_val = jnp.array(U_val * (1 - MACHINE_EPSILON) + .5 * MACHINE_EPSILON)
    callback = jax.jit(lambda params: model.metrics(params, U_val))

    if annealed:
        loss_fn = jax.jit(lambda params, lbd: model.reg_kl(params, lbd, U))
        start = time.time()
        final_state, logs = lbfgs_annealed(loss_fn, params, max_iter=max_iter, anneal_iter=20, max_lbd=1., callback=callback, max_lr=lr)
        end = time.time()
    else:
        loss_fn = jax.jit(lambda params: model.reverse_kl(params, U))
        start = time.time()
        final_state, logs = lbfgs(loss_fn, params, max_iter=max_iter, callback=callback, max_lr=lr)
        end = time.time()

    # opt = optax.adam(lr)
    # opt_state = opt.init(params)
    # best_params = params
    # best_ess = 0.
    
    # @jax_tqdm.scan_tqdm(max_iter)
    # def sgd_step(carry, t):
    #     params, opt_state, best_params, best_ess = carry
    #     lbd = jnp.clip(t / (max_iter / 1), 0., 1.)
    #     # lbd = 0.
    #     grad = jax.grad(loss_fn)(params, lbd)
    #     updates, opt_state = opt.update(grad, opt_state, params)
    #     params = optax.apply_updates(params, updates)
    #     if callback is not None:
    #         metrics = callback(params)
    #         new_best_params, new_best_ess = jax.lax.cond(
    #             jnp.isnan(metrics.ess) | (metrics.ess < best_ess),
    #             lambda: (best_params, best_ess),  
    #             lambda: (params, metrics.ess)  
    # )
    #     else:
    #         metrics = None
    #     return (params, opt_state, new_best_params, new_best_ess), metrics

    # carry = (params, opt_state, best_params, best_ess)
    # final_state, logs = jax.lax.scan(sgd_step, carry, np.arange(max_iter))
    
    print('Time elapsed', end - start)

    params = final_state[0]
    best_params = final_state[2]
    max_ess = final_state[3]
    print('final rkl', logs.rkl[-1])
    print('final reg_rkl', logs.reg_rkl[-1])
    print('final ESS', logs.ess[-1])
    print('Max ESS', max_ess)

    samples, weights = model.sample(params, nsample, seed)
    print(np.sum(weights)**2 / np.sum(weights**2))
    results = {
        'params': params,
        'time': end - start,
        'rkl': logs.rkl,
        'fkl': logs.fkl,
        'chisq': logs.chisq,
        'ess': logs.ess,
        'best_params': best_params,
        'max_ess': max_ess
    }
    savepath = os.path.join(savepath, f'tqmc_val_n_{nsample}_comp_{num_composition}_deg_{max_deg}_{optimizer}_iter_{max_iter}_lr_{lr}_{seed}.pkl')
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
    argparser.add_argument('--annealed', action='store_true')
    argparser.add_argument('--optimizer', type=str, default='lbfgs')
    argparser.add_argument('--max_iter', type=int, default=100)
    argparser.add_argument('--lr', type=float, default=1.)
    argparser.add_argument('--rootdir', type=str, default='experiment/results')

    args = argparser.parse_args()
    nsample = 2**args.m
    savepath = os.path.join(args.rootdir, args.date, args.name)
    os.makedirs(savepath, exist_ok=True)
    run_experiment(args.name, args.seed, nsample=nsample, num_composition=args.num_composition, max_deg=args.max_deg, annealed=args.annealed, optimizer=args.optimizer, max_iter=args.max_iter, lr=args.lr, savepath=savepath)
