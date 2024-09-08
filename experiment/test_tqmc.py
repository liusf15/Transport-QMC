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
from qmc_flow.train import lbfgs

MACHINE_EPSILON = np.finfo(np.float64).eps


def run_experiment(name='hmm', seed=1, nsample=64, num_composition=1, max_deg=3, max_iter=50, savepath='results'):
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

    model = TransportQMC(d, target, base_transform='logit', num_composition=num_composition, max_deg=max_deg)
    params = model.init_params()
    
    soboleng = qmc.Sobol(d, seed=seed)
    U = soboleng.random(nsample)
    U = jnp.array(U * (1 - MACHINE_EPSILON) + .5 * MACHINE_EPSILON)

    loss_fn = jax.jit(lambda params: model.reverse_kl(params, U))

    callback = jax.jit(lambda params: model.metrics(params, U))
    # loss_fn = jax.jit(model.reverse_kl)
    start = time.time()
    params, logs = lbfgs(loss_fn, params, max_iter=max_iter, callback=callback)
    end = time.time()

    logs = np.array(logs).T
    logs = pd.DataFrame(logs, columns=['rkl', 'fkl', 'chisq', 'ess'])
    print(logs.iloc[-1])
    results = {
        'time': end - start,
        'logs': logs
    }
    savepath = os.path.join(savepath, f'tqmc_n_{nsample}_comp_{num_composition}_deg_{max_deg}_iter_{max_iter}_{seed}.pkl')
    with open(savepath, 'wb') as f:
        pickle.dump(results, f)
    print('saved to', savepath)
    # opt = optax.adam(1e-1)
    # opt_state = opt.init(params)
    # max_iter = 200

    # start = time.time()
    # for it in range(max_iter):
    #     loss, grad = jax.value_and_grad(loss_fn)(params, U)
    #     updates, opt_state = opt.update(grad, opt_state, params)
    #     params = optax.apply_updates(params, updates)
    #     print(it, loss.item())
    #     # print(params[0]['b'])
    #     # print(params[0]['L'][:d])
    #     # print(params[0]['L'][d:])
    #     # print(params[0]['weights'])
        
    # end = time.time()
    # print("Time elapsed", end - start)
    # samples, weights = model.sample(params, nsample, seed=seed)
    # ess = get_effective_sample_size(weights).item()
    # print("ESS", ess)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--date', type=str, default='2024-09-07')
    argparser.add_argument('--max_iter', type=int, default=100)
    argparser.add_argument('--max_deg', type=int, default=3)
    argparser.add_argument('--name', type=str, default='hmm')
    argparser.add_argument('--num_composition', type=int, default=1)
    argparser.add_argument('--m', type=int, default=6)
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--rootdir', type=str, default='experiment/results')

    args = argparser.parse_args()
    nsample = 2**args.m
    savepath = os.path.join(args.rootdir, args.date, args.name)
    os.makedirs(savepath, exist_ok=True)
    run_experiment(args.name, args.seed, nsample=nsample, num_composition=args.num_composition, max_deg=args.max_deg, max_iter=args.max_iter, savepath=savepath)
