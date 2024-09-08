import numpy as np
import pandas as pd
import os
import argparse
import pickle
import time
import jax
import jax.numpy as jnp
import optax
from scipy.stats import qmc
from qmc_flow.targets import StanModel, Gaussian
from qmc_flow.models.tqmc import TransportQMC
from qmc_flow.train import optimize
from qmc_flow.utils import get_moments, get_effective_sample_size 
MACHINE_EPSILON = np.finfo(np.float64).eps

nsample=2**6
def run_experiment(name='hmm', seed=1):
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

    model = TransportQMC(d, target, base_transform='logit', num_composition=2, max_deg=3)
    params = model.init_params()
    
    soboleng = qmc.Sobol(d, seed=seed)
    U = soboleng.random(nsample)
    U = jnp.array(U * (1 - MACHINE_EPSILON) + .5 * MACHINE_EPSILON)

    # loss_fn = jax.jit(lambda params: model.reverse_kl(params, U))
    loss_fn = jax.jit(model.reverse_kl)

    opt = optax.adam(1e-1)
    opt_state = opt.init(params)
    max_iter = 200

    start = time.time()
    for it in range(max_iter):
        loss, grad = jax.value_and_grad(loss_fn)(params, U)
        updates, opt_state = opt.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        print(it, loss.item())
        # print(params[0]['b'])
        # print(params[0]['L'][:d])
        # print(params[0]['L'][d:])
        # print(params[0]['weights'])
        
    end = time.time()
    print("Time elapsed", end - start)
    samples, weights = model.sample(params, nsample, seed=seed)
    ess = get_effective_sample_size(weights).item()
    print("ESS", ess)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--name', type=str, default='hmm')
    argparser.add_argument('--seed', type=int, default=1)
    args = argparser.parse_args()
    run_experiment(args.name, args.seed)
