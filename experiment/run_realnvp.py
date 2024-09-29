import numpy as np
import pandas as pd
import os
import argparse
import pickle
import time
import jax
import jax.numpy as jnp
import optax

from qmc_flow.targets import StanModel, Gaussian
from qmc_flow.models.realnvp import RealNVP
from qmc_flow.models.copula import CopulaModel
from qmc_flow.train import optimize
from qmc_flow.utils import get_moments, get_effective_sample_size, sample_gaussian

num_blocks = 8
hidden_dim = 16
num_layers = 2
nsample = 2**6
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

    # RealNVP
    # nf = RealNVP(d, target, num_blocks, hidden_dim, num_layers, key=jax.random.key(seed))
    # params = nf.init_params()

    # CopulaModel
    nf = CopulaModel(d, target, max_deg=3)
    params = nf.init_params()

    loss_fn = jax.jit(nf.reverse_kl)
    
    X = sample_gaussian(nsample, d, seed=seed, sampler='rqmc')

    opt = optax.adam(1e-1)
    opt_state = opt.init(params)
    max_iter = 200
    rng = np.random.default_rng(seed)
    start = time.time()
    for it in range(max_iter):
        # X = rng.standard_normal((nsample, d))
        loss, grad = jax.value_and_grad(loss_fn)(params, X)
        updates, opt_state = opt.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        print(it, loss.item())
    end = time.time()
    print("Time elapsed", end - start)

    samples, weights = nf.sample(params, nsample, seed=seed, sampler='rqmc')
    ess = get_effective_sample_size(weights).item()
    print("ESS",ess)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--name', type=str, default='hmm')
    argparser.add_argument('--seed', type=int, default=1)
    args = argparser.parse_args()
    run_experiment(args.name, args.seed)
