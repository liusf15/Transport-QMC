import numpy as np
import pandas as pd
import scipy
import os
import argparse
import pickle
from tqdm import trange
import jax
import jax.numpy as jnp

from transport_qmc.targets import BayesianLogisticRegression
from transport_qmc.tqmc import TransportQMC_AS
from transport_qmc.train import lbfgs, sgd
from transport_qmc.utils import get_moments, sample_uniform

def run_experiment(flow_type, nsample=64, num_composition=1, max_deg=5, optimizer='lbfgs', max_iter=50, lr=1., savepath=None):
    ######################## set up target distribution #################
    d = 50
    N = 20
    rng = np.random.default_rng(0)
    rho = 0.9
    cov_X = scipy.linalg.toeplitz(rho ** np.arange(d))
    chol_x = np.linalg.cholesky(cov_X)
    X = rng.standard_normal((N, d)) @ chol_x.T / np.sqrt(N)
    beta = rng.random(d) * 2 - 1
    y = rng.binomial(1, 1 / (1 + np.exp(-jnp.dot(X, beta))))
    target = BayesianLogisticRegression(X, y, prior_scale=1.)

    ######################## find active subspace ########################
    M = 256
    key = jax.random.key(0)
    x = jax.random.normal(key, (M, d))
    G = jax.vmap(jax.grad(target.log_prob))(x) 
    G = G + x
    eigval, eigvec = np.linalg.eigh(G.T @ G / M)
    eigval = eigval[::-1]
    eigvec = eigvec[:, ::-1]

    var_explained = np.cumsum(eigval) / np.sum(eigval)
    r = np.where(var_explained > 0.99)[0][0] + 1 # rank
    print('Active subspace dimension:', r)
    V = eigvec[:]

    print("training normalizing flow ......")
    if flow_type == 'AS':
        model_as = TransportQMC_AS(d, r, V, target, base_transform='normal-icdf', nonlinearity='logit', num_composition=num_composition, max_deg=max_deg)
    elif flow_type == 'MF':
        model_as = TransportQMC_AS(d, 0, np.eye(d), target, base_transform='normal-icdf', nonlinearity='logit', num_composition=num_composition, max_deg=max_deg)
    elif flow_type == 'full':
        model_as = TransportQMC_AS(d, d, np.eye(d), target, base_transform='normal-icdf', nonlinearity='logit', num_composition=num_composition, max_deg=max_deg)
    elif flow_type == 'MF-G': # mean-field gaussian
        max_deg = 2
        model_as = TransportQMC_AS(d, 0, np.eye(d), target, base_transform='normal-icdf', nonlinearity='logit', num_composition=num_composition, max_deg=max_deg)
    elif flow_type == 'full-G': # full covariance gaussian
        max_deg = 2
        model_as = TransportQMC_AS(d, d, np.eye(d), target, base_transform='normal-icdf', nonlinearity='logit', num_composition=num_composition, max_deg=max_deg)
    else:
        raise NotImplementedError
    params = model_as.init_params()
    leaves = jax.tree_util.tree_leaves(params)
    print('num of parameters:', sum(jnp.size(leaf) for leaf in leaves))
    
    get_kl = jax.jit(model_as.reverse_kl)
    get_ess = jax.jit(model_as.ess)

    best_params = params
    max_ess = 0.
    best_seed = 0
    for seed in range(10):
        print("Seed:", seed)
        rng = np.random.default_rng(seed)

        U = jnp.array(sample_uniform(nsample, d, rng, 'rqmc'))
        loss_fn = lambda params: get_kl(params, U)

        U_val = jnp.array(sample_uniform(nsample, d, rng, 'rqmc'))
        val_fn = lambda params: get_ess(params, U_val)
        
        if optimizer == 'lbfgs':
            final_state, logs_ess = lbfgs(loss_fn, params, val_fn, max_iter=max_iter, max_lr=lr)
        else:
            final_state, logs_ess = sgd(loss_fn, params, val_fn, max_iter=max_iter, lr=lr)
        if logs_ess[-1] > max_ess:
            best_params = final_state[0]
            max_ess = logs_ess[-1]
    print("Effective sample size:", max_ess)

    print("estimating posterior moments ......")
    methods = ['mc', 'rqmc']
    m = 11
    moments_1 = {}
    moments_2 = {}
    nrep = 50
    for i in trange(nrep):
        for sampler in methods:
            X, log_weights = model_as.sample(best_params, 2**m, rng, sampler=sampler)
            weights = jnp.exp(log_weights - jnp.max(log_weights))
            moments_1[(sampler, i)], moments_2[(sampler, i)]= get_moments(X, weights)
            
    flow_params = {
        'r': r,
        'V': V,
        'best_params': best_params,
        'best_seed': best_seed
    }
    print("first moment:")
    print(pd.DataFrame(moments_1).T.groupby(level=0).mean())
    if savepath is not None:
        results = {'flow_parameters': flow_params, 
                    'moment1': moments_1,
                    'moment2': moments_2}
        os.makedirs(savepath, exist_ok=True)
        savepath = os.path.join(savepath, f'logistic_{flow_type}.pkl')
        with open(savepath, 'wb') as f:
            pickle.dump(results, f)
        print('saved to', savepath)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--flow_type', type=str, default='AS')
    argparser.add_argument('--m', type=int, default=6)
    argparser.add_argument('--num_composition', type=int, default=1)
    argparser.add_argument('--max_deg', type=int, default=5)
    argparser.add_argument('--optimizer', type=str, default='lbfgs')
    argparser.add_argument('--max_iter', type=int, default=100)
    argparser.add_argument('--lr', type=float, default=1.)
    argparser.add_argument('--savepath', type=str, default=None)

    args = argparser.parse_args()
    nsample = 2**args.m
    run_experiment(args.flow_type, 
                   nsample=nsample, 
                   num_composition=args.num_composition, 
                   max_deg=args.max_deg, 
                   optimizer=args.optimizer, 
                   max_iter=args.max_iter, 
                   lr=args.lr, 
                   savepath=args.savepath)
