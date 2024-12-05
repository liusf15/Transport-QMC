import numpy as np
import scipy
import os
import argparse
import pickle
from tqdm import trange
import jax
import jax.numpy as jnp
from scipy.stats import qmc
from qmc_flow.targets import BayesianLogisticRegression
from qmc_flow.models.tqmc_AS import TransportQMC_AS
from qmc_flow.train import lbfgs
from qmc_flow.utils import get_moments, sample_uniform, pareto_IS

MACHINE_EPSILON = jnp.finfo(jnp.float32).eps

def sample_uniform(nsample, d, rng, sampler):
    if sampler == 'mc':
        U = rng.random((nsample, d))
    else:
        soboleng =qmc.Sobol(d, scramble=True, seed=rng)    
        U = soboleng.random(nsample) 
    U = U * (1 - MACHINE_EPSILON) + MACHINE_EPSILON * .5
    return U

def get_ref_moments(d, N, cov_x, rho):
    filename = f'qmc_flow/experiment/logistic/moments/{cov_x}_d_{d}_N_{N}_rho_{rho}_n_65536_rep_1000.pkl'
    with open(filename, 'rb') as f:
        res = pickle.load(f)
    moments_1 = res['moments_1']
    moments_2 = res['moments_2']
    return np.mean(moments_1, axis=0), np.mean(moments_2, axis=0)

def run_experiment(d, cov_x, flow_type, N=20, nsample=64, num_composition=1, max_deg=3, optimizer='lbfgs', max_iter=50, lr=1., savepath='results'):
    
    ######################## generate data ########################
    rng = np.random.default_rng(0)
    if cov_x == 'equi':
        rho = 0.7
        cov_X = np.eye(d) * (1 - rho) + rho
        chol_x = np.linalg.cholesky(cov_X)
    elif cov_x == 'ar1':
        rho = 0.9
        cov_X = scipy.linalg.toeplitz(rho ** np.arange(d))
        chol_x = np.linalg.cholesky(cov_X)
    elif cov_x == 'low_rank':
        rho = 3
        D = np.ones(d) * .1
        D[:rho] = np.array([3., 2., 1.])
        orth = np.linalg.qr(rng.standard_normal((d, d)))[0]
        chol_x = orth.T @ np.diag(D)
    else:
        raise NotImplementedError
    
    X = rng.standard_normal((N, d)) @ chol_x.T / np.sqrt(N)

    beta = rng.random(d) * 2 - 1
    y = rng.binomial(1, 1 / (1 + np.exp(-jnp.dot(X, beta))))
    ref_moments_1, ref_moments_2 = get_ref_moments(d, N, cov_x, rho)

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
    print('Rank', r)
    V = eigvec[:]

    ######################## training ########################
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
    loss_fn = jax.jit(model_as.reverse_kl)

    callback_fn = jax.jit(model_as.metrics)

    best_params = params
    max_val_ess = 0.
    min_val_loss = np.Inf
    best_seed = 0
    for seed in range(10):
        print("Seed", seed)
        params = model_as.init_params()

        rng = np.random.default_rng(seed)
        U = sample_uniform(nsample, d, rng, 'rqmc')
        U_val = sample_uniform(nsample, d, rng, 'rqmc')
        callback = lambda params: callback_fn(params, U_val)

        loss = lambda params: loss_fn(params, U)    
        final_state, logs = lbfgs(loss, params, max_iter=max_iter, callback=callback, max_lr=lr)
        
        print('final rkl', logs.rkl[-1], 'final ESS', logs.ess[-1])

        if logs.rkl[-1] < min_val_loss:
            min_val_loss = logs.rkl[-1]
            best_params = final_state[0]
            max_val_ess = logs.ess[-1]
            best_seed = seed
    print(min_val_loss, max_val_ess)

    ######################## testing #######################

    def get_mse(moments_1, moments_2):
        mse_1 = (ref_moments_1 - moments_1)**2
        mse_2 = (ref_moments_2 - moments_2)**2
        return jnp.concat([mse_1, mse_2])
    
    methods = ['mc', 'rqmc']
    m_list = np.arange(3, 12)
    mse = {}
    mse_IS = {}
    mse_PSIS = {}
    moments_1 = {}
    moments_2 = {}
    moments_1_IS = {}
    moments_2_IS = {}
    moments_1_PSIS = {}
    moments_2_PSIS = {}
    nrep = 50
    for i in trange(nrep, desc='Testing'):
        for m in m_list:
            for sampler in methods:
                X, weights = model_as.sample(best_params, 2**m, rng, sampler=sampler)
                # IS
                moment_1, moment_2 = get_moments(X, weights)
                moments_1_IS[(sampler, m, i)], moments_2_IS[(sampler, m, i)] = moment_1, moment_2
                mse_IS[(sampler, m, i)] = get_mse(moment_1, moment_2)

                # no IS
                moment_1, moment_2 = get_moments(X, None)
                moments_1[(sampler, m, i)], moments_2[(sampler, m, i)] = moment_1, moment_2
                mse[(sampler, m, i)] = get_mse(moment_1, moment_2)

                # PSIS
                weights_pareto = pareto_IS(weights)
                moment_1, moment_2 = get_moments(X, weights_pareto)
                moments_1_PSIS[(sampler, m, i)], moments_2_PSIS[(sampler, m, i)] = moment_1, moment_2
                mse_PSIS[(sampler, m, i)] = get_mse(moment_1, moment_2)
            
    model_params = {
        'r': r,
        'V': V,
        'best_params': best_params,
        'best_seed': best_seed
    }
    test_results = {
        'mse': mse, 
        'moments_1': moments_1, 
        'moments_2': moments_2, 
        'mse_IS': mse_IS, 
        'moments_1_IS': moments_1_IS, 
        'moments_2_IS': moments_2_IS,
        'mse_PSIS': mse_PSIS,
        'moments_1_PSIS': moments_1_PSIS,
        'moments_2_PSIS': moments_2
    }
    results = {'model_params': model_params, 'test_results': test_results}
    savepath = os.path.join(savepath, f'{flow_type}_d_{d}_N_{N}_{cov_x}_{rho}_n_{nsample}_comp_{num_composition}_deg_{max_deg}_{optimizer}_iter_{max_iter}_lr_{lr}.pkl')
    with open(savepath, 'wb') as f:
        pickle.dump(results, f)
    print('saved to', savepath)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--date', type=str, default='2024-12-04')
    argparser.add_argument('--d', type=int, default='20')
    argparser.add_argument('--cov_x', type=str, default='ar1')
    argparser.add_argument('--flow_type', type=str, default='AS')
    argparser.add_argument('--N', type=int, default=20)
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--m', type=int, default=6)
    argparser.add_argument('--num_composition', type=int, default=1)
    argparser.add_argument('--max_deg', type=int, default=5)
    argparser.add_argument('--optimizer', type=str, default='lbfgs')
    argparser.add_argument('--max_iter', type=int, default=100)
    argparser.add_argument('--lr', type=float, default=1.)
    argparser.add_argument('--rootdir', type=str, default='experiment/results')

    args = argparser.parse_args()
    nsample = 2**args.m
    savepath = os.path.join(args.rootdir, args.date, 'logistic')
    os.makedirs(savepath, exist_ok=True)
    run_experiment(args.d, args.cov_x, args.flow_type, N=args.N, nsample=nsample, num_composition=args.num_composition, max_deg=args.max_deg, optimizer=args.optimizer, max_iter=args.max_iter, lr=args.lr, savepath=savepath)
