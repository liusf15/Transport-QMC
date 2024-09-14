import numpy as np
import pickle
import pandas as pd
import argparse
import os
import jax
import jax.numpy as jnp
from scipy.stats import qmc

from qmc_flow.models.tqmc import TransportQMC
from qmc_flow.targets import StanModel, Gaussian, BayesianLogisticRegression
from qmc_flow.utils import get_effective_sample_size, get_moments
rootdir = '/mnt/home/sliu1/ceph/normalizing_flows/qmc_flow'

MACHINE_EPSILON = np.finfo(np.float64).eps

def get_ref_moments(name):
    if name == 'corr-normal':
        ref_moments_1 = np.zeros(50)
        ref_moments_2 = np.ones(50)
    elif name == 'rosenbrock':
        ref_moments_1 = np.array([1., 1., 2., 2.])
        ref_moments_2 = np.array([2., 2., 10.1, 10.1])
    else:
        filename = f'/mnt/home/sliu1/normalizing_flows/qmc_flow/stan_models/moments_{name}_chain_20_warmup_25000_nsample_50000.pkl'
        with open(filename, 'rb') as f:
            moments = pickle.load(f)
        ref_moments_1 = moments['moment_1'].mean(0)
        ref_moments_1_var = moments['moment_1'].var(0)
        ref_moments_2 = moments['moment_2'].mean(0)
    return ref_moments_1, ref_moments_2

def get_best_params(name, date, nsample, num_composition, max_deg, max_iter):
    path = os.path.join(rootdir, date, name)
    max_ess = -1
    for seed in range(10):
        filename = os.path.join(path, f'tqmc_val_n_{nsample}_comp_{num_composition}_deg_{max_deg}_lbfgs_iter_{max_iter}_lr_1.0_{seed}.pkl')
        if not os.path.exists(filename):
            continue
        with open(filename, 'rb') as f:
            res = pickle.load(f)
        if res['ess'][-1] > max_ess:
            max_ess = res['ess'][-1]
            params = res['params']
    print(f'Best ESS: {max_ess}')
    return params

def run_experiment(name='hmm', seed=0, ntrain=64, train_date='2024-09-09', num_composition=1, max_deg=3, max_iter=200, savepath='results'):
    if name == 'logistic':
        ref_moments_1 = np.zeros(5)
        ref_moments_2 = np.zeros(5)
    else:
        ref_moments_1, ref_moments_2 = get_ref_moments(name)

    if name == 'gaussian':
        d = 10
        mean = jnp.zeros(d)
        cov = (jnp.ones((d, d)) * 0.5 + jnp.eye(d) * 0.5) * 2.
        target = Gaussian(mean, cov)
    elif name == 'logistic':
        d = 5
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

    model = TransportQMC(d, target, base_transform='logit', num_composition=num_composition, max_deg=max_deg)
    params = get_best_params(name, train_date, ntrain, num_composition, max_deg, max_iter)


    def sample_uniform(nsample, d, rng, sampler):
        if sampler == 'mc':
            U = rng.random((nsample, d))
        else:
            soboleng =qmc.Sobol(d, scramble=True, seed=rng)    
            U = soboleng.random(nsample) * (1 - MACHINE_EPSILON) + MACHINE_EPSILON * .5
        return U

    @jax.jit
    def get_samples(U, rot=np.eye(d)):
        X, log_det = jax.vmap(model.forward_rotation, in_axes=(None, 0, None))(params, U, rot)
        target_log_densities = jax.vmap(target.log_prob)(X)
        log_weights = target_log_densities + log_det
        log_weights -= jnp.mean(log_weights)
        weights = jnp.exp(log_weights)
        return X, weights

    def get_moments(X, weights):
        if getattr(target, 'param_constrain', None):
            X = target.param_constrain(np.array(X, float))
        moments_1 = jnp.sum(X * weights[:, None], axis=0) / jnp.sum(weights)
        moments_2 = jnp.sum(X**2 * weights[:, None], axis=0) / jnp.sum(weights)
        return moments_1, moments_2

    def get_mse(moments_1, moments_2):
        mse_1 = (ref_moments_1 - moments_1)**2
        mse_2 = (ref_moments_2 - moments_2)**2
        return jnp.concat([mse_1, mse_2])
    
    m_list = np.arange(3, 16)
    rng = np.random.default_rng(2024+seed)
    mse = {}
    moments_1 = {}
    moments_2 = {}

    for m in m_list:
        print(m)
        # MC
        U = sample_uniform(2**m, d, rng, 'mc')
        X, weights = get_samples(U)
        moment_1, moment_2 = get_moments(X, weights)
        moments_1[('mc', m, seed)], moments_2[('mc', m, seed)] = moment_1, moment_2
        mse[('mc', m, seed)] = get_mse(moment_1, moment_2)

        # RQMC
        U = sample_uniform(2**m, d, rng, 'rqmc')
        X, weights = get_samples(U)
        moment_1, moment_2 = get_moments(X, weights)
        moments_1[('rqmc', m, seed)], moments_2[('rqmc', m, seed)] = moment_1, moment_2
        mse[('rqmc', m, seed)] = get_mse(moment_1, moment_2)
    
    filename = os.path.join(savepath, f'ntrain_{ntrain}_comp_{num_composition}_deg_{max_deg}_iter_{max_iter}_seed_{seed}.pkl')
    mse = pd.DataFrame(mse).T.reset_index(names=['sampler', 'm', 'seed'])
    moments_1 = pd.DataFrame(moments_1).T.reset_index(names=['sampler', 'm', 'seed'])
    moments_2 = pd.DataFrame(moments_2).T.reset_index(names=['sampler', 'm', 'seed'])
    results = {'mse': mse, 'moments_1': moments_1, 'moments_2': moments_2}
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print('saved to', filename)
    

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--date', type=str, default='2024-09-09')
    argparser.add_argument('--name', type=str, default='hmm')
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--ntrain', type=int, default=64)
    argparser.add_argument('--num_composition', type=int, default=1)
    argparser.add_argument('--max_deg', type=int, default=3)
    argparser.add_argument('--max_iter', type=int, default=200)
    argparser.add_argument('--rootdir', type=str, default='experiment/results')

    args = argparser.parse_args()

    savepath = os.path.join(args.rootdir, args.date, args.name, 'mse')
    os.makedirs(savepath, exist_ok=True)
    run_experiment(args.name, args.seed, ntrain=args.ntrain, train_date='2024-09-14', num_composition=args.num_composition, max_deg=args.max_deg, max_iter=args.max_iter, savepath=savepath)
