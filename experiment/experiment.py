import numpy as np
import os
import argparse
import pickle
from tqdm import trange
from scipy.stats import qmc
import jax
import jax.numpy as jnp
from qmc_flow.targets import StanModel, Gaussian, BayesianLogisticRegression
from qmc_flow.models.tqmc import TransportQMC
from qmc_flow.train import lbfgs, sgd
from qmc_flow.utils import pareto_IS

MACHINE_EPSILON = jnp.finfo(jnp.float32).eps

def sample_uniform(nsample, d, rng, sampler):
    if sampler == 'mc':
        U = rng.random((nsample, d))
    else:
        soboleng =qmc.Sobol(d, scramble=True, seed=rng)    
        U = soboleng.random(nsample) 
    U = U * (1 - MACHINE_EPSILON) + MACHINE_EPSILON * .5
    return U

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

def run_experiment(name='hmm', nsample=64, num_composition=1, max_deg=3, optimizer='lbfgs', max_iter=50, lr=1., savepath='results'):
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
    ref_moments_1, ref_moments_2 = get_ref_moments(name)
    d = target.d

    ######################## training ########################
    model = TransportQMC(d, target, base_transform='normal-icdf', nonlinearity='logit', num_composition=num_composition, max_deg=max_deg)
    params = model.init_params()
    best_params = params
    max_val_ess = 0
    loss_fn = jax.jit(model.reverse_kl)
    callback_fn = jax.jit(model.metrics)
    for seed in range(1):
        rng = np.random.default_rng(seed)
        U = sample_uniform(nsample, d, rng, 'rqmc')
        loss = lambda params: loss_fn(params, U)
        U_val = sample_uniform(nsample, d, rng, 'rqmc')
        callback = lambda params: callback_fn(params, U_val)
        
        if optimizer == 'lbfgs':
            final_state, logs = lbfgs(loss, params, max_iter=max_iter, callback=callback, max_lr=lr)
        else:
            final_state, logs = sgd(loss, params, max_iter=max_iter, lr=lr, callback=callback)
        if logs.ess[-1] > max_val_ess:
            best_params = final_state[0]
            max_val_ess = logs.ess[-1]

    print('Max ESS', max_val_ess)
    ######################## testing ########################
    @jax.jit
    def get_samples(U):
        X, log_det = jax.vmap(model.forward, in_axes=(None, 0))(best_params, U)
        log_p = jax.vmap(target.log_prob)(X)
        log_weights = log_p + log_det
        log_weights -= jnp.nanmean(log_weights)
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
    
    m_list = np.arange(3, 14, 2)
    mse = {}
    moments_1 = {}
    moments_2 = {}
    nrep = 1
    rng = np.random.default_rng(2024)
    for i in trange(nrep, desc='Testing'):
        for m in m_list:
            for sampler in ['mc', 'rqmc']:
                U = sample_uniform(2**m, d, rng, sampler)
                X, weights = get_samples(U)
                # IS
                moment_1, moment_2 = get_moments(X, weights)
                moments_1[(sampler, 'IS', m, i)], moments_2[(sampler, 'IS', m, i)] = moment_1, moment_2
                mse[(sampler, 'IS', m, i)] = get_mse(moment_1, moment_2)

                # no IS
                moment_1, moment_2 = get_moments(X, jnp.ones_like(weights))
                moments_1[(sampler, 'no-IS', m, i)], moments_2[(sampler, 'no-IS', m, i)] = moment_1, moment_2
                mse[(sampler, 'no-IS', m, i)] = get_mse(moment_1, moment_2)

                # pareto smoothed IS
                weights_smoothed = pareto_IS(np.array(weights))
                moment_1, moment_2 = get_moments(X, weights_smoothed)
                moments_1[(sampler, 'PSIS', m, i)], moments_2[(sampler, 'PSIS', m, i)] = moment_1, moment_2
                mse[(sampler, 'PSIS', m, i)] = get_mse(moment_1, moment_2)
            
    model_params = {
        'ess': max_val_ess,
        'best_params': best_params,
    }
    test_results = {
        'mse': mse, 
        'moments_1': moments_1, 
        'moments_2': moments_2, 
    }
    results = {'model_params': model_params, 'test_results': test_results}
    savepath = os.path.join(savepath, f'mse_n_{nsample}_comp_{num_composition}_deg_{max_deg}_{optimizer}_iter_{max_iter}_lr_{lr}.pkl')
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
    argparser.add_argument('--lr', type=float, default=1.)
    argparser.add_argument('--rootdir', type=str, default='experiment/results')

    args = argparser.parse_args()
    nsample = 2**args.m
    savepath = os.path.join(args.rootdir, args.date, args.name)
    os.makedirs(savepath, exist_ok=True)
    run_experiment(args.name, nsample=nsample, num_composition=args.num_composition, max_deg=args.max_deg, optimizer=args.optimizer, max_iter=args.max_iter, lr=args.lr, savepath=savepath)
