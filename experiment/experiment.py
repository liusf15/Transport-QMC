import numpy as np
import pandas as pd
import os
import argparse
import pickle
import time
import bridgestan as bs
import cmdstanpy as csp
from scipy.stats import qmc
from scipy.special import ndtri
from scipy.optimize import minimize
import jax
import jax.numpy as jnp

from qmc_flow.nf_model import TransportMap

MACHINE_EPSILON = np.finfo(np.float64).eps

MODEL_LIST = [
    # "normal",
    "corr-normal",
    # "rosenbrock",
    # "glmm-poisson",
    "hmm",
    "garch",
    # "lotka-volterra",
    # 'irt-2pl',
    'eight-schools',
    'normal-mixture',
    # 'arma',
    'arK',
    # 'prophet',
    # 'covid19-impperial-v2',
    # 'pkpd',
]

# class stan_class:
#     def __init__(self, stan_path, data_path):
#         self.stan_model = bs.StanModel(stan_path, data_path, make_args=["STAN_THREADS=True", "TBB_CXX_TYPE=gcc"])
#         self.d = self.stan_model.param_unc_num()
    
#     def log_density(self, x):
#         val = self.stan_model.log_density(x)
#         if np.isnan(val):
#             # raise ValueError('nan in log density')
#             # print('nan in log density')
#             return -np.inf
#         return val

#     def log_density_gradient(self, x):
#         val, grad = self.stan_model.log_density_gradient(x)
#         if np.any(np.isnan(grad)) or np.isnan(val):
#             raise ValueError('nan in gradient')
#         return val, grad
    
#     def param_constrain(self, x):
#         if x.ndim == 1:
#             return self.stan_model.param_constrain(x)
#         return np.array([self.stan_model.param_constrain(x[i]) for i in range(x.shape[0])])

def make_logdensity_fn(bs_model):
    """Register a Stan model with JAX's custom VJP system via Bridgestan.

    See https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html.
    """

    @jax.custom_vjp
    def logdensity_fn(arg):
        # Cast to float64 to match Stan's dtype
        fn = lambda x: bs_model.log_density(np.array(x, dtype=np.float64))
        # Cast back to float32 to match JAX's default dtype
        result_shape = jax.ShapeDtypeStruct((), jnp.float32)
        return jax.pure_callback(fn, result_shape, arg)

    def call_grad(arg):
        fn = lambda x: bs_model.log_density_gradient(np.array(x, dtype=np.float64))[
            1
        ]
        result_shape = jax.ShapeDtypeStruct(arg.shape, arg.dtype)
        return jax.pure_callback(fn, result_shape, arg)

    def vjp_fwd(arg):
        return logdensity_fn(arg), arg

    def vjp_bwd(residuals, y_bar):
        arg = residuals
        return (call_grad(arg) * y_bar,)

    logdensity_fn.defvjp(vjp_fwd, vjp_bwd)

    return logdensity_fn


class stan_target:
    def __init__(self, stan_path, data_path):
        self.stan_model = bs.StanModel(stan_path, data_path, make_args=["STAN_THREADS=True", "TBB_CXX_TYPE=gcc"])
        self.d = self.stan_model.param_unc_num()
        self.log_prob_jax = make_logdensity_fn(self.stan_model)
    
    def log_prob(self, x):
        try:
            return self.log_prob_jax(x)
        except:
            return -jnp.inf
    
    def param_constrain(self, x):
        if x.ndim == 1:
            return self.stan_model.param_constrain(x)
        return np.array([self.stan_model.param_constrain(x[i]) for i in range(x.shape[0])])

def stan_sampler(stan, data):
    model = csp.CmdStanModel(stan_file=stan)
    fit = model.sample(
        data=data,
        seed=1,
        metric="unit_e",
        show_console=False,
        adapt_delta=0.9,
        chains=1,
        parallel_chains=2,
        iter_warmup=25_000,
        iter_sampling=50_000,
        show_progress=True,
    )
    meta_columns = len(fit.metadata.method_vars.keys())
    return fit.draws(concat_chains=True)[:, meta_columns:]

def get_moments(samples, weights=None):
    if weights is None:
        weights = np.ones(samples.shape[0]) / samples.shape[0]
    else:
        weights = weights / np.sum(weights)
    moment_1 = np.sum(samples * weights[:, None], axis=0)
    moment_2 = np.sum(samples**2 * weights[:, None], axis=0)
    return moment_1, moment_2

def get_effective_sample_size(weights):
    return np.sum(weights)**2 / np.sum(weights**2)

def get_mse(true_moments, est_moments):
    mse_1 = np.mean((true_moments[0] - est_moments[0])**2)
    mse_2 = np.mean((true_moments[1] - est_moments[1])**2)
    return mse_1, mse_2

def get_reference_moments(name):
    if name == 'corr-normal':
        return np.zeros(50), np.ones(50)
    
    moment_filename = f'qmc_flow/stan_models/{name}_moments.csv'
    if os.path.exists(moment_filename):
        moments = pd.read_csv(moment_filename)
        return moments['moment_1'].values, moments['moment_2'].values
    
    stan = f"qmc_flow/stan_models/{name}.stan"
    data = f"qmc_flow/stan_models/{name}.json"
    stan_draws = stan_sampler(stan, data)
    moment_1, moment_2 = get_moments(stan_draws)
    pd.DataFrame({'moment_1': moment_1, 'moment_2': moment_2}).to_csv(moment_filename, index=False)
    return pd.DataFrame({'moment_1': moment_1, 'moment_2': moment_2})


def run_experiment(name, max_deg, nsample, method, max_iter, seed, savepath):
    stan = f"qmc_flow/stan_models/{name}.stan"
    data = f"qmc_flow/stan_models/{name}.json"
    ref_moments = get_reference_moments(name)

    # train normalizing flow
    target = stan_target(stan, data)    
    d = target.d
    nf = TransportMap(d, target, max_deg=max_deg)

    start = time.time()
    # params_lbfgs, losses_lbfgs = nf.optimize_lbfgs_scipy(max_iter=max_iter, nsample=nsample, seed=seed, sampler='rqmc')
    params_lbfgs, losses_lbfgs = nf.optimize(
        max_iter=max_iter, 
        max_backtracking=20,
        memory_size=10,
        max_lr=1.,
        nsample=nsample, 
        seed=seed, 
        sampler=method
        )
    lbfgs_time = time.time() - start
    print('time elapsed:', lbfgs_time)
    moments_lbfgs = losses_lbfgs['moments'][-1]
    mse_lbfgs = get_mse(ref_moments, moments_lbfgs)

    # start = time.time()
    # params_adam, losses_adam = nf.optimize(max_iter=max_iter, lr=.1, nsample=nsample, seed=seed, algorithm='adam', sampler='rqmc')
    # adam_time = time.time() - start
    # print('time elapsed:', adam_time)
    # moments_adam = losses_adam['moments'][-1]
    # mse_adam = get_mse(ref_moments, moments_adam)

    results_lbfgs = {'MSE': mse_lbfgs, 'time': lbfgs_time, 'losses': losses_lbfgs['kl'], 'ESS': losses_lbfgs['ESS'], 'moments': losses_lbfgs['moments'], 'ref_moments': ref_moments}
    results = {'lbfgs': results_lbfgs}
    savepath = os.path.join(savepath, f'lbfgs_{method}_n_{nsample}_deg_{max_deg}_iter_{max_iter}_{seed}.pkl')
    with open(savepath, 'wb') as f:
        pickle.dump(results, f)
    print('saved to', savepath)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--date', type=str, default='2024-07-26')
    argparser.add_argument('--model_name', type=str, default='arK')
    argparser.add_argument('--max_iter', type=int, default=100)
    argparser.add_argument('--max_deg', type=int, default=3)
    argparser.add_argument('--m', type=int, default=10)
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--method', type=str, default='mc')
    argparser.add_argument('--rootdir', type=str, default='experiment/results')
    args = argparser.parse_args()

    nsample = 2**args.m
    savepath = os.path.join(args.rootdir, args.date, args.model_name)
    os.makedirs(savepath, exist_ok=True)

    run_experiment(args.model_name, args.max_deg, nsample, args.method, args.max_iter, args.seed, savepath)
