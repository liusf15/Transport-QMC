import numpy as np
import pandas as pd
import os
import argparse
import pickle
import bridgestan as bs
import cmdstanpy as csp
from scipy.stats import qmc
from scipy.special import ndtri
from scipy.optimize import minimize

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
    # 'eight-schools',
    # 'normal-mixture',
    # 'arma',
    'arK',
    # 'prophet',
    # 'covid19-impperial-v2',
    # 'pkpd',
]

class stan_target:
    def __init__(self, stan_path, data_path):
        self.stan_model = bs.StanModel(stan_path, data_path)
        self.d = self.stan_model.param_unc_num()
    
    def log_prob(self, x):
        if x.ndim == 1:
            return self.stan_model.log_density(x)
        return np.array([self.stan_model.log_density(x[i]) for i in range(x.shape[0])])
    
    def log_prob_grad(self, x):
        if x.ndim == 1:
            return self.stan_model.log_density_gradient(x)[1]
        return np.array([self.stan_model.log_density_gradient(x[i])[1] for i in range(x.shape[0])])
    
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

    # train normalizing flow
    target = stan_target(stan, data)    
    d = target.d
    nf = TransportMap(d, target, max_deg=max_deg)
    params = nf.init_params()
    if method == 'rqmc':
        soboleng = qmc.Sobol(d, scramble=True, seed=seed)
        X = ndtri(soboleng.random(nsample) * (1 - MACHINE_EPSILON) + .5 * MACHINE_EPSILON)
    elif method == 'mc':    
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((nsample, d))
    log_q = -0.5 * np.sum(X**2, axis=-1) - 0.5 * d * np.log(2 * np.pi)

    kl_logs = []
    moments_logs = []
    ess_logs = []
    state = {'counter': 0}
    print_every = 1
    def callback(params):
        state['counter'] += 1
        loss = nf.reverse_kl(params, X)
        kl_logs.append(loss.item())
        Z_nf, log_det = nf.forward_and_logdet(params, X)      
        proposal_log_densities = log_q - log_det
        target_log_densities = target.log_prob(np.array(Z_nf, float))
        log_weights = target_log_densities - proposal_log_densities
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        nf_samples = target.param_constrain(np.array(Z_nf, float))
        moments = get_moments(nf_samples, weights)
        moments_logs.append(moments)
        ess = get_effective_sample_size(weights)
        ess_logs.append(ess.item())
        if state['counter'] % print_every == 0:
            print(f"Iteration: {state['counter']}, Loss: {loss}, ESS: {ess}")
        if ess >= 0.99 * nsample:
            print(f"ESS is {ess}, stopping optimization")
            raise StopIteration
        

    try:
        res = minimize(nf.reverse_kl, jac=nf.reverse_kl_grad, x0=params, args=X, method='L-BFGS-B', options={'maxiter': max_iter}, callback=callback)  
        if res.success:
            print("Optimization successful")
    except StopIteration:
        print("Optimization was terminated by the callback function")

    params = res.x
    # Z_nf, log_det = nf.forward_and_logdet(params, X)
    # proposal_log_densities = log_q - log_det
    # target_log_densities = target.log_prob(np.array(Z_nf, float))
    # log_weights = target_log_densities - proposal_log_densities
    # log_weights -= np.max(log_weights)
    # weights = np.exp(log_weights)
    # nf_samples = target.param_constrain(np.array(Z_nf, float))
    # ess = get_effective_sample_size(weights)
    # moments = get_moments(nf_samples, weights)
    callback(params)
    ref_moments = get_reference_moments(name)
    # mse_1, mse_2 = get_mse(true_moments, moments)
    results = {'ESS': ess_logs, 'moments': moments_logs, 'losses': kl_logs, 'optimization': res, 'ref_moments': ref_moments}
    savepath = os.path.join(savepath, f'{method}_n_{nsample}_deg_{max_deg}_iter_{max_iter}_{seed}.pkl')
    with open(savepath, 'wb') as f:
        pickle.dump(results, f)
    # pd.DataFrame(results, index=[0]).to_csv(savepath, index=False)
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
