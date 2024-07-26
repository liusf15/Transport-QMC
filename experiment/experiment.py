import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
import os
import argparse
import bridgestan as bs
import cmdstanpy as csp
from scipy.stats import qmc
from scipy.special import ndtri
from scipy.optimize import minimize
import jax

from experiment.polynomial.targets import BayesianLogisticRegression, Gaussian
from qmc_flow.nf_model import TransportMap
MACHINE_EPSILON = np.finfo(np.float64).eps

# TODO: change the parametrization of matrix L
# play with the degree
# try a lower dimensional example

# d = 2
# mean = np.zeros(d)
# cov = np.eye(d)


class stan_target:
    def __init__(self, stan_path, data_path):
        self.stan_model = bs.StanModel(stan_path, data_path)
        self.d = self.stan_model.param_unc_num()
    
    def log_prob(self, x):
        return self.stan_model.log_density(x)
    
    def log_prob_grad(self, x):
        return self.stan_model.log_density_gradient(x)[1]


name = 'corr-normal'
name = 'arK'
stan = f"qmc_flow/stan_models/{name}.stan"
data = f"qmc_flow/stan_models/{name}.json"


def metadata_columns(fit):
    """Return the number of metadata columns in CmdStanPy's output."""
    return len(fit.metadata.method_vars.keys())

# 
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
    show_progress=False,
)

theta_draws = fit.draws(concat_chains=True)[:, metadata_columns(fit):]
# # sns.pairplot(pd.DataFrame(theta_draws))
# # plt.scatter(theta_draws[:, 0], theta_draws[:, 2])

theta_mean = theta_draws.mean(axis=0)
# theta_sd = theta_draws.std(axis=0)
# theta_sq_mean = (theta_draws**2).mean(axis=0)
# theta_sq_sd = (theta_draws**2).std(axis=0)

# plt.hist(theta_draws[:, -1], 50)

# # bs.compile_model(stan)
# model_bs = bs.StanModel(stan, data)
# model_bs.param_unc_names()


target = stan_target(stan, data)
d = target.d

nf = TransportMap(d, target, max_deg=1)
params = nf.init_params()


params_gd, losses = nf.gradient_descent(max_iter=1000, lr=1e-2, nsample=2**8, seed=0, print_every=10)

X = np.random.randn(50000, d)
Z_nf, log_det = nf.forward_and_logdet(params_gd, X) 
log_q = -0.5 * np.sum(X**2, axis=-1) - 0.5 * d * np.log(2 * np.pi)
proposal_log_densities = log_q - log_det
target_log_densities = np.array([target.log_prob(np.array(Z_nf[i], float)) for i in range(Z_nf.shape[0])])
log_weights = target_log_densities - proposal_log_densities
log_weights -= np.max(log_weights)
weights = np.exp(log_weights)   
np.sum(Z_nf * weights[:, None], axis=0) / np.sum(weights) 


plt.scatter(Z_nf[:, 0], Z_nf[:, 1])
plt.show()

# effective sample size
np.sum(weights)**2 / np.sum(weights**2)
np.mean(weights)**2 / np.mean(weights**2)


soboleng = qmc.Sobol(d, scramble=True, seed=1)
x = ndtri(soboleng.random(2**10) * (1 - MACHINE_EPSILON) + .5 * MACHINE_EPSILON)
# nf.reverse_kl(params, x)

res = minimize(nf.reverse_kl, jac=nf.reverse_kl_grad, x0=params, args=x, method='L-BFGS-B', options={'maxiter': 500})  
res
nf.reverse_kl(res.x, x)
nf.reverse_kl_grad(res.x, x)

params_ = res.x 
nf.unpack_params(params_)[1]

X = np.random.randn(50000, d)
Z_nf, log_det = nf.forward_and_logdet(params_, X) 
log_q = -0.5 * np.sum(X**2, axis=-1) - 0.5 * d * np.log(2 * np.pi)
proposal_log_densities = log_q - log_det
target_log_densities = np.array([target.log_prob(np.array(Z_nf[i], float)) for i in range(Z_nf.shape[0])])
log_weights = target_log_densities - proposal_log_densities
log_weights -= np.max(log_weights)
weights = np.exp(log_weights)   
np.sum(weights)**2 / np.sum(weights**2)
np.sum(Z_nf * weights[:, None], axis=0) / np.sum(weights)

# unconstrained samples Z_nf
target.stan_model.param_constrain(np.array(Z_nf, float)[0])


plt.hist(theta_draws[:, 6], 50)
plt.hist(np.exp(Z_nf[:, 6]), 50, alpha=0.5)
plt.show()

def run_simu(max_iter, nsample, seed, rqmc):
    d = 4
    N = 20
    rho = 0.9
    rng = np.random.default_rng(0)
    data = {}
    data['X']= rng.standard_normal((N, d)) @ np.linalg.cholesky(scipy.linalg.toeplitz(rho ** np.arange(d))).T
    data['beta'] = rng.standard_normal(d)
    data['y'] = rng.binomial(1, scipy.special.expit(data['X'] @ data['beta']))
    target = BayesianLogisticRegression(data['X'], data['y'], prior_scale=1.)
    ############################################################
    # Laplace approximation
    ############################################################
    def obj(mu):
        return np.sum(np.log(1 + np.exp(-(data['X'] @ mu) * data['y'])))  + mu.dot(mu) / 2
    res = minimize(obj, np.zeros(d))
    lap_shift = res.x

    def simu_lap(Z, shift):
        Z_shift = Z + shift
        log_weight = target.log_prob(Z_shift) - (-.5 * np.sum(Z**2, 1) - .5 * d * np.log(2 * np.pi))
        weight = np.exp(log_weight)
        normalizing_const = np.mean(weight)
        est_mean = np.mean(Z_shift * weight[:, None], 0) / normalizing_const
        est_var = np.mean(Z_shift**2 * weight[:, None], 0) / normalizing_const
        return normalizing_const, est_mean, est_var

    rng = np.random.default_rng(2024)
    nrep = 100
    nsample = 2**16
    est_consts = []
    est_means = []
    est_vars = []
    soboleng = qmc.Sobol(d, scramble=True, seed=rng)
    for i in range(nrep):
        Z = ndtri(soboleng.random(nsample) * (1 - MACHINE_EPSILON) + .5 * MACHINE_EPSILON)
        est_nc, est_mean, est_var = simu_lap(Z, lap_shift)
        est_consts.append(est_nc)
        est_means.append(est_mean)
        est_vars.append(est_var)

    true_mean = np.mean(np.array(est_means), 0)
    true_nc = np.mean(est_consts)
    true_var = np.mean(np.array(est_vars), 0)

    ############################################################
    # Normalizing flow
    ############################################################
    max_deg = 3
    nf = Copula_NF(d, target, max_deg)
    nf.init_params()
    nf.true_mean = true_mean
    nf.true_var = true_var

    nf.init_params()
    Z_nf, weights, losses, evals = nf._optimize_eval(max_iter=max_iter, rqmc=rqmc, nsample=nsample, seed=seed)
    return np.array(evals)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--date', type=str, default='2024-07-23')
    argparser.add_argument('--max_iter', type=int, default=10)
    argparser.add_argument('--m', type=int, default=6)
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--method', type=str, default='mc')
    argparser.add_argument('--rootdir', type=str, default='experiment/copula/results')
    args = argparser.parse_args()

    nsample = 2**args.m
    rqmc = args.method == 'rqmc'
    evals = run_simu(args.max_iter, nsample, args.seed, rqmc=rqmc)

    path = os.path.join(args.rootdir, args.date)
    os.makedirs(path, exist_ok=True)
    filename = f'logistic_{args.method}_{nsample}_{args.seed}.csv'
    path = os.path.join(path, filename)
    pd.DataFrame(evals).to_csv(path, index=False)

# errors_1 = {}
# errors_2 = {}
# for seed in range(10):
#     for m in [6, 8, 10, 12]:
#         nsample = 2**m
#         print(seed, nsample)
#         nf.init_params()
#         Z_nf, weights, losses, evals = nf._optimize_eval(max_iter=10, rqmc=False, nsample=nsample, seed=seed)
#         errors_1[(seed, m, 'mc')] = np.array(evals)[:, 0]
#         errors_2[(seed, m, 'mc')] = np.array(evals)[:, 1]

#         nf.init_params()
#         Z_nf, weights, losses, evals = nf._optimize_eval(max_iter=10, rqmc=True, nsample=nsample, seed=seed)
#         errors_1[(seed, m, 'rqmc')] = np.array(evals)[:, 0]
#         errors_2[(seed, m, 'rqmc')] = np.array(evals)[:, 1]

# df = pd.DataFrame(errors_1).T
# df.reset_index(names=['seed', 'm', 'method'], inplace=True)
# df = df.melt(id_vars=['seed', 'm', 'method'], var_name='iter', value_name='error')

# fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True, sharex=True)
# sns.pointplot(ax=ax[0], data=df.loc[df['method'] == 'mc'], x='iter', y='error', hue='m')
# sns.pointplot(ax=ax[1], data=df.loc[df['method'] == 'rqmc'], x='iter', y='error', hue='m')
# plt.yscale('log')
# plt.show()
