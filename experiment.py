import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
import os
import argparse
from scipy.stats import qmc
from scipy.special import ndtri
from scipy.optimize import minimize

from experiment.polynomial.targets import BayesianLogisticRegression
from experiment.copula.copula_model import Copula_NF
MACHINE_EPSILON = np.finfo(np.float64).eps

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
