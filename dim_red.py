import numpy as np
import scipy
import pandas as pd
import os
import argparse
import pickle
import time
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp, ndtri, ndtr
from scipy.stats import qmc
from qmc_flow.targets import StanModel, Gaussian, BayesianLogisticRegression
from qmc_flow.models.tqmc_AS import TransportQMC_AS
from qmc_flow.models.tqmc import TransportQMC
from qmc_flow.train import lbfgs, sgd
from qmc_flow.utils import get_moments, sample_gaussian

MACHINE_EPSILON = np.finfo(np.float64).eps

d = 100
rng = np.random.default_rng(0)
rho = 0.7
# cov_X = scipy.linalg.toeplitz(rho ** np.arange(d))
cov_X = np.eye(d) * (1 - rho) + rho
chol_X = np.linalg.cholesky(cov_X)
X = rng.standard_normal((20, d)) @ chol_X.T
beta = rng.random(d) * 2 - 1
y = rng.binomial(1, 1 / (1 + np.exp(-jnp.dot(X, beta))))
target = BayesianLogisticRegression(X, y, prior_scale=1.)

M = 1024
x = np.random.randn(M, d)
G = jax.vmap(jax.grad(target.log_prob))(x) 
G = G + x
eigval, eigvec = np.linalg.eigh(G.T @ G / M)
eigval = eigval[::-1]
eigvec = eigvec[:, ::-1]

var_explained = np.cumsum(eigval) / np.sum(eigval)
r = np.where(var_explained > 0.95)[0][0] + 1 # rank
print('Rank', r)
V = eigvec[:]
V_r = eigvec[:, :r]
P_r = V_r @ V_r.T
V_orth = eigvec[:, r:]


soboleng = qmc.Sobol(d, seed=1)
U = soboleng.random(2**6)
U = jnp.array(U * (1 - MACHINE_EPSILON) + .5 * MACHINE_EPSILON)

# model = TransportQMC(d, target, base_transform='normal-icdf', nonlinearity='logit', num_composition=1, max_deg=3)

model_as = TransportQMC_AS(d, r, V, target, base_transform='normal-icdf', nonlinearity='logit', num_composition=2, max_deg=5)
# TODO: no need to apply composition for the not important variables 
params = model_as.init_params()

loss_fn = jax.jit(lambda params: model_as.reverse_kl(params, U))

U_val = soboleng.random(2**6)
U_val = jnp.array(U_val * (1 - MACHINE_EPSILON) + .5 * MACHINE_EPSILON)
callback = jax.jit(lambda params: model_as.metrics(params, U_val))

optimizer = 'lbfgs'
max_iter = 200
lr = 1.
start = time.time()
if optimizer == 'lbfgs':
    final_state, logs = lbfgs(loss_fn, params, max_iter=max_iter, callback=callback, max_lr=lr)
else:
    final_state, logs = sgd(loss_fn, params, max_iter=max_iter, lr=lr, callback=callback)
end = time.time()

print('Time elapsed', end - start)

params = final_state[0]
best_params = final_state[2]
max_ess = final_state[3]
print('final rkl', logs.rkl[-1])
print('final ESS', logs.ess[-1])
print('Max ESS', max_ess)

results = {
    'params': params,
    'time': end - start,
    'rkl': logs.rkl,
    'fkl': logs.fkl,
    'chisq': logs.chisq,
    'ess': logs.ess,
    'best_params': best_params,
    'max_ess': max_ess
}

# target_lowdim = low_dim_target(target, r, V_r, V_orth)


# max_deg = 2
# nf = PolynomialModel(r, target_lowdim, max_deg=max_deg)
# params = nf.init_params()
# params['A'][1] = params['A'][1].at[2].set(0.25)
# params['A'][1] = params['A'][1].at[0].set(-0.5)
# params['B'][1] = params['B'][1].at[0].set(np.power(0.5, 1/4))

# X = np.array([[1., 0.3]])
# nf.forward(params, X)

# samples, weights = nf.sample(2**10, params)
# np.mean(samples, 0)
# np.mean(samples * weights[:, None], 0)
# # np.sum(weights)**2 / np.sum(weights**2)

# X = sample_gaussian(2**10, r, seed=1, sampler='rqmc')
# nf.reverse_kl(params, X)

# params, logs = optimize(nf, params, max_iter=20, max_backtracking=20, nsample=2**10, seed=1, sampler='rqmc')
# print(params)

# y_samples, weights = nf.sample(2**10, params, seed=1, sampler='rqmc', return_weights=True)
# X = sample_gaussian(2**10, d, seed=1, sampler='rqmc')
# samples = nf.forward(params, X) @ V_r.T + X[:, r:] @ V_orth.T

# import matplotlib.pyplot as plt
# plt.scatter(samples[:, 0], samples[:, 1])    
# plt.savefig('test.png')
# plt.close()

# np.mean(samples, 0)
# np.cov(samples.T)
