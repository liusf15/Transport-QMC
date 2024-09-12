import numpy as np
import itertools
from scipy.stats import qmc
import jax.numpy as jnp
from jax.scipy.special import betainc, ndtri, ndtr, logit, logsumexp
from jax.nn import sigmoid
from jax.scipy.stats.beta import pdf as beta_pdf
from jax.nn import softmax, sigmoid
import jax

from typing import NamedTuple

Metrics = NamedTuple('Metrics', [('rkl', float), ('fkl', float), ('chisq', float), ('ess', float)])

MACHINE_EPSILON = np.finfo(np.float64).eps



def mixture_beta_cdf(x, shapes, weights):
    return jnp.dot(betainc(shapes[:,0], shapes[:, 1], x), softmax(weights))

def mixture_beta_log_pdf(x, shapes, weights):
    return jnp.log(jnp.dot(beta_pdf(x, shapes[:,0], shapes[:, 1]), softmax(weights)))

class TransportQMC:
    def __init__(self, d, target, base_transform='logit', nonlinearity='logit', num_composition=1, max_deg=3):
        self.d = d
        self.target = target
        if base_transform == 'logit':
            self.base_transform = logit
            self.base_transform_grad = lambda x: 1 / (x * (1 - x))
        elif base_transform == 'normal-icdf':
            self.base_transform = ndtri
            self.base_transform_grad = lambda x: jnp.sqrt(2 * np.pi) * jnp.exp(0.5 * ndtri(x)**2)
        else:
            raise ValueError(f"'base_transform' must be 'logit' or 'normal-icdf', got {base_transform}")
        
        if nonlinearity == 'logit':
            self.F_inv, self.F = logit, sigmoid
            self.F_inv_grad = lambda x: 1 / (x * (1 - x))
            self.F_grad = lambda x: jnp.exp(-x) / (1 + jnp.exp(-x))**2
        elif nonlinearity == 'normal-icdf':
            self.F_inv, self.F = ndtri, ndtr
            self.F_inv_grad = lambda x: jnp.sqrt(2 * np.pi) * jnp.exp(0.5 * ndtri(x)**2)
            self.F_grad = lambda x: jnp.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
        else:
            raise ValueError(f"'nonlinearity' must be 'logit' or 'normal-icdf', got {nonlinearity}")
        self.num_composition = num_composition
        self.max_deg = max_deg
        self.shapes = np.array([degs for degs in itertools.product(np.arange(1, self.max_deg), repeat=2) if sum(degs) <= self.max_deg])
    
    def init_one_layer(self):
        weights = jnp.zeros((self.d, len(self.shapes)))
        weights = weights.at[:, 0].set(1.)
        return {'weights': weights, 'L': jnp.zeros(self.d * (self.d + 1) // 2), 'b': jnp.zeros(self.d)}

    def init_params(self):
        params = []
        for _ in range(self.num_composition):
            params.append(self.init_one_layer())
        return params

    def elementwise(self, weights, x):
        log_det = jnp.sum(jnp.log(self.F_grad(x)))
        x = self.F(x)
        
        log_det += jnp.sum(jax.vmap(mixture_beta_log_pdf, in_axes=[0, None, 0])(x, self.shapes, weights))
        x = jax.vmap(mixture_beta_cdf, in_axes=[0, None, 0])(x, self.shapes, weights)

        eps = jnp.finfo(jnp.float32).eps
        x = jnp.clip(x, eps * .5, 1 - eps * .5)
        log_det += jnp.sum(jnp.log(self.F_inv_grad(x)))
        x = self.F_inv(x)
        return x, log_det
    
    def forward_one_layer(self, params, x):
        weights = params['weights']
        
        L = jnp.diag(jnp.exp(params['L'][:self.d]))
        mask = np.tri(self.d, k=-1, dtype=bool)
        L = L.at[mask].set(params['L'][self.d:])

        b = params['b']        
        # linear
        x = jnp.dot(L, x) + b
        log_det = jnp.sum(params['L'][:self.d])
        # elementwise transform
        x, log_det_T = self.elementwise(weights, x)
        log_det += log_det_T
        return x, log_det

    def forward(self, params, x):
        # log_det = jnp.sum(jnp.log(self.F_inv_grad(x)))
        # x = self.F_inv(x)
        log_det = jnp.sum(jnp.log(self.base_transform_grad(x)))
        x = self.base_transform(x)
        # log_det = jnp.sum(jnp.log(jnp.sqrt(2 * np.pi) * jnp.exp(0.5 * x **2)))

        for p in params:
            x, log_det_ = self.forward_one_layer(p, x)
            log_det += log_det_
        return x, log_det

    def reverse_kl(self, params, u):
        """
        u: (num_samples, d) uniform samples
        """
        z, log_det = jax.vmap(self.forward, in_axes=(None, 0))(params, u) 
        log_p = jax.vmap(self.target.log_prob)(z)
        # log_p = log_p.at[abs(log_p) > 1e15].set(jnp.nan)
        log_p = jnp.where(jnp.abs(log_p) > 1e15, jnp.nan, log_p)
        return jnp.nanmean( - log_det - log_p)
    
    def metrics(self, params, u):
        X, log_det = jax.vmap(self.forward, in_axes=(None, 0))(params, u)
        log_p = jax.vmap(self.target.log_prob)(X)
        log_weights = log_p + log_det
        offset = jnp.max(log_weights)
        weights = jnp.exp(log_weights - offset)
        ess = jnp.sum(weights)**2 / jnp.sum(weights**2)
        chisq = logsumexp(2 * (log_weights - offset)) - jnp.log(len(log_weights)) + 2 * offset
        rkl = jnp.mean(- log_weights)
        fkl = logsumexp(log_weights, b=log_weights - offset) + offset * logsumexp(log_weights)

        return Metrics(rkl, fkl, chisq, ess)

    def sample(self, params, nsample, seed=0):
        """
        seed: either integer seed or a numpy random generator
        """
        soboleng = qmc.Sobol(self.d, scramble=True, seed=seed)
        X = soboleng.random(nsample) * (1 - MACHINE_EPSILON) + MACHINE_EPSILON * .5
        X, log_det = jax.vmap(self.forward, in_axes=(None, 0))(params, X)
        proposal_log_densities = - log_det
        target_log_densities = jax.vmap(self.target.log_prob)(X)
        log_weights = target_log_densities - proposal_log_densities
        log_weights -= np.mean(log_weights)
        weights = np.exp(log_weights)
        if getattr(self.target, 'param_constrain', None):
            X = self.target.param_constrain(np.array(X, float))
        return X, weights
    
    def forward_rotation(self, params, x, rot):
        x = ndtri(x)
        log_det = jnp.sum(jnp.log(jnp.sqrt(2 * np.pi) * jnp.exp(0.5 * x **2)))

        x = jnp.dot(rot, x)

        for p in params:
            x, log_det_ = self.forward_one_layer(p, x)
            log_det += log_det_
        
        return x, log_det
    
