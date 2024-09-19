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
from qmc_flow.models.tqmc import TransportQMC

Metrics = NamedTuple('Metrics', [('rkl', float), ('fkl', float), ('chisq', float), ('ess', float)])

MACHINE_EPSILON = np.finfo(np.float64).eps

def mixture_beta_cdf(x, shapes, weights):
    return jnp.dot(betainc(shapes[:,0], shapes[:, 1], x), softmax(weights))

def mixture_beta_log_pdf(x, shapes, weights):
    return jnp.log(jnp.dot(beta_pdf(x, shapes[:,0], shapes[:, 1]), softmax(weights)))

class TransportQMC_AS(TransportQMC):
    def __init__(self, d, r, V, target, base_transform='logit', nonlinearity='logit', num_composition=1, max_deg=3):
        super().__init__(d, target, base_transform, nonlinearity, num_composition, max_deg)
        self.r = r # number of active dimensions
        self.V = V # rotation matrix
    
    def init_one_layer(self):
        weights = jnp.zeros((self.r, len(self.shapes)))
        weights = weights.at[:, 0].set(1.)
        return {'weights': weights, 'L': jnp.zeros(self.r * (self.r + 1) // 2), 'D': jnp.zeros(self.d - self.r), 'b': jnp.zeros(self.d)}

    def init_params(self):
        params = []
        for _ in range(self.num_composition):
            params.append(self.init_one_layer())
        return params

    def elementwise(self, weights, x):
        log_det = jnp.sum(jnp.log(self.F_grad(x[:self.r])))
        y = self.F(x[:self.r])
        
        log_det += jnp.sum(jax.vmap(mixture_beta_log_pdf, in_axes=[0, None, 0])(y, self.shapes, weights))
        y = jax.vmap(mixture_beta_cdf, in_axes=[0, None, 0])(y, self.shapes, weights)

        eps = jnp.finfo(jnp.float32).eps
        y = jnp.clip(y, eps * .5, 1 - eps * .5)
        log_det += jnp.sum(jnp.log(self.F_inv_grad(y)))
        y = self.F_inv(y)
        x = x.at[:self.r].set(y)
        return x, log_det
    
    def forward_one_layer(self, params, x):
        weights = params['weights']
        
        L_r = jnp.diag(jnp.exp(params['L'][:self.r]))
        mask = np.tri(self.r, k=-1, dtype=bool)
        L_r = L_r.at[mask].set(params['L'][self.r:])
        b = params['b']        
        D = jnp.exp(params['D'])
        # linear
        y = jnp.dot(L_r, x[:self.r]) + b[:self.r]
        z = D * x[self.r:] + b[self.r:]
        
        log_det = jnp.sum(params['L'][:self.d]) + jnp.sum(params['D'])

        # elementwise transform
        y, log_det_T = self.elementwise(weights, y)
        log_det += log_det_T

        x = jnp.concatenate([y, z])
        return x, log_det

    def forward(self, params, x):
        log_det = jnp.sum(jnp.log(self.base_transform_grad(x)))
        x = self.base_transform(x)

        for p in params:
            x, log_det_ = self.forward_one_layer(p, x)
            log_det += log_det_
        return x, log_det

    def reverse_kl(self, params, u):
        """
        u: (num_samples, d) uniform samples
        """
        z, log_det = jax.vmap(self.forward, in_axes=(None, 0))(params, u) 
        log_p = jax.vmap(self.target.log_prob)(z @ self.V.T)
        return jnp.nanmean( - log_det - log_p)
    
    def metrics(self, params, u):
        X, log_det = jax.vmap(self.forward, in_axes=(None, 0))(params, u)
        log_p = jax.vmap(self.target.log_prob)(X @ self.V.T)
        log_weights = log_p + log_det
        
        offset = jnp.nanmax(log_weights)
        weights = jnp.exp(log_weights - offset)
        rkl = jnp.nanmean(- log_weights)
        ess = jnp.nansum(weights)**2 / jnp.nansum(weights**2)

        mask = ~jnp.isnan(log_weights)
        log_weights = jnp.where(mask, log_weights, -jnp.inf)
        log_weights_0 = jnp.where(mask, log_weights, 0.)
        chisq = logsumexp(2 * log_weights) - jnp.log(len(log_weights))
        fkl = logsumexp(log_weights, b=log_weights_0)

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
        log_weights -= np.nanmean(log_weights)
        weights = np.exp(log_weights)
        if getattr(self.target, 'param_constrain', None):
            X = self.target.param_constrain(np.array(X, float))
        return X, weights
    