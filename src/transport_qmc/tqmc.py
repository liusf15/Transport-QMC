import numpy as np
import itertools
from scipy.stats import qmc
import jax.numpy as jnp
from jax.scipy.special import betainc, ndtri, ndtr, logit
from jax.nn import sigmoid, logsumexp
from jax.scipy.stats.beta import pdf as beta_pdf
from jax.nn import softmax, sigmoid
import jax

from transport_qmc.utils import sample_uniform


def mixture_beta_cdf(x, shapes, weights):
    return jnp.dot(betainc(shapes[:,0], shapes[:, 1], x), softmax(weights))

def mixture_beta_log_pdf(x, shapes, weights):
    return jnp.log(jnp.dot(beta_pdf(x, shapes[:,0], shapes[:, 1]), softmax(weights)))

@jax.jit
def mixture_beta_inverse_cdf(q, shapes, weights):
    ws = softmax(weights)
    def h(x):
        return jnp.dot(betainc(shapes[:,0], shapes[:, 1], x), ws)
    def h_prime(x):
        return jnp.dot(beta_pdf(x, shapes[:,0], shapes[:, 1]), ws)
    tol = jnp.finfo(jnp.float32).eps
    def cond_fn(state):
        x, itercount = state
        return jnp.logical_and(jnp.abs(h(x) - q) > tol, itercount < 20)
    def step_fn(state):
        x, itercount = state
        return x - (h(x) - q) / h_prime(x), itercount + 1
    x = q
    x, itercount = jax.lax.while_loop(cond_fn, step_fn, (x, 0))
    return x

class TransportQMC:
    def __init__(self, d, target, base_transform='normal-icdf', nonlinearity='logit', num_composition=1, max_deg=3):
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
        log_det = jnp.sum(jnp.log(self.base_transform_grad(x)))
        x = self.base_transform(x)

        for p in params:
            x, log_det_ = self.forward_one_layer(p, x)
            log_det += log_det_
        return x, log_det
    
    def forward_from_gaussian(self, params, x):
        log_det = 0.
        for p in params:
            x, log_det_ = self.forward_one_layer(p, x)
            log_det += log_det_
        return x, log_det

    def elementwise_inverse(self, weights, x):
        log_det = jnp.sum(jnp.log(self.F_grad(x)))
        x = self.F(x)
        
        x = jax.vmap(mixture_beta_inverse_cdf, in_axes=[0, None, 0])(x, self.shapes, weights) ## TODO check order
        log_det += -jnp.sum(jax.vmap(mixture_beta_log_pdf, in_axes=[0, None, 0])(x, self.shapes, weights))

        eps = jnp.finfo(jnp.float32).eps
        x = jnp.clip(x, eps * .5, 1 - eps * .5)
        log_det += jnp.sum(jnp.log(self.F_inv_grad(x)))
        x = self.F_inv(x)
        return x, log_det

    def inverse_one_layer(self, params, x):
        weights = params['weights']
        L = jnp.diag(jnp.exp(params['L'][:self.d]))
        mask = np.tri(self.d, k=-1, dtype=bool)
        L = L.at[mask].set(params['L'][self.d:])
        b = params['b']        

        x, log_det_ = self.elementwise_inverse(weights, x)
        log_det = log_det_

        x = jax.scipy.linalg.solve_triangular(L, x - b, lower=True)
        log_det += -jnp.sum(params['L'][:self.d])
        return x, log_det

    def inverse(self, params, x):
        log_det = 0.
        for p in reversed(params):
            x, log_det_ = self.inverse_one_layer(p, x)
            log_det += log_det_
        return x, log_det

    def reverse_kl(self, params, u):
        z, log_det = jax.vmap(self.forward, in_axes=(None, 0))(params, u) 
        log_p = jax.vmap(self.target.log_prob)(z)
        return jnp.nanmean( - log_det - log_p)
    
    def ess(self, params, u):
        z, log_det = jax.vmap(self.forward, in_axes=(None, 0))(params, u)
        log_p = jax.vmap(self.target.log_prob)(z)
        log_weights = log_p + log_det
        log_weights = jnp.nan_to_num(log_weights, nan=-jnp.inf)
        log_weights -= jnp.max(log_weights)
        return jnp.exp(2 * logsumexp(log_weights) - logsumexp(2 * log_weights))

    def sample(self, params, nsample, seed=0):
        soboleng = qmc.Sobol(self.d, scramble=True, seed=seed)
        X = soboleng.random(nsample)
        eps = np.finfo(X.dtype).eps
        X = soboleng.random(nsample) * (1 - eps) + eps * .5
        X, log_det = jax.vmap(self.forward, in_axes=(None, 0))(params, X)
        log_p = jax.vmap(self.target.log_prob)(X)
        log_weights = log_p + log_det
        log_weights = jnp.nan_to_num(log_weights, nan=-jnp.inf)
        log_weights -= jnp.max(log_weights)
        if getattr(self.target, 'param_constrain', None):
            X = self.target.param_constrain(np.array(X, float))
        return X, log_weights


class TransportQMC_AS(TransportQMC):
    def __init__(self, d, r, V, target, base_transform='logit', nonlinearity='logit', num_composition=1, max_deg=3):
        super().__init__(d, target, base_transform, nonlinearity, num_composition, max_deg)
        self.r = r # number of active dimensions
        self.V = V # rotation matrix
    
    def init_one_layer(self):
        weights = jnp.zeros((self.r, len(self.shapes)))
        weights = weights.at[:, 0].set(1.)
        return {'weights': weights, 'L': jnp.zeros(self.r * (self.r + 1) // 2), 'b': jnp.zeros(self.r)}
    
    def init_params(self):
        weights_ = jnp.zeros((self.d - self.r, len(self.shapes)))
        weights_ = weights_.at[:, 0].set(1.)
        params_inactive = {'D': jnp.zeros(self.d - self.r), 'b': jnp.zeros(self.d - self.r), 'weights': weights_} 
        params_active = [self.init_one_layer() for _ in range(self.num_composition)]
        return {'active': params_active, 'inactive': params_inactive}
    
    def forward_one_layer(self, params, y):
        weights = params['weights']
        
        L_r = jnp.diag(jnp.exp(params['L'][:self.r]))
        mask = np.tri(self.r, k=-1, dtype=bool)
        L_r = L_r.at[mask].set(params['L'][self.r:])
        b = params['b']        

        # linear
        y = jnp.dot(L_r, y) + b
        log_det = jnp.sum(params['L'][:self.r])

        # elementwise transform
        y, log_det_T = self.elementwise(weights, y)
        log_det += log_det_T
        return y, log_det

    def forward(self, params, x):
        log_det = jnp.sum(jnp.log(self.base_transform_grad(x)))
        x = self.base_transform(x)

        y = x[:self.r]
        for p in params['active']:
            y, log_det_ = self.forward_one_layer(p, y)
            log_det += log_det_

        z = x[self.r:]
        z = jnp.exp(params['inactive']['D']) * z + params['inactive']['b']   
        log_det += jnp.sum(params['inactive']['D'])
        
        z, log_det_ = self.elementwise(params['inactive']['weights'], z)
        log_det += log_det_

        x = jnp.concatenate([y, z])
        return x, log_det

    def forward_from_normal(self, params, x):
        log_det = 0.
        y = x[:self.r]
        for p in params['active']:
            y, log_det_ = self.forward_one_layer(p, y)
            log_det += log_det_

        z = x[self.r:]
        z = jnp.exp(params['inactive']['D']) * z + params['inactive']['b']   
        log_det += jnp.sum(params['inactive']['D'])

        z, log_det = self.elementwise(params['inactive']['weights'], z)
        log_det += log_det

        x = jnp.concatenate([y, z])
        return x, log_det

    def reverse_kl(self, params, u):
        z, log_det = jax.vmap(self.forward, in_axes=(None, 0))(params, u) 
        log_p = jax.vmap(self.target.log_prob)(z @ self.V.T)
        return jnp.nanmean( - log_det - log_p)
    
    def ess(self, params, u):
        X, log_det = jax.vmap(self.forward, in_axes=(None, 0))(params, u)
        log_p = jax.vmap(self.target.log_prob)(X @ self.V.T)
        log_weights = log_p + log_det
        log_weights = jnp.nan_to_num(log_weights, nan=-jnp.inf)
        log_weights -= jnp.max(log_weights)
        return jnp.exp(2 * logsumexp(log_weights) - logsumexp(2 * log_weights))

    def sample(self, params, nsample, rng, sampler='rqmc'):
        X = sample_uniform(nsample, self.d, rng, sampler)
        X, log_det = jax.vmap(self.forward, in_axes=(None, 0))(params, X)
        X = X @ self.V.T
        log_p = jax.vmap(self.target.log_prob)(X)
        log_weights = log_p + log_det
        log_weights = jnp.nan_to_num(log_weights, nan=-jnp.inf)
        log_weights -= np.max(log_weights)
        if getattr(self.target, 'param_constrain', None):
            X = self.target.param_constrain(np.array(X, float))
        return X, log_weights
    