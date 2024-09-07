import numpy as np
from scipy.special import hermite
import itertools

import jax.numpy as jnp
import jax

from qmc_flow.utils import sample_gaussian
MACHINE_EPSILON = np.finfo(np.float64).eps

def hermite_polynomial(degree):
    """
    return a jax function that takes in x and returns the value of the hermite polynomial of the given degree
    """
    coefs = hermite(degree).coef
    return lambda x: jnp.polyval(coefs, x)

def multivariate_polynomial(multi_index, x):
    """
    return the value of the multivariate polynomial with the given multi_index at the given x
    """
    return jnp.prod(jnp.array([hermite_polynomial(j)(x[:, i]) for i, j in enumerate(multi_index)]), axis=0)

def int_prod_hermite(h1, h2, x):
    h_prod_int = jnp.polyint(jnp.polymul(h1, h2))
    return jnp.polyval(h_prod_int, x)

def get_index_set(d, i, max_deg, min_deg=0):
    """
    get the d-tuples such that only the first i elements are nonzero, and 
    the sum of the elements is less than or equal to max_deg, and 
    the sum of the elements is greater than or equal to min_deg
    """
    indices = itertools.product(range(min_deg, max_deg + 1), repeat=i)
    indices = [idx for idx in indices if sum(idx) <= max_deg and sum(idx) >= min_deg]
    return [idx + (0,) * (d - i) for idx in indices]

def sum_polynomials(indices, x, coefs):
    basis_functions = [multivariate_polynomial(idx, x) for idx in indices]
    basis = jnp.stack(basis_functions, axis=1)
    return jnp.dot(basis, coefs)

class PolynomialBasis:
    """
    class of d-dimensional polynomials of degree at most degree
    """
    def __init__(self, d, i, degree, key=None, init='zero'):
        self.d = d
        self.degree = degree
        self.indices = get_index_set(d, i, degree)
        
    def __call__(self, x, coefs):
        basis_functions = [multivariate_polynomial(idx, x) for idx in self.indices]
        basis = jnp.stack(basis_functions, axis=1)
        return jnp.dot(basis, coefs)
    
class PolynomialModel:
    def __init__(self, d, target, max_deg) -> None:
        self.d = d
        self.target = target
        self.max_deg = max_deg

    def init_params(self):
        d = self.d
        max_deg = self.max_deg
        self.A_basis = []
        self.B_basis = []
        self.A_coefs = []
        self.B_coefs = []
        for i in range(d):
            self.A_basis.append(PolynomialBasis(d, i, max_deg))
            self.A_coefs.append(jnp.zeros(len(self.A_basis[-1].indices)))

            self.B_basis.append(PolynomialBasis(d, i+1, 1))
            b_coef_i = jnp.zeros(len(self.B_basis[-1].indices))
            b_coef_i = b_coef_i.at[0].set(1.)
            self.B_coefs.append(b_coef_i)

        return {'A': self.A_coefs, 'B': self.B_coefs}

    def forward(self, params, x, return_log_det=False):
        d = self.d
        output = jnp.zeros_like(x)
        if return_log_det:
            log_det = jnp.zeros(x.shape[0])
        
        A_coefs = params['A']
        B_coefs = params['B']
        for i in range(d):
            a_i = self.A_basis[i]
            b_i = self.B_basis[i]

            output_i = a_i(x, A_coefs[i])
            
            b_indices = b_i.indices
            b_coefs = B_coefs[i]
            for j, j_idx in enumerate(b_indices):
                for jp, jp_idx in enumerate(b_indices):
                    j_idx_mask_i = j_idx[:i] + (0,) + j_idx[i+1:]
                    jp_idx_mask_i = jp_idx[:i] + (0,) + jp_idx[i+1:]
                    output_i += (b_coefs[j] * b_coefs[jp] * 
                                multivariate_polynomial(j_idx_mask_i, x) * 
                                multivariate_polynomial(jp_idx_mask_i, x) * 
                                int_prod_hermite(hermite(j_idx[i]).coef, 
                                                hermite(jp_idx[i]).coef, 
                                                x[:, i]))
            output = output.at[:, i].set(output_i)
        if return_log_det:
            for i in range(d):
                log_det += jnp.log((self.B_basis[i](x, B_coefs[i]))**2)
            return output, log_det
        return output

    def reverse_kl(self, params, x):
        """
        x: (num_samples, d)
        """
        log_q = -0.5 * jnp.sum(x**2, axis=-1) - 0.5 * self.d * jnp.log(2 * jnp.pi)
        z, log_det = self.forward(params, x, return_log_det=True)
        log_p = jax.vmap(self.target.log_prob)(z)
        return jnp.mean(log_q - log_det - log_p)

    def sample(self, n, params, seed=0, sampler='rqmc', return_weights=False, constrained=False):
        """
        seed: either integer seed or a numpy random generator
        """
        X = sample_gaussian(n, self.d, seed=seed, sampler=sampler)
        Z_nf, log_det = self.forward(params, X, return_log_det=True)
        if not return_weights:
            if constrained:
                return self.target.param_constrain(np.array(Z_nf, float))
        log_q = -0.5 * jnp.sum(X**2, axis=-1) - 0.5 * self.d * jnp.log(2 * jnp.pi)
        proposal_log_densities = log_q - log_det
        target_log_densities = jax.vmap(self.target.log_prob)(Z_nf)
        log_weights = target_log_densities - proposal_log_densities
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        if constrained:
            Z_nf = self.target.param_constrain(np.array(Z_nf, float))
        return Z_nf, weights
    
    # def get_params(self):
    #     coefs_a = jnp.concatenate([self.As[i].coefs.flatten() for i in range(self.d)])
    #     coefs_b = jnp.concatenate([self.Bs[i].coefs.flatten() for i in range(self.d)])
    #     params = jnp.concatenate([coefs_a, coefs_b])
    #     return params
    
    # def unpack_params(self, params):
    #     split_a = params[:len(self.coefs_a)]
    #     split_b = params[len(self.coefs_a):]
    #     coefs_a = []
    #     cur = 0
    #     for i in range(self.d):
    #         l = len(self.coefs_a[i])
    #         coefs_a.append(split_a[cur:cur + l])
    #         cur += l
        
    #     cur = 0
    #     coefs_b = []
    #     for i in range(self.d):
    #         l = len(self.coefs_b[i])
    #         coefs_b.append(split_b[cur:cur + l])
    #         cur += l
    #     return coefs_a, coefs_b

    # def set_params(self, params):
    #     cur = 0
    #     for i in range(self.d):
    #         l = len(self.coefs_a[i])
    #         self.As[i].coefs = params[cur:cur + l]
    #         cur += l
    #     for i in range(self.d):
    #         l = len(self.coefs_b[i])
    #         self.Bs[i].coefs = params[cur:cur + l]
    #         cur += l

