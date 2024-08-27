import numpy as np
from scipy.stats import qmc
from tqdm import trange
from scipy.special import hermite
import itertools

import jax.numpy as jnp
from jax.scipy.special import betainc, ndtri, ndtr, logit, logsumexp
from jax.scipy.stats.beta import pdf as beta_pdf
from jax.nn import softmax, sigmoid
import jax
import optax

from qmc_flow.utils import sample_gaussian, get_moments, get_effective_sample_size
MACHINE_EPSILON = np.finfo(np.float64).eps


class TransportMap:
    def __init__(self, d, target, max_deg) -> None:
        self.d = d
        self.target = target
        self.max_deg = max_deg
    
    def forward(self, params, x):
        raise NotImplementedError

    def forward_and_logdet(self, params, x):
        raise NotImplementedError
    
    def reverse_kl(self, params, x):
        raise NotImplementedError

    def sample(self, n, params):
        raise NotImplementedError

def get_beta_shapes(max_deg):
    return np.array([degs for degs in itertools.product(np.arange(1, max_deg), repeat=2) if sum(degs) <= max_deg])

def weighted_beta_cdf(x, shapes, weights):
    return jnp.dot(betainc(shapes[:,0], shapes[:, 1], x), weights)

def weighted_beta_pdf(x, shapes, weights):
    return jnp.dot(beta_pdf(x, shapes[:,0], shapes[:, 1]), weights)

def bernstein_poly(x, deg, weights):
    return jnp.dot(betainc(jnp.arange(1, deg + 1), jnp.arange(deg, 0, -1), x), weights)

def bernstein_poly_grad(x, deg, weights):
    return jnp.dot(beta_pdf(x, jnp.arange(1, deg + 1), jnp.arange(deg, 0, -1)), weights)

class CopulaModel(TransportMap):
    def __init__(self, d, target, max_deg, link_func=['sigmoid', 'logit']) -> None:
        super().__init__(d, target, max_deg)
        self.shapes = get_beta_shapes(max_deg)
        self.num_shapes = len(self.shapes)
        self.link_func = link_func
        if link_func[0] == 'sigmoid':
            self.R_to_01 = sigmoid
        elif link_func[0] == 'ndtr':
            self.R_to_01 = ndtr
        else:
            raise ValueError(f"Unknown link function: {link_func[0]}")
        if link_func[1] == 'logit':
            self.R_to_01_inv = logit
        elif link_func[1] == 'ndtri':
            self.R_to_01_inv = ndtri
        elif link_func[1] == 'positive_range':
            self.R_to_01_inv = lambda x: -jnp.log(1 - x)
        else:
            raise ValueError(f"Unknown link function: {link_func[1]}")
    
    def init_params(self):
        d = self.d
        num_params = d + d + d * (d - 1) // 2 + d * len(self.shapes)
        weights_unc = jnp.zeros((d, self.num_shapes))
        weights_unc = weights_unc.at[:, 0].set(1.)
        return jnp.concat([jnp.zeros(d + d + d * (d - 1) // 2), weights_unc.flatten()])
    
    def pack_params(self, mu, L, weights):
        weights_unc = jnp.log(weights)
        # weights_unc -= weights_unc[:, -1:]
        params = jnp.concatenate([mu, jnp.log(jnp.diag(L)), L[np.tri(self.d, k=-1, dtype=bool)].flatten(), weights_unc.flatten()])
        return params
    
    # def set_params(self, params):
    #     self.mu = params[:self.d]
    #     self.L = params[self.d:self.d+self.d**2].reshape(self.d, self.d)
    #     self.weights_ = params[self.d+self.d**2:].reshape(self.d, self.max_deg)
    #     self.params = params
    
    def unpack_params(self, params):
        mu = params[:self.d]
        L = jnp.diag(jnp.exp(params[self.d:self.d+self.d]))

        mask = np.tri(self.d, k=-1, dtype=bool)
        # Fill the lower triangular part
        L = L.at[mask].set(params[self.d+self.d:self.d+self.d+self.d*(self.d-1)//2])
        # L = params[self.d:self.d+self.d**2].reshape(self.d, self.d)
        weights_unc = params[self.d+self.d+self.d*(self.d-1)//2:].reshape(self.d, self.num_shapes)
        # weights_unc_0 = jnp.concatenate([weights_unc, jnp.zeros((self.d, 1))], axis=1)
        weights = softmax(weights_unc, axis=1)
        # TODO: fix the first weight
        return mu, L, weights

    def forward(self, params, x):
        mu, L, weights = self.unpack_params(params)

        # shift and scale
        t_x = mu + x @ L.T

        # map to [0, 1]
        t_x = self.R_to_01(t_x)

        # transformation by Bernstein polynomial
        output = []
        for j in range(self.d):
            # output.append(bernstein_poly(t_x[:, j:j+1], deg=self.max_deg, weights=weights[j]))
            output.append(weighted_beta_cdf(t_x[:, j:j+1], self.shapes, weights[j]))
        output = jnp.stack(output, axis=1)

        # map to R
        output = jnp.clip(output, 1e-13, 1 - 1e-13)
        output = self.R_to_01_inv(output)

        return output

    def forward_and_logdet(self, params, x):
        """
        x: (n, d)
        link_func: function that maps R to [0, 1] and [0, 1] to the desired range
        """
        mu, L, weights = self.unpack_params(params)

        t_x = mu + x @ L.T
        # log_det = jnp.log(det(L @ L.T)) / 2
        log_det = jnp.sum(params[self.d:self.d+self.d])
        
        if self.link_func[0] == 'ndtr':
            # use \Phi to map R to [0, 1]
            log_det += jnp.sum(-.5 * t_x**2 - .5 * jnp.log(2 * jnp.pi), axis=1)
            t_x = ndtr(t_x)
            t_x = jnp.clip(t_x, MACHINE_EPSILON, 1 - MACHINE_EPSILON) # avoid numerical instability

        elif self.link_func[0] == 'sigmoid':
            # use sigmoid to map R to [0, 1]
            log_det += jnp.sum(-t_x - jnp.log(1 + jnp.exp(-t_x)) * 2., axis=1)
            t_x = 1 / (1 + jnp.exp(-t_x))

        else:
            raise ValueError(f"Unknown link function: {self.link_func[0]}")

        output = []
        for j in range(self.d):
            # output.append(bernstein_poly(t_x[:, j:j+1], deg=self.max_deg, weights=weights[j]))
            output.append(weighted_beta_cdf(t_x[:, j:j+1], self.shapes, weights[j]))

            # log_det += jnp.log(bernstein_poly_grad(t_x[:, j:j+1], deg=self.max_deg, weights=weights[j]))
            log_det += jnp.log(weighted_beta_pdf(t_x[:, j:j+1], self.shapes, weights[j]))
        output = jnp.stack(output, axis=1)
        

        if self.link_func[1] == 'ndtri':
            # use \Phi^{-1} to map [0, 1] to R
            output = ndtri(output)
            log_det += jnp.sum(+.5 * output**2 + .5 * jnp.log(2 * jnp.pi), axis=1)

        elif self.link_func[1] == 'logit':
            # use logit to map [0, 1] to R
            output = jnp.clip(output, MACHINE_EPSILON * .5, 1 - MACHINE_EPSILON * .5)
            log_det += jnp.sum(-jnp.log(output) - jnp.log(1 - output), axis=1)
            output = jnp.log(output / (1 - output))

        # use log (1/(1-x)) to map [0,1] to R_+
        elif self.link_func[1] == 'positive_range':
            log_det += jnp.sum(-jnp.log(1 - output), axis=1)
            output = -jnp.log(1 - output)
        
        else:
            raise ValueError(f"Unknown link function: {self.link_func[1]}")

        return output, log_det
    
    def divergence(self, params, x, div_name, alpha=2.):
        log_q = -0.5 * jnp.sum(x**2, axis=-1) - 0.5 * self.d * jnp.log(2 * jnp.pi)
        z, log_det = self.forward_and_logdet(params, x) 
        log_p = jax.vmap(self.target.log_prob)(z)
        log_weight = log_p - log_q + log_det
        mask_inf = abs(log_weight) != jnp.inf
        # log_weight = log_weight[mask_inf]
        offset = jnp.mean(log_weight)
        if div_name == 'rkl':
            return jnp.mean(-log_weight)
        if div_name == 'fkl':
            return logsumexp(log_weight, b=log_weight - offset) + offset * logsumexp(log_weight)
        if div_name == 'chisq':
            return logsumexp(2 * (log_weight - offset)) - jnp.log(len(log_weight)) + 2 * offset # when the weights are equal, this is equal to -2 * rkl
        if div_name == 'alpha':
            return logsumexp(alpha * (log_weight - offset)) - jnp.log(len(log_weight)) + alpha * offset
        if div_name == 'all':
            rkl = jnp.mean(-log_weight)
            fkl = logsumexp(log_weight, b=log_weight - offset) + offset * logsumexp(log_weight)
            chisq = logsumexp(2 * (log_weight - offset)) - jnp.log(len(log_weight)) + 2 * offset
            return rkl, fkl, chisq

    def reverse_kl(self, params, x):
        """
        x: (num_samples, d)
        """
        log_q = -0.5 * jnp.sum(x**2, axis=-1) - 0.5 * self.d * jnp.log(2 * jnp.pi)
        z, log_det = self.forward_and_logdet(params, x) 
        log_p = jax.vmap(self.target.log_prob)(z)
        return jnp.mean(log_q - log_det - log_p)

    def forward_kl(self, params, x, offset_logweight):
        log_q = -0.5 * jnp.sum(x**2, axis=-1) - 0.5 * self.d * jnp.log(2 * jnp.pi)
        z, log_det = self.forward_and_logdet(params, x) 
        log_p = jax.vmap(self.target.log_prob)(z)
        log_weight = log_p - log_q + log_det
        weight = jnp.exp(log_weight - offset_logweight)
        return jnp.mean(weight * log_weight)

    def weight_variance(self, params, x, offset_logweight):
        log_q = -0.5 * jnp.sum(x**2, axis=-1) - 0.5 * self.d * jnp.log(2 * jnp.pi)
        z, log_det = self.forward_and_logdet(params, x) 
        log_p = jax.vmap(self.target.log_prob)(z)
        log_weight = log_p - log_q + log_det
        weight = jnp.exp(log_weight - offset_logweight)
        # return jnp.sum(weight)**2 / jnp.sum(weight**2)
        return jnp.log(jnp.sum(weight**2)) - 2 * jnp.log(jnp.sum(weight))

    def sample(self, nsample, params, constrained=True, seed=0, sampler='rqmc', return_weights=False):
        """
        seed: either integer seed or a numpy random generator
        """
        X = sample_gaussian(nsample, self.d, seed=seed, sampler=sampler)
        Z_nf, log_det = self.forward_and_logdet(params, X)
        if not return_weights:
            if constrained:
                return self.target.param_constrain(np.array(Z_nf, float))
        log_q = -0.5 * jnp.sum(X**2, axis=-1) - 0.5 * self.d * jnp.log(2 * jnp.pi)
        proposal_log_densities = log_q - log_det
        target_log_densities = jax.vmap(self.target.log_prob)(Z_nf)
        log_weights = target_log_densities - proposal_log_densities
        log_weights -= np.mean(log_weights)
        weights = np.exp(log_weights)
        if constrained:
            Z_nf = self.target.param_constrain(np.array(Z_nf, float))
        return Z_nf, weights


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
        # if init == 'zero':
        #     self.coefs = jnp.zeros(len(self.indices))
        # elif init == 'identity':
        #     self.coefs = jnp.zeros(len(self.indices))
        #     self.coefs = self.coefs.at[0].set(1.) 
        # else:
        #     self.coefs = jax.random.normal(key, (len(self.indices),))
        
    def __call__(self, x, coefs):
        basis_functions = [multivariate_polynomial(idx, x) for idx in self.indices]
        basis = jnp.stack(basis_functions, axis=1)
        return jnp.dot(basis, coefs)
    
class PolynomialModel(TransportMap):
    def __init__(self, d, target, max_deg) -> None:
        super().__init__(d, target, max_deg)

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
