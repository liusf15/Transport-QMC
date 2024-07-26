import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc
from tqdm import trange

import jax.numpy as jnp
from jax.scipy.special import betainc, ndtri, ndtr, logit
from jax.scipy.stats.beta import pdf as beta_pdf
from jax.nn import softmax, sigmoid
from jax.scipy.linalg import det
import jax
import optax

from experiment.polynomial.targets import Gaussian, LogNormal
MACHINE_EPSILON = np.finfo(np.float64).eps

def bernstein_poly(x, deg, weights):
    return jnp.dot(betainc(jnp.arange(1, deg + 1), jnp.arange(deg, 0, -1), x), weights)

def bernstein_poly_grad(x, deg, weights):
    return jnp.dot(beta_pdf(x, jnp.arange(1, deg + 1), jnp.arange(deg, 0, -1)), weights)

class TransportMap():
    def __init__(self, d, target, max_deg, link_func=['sigmoid', 'logit']) -> None:
        self.d = d
        self.target = target
        self.max_deg = max_deg
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
        num_params = d + d + d * (d - 1) // 2 + d * self.max_deg
        return jnp.zeros(num_params)
        # self.mu = jnp.zeros(self.d)
        # self.L = jnp.eye(self.d)
        # self.weights_ = jnp.zeros((self.d, self.max_deg))
        # self.params = self.pack_params()
    
    def pack_params(self):
        mask = np.tri(self.d, k=-1, dtype=bool)
        return jnp.concatenate([self.mu, np.diag(self.L), self.L[mask].flatten(), self.weights_.flatten()])
    
    def set_params(self, params):
        self.mu = params[:self.d]
        self.L = params[self.d:self.d+self.d**2].reshape(self.d, self.d)
        self.weights_ = params[self.d+self.d**2:].reshape(self.d, self.max_deg)
        self.params = params
    
    def unpack_params(self, params):
        mu = params[:self.d]
        L = jnp.diag(jnp.exp(params[self.d:self.d+self.d]))

        mask = np.tri(self.d, k=-1, dtype=bool)
        # Fill the lower triangular part
        L = L.at[mask].set(params[self.d+self.d:self.d+self.d+self.d*(self.d-1)//2])
        # L = params[self.d:self.d+self.d**2].reshape(self.d, self.d)
        weights_unc = params[self.d+self.d+self.d*(self.d-1)//2:].reshape(self.d, self.max_deg)
        weights = softmax(weights_unc, axis=1)
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
            output.append(bernstein_poly(t_x[:, j:j+1], deg=self.max_deg, weights=weights[j]))
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
        log_det = jnp.log(det(L @ L.T)) / 2
        
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
            output.append(bernstein_poly(t_x[:, j:j+1], deg=self.max_deg, weights=weights[j]))
            log_det += jnp.log(bernstein_poly_grad(t_x[:, j:j+1], deg=self.max_deg, weights=weights[j]))
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
    
    def reverse_kl(self, params, x):
        """
        x: (num_samples, d)
        """
        log_q = -0.5 * jnp.sum(x**2, axis=-1) - 0.5 * self.d * jnp.log(2 * jnp.pi)
        z, log_det = self.forward_and_logdet(params, x)
        # log_p = jnp.array([self.target.log_prob(np.array(z[i], dtype=float)) for i in range(z.shape[0])])
        log_p = jnp.array(self.target.log_prob(np.array(z, dtype=float)))
        return jnp.mean(log_q - log_det - log_p)
    
    def _log_p_grad(self, params, x):
        """
        gradient of log p(T(x; params)) w.r.t. params
        """
        z = self.forward(params, x) # TODO: use vectorized version
        T_grad = jax.jacfwd(self.forward)(params, x)
        # target_log_p = jnp.array([self.target.log_prob_grad(np.array(z[i], dtype=float)) for i in range(z.shape[0])])
        log_p_grad = jnp.array(self.target.log_prob_grad(np.array(z, dtype=float)))
        return jnp.einsum('ijk,ij->k', T_grad, log_p_grad) / z.shape[0]

    def reverse_kl_grad(self, params, x):
        def logdet_mean(params, x):
            log_det = self.forward_and_logdet(params, x)[1]
            return jnp.mean(log_det)
        log_det_grad = jax.jacfwd(logdet_mean)(params, x)
        
        # log_p_grad = np.mean([self._log_p_grad(params, x[i]) for i in range(x.shape[0])], axis=0) #!! super slow
        log_p_grad = self._log_p_grad(params, x)
        return -log_det_grad - log_p_grad
    
    def gradient_descent(self, max_iter=50, lr=1e-2, nsample=2**10, seed=0, print_every=100):
        params = self.init_params()
        losses = []

        soboleng = qmc.Sobol(self.d, scramble=True, seed=seed)
        X = ndtri(soboleng.random(nsample) * (1 - MACHINE_EPSILON) + .5 * MACHINE_EPSILON)
        for t in range(max_iter):
            loss = self.reverse_kl(params, X)
            grad = self.reverse_kl_grad(params, X) # redundant computation
            params = params - lr * grad
            losses.append(loss.item())
            if t % print_every == 0:
                print(f"Iteration {t}, Loss {loss.item()}")
            if np.linalg.norm(lr * grad) < 1e-3:
                print("Converged")
                break
        return params, losses

    def optimize(self, max_iter=50, lr=None, nsample=2**10, seed=0):
        optimizer = optax.lbfgs(learning_rate=lr)
        params = self.init_params()
        opt_state = optimizer.init(params)

        # @jax.jit
        def step(params, opt_state, x):
            # loss, grad = jax.value_and_grad(self.reverse_kld)(params, x)
            loss = self.reverse_kl(params, x)
            grad = self.reverse_kl_grad(params, x)
            updates, opt_state = optimizer.update(grad, opt_state, params, value=loss, grad=grad, value_fn=lambda params: self.reverse_kl(params, x))
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        soboleng = qmc.Sobol(self.d, scramble=True, seed=seed)
        X = ndtri(soboleng.random(nsample) * (1 - MACHINE_EPSILON) + .5 * MACHINE_EPSILON)
        losses = []
        pbar = trange(max_iter)
        for t in pbar:
            params, opt_state, loss = step(params, opt_state, X)
            losses.append(loss.item())
            pbar.set_description(f"Loss: {loss.item()}")
        self.set_params(params)
        log_q = -0.5 * jnp.sum(X**2, axis=-1) - 0.5 * self.d * jnp.log(2 * jnp.pi)
        Z_nf, log_det = self.forward_and_logdet(X)
        log_p = self.target.log_prob(Z_nf)
        weights = np.exp(log_p - log_q + log_det)
        return Z_nf, weights, losses
    
    def _optimize_eval(self, max_iter=50, lr=None, rqmc=True, nsample=2**10, seed=0):
        optimizer = optax.lbfgs(learning_rate=lr)
        params = self.pack_params()
        opt_state = optimizer.init(params)

        @jax.jit
        def step(params, opt_state, x):
            loss, grad = jax.value_and_grad(self.reverse_kld)(params, x)
            updates, opt_state = optimizer.update(grad, opt_state, params, value=loss, grad=grad, value_fn=lambda params: self.reverse_kld(params, x))
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        if rqmc:
            soboleng = qmc.Sobol(self.d, scramble=True, seed=seed)
            X = ndtri(soboleng.random(nsample) * (1 - MACHINE_EPSILON) + .5 * MACHINE_EPSILON)
        else:
            rng = np.random.default_rng(seed)
            X = rng.standard_normal((nsample, self.d))
        losses = []
        evals = []
        pbar = trange(max_iter)
        for t in pbar:
            params, opt_state, loss = step(params, opt_state, X)
            losses.append(loss.item())
            pbar.set_description(f"Loss: {loss.item()}")

            self.set_params(params)
            log_q = -0.5 * jnp.sum(X**2, axis=-1) - 0.5 * self.d * jnp.log(2 * jnp.pi)
            Z_nf, log_det = self.forward_and_logdet(X)
            log_p = self.target.log_prob(Z_nf)
            weights = np.exp(log_p - log_q + log_det)
            mean_est = np.mean(Z_nf * weights[:, None], 0) / np.mean(weights)
            var_est = np.mean(Z_nf**2 * weights[:, None], 0) / np.mean(weights)
            err_1 = np.mean((mean_est - self.true_mean)**2).item()
            err_2 = np.mean((var_est - self.true_var)**2).item()
            evals.append((err_1, err_2))
        return Z_nf, weights, losses, evals
    

    def sample(self, n, params, constrained=True, seed=0, rqmc=True, return_weights=False):
        """
        seed: either integer seed or a numpy random generator
        """
        if not rqmc:
            rng = np.random.default_rng(seed)
            X = rng.standard_normal((n, self.d))
        else:
            soboleng = qmc.Sobol(self.d, scramble=True, seed=seed)
            X = ndtri(soboleng.random(n) * (1 - MACHINE_EPSILON) + .5 * MACHINE_EPSILON)
        Z_nf, log_det = self.forward_and_logdet(params, X)
        if not return_weights:
            if constrained:
                return self.target.param_constrain(np.array(Z_nf, float))
        log_q = -0.5 * jnp.sum(X**2, axis=-1) - 0.5 * self.d * jnp.log(2 * jnp.pi)
        proposal_log_densities = log_q - log_det
        target_log_densities = self.target.log_prob(np.array(Z_nf, float))
        log_weights = target_log_densities - proposal_log_densities
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        if constrained:
            Z_nf = self.target.param_constrain(np.array(Z_nf, float))
        return Z_nf, weights



