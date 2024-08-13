import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.optimize import minimize
from scipy.optimize import fmin_l_bfgs_b
from tqdm import trange

import jax.numpy as jnp
from jax.scipy.special import betainc, ndtri, ndtr, logit
from jax.scipy.stats.beta import pdf as beta_pdf
from jax.nn import softmax, sigmoid
from jax.scipy.linalg import det
import jax
import optax

from qmc_flow.utils import sample_gaussian 
MACHINE_EPSILON = np.finfo(np.float64).eps

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
        # TODO: fix the last one to 0
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
        z, log_det = self.forward_and_logdet(params, x) # use vmap, faster?
        # log_p = jnp.array([self.target.log_prob(np.array(z[i], dtype=float)) for i in range(z.shape[0])])
        log_p = jax.vmap(self.target.log_prob)(z)
        return jnp.mean(log_q - log_det - log_p)
    
    # def _log_p_grad(self, params, x):
    #     """
    #     gradient of log p(T(x; params)) w.r.t. params
    #     """
    #     z = self.forward(params, x) # TODO: use vectorized version
    #     T_grad = jax.jacfwd(self.forward)(params, x)
    #     # target_log_p = jnp.array([self.target.log_prob_grad(np.array(z[i], dtype=float)) for i in range(z.shape[0])])
    #     log_p_grad = jnp.array(self.target.log_prob_grad(np.array(z, dtype=float)))
    #     return jnp.einsum('ijk,ij->k', T_grad, log_p_grad) / z.shape[0]

    # def reverse_kl_grad(self, params, x):
    #     def logdet_mean(params, x):
    #         log_det = self.forward_and_logdet(params, x)[1]
    #         return jnp.mean(log_det)
    #     log_det_grad = jax.jacfwd(logdet_mean)(params, x)
        
    #     # log_p_grad = np.mean([self._log_p_grad(params, x[i]) for i in range(x.shape[0])], axis=0) #!! super slow
    #     log_p_grad = self._log_p_grad(params, x)
    #     return -log_det_grad - log_p_grad
    
    def gradient_descent(self, max_iter=50, lr=1e-2, nsample=2**10, seed=0, print_every=100):
        params = self.init_params()
        losses = []

        soboleng = qmc.Sobol(self.d, scramble=True, seed=seed)
        X = ndtri(soboleng.random(nsample) * (1 - MACHINE_EPSILON) + .5 * MACHINE_EPSILON)
        for t in range(max_iter):
            loss, grad = jax.value_and_grad(self.reverse_kl)(params, X)
            params = params - lr * grad
            losses.append(loss.item())
            if t % print_every == 0:
                print(f"Iteration {t}, Loss {loss.item()}")
            if np.linalg.norm(lr * grad) < 1e-3:
                print("Converged")
                break
        return params, losses

    def optimize_lbfgs_scipy(self, max_iter=200, nsample=2**10, sampler='mc', seed=0):
        X = sample_gaussian(nsample, self.d, seed=seed, sampler=sampler)
        log_q = -0.5 * jnp.sum(X**2, axis=-1) - 0.5 * self.d * jnp.log(2 * jnp.pi)

        kl_logs = []
        moments_logs = []
        ess_logs = []
        state = {'counter': 0}
        print_every = 1
        def callback(params):
            state['counter'] += 1
            
            Z_nf, log_det = self.forward_and_logdet(params, X)      
            proposal_log_densities = - log_det
            
            target_log_densities = jax.vmap(self.target.log_prob)(Z_nf)
            log_weights = target_log_densities - proposal_log_densities
            log_weights -= np.max(log_weights)
            weights = np.exp(log_weights)

            loss = jnp.mean(log_q - log_det - target_log_densities)
            kl_logs.append(loss.item()) 

            nf_samples = self.target.param_constrain(np.array(Z_nf, float))
            moments = get_moments(nf_samples, weights)
            moments_logs.append(moments)
            ess = get_effective_sample_size(weights)
            ess_logs.append(ess.item())
            if state['counter'] % print_every == 0:
                print(f"Iteration: {state['counter']}, Loss: {loss}, ESS: {ess}")
            if ess >= 0.999 * nsample:
                print(f"ESS is {ess}, stopping optimization")
                raise StopIteration
        try:
            params = self.init_params()
            # res = minimize(self.reverse_kl, jac=jax.grad(self.reverse_kl), x0=params, args=X, method='L-BFGS-B', options={'maxiter': max_iter}, callback=callback)  
            res = fmin_l_bfgs_b(func=jax.value_and_grad(self.reverse_kl), x0=params, args=(X,), maxiter=max_iter, callback=callback, m=20)
            # if res.success:
            #     print("Optimization successful")
        except StopIteration:
            print("Optimization was terminated by the callback function")

        return res[0], {'kl': kl_logs, 'moments': moments_logs, 'ESS': ess_logs}

    def optimize(self, max_iter=50, max_backtracking=20, slope_rtol=1e-4, memory_size=10, max_lr=1., nsample=2**10, seed=0, sampler='rqmc'):
        X = sample_gaussian(nsample, self.d, seed=seed, sampler=sampler)
        kl_logs = []
        moments_logs = []
        ess_logs = []
        params = self.init_params()

        # first run Adam for a few iterations
        # def step(params, opt_state, x):
        #     loss, grad = jax.value_and_grad(self.reverse_kl)(params, x)
        #     updates, opt_state = opt_lbfgs.update(grad, opt_state, params)
        #     params = params + updates
        #     return params, opt_state, loss

        # pbar = trange(memory_size, desc="Warm-up")
        # for t in pbar:
        #     loss, grad = jax.value_and_grad(self.reverse_kl)(params, X)
        #     params = params - lr * grad
        #     kl_logs.append(loss.item())
        #     pbar.set_description(f"Warm-up Loss: {loss.item()}")

            
        
        # now run LBFGS
        # linesearch = optax.scale_by_backtracking_linesearch(
        #     max_backtracking_steps=10, 
        #     decrease_factor=0.5,
        #     slope_rtol=1e-5,
        #     increase_factor=jnp.inf,
        #     max_learning_rate=0.01, 
        #     store_grad=True)
        # doesn't work; 
        # loss, grad = optax.value_and_grad_from_state(self.reverse_kl)(params, state=opt_state, x=X)
        # updates, opt_state = opt.update(grad, opt_state, params, value=loss, grad=grad, value_fn=self.reverse_kl, x=X)    
        # print('learning rate', opt_state[1].learning_rate)
        # params = params + updates
        
        opt_lbfgs = optax.scale_by_lbfgs(memory_size=memory_size)
        # opt = optax.chain(opt_lbfgs, linesearch)
        opt = opt_lbfgs
        opt_state = opt.init(params)
        pbar = trange(max_iter)
        for t in pbar:
            loss, grad = jax.value_and_grad(self.reverse_kl)(params, X)
            updates, opt_state = opt_lbfgs.update(grad, opt_state, params)
            alpha = max_lr
            for s in range(max_backtracking):
                new_params = params - alpha * updates
                try:
                    new_loss = self.reverse_kl(new_params, X)
                    if jnp.isnan(new_loss):
                        raise ValueError
                except:
                    alpha *= 0.5
                    print(s, "Error, decrease alpha to", alpha)
                    continue
                if new_loss < loss - slope_rtol * alpha * jnp.dot(updates, grad):
                    print("Decrease condition satisfied")
                    break
                alpha *= 0.5
            params = params - alpha * updates

            kl_logs.append(loss.item())
            pbar.set_description(f"Loss: {loss.item()}")
            # evaluation
            Z_nf, log_det = self.forward_and_logdet(params, X)      
            proposal_log_densities = - log_det
            target_log_densities = jax.vmap(self.target.log_prob)(Z_nf)
            log_weights = target_log_densities - proposal_log_densities
            log_weights -= np.max(log_weights)
            weights = np.exp(log_weights)
            nf_samples = self.target.param_constrain(np.array(Z_nf, float))
            moments_logs.append(get_moments(nf_samples, weights))
            ess_logs.append(get_effective_sample_size(weights).item())

            if jnp.linalg.norm(updates).item() < 1e-3:
                print("Converged")

        return params, {'kl': kl_logs, 'moments': moments_logs, 'ESS': ess_logs}
    
    def _optimize_eval(self, max_iter=50, lr=None, rqmc=True, nsample=2**10, seed=0, algorithm='adam'):
        if algorithm == 'lbfgs':
            linesearch = optax.scale_by_backtracking_linesearch(max_backtracking_steps=15, max_learning_rate=1., store_grad=True)
            # linesearch = optax.scale_by_zoom_linesearch(max_linesearch_steps=15, verbose=True)
            optimizer = optax.lbfgs(linesearch=linesearch)
        elif algorithm == 'adam':
            optimizer = optax.adam(learning_rate=lr)

        params = self.init_params()
        opt_state = optimizer.init(params)

        @jax.jit
        def step(params, opt_state, x):
            loss, grad = jax.value_and_grad(self.reverse_kl)(params, x)
            updates, opt_state = optimizer.update(grad, opt_state, params, value=loss, grad=grad, value_fn=self.reverse_kl, x=x)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        if rqmc:
            soboleng = qmc.Sobol(self.d, scramble=True, seed=seed)
            X = ndtri(soboleng.random(nsample) * (1 - MACHINE_EPSILON) + .5 * MACHINE_EPSILON)
        else:
            rng = np.random.default_rng(seed)
            X = rng.standard_normal((nsample, self.d))
        kl_logs = []
        moments_logs = []
        ess_logs = []
        pbar = trange(max_iter)
        for t in pbar:
            params, opt_state, loss = step(params, opt_state, X)
            kl_logs.append(loss.item())

            Z_nf, log_det = self.forward_and_logdet(params, X)      
            proposal_log_densities = - log_det
            target_log_densities = jax.vmap(self.target.log_prob)(Z_nf)
            log_weights = target_log_densities - proposal_log_densities
            log_weights -= np.max(log_weights)
            weights = np.exp(log_weights)
            nf_samples = self.target.param_constrain(np.array(Z_nf, float))
            moments = get_moments(nf_samples, weights)
            moments_logs.append(moments)
            ess = get_effective_sample_size(weights)
            ess_logs.append(ess.item())
        return Z_nf, weights, {'kl_logs': kl_logs, 'moments_logs': moments_logs, 'ess_logs': ess_logs}
    

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
        target_log_densities = self.target.log_prob(Z_nf)
        log_weights = target_log_densities - proposal_log_densities
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        if constrained:
            Z_nf = self.target.param_constrain(np.array(Z_nf, float))
        return Z_nf, weights



