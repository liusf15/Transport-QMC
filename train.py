import numpy as np
import scipy
from scipy.special import hermite, ndtri
from scipy.stats import qmc
import itertools
import jax_tqdm
import jax
import jax.numpy as jnp
import jax.scipy.optimize
import optax
from optax.tree_utils import tree_add_scalar_mul, tree_vdot, tree_l2_norm
from functools import partial
from typing import NamedTuple
import matplotlib.pyplot as plt
from tqdm import trange
from qmc_flow.utils import sample_gaussian, get_moments, get_effective_sample_size
MACHINE_EPSILON = np.finfo(np.float64).eps

class LineSearchState(NamedTuple):
    alpha: float
    new_loss: float
    params: jnp.ndarray
    loss: float
    updates: jnp.ndarray
    v_g_prod: float

def optimize(model, params0, div_name, max_iter=50, max_backtracking=20, slope_rtol=1e-4, memory_size=10, max_lr=1., nsample=2**10, seed=0, sampler='rqmc'):
    X = sample_gaussian(nsample, model.d, seed=seed, sampler=sampler)
    min_lr = max_lr / (1 << max_backtracking)

    loss_fn = jax.jit(model.divergence, static_argnames=('div_name', ))

    @jax.jit
    def LS_cond(state: LineSearchState):
        return jnp.logical_and(state.alpha > min_lr, 
                        jnp.isnan(state.new_loss) | (state.new_loss > state.loss - slope_rtol * state.alpha * state.v_g_prod))

    @jax.jit
    def LS_step(state: LineSearchState):
        new_alpha = state.alpha * 0.5
        new_params = tree_add_scalar_mul(state.params, -new_alpha, state.updates)
        new_loss = loss_fn(new_params, X, div_name)
        state = state._replace(alpha=new_alpha, new_loss=new_loss)
        return state

    params = params0
    
    opt = optax.scale_by_lbfgs(memory_size=memory_size)
    opt_state = opt.init(params)

    @jax_tqdm.scan_tqdm(max_iter)
    def lbfgs_step(carry, t):
        params, opt_state = carry
        # jax.debug.print("{}", params)
        loss, grad = jax.value_and_grad(loss_fn)(params, X, div_name)
        updates, opt_state = opt.update(grad, opt_state, params)

        alpha = max_lr
        new_params = tree_add_scalar_mul(params, -alpha, updates)
        new_loss = loss_fn(new_params, X, div_name)
        init_state = LineSearchState(alpha, new_loss, params, loss, updates, tree_vdot(updates, grad))
        final_state = jax.lax.while_loop(LS_cond, LS_step, init_state)
        params = tree_add_scalar_mul(params, -final_state.alpha, updates)
        # jax.debug.breakpoint()
        return (params, opt_state), loss_fn(params, X, 'all')

    carry = (params, opt_state)
    final_state, losses = jax.lax.scan(lbfgs_step, carry, np.arange(max_iter))
    return final_state[0], losses
    # pbar = trange(max_iter)
    # loss_fn = lambda params, X: model.divergence(params, X, div_name)
    # loss_fn = jax.jit(loss_fn)
    
    
    # for t in pbar:
        # loss, grad = jax.value_and_grad(loss_fn)(params, X, div_name)
        # div_logs.append(loss)
        # loss, grad = jax.value_and_grad(model.divergence)(params, X, div_name)
        # updates, opt_state = opt.update(grad, opt_state, params)
        # alpha = max_lr
        
        # new_params = tree_add_scalar_mul(params, -alpha, updates)
        # new_loss = loss_fn(new_params, X, div_name)
        # init_state = LineSearchState(alpha, new_loss, params, loss, updates, tree_vdot(updates, grad))
        # final_state = jax.lax.while_loop(LS_cond, LS_step, init_state)
        # params = tree_add_scalar_mul(params, -final_state.alpha, updates)
        # params, opt_state = lbfgs_step(params, opt_state)
        
        # def body_fun(alpha):
        #     return alpha * 0.5

        # jax.lax.while_loop(cond_fun, body_fun, alpha)
        # for s in range(max_backtracking):
        #     new_params = tree_add_scalar_mul(params, -alpha, updates)

        #     new_loss = fn(new_params, X)
        #     if jnp.isnan(new_loss):
        #         alpha *= .5
        #     elif new_loss > loss - slope_rtol * alpha * tree_vdot(updates, grad):
        #         alpha *= 0.5
        #     else:
        #         break
            
        # params = tree_add_scalar_mul(params, -alpha, updates) 


        # div_logs.append(fn(params, X, 'all'))
        # evaluation
        # Z_nf, log_det = model.forward_and_logdet(params, X)      
        # log_q = -0.5 * jnp.sum(X**2, axis=-1) - 0.5 * model.d * jnp.log(2 * jnp.pi)
        # proposal_log_densities = log_q - log_det
        # target_log_densities = jax.vmap(model.target.log_prob)(Z_nf)
        # log_weights = target_log_densities - proposal_log_densities
        # log_weights -= np.max(log_weights)
        # weights = np.exp(log_weights)
        # try:
        #     nf_samples = model.target.param_constrain(np.array(Z_nf, float))
        # except:
        #     nf_samples = np.array(Z_nf, float)
        # moments_logs.append(get_moments(nf_samples, weights))
        # ess_logs.append(get_effective_sample_size(weights).item())
        # pbar.set_description(f"Loss: {loss.item()}, ESS: {ess_logs[-1]}")
        # pbar.set_description(f"Loss: {loss.item()}")
        # if tree_l2_norm(updates).item() < 1e-3:
        #     print("Converged")

    # return params, {'div_logs': div_logs, 'moments': moments_logs, 'ESS': ess_logs}

def optimize_variance(model, params0, offset_logweight, max_iter=50, max_backtracking=20, slope_rtol=1e-4, memory_size=10, max_lr=1., nsample=2**10, seed=0, sampler='rqmc'):
    X = sample_gaussian(nsample, model.d, seed=seed, sampler=sampler)
    kl_logs = []
    moments_logs = []
    ess_logs = []
    params = params0
    
    opt = optax.scale_by_lbfgs(memory_size=memory_size)
    opt_state = opt.init(params)
    pbar = trange(max_iter)
    for t in pbar:
        loss, grad = jax.value_and_grad(model.forward_kl)(params, X, offset_logweight)
        updates, opt_state = opt.update(grad, opt_state, params)
        alpha = max_lr
        for s in range(max_backtracking):
            new_params = tree_add_scalar_mul(params, -alpha, updates)
            try:
                new_loss = model.forward_kl(new_params, X, offset_logweight)
                if jnp.isnan(new_loss):
                    raise ValueError
            except:
                alpha *= 0.5
                print(s, "Error, decrease alpha to", alpha)
                continue
            if new_loss < loss - slope_rtol * alpha * tree_vdot(updates, grad):
                print("Decrease condition satisfied at", alpha)
                break
            alpha *= 0.5
        # params = params - alpha * updates
        params = tree_add_scalar_mul(params, -alpha, updates) 

        kl_logs.append(model.reverse_kl(params, X).item())
        
        # evaluation
        Z_nf, log_det = model.forward_and_logdet(params, X)      
        proposal_log_densities = - log_det
        target_log_densities = jax.vmap(model.target.log_prob)(Z_nf)
        log_weights = target_log_densities - proposal_log_densities
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        # nf_samples = model.target.param_constrain(np.array(Z_nf, float))
        nf_samples = np.array(Z_nf, float)
        moments_logs.append(get_moments(nf_samples, weights))
        ess_logs.append(get_effective_sample_size(weights).item())
        pbar.set_description(f"Loss: {loss.item()}, ESS: {ess_logs[-1]}, KL: {kl_logs[-1]}")
        if tree_l2_norm(updates).item() < 1e-3:
            print("Converged")

    return params, {'kl': kl_logs, 'moments': moments_logs, 'ESS': ess_logs}
