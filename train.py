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
from qmc_flow.utils import sample_gaussian, get_moments, get_effective_sample_size, sample_t
MACHINE_EPSILON = np.finfo(np.float64).eps

class LineSearchState(NamedTuple):
    alpha: float
    new_loss: float
    params: jnp.ndarray
    loss: float
    updates: jnp.ndarray
    v_g_prod: float
    lbd: float

def lbfgs(loss_fn, params0, max_iter=50, max_backtracking=20, slope_rtol=1e-4, memory_size=20, max_lr=1., callback=None):
    min_lr = max_lr / (1 << max_backtracking)
    @jax.jit
    def LS_cond(state: LineSearchState):
        return jnp.logical_and(state.alpha > min_lr, 
                        jnp.isnan(state.new_loss) | (state.new_loss > state.loss - slope_rtol * state.alpha * state.v_g_prod))

    @jax.jit
    def LS_step(state: LineSearchState):
        new_alpha = state.alpha * 0.5
        new_params = tree_add_scalar_mul(state.params, -new_alpha, state.updates)
        new_loss = loss_fn(new_params)
        state = state._replace(alpha=new_alpha, new_loss=new_loss)
        return state

    params = params0
    best_params = params0
    best_ess = 0.
    
    opt = optax.scale_by_lbfgs(memory_size=memory_size)
    opt_state = opt.init(params)

    @jax_tqdm.scan_tqdm(max_iter)
    def lbfgs_step(carry, t):
        params, opt_state, best_params, best_ess = carry
        # jax.debug.print("{}", params)
        loss, grad = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grad, opt_state, params)

        alpha = max_lr
        new_params = tree_add_scalar_mul(params, -alpha, updates)
        new_loss = loss_fn(new_params)
        init_state = LineSearchState(alpha, new_loss, params, loss, updates, tree_vdot(updates, grad), 0.)
        final_state = jax.lax.while_loop(LS_cond, LS_step, init_state)
        params = tree_add_scalar_mul(params, -final_state.alpha, updates)
        
        if callback is not None:
            metrics = callback(params)
            new_best_params, new_best_ess = jax.lax.cond(
                jnp.isnan(metrics.ess) | (metrics.ess < best_ess),
                lambda: (best_params, best_ess),  
                lambda: (params, metrics.ess)
                )
            return (params, opt_state, new_best_params, new_best_ess), metrics
        else:
            metrics = None
            return (params, opt_state, None, None), loss
        
    if callback is not None:
        carry = (params, opt_state, best_params, best_ess)
    else:
        carry = (params, opt_state, None, None)
    final_state, losses = jax.lax.scan(lbfgs_step, carry, np.arange(max_iter))
    return final_state, losses

def lbfgs_annealed(loss_fn, params0, max_iter=50, anneal_iter=50, max_lbd=1., max_backtracking=20, slope_rtol=1e-4, memory_size=20, max_lr=1., callback=None):
    min_lr = max_lr / (1 << max_backtracking)
    @jax.jit
    def LS_cond(state: LineSearchState):
        return jnp.logical_and(state.alpha > min_lr, 
                        jnp.isnan(state.new_loss) | (state.new_loss > state.loss - slope_rtol * state.alpha * state.v_g_prod))

    @jax.jit
    def LS_step(state: LineSearchState):
        new_alpha = state.alpha * 0.5
        new_params = tree_add_scalar_mul(state.params, -new_alpha, state.updates)
        new_loss = loss_fn(new_params, state.lbd)
        state = state._replace(alpha=new_alpha, new_loss=new_loss)
        return state

    params = params0
    best_params = params0
    best_ess = 0.
    
    opt = optax.scale_by_lbfgs(memory_size=memory_size)
    opt_state = opt.init(params)

    @jax_tqdm.scan_tqdm(max_iter)
    def lbfgs_step(carry, t):
        params, opt_state, best_params, best_ess = carry
        # jax.debug.print("{}", params)
        lbd = jnp.clip(max_lbd / anneal_iter, 0., max_lbd)
        loss, grad = jax.value_and_grad(loss_fn)(params, lbd)
        updates, opt_state = opt.update(grad, opt_state, params)

        alpha = max_lr
        new_params = tree_add_scalar_mul(params, -alpha, updates)
        new_loss = loss_fn(new_params, lbd)
        init_state = LineSearchState(alpha, new_loss, params, loss, updates, tree_vdot(updates, grad), lbd)
        final_state = jax.lax.while_loop(LS_cond, LS_step, init_state)
        params = tree_add_scalar_mul(params, -final_state.alpha, updates)
        
        if callback is not None:
            metrics = callback(params)
            new_best_params, new_best_ess = jax.lax.cond(
                jnp.isnan(metrics.ess) | (metrics.ess < best_ess),
                lambda: (best_params, best_ess),  
                lambda: (params, metrics.ess)
                )
            return (params, opt_state, new_best_params, new_best_ess), metrics
        else:
            metrics = None
            return (params, opt_state, None, None), loss
        
    if callback is not None:
        carry = (params, opt_state, best_params, best_ess)
    else:
        carry = (params, opt_state, None, None)
    final_state, losses = jax.lax.scan(lbfgs_step, carry, np.arange(max_iter))
    return final_state, losses

def sgd(loss_fn, params0, max_iter=50, lr=1e-3, callback=None):
    opt = optax.adam(lr)
    opt_state = opt.init(params0)
    params = params0
    best_params = params0
    best_ess = 0.
    
    @jax_tqdm.scan_tqdm(max_iter)
    def sgd_step(carry, t):
        params, opt_state, best_params, best_ess = carry

        grad = jax.grad(loss_fn)(params)
        updates, opt_state = opt.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        if callback is not None:
            metrics = callback(params)
            new_best_params, new_best_ess = jax.lax.cond(
                jnp.isnan(metrics.ess) | (metrics.ess < best_ess),
                lambda: (best_params, best_ess),  
                lambda: (params, metrics.ess)  
    )
        else:
            metrics = None
        return (params, opt_state, new_best_params, new_best_ess), metrics

    carry = (params, opt_state, best_params, best_ess)
    final_state, losses = jax.lax.scan(sgd_step, carry, np.arange(max_iter))
    return final_state, losses

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
