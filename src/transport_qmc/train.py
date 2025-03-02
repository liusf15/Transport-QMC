import numpy as np
import jax
import jax_tqdm
import jax.numpy as jnp
import jax.scipy.optimize
import optax
from optax.tree_utils import tree_add_scalar_mul, tree_vdot
from typing import NamedTuple

class LineSearchState(NamedTuple):
    alpha: float
    new_loss: float
    params: jnp.ndarray
    loss: float
    updates: jnp.ndarray
    v_g_prod: float
    lbd: float


def lbfgs(loss_fn, params0, validation_fn, max_iter=50, max_backtracking=20, slope_rtol=1e-4, memory_size=20, max_lr=1.):
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
    
    opt = optax.scale_by_lbfgs(memory_size=memory_size)
    opt_state = opt.init(params)

    @jax_tqdm.scan_tqdm(max_iter)
    def lbfgs_step(carry, t):
        params, opt_state = carry
        loss, grad = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grad, opt_state, params)

        alpha = max_lr
        new_params = tree_add_scalar_mul(params, -alpha, updates)
        new_loss = loss_fn(new_params)
        init_state = LineSearchState(alpha, new_loss, params, loss, updates, tree_vdot(updates, grad), 0.)
        final_state = jax.lax.while_loop(LS_cond, LS_step, init_state)
        params = tree_add_scalar_mul(params, -final_state.alpha, updates)

        val_loss = validation_fn(params)
        return (params, opt_state), val_loss
        
    carry = (params, opt_state)
    final_state, val_losses = jax.lax.scan(lbfgs_step, carry, jnp.arange(max_iter))
    return final_state, val_losses



def sgd(loss_fn, params0, validation_fn, max_iter=50, lr=1e-3):
    opt = optax.adam(lr)
    opt_state = opt.init(params0)
    params = params0
    
    @jax_tqdm.scan_tqdm(max_iter)
    def sgd_step(carry, t):
        params, opt_state = carry

        grad = jax.grad(loss_fn)(params)
        updates, opt_state = opt.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        val_loss = validation_fn(params)
        return (params, opt_state), val_loss

    carry = (params, opt_state)
    final_state, losses = jax.lax.scan(sgd_step, carry, np.arange(max_iter))
    return final_state, losses

# def lbfgs(loss_fn, params0, max_iter=50, max_backtracking=20, slope_rtol=1e-4, memory_size=20, max_lr=1., callback=None):
#     min_lr = max_lr / (1 << max_backtracking)
#     @jax.jit
#     def LS_cond(state: LineSearchState):
#         return jnp.logical_and(state.alpha > min_lr, 
#                         jnp.isnan(state.new_loss) | (state.new_loss > state.loss - slope_rtol * state.alpha * state.v_g_prod))

#     @jax.jit
#     def LS_step(state: LineSearchState):
#         new_alpha = state.alpha * 0.5
#         new_params = tree_add_scalar_mul(state.params, -new_alpha, state.updates)
#         new_loss = loss_fn(new_params)
#         state = state._replace(alpha=new_alpha, new_loss=new_loss)
#         return state

#     params = params0
#     best_params = params0
#     best_ess = 0.
    
#     opt = optax.scale_by_lbfgs(memory_size=memory_size)
#     opt_state = opt.init(params)

#     def lbfgs_step(carry, t):
#         params, opt_state, best_params, best_ess = carry
#         # jax.debug.print("{}", params)
#         loss, grad = jax.value_and_grad(loss_fn)(params)
#         updates, opt_state = opt.update(grad, opt_state, params)

#         alpha = max_lr
#         new_params = tree_add_scalar_mul(params, -alpha, updates)
#         new_loss = loss_fn(new_params)
#         init_state = LineSearchState(alpha, new_loss, params, loss, updates, tree_vdot(updates, grad), 0.)
#         final_state = jax.lax.while_loop(LS_cond, LS_step, init_state)
#         params = tree_add_scalar_mul(params, -final_state.alpha, updates)
        
#         if callback is not None:
#             metrics = callback(params)
#             new_best_params, new_best_ess = jax.lax.cond(
#                 jnp.isnan(metrics.ess) | (metrics.ess < best_ess),
#                 lambda: (best_params, best_ess),  
#                 lambda: (params, metrics.ess)
#                 )
#             return (params, opt_state, new_best_params, new_best_ess), metrics
#         else:
#             metrics = None
#             return (params, opt_state, None, None), loss
        
#     if callback is not None:
#         carry = (params, opt_state, best_params, best_ess)
#     else:
#         carry = (params, opt_state, None, None)
#     final_state, losses = jax.lax.scan(lbfgs_step, carry, np.arange(max_iter))
#     return final_state, losses
