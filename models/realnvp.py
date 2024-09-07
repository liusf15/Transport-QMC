import numpy as np
import jax.numpy as jnp
import jax

from qmc_flow.utils import sample_gaussian
MACHINE_EPSILON = np.finfo(np.float64).eps

def leaky_relu(x, alpha=0.01):
    return jnp.where(x > 0, x, alpha * x)

class MLP:
    def __init__(self, params, activation='leakyRelu') -> None:
        self.params = params
        if activation == 'leakyRelu':
            self.act = leaky_relu
        else:
            raise ValueError(f"Unknown activation function: {activation}")
    
    def forward(self, x):
        for W, b in self.params[:-1]:
            x = self.act(x @ W + b)
        W, b = self.params[-1]
        return x @ W + b

class AffineCouplingBlock:
    def __init__(self, d, hidden_dim, num_layers, mask) -> None:
        self.d = d
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.mask = mask
    
    def init_params(self, key=jax.random.PRNGKey(0)):
        params = {'t': [], 's': []}
        for l in range(self.num_layers):
            if l == 0:
                input_dim = self.d
            else:
                input_dim = self.hidden_dim
            
            if l == self.num_layers - 1:
                output_dim = self.d
            else:
                output_dim = self.hidden_dim

            key, subkey = jax.random.split(key, 2)
            W, b = jax.random.normal(jax.random.PRNGKey(subkey), (2, input_dim, output_dim)), jnp.zeros(self.hidden_dim)
            params['t'].append((W[0], b))
            params['s'].append((W[1], b))
        return params

    def forward(self, params, x):
        x_msk = x * self.mask # unchanged part
        x_ = x * (1 - self.mask)
        shift = MLP(params['t'], activation='leakyRelu').forward(x_)
        scale = MLP(params['s'], activation='leakyRelu').forward(x_)

        y = x_msk + (1 - self.mask) * (x_ * jnp.exp(scale) + shift)
        return y

    def forward_and_logdet(self, params, x):
        x_ = x * (1 - self.mask)
        shift = MLP(params['t'], activation='leakyRelu').forward(x_)
        scale = MLP(params['s'], activation='leakyRelu').forward(x_)

        y = x * self.mask + (1 - self.mask) * (x_ * jnp.exp(scale) + shift)
        log_det = jnp.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return y, log_det

class RealNVP:
    def __init__(self, d, target, num_blocks, hidden_dim, num_layers, key=jax.random.key(0)) -> None:
        self.d = d
        self.target = target
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.masks = self.init_masks(key)

    def init_block(self, key, init_zero=True):
        params = {'t': [], 's': []}
        for l in range(self.num_layers):
            if l == 0:
                input_dim = self.d
            else:
                input_dim = self.hidden_dim
            
            if l == self.num_layers - 1:
                output_dim = self.d
            else:
                output_dim = self.hidden_dim

            key, subkey = jax.random.split(key, 2)
            if init_zero:
                W = jnp.zeros((2, input_dim, output_dim))
            else:
                W = jax.random.normal(subkey, (2, input_dim, output_dim))
            b = jnp.zeros(output_dim)
            params['t'].append((W[0], b))
            params['s'].append((W[1], b))
        return params
    
    def init_masks(self, key=jax.random.key(0)):    
        masks = []
        for b in range(self.num_blocks):
            key, subkey = jax.random.split(key)
            msk = jnp.ones(self.d, dtype=bool)
            idx = jax.random.choice(subkey, self.d, (self.d // 2,), replace=False)
            msk = msk.at[idx].set(False)
            masks.append(msk)
        return masks
    
    def init_params(self, key=jax.random.key(0), init_zero=True):
        params = []
        for b in range(self.num_blocks):
            key, subkey = jax.random.split(key)
            params.append(self.init_block(subkey, init_zero=init_zero))
        return params
    
    def affine_coupling(self, mask, params, x):
        x_msk = x * mask # unchanged part
        x_ = x * (1 - mask)
        shift = MLP(params['t'], activation='leakyRelu').forward(x_)
        scale = MLP(params['s'], activation='leakyRelu').forward(x_)

        y = x_msk + (1 - mask) * (x_ * jnp.exp(scale) + shift)
        return y, jnp.sum((1 - mask) * scale, axis=-1)

    def forward_and_logdet(self, params, x):
        log_det = 0
        for b in range(self.num_blocks):
            x, log_det_b = self.affine_coupling(self.masks[b], params[b], x)
            log_det += log_det_b
        return x, log_det

    def reverse_kl(self, params, x):
        """
        x: (num_samples, d)
        """
        # log_q = -0.5 * jnp.sum(x**2, axis=-1) - 0.5 * self.d * jnp.log(2 * jnp.pi)
        log_q = 0.
        z, log_det = self.forward_and_logdet(params, x)
        log_p = jax.vmap(self.target.log_prob)(z)
        return jnp.mean(log_q - log_det - log_p)

    def sample(self, params, nsample, seed=0, sampler='rqmc'):
        """
        seed: either integer seed or a numpy random generator
        """
        X = sample_gaussian(nsample, self.d, seed=seed, sampler=sampler)
        Z_nf, log_det = self.forward_and_logdet(params, X)
        log_q = -0.5 * jnp.sum(X**2, axis=-1) - 0.5 * self.d * jnp.log(2 * jnp.pi)
        proposal_log_densities = log_q - log_det
        target_log_densities = jax.vmap(self.target.log_prob)(Z_nf)
        log_weights = target_log_densities - proposal_log_densities
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        if getattr(self.target, 'param_constrain', None):
            Z_nf = self.target.param_constrain(np.array(Z_nf, float))
        return Z_nf, weights
