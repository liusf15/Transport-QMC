"""
Ill conditioned Gaussian

Banana

Neal's funnel

Bayesian regression

BIP

"""
import numpy as np
import scipy
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from jax.scipy.stats import norm, cauchy
import json
import bridgestan as bs

class IllGaussian:
    def __init__(self, dim):
        self.dim = dim
        self.cov = jnp.diag(jnp.logspace(0, -6, dim))

    def log_prob(self, x):
        return stats.multivariate_normal.logpdf(x, mean=jnp.zeros(self.dim), cov=self.cov)

class Gaussian:
    def __init__(self, mean, cov) -> None:
        self.d = mean.shape[0]
        self.mean = mean
        self.cov = cov
        self.prec = np.linalg.inv(self.cov)
        self.const = -0.5 * self.d * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(self.cov))
    
    def log_prob(self, x):
        tmp = (x - self.mean) @ self.prec
        log_p = -0.5 * (tmp * (x - self.mean)).sum(axis=-1) + self.const
        return log_p

class GaussianMixture:
    def __init__(self, means, covs, weights):
        self.d = len(means[0])
        self.means = means
        self.covs = covs
        self.weights = weights
        self.n_components = len(weights)

    def log_prob(self, x):
        log_probs = jnp.array([stats.multivariate_normal.logpdf(x, mean=mean, cov=cov) for mean, cov in zip(self.means, self.covs)])
        return jnp.log(jnp.sum(jnp.exp(log_probs + jnp.log(self.weights[:, None])), axis=0))

class LogNormal:
    def __init__(self, mean, cov):
        self.d = mean.shape[0]
        self.mean = mean
        self.cov = cov

    def log_prob(self, x):
        return stats.multivariate_normal.logpdf(jnp.log(x), mean=self.mean, cov=self.cov) - jnp.log(x).sum(axis=-1)

class Banana:
    def __init__(self, d):
        self.d = d
        self.mean = np.zeros(d)
        cov_ = np.array([[1., 0.], [0., 1+ 2.]])
        self.cov = scipy.linalg.block_diag(cov_, np.eye(d - 2))

    def log_prob(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        term1 = -0.5 * x1**2 - 0.5 * jnp.log(2 * jnp.pi)
        term2 = -0.5 * (x2 - (x1**2 - 1))**2 - 0.5 * jnp.log(2 * jnp.pi)
        term3 = -0.5 * jnp.sum(x[:, 2:]**2, axis=1) - 0.5 * (self.d - 2) * jnp.log(2 * jnp.pi)
        return term1 + term2 + term3

class BananaNormal:
    def __init__(self, d):
        self.d = d
        self.mean = np.zeros(d)
        cov_ = np.array([[1., 0.], [0., 2+ .5]])
        self.cov = scipy.linalg.block_diag(cov_, np.eye(d - 2))

    def log_prob(self, x):
        x1, x2 = x[0], x[1]
        term1 = -0.5 * x1**2 - 0.5 * jnp.log(2 * jnp.pi)
        term2 = -0.5 * 2 * (x2 - (x1**2 - 0.5 - 0.5))**2 - 0.5 * jnp.log(2 * jnp.pi * .5)
        term3 = -0.5 * jnp.sum(x[2:]**2) - 0.5 * (self.d - 2) * jnp.log(2 * jnp.pi)
        return term1 + term2 + term3
    
    def density(self, x):
        return jnp.exp(self.log_prob(x))
        
class NealsFunnel:
    def __init__(self, dim):
        self.dim = dim
        self.mean = np.zeros(dim)
        self.cov = np.diag(np.concatenate([[3.], np.exp(6 * np.ones(dim - 1))]))

    def log_prob(self, x):
        v = x[:, 0]
        z = x[:, 1:]
        term1 = -0.5 * v**2 / 3 - 0.5 * jnp.log(3 * 2 * jnp.pi)
        term2 = -0.5 * jnp.sum(z**2 * jnp.exp(-2 * v[:, None]), axis=1) - 0.5 * (self.dim - 1) * jnp.log(2 * jnp.pi)
        return term1 + term2

class BayesianLogisticRegression:
    def __init__(self, X, y, prior_scale=1.0):
        self.d = X.shape[1]
        self.X = X
        self.y = y # 1 or 0
        self.prior_scale = prior_scale

    def log_prob(self, w):
        logits = jnp.dot(self.X, w)
        log_prior = -0.5 * jnp.sum((w / self.prior_scale)**2) - 0.5 * self.d * jnp.log(2 * jnp.pi)
        log_lik = jnp.sum(logits * self.y - jnp.logaddexp(0, logits))
        return log_prior + log_lik

class arK:
    def __init__(self, datapath):
        with open(datapath, 'r') as f:
            data = json.load(f)
        self.T = data['T']
        self.K = data['K']
        self.y = jnp.array(data['y'])
        self.d = self.K + 2

    def param_constrain(self, param_unc):
        return param_unc[0], param_unc[1:-1], jnp.exp(param_unc[-1])

    def log_prob(self, param_unc):
        alpha, beta, sigma = self.param_constrain(param_unc)

        # Log-prior for alpha
        log_prior_alpha = norm.logpdf(alpha, 0, 10)
        
        # Log-prior for beta
        log_prior_beta = jnp.sum(norm.logpdf(beta, 0, 10))
        
        # Log-prior for sigma
        log_prior_sigma = cauchy.logpdf(sigma, 0, 2.5)
        
        # Log-likelihood
        log_likelihood = 0
        for t in range(self.K, self.T):
            mu = alpha + jnp.sum(beta * self.y[t-self.K:t])
            log_likelihood += norm.logpdf(self.y[t], mu, sigma)
        
        # Total log-posterior
        log_posterior_value = log_prior_alpha + log_prior_beta + log_prior_sigma + log_likelihood + jnp.log(sigma)
        
        return log_posterior_value

class Posterior:
    def __init__(self, seed, model_path, data_path):
        self.bsmodel = bs.StanModel(
            model_lib=model_path,
            data=data_path,
            seed=seed,
            make_args=["STAN_THREADS=True", "TBB_CXX_TYPE=gcc"],
        )
        self.dimensions = self.bsmodel.param_unc_num()

    def log_density(self, x):
        try:
            log_density = self.bsmodel.log_density(x)
            if jnp.abs(log_density) > 1e15:
                return jnp.nan
            return log_density
        except:
            return jnp.nan

    def log_density_gradient(self, x):
        try:
            log_density, gradient = self.bsmodel.log_density_gradient(x)
            if jnp.abs(log_density) > 1e15 or jnp.isnan(gradient).any():
                return jnp.nan, jnp.zeros(x.shape)
            return log_density, gradient
        except:
            return jnp.nan, jnp.zeros(x.shape) 

    def param_unconstrain(self, x):
        return self.bsmodel.param_unconstrain(x)

    def param_constrain(self, x):
        return self.bsmodel.param_constrain(x)

    @property
    def dims(self):
        return self.dimensions
    
    def param_unc_num(self):
        return self.bsmodel.param_unc_num()
    
def make_logdensity_fn(bs_model):
    """Register a Stan model with JAX's custom VJP system via Bridgestan.

    See https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html.
    """

    @jax.custom_vjp
    def log_density(arg):
        # Cast to float64 to match Stan's dtype
        fn = lambda x: bs_model.log_density(np.array(x, dtype=np.float64))
        # Cast back to float32 to match JAX's default dtype
        result_shape = jax.ShapeDtypeStruct((), jnp.float32)
        return jax.pure_callback(fn, result_shape, arg)

    def call_grad(arg):
        fn = lambda x: bs_model.log_density_gradient(np.array(x, dtype=np.float64))[
            1
        ]
        result_shape = jax.ShapeDtypeStruct(arg.shape, arg.dtype)
        return jax.pure_callback(fn, result_shape, arg)

    def vjp_fwd(arg):
        return log_density(arg), arg

    def vjp_bwd(residuals, y_bar):
        arg = residuals
        return (call_grad(arg) * y_bar,)

    log_density.defvjp(vjp_fwd, vjp_bwd)

    return log_density

class StanModel:
    def __init__(self, stan_path, data_path):
        self.stan_model = Posterior(0, stan_path, data_path)
        self.d = self.stan_model.param_unc_num()
        self.log_prob_jax = make_logdensity_fn(self.stan_model)
    
    def log_prob(self, x):
        return self.log_prob_jax(x)
    
    def param_constrain(self, x):
        if x.ndim == 1:
            return self.stan_model.param_constrain(x)
        return np.array([self.stan_model.param_constrain(x[i]) for i in range(x.shape[0])])