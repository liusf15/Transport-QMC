import numpy as np
from scipy.stats import multivariate_t, mvn
from scipy.special import stdtrit, ndtri
from importlib import reload
import qmc_flow.utils; reload(qmc_flow.utils)
from qmc_flow.utils import sample_t

samples, log_q = sample_t(1024, 2, df=3, seed=1, sampler='rqmc', return_logp=True)

stdtrit(np.Inf, 0.6)
ndtri(0.6)

mvn()

np.random.standard_t(df=2, size=100)
d = 2
mean = np.zeros(d)
cov = np.eye(d) 
t_dist = multivariate_t(loc=mean, shape=cov, df=5, seed=1)
samples = t_dist.rvs(size=1000)
np.mean(samples, axis=0), np.cov(samples.T)
