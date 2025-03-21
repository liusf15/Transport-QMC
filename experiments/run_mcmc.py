import numpy as np
import jax
from numpyro.infer import NUTS, MCMC
import pickle
import argparse

import transport_qmc.targets as Targets
import numpyro

numpyro.set_host_device_count(20)

print(jax.local_device_count())

def main(posterior_name, num_warmup, num_samples, num_chains):
    data_file = f"stan/{posterior_name}.json"
    target = getattr(Targets, posterior_name)(data_file)

    nuts_kernel = NUTS(target.numpyro_model)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(jax.random.key(0))

    mcmc_samples = mcmc.get_samples()

    param_names = target.param_unc_names()
    samples_unc = None
    for key in param_names:
        sample = mcmc_samples[key]
        if sample.ndim == 1:
            sample = sample.reshape(-1, 1)
        if samples_unc is None:
            samples_unc = sample
        else:
            samples_unc = np.concatenate([samples_unc, sample], axis=1)

    samples = target.param_constrain(samples_unc)
    samples = samples.reshape(num_chains, -1, samples.shape[1])
    moments_1 = np.mean(samples, axis=1)
    moments_2 = np.mean(samples**2, axis=1)

    with open(f"experiments/results/{posterior_name}_mcmc_moments.pkl", 'wb') as f:
        pickle.dump({'moments_1': moments_1, 'moments_2': moments_2}, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--posterior_name', type=str, default='arK')
    parser.add_argument('--num_warmup', type=int, default=25_000)
    parser.add_argument('--num_samples', type=int, default=50_000)
    parser.add_argument('--num_chains', type=int, default=20)
    args = parser.parse_args()
    main(args.posterior_name, args.num_warmup, args.num_samples, args.num_chains)
