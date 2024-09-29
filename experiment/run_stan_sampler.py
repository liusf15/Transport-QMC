import numpy as np
import pandas as pd
import cmdstanpy as csp
import bridgestan as bs
import pickle
import argparse

def run_sampler(name, chains=20, warmup=25_000, nsample=50_000):
    stan = f"qmc_flow/stan_models/{name}.stan"
    data = f"qmc_flow/stan_models/{name}.json"
    bs_model = bs.StanModel(stan, data, make_args=["STAN_THREADS=True", "TBB_CXX_TYPE=gcc"])
    d = bs_model.param_num()
    model = csp.CmdStanModel(stan_file=stan)
    fit = model.sample(
        data=data,
        seed=1,
        metric="unit_e",
        show_console=False,
        adapt_delta=0.9,
        chains=chains,
        parallel_chains=4,
        iter_warmup=warmup,
        iter_sampling=nsample,
        show_progress=True,
    )
    meta_columns = len(fit.metadata.method_vars.keys())
    # param_names = bs_model.param_names()
    # col_idx = [fit.column_names.index(var_name) for var_name in param_names]
    stan_draws = fit.draws(concat_chains=False)[:, :, meta_columns:meta_columns+d]
    moments_1 = np.mean(stan_draws, axis=0)
    moments_2 = np.mean(stan_draws**2, axis=0)
    
    moment_filename = f'qmc_flow/stan_models/moments_{name}_chain_{chains}_warmup_{warmup}_nsample_{nsample}.pkl'
    results = {'moment_1': moments_1, 'moment_2': moments_2}
    with open(moment_filename, 'wb') as f:
        pickle.dump(results, f)

MODEL_LIST = [
    # "normal",
    # "corr-normal",
    # "rosenbrock",
    # "glmm-poisson",
    "hmm",
    "garch",
    # "lotka-volterra",
    # 'irt-2pl',
    'eight-schools',
    'normal-mixture',
    # 'arma',
    'arK',
    # 'prophet',
    # 'covid19-impperial-v2',
    # 'pkpd',
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobid', type=int, default=0)
    parser.add_argument('--name', type=str)
    args = parser.parse_args()

    name = args.name
    print(name)

    
    # run_sampler(name, chains=1, warmup=100, nsample=100)
    run_sampler(name)


