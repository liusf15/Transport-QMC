import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import qmc
import matplotlib.pyplot as plt
import seaborn as sns

MACHINE_EPSILON = np.finfo(np.float32).eps

def sample_uniform(nsample, d, rng, sampler):
    if sampler == 'mc':
        U = rng.random((nsample, d))
    else:
        soboleng =qmc.Sobol(d, scramble=True, seed=rng)    
        U = soboleng.random(nsample) * (1 - MACHINE_EPSILON) + MACHINE_EPSILON * .5
    return U

def sample_gaussian(nsample, d, seed=0, sampler='rqmc'):
    if sampler == 'rqmc':
        soboleng = qmc.Sobol(d, scramble=True, seed=seed)
        X = ndtri(soboleng.random(nsample) * (1 - MACHINE_EPSILON) + .5 * MACHINE_EPSILON)
    elif sampler == 'mc':    
        if isinstance(seed, int):
            rng = np.random.default_rng(seed)
            X = rng.standard_normal((nsample, d))
        else:
            X = seed.standard_normal((nsample, d))
    else:
        raise NotImplementedError
    return X


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

def get_mse(true_moments, est_moments):
    mse_1 = np.mean((true_moments[0] - est_moments[0])**2)
    mse_2 = np.mean((true_moments[1] - est_moments[1])**2)
    return mse_1, mse_2


def plot_mse(mse, figname):
    mse = pd.DataFrame(mse).T.reset_index(names=['sampler', 'm', 'seed'])
    m_list = mse['m'].unique()
    d = mse.shape[1] - 3
    # mse.set_index(['sampler', 'm'], inplace=True)
    sns.set_theme(context='paper', style='whitegrid', font_scale=1.5)
    fig, ax = plt.subplots(1, d, figsize=(3 * d, 2.5), sharey=True)
    for j in range(d):
        sns.pointplot(ax=ax[j], data=mse, x='m', y=j, hue='sampler', ls='', capsize=0.1, markers=['.', '*'], markersize=10)
    
        ax[j].set_ylabel('MSE')

        l1 = 1 / (2**m_list)
        l1 = l1 / l1[0] * mse.loc[(mse['sampler'] == 'mc') & (mse['m'] == m_list.min())][j].mean()
        l2 = 1 / (4**m_list)
        l2 = l2 / l2[0] * mse.loc[(mse['sampler'] == 'rqmc') & (mse['m'] == m_list.min())][j].mean()
        ax[j].plot(np.arange(len(m_list)), l1, ls='--', c='gray')
        ax[j].plot(np.arange(len(m_list)), l2, ls=':', c='gray')
        
        if j == 0:
            handles, labels = ax[j].get_legend_handles_labels()
            labels = ['MC', 'RQMC']
            ax[j].legend(handles, labels, title='', markerscale=1., fontsize=13)
            ax[j].annotate(r'$n^{-1}$', (len(m_list)-1.5, l1[-1]), fontsize=13)
            ax[j].annotate(r'$n^{-2}$', (len(m_list)-1.5, l2[-1]), fontsize=13)
        else:
            ax[j].legend().remove()
        ax[j].set_xticks(np.arange(len(m_list)))
        ax[j].set_xticklabels([r"$2^{{{:.0f}}}$".format(m) for m in m_list])
        ax[j].set_xlabel('Sample size', fontsize=12)

    plt.yscale('log', base=2)
    plt.tight_layout()
    plt.savefig(figname, bbox_inches='tight')
    print('Figure saved to', figname)
    plt.close()

