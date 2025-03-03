# Transport-QMC

A python package for [Transport Quasi-Monte Carlo](https://arxiv.org/abs/2412.16416)

Installation

```
git clone github.com:liusf15/Transport-QMC.git
cd Transport-QMC
pip install -e .
```

See [example.ipynb](experiments/example.ipynb) for an example of using the package.

To reproduce the experiments with posteriordb in the paper, run
```
python -m experiments.posteriordb_simulation --posterior_name hmm
```