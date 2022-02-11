# Wide BNN sampling

Code for the paper **"Simplicity of the wide Bayesian neural network weight 
posterior: theory and accelerated sampling"** by 
[Jiri Hron](https://sites.google.com/corp/view/jirihron), 
[Roman Novak](https://research.google/people/RomanNovak/), 
[Jeffrey Pennington](https://scholar.google.com/citations?user=cn_FoswAAAAJ&hl=en),
and [Jascha Sohl-Dickstein](http://www.sohldickstein.com/).

The main contribution is a reparametrisation of Bayesian neural network (BNN) 
posteriors which enables 10-200x faster mixing compared to standard parametrisation
when combined with Hamiltonian Monte Carlo. Intriguingly, the **sampling speed
becomes higher the larger the BNN**. The reparametrisation is derived using
the large width BNN theory 
(e.g., [Matthews et al.](https://arxiv.org/abs/1804.11271),
[Lee et al.](https://arxiv.org/abs/1711.00165), 
[Hron et al.](https://arxiv.org/abs/2006.10541)),
and can be shown to transform the *exact* BNN weight space posterior into a
distribution whose KL divergence from the multivariate standard normal
distribution vanishes in the large width limit. This is the source of the 
speed-up at large width, but we have sometimes observed 10x faster mixing even 
when very far from the wide regime (i.e., when width is much smaller than the
dataset size).

The code in this repository provides an efficient way of computing both the
reparametrised density and the parameters at the same time. As detailed in the
paper, the implementation is based on Cholesky decomposition, and a forward
and backward solve akin to the usual implementation of the Cholesky solver.

We rely on [JAX](https://github.com/google/jax), a high-performance machine
learning library based on [XLA](https://www.tensorflow.org/xla) with simple 
NumPy/Autograd like API, and
[Neural Tangents](https://github.com/google/neural-tangents),
a high-level neural network API enabling computation with finite as well as
*infinite* neural networks.
See `setup.py` for other dependencies.

## Using the code

The code has several dependencies described in `setup.py`. To install them
automatically, use
```console
git clone https://github.com/google/wide_bnn_sampling
cd wide_bnn_sampling
pip install -e .
```
A dependency not included is `jaxlib` whose installation differs based on the
available hardware; please follow the relevant instructions from 
[JAX's repository](https://github.com/google/jax#installation).
If you want to just quickly try the code with CPU backend, you can run
`pip install jax jaxlib --upgrade`.

To set off an experiment, you can modify the provided `config.py` as needed,
and invoke
```console
python3 wide_bnn_sampling/main.py --config wide_bnn_sampling/config.py --store_dir <results-directory>
```

The high-level structure of `main.py` dependencies is descibed below:
   * `config.py`: Configuration flags for the dataset, neural network
   architecture, the sampler, and auxiliary experiment run settings.
   * `datasets.py`: Loading and preprocessing of data.
   * `measurements.py`: Logging utilities.
   * `models.py`: Constructs neural network architectures with
   [Neural Tangents](https://github.com/google/neural-tangents).
   * `reparametrisation.py`: Effective implementation of the reparametrisation
   under the assumption of Gaussian likelihood and prior (details in the paper).
   * `samplers.py`: Custom implementation of HMC/LMC, and Metropolis-Hastings
   with a simple Gaussian proposal.
   * `utils.py`: Auxiliary methods primarily used within `main.py`.

**CAVEAT:** Despite using several tricks for improved stability, we observed 
significant deterioration of acceptance probabilities when computational 
precisions is low. We recommend using at least `float32`, but preferring
`float64` where feasible. The relevant flags in JAX are `jax_enable_x64` (and
`jax_default_matmul_precision` if on TPU).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## License

Apache 2.0; see [`LICENSE`](LICENSE) for details.

## Disclaimer

This project is not an official Google project. It is not supported by
Google and Google specifically disclaims all warranties as to its quality,
merchantability, or fitness for a particular purpose.


