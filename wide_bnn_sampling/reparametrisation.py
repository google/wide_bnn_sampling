# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reparametrise from the 'prior' phi to the posterior 'theta'."""
import jax
import jax.numpy as jnp


def _get_ident_energy_fn(f, noise_scale, parallel):
  """
  Return the negative log density of the BNN posterior induced by a multivariate
  standard normal prior, and Gaussian likelihood on top of the NN outputs.

  Args:
    f:
      A function which takes a *flattened* vector of NN parameters, and an
      array of inputs, and returns the corresponding NN predictions.
    noise_scale:
      Standard deviation used for the Gaussian likelihood.
    parallel:
      Boolean indicating whether we are running in the `parallel` mode, with
      dataset sharded over multiple devices.

  Returns:
    A function which takes a *flattened* vector of NN parameters, and arrays
    of inputs and outputs, and returns the corresponding unnormalised energy
    (i.e., negative log posterior density up to an additive constant), and
    a 3-tuple `(theta, logdet, stats)` where `stats` is a dictionary of
    statistics containing values of the individual components, `logdet` is
    the absolute value of the log determinant of the transformation Jacobian,
    and `theta` is the reparametrised *flattened* parameter vector. Here, no
    reparametrisation is applied, hence the log determinant is zero, and `theta`
    equals the input flattened vector.
  """
  lvar = noise_scale**2

  def energy_fn(phi, x, y):
    theta = phi

    prior = -0.5 * jnp.sum(theta ** 2 - 1)
    likelihood = -0.5 * jnp.sum((y - f(theta, x)) ** 2 - lvar) / lvar

    if parallel:
      likelihood = jax.lax.psum(likelihood, axis_name='i')

    logdet = 0.0  # theta = phi -> log|det(I)| = 0

    energy = - (prior + likelihood + logdet)
    stats = {'prior': prior, 'likelihood': likelihood,
             'logdet': logdet, 'phi': -prior}

    return energy, (theta, logdet, stats)

  return energy_fn


def _get_nngp_energy_fn(
    embed, unflatten, kernel_reg, noise_scale, w_std_out, b_std_out, parallel):
  """
  Return the negative log density of the BNN posterior induced by a multivariate
  standard normal prior, and Gaussian likelihood on top of the NN outputs, after
  *reparametrisation*. Uses the effective implementation via Cholesky. Assumes
  `'ntk'` parametrisation, and that the readout layer is linear.

  Args:
    embed:
      A function which takes a *flattened* vector of NN parameters, and an
      array of inputs, and returns the corresponding NN top-layer embeddings
      scaled by the square root of their dimension, but without the additional
      constant entry for the bias term, nor scaling by the readout weight and
      bias (prior) deviations.
    unflatten:
      The unflattening function returned by `jax.flatten_util.ravel_pytree`.
    kernel_reg:
      A float used to regularise the empirical Gram matrix of top-layer
      embeddings before performing Cholesky decomposition. If set to
      `noise_scale ** 2`, it will lead to exact posterior samples in a
      Bayesian linear model, under the same prior and likelihood
      assumptions as described above.
    noise_scale:
      Standard deviation used for the Gaussian likelihood.
    w_std_out:
      Standard deviation (prior) for the readout weights.
    b_std_out:
      Standard deviation (prior) for the readout biases.
    parallel:
      Boolean indicating whether we are running in the `parallel` mode, with
      dataset sharded over multiple devices.

  Returns:
    A function which takes a *flattened* vector of NN parameters, and arrays
    of inputs and outputs, and returns the corresponding unnormalised energy
    (i.e., negative log posterior density up to an additive constant), and
    a 3-tuple `(theta, logdet, stats)` where `stats` is a dictionary of
    statistics containing values of the individual components, `logdet` is
    the absolute value of the log determinant of the transformation Jacobian,
    and `theta` is the reparametrised *flattened* parameter vector.
  """
  lvar = noise_scale**2

  def reparam_fn(phi, x, y):
    phi_unflat = unflatten(phi)
    phi_out = jnp.vstack(phi_unflat[-1])

    # compute Gram matrix (the counterpart of the empirical NNGP kernel matrix)
    emb = embed(phi, x)  # ignores the output layer parameters
    temb = jnp.hstack((w_std_out * emb, b_std_out * jnp.ones((len(emb), 1))))
    gram = temb.T @ temb

    # if dataset is sharded, use the sum formula for gram-matrix
    if parallel:
      gram = jax.lax.psum(gram, axis_name='i')

    # add regularisation and compute the Cholesky decomposition
    gram += kernel_reg * jnp.eye(temb.shape[-1])
    cho, lower = jax.scipy.linalg.cho_factor(gram)

    # incremental computation of the reparametrised readout layer
    theta_out = kernel_reg**0.5 * phi_out

    # prepare the right-hand side for the first triangular solve
    rhs = temb.T @ y
    if parallel:
      rhs = jax.lax.psum(rhs, axis_name='i')

    # first triangular solve (result serves as the rhs for the second one)
    theta_out += jax.lax.linalg.triangular_solve(
        cho, rhs, left_side=True, lower=lower, transpose_a=not lower)

    # second triangular solve
    theta_out = jax.lax.linalg.triangular_solve(
        cho, theta_out, left_side=True, lower=lower, transpose_a=lower)

    # replace readout with reparametrised values, and flatten again
    theta_unflat = phi_unflat[:-1] + [(theta_out[:-1], theta_out[-1:])]
    theta = jax.flatten_util.ravel_pytree(theta_unflat)[0]

    return theta, (theta_out, temb, (cho, lower))

  def energy_fn(phi, x, y):
    d_out = y.shape[-1]
    theta, (theta_out, temb, (cho, _)) = reparam_fn(phi, x, y)

    pred = temb @ theta_out  # parameters affecting embedding do not change
    prior = -0.5 * jnp.sum(theta ** 2 - 1)
    likelihood = -0.5 * jnp.sum((y - pred) ** 2 - lvar) / lvar

    if parallel:
      likelihood = jax.lax.psum(likelihood, axis_name='i')

    logdet = d_out * jnp.sum(0.5 * jnp.log(kernel_reg) - jnp.log(jnp.diag(cho)))
    # `abs` is not needed for Chol factors (positive diag by definition)

    energy = - (prior + likelihood + logdet)
    stats = {'prior': prior, 'likelihood': likelihood,
             'logdet': logdet, 'phi': 0.5 * jnp.sum(phi ** 2 - 1)}
    return energy, (theta, logdet, stats)

  return energy_fn


def get_energy_fn(
    reparam_type, apply_flat, embed_flat, unflatten, kernel_reg, noise_scale,
    w_std_out, b_std_out, parallel):
  """
  Return the negative log density of the BNN posterior induced by a multivariate
  standard normal prior, and Gaussian likelihood on top of the NN outputs, after
  *reparametrisation*. Two (re)parametrisations are available: `'identity'` (no
  reparametrisation, and `'nngp'` (reparmetrisation of the readout weights which
  maps multivariate standard normal samples to a distribution vanishingly close
  to the posterior in wide BNNs). Assumes the readout layer is linear, and the
  `'ntk'` parametrisation if `'nngp'` is selected.

  Args:
    reparam_type:
      A string specifying which reparametrisation is to be used. As described
      above, supported values are `'identity'` and `'nngp'`.
    apply_flat:
      A function which takes a *flattened* vector of NN parameters, and an
      array of inputs, and returns the corresponding NN predictions.
    embed_flat:
      A function which takes a *flattened* vector of NN parameters, and an
      array of inputs, and returns the corresponding NN top-layer embeddings
      scaled by the square root of their dimension, but without the additional
      constant entry for the bias term, nor scaling by the readout weight and
      bias (prior) deviations.
    unflatten:
      The unflattening function returned by `jax.flatten_util.ravel_pytree`.
    kernel_reg:
      A float used to regularise the empirical Gram matrix of top-layer
      embeddings before performing Cholesky decomposition. If set to
      `noise_scale ** 2`, it will lead to exact posterior samples in a
      Bayesian linear model, under the same prior and likelihood
      assumptions as described above.
    noise_scale:
      Standard deviation used for the Gaussian likelihood.
    w_std_out:
      Standard deviation (prior) for the readout weights.
    b_std_out:
      Standard deviation (prior) for the readout biases.
    parallel:
      Boolean indicating whether we are running in the `parallel` mode, with
      dataset sharded over multiple devices.

  Returns:
    A function which takes a *flattened* vector of NN parameters, and arrays
    of inputs and outputs, and returns the corresponding unnormalised energy
    (i.e., negative log posterior density up to an additive constant), and
    a 3-tuple `(theta, logdet, stats)` where `stats` is a dictionary of
    statistics containing values of the individual components (`prior`, `logdet`
    `likelihood`, and also the norm of `phi` for logging), `logdet` is the
    absolute value of the log determinant of the transformation Jacobian, and
    `theta` is the reparametrised *flattened* parameter vector.
  """
  if reparam_type == 'identity':  # standard parametrisation
    energy_fn = _get_ident_energy_fn(
        f=apply_flat, noise_scale=noise_scale, parallel=parallel)
  elif reparam_type in ('nngp', 'nngp_chol'):  # our (re)parametrisation
    energy_fn = _get_nngp_energy_fn(
        embed=embed_flat, unflatten=unflatten, parallel=parallel,
        kernel_reg=kernel_reg, noise_scale=noise_scale,
        w_std_out=w_std_out, b_std_out=b_std_out)
  else:
    raise NotImplementedError(f'unknown `reparam_type` "{reparam_type}"')

  return energy_fn


def get_logp_diff_fn(apply_flat, noise_scale, parallel):
  """Returned function computes `log p(theta'|X,y) - log p(theta|X,y)`."""
  lvar = noise_scale**2

  def logp_diff_fn(theta, theta_prime, x, y):
    preds, preds_prime = apply_flat(theta, x), apply_flat(theta_prime, x)

    diff = 0.5 * jnp.sum((y - preds) ** 2 - (y - preds_prime) ** 2) / lvar
    if parallel:
      diff = jax.lax.psum(diff, axis_name='i')
    diff += 0.5 * jnp.sum(theta ** 2 - theta_prime ** 2)

    return diff

  return logp_diff_fn
