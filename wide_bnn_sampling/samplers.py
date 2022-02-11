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

"""MCMC sammplers."""
import jax
from jax import random
import jax.numpy as jnp


def gauss_mh(energy_fn, logp_diff_fn, mcmc_beta):
  """Metropolis-Hastings with Gaussian proposal."""

  def init_fn(_, phi, x, y):
    """Returns sampler state."""
    energy, (theta, logdet, _) = energy_fn(phi, x, y)
    return {'phi': phi, 'theta': theta, 'logdet': logdet, 'energy': energy}

  def sample_fn(key, state, x, y):
    """Executes a single sampler step.

    Args:
      key:
        A PRNG key.
      state:
        Current state of the sampler. Initial state obtained from `init_fn`.
      x:
        A `jax.numpy.ndarray` of inputs with dataset size as leading dimension.
      y:
        A `jax.numpy.ndarray` of targets with dataset size as leading dimension.

    Returns:
      A tuple `(state, stats)`. `state` is a dictionary representing the
      internal state of the sampler; it is guaranteed to contain `phi`
      (flattened parameter vector---non-reparametrised), `theta` (flattened
      parameter vector---reparametrised), `logdet` (log determinant of the
      reparametrised posterior density corresponding to `phi`), and `energy`
      (negative log posterior density of `phi` up to an additive constant).
      `stats` is a dictionary which combines the auxiliary statistics returned
      by the `energy_fn` (see its documentation) with sampler statistics; it is
      guaranteed to contain energy values for the original state (`energy`),
      the new state (`energy_new`), and the acceptance probability from the last
      step (`p_acc`).
    """
    phi, theta = state['phi'], state['theta']
    logdet, energy = state['logdet'], state['energy']

    # get a new proposal
    key, pkey = random.split(key)
    phi_new = jnp.sqrt(1.0 - mcmc_beta) * phi
    phi_new += jnp.sqrt(mcmc_beta) * random.normal(pkey, phi.shape)
    energy_new, (theta_new, logdet_new, stats) = energy_fn(phi_new, x, y)

    # accept is 1 if proposal accepted, 0 otherwise
    log_p_acc = logp_diff_fn(theta, theta_new, x, y) + logdet_new - logdet
    log_p_acc += 0.5 * jnp.sum(phi_new**2 - phi**2)
    p_acc = jnp.minimum(1, jnp.exp(log_p_acc))

    key, akey = random.split(key)
    accept = random.bernoulli(akey, p_acc).astype(phi.dtype)
    update = lambda old, new: jnp.where(accept, new, old)

    state['phi'] = update(phi, phi_new)
    state['theta'] = update(theta, theta_new)
    state['logdet'] = update(logdet, logdet_new)
    state['energy'] = update(energy, energy_new)

    stats.update({'p_acc': p_acc, 'energy': energy, 'energy_new': energy_new})
    return state, stats

  return init_fn, sample_fn


def hmc(energy_fn, logp_diff_fn, n_steps, stepsize, mcmc_beta, parallel,
        mass=1.0, skip_mh=False):
  """
  Underdamped Hamiltonian Monte Carlo (HMC). Langevin Monte Carlo (LMC) is HMC
  with `n_steps == 1`. Usual Guassian density for the auxiliary momentum
  variables is assumed.

  Args:
    energy_fn:
      A function computing the neg. log posterior density (up to a constant).
      See `get_energy_fn` in `reparametriation.py` for details.
    logp_diff_fn:
      A function which computes the difference between log density values at
      two different values of `theta` (i.e., the parameter itself for the
      Identity, and the `theta = T(phi)` value for the NNGP parametrisation.
      See `get_logp_diff_fn` in `reparametriation.py` for details.
    n_steps:
      Number of Leapfrog steps to take to obtain a new proposal. Set to one
      for LMC.
    stepsize:
      Stepsize for the Leapfrog integrator. Should be positive.
    mcmc_beta:
      Damping factor. New momentum is computed as
      `m' = sqrt(1 - beta) * m + sqrt(beta) * sqrt(mass) * gauss_noise`
      where the `gauss_noise` is a vector of i.i.d. Gaussian random variables.
      Must be in `[0, 1]`; set to one to turn off damping.
    parallel:
      Boolean indicating whether we are running in the `parallel` mode, with
      dataset sharded over multiple devices.
    mass:
      The momentum variables are evolved according to the density of
      `N(0, mass * I)`. In Leapfrog, it divides the gradients, so can be used
      for rescaling.
    skip_mh:
      Whether to skip the Metropolis-Hastings correction. Common in LMC.
  Returns:
      A tuple formed by updated sampler state, and a dictionary of statistics
      useful for monitoring the sampler.
  """
  if n_steps < 1:
    raise ValueError(f'must take at least one step; `n_steps={n_steps}`')
  energy_vg = jax.value_and_grad(energy_fn, has_aux=True)

  def init_fn(key, phi, x, y):
    key, mkey = random.split(key)

    m = jnp.sqrt(mass) * random.normal(mkey, phi.shape)
    (energy, (theta, logdet, _)), g = energy_vg(phi, x, y)
    if parallel:
      g = jax.lax.pmean(g, axis_name='i')

    return {'phi': phi, 'theta': theta, 'g': g, 'm': m,
            'logdet': logdet, 'energy': energy}

  def sample_fn(key, state, x, y):
    phi, theta, g = state['phi'], state['theta'], state['g']
    m, logdet, energy = state['m'], state['logdet'], state['energy']

    # partial momentum refresh
    key, mkey = random.split(key)
    m *= jnp.sqrt(1 - mcmc_beta)
    m += jnp.sqrt(mass * mcmc_beta) * random.normal(mkey, m.shape)

    # leapfrog
    def _leapfrog(carry, _):
      phi_new, theta_new, g_new, m_new = carry

      m_new = m_new - 0.5 * stepsize * g_new
      phi_new = phi_new + stepsize * m_new / mass
      (energy_new, aux), g_new = energy_vg(phi_new, x, y)
      if parallel:
        g_new = jax.lax.pmean(g_new, axis_name='i')
      m_new = m_new - 0.5 * stepsize * g_new

      theta_new, logdet_new, stats_new = aux
      carry = (phi_new, theta_new, g_new, m_new)
      return carry, (energy_new, logdet_new, stats_new)

    (phi_new, theta_new, g_new, m_new), aux = jax.lax.scan(
        _leapfrog, init=(phi, theta, g, m), xs=None, length=n_steps)
    energy_new, logdet_new, stats = jax.tree_map(lambda p: p[-1], aux)

    # MH correction
    m_new *= -1.0  # 1st sign flip

    log_p_acc = logp_diff_fn(theta, theta_new, x, y) + logdet_new - logdet
    log_p_acc += 0.5 * jnp.sum(m**2 - m_new**2) / mass
    p_acc = jnp.minimum(1.0, jnp.exp(log_p_acc))

    key, akey = random.split(key)
    accept = 1 if skip_mh else random.bernoulli(akey, p_acc).astype(m.dtype)
    update = lambda old, new: (1 - accept) * old + accept * new

    state['m'] = update(m, m_new)
    state['g'] = update(g, g_new)
    state['phi'] = update(phi, phi_new)
    state['theta'] = update(theta, theta_new)
    state['logdet'] = update(logdet, logdet_new)
    state['energy'] = update(energy, energy_new)

    state['m'] *= -1.0  # 2nd sign flip

    stats.update({'p_acc': p_acc, 'energy': energy, 'energy_new': energy_new})
    return state, stats

  return init_fn, sample_fn


def gd(energy_fn, stepsize, parallel, mass=1.0):
  """
  A vanilla gradient descent 'sampler'. Generally will not sample from the
  true posterior. Useful for debugging and reference.

  Args:
    energy_fn:
      A function computing the neg. log posterior density (up to a constant).
      See `get_energy_fn` in `reparametriation.py` for details.
    stepsize:
      Stepsize for the gradient update. Should be positive.
    parallel:
      Boolean indicating whether we are running in the `parallel` mode, with
      dataset sharded over multiple devices.
    mass:
      Useful for comparing with the HMC behaviour, since `mass` there can be
      interpreted as a simple scaling of gradients/stepsize. See the `hmc`
      documentation for details.
  Returns:
      A tuple formed by updated sampler state, and a dictionary of statistics
      useful for monitoring the sampler.
  """
  energy_vg = jax.value_and_grad(energy_fn, has_aux=True)

  def init_fn(_, phi, x, y):
    # initial theta not used -> don't bother computing
    return {'phi': phi, 'theta': phi}

  def sample_fn(_, state, x, y):
    phi = state['phi']

    (energy, (theta, _, stats)), g = energy_vg(phi, x, y)
    if parallel:
      g = jax.lax.pmean(g, axis_name='i')
    phi -= 0.5 * stepsize**2 * g / mass  # match hmc stepsize scaling

    stats.update({'p_acc': -1.0, 'energy': energy, 'energy_new': energy})
    return {'phi': phi, 'theta': theta}, stats

  return init_fn, sample_fn
