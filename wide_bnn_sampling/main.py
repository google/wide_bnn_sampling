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

"""Sampling from a finite BNN."""
import itertools
import os
import pickle
import time

from absl import app
from absl import flags

import jax
from jax import random
import jax.numpy as jnp

from ml_collections import config_dict
from ml_collections import config_flags

import numpy as onp
import tensorflow.compat.v1.gfile as gfile

import wide_bnn_sampling.checkpoint as checkpoint
import wide_bnn_sampling.datasets as datasets
import wide_bnn_sampling.measurements as measurements
import wide_bnn_sampling.models as models
import wide_bnn_sampling.reparametrisation as reparam
import wide_bnn_sampling.samplers as samplers
import wide_bnn_sampling.utils as u


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'store_dir', '~/wide_bnn_sampling', 'storage location')
flags.DEFINE_string('init_store_dir', None, 'initial checkpoint')
config_flags.DEFINE_config_file(
    name='config',
    default=None,
    help_string='training configuration')

jax.config.update('jax_numpy_rank_promotion', 'raise')
jax.config.parse_flags_with_absl()


def _init_mcmc(init_fn, apply_fn, embed_fn, unflatten, C, sampler='hmc'):
  """Initialises the sampler.

  Args:
    init_fn:
      Network initialisation function returned by `neural_tangents`.
    apply_fn:
      Network forward pass function returned by `neural_tangents`.
    embed_fn:
      Network embedding computation as returned by `neural_tangents`.
    unflatten:
      The unflattening function returned by `jax.flatten_util.ravel_pytree`.
    C:
      An `ml_collections.config_dict` instance.
    sampler:
      Specifies which sampler should be used. Available options: `hmc`, `gd`,
      and the (Gaussian) `mh`. See `samplers.py` for details on each of these.

  Returns:
    Sampler initialisation and step functions. See `samplers.py` for details.
  """
  stepsize = C.stepsize
  mcmc_beta = C.mcmc_beta
  kernel_reg = C.noise_scale**2

  if C.kernel_reg_mult is not None and C.kernel_reg_mult >= 0:
    kernel_reg *= C.n_train * C.kernel_reg_mult
  if C.scale_stepsize:
    stepsize *= C.noise_scale / (C.n_train**0.5)
  if C.scale_mcmc_beta:
    mcmc_beta = C.mcmc_beta * stepsize  # keep per-step noise constant
    # mcmc_beta *= hmc_steps  # noise injected only between leapfrog simulations
    mcmc_beta = min(1.0, max(0.0, mcmc_beta))  # correct numerical errors
  u.info('stepsize=%s beta=%s kreg=%s', stepsize, mcmc_beta, kernel_reg)

  apply_flat = lambda pf, x: apply_fn(unflatten(pf), x)
  embed_flat = lambda pf, x: embed_fn(unflatten(pf), x)

  energy_fn = reparam.get_energy_fn(
      reparam_type=C.reparam_type, apply_flat=apply_flat, embed_flat=embed_flat,
      unflatten=unflatten, noise_scale=C.noise_scale, kernel_reg=kernel_reg,
      w_std_out=C.w_std_out, b_std_out=C.b_std_out, parallel=C.parallel)
  logp_diff_fn = reparam.get_logp_diff_fn(
      apply_flat=apply_flat, noise_scale=C.noise_scale, parallel=C.parallel)

  if sampler == 'hmc':
    _sampler_init, _sampler_step = samplers.hmc(  # pylint: disable=invalid-name
        energy_fn=energy_fn, mcmc_beta=mcmc_beta, logp_diff_fn=logp_diff_fn,
        stepsize=stepsize, n_steps=1, parallel=C.parallel, mass=1.0,
        skip_mh=C.hmc_skip_mh)
  elif sampler in ('gd', 'gradient_descent'):
    _sampler_init, _sampler_step = samplers.gd(  # pylint: disable=invalid-name
        energy_fn=energy_fn, stepsize=stepsize, parallel=C.parallel, mass=1.0)
  elif sampler in ('mh', 'metropolis_hastings'):
    _sampler_init, _sampler_step = samplers.gauss_mh(  # pylint: disable=invalid-name
        energy_fn=energy_fn, mcmc_beta=mcmc_beta, logp_diff_fn=logp_diff_fn)
  else:
    raise NotImplementedError(f'unknown sampler `{sampler}`')

  def sampler_init(key, x, y):
    key, pkey, skey = random.split(key, 3)
    _, phi_init = init_fn(pkey, x.shape)
    phi, _ = jax.flatten_util.ravel_pytree(phi_init)
    return key, _sampler_init(skey, phi, x, y)

  def sampler_step(key, state, x, y):
    key, skey = random.split(key)
    state, stats = _sampler_step(skey, state, x, y)
    return key, state, stats

  if C.parallel:
    sampler_step = jax.pmap(sampler_step, axis_name='i')
    sampler_init = jax.pmap(sampler_init, axis_name='i')
  else:
    sampler_step = jax.jit(sampler_step)
    sampler_init = jax.jit(sampler_init)

  return sampler_init, sampler_step


def _init_run(
    key, x, y, sampler_init, accum_init, store_dir, init_store_dir, parallel):
  """
  Initialise auxiliary variables for the main loop. If checkpoints are present,
  attempts to load the most recent one.

  Args:
    key:
      PRNG key.
    x:
      Array of training inputs, one per row. If `parallel == True`, the leading
      dimension should equal the number of devices (sharding).
    y:
      Array of training targets, one per row. If `parallel == True`, the leading
      dimension should equal the number of devices (sharding).
    sampler_init:
      Function which initialises the sampler. See `samplers.py` for details.
    accum_init:
      Function which initialises the accumulators. See `get_accum_fns` in
      `utils.py`.
    store_dir:
      Directory which contains `'./checkpoint'` where the checkpoints indexed by
      sampler step are saved.
    init_store_dir:
      Alternative directory to `store_dir`. Used when no checkpoint can be
      successfully recovered from `store_dir`. Pass `None` the experiment shall
      start from scratch in case of no valid checkpoint in `store_dir`.
    parallel:
      Boolean indicating whether we are running in the `parallel` mode, with
      dataset sharded over multiple devices.

  Raises:
    IOError: If checkpoint is successfuly loaded, but its state is inconsistent.

  Returns:
    Tuple `(chain, aux)`, where `chain` is a tuple of `(mc_key, mc_state)` with
    the former an array of PRNG keys used by the sampler to generate randomness
    in each step, and the latter a state dictionary of the MCMC sampler: see
    `samplers.py` for details. The `aux` is also a tuple of `(mmnts, stats,
    samples, accum_counter, first_step)` where `mmnts` is a dictionary of
    accumulated 1st and 2nd order moments for phi, theta, and test predictions
    after burn-in, `stats` is a dictionary of *projected* phi, theta, and test
    predictions, `accum_counter` is an `itertools.counter` used to monitor how
    many samples were collected so far (different from the number of sampler
    steps when thinning factor is not one), and `first_step` is the zero-indexed
    number of the first sampler step (higher than zero when starting from a
    checkpoint).
  """
  ckpt = checkpoint.load_checkpoint(store_dir)
  if ckpt is None:  # if no checkpoint exists, try in the init_store_dir
    ckpt = checkpoint.load_checkpoint(init_store_dir)

  if ckpt is None:
    stats, samples = {}, {}
    first_step, accum_counter = 0, itertools.count(1)

    key, skey = random.split(key)
    if parallel:
      skey = jax.device_put_replicated(skey, jax.local_devices())
    mc_key, mc_state = sampler_init(skey, x, y)
    mmnts = accum_init(mc_state['theta'].shape[-1])
  else:
    if parallel and jax.process_count() > 1:
      # Check if the same checkpoint has been loaded on all hosts.
      sid = jnp.array(ckpt['sid'])
      sid = jax.device_put_replicated(sid, jax.local_devices())
      all_equal = bool(u.all_equal(sid)[0])
      if not all_equal:
        u.error(f'Process {jax.process_index()} loaded ckpt {sid}.')
        raise IOError('Different checkpoints loaded')

    stats, samples = ckpt['stats'], ckpt['samples']
    mc_key, mc_state, mmnts = ckpt['key'], ckpt['state'], ckpt['mmnts']
    if parallel:
      mc_key = jax.device_put_replicated(mc_key, jax.local_devices())
      mc_state = jax.device_put_replicated(mc_state, jax.local_devices())
      mmnts = u.shard_moments(mmnts)

    first_step, accum_counter = ckpt['sid'], itertools.count(ckpt['n_acc'])

  return (mc_key, mc_state), (mmnts, stats, samples, accum_counter, first_step)


def run_experiment(C: config_dict.ConfigDict, m: measurements.Measurements):
  """Run finite BNN experiment, optionally sharding data over devices."""
  key = random.PRNGKey(C.seed)

  ## DATA
  ds = datasets.cifar10_tfds(
      n_train=C.n_train, n_test=C.n_test, flatten=C.architecture == 'fcn',
      regression=True, data_dir=C.data_dir)

  if C.parallel:  # shard data equally over devices
    ds = u.shard(ds)

  (x, y), (x_test, y_test) = ds
  u.info('x shape: %s', x.shape)

  d_out = y.shape[-1]
  if C.n_units < C.n_train * d_out:
    u.warning('we may be far from the ntk regime')

  ## MODEL
  init_fn, apply_fn, _, embed_fn = models.get_network(d_out=d_out, C=C)

  x_shape = x.shape[1:] if C.parallel else x.shape
  _, unflatten = jax.flatten_util.ravel_pytree(init_fn(key, x_shape)[1])

  # for evaluation on the test set
  def f_test(pf, x_test):
    return apply_fn(unflatten(pf), x_test)

  proj_param_key = random.PRNGKey(1)  # constant key for projections
  if C.parallel:
    proj_preds_key = random.split(
        random.fold_in(proj_param_key, jax.process_index()),
        jax.local_device_count())
    proj_param_key = jax.device_put_replicated(
        proj_param_key, jax.local_devices())
    f_test = jax.pmap(f_test, axis_name='i')
  else:
    proj_preds_key = random.split(proj_param_key, 1)[0]
    f_test = jax.jit(f_test)

  ## MCMC CHAIN
  accum_init, accum_update = u.get_accum_fns(y_test.shape, C.parallel)
  sampler_init, sampler_step = _init_mcmc(
      init_fn=init_fn, apply_fn=apply_fn, embed_fn=embed_fn,
      unflatten=unflatten, C=C)
  project_fn = u.get_projection_fn(
      param_key=proj_param_key, pred_key=proj_preds_key, n_proj=C.n_proj,
      parallel=C.parallel, vmap_size=C.proj_vmap_size)

  ## MAIN LOOP
  key, run_init_key = random.split(key)
  chain, (mmnts, stats, samples, accum_counter, first_step) = _init_run(
      run_init_key, x=x, y=y, accum_init=accum_init, sampler_init=sampler_init,
      parallel=C.parallel, store_dir=m.save_dir, init_store_dir=m.init_save_dir)
  mc_key, mc_state = chain

  start = time.time()
  n_steps = C.burn_in + C.step_count
  for sid in range(first_step, n_steps):
    # single sampler step
    mc_key, mc_state, iter_stats = sampler_step(mc_key, mc_state, x, y)

    # save stats
    for k, v in iter_stats.items():
      if C.parallel:
        v = v[0]
      stats.setdefault(k, []).append(onp.array(jax.device_get(v)))

    # update accumulators
    if sid >= C.burn_in and (sid - C.burn_in) % C.thin == 0:
      phi, theta = mc_state['phi'], mc_state['theta']
      preds, n_acc = f_test(theta, x_test), next(accum_counter)

      mmnts = accum_update(
          n_acc, preds=preds, phi=phi, theta=theta, mmnts=mmnts)  # pytype:disable=wrong-keyword-args

      # store the projected samples
      for k, v in project_fn(preds, phi, theta).items():
        if C.parallel:
          v = v[0]
        samples.setdefault(k, []).append(onp.array(jax.device_get(v)))

      del preds, phi, theta  # release memory

    # log progress
    if sid % C.log_freq == 0:
      pst = (time.time() - start) / (sid + 1)
      m.log('step_time', pst, step=sid)

      for s in ('energy', 'likelihood', 'prior', 'logdet', 'p_acc'):
        val = jax.device_get(iter_stats[s]).mean().item()
        m.log(s, val, step=sid)

    # checkpoint
    # if time.time() - last_ckpt_save > C.ckpt_freq:  # breaks w/ multi-device
    if sid % C.ckpt_freq == 0 and sid != first_step:
      u.info('saving checkpoint after %s steps', sid)
      checkpoint.save_checkpoint(
          key=mc_key, state=mc_state, samples=samples, mmnts=mmnts, stats=stats,
          n_acc=u.counter2val(accum_counter), sid=sid, parallel=C.parallel,
          save_dir=m.save_dir, max_n_checkpoints=C.max_n_checkpoints,
          keep_every_n_checkpoint=C.keep_every_n_checkpoint)

  # finish & clean-up
  duration = time.time() - start
  n_acc = u.counter2val(accum_counter)
  stats = {k: onp.array(jax.device_get(stats[k])) for k in stats}
  p_acc = onp.mean(stats['p_acc'][C.burn_in:]).item()
  rmse, acc = u.rmse_and_acc(
      y_test, preds=mmnts['preds'][0], parallel=C.parallel)

  step = int(onp.round(onp.log10(C.noise_scale)))  # can't handle non-integers
  m.log('acc', acc, step=step)
  m.log('rmse', rmse, step=step)
  m.log('duration', duration, step=step)
  m.log('mean_p_acc', p_acc, step=step)
  u.info('final: p_acc %s rmse %s acc %s time %s', p_acc, rmse, acc, duration)

  samples = {k: onp.array(jax.device_get(samples[k])) for k in samples}
  results = {
      'rmse': rmse, 'acc': acc, 'p_acc': p_acc, 'stats': stats,
      'n_acc': n_acc, 'save_dir': m.save_dir}

  # save results
  if C.parallel:
    mmnts = u.unshard_moments(mmnts)

  if jax.process_index() == 0 and C.save_stats:
    with gfile.Open(os.path.join(m.save_dir, 'samples'), 'wb') as f:
      pickle.dump(samples, f)
    with gfile.Open(os.path.join(m.save_dir, 'mmnts'), 'wb') as f:
      pickle.dump(mmnts, f)
    with gfile.Open(os.path.join(m.save_dir, 'results'), 'wb') as f:
      pickle.dump(results, f)


def main(_):
  C = FLAGS.config
  m = measurements.Measurements(FLAGS.store_dir, FLAGS.init_store_dir)

  u.info(f'config:\n{C}')
  u.info(f'number of processes: {jax.process_count()}')
  u.info(f'number of devices: {jax.device_count()}')
  u.info(f'number of local devices: {jax.local_device_count()}')

  run_experiment(C, m)


if __name__ == '__main__':
  app.run(main)
