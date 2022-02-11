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

"""Misc helper functions that do not fit elsewhere."""
import logging
import math

import jax
from jax import numpy as jnp
from jax import random

import numpy as onp


def counter2val(counter):
  """Extract current value of an `itertools.count()` w/o incrementing it."""
  return counter.__reduce__()[1][0]


def tree_shape(tree):
  """Get shapes of values in a pytree."""
  return jax.tree_map(lambda p: p.shape, tree)


def tree_take_0(tree):
  return jax.tree_map(lambda x: x[0], tree)


def _rmse_and_acc(y_test, preds):
  rmse = jnp.mean(jnp.sum((y_test - preds) ** 2, -1))
  rmse = jax.lax.pmean(rmse, axis_name='i') ** 0.5

  acc = jnp.mean(jnp.argmax(y_test, -1) == jnp.argmax(preds, -1))
  acc = jax.lax.pmean(acc, axis_name='i')
  return rmse, acc
_rmse_and_acc = jax.pmap(_rmse_and_acc, axis_name='i')


def rmse_and_acc(y_test, preds, parallel):
  """Compute root mean squared error and accuracy from predictive mean.

  Args:
    y_test:
      Array of test targets, one per row.
    preds:
      Array of test predictions, one per row, and sharded if `parallel` is true.
    parallel:
      Boolean indicating whether we are running in the `parallel` mode, with
      dataset sharded over multiple devices.
  Returns:
    Tuple of root mean squared error and accuracy.
  """
  if parallel:
    rmse, acc = _rmse_and_acc(y_test, preds)
    assert rmse.shape == (jax.local_device_count(),), rmse.shape
    assert acc.shape == (jax.local_device_count(),), acc.shape
    rmse, acc = float(rmse[0]), float(acc[0])
  else:
    preds = jax.device_get(preds)
    rmse = onp.mean(onp.sum((y_test - preds)**2, -1)).item() ** 0.5
    acc = onp.mean(onp.argmax(y_test, -1) == onp.argmax(preds, -1)).item()

  return rmse, acc


def get_accum_fns(out_shape, parallel):
  """
  Create accumulators for first and second order moments of the network weights
  and test set predictions. Maintains running estimates instead of storing all
  samples, and computing the moments at the end, to avoid OOM issues.

  Args:
    out_shape:
      Shape of the test set predictions.
    parallel:
      Boolean indicating whether we are running in the `parallel` mode, with
      dataset sharded over multiple devices.

  Returns:
    A tuple of functions `(accum_init, accum_update)`. `accum_init` accepts
    the dimension of the flattened neural network parameters (integer), and
    returns a dictionary with `'preds', 'phi', 'theta'` for keys, and their
    corresponding accumulators of matching dimension as the dictionary values.
    `accum_update` then accepts `step` (number of updates of the accumulator;
    differs from the number of sampler steps if thinning factor is not one),
    `preds` (test set predictions from the last sampler step), `phi` and `theta`
    (both also from the last sampler step), and `mmnts` which is the dictionary
    returned by the `accum_init` function.
  """
  if parallel:
    def accum_init(dim_param):
      tzeros = lambda shape: jax.device_put_replicated(
          (jnp.zeros(shape), jnp.zeros(shape)), jax.local_devices())
      return {
          'preds': tzeros(out_shape[1:]),
          'phi': tzeros(dim_param),
          'theta': tzeros(dim_param)
      }

  else:
    def accum_init(dim_param):
      tzeros = lambda shape: (jnp.zeros(shape), jnp.zeros(shape))
      return {
          'preds': tzeros(out_shape),
          'phi': tzeros(dim_param),
          'theta': tzeros(dim_param)
      }

  def _accum_update(step, val, mmnt):
    mean, var = mmnt

    # update mean
    delta1 = val - mean
    mean += delta1 / step  # delta from the previous mean

    # update variance
    delta2 = val - mean  # delta from the new mean
    var += (delta1 * delta2 - var) / step

    return mean, var

  def accum_update(step, preds, phi, theta, mmnts):
    values = {'preds': preds, 'phi': phi, 'theta': theta}
    return {k: _accum_update(step, values[k], mmnts[k]) for k in mmnts}

  if parallel:
    accum_update = jax.pmap(
        accum_update, axis_name='i', static_broadcasted_argnums=(0,))
  else:
    accum_update = jax.jit(accum_update, static_argnums=(0,))

  return accum_init, accum_update


def get_projection_fn(param_key, pred_key, n_proj, parallel, vmap_size=None):
  """
  Fixed random projections of predictive logits, theta, and phi.

  Args:
    param_key:
      PRNG key used to generate random projection directions for phi & theta.
    pred_key:
      PRNG key used to generate random projection directions for predictions.
    n_proj:
      The total number of projection dimensions.
    parallel:
      Boolean indicating whether we are running in the `parallel` mode, with
      dataset sharded over multiple devices.
    vmap_size:
      The number of projections to compute in parallel. Should divide `n_proj`;
      if it does not, `n_proj` is increased to the next higher number which is
      divisible. Pass `None` to set `vmap_size == n_proj`
  Returns:
      A function which takes in the predictions, phi, and theta, and returns
      a dictionary with the projected values. The random keys are fixed, i.e.,
      the projection directions are the same at every call.
  """
  vmap_size = n_proj if vmap_size is None else vmap_size
  n_batches = math.ceil(n_proj / vmap_size)

  def _project(param_key, pred_key, preds, phi, theta):
    param_shape, preds_shape = phi.shape, preds.shape

    def _batch_project(param_and_pred_key):
      param_key, pred_key = param_and_pred_key
      param_vec = random.normal(param_key, param_shape + (vmap_size,))
      param_vec /= onp.prod(param_shape)**0.5

      bphi, btheta = phi @ param_vec, theta @ param_vec

      preds_vec = random.normal(pred_key, preds_shape + (vmap_size,))
      preds_vec /= onp.prod(preds_shape)**0.5
      if parallel:
        preds_vec /= jax.device_count()**0.5

      blogits = jnp.einsum('xo,xoj->j', preds, preds_vec, optimize='greedy')
      if parallel:
        blogits = jax.lax.psum(blogits, axis_name='i')

      return {'phi': bphi, 'theta': btheta, 'preds': blogits}

    if n_batches > 1:
      param_and_pred_key = (random.split(param_key, n_batches),
                            random.split(pred_key, n_batches))
      out = jax.lax.map(_batch_project, param_and_pred_key)
      out = jax.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), out)

    else:
      out = _batch_project((param_key, pred_key))

    return out

  if parallel:
    _project = jax.pmap(_project, axis_name='i')  # pylint: disable=invalid-name
  else:
    _project = jax.jit(_project)  # pylint: disable=invalid-name

  def project(preds, phi, theta):  # avoids jitting the key (causes OOM)
    return _project(param_key, pred_key, preds, phi, theta)

  return project


def all_equal(x):
  return jax.lax.pmin(x, 'i') == jax.lax.pmax(x, 'i')
all_equal = jax.pmap(all_equal, 'i')


def info(*args, **kwargs):
  # print(*args, **kwargs)
  logging.info(*args, **kwargs)
  # sys.stdout.flush()


def warning(*args, **kwargs):
  # print(*args, **kwargs)
  logging.warning(*args, **kwargs)
  # sys.stdout.flush()


def error(*args, **kwargs):
  # print(*args, **kwargs)
  logging.error(*args, **kwargs)
  # sys.stdout.flush()


def _all_gather(x):
  x = jax.lax.all_gather(x, axis_name='i')
  x = jax.tree_map(lambda x: jnp.reshape(x, (-1, *(x.shape[2:]))), x)
  return x
all_gather = jax.pmap(_all_gather, axis_name='i')


def shard(x):
  def _shard(x):
    n_hosts = jax.process_count()
    host_id = jax.process_index()
    x = jnp.array_split(x, n_hosts)[host_id]
    x = jnp.array_split(x, jax.local_device_count())
    x = jax.device_put_sharded(x, jax.local_devices())
    return x

  return jax.tree_map(_shard, x)


def shard_moments(mmnts):
  return {
      'preds': shard(mmnts['preds']),
      'phi': jax.device_put_replicated(mmnts['phi'], jax.local_devices()),
      'theta': jax.device_put_replicated(mmnts['theta'], jax.local_devices())
  }

def unshard_moments(mmnts):
  mmnts = mmnts.copy()  # don't modify original dict as a side-effect
  # preds, collect from all devices.
  mmnts['preds'] = all_gather(mmnts['preds'])
  # others - replicated, take 0s element.
  mmnts = tree_take_0(mmnts)
  return mmnts

