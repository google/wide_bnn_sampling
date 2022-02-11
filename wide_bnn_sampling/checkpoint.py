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

"""Checkpointing."""
import os
import h5py

import jax
from jax import numpy as jnp

import tensorflow.compat.v1.gfile as gfile
import wide_bnn_sampling.utils as u



def _get_checkpoint_path(save_dir, sid):
  path = os.path.join(save_dir, 'checkpoint')
  if sid is not None:
    path = os.path.join(path, str(sid))
  return path


def load_checkpoint(save_dir):
  """
  Load from a checkpoint. Cycles through all checkpoints in the directory,
  identified by their sampler step number, and returns the most recent one
  which can be loaded (`None` if none exists, or all fail to load.)

  Args:
    save_dir:
      Directory which contains `'./checkpoint'` where the checkpoints indexed by
      sampler step are saved.

  Returns:
    A dictionary with: `sid` (step ID == zero indexed count of sampler steps);
    `n_acc` (zero indexed number of accumulator steps; agrees with `sid` when
    thinning factor is one); `key` (an array of PRNG keys used by the sampler
    to generate randomness in each step); `state` (state dictionary of the
    MCMC sampler: see `samplers.py` for details), `stats` (a dictionary of
    auxiliary statistics collected by the sampler), `samples` (a dictionary of
    *projected* phi, theta, and test predictions), and `mmnts` (a dictionary of
    accumulated 1st and 2nd order moments for phi, theta, and test predictions
    after burn-in).
  """
  if save_dir is None:
    return None

  path = _get_checkpoint_path(save_dir, None)
  if not gfile.Exists(path):
    u.warning('no path %s found' % path)
    return None

  ckpt = None
  sids = gfile.ListDirectory(path)
  for sid in sorted(map(int, sids), reverse=True):
    sid_path = f'{path}/{sid}'
    try:
      u.info(f'Loading {sid_path}...')
      ckpt = _load_checkpoint(sid_path)
    except OSError as e:
      u.error('failed to load')
      u.error(e)
    except Exception as e:
      u.error('failed to unpack')
      u.error(e)

    if ckpt is not None:
      u.info('starting from a checkpoint at sid %s', ckpt['sid'])
      return ckpt

  u.warning(f'did not find any checkpoints among {sids}')
  return None


def _load_checkpoint(path):
  ckpt = {}

  # mcmc chain state
  with gfile.Open(os.path.join(path, 'chain'), 'rb') as fsource:
    with h5py.File(fsource, 'r') as f:
      ckpt['sid'] = f['sid'][()]
      ckpt['n_acc'] = f['n_acc'][()]
      ckpt['key'] = jnp.array(f['key'][:])  # OK for PRNGKey to be a DeviceArray
      ckpt['state'] = {k: jnp.array(v[()]) for k, v in f['state'].items()}

  # samples and statistics
  for name in ('stats', 'samples'):
    with gfile.Open(os.path.join(path, name), 'rb') as fsource:
      with h5py.File(fsource, 'r') as f:
        ckpt[name] = {}
        for k, v in f.items():
          # load & convert into a list, but don't distribute across devices
          ckpt[name][k] = list(v[()])

  # moment accumulators
  mmnts = {}
  for l in ('preds', 'phi', 'theta'):
    with gfile.Open(os.path.join(path, f'{l}_mmnts'), 'rb') as fsource:
      with h5py.File(fsource, 'r') as f:
        mmnts[l] = (jnp.array(f['mean'][:]), jnp.array(f['var'][:]))
  ckpt['mmnts'] = mmnts

  return ckpt


def save_checkpoint(
    key, state, mmnts, stats, samples, sid, n_acc, save_dir, parallel,
    max_n_checkpoints, keep_every_n_checkpoint, compression=None):
  """
  Save a checkpoint. If more than `max_n_checkpoints` has been already saved,
  the oldest ones are automatically deleted.

  Args:
    key:
      Array of PRNG keys used to generate randomness in each sampler step.
    state:
      State dictionary of the MCMC sampler: see `samplers.py` for details.
    mmnts:
      Dictionary of accumulated first and second order moments for phi, theta,
      and test predictions after burn-in.
    stats:
      Dictionary of auxiliary statistics collected by the sampler.
    samples:
      Dictionary of *projected* phi, theta, and test predictions.
    sid:
      Step ID == zero indexed count of sampler steps.
    n_acc:
      Zero indexed number of accumulator steps; agrees with `sid` when the
      thinning factor is one.
    save_dir:
      Directory which contains `'./checkpoint'` where the checkpoints indexed by
      sampler step are saved.
    parallel:
      Boolean indicating whether we are running in the `parallel` mode, with
      dataset sharded over multiple devices.
    max_n_checkpoints:
      Number of past checkpoints to keep. Aimed to prevent issues with
      corrupted checkpoints due to pre-emption and OS errors.
    keep_every_n_checkpoint:
      Optionally keep checkpoints with a fixed frequency. Pass `None` to switch
      off this functionality.
    compression:
      Argument for `h5py`'s `create_dataset` method.
  """
  is_float_or_array = lambda p: isinstance(p, (float, jnp.ndarray))
  if not all(list(jax.tree_map(is_float_or_array, state).values())):
    raise ValueError('state values must be `np.ndarray` or `float`')

  if parallel:
    key = u.tree_take_0(key)
    state = u.tree_take_0(state)
    mmnts = u.unshard_moments(mmnts)

  if jax.process_index() == 0:
    # if no checkpoint base directory exists, create it
    base_path = _get_checkpoint_path(save_dir, None)
    if not gfile.Exists(base_path):
      gfile.MakeDirs(base_path)

    path = _get_checkpoint_path(save_dir, sid)
    if not os.path.exists(path):
      os.makedirs(path)

    with h5py.File(os.path.join(path, 'chain'), 'w') as f:
      f.create_dataset('sid', data=sid)
      f.create_dataset('n_acc', data=n_acc)
      f.create_dataset('key', data=key, compression=compression)

      state_grp = f.create_group('state')
      for k, v in state.items():
        state_grp.create_dataset(k, data=v, compression=compression)

    with h5py.File(os.path.join(path, 'stats'), 'w') as f:
      for k, v in stats.items():
        f.create_dataset(k, data=v, compression=compression)

    with h5py.File(os.path.join(path, 'samples'), 'w') as f:
      for k, v in samples.items():
        f.create_dataset(k, data=v, compression=compression)

    for l, m in mmnts.items():
      with h5py.File(os.path.join(path, f'{l}_mmnts'), 'w') as f:
        f.create_dataset('mean', data=m[0], compression=compression)
        f.create_dataset('var', data=m[1], compression=compression)

    u.info('SAVED checkpoint at sid %s', sid)

    # after saving, check if we can delete an old checkpoint
    sids = list(map(int, gfile.ListDirectory(base_path)))
    if keep_every_n_checkpoint is not None:
      sids = [s for s in sids if s % keep_every_n_checkpoint != 0]
    if len(sids) > max_n_checkpoints:
      min_sid = min(sids)
      min_path = _get_checkpoint_path(save_dir, min_sid)
      u.info('Deleting checkpoint at sid %s', min_sid)
      if gfile.Exists(min_path):
        gfile.DeleteRecursively(path)
      u.info('DELETED checkpoint at sid %s', min_sid)
