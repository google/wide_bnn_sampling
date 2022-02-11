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

"""Experiment measurements utility."""
import tensorflow.compat.v1.gfile as gfile
import wide_bnn_sampling.utils as u


class Measurements(object):
  """Utility class for logging experiment measurements."""

  def __init__(self, store_dir: str, init_store_dir: str):

    """Creates a manager for both XM and local disk measurements.

    Args:
      store_dir:
        Directory where the statistics will be stored.
      init_store_dir:
        Directory from which to initialise. Can be used to start an experiment
        from a checkpoint created in a different run. Note that the supplied
        directory must contain work units with IDs which match that of the
        new experiment, i.e., this functionality is useful when restarting the
        same experiment with modified hyperparameters, but does not allow
        more advanced initialisation of units based on an experiment with
        different structure.
    """
    # preprocessing of supplied initial checkpoint path
    if init_store_dir is not None:
      if (gfile.Exists(init_store_dir) and
          gfile.IsDirectory(init_store_dir) and
          'checkpoint' in gfile.ListDirectory(init_store_dir)):
        u.info('setting `init_save_dir` to "%s"', init_store_dir)
      else:
        init_store_dir = None
        u.warning('invalid `init_save_dir` "%s"', init_store_dir)
    else:
      u.info('no `init_save_dir` -> starting from scratch')

    self.save_dir = store_dir
    self.init_save_dir = init_store_dir


  def log(self, label: str, value: float, step: int):
    s = f'[step {step}]: \t{label} \t= {value}'
    u.info(s)
