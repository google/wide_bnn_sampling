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

"""Config file for the BNN experiment."""
from ml_collections import ConfigDict


def get_config():
  """Returns the default configuration as instance of ConfigDict."""
  C = ConfigDict()

  C.seed = 0

  C.n_train = 128
  C.n_test = 128
  C.noise_scale = 0.1  # variance of the Gaussian likelihood

  C.stepsize = 0.1  # sampler stepsize (applies to `hmc` and `gd`)
  C.mcmc_beta = 0.01  # damping factor for HMC/LMC (`hmc`)
  C.scale_stepsize = False  # scale stepsize by `noise_scale / sqrt(n_train)`
  C.scale_mcmc_beta = False  # scale by *effective* stepsize (`scale_stepsize`)
  C.hmc_skip_mh = False  # skip Metropolis-Hastings correction in HMC/LMC

  C.step_count = 3000  # number of steps the sampler is to take
  C.burn_in = 100  # number of burn-in steps before `step_count` starts
  C.thin = 50  # thinning factor (does not increase `step_count`!)
  C.n_proj = 100  # samples can be projected, and the result stored
  C.proj_vmap_size = 100  # if `n_proj` is too high, initialising all projection
  # dimensions at the same time may result in OOM; `proj_vmap_size` sets the
  # number of projections that are computed at the same time (batching)

  C.depth = 1  # network depth
  C.n_units = 32  # number of hidden embeddings dimensions (see `models.py`)
  C.nonlin = 'gelu'  # 'gelu', 'relu', 'erf'
  C.architecture = 'fcn'  # 'fcn', 'resnet'
  C.resid_scaling = 'progressive_one'
  # 'uniform', 'progressive_one', 'progressive_depth' (see `models.py`)

  C.w_std = 2.0**0.5  # weight scaling, pre-readout
  C.b_std = 0.1  # bias scaling, pre-readout
  C.w_std_out = 1.0  # weight scaling, readout
  C.b_std_out = 0.1  # bias scaling, readout

  C.reparam_type = 'identity'  # 'identity', 'nngp'
  C.kernel_reg_mult = -1.0  # negative value for `kernel_reg == noise_scale**2`

  C.max_n_checkpoints = 5
  C.ckpt_freq = 10000
  C.keep_every_n_checkpoint = 250_000

  C.parallel = True
  C.data_dir = ''
  C.log_freq = 100
  C.save_stats = True

  return C
