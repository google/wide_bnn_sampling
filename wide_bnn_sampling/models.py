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

"""Models."""
from functools import partial
import itertools

import neural_tangents as nt


@nt.stax.layer
@nt.stax.supports_masking(remask_kernel=False)
def Rescaling(scaling) -> nt.stax.InternalLayer:  # pylint: disable=invalid-name
  """Scale layer inputs by a fixed constant."""
  def init_fn(rng, input_shape):
    return input_shape, ()

  def apply_fn(params, inputs, **kwargs):
    return inputs * scaling

  def kernel_fn(k, **kwargs):
    return k * scaling

  return init_fn, apply_fn, kernel_fn


# pylint: disable=invalid-name
def WideResnetBlock(n_units, nonlin, strides=(1, 1), w_std=1.0, b_std=None,
                    downsample=False, scaling=1 / (2**0.5)):
  """A single skip-connection block for a normaliser-free Wide ResNet."""
  conv = partial(nt.stax.Conv, padding='SAME', W_std=w_std, b_std=b_std)
  if callable(scaling):
    resid_weight, skip_weight = scaling()
  else:
    resid_weight = skip_weight = scaling

  main = nt.stax.serial(
      nonlin(), conv(n_units, (3, 3), strides),
      nonlin(), conv(n_units, (3, 3), (1, 1)),
      Rescaling(resid_weight)
  )
  shortcut = nt.stax.serial(
      nt.stax.Identity() if not downsample else nt.stax.Conv(
          n_units, (3, 3), strides, padding='SAME', W_std=1.0, b_std=b_std),
      Rescaling(skip_weight)
  )

  return nt.stax.serial(
      nt.stax.FanOut(2),
      nt.stax.parallel(main, shortcut),
      nt.stax.FanInSum()
  )


def WideResnetGroup(  # pylint: disable=invalid-name
    n_blocks, n_units, nonlin, scaling, w_std, b_std, strides=(1, 1)):
  """A group or normaliser-free Wide ResNet blocks with initial downsampling."""
  block = partial(
      WideResnetBlock, nonlin=nonlin, scaling=scaling, w_std=w_std, b_std=b_std)

  layers = [block(n_units, strides=strides, downsample=True)]
  layers += [block(n_units, strides=(1, 1)) for _ in range(n_blocks - 1)]

  return nt.stax.serial(*layers)


def resnet(n_units, depth, w_std, b_std, nonlin_init, resid_scaling='uniform'):
  """
  A single skip-connection block for a normaliser-free Wide ResNet.

  Args:
    n_units:
      Integer. Sets the number of embedding dimensions (channels) in the
      first group of residual layers, i.e., the *narrowest* one. The number
      of embedding dimensions doubles in each group, i.e., the widest layer
      will have 4x as many embedding dimensions (3 groups -> 2 doublings).
    depth:
      Integer. Set the number of layers. Must satisfy `(depth - 2) % 6 == 0`.
    w_std:
      Float. Initial (prior) variance of all but the readout weights.
    b_std:
      Float. Initial (prior) variance of all but the readout biases.
    nonlin_init:
      Callable. Should return a `neural_tangents` activation, represented by
      the usual `(init_fn, kernel_fn, apply_fn)` tuple, and take no arguments.
    resid_scaling:
      String. Determines how the contributions from the residual and the skip
      connection are weighted. Except for `None` (both weighted by one), and
      `'uniform'` (both weighted by `1 / sqrt(2)`, we support the scaling from
      Shao et al. (2020) "Is normalization indispensable for training deep
      neural network?" with two choices for their hyperparameter `c`: passing
      `'progressive_one'` will result in `c == 1`, whereas `'progressive_depth'`
      sets `c == (depth - 2) // 3` which is half the number of skip connections.
  Returns:
    A list of the layers with a single-layer CNN stem, followed by 3 groups of
    skip-connection blocks, an `8x8` Average Pooling, and Flattening, but no
    readout layer! Due to the dowwnsampling at the beginning of each block,
    the final spatial dimensions are `orig_dimension / 4`, so `8 x 8` on
    cifar-10 (32x32 inputs).
  """
  if (depth - 2) % 6 != 0:
    raise ValueError('depth must be equal to `6n + 2` for some integer `n`')

  n_blocks = (depth - 2) // 6  # 3 groups each with 2 layers per block
  if resid_scaling == 'uniform':
    scaling = 1 / 2**0.5
  elif resid_scaling.startswith('progressive'):
    if resid_scaling == 'progressive_one':
      c = 1.0
    elif resid_scaling == 'progressive_depth':
      c = 3 * n_blocks / 2  # no. of skip connections / 2
    else:
      raise NotImplementedError(resid_scaling)
    resid_counter = itertools.count(1)
    def scaling():
      k = next(resid_counter)
      weight_resid = 1 / (k + c) ** 0.5
      return weight_resid, (1 - weight_resid**2)**0.5
  else:
    raise NotImplementedError(
        f'unknown residual scaling strategy `{resid_scaling}`')

  group = partial(WideResnetGroup, nonlin=nonlin_init, scaling=scaling,
                  w_std=w_std, b_std=b_std)

  return [
      nt.stax.Conv(n_units, (3, 3), padding='SAME', W_std=1.0, b_std=b_std),
      group(n_blocks, n_units),
      group(n_blocks, n_units * 2, strides=(2, 2)),
      group(n_blocks, n_units * 4, strides=(2, 2)),
      nt.stax.AvgPool((8, 8)), nt.stax.Flatten(),
  ]


def get_network(d_out, C):
  """Construct a model of given architecture using `neural_tangents`.

  Args:
    d_out: Integer. Number of outputs.
    C:
      Config. Instance of `ml_collections.config_dict`. Should contain: `depth`
        (no. of hidden layers), `d_out` (output dimension), `w_std` (initial
        (prior) variance of all but the readout weights), `b_std` (initial
        (prior) variance of all but the readout biases), `w_std_out` (initial
        (prior) variance of the readout weights), `b_std_out` (initial (prior)
        variance of the readout biases), `architecture` (string specifying
        network architecture), `nonlin` (string specifying nonlinearity), and
        `resid_scaling` (string specifying type of residual connection scaling).

  Returns:
    A `neural_tangents` model, i.e., a triple `init_fn, apply_fn, kernel_fn`,
    and an `embed_fn`, which computes the top-layer embeddings scaled by the
    square root of their dimension, as the 4th output.
  """
  b_std_out = C.b_std if C.b_std_out is None else C.b_std_out

  if C.nonlin == 'erf':
    nonlin_init = nt.stax.Erf
  elif C.nonlin == 'relu':
    nonlin_init = nt.stax.Relu
  elif C.nonlin == 'gelu':
    nonlin_init = nt.stax.Gelu
  else:
    raise NotImplementedError(f'nonlinearity "{C.nonlin}" not implemented')

  # pre-readout layers
  if C.architecture == 'fcn':
    layers = []
    layer_init = partial(nt.stax.Dense, parameterization='ntk')
    for _ in range(C.depth):
      layers.append(layer_init(C.n_units, W_std=C.w_std, b_std=C.b_std))
      layers.append(nonlin_init())
  elif C.architecture == 'resnet':
    layers = resnet(
        n_units=C.n_units, depth=C.depth, w_std=C.w_std, b_std=C.b_std,
        nonlin_init=nonlin_init, resid_scaling=C.resid_scaling)
  else:
    raise NotImplementedError(
        f'architecture "{C.architecture}" not implemented')

  # function computing top-layer embeddings scaled by the sqrt of their dim
  if C.depth > 0:
    _, _embed_fn, _ = nt.stax.serial(*layers)
    def embed_fn(params, inputs, **kwargs):
      emb = _embed_fn(params, inputs, **kwargs)
      return emb / (emb.shape[-1]**0.5)
  else:
    raise ValueError(f'depth must be greater than zero; was {C.depth}')

  # readout layer
  layers.append(nt.stax.Dense(
      d_out, W_std=C.w_std_out, b_std=b_std_out, parameterization='ntk'))

  init_fn, apply_fn, kernel_fn = nt.stax.serial(*layers)
  return init_fn, apply_fn, kernel_fn, embed_fn

