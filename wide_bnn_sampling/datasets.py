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

"""Datasets."""
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def _partial_flatten_and_normalize(x):
  """Flatten all but the first dimension of an `np.ndarray`."""
  x = np.reshape(x, (x.shape[0], -1))
  return (x - np.mean(x)) / np.std(x)


def _one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)


def get_dataset(
    name,
    n_train=None,
    n_test=None,
    permute_train=False,
    do_flatten_and_normalize=True,
    data_dir=None,
    input_key='image'):
  """Download, parse and process a dataset to unit scale and one-hot labels."""
  tf.config.set_visible_devices([], 'GPU')

  ds_builder = tfds.builder(name)

  ds_train, ds_test = tfds.as_numpy(
      tfds.load(
          name + (':3.*.*' if name != 'imdb_reviews' else ''),
          split=['train' + ('[:%d]' % n_train if n_train is not None else ''),
                 'test' + ('[:%d]' % n_test if n_test is not None else '')],
          batch_size=-1,
          as_dataset_kwargs={'shuffle_files': False},
          data_dir=data_dir))

  train_images, train_labels, test_images, test_labels = (ds_train[input_key],
                                                          ds_train['label'],
                                                          ds_test[input_key],
                                                          ds_test['label'])

  if do_flatten_and_normalize:
    train_images = _partial_flatten_and_normalize(train_images)
    test_images = _partial_flatten_and_normalize(test_images)

  num_classes = ds_builder.info.features['label'].num_classes
  train_labels = _one_hot(train_labels, num_classes)
  test_labels = _one_hot(test_labels, num_classes)

  if permute_train:
    perm = np.random.RandomState(0).permutation(train_images.shape[0])
    train_images = train_images[perm]
    train_labels = train_labels[perm]

  return train_images, train_labels, test_images, test_labels


def cifar10_tfds(n_train,
                 n_test,
                 flatten=False,
                 regression=False,
                 data_dir=None):
  """Load the cifar-10 dataset using the `neural_tangents` library.

  Args:
    n_train: Integer. Number of training points.
    n_test: Integer. Number of test points.
    flatten: Boolean. Indicates if the images should be flattened.
    regression: Boolean. Inidicates if the labels should be encoded as one-hot
      vectors (`False`), or one-hot vectors shifted by `-0.1` to make them
      centred (`True`).
    data_dir: Directory from which to load `cifar-10`. (Optional.)

  Returns:
    Tuple `(x_train, y_train), (x_test, y_test)`.
  """
  data_dir = None if data_dir == '' else data_dir  # pylint: disable=g-explicit-bool-comparison

  x_train, y_train, x_test, y_test = get_dataset(
      'cifar10',
      n_train=n_train,
      n_test=n_test,
      permute_train=False,
      do_flatten_and_normalize=False,
      data_dir=data_dir)

  # standardise
  x_mean, x_std = jnp.mean(x_train), jnp.std(x_train)
  x_train, x_test = ((x - x_mean) / x_std for x in (x_train, x_test))

  if flatten:  # flatten the spatial dimensions
    x_train, x_test = (x.reshape((len(x), -1)) for x in (x_train, x_test))
  if regression:  # shift the labels for regression
    y_train, y_test = (y - 0.1 for y in (y_train, y_test))

  return (x_train, y_train), (x_test, y_test)
