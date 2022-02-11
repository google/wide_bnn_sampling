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

"""Setup the package with pip."""
import os
import setuptools


# https://packaging.python.org/guides/making-a-pypi-friendly-readme/
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()


INSTALL_REQUIRES = [
    'h5py>=3.6.0',
    'jax>=0.3.0',
    'ml_collections>=0.1.1',
    'neural-tangents>=0.3.9',
    'tensorflow>=2.8.0',
    'tensorflow-datasets>=4.5.2',
]


def _get_version() -> str:
  """Returns the package version.

  Returns:
    Version number.
  """
  return '0.1'


setuptools.setup(
    name='wide_bnn_samplng',
    version=_get_version(),
    license='Apache 2.0',
    author='Jiri Hron',
    author_email='jirihron@google.com',
    install_requires=INSTALL_REQUIRES,
    url='https://github.com/google/wide_bnn_sampling',
    download_url='https://github.com/google/wide_bnn_sampling',
    project_urls={
        'Source Code': 'https://github.com/google/wide_bnn_sampling',
        'Paper': 'TBA',
    },
    packages=setuptools.find_packages(exclude=('presentation',)),
    long_description=long_description,
    long_description_content_type='text/markdown',
    description='Simplicity of the wide Bayesian neural networks weight '
                'posterior: theory and accelerated sampling',
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Development Status :: 4 - Beta',
    ])
