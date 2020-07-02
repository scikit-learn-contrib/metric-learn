#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup
import os
import io
import sys


CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 6)

# This check and everything above must remain compatible with Python 2.7.
if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write("""
==========================
Unsupported Python version
==========================
This version of metric-learn requires Python {}.{}, but you're trying to
install it on Python {}.{}.
This may be because you are using a version of pip that doesn't
understand the python_requires classifier. Make sure you
have pip >= 9.0 and setuptools >= 24.2, then try again:
    $ python -m pip install --upgrade pip setuptools
    $ python -m pip install django
This will install the latest version of metric-learn which works on your
version of Python. If you can't upgrade your pip (or Python), request
an older version of metric-learn:
    $ python -m pip install "metric-learn<0.6.0"
""".format(*(REQUIRED_PYTHON + CURRENT_PYTHON)))
    sys.exit(1)


version = {}
with io.open(os.path.join('metric_learn', '_version.py')) as fp:
  exec(fp.read(), version)

# Get the long description from README.md
with io.open('README.rst', encoding='utf-8') as f:
  long_description = f.read()

setup(name='metric-learn',
      version=version['__version__'],
      description='Python implementations of metric learning algorithms',
      long_description=long_description,
      python_requires='>={}.{}'.format(*REQUIRED_PYTHON),
      author=[
          'CJ Carey',
          'Yuan Tang',
          'William de Vazelhes',
          'Aurélien Bellet',
          'Nathalie Vauquier'
      ],
      author_email='ccarey@cs.umass.edu',
      url='http://github.com/scikit-learn-contrib/metric-learn',
      license='MIT',
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering'
      ],
      packages=['metric_learn'],
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn>=0.20.3',
      ],
      extras_require=dict(
          docs=['sphinx', 'shinx_rtd_theme', 'numpydoc'],
          demo=['matplotlib'],
          sdml=['skggm>=0.2.9']
      ),
      test_suite='test',
      keywords=[
          'Metric Learning',
          'Large Margin Nearest Neighbor',
          'Information Theoretic Metric Learning',
          'Sparse Determinant Metric Learning',
          'Least Squares Metric Learning',
          'Neighborhood Components Analysis',
          'Local Fisher Discriminant Analysis',
          'Relative Components Analysis',
          'Mahalanobis Metric for Clustering',
          'Metric Learning for Kernel Regression'
      ])
