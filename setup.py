#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup
import os
import io

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
      author=['CJ Carey', 'Yuan Tang'],
      author_email='ccarey@cs.umass.edu',
      url='http://github.com/metric-learn/metric-learn',
      license='MIT',
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering'
      ],
      packages=['metric_learn'],
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn',
          'six'
      ],
      extras_require=dict(
          docs=['sphinx', 'shinx_rtd_theme', 'numpydoc'],
          demo=['matplotlib'],
          sdml=['skggm']
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
