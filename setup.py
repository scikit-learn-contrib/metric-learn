#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup

version = "0.1.0"
setup(name='metric_learn',
      version=version,
      description='Python implementations of metric learning algorithms',
      author='CJ Carey',
      author_email='ccarey@cs.umass.edu',
      url='http://github.com/all-umass/metric_learn',
      license='MIT',
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python',
      ],
      packages=['metric_learn'],
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn'
      ],
      test_suite='test'
      )
