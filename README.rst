|Travis-CI Build Status| |License| |PyPI version| |Code coverage|

metric-learn
=============

Metric Learning algorithms in Python. 

As part of `scikit-learn-contrib <https://github.com/scikit-learn-contrib>`_, the API of metric-learn is compatible with `scikit-learn <http://scikit-learn.org/stable/>`_, the leading library for machine learning in Python. This allows to use of all the scikit-learn routines (for pipelining, model selection, etc) with metric learning algorithms.

**Algorithms**

-  Large Margin Nearest Neighbor (LMNN)
-  Information Theoretic Metric Learning (ITML)
-  Sparse Determinant Metric Learning (SDML)
-  Least Squares Metric Learning (LSML)
-  Neighborhood Components Analysis (NCA)
-  Local Fisher Discriminant Analysis (LFDA)
-  Relative Components Analysis (RCA)
-  Metric Learning for Kernel Regression (MLKR)
-  Mahalanobis Metric for Clustering (MMC)

**Dependencies**

-  Python 2.7+, 3.4+
-  numpy, scipy, scikit-learn>=0.20.3

**Optional dependencies**

- For SDML, using skggm will allow the algorithm to solve problematic cases
  (install from commit `a0ed406 <https://github.com/skggm/skggm/commit/a0ed406586c4364ea3297a658f415e13b5cbdaf8>`_).
-  For running the examples only: matplotlib

**Installation/Setup**

Run ``pip install metric-learn`` to download and install from PyPI.

Run ``python setup.py install`` for default installation.

Run ``pytest test`` to run all tests (you will need to have the ``pytest``
package installed).

**Usage**

See the `sphinx documentation`_ for full documentation about installation, API, usage, and examples.


.. _sphinx documentation: http://contrib.scikit-learn.org/metric-learn/

.. |Travis-CI Build Status| image:: https://api.travis-ci.org/scikit-learn-contrib/metric-learn.svg?branch=master
   :target: https://travis-ci.org/scikit-learn-contrib/metric-learn
.. |License| image:: http://img.shields.io/:license-mit-blue.svg?style=flat
   :target: http://badges.mit-license.org
.. |PyPI version| image:: https://badge.fury.io/py/metric-learn.svg
   :target: http://badge.fury.io/py/metric-learn
.. |Code coverage| image:: https://codecov.io/gh/scikit-learn-contrib/metric-learn/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/scikit-learn-contrib/metric-learn
