|Travis-CI Build Status| |License| |PyPI version|

metric-learn
=============

Metric Learning algorithms in Python.

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
-  numpy, scipy, scikit-learn, and skggm (commit `a0ed406 <https://github.com/skggm/skggm/commit/a0ed406586c4364ea3297a658f415e13b5cbdaf8>`_ for `SDML`
-  (for running the examples only: matplotlib)

**Installation/Setup**

Run ``pip install metric-learn`` to download and install from PyPI.

Run ``python setup.py install`` for default installation.

Run ``pytest test`` to run all tests (you will need to have the ``pytest``
package installed).

**Usage**

See the `sphinx documentation`_ for full documentation about installation, API, usage, and examples.

**Notes**

If a recent version of the Shogun Python modular (``modshogun``) library
is available, the LMNN implementation will use the fast C++ version from
there. The two implementations differ slightly, and the C++ version is
more complete.


.. _sphinx documentation: http://metric-learn.github.io/metric-learn/

.. |Travis-CI Build Status| image:: https://api.travis-ci.org/metric-learn/metric-learn.svg?branch=master
   :target: https://travis-ci.org/metric-learn/metric-learn
.. |License| image:: http://img.shields.io/:license-mit-blue.svg?style=flat
   :target: http://badges.mit-license.org
.. |PyPI version| image:: https://badge.fury.io/py/metric-learn.svg
   :target: http://badge.fury.io/py/metric-learn

