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

**Dependencies**

-  Python 2.7+, 3.4+
-  numpy, scipy, scikit-learn
-  (for running the examples only: matplotlib)

**Installation/Setup**

Run ``pip install metric-learn`` to download and install from PyPI.

Run ``python setup.py install`` for default installation.

Run ``python setup.py test`` to run all tests.

**Usage**

For full usage examples, see the `sphinx documentation`_.

Each metric is a subclass of ``BaseMetricLearner``, which provides
default implementations for the methods ``metric``, ``transformer``, and
``transform``. Subclasses must provide an implementation for either
``metric`` or ``transformer``.

For an instance of a metric learner named ``foo`` learning from a set of
``d``-dimensional points, ``foo.metric()`` returns a ``d`` by ``d``
matrix ``M`` such that a distance between vectors ``x`` and ``y`` is
expressed ``(x-y).dot(M).dot(x-y)``.

In the same scenario, ``foo.transformer()`` returns a ``d`` by ``d``
matrix ``L`` such that a vector ``x`` can be represented in the learned
space as the vector ``x.dot(L.T)``.

For convenience, the function ``foo.transform(X)`` is provided for
converting a matrix of points (``X``) into the learned space, in which
standard Euclidean distance can be used.

**Notes**

If a recent version of the Shogun Python modular (``modshogun``) library
is available, the LMNN implementation will use the fast C++ version from
there. The two implementations differ slightly, and the C++ version is
more complete.


.. _sphinx documentation: http://all-umass.github.io/metric-learn/

.. |Travis-CI Build Status| image:: https://api.travis-ci.org/all-umass/metric-learn.svg?branch=master
   :target: https://travis-ci.org/all-umass/metric-learn
.. |License| image:: http://img.shields.io/:license-mit-blue.svg?style=flat
   :target: http://badges.mit-license.org
.. |PyPI version| image:: https://badge.fury.io/py/metric-learn.svg
   :target: http://badge.fury.io/py/metric-learn

