metric-learn: Metric Learning in Python  
=====================================
|License| |PyPI version|

Distance metric is widely used in the machine learning literature. We used to choose a distance metric according to a priori (Euclidean Distance , L1 Distance, etc.) or according to the result of cross validation within small class of functions (e.g. choosing order of polynomial for a kernel). Actually, with priori knowledge of the data, we could learn a more suitable distance metric with metric learning techniques. metric-learn  contains implementations of the state-of-the-art algorithms for metric learning. These metric learning methods are widely applied in feature extraction, dimensionality reduction, clustering, classification, information retrieval, and computer vision problems.

**Algorithms**

-  Large Margin Nearest Neighbor (LMNN)
-  Information Theoretic Metric Learning (ITML)
-  Sparse Determinant Metric Learning (SDML)
-  Least Squares Metric Learning (LSML)
-  Neighborhood Components Analysis (NCA)
-  Local Fisher Discriminant Analysis (LFDA)
-  Relative Components Analysis (RCA)

**Dependencies**

-  Python 2.6+
-  numpy, scipy, scikit-learn
-  (for running the examples only: matplotlib)

**Installation/Setup**

Run ``pip install metric-learn`` to download and install from PyPI.

Run ``python setup.py install`` for default installation.

Run ``python setup.py test`` to run all tests.

**Usage**

For full usage examples, see the ``test`` and ``examples`` directories.

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
space as the vector ``L.dot(x)``.

For convenience, the function ``foo.transform(X)`` is provided for
converting a matrix of points (``X``) into the learned space, in which
standard Euclidean distance can be used.

**Notes**

If a recent version of the Shogun Python modular (``modshogun``) library
is available, the LMNN implementation will use the fast C++ version from
there. The two implementations differ slightly, and the C++ version is
more complete.

**TODO**

- implement the rest of the methods on `this site`_

.. _this site: http://www.cs.cmu.edu/~liuy/distlearn.htm

.. |PyPI version| image:: https://badge.fury.io/py/metric-learn.svg
   :target: http://badge.fury.io/py/metric-learn
.. |License| image:: http://img.shields.io/:license-mit-blue.svg?style=flat
   :target: http://badges.mit-license.org