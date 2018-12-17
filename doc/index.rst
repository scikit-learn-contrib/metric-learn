metric-learn: Metric Learning in Python
=======================================
|License| |PyPI version|

Distance metrics are widely used in the machine learning literature.
Traditionally, practicioners would choose a standard distance metric
(Euclidean, City-Block, Cosine, etc.) using a priori knowledge of
the domain.
Distance metric learning (or simply, metric learning) is the sub-field of
machine learning dedicated to automatically constructing optimal distance
metrics.

This package contains efficient Python implementations of several popular
metric learning algorithms.

Supervised Algorithms
---------------------
Supervised metric learning algorithms take as inputs points `X` and target
labels `y`, and learn a distance matrix that make points from the same class
(for classification) or with close target value (for regression) close to
each other, and points from different classes or with distant target values
far away from each other.

- `Covariance <metric_learn.covariance.html>`_
- `LMNN <metric_learn.lmnn.html>`_
- `NCA <metric_learn.nca.html>`_
- `LFDA <metric_learn.covariance.html>`_
- `MLKR <metric_learn.mlkr.html>`_

Weakly-Supervised Algorithms
--------------------------
Weakly supervised algorithms work on weaker information about the data points
than supervised algorithms. Rather than labeled points, they take as input
similarity judgments on tuples of data points, for instance pairs of similar
and dissimilar points. Refer to the documentation of each algorithm for its
particular form of input data.

- `ITML <metric_learn.itml.html>`_
- `LSML <metric_learn.lsml.html>`_
- `SDML <metric_learn.sdml.html>`_
- `RCA <metric_learn.rca.html>`_
- `MMC <metric_learn.mmc.html>`_

Note that each weakly-supervised algorithm has a supervised version of the form
`*_Supervised` where similarity constraints are generated from
the labels information and passed to the underlying algorithm.

Each metric learning algorithm supports the following methods:

-  ``fit(...)``, which learns the model.
-  ``transformer()``, which returns a transformation matrix
   :math:`L \in \mathbb{R}^{D \times d}`, which can be used to convert a
   data matrix :math:`X \in \mathbb{R}^{n \times d}` to the
   :math:`D`-dimensional learned metric space :math:`X L^{\top}`,
   in which standard Euclidean distances may be used.
-  ``transform(X)``, which applies the aforementioned transformation.
-  ``metric()``, which returns a Mahalanobis matrix
   :math:`M = L^{\top}L` such that distance between vectors ``x`` and
   ``y`` can be computed as :math:`\left(x-y\right)M\left(x-y\right)`.


Installation and Setup
======================

Run ``pip install metric-learn`` to download and install from PyPI.

Alternately, download the source repository and run:

-  ``python setup.py install`` for default installation.
-  ``python setup.py test`` to run all tests.

**Dependencies**

-  Python 2.7+, 3.4+
-  numpy, scipy, scikit-learn
-  (for running the examples only: matplotlib)

**Notes**

If a recent version of the Shogun Python modular (``modshogun``) library
is available, the LMNN implementation will use the fast C++ version from
there. The two implementations differ slightly, and the C++ version is
more complete.

Navigation
----------


.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   user_guide

:ref:`genindex` | :ref:`modindex` | :ref:`search`

.. toctree::
   :maxdepth: 4
   :hidden:

   Package Overview <metric_learn>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorial - Examples

   auto_examples/index


.. |PyPI version| image:: https://badge.fury.io/py/metric-learn.svg
   :target: http://badge.fury.io/py/metric-learn
.. |License| image:: http://img.shields.io/:license-mit-blue.svg?style=flat
   :target: http://badges.mit-license.org
