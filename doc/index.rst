metric-learn: Metric Learning in Python
=======================================
|License| |PyPI version|

Many approaches in machine learning require a measure of distance between data
points. Traditionally, practitioners would choose a standard distance metric
(Euclidean, City-Block, Cosine, etc.) using a priori knowledge of the domain,
which is often difficult.
In contrast, distance metric learning (or simply, metric learning) aims at
automatically constructing task-specific distance metrics from (weakly)
supervised data. The learned distance metric can then be used to perform
various tasks (e.g., k-NN classification, clustering, information retrieval).

This package contains efficient Python implementations of several popular
supervised and weakly-supervised metric learning algorithms. The API of
metric-learn is compatible with scikit-learn, allowing the use of all the
scikit-learn routines (for pipelining, model selection, etc) with metric
learning algorithms.

Documentation outline
---------------------

.. toctree::
   :maxdepth: 2

   getting_started

.. toctree::
   :maxdepth: 2

   user_guide

.. toctree::
   :maxdepth: 2

   Package Overview <metric_learn>

.. toctree::
   :maxdepth: 2

   auto_examples/index

:ref:`genindex` | :ref:`modindex` | :ref:`search`

.. |PyPI version| image:: https://badge.fury.io/py/metric-learn.svg
   :target: http://badge.fury.io/py/metric-learn
.. |License| image:: http://img.shields.io/:license-mit-blue.svg?style=flat
   :target: http://badges.mit-license.org
