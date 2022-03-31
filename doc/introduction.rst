.. _intro_metric_learning:

========================
What is Metric Learning?
========================

Many approaches in machine learning require a measure of distance (or similarity)
between data points. Traditionally, practitioners would choose a standard metric
(Euclidean, City-Block, Cosine, etc.) using a priori knowledge of the
domain. However, it is often difficult to design metrics that are well-suited
to the particular data and task of interest.

Metric learning (or simply, metric learning) aims at automatically constructing
task-specific metrics from (weakly) supervised data, in a machine learning manner.
The learned metric can then be used to perform various tasks (e.g.,
k-NN classification, clustering, information retrieval).

Problem Setting
===============

Metric learning problems fall into two main categories depending on the type
of supervision available about the training data:

- :doc:`Supervised learning <supervised>`: the algorithm has access to
  a set of data points, each of them belonging to a class (label) as in a
  standard classification problem.
  Broadly speaking, the goal in this setting is to learn a metric
  that puts points with the same label close together while pushing away
  points with different labels.
- :doc:`Weakly supervised learning <weakly_supervised>`: the
  algorithm has access to a set of data points with supervision only
  at the tuple level (typically pairs, triplets, or quadruplets of
  data points). A classic example of such weaker supervision is a set of
  positive and negative pairs: in this case, the goal is to learn a
  metric that puts positive pairs close together and negative pairs far away.

Based on the above (weakly) supervised data, the metric learning problem is
generally formulated as an optimization problem where one seeks to find the
parameters of a function that optimize some objective function
measuring the agreement with the training data.

.. _mahalanobis_distances:

Mahalanobis Distances
=====================

In the metric-learn package, most algorithms currently implemented learn 
so-called Mahalanobis distances. Given a real-valued parameter matrix
:math:`L` of shape ``(num_dims, n_features)`` where ``n_features`` is the
number features describing the data, the Mahalanobis distance associated with
:math:`L` is defined as follows:

.. math:: D(x, x') = \sqrt{(Lx-Lx')^\top(Lx-Lx')}

In other words, a Mahalanobis distance is a Euclidean distance after a
linear transformation of the feature space defined by :math:`L` (taking
:math:`L` to be the identity matrix recovers the standard Euclidean distance).
Mahalanobis distance metric learning can thus be seen as learning a new
embedding space of dimension ``num_dims``. Note that when ``num_dims`` is
smaller than ``n_features``, this achieves dimensionality reduction.

Strictly speaking, Mahalanobis distances are "pseudo-metrics": they satisfy
three of the `properties of a metric <https://en.wikipedia.org/wiki/Metric_
(mathematics)>`_ (non-negativity, symmetry, triangle inequality) but not
necessarily the identity of indiscernibles.

.. note::

  Mahalanobis distances can also be parameterized by a `positive semi-definite 
  (PSD) matrix
  <https://en.wikipedia.org/wiki/Positive-definite_matrix#Positive_semidefinite>`_
  :math:`M`:

  .. math:: D(x, x') = \sqrt{(x-x')^\top M(x-x')}

  Using the fact that a PSD matrix :math:`M` can always be decomposed as
  :math:`M=L^\top L` for some  :math:`L`, one can show that both
  parameterizations are equivalent. In practice, an algorithm may thus solve
  the metric learning problem with respect to either :math:`M` or :math:`L`.

.. _bilinear_similarity:

Bilinear Similarities
=====================

Some algorithms in the package learn bilinear similarity functions. These
similarity functions are not pseudo-distances: they simply output real values
such that the larger the similarity value, the more similar the two examples.
Given a real-valued parameter matrix :math:`W` of shape
``(n_features, n_features)`` where ``n_features`` is the number features
describing the data, the bilinear similarity associated with :math:`W` is
defined as follows:

.. math:: S_W(x, x') = x^\top W x'

The matrix :math:`W` is not required to be positive semi-definite (PSD) or
even symmetric, so the distance properties (nonnegativity, identity of
indiscernibles, symmetry and triangle inequality) do not hold in general.

This allows some algorithms to optimize :math:`S_W` in an online manner using a
simple and efficient procedure, and thus can be applied to problems with
millions of training instances and achieves state-of-the-art performance
on an image search task using :math:`k`-NN.

The absence of PSD constraint can enable the design of more efficient
algorithms. It is also relevant in applications where the underlying notion
of similarity does not satisfy the triangle inequality, as known to be the
case for visual judgments.

.. _use_cases:

Use-cases
=========

There are many use-cases for metric learning. We list here a few popular
examples (for code illustrating some of these use-cases, see the
:doc:`examples <auto_examples/index>` section of the documentation):

- `Nearest neighbors models
  <https://scikit-learn.org/stable/modules/neighbors.html>`_: the learned
  metric can be used to improve nearest neighbors learning models for
  classification, regression, anomaly detection...
- `Clustering <https://scikit-learn.org/stable/modules/clustering.html>`_:
  metric learning provides a way to bias the clusters found by algorithms like
  K-Means towards the intended semantics.
- Information retrieval: the learned metric can be used to retrieve the
  elements of a database that are semantically closest to a query element.
- Dimensionality reduction: metric learning may be seen as a way to reduce the
  data dimension in a (weakly) supervised setting.
- More generally with Mahalanobis distances, the learned transformation :math:`L`
  can be used to project the data into a new embedding space before feeding it
  into another machine learning algorithm.

The API of metric-learn is compatible with `scikit-learn
<https://scikit-learn.org/>`_, the leading library for machine
learning in Python. This allows to easily pipeline metric learners with other
scikit-learn estimators to realize the above use-cases, to perform joint
hyperparameter tuning, etc.

Further reading
===============

For more information about metric learning and its applications, one can refer
to the following resources:

- **Tutorial:** `Similarity and Distance Metric Learning with Applications to
  Computer Vision
  <http://researchers.lille.inria.fr/abellet/talks/metric_learning_tutorial_ECML_PKDD.pdf>`_ (2015)
- **Surveys:** `A Survey on Metric Learning for Feature Vectors and Structured
  Data <https://arxiv.org/pdf/1306.6709.pdf>`_ (2013), `Metric Learning: A
  Survey <http://dx.doi.org/10.1561/2200000019>`_ (2012)
- **Book:** `Metric Learning
  <http://dx.doi.org/10.2200/S00626ED1V01Y201501AIM030>`_ (2015)
