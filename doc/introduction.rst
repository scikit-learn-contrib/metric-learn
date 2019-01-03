========================
What is Metric Learning?
========================

Many approaches in machine learning require a measure of distance between data
points. Traditionally, practitioners would choose a standard distance metric
(Euclidean, City-Block, Cosine, etc.) using a priori knowledge of the domain,
but it is often difficult to design metrics that are well-suited to the data
and task of interest.

Distance metric learning (or simply, metric learning) aims at
automatically constructing task-specific distance metrics from (weakly)
supervised data, in a machine learning manner. The learned distance metric can
then be used to perform various tasks (e.g., k-NN classification, clustering,
information retrieval).

In the rest of this section, we introduce the main ideas of metric learning.

Problem Setting
===============

Metric learning problems fall into two main categories depending on the type
of supervision available about the training data:

- :ref:`Supervised learning <supervised_section>`: the algorithm has access to
  a set of data points where each of them belongs to a class (label), as in a
  standard classification problem.
  Broadly speaking, the goal is to learn a distance metric that puts points
  with the same label close together while pushing away points with different
  labels.
- :ref:`Weakly supervised learning <weakly_supervised_section>`: the
  algorithm has access to a set of data points with supervision only
  at the tuple level (typically pairs, triplets, or quadruplets of
  data points). A classic example of such weaker supervision is a set of
  positive and negative pairs: in this case, the goal is to learn a distance
  metric that puts positive pairs close together and negative pairs far away.

Based on the above (weakly) supervised data, the metric learning problem is
generally formulated as an optimization problem where one seeks to find the
parameters of a distance function that minimize some objective function
measuring the agreement with the training data.

Mahalanobis Distances
=====================

In the metric-learn package, all algorithms currently implemented learn 
so-called Mahalanobis distances. Given a real-valued parameter matrix
:math:`L`, the Mahalanobis distance associated with :math:`L` is computed as
follows:

.. math:: D(x, x') = \sqrt{(Lx-Lx')^\top(Lx-Lx')}

In other words, a Mahalanobis distance is a Euclidean distance after a
linear transformation of the feature space defined by :math:`L`. Mahalanobis
distance metric learning can thus be seen as learning a new embedding space 
(representation learning). Note that when :math:`L` is the identity, one
recovers the Euclidean distance in the original feature space.

Mahalanobis distances can also be parameterized by a `positive semi-definite 
(PSD) matrix
<https://en.wikipedia.org/wiki/Positive-definite_matrix#Positive_semidefinite>`_
:math:`M`:

.. math:: D(x, x') = \sqrt{(x-x')^\top M(x-x')}

Using the fact that a PSD matrix :math:`M` can always be decomposed as
:math:`M=L^\top L`, one can see that both parameterizations are equivalent. In
practice, an algorithm may thus solve the metric problem in  :math:`M` or
:math:`L`.

.. note::

  Strictly speaking, Mahalanobis distances are called pseudo-metrics: they
  satisfy
  three of
  the `properties of a metric <https://en.wikipedia.org/wiki/Metric_
  (mathematics)>`_ (non-negativity, symmetry, triangle inequality) but not necessarily the identity of indiscernibles.

Use-cases
=========

K-NN, clustering, dimensionality reduction, retrieval

Additional Resources
====================

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

.. Methods [TO MOVE TO SUPERVISED/WEAK SECTIONS]
.. =============================================

.. Currently, each metric learning algorithm supports the following methods:

.. -  ``fit(...)``, which learns the model.
.. -  ``metric()``, which returns a Mahalanobis matrix
..    :math:`M = L^{\top}L` such that distance between vectors ``x`` and
..    ``y`` can be computed as :math:`\sqrt{\left(x-y\right)M\left(x-y\right)}`.
.. -  ``transformer_from_metric(metric)``, which returns a transformation matrix
..    :math:`L \in \mathbb{R}^{D \times d}`, which can be used to convert a
..    data matrix :math:`X \in \mathbb{R}^{n \times d}` to the
..    :math:`D`-dimensional learned metric space :math:`X L^{\top}`,
..    in which standard Euclidean distances may be used.
.. -  ``transform(X)``, which applies the aforementioned transformation.
.. - ``score_pairs(pairs)`` which returns the distance between pairs of
..   points. ``pairs`` should be a 3D array-like of pairs of shape ``(n_pairs,
..   2, n_features)``, or it can be a 2D array-like of pairs indicators of
..   shape ``(n_pairs, 2)`` (see section :ref:`preprocessor_section` for more
..   details).