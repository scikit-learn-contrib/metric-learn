============
Introduction
============

Distance metrics are widely used in the machine learning literature.
Traditionally, practitioners would choose a standard distance metric
(Euclidean, City-Block, Cosine, etc.) using a priori knowledge of
the domain.
Distance metric learning (or simply, metric learning) is the sub-field of
machine learning dedicated to automatically constructing optimal distance
metrics.

This package contains a efficient Python implementations of several popular
metric learning algorithms, compatible with scikit-learn. This allows to use
all the scikit-learn routines for pipelining and model selection for
metric learning algorithms.


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