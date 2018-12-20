.. _weakly_supervised_section:

=================================
Weakly Supervised Metric Learning
=================================

Weakly supervised algorithms work on weaker information about the data points
than supervised algorithms. Rather than labeled points, they take as input
similarity judgments on tuples of data points, for instance pairs of similar
and dissimilar points. Refer to the documentation of each algorithm for its
particular form of input data.


Input data
==========

In the following paragraph we talk about tuples for sake of generality. These
can be pairs, triplets, quadruplets etc, depending on the particular metric
learning algorithm we use.

Basic form
----------
Every weakly supervised algorithm will take as input tuples of points, and if
needed labels for theses tuples.


The `tuples` argument is the first argument of every method (like the X
argument for classical algorithms in scikit-learn). The second argument is the
label of the tuple: its semantic depends on the algorithm used. For instance
for pairs learners ``y`` is a label indicating whether the pair is of similar
samples or dissimilar samples.

Then one can fit a Weakly Supervised Metric Learner on this tuple, like this:

>>> my_algo.fit(tuples, y)

Like in a classical setting we split the points ``X`` between train and test,
here we split the ``tuples`` between train and test.

>>> from sklearn.model_selection import train_test_split
>>> pairs_train, pairs_test, y_train, y_test = train_test_split(pairs, y)

These are two data structures that can be used to represent tuple in metric
learn:

3D array of tuples
------------------

The most intuitive way to represent tuples is to provide the algorithm with a
3D array-like of tuples of shape ``(n_tuples, t, n_features)``, where
``n_tuples`` is the number of tuples, ``tuple_size`` is the number of elements
in a tuple (2 for pairs, 3 for triplets for instance), and ``n_features`` is
the number of features of each point.

.. topic:: Example:
   Here is an artificial dataset of 4 pairs of 2 points of 3 features each:

>>> import numpy as np
>>> tuples = np.array([[[-0.12, -1.21, -0.20],
>>>                     [+0.05, -0.19, -0.05]],
>>>
>>>                    [[-2.16, +0.11, -0.02],
>>>                     [+1.58, +0.16, +0.93]],
>>>
>>>                    [[+1.58, +0.16, +0.93 ],  # same as tuples[1, 1, :]
>>>                     [+0.89, -0.34, +2.41]],
>>>
>>>                    [[-0.12, -1.21, -0.20 ],  # same as tuples[0, 0, :]
>>>                     [-2.16, +0.11, -0.02]]])  # same as tuples[1, 0, :]
>>> y = np.array([-1, 1, 1, -1])

.. warning:: This way of specifying pairs is not recommended for a large number
   of tuples, as it is redundant (see the comments in the example) and hence
   takes a lot of memory. Indeed each feature vector of a point will be
   replicated as many times as a point is involved in a tuple. The second way
   to specify pairs is more efficient


2D array of indicators + preprocessor
-------------------------------------

Instead of forming each point in each tuple, a more efficient representation
would be to keep the dataset of points ``X`` aside, and just represent tuples
as a collection of tuples of *indices* from the points in ``X``. Since we loose
the feature dimension there, the resulting array is 2D.

.. topic:: Example: An equivalent representation of the above pairs would be:

>>> X = np.array([[-0.12, -1.21, -0.20],
>>>               [+0.05, -0.19, -0.05],
>>>               [-2.16, +0.11, -0.02],
>>>               [+1.58, +0.16, +0.93],
>>>               [+0.89, -0.34, +2.41]])
>>>
>>> tuples_indices = np.array([[0, 1],
>>>                            [2, 3],
>>>                            [3, 4],
>>>                            [0, 2]])
>>> y = np.array([-1, 1, 1, -1])

In order to fit metric learning algorithms with this type of input, we need to
give the original dataset of points ``X`` to the estimator so that it knows
the points the indices refer to. We do this when initializing the estimator,
through the argument `preprocessor`.

.. topic:: Example:

>>> from metric_learn import MMC
>>> mmc = MMC(preprocessor=X)
>>> mmc.fit(pairs_indice, y)


.. note::

   Instead of an array-like, you can give a callable in the argument
   ``preprocessor``, which will go fetch and form the tuples. This allows to
   give more general indicators than just indices from an array (for instance
   paths in the filesystem, name of records in a database etc...) See section
   :ref:`preprocessor_section` for more details on how to use the preprocessor.


Scikit-learn compatibility
==========================

Weakly supervised estimators are compatible with scikit-learn routines for
model selection (grid-search, cross-validation etc). See the scoring section
for more details on the scoring used in the case of Weakly Supervised
Metric Learning.

.. topic:: Example

>>> from metric_learn import MMC
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import cross_val_score
>>> rng = np.random.RandomState(42)
>>> X, _ = load_iris(return_X_y=True)
>>> # let's sample 30 random pairs and labels of pairs
>>> pairs_indices = rng.randint(X.shape[0], size=(30, 2))
>>> y = rng.randint(2, size=30)
>>> mmc = MMC(preprocessor=X)
>>> cross_val_score(mmc, pairs_indices, y)

Scoring
=======

Some default scoring are implemented in metric-learn, depending on the kind of
tuples you're working with (pairs, triplets...). See the docstring of the
`score` method of the estimator you use.


Algorithms
==================

ITML
----

Information Theoretic Metric Learning, Davis et al., ICML 2007

`ITML` minimizes the differential relative entropy between two multivariate
Gaussians under constraints on the distance function, which can be formulated
into a Bregman optimization problem by minimizing the LogDet divergence subject
to linear constraints. This algorithm can handle a wide variety of constraints
and can optionally incorporate a prior on the distance function. Unlike some
other methods, ITML does not rely on an eigenvalue computation or semi-definite
programming.

.. topic:: Example Code:

::

    from metric_learn import ITML

    pairs = [[[1.2, 7.5], [1.3, 1.5]],
             [[6.4, 2.6], [6.2, 9.7]],
             [[1.3, 4.5], [3.2, 4.6]],
             [[6.2, 5.5], [5.4, 5.4]]]
    y = [1, 1, -1, -1]

    # in this task we want points where the first feature is close to be closer
    # to each other, no matter how close the second feature is


    itml = ITML()
    itml.fit(pairs, y)

.. topic:: References:

    .. [1] `Information-theoretic Metric Learning <http://machinelearning.wustl
       .edu/mlpapers/paper_files/icml2007_DavisKJSD07.pdf>`_ Jason V. Davis,
       et al.

    .. [2] Adapted from Matlab code at http://www.cs.utexas.edu/users/pjain/
       itml/


LSML
----

`LSML`: Metric Learning from Relative Comparisons by Minimizing Squared
Residual

.. topic:: Example Code:

::

    from metric_learn import LSML

    quadruplets = [[[1.2, 7.5], [1.3, 1.5], [6.4, 2.6], [6.2, 9.7]],
                   [[1.3, 4.5], [3.2, 4.6], [6.2, 5.5], [5.4, 5.4]],
                   [[3.2, 7.5], [3.3, 1.5], [8.4, 2.6], [8.2, 9.7]],
                   [[3.3, 4.5], [5.2, 4.6], [8.2, 5.5], [7.4, 5.4]]]

    # we want to make closer points where the first feature is close, and
    # further if the second feature is close

    lsml = LSML()
    lsml.fit(quadruplets)

.. topic:: References:

    .. [1] Liu et al.
       "Metric Learning from Relative Comparisons by Minimizing Squared
       Residual". ICDM 2012. http://www.cs.ucla.edu/~weiwang/paper/ICDM12.pdf

    .. [2] Adapted from https://gist.github.com/kcarnold/5439917


SDML
----

`SDML`: An efficient sparse metric learning in high-dimensional space via
L1-penalized log-determinant regularization

.. topic:: Example Code:

::

    from metric_learn import SDML

    pairs = [[[1.2, 7.5], [1.3, 1.5]],
             [[6.4, 2.6], [6.2, 9.7]],
             [[1.3, 4.5], [3.2, 4.6]],
             [[6.2, 5.5], [5.4, 5.4]]]
    y = [1, 1, -1, -1]

    # in this task we want points where the first feature is close to be closer
    # to each other, no matter how close the second feature is

    sdml = SDML()
    sdml.fit(pairs, y)

.. topic:: References:

    .. [1] Qi et al.
       An efficient sparse metric learning in high-dimensional space via
       L1-penalized log-determinant regularization. ICML 2009.
       http://lms.comp.nus.edu.sg/sites/default/files/publication-attachments/
       icml09-guojun.pdf

    .. [2] Adapted from https://gist.github.com/kcarnold/5439945


RCA
---

Relative Components Analysis (RCA)

`RCA` learns a full rank Mahalanobis distance metric based on a weighted sum of
in-class covariance matrices. It applies a global linear transformation to
assign large weights to relevant dimensions and low weights to irrelevant
dimensions. Those relevant dimensions are estimated using "chunklets", subsets
of points that are known to belong to the same class.

.. topic:: Example Code:

::

    from metric_learn import RCA

    pairs = [[[1.2, 7.5], [1.3, 1.5]],
             [[6.4, 2.6], [6.2, 9.7]],
             [[1.3, 4.5], [3.2, 4.6]],
             [[6.2, 5.5], [5.4, 5.4]]]
    y = [1, 1, -1, -1]

    # in this task we want points where the first feature is close to be closer
    # to each other, no matter how close the second feature is

    rca = RCA()
    rca.fit(pairs, y)


.. topic:: References:

    .. [1] `Adjustment learning and relevant component analysis
       <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.19.2871
       &rep=rep1&type=pdf>`_ Noam Shental, et al.

    .. [2] 'Learning distance functions using equivalence relations', ICML 2003

    .. [3]'Learning a Mahalanobis metric from equivalence constraints', JMLR
       2005

MMC
---

Mahalanobis Metric Learning with Application for Clustering with
Side-Information, Xing et al., NIPS 2002

`MMC` minimizes the sum of squared distances between similar examples, while
enforcing the sum of distances between dissimilar examples to be greater than a
certain margin. This leads to a convex and, thus, local-minima-free
optimization problem that can be solved efficiently. However, the algorithm
involves the computation of eigenvalues, which is the main speed-bottleneck.
Since it has initially been designed for clustering applications, one of the
implicit assumptions of MMC is that all classes form a compact set, i.e.,
follow a unimodal distribution, which restricts the possible use-cases of this
method. However, it is one of the earliest and a still often cited technique.

.. topic:: Example Code:

::

    from metric_learn import MMC

    pairs = [[[1.2, 7.5], [1.3, 1.5]],
             [[6.4, 2.6], [6.2, 9.7]],
             [[1.3, 4.5], [3.2, 4.6]],
             [[6.2, 5.5], [5.4, 5.4]]]
    y = [1, 1, -1, -1]

    # in this task we want points where the first feature is close to be closer
    # to each other, no matter how close the second feature is

    mmc = MMC()
    mmc.fit(pairs, y)

.. topic:: References:

  .. [1] `Distance metric learning with application to clustering with
        side-information <http://papers.nips
        .cc/paper/2164-distance-metric-learning-with-application-to-clustering
        -with-side-information.pdf>`_ Xing, Jordan, Russell, Ng.
  .. [2] Adapted from Matlab code `here <http://www.cs.cmu
     .edu/%7Eepxing/papers/Old_papers/code_Metric_online.tar.gz>`_.
