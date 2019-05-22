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


Learning on pairs
=================

Some metric learning algorithms learn on pairs of samples. In this case, one
should provide the algorithm with ``n_samples`` pairs of points, with a
corresponding target containing ``n_samples`` values being either +1 or -1.
These values indicate whether the given pairs are similar points or
dissimilar points.


.. _calibration:

Thresholding
------------
In order to predict whether a new pair represents similar or dissimilar
samples, we need to set a distance threshold, so that points closer (in the
learned space) than this threshold are predicted as similar, and points further
away are predicted as dissimilar. Several methods are possible for this
thresholding.

- **At fit time**: The threshold is set with `calibrate_threshold` (see
  below) on the trainset. You can specify the calibration parameters directly
  in the `fit` method with the `threshold_params` parameter (see the
  documentation of the `fit` method of any metric learner that learns on pairs
  of points for more information). This method can cause a little bit of
  overfitting. If you want to avoid that, calibrate the threshold after
  fitting, on a validation set.

- **Manual**: calling `set_threshold` will set the threshold to a
  particular value.

- **Calibration**: calling `calibrate_threshold` will calibrate the
  threshold to achieve a particular score on a validation set, the score
  being among the classical scores for classification (accuracy, f1 score...).


See also: `sklearn.calibration`.


Algorithms
==========

.. _itml:

:py:class:`ITML <metric_learn.itml.ITML>`
-----------------------------------------

Information Theoretic Metric Learning(:py:class:`ITML <metric_learn.itml.ITML>`)

`ITML` minimizes the (differential) relative entropy, aka Kullback–Leibler 
divergence, between two multivariate Gaussians subject to constraints on the 
associated Mahalanobis distance, which can be formulated into a Bregman 
optimization problem by minimizing the LogDet divergence subject to 
linear constraints. This algorithm can handle a wide variety of constraints
and can optionally incorporate a prior on the distance function. Unlike some
other methods, `ITML` does not rely on an eigenvalue computation or 
semi-definite programming.


Given a Mahalanobis distance parameterized by :math:`A`, its corresponding 
multivariate Gaussian is denoted as:

.. math::
    p(\mathbf{x}; \mathbf{A}) = \frac{1}{Z}\exp(-\frac{1}{2}d_\mathbf{A}
    (\mathbf{x}, \mu)) 
    =  \frac{1}{Z}\exp(-\frac{1}{2}((\mathbf{x} - \mu)^T\mathbf{A}
    (\mathbf{x} - \mu)) 

where :math:`Z` is the normalization constant, the inverse of Mahalanobis 
matrix :math:`\mathbf{A}^{-1}` is the covariance of the Gaussian.

Given pairs of similar points :math:`S` and pairs of dissimilar points 
:math:`D`, the distance metric learning problem is to minimize the LogDet
divergence, which is equivalent as minimizing :math:`\textbf{KL}(p(\mathbf{x}; 
\mathbf{A}_0) || p(\mathbf{x}; \mathbf{A}))`:

.. math::

    \min_\mathbf{A} D_{\ell \mathrm{d}}\left(A, A_{0}\right) = 
    \operatorname{tr}\left(A A_{0}^{-1}\right)-\log \operatorname{det}
    \left(A A_{0}^{-1}\right)-n\\
    \text{subject to } \quad d_\mathbf{A}(\mathbf{x}_i, \mathbf{x}_j) 
    \leq u \qquad (\mathbf{x}_i, \mathbf{x}_j)\in S \\
    d_\mathbf{A}(\mathbf{x}_i, \mathbf{x}_j) \geq l \qquad (\mathbf{x}_i, 
    \mathbf{x}_j)\in D


where :math:`u` and :math:`l` is the upper and the lower bound of distance
for similar and dissimilar pairs respectively, and :math:`\mathbf{A}_0` 
is the prior distance metric, set to identity matrix by default, 
:math:`D_{\ell \mathrm{d}}(\cdot)` is the log determinant.

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


.. _sdml:

:py:class:`SDML <metric_learn.sdml.SDML>`
-----------------------------------------

Sparse High-Dimensional Metric Learning
(:py:class:`SDML <metric_learn.sdml.SDML>`)

`SDML` is an efficient sparse metric learning in high-dimensional space via 
double regularization: an L1-penalization on the off-diagonal elements of the 
Mahalanobis matrix :math:`\mathbf{M}`, and a log-determinant divergence between 
:math:`\mathbf{M}` and :math:`\mathbf{M_0}` (set as either :math:`\mathbf{I}` 
or :math:`\mathbf{\Omega}^{-1}`, where :math:`\mathbf{\Omega}` is the 
covariance matrix).

The formulated optimization on the semidefinite matrix :math:`\mathbf{M}` 
is convex:

.. math::

    \min_{\mathbf{M}} = \text{tr}((\mathbf{M}_0 + \eta \mathbf{XLX}^{T})
    \cdot \mathbf{M}) - \log\det \mathbf{M} + \lambda ||\mathbf{M}||_{1, off}

where :math:`\mathbf{X}=[\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_n]` is 
the training data, the incidence matrix :math:`\mathbf{K}_{ij} = 1` if 
:math:`(\mathbf{x}_i, \mathbf{x}_j)` is a similar pair, otherwise -1. The 
Laplacian matrix :math:`\mathbf{L}=\mathbf{D}-\mathbf{K}` is calculated from 
:math:`\mathbf{K}` and :math:`\mathbf{D}`, a diagonal matrix whose entries are 
the sums of the row elements of :math:`\mathbf{K}`., :math:`||\cdot||_{1, off}` 
is the off-diagonal L1 norm.


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

.. _rca:

:py:class:`RCA <metric_learn.rca.RCA>`
--------------------------------------

Relative Components Analysis (:py:class:`RCA <metric_learn.rca.RCA>`)

`RCA` learns a full rank Mahalanobis distance metric based on a weighted sum of
in-chunklets covariance matrices. It applies a global linear transformation to 
assign large weights to relevant dimensions and low weights to irrelevant 
dimensions. Those relevant dimensions are estimated using "chunklets", subsets 
of points that are known to belong to the same class.

For a training set with :math:`n` training points in :math:`k` chunklets, the 
algorithm is efficient since it simply amounts to computing

.. math::

      \mathbf{C} = \frac{1}{n}\sum_{j=1}^k\sum_{i=1}^{n_j}
      (\mathbf{x}_{ji}-\hat{\mathbf{m}}_j)
      (\mathbf{x}_{ji}-\hat{\mathbf{m}}_j)^T


where chunklet :math:`j` consists of :math:`\{\mathbf{x}_{ji}\}_{i=1}^{n_j}` 
with a mean :math:`\hat{m}_j`. The inverse of :math:`\mathbf{C}^{-1}` is used 
as the Mahalanobis matrix.

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

.. _mmc:

:py:class:`MMC <metric_learn.mmc.MMC>`
--------------------------------------

Metric Learning with Application for Clustering with Side Information
(:py:class:`MMC <metric_learn.mmc.MMC>`)

`MMC` minimizes the sum of squared distances between similar points, while
enforcing the sum of distances between dissimilar ones to be greater than one. 
This leads to a convex and, thus, local-minima-free optimization problem that 
can be solved efficiently. 
However, the algorithm involves the computation of eigenvalues, which is the 
main speed-bottleneck. Since it has initially been designed for clustering 
applications, one of the implicit assumptions of MMC is that all classes form 
a compact set, i.e., follow a unimodal distribution, which restricts the 
possible use-cases of this method. However, it is one of the earliest and a 
still often cited technique.

The algorithm aims at minimizing the sum of distances between all the similar 
points, while constrains the sum of distances between dissimilar points:

.. math::

      \min_{\mathbf{M}\in\mathbb{S}_+^d}\sum_{(\mathbf{x}_i, 
      \mathbf{x}_j)\in S} d_{\mathbf{M}}(\mathbf{x}_i, \mathbf{x}_j)
      \qquad \qquad \text{s.t.} \qquad \sum_{(\mathbf{x}_i, \mathbf{x}_j)
      \in D} d^2_{\mathbf{M}}(\mathbf{x}_i, \mathbf{x}_j) \geq 1

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

Learning on quadruplets
=======================

A type of information even weaker than pairs is information about relative
comparisons between pairs. The user should provide the algorithm with a
quadruplet of points, where the two first points are closer than the two
last points. No target vector (``y``) is needed, since the supervision is
already in the order that points are given in the quadruplet.

Algorithms
==========

.. _lsml:

:py:class:`LSML <metric_learn.lsml.LSML>`
-----------------------------------------

Metric Learning from Relative Comparisons by Minimizing Squared Residual
(:py:class:`LSML <metric_learn.lsml.LSML>`)

`LSML` proposes a simple, yet effective, algorithm that minimizes a convex 
objective function corresponding to the sum of squared residuals of 
constraints. This algorithm uses the constraints in the form of the 
relative distance comparisons, such method is especially useful where 
pairwise constraints are not natural to obtain, thus pairwise constraints 
based algorithms become infeasible to be deployed. Furthermore, its sparsity 
extension leads to more stable estimation when the dimension is high and 
only a small amount of constraints is given.

The loss function of each constraint 
:math:`d(\mathbf{x}_a, \mathbf{x}_b) < d(\mathbf{x}_c, \mathbf{x}_d)` is 
denoted as:

.. math::

    H(d_\mathbf{M}(\mathbf{x}_a, \mathbf{x}_b) 
    - d_\mathbf{M}(\mathbf{x}_c, \mathbf{x}_d))

where :math:`H(\cdot)` is the squared Hinge loss function defined as:

.. math::

    H(x) = \left\{\begin{aligned}0 \qquad x\leq 0 \\
    \,\,x^2 \qquad x>0\end{aligned}\right.\\

The summed loss function :math:`L(C)` is the simple sum over all constraints 
:math:`C = \{(\mathbf{x}_a , \mathbf{x}_b , \mathbf{x}_c , \mathbf{x}_d) 
: d(\mathbf{x}_a , \mathbf{x}_b) < d(\mathbf{x}_c , \mathbf{x}_d)\}`. The 
original paper suggested here should be a weighted sum since the confidence 
or probability of each constraint might differ. However, for the sake of 
simplicity and assumption of no extra knowledge provided, we just deploy 
the simple sum here as well as what the authors did in the experiments.

The distance metric learning problem becomes minimizing the summed loss 
function of all constraints plus a regularization term w.r.t. the prior 
knowledge:

.. math::

    \min_\mathbf{M}(D_{ld}(\mathbf{M, M_0}) + \sum_{(\mathbf{x}_a, 
    \mathbf{x}_b, \mathbf{x}_c, \mathbf{x}_d)\in C}H(d_\mathbf{M}(
    \mathbf{x}_a, \mathbf{x}_b) - d_\mathbf{M}(\mathbf{x}_c, \mathbf{x}_c))\\

where :math:`\mathbf{M}_0` is the prior metric matrix, set as identity 
by default, :math:`D_{ld}(\mathbf{\cdot, \cdot})` is the LogDet divergence:

.. math::

    D_{ld}(\mathbf{M, M_0}) = \text{tr}(\mathbf{MM_0}) − \text{logdet}
    (\mathbf{M})

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


