==========================
Supervised Metric Learning
==========================

Supervised metric learning algorithms take as inputs points `X` and target
labels `y`, and learn a distance matrix that make points from the same class
(for classification) or with close target value (for regression) close to each
other, and points from different classes or with distant target values far away
from each other.

General API
===========

Supervised metric learning algorithms essentially use the same API as
scikit-learn.

Input data
----------
In order to train a model, you need two `array-like <https://scikit-learn\
.org/stable/glossary.html#term-array-like>`_ objects, `X` and `y`. `X`
should be a 2D array-like of shape `(n_samples, n_features)`, where
`n_samples` is the number of points of your dataset and `n_features` is the
number of attributes describing each point. `y` should be a 1D
array-like
of shape `(n_samples,)`, containing for each point in `X` the class it
belongs to (or the value to regress for this sample, if you use `MLKR` for
instance).

Here is an example of a dataset of two dogs and one
cat (the classes are 'dog' and 'cat') an animal being represented by
two numbers.

>>> import numpy as np
>>> X = np.array([[2.3, 3.6], [0.2, 0.5], [6.7, 2.1]])
>>> y = np.array(['dog', 'cat', 'dog'])

.. note::

   You can also use a preprocessor instead of directly giving the inputs as
   2D arrays. See the :ref:`preprocessor_section` section for more details.

Fit, transform, and so on
-------------------------
The goal of supervised metric-learning algorithms is to transform
points in a new space, in which the distance between two points from the
same class will be small, and the distance between two points from different
classes will be large. To do so, we fit the metric learner (example:
`NCA`).

>>> from metric_learn import NCA
>>> nca = NCA(random_state=42)
>>> nca.fit(X, y)
NCA(init='auto', max_iter=100, n_components=None,
  preprocessor=None, random_state=42, tol=None, verbose=False)


Now that the estimator is fitted, you can use it on new data for several
purposes.

First, you can transform the data in the learned space, using `transform`:
Here we transform two points in the new embedding space.

>>> X_new = np.array([[9.4, 4.1], [2.1, 4.4]])
>>> nca.transform(X_new)
array([[ 5.91884732, 10.25406973],
       [ 3.1545886 ,  6.80350083]])

Also, as explained before, our metric learners has learn a distance between
points. You can use this distance in two main ways:

- You can either return the distance between pairs of points using the
  `pair_distance` function:

>>> nca.pair_distance([[[3.5, 3.6], [5.6, 2.4]], [[1.2, 4.2], [2.1, 6.4]]])
array([0.49627072, 3.65287282])

- Or you can return a function that will return the distance (in the new
  space) between two 1D arrays (the coordinates of the points in the original
  space), similarly to distance functions in `scipy.spatial.distance`.

>>> metric_fun = nca.get_metric()
>>> metric_fun([3.5, 3.6], [5.6, 2.4])
0.4962707194621285

.. note::

    If the metric learner that you use learns a :ref:`Mahalanobis distance
    <mahalanobis_distances>` (like it is the case for all algorithms
    currently in metric-learn), you can get the plain learned Mahalanobis
    matrix using `get_mahalanobis_matrix`.

    >>> nca.get_mahalanobis_matrix()
    array([[0.43680409, 0.89169412],
           [0.89169412, 1.9542479 ]])

.. TODO: remove the "like it is the case etc..." if it's not the case anymore

Scikit-learn compatibility
--------------------------

All supervised algorithms are scikit-learn estimators 
(`sklearn.base.BaseEstimator`) and transformers 
(`sklearn.base.TransformerMixin`) so they are compatible with pipelines 
(`sklearn.pipeline.Pipeline`) and
scikit-learn model selection routines 
(`sklearn.model_selection.cross_val_score`,
`sklearn.model_selection.GridSearchCV`, etc).

Algorithms
==========

.. _lmnn:

:py:class:`LMNN <metric_learn.LMNN>`
-----------------------------------------

Large Margin Nearest Neighbor Metric Learning
(:py:class:`LMNN <metric_learn.LMNN>`)

`LMNN` learns a Mahalanobis distance metric in the kNN classification
setting. The learned metric attempts to keep close k-nearest neighbors 
from the same class, while keeping examples from different classes 
separated by a large margin. This algorithm makes no assumptions about
the distribution of the data.

The distance is learned by solving the following optimization problem:

.. math::

      \min_\mathbf{L}\sum_{i, j}\eta_{ij}||\mathbf{L(x_i-x_j)}||^2 + 
      c\sum_{i, j, l}\eta_{ij}(1-y_{ij})[1+||\mathbf{L(x_i-x_j)}||^2-||
      \mathbf{L(x_i-x_l)}||^2]_+)

where :math:`\mathbf{x}_i` is a data point, :math:`\mathbf{x}_j` is one 
of its k-nearest neighbors sharing the same label, and :math:`\mathbf{x}_l` 
are all the other instances within that region with different labels, 
:math:`\eta_{ij}, y_{ij} \in \{0, 1\}` are both the indicators, 
:math:`\eta_{ij}` represents :math:`\mathbf{x}_{j}` is the k-nearest 
neighbors (with same labels) of :math:`\mathbf{x}_{i}`, :math:`y_{ij}=0` 
indicates :math:`\mathbf{x}_{i}, \mathbf{x}_{j}` belong to different classes, 
:math:`[\cdot]_+=\max(0, \cdot)` is the Hinge loss.

.. topic:: Example Code:

::

    import numpy as np
    from metric_learn import LMNN
    from sklearn.datasets import load_iris

    iris_data = load_iris()
    X = iris_data['data']
    Y = iris_data['target']

    lmnn = LMNN(k=5, learn_rate=1e-6)
    lmnn.fit(X, Y, verbose=False)

.. topic:: References:

    .. [1] Weinberger et al. `Distance Metric Learning for Large Margin
       Nearest Neighbor Classification
       <http://jmlr.csail.mit.edu/papers/volume10/weinberger09a/weinberger09a.pdf>`_.
       JMLR 2009

    .. [2] `Wikipedia entry on Large Margin Nearest Neighbor <https://en.wikipedia.org/wiki/Large_margin_nearest_neighbor>`_
       

.. _nca:

:py:class:`NCA <metric_learn.NCA>`
--------------------------------------

Neighborhood Components Analysis (:py:class:`NCA <metric_learn.NCA>`)

`NCA` is a distance metric learning algorithm which aims to improve the 
accuracy of nearest neighbors classification compared to the standard 
Euclidean distance. The algorithm directly maximizes a stochastic variant 
of the leave-one-out k-nearest neighbors (KNN) score on the training set. 
It can also learn a low-dimensional linear transformation of data that can 
be used for data visualization and fast classification.

They use the decomposition :math:`\mathbf{M} = \mathbf{L}^T\mathbf{L}` and 
define the probability :math:`p_{ij}` that :math:`\mathbf{x}_i` is the 
neighbor of :math:`\mathbf{x}_j` by calculating the softmax likelihood of 
the Mahalanobis distance:

.. math::

      p_{ij} = \frac{\exp(-|| \mathbf{Lx}_i - \mathbf{Lx}_j ||_2^2)}
      {\sum_{l\neq i}\exp(-||\mathbf{Lx}_i - \mathbf{Lx}_l||_2^2)}, 
      \qquad p_{ii}=0

Then the probability that :math:`\mathbf{x}_i` will be correctly classified 
by the stochastic nearest neighbors rule is:

.. math::

      p_{i} = \sum_{j:j\neq i, y_j=y_i}p_{ij}

The optimization problem is to find matrix :math:`\mathbf{L}` that maximizes 
the sum of probability of being correctly classified:

.. math::

      \mathbf{L} = \text{argmax}\sum_i p_i

.. topic:: Example Code:

::

    import numpy as np
    from metric_learn import NCA
    from sklearn.datasets import load_iris

    iris_data = load_iris()
    X = iris_data['data']
    Y = iris_data['target']

    nca = NCA(max_iter=1000)
    nca.fit(X, Y)

.. topic:: References:

    .. [1] Goldberger et al.
       `Neighbourhood Components Analysis <https://papers.nips.cc/paper/2566-neighbourhood-components-analysis.pdf>`_.
       NIPS 2005

    .. [2] `Wikipedia entry on Neighborhood Components Analysis <https://en.wikipedia.org/wiki/Neighbourhood_components_analysis>`_
       

.. _lfda:

:py:class:`LFDA <metric_learn.LFDA>`
-----------------------------------------

Local Fisher Discriminant Analysis (:py:class:`LFDA <metric_learn.LFDA>`)

`LFDA` is a linear supervised dimensionality reduction method which effectively combines the ideas of `Linear Discriminant Analysis <https://en.wikipedia.org/wiki/Linear_discriminant_analysis>` and Locality-Preserving Projection . It is
particularly useful when dealing with multi-modality, where one ore more classes
consist of separate clusters in input space. The core optimization problem of
LFDA is solved as a generalized eigenvalue problem.


The algorithm define the Fisher local within-/between-class scatter matrix 
:math:`\mathbf{S}^{(w)}/ \mathbf{S}^{(b)}` in a pairwise fashion:

.. math::

    \mathbf{S}^{(w)} = \frac{1}{2}\sum_{i,j=1}^nW_{ij}^{(w)}(\mathbf{x}_i - 
    \mathbf{x}_j)(\mathbf{x}_i - \mathbf{x}_j)^T,\\
    \mathbf{S}^{(b)} = \frac{1}{2}\sum_{i,j=1}^nW_{ij}^{(b)}(\mathbf{x}_i - 
    \mathbf{x}_j)(\mathbf{x}_i - \mathbf{x}_j)^T,\\

where 

.. math::

    W_{ij}^{(w)} = \left\{\begin{aligned}0 \qquad y_i\neq y_j \\
    \,\,\mathbf{A}_{i,j}/n_l \qquad y_i = y_j\end{aligned}\right.\\
    W_{ij}^{(b)} = \left\{\begin{aligned}1/n \qquad y_i\neq y_j \\
    \,\,\mathbf{A}_{i,j}(1/n-1/n_l) \qquad y_i = y_j\end{aligned}\right.\\

here :math:`\mathbf{A}_{i,j}` is the :math:`(i,j)`-th entry of the affinity
matrix :math:`\mathbf{A}`:, which can be calculated with local scaling methods, `n` and `n_l` are the total number of points and the number of points per cluster `l` respectively.

Then the learning problem becomes derive the LFDA transformation matrix 
:math:`\mathbf{L}_{LFDA}`:

.. math::

    \mathbf{L}_{LFDA} = \arg\max_\mathbf{L}
    [\text{tr}((\mathbf{L}^T\mathbf{S}^{(w)}
    \mathbf{L})^{-1}\mathbf{L}^T\mathbf{S}^{(b)}\mathbf{L})]

That is, it is looking for a transformation matrix :math:`\mathbf{L}` such that 
nearby data pairs in the same class are made close and the data pairs in 
different classes are separated from each other; far apart data pairs in the 
same class are not imposed to be close.

.. topic:: Example Code:

::

    import numpy as np
    from metric_learn import LFDA
    from sklearn.datasets import load_iris

    iris_data = load_iris()
    X = iris_data['data']
    Y = iris_data['target']

    lfda = LFDA(k=2, dim=2)
    lfda.fit(X, Y)

.. note::
    LDFA suffers from a problem called “sign indeterminacy”, which means the sign of the ``components`` and the output from transform depend on a random state. This is directly related to the calculation of eigenvectors in the algorithm. The same input ran in different times might lead to different transforms, but both valid.
    
    To work around this, fit instances of this class to data once, then keep the instance around to do transformations.

.. topic:: References:

    .. [1] Sugiyama. `Dimensionality Reduction of Multimodal Labeled Data by Local
       Fisher Discriminant Analysis <http://www.jmlr.org/papers/volume8/sugiyama07b/sugiyama07b.pdf>`_.
       JMLR 2007

    .. [2] Tang. `Local Fisher Discriminant Analysis on Beer Style Clustering
       <https://gastrograph.com/resources/whitepapers/local-fisher
       -discriminant-analysis-on-beer-style-clustering.html#>`_.

.. _mlkr:

:py:class:`MLKR <metric_learn.MLKR>`
-----------------------------------------

Metric Learning for Kernel Regression (:py:class:`MLKR <metric_learn.MLKR>`)

`MLKR` is an algorithm for supervised metric learning, which learns a
distance function by directly minimizing the leave-one-out regression error.
This algorithm can also be viewed as a supervised variation of PCA and can be
used for dimensionality reduction and high dimensional data visualization.

Theoretically, `MLKR` can be applied with many types of kernel functions and 
distance metrics, we hereafter focus the exposition on a particular instance 
of the Gaussian kernel and Mahalanobis metric, as these are used in our 
empirical development. The Gaussian kernel is denoted as:

.. math::

    k_{ij} = \frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{d(\mathbf{x}_i, 
    \mathbf{x}_j)}{\sigma^2})

where :math:`d(\cdot, \cdot)` is the squared distance under some metrics, 
here in the fashion of Mahalanobis, it should be :math:`d(\mathbf{x}_i, 
\mathbf{x}_j) = ||\mathbf{L}(\mathbf{x}_i - \mathbf{x}_j)||`, the transition 
matrix :math:`\mathbf{L}` is derived from the decomposition of Mahalanobis 
matrix :math:`\mathbf{M=L^TL}`.

Since :math:`\sigma^2` can be integrated into :math:`d(\cdot)`, we can set 
:math:`\sigma^2=1` for the sake of simplicity. Here we use the cumulative 
leave-one-out quadratic regression error of the training samples as the 
loss function:

.. math::

    \mathcal{L} = \sum_i(y_i - \hat{y}_i)^2

where the prediction :math:`\hat{y}_i` is derived from kernel regression by 
calculating a weighted average of all the training samples:

.. math::

    \hat{y}_i = \frac{\sum_{j\neq i}y_jk_{ij}}{\sum_{j\neq i}k_{ij}}

.. topic:: Example Code:

::

    from metric_learn import MLKR
    from sklearn.datasets import load_iris

    iris_data = load_iris()
    X = iris_data['data']
    Y = iris_data['target']

    mlkr = MLKR()
    mlkr.fit(X, Y)

.. topic:: References:

    .. [1] Weinberger et al. `Metric Learning for Kernel Regression <http://proceedings.mlr.
       press/v2/weinberger07a/weinberger07a.pdf>`_. AISTATS 2007


.. _supervised_version:

Supervised versions of weakly-supervised algorithms
---------------------------------------------------

Each :ref:`weakly-supervised algorithm <weakly_supervised_section>`
has a supervised version of the form `*_Supervised` where similarity tuples are
randomly generated from the labels information and passed to the underlying
algorithm. 

.. warning::
    Supervised versions of weakly-supervised algorithms interpret label -1
    (or any negative label) as a point with unknown label.
    Those points are discarded in the learning process.

For pairs learners (see :ref:`learning_on_pairs`), pairs (tuple of two points
from the dataset), and pair labels (`int` indicating whether the two points
are similar (+1) or dissimilar (-1)), are sampled with the function
`metric_learn.constraints.positive_negative_pairs`. To sample positive pairs
(of label +1), this method will look at all the samples from the same label and
sample randomly a pair among them. To sample negative pairs (of label -1), this
method will look at all the samples from a different class and sample randomly
a pair among them. The method will try to build `num_constraints` positive
pairs and `num_constraints` negative pairs, but sometimes it cannot find enough
of one of those, so forcing `same_length=True` will return both times the
minimum of the two lenghts.

For using quadruplets learners (see :ref:`learning_on_quadruplets`) in a
supervised way, positive and negative pairs are sampled as above and
concatenated so that we have a 3D array of
quadruplets, where for each quadruplet the two first points are from the same
class, and the two last points are from a different class (so indeed the two
last points should be less similar than the two first points).

.. topic:: Example Code:

::

    from metric_learn import MMC_Supervised
    from sklearn.datasets import load_iris

    iris_data = load_iris()
    X = iris_data['data']
    Y = iris_data['target']

    mmc = MMC_Supervised(num_constraints=200)
    mmc.fit(X, Y)
