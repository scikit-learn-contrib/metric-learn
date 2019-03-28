==========================
Supervised Metric Learning
==========================

Supervised metric learning algorithms take as inputs points `X` and target
labels `y`, and learn a distance matrix that make points from the same class
(for classification) or with close target value (for regression) close to each
other, and points from different classes or with distant target values far away
from each other.

Scikit-learn compatibility
==========================

All supervised algorithms are scikit-learn `Estimators`, so they are
compatible with Pipelining and scikit-learn model selection routines.

Algorithms
==========

Covariance
----------

.. todo:: Covariance is unsupervised, so its doc should not be here.

`Covariance` does not "learn" anything, rather it calculates
the covariance matrix of the input data. This is a simple baseline method.

.. topic:: Example Code:

::

    from metric_learn import Covariance
    from sklearn.datasets import load_iris

    iris = load_iris()['data']

    cov = Covariance().fit(iris)
    x = cov.transform(iris)

.. topic:: References:

    .. [1] On the Generalized Distance in Statistics, P.C.Mahalanobis, 1936


.. _lmnn:

LMNN
-----

Large-margin nearest neighbor metric learning.

`LMNN` learns a Mahalanobis distance metric in the kNN classification
setting. The learned metric attempts to keep close k-nearest neighbors 
from the same class, while keeping examples from different classes 
separated by a large margin. This algorithm makes no assumptions about
the distribution of the data.

The distance is learned using the following optimization:

.. math::

      \min_\mathbf{L}\sum_{i, j}\eta_{ij}||\mathbf{L}(x_i-x_j)||^2 + 
      c\sum_{i, j, l}\eta_{ij}(1-y_{ij})[1+||\mathbf{L}(x_i-x_j)||^2-||
      \mathbf{L}(x_i-x_l)||^2]_+)

where :math:`x_i` is an data point, :math:`x_j` are its k nearest neighbors 
sharing the same label, and :math:`x_l` are all the other instances within 
that region with different labels, :math:`\eta_{ij}, y_{ij} \in \{0, 1\}` 
are both the indicators, :math:`\eta_{ij}` represents :math:`x_{j}` is the 
k nearest neighbors(with same labels) of :math:`x_{i}`, :math:`y_{ij}=0` 
indicates :math:`x_{i}, x_{j}` belong to different class, :math:`[\cdot]_+` 
is the Hinge loss :math:`[\cdot]_+=\max(0, \cdot)`.

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

If a recent version of the Shogun Python modular (``modshogun``) library
is available, the LMNN implementation will use the fast C++ version from
there. Otherwise, the included pure-Python version will be used.
The two implementations differ slightly, and the C++ version is more complete.

.. topic:: References:

    .. [1] `Distance Metric Learning for Large Margin Nearest Neighbor
       Classification
       <http://papers.nips.cc/paper/2795-distance-metric-learning-for-large
       -margin -nearest-neighbor-classification>`_ Kilian Q. Weinberger, John
       Blitzer, Lawrence K. Saul


.. _nca:

NCA
---

Neighborhood Components Analysis (`NCA`) is a distance metric learning
algorithm which aims to improve the accuracy of nearest neighbors
classification compared to the standard Euclidean distance. The algorithm
directly maximizes a stochastic variant of the leave-one-out k-nearest
neighbors (KNN) score on the training set. It can also learn a low-dimensional
linear transformation of data that can be used for data visualization and fast
classification.

They use the decomposition :math:`\mathbf{M} = \mathbf{L}^T\mathbf{L}` and 
define the probability :math:`p_{ij}` that :math:`x_i` is the neighbor of 
:math:`x_j` by calculating the softmax likelihood of the Mahalanobis distance:

.. math::

      p_{ij} = \frac{\exp(-|| \mathbf{L}x_i - \mathbf{L}x_j ||_2^2)}
      {\sum_{l\neq i}\exp(-||\mathbf{L}x_i - \mathbf{L}x_l||_2^2)}, 
      \qquad p_{ii}=0

Then the probability that :math:`x_i` will be correctly classified by the 
stochastic nearest neighbors rule is:

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

    .. [1] J. Goldberger, G. Hinton, S. Roweis, R. Salakhutdinov.
       "Neighbourhood Components Analysis". Advances in Neural Information
       Processing Systems. 17, 513-520, 2005.
       http://www.cs.nyu.edu/~roweis/papers/ncanips.pdf

    .. [2] Wikipedia entry on Neighborhood Components Analysis
       https://en.wikipedia.org/wiki/Neighbourhood_components_analysis

LFDA
----

Local Fisher Discriminant Analysis (LFDA)

`LFDA` is a linear supervised dimensionality reduction method. It is
particularly useful when dealing with multimodality, where one ore more classes
consist of separate clusters in input space. The core optimization problem of
LFDA is solved as a generalized eigenvalue problem.

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

.. topic:: References:

    .. [1] `Dimensionality Reduction of Multimodal Labeled Data by Local
       Fisher Discriminant Analysis <http://www.ms.k.u-tokyo.ac.jp/2007/LFDA
       .pdf>`_ Masashi Sugiyama.

    .. [2] `Local Fisher Discriminant Analysis on Beer Style Clustering
       <https://gastrograph.com/resources/whitepapers/local-fisher
       -discriminant-analysis-on-beer-style-clustering.html#>`_ Yuan Tang.


MLKR
----

Metric Learning for Kernel Regression.

`MLKR` is an algorithm for supervised metric learning, which learns a
distance function by directly minimising the leave-one-out regression error.
This algorithm can also be viewed as a supervised variation of PCA and can be
used for dimensionality reduction and high dimensional data visualization.

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

    .. [1] `Metric Learning for Kernel Regression <http://proceedings.mlr.
       press/v2/weinberger07a/weinberger07a.pdf>`_ Kilian Q. Weinberger,
       Gerald Tesauro


Supervised versions of weakly-supervised algorithms
---------------------------------------------------

Note that each :ref:`weakly-supervised algorithm <weakly_supervised_section>`
has a supervised version of the form `*_Supervised` where similarity tuples are
generated from the labels information and passed to the underlying algorithm.

.. todo:: add more details about that (see issue `<https://github
          .com/metric-learn/metric-learn/issues/135>`_)

.. topic:: Example Code:

::

    from metric_learn import MMC_Supervised
    from sklearn.datasets import load_iris

    iris_data = load_iris()
    X = iris_data['data']
    Y = iris_data['target']

    mmc = MMC_Supervised(num_constraints=200)
    mmc.fit(X, Y)
