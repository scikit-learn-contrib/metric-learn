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

Large Margin Nearest Neighbor Metric Learning
(:py:class:`LMNN <metric_learn.lmnn.LMNN>`)

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

where :math:`\mathbf{x}_i` is an data point, :math:`\mathbf{x}_j` is one 
of its k nearest neighbors sharing the same label, and :math:`\mathbf{x}_l` 
are all the other instances within that region with different labels, 
:math:`\eta_{ij}, y_{ij} \in \{0, 1\}` are both the indicators, 
:math:`\eta_{ij}` represents :math:`\mathbf{x}_{j}` is the k nearest 
neighbors(with same labels) of :math:`\mathbf{x}_{i}`, :math:`y_{ij}=0` 
indicates :math:`\mathbf{x}_{i}, \mathbf{x}_{j}` belong to different class, 
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

Neighborhood Components Analysis(:py:class:`NCA <metric_learn.nca.NCA>`)

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

    .. [1] J. Goldberger, G. Hinton, S. Roweis, R. Salakhutdinov.
       "Neighbourhood Components Analysis". Advances in Neural Information
       Processing Systems. 17, 513-520, 2005.
       http://www.cs.nyu.edu/~roweis/papers/ncanips.pdf

    .. [2] Wikipedia entry on Neighborhood Components Analysis
       https://en.wikipedia.org/wiki/Neighbourhood_components_analysis

.. _lfda:

LFDA
----

Local Fisher Discriminant Analysis(:py:class:`LFDA <metric_learn.lfda.LFDA>`)

`LFDA` is a linear supervised dimensionality reduction method. It is
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
matrix :math:`\mathbf{A}`:, which can be calculated with local scaling methods.

Then the learning problem becomes derive the LFDA transformation matrix 
:math:`\mathbf{T}_{LFDA}`:

.. math::

    \mathbf{T}_{LFDA} = \arg\max_\mathbf{T}
    [\text{tr}((\mathbf{T}^T\mathbf{S}^{(w)}
    \mathbf{T})^{-1}\mathbf{T}^T\mathbf{S}^{(b)}\mathbf{T})]

That is, it is looking for a transformation matrix :math:`\mathbf{T}` such that 
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

.. topic:: References:

    .. [1] `Dimensionality Reduction of Multimodal Labeled Data by Local
       Fisher Discriminant Analysis <http://www.ms.k.u-tokyo.ac.jp/2007/LFDA
       .pdf>`_ Masashi Sugiyama.

    .. [2] `Local Fisher Discriminant Analysis on Beer Style Clustering
       <https://gastrograph.com/resources/whitepapers/local-fisher
       -discriminant-analysis-on-beer-style-clustering.html#>`_ Yuan Tang.

.. _mlkr:

MLKR
----

Metric Learning for Kernel Regression(:py:class:`MLKR <metric_learn.mlkr.MLKR>`)

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
\mathbf{x}_j) = ||\mathbf{A}(\mathbf{x}_i - \mathbf{x}_j)||`, the transition 
matrix :math:`\mathbf{A}` is derived from the decomposition of Mahalanobis 
matrix :math:`\mathbf{M=A^TA}`.

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
