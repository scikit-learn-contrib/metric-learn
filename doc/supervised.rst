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

LMNN
-----

Large-margin nearest neighbor metric learning.

`LMNN` learns a Mahanalobis distance metric in the kNN classification
setting using semidefinite programming. The learned metric attempts to keep
k-nearest neighbors in the same class, while keeping examples from different
classes separated by a large margin. This algorithm makes no assumptions about
the distribution of the data.

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

NCA
---

Neighborhood Components Analysis (`NCA`) is a distance metric learning
algorithm which aims to improve the accuracy of nearest neighbors
classification compared to the standard Euclidean distance. The algorithm
directly  maximizes  a stochastic  variant  of  the leave-one-out k-nearest
neighbors (KNN) score on the training set.  It can also learn a low-dimensional
linear  embedding  of  data  that  can  be used for data visualization and fast
classification.

.. topic:: Example Code:

::

    import numpy as np
    from metric_learn import NCA
    from sklearn.datasets import load_iris

    iris_data = load_iris()
    X = iris_data['data']
    Y = iris_data['target']

    nca = NCA(max_iter=1000, learning_rate=0.01)
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
