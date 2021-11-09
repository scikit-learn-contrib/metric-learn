============================
Unsupervised Metric Learning
============================

Unsupervised metric learning algorithms only take as input an (unlabeled)
dataset `X`. For now, in metric-learn, there only is `Covariance`, which is a
simple baseline algorithm (see below).


Algorithms
==========
.. _covariance:

Covariance
----------

`Covariance` does not "learn" anything, rather it calculates
the covariance matrix of the input data. This is a simple baseline method.
It can be used for ZCA whitening of the data (see the Wikipedia page of
`whitening transformation <https://en.wikipedia.org/wiki/\
Whitening_transformation>`_).

.. rubric:: Example Code

::

    from metric_learn import Covariance
    from sklearn.datasets import load_iris

    iris = load_iris()['data']

    cov = Covariance().fit(iris)
    x = cov.transform(iris)

.. rubric:: References


.. container:: hatnote hatnote-gray

      [1]. On the Generalized Distance in Statistics, P.C.Mahalanobis, 1936.