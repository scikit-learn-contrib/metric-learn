Covariance metric (baseline method)
===================================

.. automodule:: metric_learn.covariance
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Example Code
------------

::

    from metric_learn import Covariance
    from sklearn.datasets import load_iris

    iris = load_iris()['data']

    cov = Covariance().fit(iris)
    x = cov.transform(iris)
