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

    iris_data = load_iris()

    cov = Covariance()
    x = cov.fit_transform(iris_data['data'])
