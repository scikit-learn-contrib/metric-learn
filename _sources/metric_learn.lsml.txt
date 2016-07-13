Least Squares Metric Learning (LSML)
====================================

.. automodule:: metric_learn.lsml
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Example Code
------------

::

    from metric_learn import LSML_Supervised
    from sklearn.datasets import load_iris

    iris_data = load_iris()
    X = iris_data['data']
    Y = iris_data['target']

    lsml = LSML_Supervised(num_constraints=200)
    isml.fit(X, Y)

References
----------

