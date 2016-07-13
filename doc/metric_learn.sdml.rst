Sparse Determinant Metric Learning (SDML)
=========================================

.. automodule:: metric_learn.sdml
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Example Code
------------

::

    from metric_learn import SDML_Supervised
    from sklearn.datasets import load_iris

    iris_data = load_iris()
    X = iris_data['data']
    Y = iris_data['target']

    sdml = SDML_Supervised(num_constraints=200)
    sdml.fit(X, Y)

References
------------------
