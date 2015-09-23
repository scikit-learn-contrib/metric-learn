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

    import numpy as np
    from metric_learn import SDML
    from sklearn.datasets import load_iris

    iris_data = load_iris()
    X = iris_data['data']
    Y = iris_data['target']

    sdml = SDML()
    W = SDML.prepare_constraints(Y, X.shape[0], 1500)
    sdml.fit(X, W)

References
------------------
