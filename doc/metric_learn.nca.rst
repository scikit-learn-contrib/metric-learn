Neighborhood Components Analysis (NCA)
======================================

.. automodule:: metric_learn.nca
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Example Code
------------

::

    import numpy as np
    from metric_learn import NCA
    from sklearn.datasets import load_iris

    iris_data = load_iris()
    X = iris_data['data']
    Y = iris_data['target']

    nca = NCA(max_iter=1000, learning_rate=0.01)
    nca.fit(X, Y)

References
----------

