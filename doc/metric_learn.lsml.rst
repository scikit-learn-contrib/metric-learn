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

    import numpy as np
    from metric_learn import LSML
    from sklearn.datasets import load_iris

    iris_data = load_iris()
    X = iris_data['data']
    Y = iris_data['target']

    lsml = LSML()
    C = LSML.prepare_constraints(Y, 200)
    isml.fit(X, C, verbose=False)

References
----------

