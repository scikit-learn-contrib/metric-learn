Information Theoretic Metric Learning (ITML)
============================================

.. automodule:: metric_learn.itml
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Example Code
------------

::

    import numpy as np
    from metric_learn import ITML
    from sklearn.datasets import load_iris

    iris_data = load_iris()
    X = iris_data['data']
    Y = iris_data['target']

    itml = ITML()

    num_constraints = 200
    C = ITML.prepare_constraints(Y, X.shape[0], num_constraints)
    itml.fit(X, C, verbose=False)

References
----------
`Information-theoretic Metric Learning <http://machinelearning.wustl.edu/mlpapers/paper_files/icml2007_DavisKJSD07.pdf>`_ Jason V. Davis, et al.
