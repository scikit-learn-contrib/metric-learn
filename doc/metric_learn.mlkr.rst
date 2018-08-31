Metric Learning for Kernel Regression (MLKR)
============================================

.. automodule:: metric_learn.mlkr
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :special-members: __init__

Example Code
------------

::

    from metric_learn import MLKR
    from sklearn.datasets import load_iris

    iris_data = load_iris()
    X = iris_data['data']
    Y = iris_data['target']

    mlkr = MLKR()
    mlkr.fit(X, Y)

References
----------
`Information-theoretic Metric Learning <http://machinelearning.wustl.edu/mlpapers/paper_files/icml2007_DavisKJSD07.pdf>`_ Jason V. Davis, et al.
