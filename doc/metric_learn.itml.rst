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

    from metric_learn import ITML_Supervised
    from sklearn.datasets import load_iris

    iris_data = load_iris()
    X = iris_data['data']
    Y = iris_data['target']

    itml = ITML_Supervised(num_constraints=200)
    itml.fit(X, Y)

References
----------
`Information-theoretic Metric Learning <http://machinelearning.wustl.edu/mlpapers/paper_files/icml2007_DavisKJSD07.pdf>`_ Jason V. Davis, et al.
