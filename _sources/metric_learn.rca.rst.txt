Relative Components Analysis (RCA)
==================================

.. automodule:: metric_learn.rca
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :special-members: __init__

Example Code
------------

::

    from metric_learn import RCA_Supervised
    from sklearn.datasets import load_iris

    iris_data = load_iris()
    X = iris_data['data']
    Y = iris_data['target']

    rca = RCA_Supervised(num_chunks=30, chunk_size=2)
    rca.fit(X, Y)

References
------------------
`Adjustment learning and relevant component analysis <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.19.2871&rep=rep1&type=pdf>`_ Noam Shental, et al.
