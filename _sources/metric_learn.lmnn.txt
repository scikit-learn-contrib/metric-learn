Large Margin Nearest Neighbor (LMNN)
====================================

.. automodule:: metric_learn.lmnn
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Example Code
------------

::

    import numpy as np
    from metric_learn import LMNN
    from sklearn.datasets import load_iris

    iris_data = load_iris()
    X = iris_data['data']
    Y = iris_data['target']

    lmnn = LMNN(k=5, learn_rate=1e-6)
    lmnn.fit(X, Y, verbose=False)

If a recent version of the Shogun Python modular (``modshogun``) library
is available, the LMNN implementation will use the fast C++ version from
there. Otherwise, the included pure-Python version will be used.
The two implementations differ slightly, and the C++ version is more complete.

References
----------
`Distance Metric Learning for Large Margin Nearest Neighbor Classification <http://papers.nips.cc/paper/2795-distance-metric-learning-for-large-margin-nearest-neighbor-classification>`_ Kilian Q. Weinberger, John Blitzer, Lawrence K. Saul
