Local Fisher Discriminant Analysis (LFDA)
=========================================

.. automodule:: metric_learn.lfda
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Example Code
------------

::

    import numpy as np
    from metric_learn import LFDA
    from sklearn.datasets import load_iris

    iris_data = load_iris()
    X = iris_data['data']
    Y = iris_data['target']

    lfda = LFDA(k=2, dim=2)
    lfda.fit(X, Y)

References
------------------
`Dimensionality Reduction of Multimodal Labeled Data by Local Fisher Discriminant Analysis <http://www.ms.k.u-tokyo.ac.jp/2007/LFDA.pdf>`_ Masashi Sugiyama.

`Local Fisher Discriminant Analysis on Beer Style Clustering <https://gastrograph.com/resources/whitepapers/local-fisher-discriminant-analysis-on-beer-style-clustering.html#>`_ Yuan Tang.
