|GitHub Actions Build Status| |License| |PyPI version| |Code coverage|

metric-learn: Metric Learning in Python
=======================================

metric-learn contains efficient Python implementations of several popular supervised and weakly-supervised metric learning algorithms. As part of `scikit-learn-contrib <https://github.com/scikit-learn-contrib>`_, the API of metric-learn is compatible with `scikit-learn <http://scikit-learn.org/stable/>`_, the leading library for machine learning in Python. This allows to use all the scikit-learn routines (for pipelining, model selection, etc) with metric learning algorithms through a unified interface.

**Algorithms**

-  Large Margin Nearest Neighbor (LMNN)
-  Information Theoretic Metric Learning (ITML)
-  Sparse Determinant Metric Learning (SDML)
-  Least Squares Metric Learning (LSML)
-  Sparse Compositional Metric Learning (SCML)
-  Neighborhood Components Analysis (NCA)
-  Local Fisher Discriminant Analysis (LFDA)
-  Relative Components Analysis (RCA)
-  Metric Learning for Kernel Regression (MLKR)
-  Mahalanobis Metric for Clustering (MMC)
-  Online Algorithm for Scalable Image Similarity (OASIS)

**Dependencies**

-  Python 3.6+ (the last version supporting Python 2 and Python 3.5 was
   `v0.5.0 <https://pypi.org/project/metric-learn/0.5.0/>`_)
-  numpy, scipy, scikit-learn>=0.20.3

**Optional dependencies**

- For SDML, using skggm will allow the algorithm to solve problematic cases
  (install from commit `a0ed406 <https://github.com/skggm/skggm/commit/a0ed406586c4364ea3297a658f415e13b5cbdaf8>`_).
  ``pip install 'git+https://github.com/skggm/skggm.git@a0ed406586c4364ea3297a658f415e13b5cbdaf8'`` to install the required version of skggm from GitHub.
-  For running the examples only: matplotlib

**Installation/Setup**

- If you use Anaconda: ``conda install -c conda-forge metric-learn``. See more options `here <https://github.com/conda-forge/metric-learn-feedstock#installing-metric-learn>`_.

- To install from PyPI: ``pip install metric-learn``.

- For a manual install of the latest code, download the source repository and run ``python setup.py install``. You may then run ``pytest test`` to run all tests (you will need to have the ``pytest`` package installed).

**Usage**

See the `sphinx documentation`_ for full documentation about installation, API, usage, and examples.

**Citation**

If you use metric-learn in a scientific publication, we would appreciate
citations to the following paper:

`metric-learn: Metric Learning Algorithms in Python
<http://www.jmlr.org/papers/volume21/19-678/19-678.pdf>`_, de Vazelhes
*et al.*, Journal of Machine Learning Research, 21(138):1-6, 2020.

Bibtex entry::

  @article{metric-learn,
    title = {metric-learn: {M}etric {L}earning {A}lgorithms in {P}ython},
    author = {{de Vazelhes}, William and {Carey}, CJ and {Tang}, Yuan and
              {Vauquier}, Nathalie and {Bellet}, Aur{\'e}lien},
    journal = {Journal of Machine Learning Research},
    year = {2020},
    volume = {21},
    number = {138},
    pages = {1--6}
  }

.. _sphinx documentation: http://contrib.scikit-learn.org/metric-learn/

.. |GitHub Actions Build Status| image:: https://github.com/scikit-learn-contrib/metric-learn/workflows/CI/badge.svg
   :target: https://github.com/scikit-learn-contrib/metric-learn/actions?query=event%3Apush+branch%3Amaster
.. |License| image:: http://img.shields.io/:license-mit-blue.svg?style=flat
   :target: http://badges.mit-license.org
.. |PyPI version| image:: https://badge.fury.io/py/metric-learn.svg
   :target: http://badge.fury.io/py/metric-learn
.. |Code coverage| image:: https://codecov.io/gh/scikit-learn-contrib/metric-learn/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/scikit-learn-contrib/metric-learn
