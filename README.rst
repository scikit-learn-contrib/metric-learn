|Travis-CI Build Status| |License| |PyPI version| |Code coverage|

metric-learn: Metric Learning in Python
=======================================

metric-learn contains efficient Python implementations of several popular supervised and weakly-supervised metric learning algorithms. As part of `scikit-learn-contrib <https://github.com/scikit-learn-contrib>`_, the API of metric-learn is compatible with `scikit-learn <http://scikit-learn.org/stable/>`_, the leading library for machine learning in Python. This allows to use all the scikit-learn routines (for pipelining, model selection, etc) with metric learning algorithms through a unified interface.

**Algorithms**

-  Large Margin Nearest Neighbor (LMNN)
-  Information Theoretic Metric Learning (ITML)
-  Sparse Determinant Metric Learning (SDML)
-  Least Squares Metric Learning (LSML)
-  Neighborhood Components Analysis (NCA)
-  Local Fisher Discriminant Analysis (LFDA)
-  Relative Components Analysis (RCA)
-  Metric Learning for Kernel Regression (MLKR)
-  Mahalanobis Metric for Clustering (MMC)

**Dependencies**

-  Python 2.7+, 3.4+
-  numpy, scipy, scikit-learn>=0.20.3

**Optional dependencies**

- For SDML, using skggm will allow the algorithm to solve problematic cases
  (install from commit `a0ed406 <https://github.com/skggm/skggm/commit/a0ed406586c4364ea3297a658f415e13b5cbdaf8>`_).
  ``pip install 'git+https://github.com/skggm/skggm.git@a0ed406586c4364ea3297a658f415e13b5cbdaf8'`` to install the required version of skggm from GitHub.
-  For running the examples only: matplotlib

**Installation/Setup**

- If you use Anaconda: ``conda install -c conda-forge metric-learn``. See more
options `here <https://github.com/conda-forge/metric-learn-feedstock#installing-metric-learn>`_.

- To install from PyPi: ``pip install metric-learn``.

- For a manual install of the latest code, download the package from GitHub and run ``python setup.py install``. You may then run ``pytest test`` to run all tests (you will need to have the ``pytest`` package installed).

**Usage**

See the `sphinx documentation`_ for full documentation about installation, API, usage, and examples.

**Citation**

If you use metric-learn in a scientific publication, we would appreciate
citations to the following paper:

`metric-learn: Metric Learning Algorithms in Python
<https://arxiv.org/abs/1908.04710>`_, de Vazelhes
*et al.*, arXiv:1908.04710, 2019.

Bibtex entry::

  @techreport{metric-learn,
    title = {metric-learn: {M}etric {L}earning {A}lgorithms in {P}ython},
    author = {{de Vazelhes}, William and {Carey}, CJ and {Tang}, Yuan and
              {Vauquier}, Nathalie and {Bellet}, Aur{\'e}lien},
    institution = {arXiv:1908.04710},
    year = {2019}
  }

.. _sphinx documentation: http://contrib.scikit-learn.org/metric-learn/

.. |Travis-CI Build Status| image:: https://api.travis-ci.org/scikit-learn-contrib/metric-learn.svg?branch=master
   :target: https://travis-ci.org/scikit-learn-contrib/metric-learn
.. |License| image:: http://img.shields.io/:license-mit-blue.svg?style=flat
   :target: http://badges.mit-license.org
.. |PyPI version| image:: https://badge.fury.io/py/metric-learn.svg
   :target: http://badge.fury.io/py/metric-learn
.. |Code coverage| image:: https://codecov.io/gh/scikit-learn-contrib/metric-learn/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/scikit-learn-contrib/metric-learn
