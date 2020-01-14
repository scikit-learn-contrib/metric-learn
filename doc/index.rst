metric-learn: Metric Learning in Python
=======================================
|Travis-CI Build Status| |License| |PyPI version| |Code coverage|

`metric-learn <https://github.com/scikit-learn-contrib/metric-learn>`_
contains efficient Python implementations of several popular supervised and
weakly-supervised metric learning algorithms. As part of `scikit-learn-contrib
<https://github.com/scikit-learn-contrib>`_, the API of metric-learn is compatible with `scikit-learn
<https://scikit-learn.org/>`_, the leading library for machine learning in
Python. This allows to use all the scikit-learn routines (for pipelining,
model selection, etc) with metric learning algorithms through a unified
interface.

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


Documentation outline
---------------------

.. toctree::
   :maxdepth: 2

   getting_started

.. toctree::
   :maxdepth: 2

   user_guide

.. toctree::
   :maxdepth: 2

   Package Contents <metric_learn>

.. toctree::
   :maxdepth: 2

   auto_examples/index

:ref:`genindex` | :doc:`Modules <./metric_learn>` | :ref:`search`

.. |Travis-CI Build Status| image:: https://api.travis-ci.org/scikit-learn-contrib/metric-learn.svg?branch=master
   :target: https://travis-ci.org/scikit-learn-contrib/metric-learn
.. |PyPI version| image:: https://badge.fury.io/py/metric-learn.svg
   :target: http://badge.fury.io/py/metric-learn
.. |License| image:: http://img.shields.io/:license-mit-blue.svg?style=flat
   :target: http://badges.mit-license.org
.. |Code coverage| image:: https://codecov.io/gh/scikit-learn-contrib/metric-learn/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/scikit-learn-contrib/metric-learn
