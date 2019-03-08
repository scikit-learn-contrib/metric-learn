###############
Getting started
###############

Installation and Setup
======================

Run ``pip install metric-learn`` to download and install from PyPI.

Alternately, download the source repository and run:

-  ``python setup.py install`` for default installation.
-  ``python setup.py test`` to run all tests.

**Dependencies**

-  Python 2.7+, 3.4+
-  numpy, scipy, scikit-learn, and skggm (commit `a0ed406 <https://github.com/skggm/skggm/commit/a0ed406586c4364ea3297a658f415e13b5cbdaf8>`_) for `SDML`
-  (for running the examples only: matplotlib)

**Notes**

If a recent version of the Shogun Python modular (``modshogun``) library
is available, the LMNN implementation will use the fast C++ version from
there. The two implementations differ slightly, and the C++ version is
more complete.


Quick start
===========

This example loads the iris dataset, and evaluates a k-nearest neighbors
algorithm on an embedding space learned with `NCA`.

>>> from metric_learn import NCA
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import cross_val_score
>>> from sklearn.pipeline import make_pipeline
>>>
>>> X, y = load_iris(return_X_y=True)
>>> clf = make_pipeline(NCA(), KNeighborsClassifier())
>>> cross_val_score(clf, X, y)
