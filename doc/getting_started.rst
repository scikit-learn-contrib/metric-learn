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
-  numpy, scipy, scikit-learn>=0.20.3

**Optional dependencies**

- For SDML, using skggm will allow the algorithm to solve problematic cases
  (install from commit `a0ed406 <https://github.com/skggm/skggm/commit/a0ed406586c4364ea3297a658f415e13b5cbdaf8>`_).
-  For running the examples only: matplotlib

Quick start
===========

This example loads the iris dataset, and evaluates a k-nearest neighbors
algorithm on an embedding space learned with `NCA`.

::

    from metric_learn import NCA
    from sklearn.datasets import load_iris
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.neighbors import KNeighborsClassifier
    
    X, y = load_iris(return_X_y=True)
    clf = make_pipeline(NCA(), KNeighborsClassifier())
    cross_val_score(clf, X, y)
