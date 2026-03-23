.. _api-structure:
 
=============
API Structure
=============

The API structure of metric-learn is insipred on the main classes from scikit-learn:
``Estimator``, ``Predictor``, ``Transformer`` (check them
`here <https://scikit-learn.org/stable/developers/develop.html>`_).


BaseMetricLearner
^^^^^^^^^^^^^^^^^

All learners are ``BaseMetricLearner`` wich inherit from scikit-learn's ``BaseEstimator``
class, so all of them have a ``fit`` method to learn from data, either:

.. code-block::
  
  estimator = estimator.fit(data, targets)

or 

.. code-block::
  
  estimator = estimator.fit(data)

This class has three main abstract methods that all learners need to implement:

+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Abstract method** | **Description**                                                                                                                                                                                                    |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| pair_score          | Returns the similarity score between pairs of points (the larger the score, the more similar the pair). For metric learners that learn a distancethe score is simply the opposite of the distance between pairs.   |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| pair_distance       | Returns the (pseudo) distance between pairs, when available. For metric learrners that do not learn a (pseudo) distance, an error is thrown instead.                                                               |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| get_metric          | Returns a function that takes as input two 1D arrays and outputs the value of the learned metric on these two points. Depending on the algorithm, it can return a distance or a similarity function between pairs. |
+---------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

As you may noticed, the algorithms can learn a (pseudo) distance or a similarity. Most
algorithms in the package learn a Mahalanobis metric, and have these three methods
available, but for similarity learners ``pair_distance`` must throw an error. If you
want to implement an algorithm of this kind, take this into account.

MetricTransformer
^^^^^^^^^^^^^^^^^

Following scikit-learn's ``Transformer`` class gidelines, Mahalanobis learners inherit
from a custom class named ``MetricTransformer`` wich only has the ``transform`` method.
With it, these learners can apply a linear transformation to the input:

.. code-block::

  new_data = transformer.transform(data)

Mixins
^^^^^^

Mixins represent the `metric` that algorithms need to learn. As of now, two main
mixins are available: ``MahalanobisMixin`` and ``BilinearMixin``. They inherit from
``BaseMetricLearner``, and/or ``MetricTransformer`` and **implement the abstract methods**
needed. Later on, the algorithms inherit from the Mixin to access these methods while
computing distance or the similarity score.

As many algorithms learn the same metric, such as Mahalanobis, its useful to have the
Mixins to avoid duplicated code, and to make sure that these metrics are computed
correctly.

Classifiers
^^^^^^^^^^^

Weakly-Supervised algorithms that learn from tuples such as pairs, triplets or quadruplets
can also classify unseen points, using the learned metric.

Metric-learn has three specific plug-and-play classes for this: ``_PairsClassifierMixin``,
``_TripletsClassifierMixin`` and ``_QuadrupletsClassifierMixin``. All inherit from
``BaseMetricLearner`` to access the methods described earlier.

All these classifiers implement the following methods:

+---------------------+-------------------------------------------------------------------------------------+
| **Abstract method** | **Description**                                                                     |
+---------------------+-------------------------------------------------------------------------------------+
| predict             | Predicts the ordering between sample distances in input pairs/triplets/quadruplets. |
+---------------------+-------------------------------------------------------------------------------------+
| decision_function   | Returns the decision function used to classify the pairs.                           |
+---------------------+-------------------------------------------------------------------------------------+
| score               | Computes score of pairs/triplets/quadruplets similarity prediction.                 |
+---------------------+-------------------------------------------------------------------------------------+
