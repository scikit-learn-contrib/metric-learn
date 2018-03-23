.. _wsml:

Weakly Supervised Learning (General information)
================================================

Introduction
------------

In Distance Metric Learning, we are interested in learning a metric between
points that takes into account some supervised information about the
similarity between those points. If each point has a class, we can use this
information by saying that all intra-class points are similar, and inter-class
points are dissimilar.

However, sometimes we do not have a class for each sample. Instead, we can have
pairs of points and a label for each saying whether the points in each pair are
similar or not. Indeed, if we imagine a hand labeled dataset of images with a
huge number of classes, it will be easier for a human to say whether two images
are similar rather that telling, among the huge number of classes, which one is
that of the shown image. We can also have a dataset of triplets of points where
we know the first sample is more similar to the second than the third. Or we
could also have quadruplets of points where the two first points are more
similar than the two last are. In fact, some metric learning algorithms are
made to use this kind of data. These are Weakly Supervised Metric Learners. For
instance, `ITML`, `MMC` and `SDML` work on labeled pairs, and `LSML` works on
unlabeled quadruplets.

In the ``metric-learn`` package, we use an object called `ConstrainedDataset`
to store these kinds of datasets where each sample/line is a tuple of points
from an initial dataset. Contrary to a 3D numpy array where each line would be
a tuple of ``t`` points from an initial dataset,  `ConstrainedDataset` is
memory efficient as it does not duplicate points in the underlying memory.
Instead, it stores indices of points involved in every tuple, as well as the
initial dataset. Plus, it supports slicing on tuples to be compatible with
scikit-learn utilities for cross-validation (see :ref:`performance_ws`).

See documentation of `ConstrainedDataset` `here<ConstrainedDataset>` for more
information.



.. _workflow_ws:

Basic worflow
-------------

Let us see how we can use weakly supervised metric learners in a basic
scikit-learn like workflow with ``fit``, ``predict``, ``transform``,
``score`` etc.

- Fitting

Let's say we have a dataset of samples and we also know for some pairs of them
if they are similar of dissimilar. We want to fit a metric learner on this
data. First, we recognize this data is made of labeled pairs. What we will need
to do first is therefore to make a `ConstrainedDataset` with as input the
points ``X`` (an array of shape ``(n_samples, n_features)``, and the
constraints ``c`` (an array of shape ``(n_constraints, 2))`` of indices of
pairs. We also need to have a vector ``y_constraints`` of shape
``(n_constraints,)`` where each ``y_constraints_i`` is 1 if sample
``X[c[i,0]]`` is similar to sample ``X[c[i, 1]]`` and 0 if they are dissimilar.

.. code:: python

    from metric_learn import ConstrainedDataset
    X_constrained = ConstrainedDataset(X, c)

Then we can fit a Weakly Supervised Metric Learner (here that inherits from
`PairsMixin`), on this data (let's use `MMC` for instance):

.. code:: python

    from metric_learn import MMC
    mmc = MMC()
    mmc.fit(X_constrained, y_constraints)

.. _transform_ws:

- Transforming

Weakly supervised metric learners can also be used as transformers. Let us say
we have a fitted estimator. At ``transform`` time, they can independently be
used on arrays of samples as well as `ConstrainedDataset`s. Indeed, they will
return transformed samples and thus only need input samples (they will ignore
any information on constraints in the input). The transformed samples are the
new points in an embedded space.  See :ref:`this section<transform_ml>` for
more details about this transformation.

.. code:: python

    mmc.transform(X)

- Predicting

Weakly Supervised Metric Learners work on lines of data where each line is a
tuple of points of an original dataset. For some of these, we should also have
a label for each line (for instance in the cases of learning on pairs, each
label ``y_constraints_i`` should tell whether the pair in line ``i`` is a
similar or dissimilar pair). So for these algorithm, applying ``predict`` to an
input ConstrainedDataset will predict scalars related to this task for each
tuple. For instance in the case of pairs, ``predict`` will return for each
input pair a float measuring the similarity between samples in the pair.

See the API documentation for `WeaklySupervisedMixin`'s childs
( `PairsMixin`,
`TripletsMixin`, `QuadrupletsMixin`) for the particular prediction functions of
each type of Weakly Supervised Metric Learner.

.. code:: python

    mmc.predict(X_constrained)

- Scoring

We can also use scoring functions like this, calling the default scoring
function of the Weakly Supervised Learner we use:

.. code:: python

    mmc.score(X_constrained, y_constraints)

The type of score depends on the type of Weakly Supervised Metric Learner
used. See the API documentation for `WeaklySupervisedMixin`'s childs
(`PairsMixin`, `TripletsMixin`, `QuadrupletsMixin`) for the particular
default scoring functions of each type of estimator.

See also :ref:`performance_ws`, for how to use scikit-learn's
cross-validation routines with Weakly Supervised Metric Learners.


.. _supervised_version:

Supervised Version
------------------

Weakly Supervised Metric Learners can also be used in a supervised way: the
corresponding supervised algorithm will create a
`ConstrainedDataset` ``X_constrained``
and labels
``y_constraints`` of tuples from a supervised dataset with labels. For
instance if we want to use the algorithm `MMC` on a dataset of points and
labels
(``X`` and ``y``),
we should use ``MMC_Supervised`` (the underlying code will create pairs of
samples from the same class and labels saying that they are similar, and pairs
of samples from a different class and labels saying that they are
dissimilar, before calling `MMC`).

Example:

.. code:: python

    from sklearn.datasets import make_classification

    X, y = make_classification()
    mmc_supervised = MMC_Supervised()
    mmc_supervised.fit_transform(X, y)


.. _performance_ws:

Evaluating the performance of weakly supervised metric learning algorithms
--------------------------------------------------------------------------

To evaluate the performance of a classical supervised algorithm that takes in
an input dataset ``X`` and some labels ``y``, we can compute a cross-validation
score. However, weakly supervised algorithms cannot  ``predict`` on one sample,
so we cannot split on samples to make a training set and a test set the same
way as we do with usual estimators. Instead, metric learning algorithms output
a score on a **tuple** of samples: for instance a similarity score on pairs of
samples. So doing cross-validation scoring for metric learning algorithms
implies to split on **tuples** of samples. Hopefully, `ConstrainedDataset`
allows to do so naturally.

Here is how we would get the cross-validation score for the ``MMC`` algorithm:

.. code:: python

    from sklearn.model_selection import cross_val_score
    cross_val_score(mmc, X_constrained, y_constraints)


Pipelining
----------

Weakly Supervised Learners can also be embedded in scikit-learn pipelines.
However, they can only be combined with Transformers. This is because there
is already supervision from constraints and we cannot add more
supervision that would be used from scikit-learn's supervised estimators.

For instance, you can combine it with another transformer like PCA or KMeans:

.. code:: python

    from sklearn.decomposition import PCA
    from sklearn.clustering import KMeans
    from sklearn.pipeline import make_pipeline

    pipe_pca = make_pipeline(MMC(), PCA())
    pipe_pca.fit(X_constrained, y)
    pipe_clustering = make_pipeline(MMC(), KMeans())
    pipe_clustering.fit(X_constrained, y)

There are also some other things to keep in mind:

- The ``X`` type input of the pipeline should be a `ConstrainedDataset` when
  fitting, but when transforming or predicting it can be an array of samples.
  Therefore, all the following lines are valid:

  .. code:: python

      pipe_pca.transform(X_constrained)
      pipe_pca.fit_transform(X_constrained)
      pipe_pca.transform(X_constrained.X)

- You should also not try to cross-validate those pipelines with scikit-learn's
  cross-validation functions (as their input data is a `ConstrainedDataset`
  which when splitting can contain same points between train and test (but
  of course not the same tuple of points)).

