:ref:`preprocessor`

============
Preprocessor
============

Estimators in metric-learn all have a ``preprocessor`` option at instantiation.
Filling this argument allows them to take more compact input representation
when fitting, predicting etc...

If ``preprocessor=None``, no preprocessor will be used and the user must
provide the classical representation to the fit/predict/score/etc... methods of
the estimators (see the documentation of the particular estimator to know what
type of input it accepts). Otherwise, two types of objects can be put in this
argument:

Array-like
----------
You can specify ``preprocessor=X`` where ``X`` is an array-like containing the
dataset of points. In this case, the fit/predict/score/etc... methods of the
estimator will be able to take as inputs an array-like of indices, replacing
under the hood each index by the corresponding sample.


Example with a supervised metric learner:

>>> from metric_learn import NCA
>>>
>>> X = np.array([[-0.7 , -0.23],
>>>               [-0.43, -0.49],
>>>               [ 0.14, -0.37]])  # array of 3 samples of 2 features
>>> points_indices = np.array([2, 0, 1, 0])
>>> y = np.array([1, 0, 1, 1])
>>>
>>> nca = NCA(preprocessor=X)
>>> nca.fit(points_indices, y)
>>> # under the hood the algorithm will create
>>> # points = np.array([[ 0.14, -0.37],
>>> #                    [-0.7 , -0.23],
>>> #                    [-0.43, -0.49],
>>> #                    [ 0.14, -0.37]]) and fit on it


Example with a weakly supervised metric learner:

>>> from metric_learn import MMC
>>> X = np.array([[-0.7 , -0.23],
>>>               [-0.43, -0.49],
>>>               [ 0.14, -0.37]])  # array of 3 samples of 2 features
>>> pairs_indices = np.array([[2, 0], [1, 0]])
>>> y_pairs = np.array([1, -1])
>>>
>>> mmc = MMC(preprocessor=X)
>>> mmc.fit(pairs_indices, y_pairs)
>>> # under the hood the algorithm will create
>>> # pairs = np.array([[[ 0.14, -0.37], [-0.7 , -0.23]],
>>> #                    [[-0.43, -0.49], [-0.7 , -0.23]]]) and fit on it

Callable
--------
Instead, you can provide a callable in the argument ``preprocessor``. Then the
estimator will accept indicators of points instead of points. Under the hood,
the estimator will call this callable on the indicators you provide as input
when fitting, predicting etc... Using a callable can be really useful to
represent lazily a dataset of images stored on the file system for instance.
The callable should take as an input a 1D array-like, and return a 2D
array-like. For supervised learners it will be applied on the whole 1D array of
indicators at once, and for weakly supervised learners it will be applied on
each column of the 2D array of tuples.

Example with a supervised metric learner:

>>> def find_images(file_paths):
>>>    # each file contains a small image to use as an input datapoint
>>>    return np.row_stack([imread(f).ravel() for f in file_paths])
>>>
>>> nca = NCA(preprocessor=find_images)
>>> nca.fit(['img01.png', 'img00.png', 'img02.png'], [1, 0, 1])
>>> # under the hood preprocessor(indicators) will be called


Example with a weakly supervised metric learner:

>>> pairs_images_paths = [['img02.png', 'img00.png'],
>>>                       ['img01.png', 'img00.png']]
>>> y_pairs = np.array([1, -1])
>>>
>>> mmc = NCA(preprocessor=find_images)
>>> mmc.fit(pairs_images_paths, y_pairs)
>>> # under the hood preprocessor(pairs_indicators[i]) will be called for each
>>> # i in [0, 1]


.. note:: Note that when you fill the ``preprocessor`` option, it allows you
 to give more compact inputs, but the classical way of providing inputs
 stays valid (2D array-like for supervised learners and 3D array-like of
 tuples for weakly supervised learners). If a classical input
 is provided, the metric learner will not use the preprocessor.

 Example: This will work:

 >>> from metric_learn import MMC
 >>> def preprocessor_wip(array):
 >>>    raise NotImplementedError("This preprocessor does nothing yet.")
 >>>
 >>> pairs = np.array([[[ 0.14, -0.37], [-0.7 , -0.23]],
 >>>                   [[-0.43, -0.49], [-0.7 , -0.23]]])
 >>> y_pairs = np.array([1, -1])
 >>>
 >>> mmc = MMC(preprocessor=preprocessor_wip)
 >>> mmc.fit(pairs, y_pairs)  # preprocessor_wip will not be called here
