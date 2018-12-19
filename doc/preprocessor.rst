:ref:`preprocessor`

============
Preprocessor
============

Estimators in metric-learn all have a ``preprocessor`` option at instantiation.
Filling this argument allows them to take more compact input representation
when fitting, predicting etc...

Two types of objects can be put in this argument:

Array-like
----------
You can specify ``preprocessor=X`` where ``X`` is an array-like containing the
dataset of points. In this case, the estimator will be able to take as
inputs an array-like of indices, replacing under the hood each index by the
corresponding sample.


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
Instead, you can provide a callable in the argument ``preprocessor``.
Then the estimator will accept indicators of points instead of points.
Under the hood, the estimator will call this callable on the indicators you
provide as input when fitting, predicting etc...
Using a callable can be really useful to represent lazily a dataset of
images stored on the file system for instance.
The callable should take as an input an array-like, and return a 2D
array-like. For supervised learners it will be applied on the whole array of
indicators at once, and for weakly supervised learners it will be applied
on each column of the array of tuples.

Example with a supervised metric learner:

The callable should take as input an array-like, and return a 2D array-like.

>>> def find_images(arr):
>>>     X = np.array([[-0.7 , -0.23],
>>>                   [-0.43, -0.49],
>>>                   [ 0.14, -0.37]])  # array of 3 samples of 2 features
>>>     result = []
>>>     for img_path in arr:
>>>         result.append(X[int(img_path[3:5])])
>>>         # transforms 'img01.png' into X[1]
>>>     return np.array(result)
>>> images_paths = ['img01.png', 'img00.png', 'img02.png']
>>> y = np.array([1, 0, 1])
>>>
>>> nca = NCA(preprocessor=find_images)
>>> nca.fit(images_paths, y)
>>> # under the hood preprocessor(indicators) will be called


Example with a weakly supervised metric learner:

The given callable should take as input an array-like, and return a
2D array-like. It will be called on each column of the input tuples of
indicators.

>>> def find_images(arr):
>>>     X = np.array([[-0.7 , -0.23],
>>>                   [-0.43, -0.49],
>>>                   [ 0.14, -0.37]])  # array of 3 samples of 2 features
>>>     result = []
>>>     for img_path in arr:
>>>         result.append(X[int(img_path[3:5])])
>>>         # transforms 'img01.png' into X[1]
>>>     return np.array(result)
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
 stays valid (2D array-like for ``X`` for supervised learners and 3D
 array-like of tuples for weakly supervised learners).

 Example: This would work:

 >>> from metric_learn import MMC
 >>> X = np.array([[-0.7 , -0.23],
 >>>               [-0.43, -0.49],
 >>>               [ 0.14, -0.37]])  # array of 3 samples of 2 features
 >>> pairs = np.array([[[ 0.14, -0.37], [-0.7 , -0.23]],
 >>>                   [[-0.43, -0.49], [-0.7 , -0.23]]])
 >>> y_pairs = np.array([1, -1])
 >>>
 >>> mmc = MMC(preprocessor=X)
 >>> mmc.fit(pairs, y_pairs)
