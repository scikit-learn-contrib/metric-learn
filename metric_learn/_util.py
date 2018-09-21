import numpy as np
import six
from sklearn.utils import check_array, column_or_1d
from sklearn.utils.validation import (_assert_all_finite,
                                      check_consistent_length)

# hack around lack of axis kwarg in older numpy versions
try:
  np.linalg.norm([[4]], axis=1)
except TypeError:
  def vector_norm(X):
    return np.apply_along_axis(np.linalg.norm, 1, X)
else:
  def vector_norm(X):
    return np.linalg.norm(X, axis=1)


def check_tuples(tuples, preprocessor=False, t=None, dtype="auto",
                 order=None, copy=False, force_all_finite=True,
                 ensure_min_samples=1, ensure_min_features=1,
                 warn_on_dtype=False, estimator=None):
  """Check that `tuples` is a valid array of tuples.

  Depending on whether a preprocessor is available or not, `tuples` should be:
      - a 3D array of formed tuples or a 2D array of tuples of indicators if a
        preprocessor is available
      - a 3D array of formed tuples if no preprocessor is available

  The number of elements in a tuple (e.g. 2 for pairs) should be the right
  one, specified by the parameter `t`.
  `check_tuples` will then convert the tuples to the right format as
  `sklearn.utils.validation.check_array` would do. See
  `sklearn.utils.validation.check_array` for more details.

  Parameters
  ----------
  tuples : object
    The tuples to check.

  t : int or None (default=None)
    The number of elements to ensure there is in every tuple (e.g. 2 for
    pairs). If None, the number of tuples is not checked.

  dtype : string, type, list of types or None (default="auto")
      Data type of result. If None, the dtype of the input is preserved.
      If "numeric", dtype is preserved unless array.dtype is object.
      If dtype is a list of types, conversion on the first type is only
      performed if the dtype of the input is not in the list. If
      "auto", will we be set to "numeric" if `preprocessor=True`,
      else to None.

  order : 'F', 'C' or None (default=None)
      Whether an array will be forced to be fortran or c-style.

  copy : boolean (default=False)
      Whether a forced copy will be triggered. If copy=False, a copy might
      be triggered by a conversion.

  force_all_finite : boolean or 'allow-nan', (default=True)
      Whether to raise an error on np.inf and np.nan in `tuples`.
      This parameter does not influence whether y can have np.inf or np.nan
      values. The possibilities are:

      - True: Force all values of `tuples` to be finite.
      - False: accept both np.inf and np.nan in `tuples`.
      - 'allow-nan':  accept  only  np.nan  values in  `tuples`.  Values
        cannot  be infinite.

  ensure_min_samples : int (default=1)
      Make sure that `tuples` has a minimum number of samples in its first
      axis (rows for a 2D array).

  ensure_min_features : int (default=1)
      Only used when using no preprocessor. Make sure that each point in the 3D
      array of tuples has some minimum number of features (axis=2). The default
      value of 1 rejects empty datasets. This check is only enforced when X has
      effectively 3 dimensions. Setting to 0 disables this check.

  warn_on_dtype : boolean (default=False)
      Raise DataConversionWarning if the dtype of the input data structure
      does not match the requested dtype, causing a memory copy.

  estimator : str or estimator instance (default=None)
      If passed, include the name of the estimator in warning messages.

  Returns
  -------
  tuples_valid : object
    The validated tuples.
  """
  if dtype == "auto":
    dtype = 'numeric' if not preprocessor else None

  context = make_context(estimator)
  tuples = check_array(tuples, dtype=dtype, accept_sparse=False, copy=copy,
                       force_all_finite=force_all_finite,
                       order=order,
                       ensure_2d=False,  # tuples can be 2D or 3D
                       allow_nd=True,
                       ensure_min_samples=ensure_min_samples,
                       # ensure_min_features only works if ndim=2, so we will
                       # have to check again if input is 3D (see below)
                       ensure_min_features=0,
                       # if 2D and preprocessor, no notion of
                       # "features". If 3D and no preprocessor, min_features
                       # is checked below
                       estimator=estimator,
                       warn_on_dtype=warn_on_dtype)

  if tuples.ndim == 2 and preprocessor:  # in this case there is left to check
      # if t is OK
    check_t(tuples, t, context)
  elif tuples.ndim == 3:
    # if the dimension is 3 we still have to check that the num_features is OK
    if ensure_min_features > 0:
      n_features = tuples.shape[2]
      if n_features < ensure_min_features:
        raise ValueError("Found array with {} feature(s) (shape={}) while"
                         " a minimum of {} is required{}."
                         .format(n_features, tuples.shape, ensure_min_features,
                                 context))
    # then we should also check that t is OK
    check_t(tuples, t, context)
  else:
    expected_shape = ('2D array of indicators or 3D array of formed tuples'
                      if preprocessor else '3D array of formed tuples')
    with_prep = ' when using a preprocessor' if preprocessor else ''
    should_use_prep = ' and/or use a preprocessor' if not preprocessor else ''
    raise ValueError("{} expected{}{}. Found {}D array "
                     "instead:\ninput={}. Reshape your data{}.\n"
                     .format(expected_shape, context, with_prep,
                             tuples.ndim, tuples, should_use_prep))
  return tuples


def check_tuples_y(tuples, y, preprocessor=False, t=None, dtype="auto",
                   order=None, copy=False, force_all_finite=True,
                   multi_output=False, ensure_min_samples=1,
                   ensure_min_features=1, y_numeric=False,
                   warn_on_dtype=False, estimator=None):
  """Input validation for standard estimators.

  Adapted from `sklearn.utils.validation.check_X_y`.
  Checks tuples with `check_tuples`, and checks that the size of `y` and
  `tuples` are consistent. In addition, standard input checks are only
  applied to y, such as checking that y does not have np.nan or np.inf
  targets. For multi-label y, set multi_output=True to allow 2d and sparse y.

  Parameters
  ----------
  tuples : 3D array of formed tuples or 2D array of tuples indicators
      Input tuples.

  y : nd-array, list or sparse matrix
      Labels.

  preprocessor : boolean
      Whether a preprocessor is available or not (the input format depends
      on that) (See `check_tuples` for more information)

  dtype : string, type, list of types or None (default="numeric")
      Data type of result. If None, the dtype of the input is preserved.
      If "numeric", dtype is preserved unless array.dtype is object.
      If dtype is a list of types, conversion on the first type is only
      performed if the dtype of the input is not in the list.

  order : 'F', 'C' or None (default=None)
      Whether an array will be forced to be fortran or c-style.

  copy : boolean (default=False)
      Whether a forced copy will be triggered. If copy=False, a copy might
      be triggered by a conversion.

  force_all_finite : boolean (default=True)
      Whether to raise an error on np.inf and np.nan in X. This parameter
      does not influence whether y can have np.inf or np.nan values.

  multi_output : boolean (default=False)
      Whether to allow 2-d y (array or sparse matrix). If false, y will be
      validated as a vector. y cannot have np.nan or np.inf values if
      multi_output=True.

  ensure_min_samples : int (default=1)
      Make sure that X has a minimum number of samples in its first
      axis (rows for a 2D array).

  ensure_min_features : int (default=1)
      Make sure that the tuples has some minimum number of features
      The default value of 1 rejects empty datasets.
      Setting to 0 disables this check.

  y_numeric : boolean (default=False)
      Whether to ensure that y has a numeric type. If dtype of y is object,
      it is converted to float64. Should only be used for regression
      algorithms.

  warn_on_dtype : boolean (default=False)
      Raise DataConversionWarning if the dtype of the input data structure
      does not match the requested dtype, causing a memory copy.

  estimator : str or estimator instance (default=None)
      If passed, include the name of the estimator in warning messages.

  Returns
  -------
  tuples_converted : object
      The converted and validated tuples.

  y_converted : object
      The converted and validated y.
  """
  tuples = check_tuples(tuples,
                        preprocessor=preprocessor,
                        t=t,
                        dtype=dtype,
                        order=order, copy=copy,
                        force_all_finite=force_all_finite,
                        ensure_min_samples=ensure_min_samples,
                        ensure_min_features=ensure_min_features,
                        warn_on_dtype=warn_on_dtype,
                        estimator=estimator)
  if multi_output:
      y = check_array(y, 'csr', force_all_finite=True, ensure_2d=False,
                      dtype=None)
  else:
      y = column_or_1d(y, warn=True)
      _assert_all_finite(y)
  if y_numeric and y.dtype.kind == 'O':
      y = y.astype(np.float64)

  check_consistent_length(tuples, y)

  return tuples, y


def check_points(points, preprocessor=False, accept_sparse=False,
                 dtype="auto", order=None, copy=False, force_all_finite=True,
                 ensure_min_samples=1, ensure_min_features=1,
                 warn_on_dtype=False, estimator=None):
  """Checks that `points` is a valid dataset of points

  Depending on whether a preprocessor is available or not, `points` should
  be:
      - a 2D array of formed points or a 1D array of indicators if a
        preprocessor is available
      - a 3D array of formed tuples if no preprocessor is available

  `check_points` will then convert the points to the right format as
  `sklearn.utils.validation.check_array` would do. See
  `sklearn.utils.validation.check_array` for more details.

  Parameters
  ----------
  points : object
      Input object to check / convert.

  accept_sparse : string, boolean or list/tuple of strings (default=False)
      String[s] representing allowed sparse matrix formats, such as 'csc',
      'csr', etc. If the input is sparse but not in the allowed format,
      it will be converted to the first listed format. True allows the input
      to be any format. False means that a sparse matrix input will
      raise an error.

  dtype : string, type, list of types or None (default="numeric")
      Data type of result. If None, the dtype of the input is preserved.
      If "numeric", dtype is preserved unless points.dtype is object.
      If dtype is a list of types, conversion on the first type is only
      performed if the dtype of the input is not in the list.

  order : 'F', 'C' or None (default=None)
      Whether an array will be forced to be fortran or c-style.
          When order is None (default), then if copy=False, nothing is ensured
      about the memory layout of the output array; otherwise (copy=True)
      the memory layout of the returned array is kept as close as possible
      to the original array.

  copy : boolean (default=False)
      Whether a forced copy will be triggered. If copy=False, a copy might
      be triggered by a conversion.

  force_all_finite : boolean (default=True)
      Whether to raise an error on np.inf and np.nan in `points`.

  ensure_min_samples : int (default=1)
      Make sure that the array has a minimum number of samples in its first
      axis (rows for a 2D array). Setting to 0 disables this check.

  ensure_min_features : int (default=1)
      Make sure that the 2D array has some minimum number of features
      (columns). The default value of 1 rejects empty datasets.
      This check is only enforced when the input data has effectively 2
      dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
      disables this check.

  warn_on_dtype : boolean (default=False)
      Raise DataConversionWarning if the dtype of the input data structure
      does not match the requested dtype, causing a memory copy.

  estimator : str or estimator instance (default=None)
      If passed, include the name of the estimator in warning messages.

  Returns
  -------
  points_converted : object
      The converted and validated array of points.

  """
  if dtype == "auto":
      dtype = 'numeric' if preprocessor is not None else None

  context = make_context(estimator)
  points = check_array(points, dtype=dtype, accept_sparse=accept_sparse,
                       copy=copy,
                       force_all_finite=force_all_finite,
                       order=order,
                       ensure_2d=False,  # input can be 1D
                       allow_nd=True,  # true, to throw custom error message
                       ensure_min_samples=ensure_min_samples,
                       ensure_min_features=ensure_min_features,
                       estimator=estimator,
                       warn_on_dtype=warn_on_dtype)
  if (points.ndim == 1 and preprocessor) or points.ndim == 2:
    return points
  else:
    expected_shape = ('1D array of indicators or 2D array of formed points'
                      if preprocessor else '2D array of formed points')
    with_prep = ' when using a preprocessor' if preprocessor else ''
    should_use_prep = ' and/or use a preprocessor' if not preprocessor else ''
    raise ValueError("{} expected{}{}. Found {}D array "
                     "instead:\ninput={}. Reshape your data{}.\n"
                     .format(expected_shape, context, with_prep,
                             points.ndim, points, should_use_prep))
  return points


def check_points_y(points, y, preprocessor=False, accept_sparse=False,
                   dtype="auto",
                   order=None, copy=False, force_all_finite=True,
                   multi_output=False, ensure_min_samples=1,
                   ensure_min_features=1, y_numeric=False,
                   warn_on_dtype=False, estimator=None):
  """Input validation for standard estimators.

  Checks `points` and `y` for consistent length, enforces `points` is a 2d
  array of formed points or 1d array of indicators of points, and y is 1d.
  Standard input checks are only applied to y, such as checking that y
  does not have np.nan or np.inf targets. For multi-label y, set
  multi_output=True to allow 2d and sparse y.  If the dtype of `points` is
  object, attempt converting to float, raising on failure.
  Adapted from :func:`sklearn.utils.validation.check_X_y`.

  Parameters
  ----------
  points : nd-array, list or sparse matrix
      Input data.

  y : nd-array, list or sparse matrix
      Labels.

  accept_sparse : string, boolean or list of string (default=False)
      String[s] representing allowed sparse matrix formats, such as 'csc',
      'csr', etc. If the input is sparse but not in the allowed format,
      it will be converted to the first listed format. True allows the input
      to be any format. False means that a sparse matrix input will
      raise an error.

      .. deprecated:: 0.19
         Passing 'None' to parameter ``accept_sparse`` in methods is
         deprecated in version 0.19 "and will be removed in 0.21. Use
         ``accept_sparse=False`` instead.

  dtype : string, type, list of types or None (default="numeric")
      Data type of result. If None, the dtype of the input is preserved.
      If "numeric", dtype is preserved unless array.dtype is object.
      If dtype is a list of types, conversion on the first type is only
      performed if the dtype of the input is not in the list.

  order : 'F', 'C' or None (default=None)
      Whether an array will be forced to be fortran or c-style.

  copy : boolean (default=False)
      Whether a forced copy will be triggered. If copy=False, a copy might
      be triggered by a conversion.

  force_all_finite : boolean (default=True)
      Whether to raise an error on np.inf and np.nan in `points`.
      This parameter does not influence whether y can have np.inf or np.nan
      values.

  ensure_2d : boolean (default=True)
      Whether to make `points` at least 2d.

  allow_nd : boolean (default=False)
      Whether to allow points.ndim > 2.

  multi_output : boolean (default=False)
      Whether to allow 2-d y (array or sparse matrix). If false, y will be
      validated as a vector. y cannot have np.nan or np.inf values if
      multi_output=True.

  ensure_min_samples : int (default=1)
      Make sure that `points` has a minimum number of samples in its first
      axis (rows for a 2D array).

  ensure_min_features : int (default=1)
      Make sure that the 2D array has some minimum number of features
      (columns). The default value of 1 rejects empty datasets.
      This check is only enforced when `points` has effectively 2 dimensions or
      is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
      this check.

  y_numeric : boolean (default=False)
      Whether to ensure that y has a numeric type. If dtype of y is object,
      it is converted to float64. Should only be used for regression
      algorithms.

  warn_on_dtype : boolean (default=False)
      Raise DataConversionWarning if the dtype of the input data structure
      does not match the requested dtype, causing a memory copy.

  estimator : str or estimator instance (default=None)
      If passed, include the name of the estimator in warning messages.

  Returns
  -------
  points_converted : object
      The converted and validated points.

  y_converted : object
      The converted and validated y.
  """
  points = check_points(points, preprocessor=preprocessor,
                        accept_sparse=accept_sparse,
                        dtype=dtype,
                        order=order, copy=copy,
                        force_all_finite=force_all_finite,
                        ensure_min_samples=ensure_min_samples,
                        ensure_min_features=ensure_min_features,
                        warn_on_dtype=warn_on_dtype,
                        estimator=estimator)
  if multi_output:
      y = check_points(y, 'csr', force_all_finite=True, dtype=None,
                       preprocessor=preprocessor)
  else:
      y = column_or_1d(y, warn=True)
      _assert_all_finite(y)
  if y_numeric and y.dtype.kind == 'O':
      y = y.astype(np.float64)

  check_consistent_length(points, y)

  return points, y


def preprocess_tuples(tuples, preprocessor, estimator=None):
  """form tuples if there is a preprocessor else keep them as such (assumes
  that check_tuples has already been called)"""
  if estimator is not None:
    estimator_name = make_name(estimator) + (' after the preprocessor '
                                             'has been applied')
  else:
    estimator_name = ('objects that will use preprocessed tuples')

  if preprocessor is not None and tuples.ndim == 2:
    print("Preprocessing tuples...")
    tuples = np.column_stack([preprocessor(tuples[:, i])[:, np.newaxis] for
                              i in range(tuples.shape[1])])
  tuples = check_tuples(tuples, preprocessor=False, estimator=estimator_name)
  # normally we shouldn't need to enforce the t, since a preprocessor shouldn't
  # be able to transform a t tuples array into a t' tuples array
  return tuples


def preprocess_points(points, preprocessor, estimator=None):
  """form points if there is a preprocessor else keep them as such (assumes
  that check_points has already been called)"""
  if estimator is not None:
    estimator_name = make_name(estimator) + (' after the preprocessor '
                                             'has been applied')
  else:
    estimator_name = ('objects that will use preprocessed points')

  if preprocessor is not None and points.ndim == 1:
    print("Preprocessing points...")
    points = preprocessor(points)
  points = check_points(points, preprocessor=False, estimator=estimator_name)
  return points


def make_context(estimator):
  """Helper function to create a string with the estimator name.
  Taken from check_array function in scikit-learn.
  Will return the following for instance:
  NCA: ' by NCA'
  'NCA': ' by NCA'
  None: ''
  """
  estimator_name = make_name(estimator)
  context = (' by ' + estimator_name) if estimator_name is not None else ''
  return context


def make_name(estimator):
  """Helper function that returns the name of estimator or the given string
  if a string is given
  """
  if estimator is not None:
    if isinstance(estimator, six.string_types):
      estimator_name = estimator
    else:
      estimator_name = estimator.__class__.__name__
  else:
    estimator_name = None
  return estimator_name


def check_t(tuples, t, context):
  """Helper function to check that the number of points in each tuple is
  equal to t (e.g. 2 for pairs), and raise a `ValueError` otherwise"""
  if t is not None and tuples.shape[1] != t:
    msg_t = (("Tuples of {} element(s) expected{}. Got tuples of {} "
             "element(s) instead (shape={}):\ninput={}.\n")
             .format(t, context, tuples.shape[1], tuples.shape, tuples))
    raise ValueError(msg_t)


class ArrayIndexer():

  def __init__(self, X):
    self.X = X

  def __call__(self, indices):
    return self.X[indices]
