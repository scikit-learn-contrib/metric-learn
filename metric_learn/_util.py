import numpy as np
import six
from sklearn.utils import check_array


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

  Parameters
  ----------
  tuples : object
    The tuples to check.

  t : int
    The number of elements in a tuple (e.g. 2 for pairs).

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
      Whether to raise an error on np.inf and np.nan in X. This parameter
      does not influence whether y can have np.inf or np.nan values.
      The possibilities are:

      - True: Force all values of X to be finite.
      - False: accept both np.inf and np.nan in X.
      - 'allow-nan':  accept  only  np.nan  values in  X.  Values  cannot  be
        infinite.

  ensure_min_samples : int (default=1)
      Make sure that X has a minimum number of samples in its first
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
      dtype = 'numeric' if preprocessor else None

  name = make_name(estimator, preprocessor)
  context = ' by ' + name if name is not None else ''
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
                       estimator=name,
                       warn_on_dtype=warn_on_dtype)

  if tuples.ndim == 2:  # in this case there is left to check if t is OK
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
    expected_shape = 2 if preprocessor else 3
    raise ValueError("{}D array expected{}. Found {}D array "
                     "instead:\ninput={}.\n"
                     .format(context, expected_shape, tuples.ndim, tuples))
  return tuples


def make_name(estimator, preprocessor):
  """Helper function to create a string with the estimator name and tell if
  it is using a preprocessor. Will return the following for instance:
  NCA + preprocessor: 'NCA's preprocessor'
  NCA + no preprocessor: 'NCA'
  None + preprocessor: 'a preprocessor'
  None + None: None"""
  if estimator is not None:
      with_preprocessor = "'s preprocessor" if preprocessor else ''
      if isinstance(estimator, six.string_types):
          estimator_name = estimator + with_preprocessor
      else:
          estimator_name = estimator.__class__.__name__ + with_preprocessor
  else:
      estimator_name = None if not preprocessor else 'a preprocessor'
  return estimator_name


def check_t(tuples, t, context):
  """Helper function to check that the number of points in each tuple is
  equal to t (e.g. 2 for pairs), and raise a `ValueError` otherwise"""
  if t is not None and tuples.shape[1] != t:
    msg_t = (("Tuples of {} element(s) expected{}. Got tuples of {} "
             "element(s) instead (shape={}):\ninput={}.\n")
             .format(t, context, tuples.shape[1], tuples.shape, tuples))
    raise ValueError(msg_t)


class SimplePreprocessor():

  def __init__(self, X):
    self.X = X

  def __call__(self, indices):
    return self.X[indices]
