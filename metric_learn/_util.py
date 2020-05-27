import numpy as np
from numpy.linalg import LinAlgError
from sklearn.datasets import make_spd_matrix
from sklearn.decomposition import PCA
from sklearn.utils import check_array
from sklearn.utils.validation import check_X_y, check_random_state
from .exceptions import PreprocessorError, NonPSDError
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.linalg import pinvh, eigh
import sys
import time
import warnings

# hack around lack of axis kwarg in older numpy versions
try:
  np.linalg.norm([[4]], axis=1)
except TypeError:
  def vector_norm(X):
    return np.apply_along_axis(np.linalg.norm, 1, X)
else:
  def vector_norm(X):
    return np.linalg.norm(X, axis=1)


def check_input(input_data, y=None, preprocessor=None,
                type_of_inputs='classic', tuple_size=None, accept_sparse=False,
                dtype='numeric', order=None,
                copy=False, force_all_finite=True,
                multi_output=False, ensure_min_samples=1,
                ensure_min_features=1, y_numeric=False, estimator=None):
  """Checks that the input format is valid, and converts it if specified
  (this is the equivalent of scikit-learn's `check_array` or `check_X_y`).
  All arguments following tuple_size are scikit-learn's `check_X_y`
  arguments that will be enforced on the data and labels array. If
  indicators are given as an input data array, the returned data array
  will be the formed points/tuples, using the given preprocessor.

  Parameters
  ----------
  input : array-like
    The input data array to check.

  y : array-like
    The input labels array to check.

  preprocessor : callable (default=`None`)
    The preprocessor to use. If None, no preprocessor is used.

  type_of_inputs : `str` {'classic', 'tuples'}
    The type of inputs to check. If 'classic', the input should be
    a 2D array-like of points or a 1D array like of indicators of points. If
    'tuples', the input should be a 3D array-like of tuples or a 2D
    array-like of indicators of tuples.

  accept_sparse : `bool`
    Set to true to allow sparse inputs (only works for sparse inputs with
    dim < 3).

  tuple_size : int
    The number of elements in a tuple (e.g. 2 for pairs).

  dtype : string, type, list of types or None (default='numeric')
    Data type of result. If None, the dtype of the input is preserved.
    If 'numeric', dtype is preserved unless array.dtype is object.
    If dtype is a list of types, conversion on the first type is only
    performed if the dtype of the input is not in the list.

  order : 'F', 'C' or None (default=`None`)
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
    Make sure that the 2D array has some minimum number of features
    (columns). The default value of 1 rejects empty datasets.
    This check is only enforced when X has effectively 2 dimensions or
    is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
    this check.

  estimator : str or estimator instance (default=`None`)
    If passed, include the name of the estimator in warning messages.

  Returns
  -------
  X : `numpy.ndarray`
    The checked input data array.

  y: `numpy.ndarray` (optional)
    The checked input labels array.
  """

  context = make_context(estimator)

  args_for_sk_checks = dict(accept_sparse=accept_sparse,
                            dtype=dtype, order=order,
                            copy=copy, force_all_finite=force_all_finite,
                            ensure_min_samples=ensure_min_samples,
                            ensure_min_features=ensure_min_features,
                            estimator=estimator)

  # We need to convert input_data into a numpy.ndarray if possible, before
  # any further checks or conversions, and deal with y if needed. Therefore
  # we use check_array/check_X_y with fixed permissive arguments.
  if y is None:
    input_data = check_array(input_data, ensure_2d=False, allow_nd=True,
                             copy=False, force_all_finite=False,
                             accept_sparse=True, dtype=None,
                             ensure_min_features=0, ensure_min_samples=0)
  else:
    input_data, y = check_X_y(input_data, y, ensure_2d=False, allow_nd=True,
                              copy=False, force_all_finite=False,
                              accept_sparse=True, dtype=None,
                              ensure_min_features=0, ensure_min_samples=0,
                              multi_output=multi_output,
                              y_numeric=y_numeric)

  if type_of_inputs == 'classic':
    input_data = check_input_classic(input_data, context, preprocessor,
                                     args_for_sk_checks)

  elif type_of_inputs == 'tuples':
    input_data = check_input_tuples(input_data, context, preprocessor,
                                    args_for_sk_checks, tuple_size)

    # if we have y and the input data are pairs, we need to ensure
    # the labels are in [-1, 1]:
    if y is not None and input_data.shape[1] == 2:
      check_y_valid_values_for_pairs(y)

  else:
    raise ValueError("Unknown value {} for type_of_inputs. Valid values are "
                     "'classic' or 'tuples'.".format(type_of_inputs))

  return input_data if y is None else (input_data, y)


def check_input_tuples(input_data, context, preprocessor, args_for_sk_checks,
                       tuple_size):
  preprocessor_has_been_applied = False
  if input_data.ndim == 2:
    if preprocessor is not None:
      input_data = preprocess_tuples(input_data, preprocessor)
      preprocessor_has_been_applied = True
    else:
      make_error_input(201, input_data, context)
  elif input_data.ndim == 3:
    pass
  else:
    if preprocessor is not None:
      make_error_input(420, input_data, context)
    else:
      make_error_input(200, input_data, context)
  input_data = check_array(input_data, allow_nd=True, ensure_2d=False,
                           **args_for_sk_checks)
  # we need to check num_features because check_array does not check it
  # for 3D inputs:
  if args_for_sk_checks['ensure_min_features'] > 0:
    n_features = input_data.shape[2]
    if n_features < args_for_sk_checks['ensure_min_features']:
      raise ValueError("Found array with {} feature(s) (shape={}) while"
                       " a minimum of {} is required{}."
                       .format(n_features, input_data.shape,
                               args_for_sk_checks['ensure_min_features'],
                               context))
  #  normally we don't need to check_tuple_size too because tuple_size
  # shouldn't be able to be modified by any preprocessor
  if input_data.ndim != 3:
    # we have to ensure this because check_array above does not
    if preprocessor_has_been_applied:
      make_error_input(211, input_data, context)
    else:
      make_error_input(201, input_data, context)
  check_tuple_size(input_data, tuple_size, context)
  return input_data


def check_input_classic(input_data, context, preprocessor, args_for_sk_checks):
  preprocessor_has_been_applied = False
  if input_data.ndim == 1:
    if preprocessor is not None:
      input_data = preprocess_points(input_data, preprocessor)
      preprocessor_has_been_applied = True
    else:
      make_error_input(101, input_data, context)
  elif input_data.ndim == 2:
    pass  # OK
  else:
    if preprocessor is not None:
      make_error_input(320, input_data, context)
    else:
      make_error_input(100, input_data, context)

  input_data = check_array(input_data, allow_nd=True, ensure_2d=False,
                           **args_for_sk_checks)
  if input_data.ndim != 2:
    # we have to ensure this because check_array above does not
    if preprocessor_has_been_applied:
      make_error_input(111, input_data, context)
    else:
      make_error_input(101, input_data, context)
  return input_data


def make_error_input(code, input_data, context):
  code_str = {'expected_input': {'1': '2D array of formed points',
                                 '2': '3D array of formed tuples',
                                 '3': ('1D array of indicators or 2D array of '
                                       'formed points'),
                                 '4': ('2D array of indicators or 3D array '
                                       'of formed tuples')},
              'additional_context': {'0': '',
                                     '2': ' when using a preprocessor',
                                     '1': (' after the preprocessor has been '
                                           'applied')},
              'possible_preprocessor': {'0': '',
                                        '1': ' and/or use a preprocessor'
                                        }}
  code_list = str(code)
  err_args = dict(expected_input=code_str['expected_input'][code_list[0]],
                  additional_context=code_str['additional_context']
                  [code_list[1]],
                  possible_preprocessor=code_str['possible_preprocessor']
                  [code_list[2]],
                  input_data=input_data, context=context,
                  found_size=input_data.ndim)
  err_msg = ('{expected_input} expected'
             '{context}{additional_context}. Found {found_size}D array '
             'instead:\ninput={input_data}. Reshape your data'
             '{possible_preprocessor}.\n')
  raise ValueError(err_msg.format(**err_args))


def preprocess_tuples(tuples, preprocessor):
  try:
    tuples = np.column_stack([preprocessor(tuples[:, i])[:, np.newaxis] for
                              i in range(tuples.shape[1])])
  except Exception as e:
    raise PreprocessorError(e)
  return tuples


def preprocess_points(points, preprocessor):
  """form points if there is a preprocessor else keep them as such (assumes
  that check_points has already been called)"""
  try:
    points = preprocessor(points)
  except Exception as e:
    raise PreprocessorError(e)
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
    if isinstance(estimator, str):
      estimator_name = estimator
    else:
      estimator_name = estimator.__class__.__name__
  else:
    estimator_name = None
  return estimator_name


def check_tuple_size(tuples, tuple_size, context):
  """Helper function to check that the number of points in each tuple is
  equal to tuple_size (e.g. 2 for pairs), and raise a `ValueError` otherwise"""
  if tuple_size is not None and tuples.shape[1] != tuple_size:
    msg_t = (("Tuples of {} element(s) expected{}. Got tuples of {} "
             "element(s) instead (shape={}):\ninput={}.\n")
             .format(tuple_size, context, tuples.shape[1], tuples.shape,
                     tuples))
    raise ValueError(msg_t)


def check_y_valid_values_for_pairs(y):
  """Checks that y values are in [-1, 1]"""
  if not np.array_equal(np.abs(y), np.ones_like(y)):
    raise ValueError("When training on pairs, the labels (y) should contain "
                     "only values in [-1, 1]. Found an incorrect value.")


class ArrayIndexer:

  def __init__(self, X):
    # we check the array-like preprocessor here, and we as much permissive
    # as possible (because the user will check for the desired
    # format with arguments in check_input, and only this latter function
    # should return the appropriate errors). We do this only to have a numpy
    # array object which can be indexed by another numpy array object.
    X = check_array(X,
                    accept_sparse=True, dtype=None,
                    force_all_finite=False,
                    ensure_2d=False, allow_nd=True,
                    ensure_min_samples=0, ensure_min_features=0,
                    estimator=None)
    self.X = X

  def __call__(self, indices):
    return self.X[indices]


def check_collapsed_pairs(pairs):
    num_ident = (vector_norm(pairs[:, 0] - pairs[:, 1]) < 1e-9).sum()
    if num_ident:
      raise ValueError("{} collapsed pairs found (where the left element is "
                       "the same as the right element), out of {} pairs "
                       "in total.".format(num_ident, pairs.shape[0]))


def _check_sdp_from_eigen(w, tol=None):
  """Checks if some of the eigenvalues given are negative, up to a tolerance
  level, with a default value of the tolerance depending on the eigenvalues.
  It also returns whether the matrix is positive definite, up to the above
  tolerance.

  Parameters
  ----------
  w : array-like, shape=(n_eigenvalues,)
    Eigenvalues to check for non semidefinite positiveness.

  tol : positive `float`, optional
    Absolute eigenvalues below tol are considered zero. If
    tol is None, and eps is the epsilon value for datatype of w, then tol
    is set to abs(w).max() * len(w) * eps.

  Returns
  -------
  is_definite : bool
    Whether the matrix is positive definite or not.

  See Also
  --------
  np.linalg.matrix_rank for more details on the choice of tolerance (the same
    strategy is applied here)
  """
  if tol is None:
    tol = np.abs(w).max() * len(w) * np.finfo(w.dtype).eps
  if tol < 0:
    raise ValueError("tol should be positive.")
  if any(w < - tol):
    raise NonPSDError()
  if any(abs(w) < tol):
    return False
  return True


def components_from_metric(metric, tol=None):
  """Returns the transformation matrix from the Mahalanobis matrix.

  Returns the transformation matrix from the Mahalanobis matrix, i.e. the
  matrix L such that metric=L.T.dot(L).

  Parameters
  ----------
  metric : symmetric `np.ndarray`, shape=(d x d)
    The input metric, from which we want to extract a transformation matrix.

  tol : positive `float`, optional
    Eigenvalues of `metric` between 0 and - tol are considered zero. If tol is
    None, and w_max is `metric`'s largest eigenvalue, and eps is the epsilon
    value for datatype of w, then tol is set to w_max * metric.shape[0] * eps.

  Returns
  -------
  L : np.ndarray, shape=(d x d)
    The transformation matrix, such that L.T.dot(L) == metric.
  """
  if not np.allclose(metric, metric.T):
    raise ValueError("The input metric should be symmetric.")
  # If M is diagonal, we will just return the elementwise square root:
  if np.array_equal(metric, np.diag(np.diag(metric))):
    _check_sdp_from_eigen(np.diag(metric), tol)
    return np.diag(np.sqrt(np.maximum(0, np.diag(metric))))
  else:
    try:
      # if `M` is positive semi-definite, it will admit a Cholesky
      # decomposition: L = cholesky(M).T
      return np.linalg.cholesky(metric).T
    except LinAlgError:
      # However, currently np.linalg.cholesky does not support indefinite
      # matrices. So if the latter does not work we will return L = V.T w^(
      # -1/2), with M = V*w*V.T being the eigenvector decomposition of M with
      # the eigenvalues in the diagonal matrix w and the columns of V being the
      # eigenvectors.
      w, V = np.linalg.eigh(metric)
      _check_sdp_from_eigen(w, tol)
      return V.T * np.sqrt(np.maximum(0, w[:, None]))


def validate_vector(u, dtype=None):
  # replica of scipy.spatial.distance._validate_vector, for making scipy
  # compatible functions on vectors (such as distances computations)
  u = np.asarray(u, dtype=dtype, order='c').squeeze()
  # Ensure values such as u=1 and u=[1] still return 1-D arrays.
  u = np.atleast_1d(u)
  if u.ndim > 1:
    raise ValueError("Input vector should be 1-D.")
  return u


def _initialize_components(n_components, input, y=None, init='auto',
                           verbose=False, random_state=None,
                           has_classes=True):
  """Returns the initial transformation to be used depending on the arguments.

  Parameters
  ----------
  n_components : int
    The number of components to take. (Note: it should have been checked
    before, meaning it should not be None and it should be a value in
    [1, X.shape[1]])

  input : array-like
    The input samples (can be tuples or regular samples).

  y : array-like or None
    The input labels (or not if there are no labels).

  init : string or numpy array, optional (default='auto')
    Initialization of the linear transformation. Possible options are
    'auto', 'pca', 'lda', 'identity', 'random', and a numpy array of shape
    (n_features_a, n_features_b).

    'auto'
      Depending on ``n_components``, the most reasonable initialization
      will be chosen. If ``n_components <= n_classes`` we use 'lda' (see
      the description of 'lda' init), as it uses labels information. If
      not, but ``n_components < min(n_features, n_samples)``, we use 'pca',
      as it projects data onto meaningful directions (those of higher
      variance). Otherwise, we just use 'identity'.

    'pca'
      ``n_components`` principal components of the inputs passed
      to :meth:`fit` will be used to initialize the transformation.
      (See `sklearn.decomposition.PCA`)

    'lda'
      ``min(n_components, n_classes)`` most discriminative
      components of the inputs passed to :meth:`fit` will be used to
      initialize the transformation. (If ``n_components > n_classes``,
      the rest of the components will be zero.) (See
      `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`).
      This initialization is possible only if `has_classes == True`.

    'identity'
      The identity matrix. If ``n_components`` is strictly smaller than the
      dimensionality of the inputs passed to :meth:`fit`, the identity
      matrix will be truncated to the first ``n_components`` rows.

    'random'
      The initial transformation will be a random array of shape
      `(n_components, n_features)`. Each value is sampled from the
      standard normal distribution.

    numpy array
      n_features_b must match the dimensionality of the inputs passed to
      :meth:`fit` and n_features_a must be less than or equal to that.
      If ``n_components`` is not None, n_features_a must match it.

  verbose : bool
    Whether to print the details of the initialization or not.

  random_state : int or `numpy.RandomState` or None, optional (default=None)
    A pseudo random number generator object or a seed for it if int. If
    ``init='random'``, ``random_state`` is used to initialize the random
    transformation. If ``init='pca'``, ``random_state`` is passed as an
    argument to PCA when initializing the transformation.

  has_classes : bool (default=True)
    Whether the labels are in fact classes. If true, this will allow to use
    the 'lda' initialization.

  Returns
  -------
  init_components : `numpy.ndarray`
    The initial transformation to use.
  """
  # if we are doing a regression we cannot use lda:
  n_features = input.shape[-1]
  authorized_inits = ['auto', 'pca', 'identity', 'random']
  if has_classes:
    authorized_inits.append('lda')

  if isinstance(init, np.ndarray):
    # we copy the array, so that if we update the metric, we don't want to
    # update the init
    init = check_array(init, copy=True)

    # Assert that init.shape[1] = X.shape[1]
    if init.shape[1] != n_features:
      raise ValueError('The input dimensionality ({}) of the given '
                       'linear transformation `init` must match the '
                       'dimensionality of the given inputs `X` ({}).'
                       .format(init.shape[1], n_features))

    # Assert that init.shape[0] <= init.shape[1]
    if init.shape[0] > init.shape[1]:
      raise ValueError('The output dimensionality ({}) of the given '
                       'linear transformation `init` cannot be '
                       'greater than its input dimensionality ({}).'
                       .format(init.shape[0], init.shape[1]))

    # Assert that self.n_components = init.shape[0]
    if n_components != init.shape[0]:
      raise ValueError('The preferred dimensionality of the '
                       'projected space `n_components` ({}) does'
                       ' not match the output dimensionality of '
                       'the given linear transformation '
                       '`init` ({})!'
                       .format(n_components,
                               init.shape[0]))
  elif init not in authorized_inits:
    raise ValueError(
        "`init` must be '{}' "
        "or a numpy array of shape (n_components, n_features)."
        .format("', '".join(authorized_inits)))

  random_state = check_random_state(random_state)
  if isinstance(init, np.ndarray):
    return init
  n_samples = input.shape[0]
  if init == 'auto':
    if has_classes:
      n_classes = len(np.unique(y))
    else:
      n_classes = -1
    init = _auto_select_init(has_classes, n_features, n_samples, n_components,
                             n_classes)
  if init == 'identity':
    return np.eye(n_components, input.shape[-1])
  elif init == 'random':
    return random_state.randn(n_components, input.shape[-1])
  elif init in {'pca', 'lda'}:
    init_time = time.time()
    if init == 'pca':
      pca = PCA(n_components=n_components,
                random_state=random_state)
      if verbose:
        print('Finding principal components... ')
        sys.stdout.flush()
      pca.fit(input)
      transformation = pca.components_
    elif init == 'lda':
      lda = LinearDiscriminantAnalysis(n_components=n_components)
      if verbose:
        print('Finding most discriminative components... ')
        sys.stdout.flush()
      lda.fit(input, y)
      transformation = lda.scalings_.T[:n_components]
    if verbose:
      print('done in {:5.2f}s'.format(time.time() - init_time))
    return transformation


def _auto_select_init(has_classes, n_features, n_samples, n_components,
                      n_classes):
  if has_classes and n_components <= min(n_features, n_classes - 1):
    init = 'lda'
  elif n_components < min(n_features, n_samples):
    init = 'pca'
  else:
    init = 'identity'
  return init


def _initialize_metric_mahalanobis(input, init='identity', random_state=None,
                                   return_inverse=False, strict_pd=False,
                                   matrix_name='matrix'):
  """Returns a PSD matrix that can be used as a prior or an initialization
  for the Mahalanobis distance

  Parameters
  ----------
  input : array-like
    The input samples (can be tuples or regular samples).

  init : string or numpy array, optional (default='identity')
    Specification for the matrix to initialize. Possible options are
    'identity', 'covariance', 'random', and a numpy array of shape
    (n_features, n_features).

    'identity'
      An identity matrix of shape (n_features, n_features).

    'covariance'
      The (pseudo-)inverse covariance matrix (raises an error if the
      covariance matrix is not definite and `strict_pd == True`)

    'random'
      A random positive definite (PD) matrix of shape
      `(n_features, n_features)`, generated using
      `sklearn.datasets.make_spd_matrix`.

    numpy array
      A PSD matrix (or strictly PD if strict_pd==True) of
      shape (n_features, n_features), that will be used as such to
      initialize the metric, or set the prior.

  random_state : int or `numpy.RandomState` or None, optional (default=None)
    A pseudo random number generator object or a seed for it if int. If
    ``init='random'``, ``random_state`` is used to set the random Mahalanobis
    matrix. If ``init='pca'``, ``random_state`` is passed as an
    argument to PCA when initializing the matrix.

  return_inverse : bool, optional (default=False)
    Whether to return the inverse of the specified matrix. This
    can be sometimes useful. It will return the pseudo-inverse (which is the
    same as the inverse if the matrix is definite (i.e. invertible)). If
    `strict_pd == True` and the matrix is not definite, it will return an
    error.

  strict_pd : bool, optional (default=False)
    Whether to enforce that the provided matrix is definite (in addition to
    being PSD).

  param_name : str, optional (default='matrix')
    The name of the matrix used (example: 'init', 'prior'). Will be used in
    error messages.

  Returns
  -------
  M, or (M, M_inv) : `numpy.ndarray`
    The initial matrix to use M, and its inverse if `return_inverse=True`.
  """
  n_features = input.shape[-1]
  if isinstance(init, np.ndarray):
    # we copy the array, so that if we update the metric, we don't want to
    # update the init
    init = check_array(init, copy=True)

    # Assert that init.shape[1] = n_features
    if init.shape != (n_features,) * 2:
      raise ValueError('The input dimensionality {} of the given '
                       'mahalanobis matrix `{}` must match the '
                       'dimensionality of the given inputs ({}).'
                       .format(init.shape, matrix_name, n_features))

    # Assert that the matrix is symmetric
    if not np.allclose(init, init.T):
      raise ValueError("`{}` is not symmetric.".format(matrix_name))

  elif init not in ['identity', 'covariance', 'random']:
    raise ValueError(
        "`{}` must be 'identity', 'covariance', 'random' "
        "or a numpy array of shape (n_features, n_features)."
        .format(matrix_name))

  random_state = check_random_state(random_state)
  M = init
  if isinstance(M, np.ndarray):
    w, V = eigh(M, check_finite=False)
    init_is_definite = _check_sdp_from_eigen(w)
    if strict_pd and not init_is_definite:
      raise LinAlgError("You should provide a strictly positive definite "
                        "matrix as `{}`. This one is not definite. Try another"
                        " {}, or an algorithm that does not "
                        "require the {} to be strictly positive definite."
                        .format(*((matrix_name,) * 3)))
    elif return_inverse and not init_is_definite:
      warnings.warn('The initialization matrix is not invertible: '
                    'using the pseudo-inverse instead.')
    if return_inverse:
      M_inv = _pseudo_inverse_from_eig(w, V)
      return M, M_inv
    else:
      return M
  elif init == 'identity':
    M = np.eye(n_features, n_features)
    if return_inverse:
      M_inv = M.copy()
      return M, M_inv
    else:
      return M
  elif init == 'covariance':
    if input.ndim == 3:
      # if the input are tuples, we need to form an X by deduplication
      X = np.vstack({tuple(row) for row in input.reshape(-1, n_features)})
    else:
      X = input
    # atleast2d is necessary to deal with scalar covariance matrices
    M_inv = np.atleast_2d(np.cov(X, rowvar=False))
    w, V = eigh(M_inv, check_finite=False)
    cov_is_definite = _check_sdp_from_eigen(w)
    if strict_pd and not cov_is_definite:
      raise LinAlgError("Unable to get a true inverse of the covariance "
                        "matrix since it is not definite. Try another "
                        "`{}`, or an algorithm that does not "
                        "require the `{}` to be strictly positive definite."
                        .format(*((matrix_name,) * 2)))
    elif not cov_is_definite:
      warnings.warn('The covariance matrix is not invertible: '
                    'using the pseudo-inverse instead.'
                    'To make the covariance matrix invertible'
                    ' you can remove any linearly dependent features and/or '
                    'reduce the dimensionality of your input, '
                    'for instance using `sklearn.decomposition.PCA` as a '
                    'preprocessing step.')
    M = _pseudo_inverse_from_eig(w, V)
    if return_inverse:
      return M, M_inv
    else:
      return M
  elif init == 'random':
    # we need to create a random symmetric matrix
    M = make_spd_matrix(n_features, random_state=random_state)
    if return_inverse:
      # we use pinvh even if we know the matrix is definite, just because
      # we need the returned matrix to be symmetric (and sometimes
      # np.linalg.inv returns not symmetric inverses of symmetric matrices)
      # TODO: there might be a more efficient method to do so
      M_inv = pinvh(M)
      return M, M_inv
    else:
      return M


def _check_n_components(n_features, n_components):
  """Checks that n_components is less than n_features and deal with the None
  case"""
  if n_components is None:
    return n_features
  if 0 < n_components <= n_features:
    return n_components
  raise ValueError('Invalid n_components, must be in [1, %d]' % n_features)


def _pseudo_inverse_from_eig(w, V, tol=None):
  """Compute the (Moore-Penrose) pseudo-inverse of the EVD of a symetric
  matrix.

  Parameters
  ----------
  w : (..., M) ndarray
    The eigenvalues in ascending order, each repeated according to
    its multiplicity.

  v : {(..., M, M) ndarray, (..., M, M) matrix}
    The column ``v[:, i]`` is the normalized eigenvector corresponding
    to the eigenvalue ``w[i]``.  Will return a matrix object if `a` is
    a matrix object.

  tol : positive `float`, optional
    Absolute eigenvalues below tol are considered zero.

  Returns
  -------
  output : (..., M, N) array_like
    The pseudo-inverse given by the EVD.
  """
  if tol is None:
    tol = np.amax(w) * np.max(w.shape) * np.finfo(w.dtype).eps
  # discard small eigenvalues and invert the rest
  large = np.abs(w) > tol
  w = np.divide(1, w, where=large, out=w)
  w[~large] = 0

  return np.dot(V * w, np.conjugate(V).T)
