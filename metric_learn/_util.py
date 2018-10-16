import numpy as np
import six
from sklearn.utils import check_array
from sklearn.utils.validation import check_X_y

# hack around lack of axis kwarg in older numpy versions
try:
  np.linalg.norm([[4]], axis=1)
except TypeError:
  def vector_norm(X):
    return np.apply_along_axis(np.linalg.norm, 1, X)
else:
  def vector_norm(X):
    return np.linalg.norm(X, axis=1)


def check_input(input, y=None, preprocessor=None,
                type_of_inputs='classic', t=None, accept_sparse=False,
                dtype="numeric", order=None,
                copy=False, force_all_finite=True,
                multi_output=False, ensure_min_samples=1,
                ensure_min_features=1, y_numeric=False,
                warn_on_dtype=False, estimator=None):
  """Checks that the input format is valid and does conversions if specified
  (this is the equivalent of scikit-learn's `check_array` or `check_X_y`).
  All arguments following t are scikit-learn's `check_array` or `check_X_y`
  arguments that will be enforced on the output array

  Parameters
  ----------
  input : object
    The input to check
  y : object (optional, default=None)
    The
  preprocessor
  type_of_inputs
  t
  accept_sparse
  dtype
  order
  copy
  force_all_finite
  multi_output
  ensure_min_samples
  ensure_min_features
  y_numeric
  warn_on_dtype
  estimator

  Returns
  -------

  """
  # todo: faire attention a la copie
  # todo: faire attention aux trucs sparses

  context = make_context(estimator)

  args_for_sk_checks = dict(accept_sparse=accept_sparse,
                            dtype=dtype, order=order,
                            copy=copy, force_all_finite=force_all_finite,
                            ensure_min_samples=ensure_min_samples,
                            ensure_min_features=ensure_min_features,
                            warn_on_dtype=warn_on_dtype, estimator=estimator)
  if y is None:
    input = check_array(input, ensure_2d=False, allow_nd=True,
                        copy=False, force_all_finite=False,
                        accept_sparse=True, dtype=None,
                        ensure_min_features=0, ensure_min_samples=0)
  else:
    input, y = check_X_y(input, y, ensure_2d=False, allow_nd=True,
                         copy=False, force_all_finite=False,
                         accept_sparse=True, dtype=None,
                         ensure_min_features=0, ensure_min_samples=0,
                         multi_output=multi_output,
                         y_numeric=y_numeric)
    # we try to allow the more possible stuff here
  preprocessor_has_been_applied = False

  if type_of_inputs == 'classic':
    if input.ndim == 1:
        if preprocessor is not None:
                input = preprocess_points(input, preprocessor)
                preprocessor_has_been_applied = True
        else:
          raise ValueError('2D array of formed points expected{}. Found {}D array '
                  'instead:\ninput={}. Reshape '
                  'your data and/or use a preprocessor.\n'.format(
                      context, input.ndim, input))
    elif input.ndim == 2:
        pass # OK
    else:
      with_prep = (('1D array of indicators or 2D array of formed points',
                    ' when using a preprocessor')
                   if preprocessor is not None else
                   ('2D array of formed points', ''))
      raise ValueError("{} expected{}{}. Found {}D array "
                       "instead:\ninput={}. Reshape your data.\n"
                       .format(with_prep[0], context, with_prep[1],
                               input.ndim, input))

    input = check_array(input, allow_nd=True, ensure_2d=False,
                        # arguments that come from before
                        **args_for_sk_checks)
    if input.ndim != 2:  # we have to ensure this because check_array above
        # does not

      raise ValueError('2D array of formed points expected{}{}. '
              'Found {}D array '
              'instead:\ninput={}. Reshape your data '
              'and/or use a preprocessor.\n'.format(context,
                  ' after the preprocessor has been applied' if
                  preprocessor_has_been_applied else '',
                  input.ndim, input))

  elif type_of_inputs == 'tuples':
    if input.ndim == 2:
      if preprocessor is not None:
          input = preprocess_tuples(input, preprocessor)
          preprocessor_has_been_applied = True
      else:

          raise ValueError('3D array of formed tuples expected{}. '
                  'Found {}D array '
                  'instead:\ninput={}. '
                  'Reshape your data '
                  'and/or use a preprocessor.\n'.format(context,
                      input.ndim,
                      input))
    elif input.ndim == 3:  # we should check_num_features which is not checked
        #  after
        pass
    else:

      with_prep =(('2D array of indicators or 3D array of formed tuples',
                   ' when using a preprocessor')
                  if preprocessor is not None else
                  ('3D array of formed tuples', ''))
      raise ValueError("{} expected{}{}. Found {}D array "
                       "instead:\ninput={}. Reshape your data.\n"
                       .format(with_prep[0], context, with_prep[1],
                               input.ndim, input))

    input = check_array(input, allow_nd=True, ensure_2d=False,
                      # arguments that come from before
                      **args_for_sk_checks)
    if ensure_min_features > 0:
      n_features = input.shape[2]
      if n_features < ensure_min_features:
          raise ValueError("Found array with {} feature(s) (shape={}) while"
                  " a minimum of {} is required{}."
                  .format(n_features, input.shape,
                      ensure_min_features, context))
                  #  normally we don't need to check_t too because t should'nt be able to
    # be modified by any preprocessor
    if input.ndim != 3:  # we have to ensure this because check_array above
      # does not

      raise ValueError('3D array of formed tuples expected{}{}. '
                       'Found {}D array '
                       'instead:\ninput={}. Reshape your data '
                       'and/or use a preprocessor.\n'.format(context,
                       ' after the preprocessor has been applied' if
                        preprocessor_has_been_applied else '',
                        input.ndim, input))

    check_t(input, t, context)

  return input if y is None else (input, y)


def preprocess_tuples(tuples, preprocessor):
  print("Preprocessing tuples...")
  tuples = np.column_stack([preprocessor(tuples[:, i])[:, np.newaxis] for
      i in range(tuples.shape[1])])
  return tuples


def preprocess_points(points, preprocessor):
  """form points if there is a preprocessor else keep them as such (assumes
  that check_points has already been called)"""
  print("Preprocessing points...")
  points = preprocessor(points)
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
