import numpy as np


# hack around lack of axis kwarg in older numpy versions
try:
  np.linalg.norm([[4]], axis=1)
except TypeError:
  def vector_norm(X):
    return np.apply_along_axis(np.linalg.norm, 1, X)
else:
  def vector_norm(X):
    return np.linalg.norm(X, axis=1)


def check_tuples(tuples):
  """Check that the input is a valid 3D array representing a dataset of tuples.

  Equivalent of `check_array` in scikit-learn.

  Parameters
  ----------
  tuples : object
    The tuples to check.

  Returns
  -------
  tuples_valid : object
    The validated input.
  """
  # If input is scalar raise error
  if np.isscalar(tuples):
    raise ValueError(
      "Expected 3D array, got scalar instead. Cannot apply this function on "
      "scalars.")
  # If input is 1D raise error
  if len(tuples.shape) == 1:
    raise ValueError(
      "Expected 3D array, got 1D array instead:\ntuples={}.\n"
      "Reshape your data using tuples.reshape(1, -1, 1) if it contains a "
      "single tuple and the points in the tuple have a single "
      "feature.".format(tuples))
  # If input is 2D raise error
  if len(tuples.shape) == 2:
    raise ValueError(
      "Expected 3D array, got 2D array instead:\ntuples={}.\n"
      "Reshape your data either using tuples.reshape(-1, {}, 1) if "
      "your data has a single feature or tuples.reshape(1, {}, -1) "
      "if it contains a single tuple.".format(tuples, tuples.shape[1],
                                              tuples.shape[0]))
  return tuples
