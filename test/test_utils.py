import numpy as np
import pytest
from metric_learn._util import check_tuples


def test_check_tuples():
  X = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
  check_tuples(X)

  X = np.array(5)
  msg = ("Expected 3D array, got scalar instead. Cannot apply this function "
         "on scalars.")
  with pytest.raises(ValueError, message=msg):
    check_tuples(X)

  X = np.array([1, 2, 3])
  msg = ("Expected 3D array, got 1D array instead:\ntuples=[1, 2, 3].\n"
         "Reshape your data using tuples.reshape(1, -1, 1) if it contains a "
         "single tuple and the points in the tuple have a single feature.")
  with pytest.raises(ValueError, message=msg):
    check_tuples(X)

  X = np.array([[1, 2, 3], [2, 3, 5]])
  msg = ("Expected 3D array, got 2D array instead:\ntuples=[[1, 2, 3], "
         "[2, 3, 5]].\nReshape your data either using "
         "tuples.reshape(-1, 3, 1) if your data has a single feature or "
         "tuples.reshape(1, 2, -1) if it contains a single tuple.")
  with pytest.raises(ValueError, message=msg):
    check_tuples(X)
