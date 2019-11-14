import numpy as np
import pytest
from numpy.linalg import LinAlgError
from scipy.stats import ortho_group

rng = np.random.RandomState(42)

# an orthonormal matrix useful for creating matrices with given
# eigenvalues:
P = ortho_group.rvs(7, random_state=rng)

# matrix with a determinant still high but which should be considered as a
# non-definite matrix (to check we don't test the definiteness with the
# determinant which is a bad strategy)
M = np.diag([1e5, 1e5, 1e5, 1e5, 1e5, 1e5, 1e-20])
M = P.dot(M).dot(P.T)
assert np.abs(np.linalg.det(M)) > 10
assert np.linalg.slogdet(M)[1] > 1  # (just to show that the computed
# determinant is far from null)
with pytest.raises(LinAlgError) as err_msg:
  np.linalg.cholesky(M)
