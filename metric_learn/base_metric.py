from numpy.linalg import inv,cholesky


class BaseMetricLearner(object):
  def __init__(self):
    raise NotImplementedError('BaseMetricLearner should not be instantiated')

  def metric(self):
    """Computes the Mahalanobis matrix from the transformation matrix.

    .. math:: M = L^{\\top} L

    Returns
    -------
    M : (d x d) matrix
    """
    L = self.transformer()
    return L.T.dot(L)

  def transformer(self):
    """Computes the transformation matrix from the Mahalanobis matrix.

    L = inv(cholesky(M))

    Returns
    -------
    L : (d x d) matrix
    """
    return inv(cholesky(self.metric()))

  def transform(self, X=None):
    """Applies the metric transformation.

    Parameters
    ----------
    X : (n x d) matrix, optional
        Data to transform. If not supplied, the training data will be used.

    Returns
    -------
    transformed : (n x d) matrix
        Input data transformed to the metric space by :math:`XL^{\\top}`
    """
    if X is None:
      X = self.X
    L = self.transformer()
    return X.dot(L.T)

  def get_params(self, deep=False):
    """Get parameters for this metric learner.

    Parameters
    ----------
    deep: boolean, optional
        @WARNING doesn't do anything, only exists because
        scikit-learn has this on BaseEstimator.

    Returns
    -------
    params : mapping of string to any
        Parameter names mapped to their values.
    """
    return self.params

  def set_params(self, **kwarg):
    """Set the parameters of this metric learner.

    Overwrites any default parameters or parameters specified in constructor.

    Returns
    -------
    self
    """
    self.params.update(kwarg)
    return self
