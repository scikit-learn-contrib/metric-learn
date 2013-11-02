from numpy.linalg import inv,cholesky


class BaseMetricLearner(object):
  def __init__(self):
    raise NotImplementedError('BaseMetricLearner should not be instantiated')

  def metric(self):
    L = self.transformer()
    return L.T.dot(L)

  def transformer(self):
    return inv(cholesky(self.metric()))

  def transform(self, X=None):
    if X is None:
      X = self.X
    L = self.transformer()
    return L.dot(X.T).T
