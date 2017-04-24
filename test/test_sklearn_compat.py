import numpy as np
import unittest
from sklearn.utils.estimator_checks import check_estimator

from metric_learn import (
    LMNN, NCA, LFDA, Covariance, MLKR, MetricEvolution,
    LSML_Supervised, ITML_Supervised, SDML_Supervised, RCA_Supervised)


# Wrap the _Supervised methods with a deterministic wrapper for testing.
class deterministic_mixin(object):
  def fit(self, X, y):
    rs = np.random.RandomState(1234)
    return super(deterministic_mixin, self).fit(X, y, random_state=rs)


class dLSML(deterministic_mixin, LSML_Supervised):
  pass


class dITML(deterministic_mixin, ITML_Supervised):
  pass


class dSDML(deterministic_mixin, SDML_Supervised):
  pass


class dRCA(deterministic_mixin, RCA_Supervised):
  pass


class TestSklearnCompat(unittest.TestCase):
  def test_covariance(self):
    check_estimator(Covariance)

  def test_lmnn(self):
    check_estimator(LMNN)

  def test_lfda(self):
    check_estimator(LFDA)

  def test_mlkr(self):
    check_estimator(MLKR)

  def test_nca(self):
    check_estimator(NCA)

  def test_lsml(self):
    check_estimator(dLSML)

  def test_itml(self):
    check_estimator(dITML)

  # This fails in check_dtype_object, not sure why
  # def test_evolution(self):
  #   check_estimator(MetricEvolution)

  # This fails due to a FloatingPointError
  # def test_sdml(self):
  #   check_estimator(dSDML)

  # This fails because the default num_chunks isn't data-dependent.
  # def test_rca(self):
  #   check_estimator(RCA_Supervised)


if __name__ == '__main__':
  unittest.main()
