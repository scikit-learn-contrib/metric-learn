import unittest
import re
import pytest
import numpy as np
from scipy.optimize import check_grad
from six.moves import xrange
from sklearn.metrics import pairwise_distances
from sklearn.datasets import load_iris, make_classification, make_regression
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.testing import assert_warns_message
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_X_y

from metric_learn import (
    LMNN, NCA, LFDA, Covariance, MLKR, MMC,
    LSML_Supervised, ITML_Supervised, SDML_Supervised, RCA_Supervised,
    MMC_Supervised)
# Import this specially for testing.
from metric_learn._util import has_installed_skggm
from metric_learn.constraints import wrap_pairs
from metric_learn.lmnn import python_LMNN


def class_separation(X, labels):
  unique_labels, label_inds = np.unique(labels, return_inverse=True)
  ratio = 0
  for li in xrange(len(unique_labels)):
    Xc = X[label_inds==li]
    Xnc = X[label_inds!=li]
    ratio += pairwise_distances(Xc).mean() / pairwise_distances(Xc,Xnc).mean()
  return ratio / len(unique_labels)


class MetricTestCase(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    # runs once per test class
    iris_data = load_iris()
    self.iris_points = iris_data['data']
    self.iris_labels = iris_data['target']
    np.random.seed(1234)


class TestCovariance(MetricTestCase):
  def test_iris(self):
    cov = Covariance()
    cov.fit(self.iris_points)

    csep = class_separation(cov.transform(self.iris_points), self.iris_labels)
    # deterministic result
    self.assertAlmostEqual(csep, 0.72981476)


class TestLSML(MetricTestCase):
  def test_iris(self):
    lsml = LSML_Supervised(num_constraints=200)
    lsml.fit(self.iris_points, self.iris_labels)

    csep = class_separation(lsml.transform(self.iris_points), self.iris_labels)
    self.assertLess(csep, 0.8)  # it's pretty terrible

  def test_deprecation_num_labeled(self):
    # test that a deprecation message is thrown if num_labeled is set at
    # initialization
    # TODO: remove in v.0.6
    X = np.array([[0, 0], [0, 1], [2, 0], [2, 1]])
    y = np.array([1, 0, 1, 0])
    lsml_supervised = LSML_Supervised(num_labeled=np.inf)
    msg = ('"num_labeled" parameter is not used.'
           ' It has been deprecated in version 0.5.0 and will be'
           'removed in 0.6.0')
    assert_warns_message(DeprecationWarning, msg, lsml_supervised.fit, X, y)


class TestITML(MetricTestCase):
  def test_iris(self):
    itml = ITML_Supervised(num_constraints=200)
    itml.fit(self.iris_points, self.iris_labels)

    csep = class_separation(itml.transform(self.iris_points), self.iris_labels)
    self.assertLess(csep, 0.2)

  def test_deprecation_num_labeled(self):
    # test that a deprecation message is thrown if num_labeled is set at
    # initialization
    # TODO: remove in v.0.6
    X = np.array([[0, 0], [0, 1], [2, 0], [2, 1]])
    y = np.array([1, 0, 1, 0])
    itml_supervised = ITML_Supervised(num_labeled=np.inf)
    msg = ('"num_labeled" parameter is not used.'
           ' It has been deprecated in version 0.5.0 and will be'
           'removed in 0.6.0')
    assert_warns_message(DeprecationWarning, msg, itml_supervised.fit, X, y)

  def test_deprecation_bounds(self):
    # test that a deprecation message is thrown if bounds is set at
    # initialization
    # TODO: remove in v.0.6
    X = np.array([[0, 0], [0, 1], [2, 0], [2, 1]])
    y = np.array([1, 0, 1, 0])
    itml_supervised = ITML_Supervised(bounds=None)
    msg = ('"bounds" parameter from initialization is not used.'
           ' It has been deprecated in version 0.5.0 and will be'
           'removed in 0.6.0. Use the "bounds" parameter of this '
           'fit method instead.')
    assert_warns_message(DeprecationWarning, msg, itml_supervised.fit, X, y)


class TestLMNN(MetricTestCase):
  def test_iris(self):
    # Test both impls, if available.
    for LMNN_cls in set((LMNN, python_LMNN)):
      lmnn = LMNN_cls(k=5, learn_rate=1e-6, verbose=False)
      lmnn.fit(self.iris_points, self.iris_labels)

      csep = class_separation(lmnn.transform(self.iris_points),
                              self.iris_labels)
      self.assertLess(csep, 0.25)


def test_convergence_simple_example(capsys):
  # LMNN should converge on this simple example, which it did not with
  # this issue: https://github.com/metric-learn/metric-learn/issues/88
  X, y = make_classification(random_state=0)
  lmnn = python_LMNN(verbose=True)
  lmnn.fit(X, y)
  out, _ = capsys.readouterr()
  assert "LMNN converged with objective" in out


def test_no_twice_same_objective(capsys):
  # test that the objective function never has twice the same value
  # see https://github.com/metric-learn/metric-learn/issues/88
  X, y = make_classification(random_state=0)
  lmnn = python_LMNN(verbose=True)
  lmnn.fit(X, y)
  out, _ = capsys.readouterr()
  lines = re.split("\n+", out)
  # we get only objectives from each line:
  # the regexp matches a float that follows an integer (the iteration
  # number), and which is followed by a (signed) float (delta obj). It
  # matches for instance:
  # 3 **1113.7665747189938** -3.182774197440267 46431.0200999999999998e-06
  objectives = [re.search("\d* (?:(\d*.\d*))[ | -]\d*.\d*", s)
                for s in lines]
  objectives = [match.group(1) for match in objectives if match is not None]
  # we remove the last element because it can be equal to the penultimate
  # if the last gradient update is null
  assert len(objectives[:-1]) == len(set(objectives[:-1]))


if has_installed_skggm():
  class TestSDML(MetricTestCase):
    def test_iris(self):
      # Note: this is a flaky test, which fails for certain seeds.
      # TODO: un-flake it!
      rs = np.random.RandomState(5555)

      sdml = SDML_Supervised(num_constraints=1500)
      sdml.fit(self.iris_points, self.iris_labels, random_state=rs)
      csep = class_separation(sdml.transform(self.iris_points),
                              self.iris_labels)
      self.assertLess(csep, 0.20)

    def test_deprecation_num_labeled(self):
      # test that a deprecation message is thrown if num_labeled is set at
      # initialization
      # TODO: remove in v.0.6
      X = np.array([[0, 0], [0, 1], [2, 0], [2, 1]])
      y = np.array([1, 0, 1, 0])
      sdml_supervised = SDML_Supervised(num_labeled=np.inf)
      msg = ('"num_labeled" parameter is not used.'
             ' It has been deprecated in version 0.5.0 and will be'
             'removed in 0.6.0')
      assert_warns_message(DeprecationWarning, msg, sdml_supervised.fit, X, y)


class TestNCA(MetricTestCase):
  def test_iris(self):
    n = self.iris_points.shape[0]

    # Without dimension reduction
    nca = NCA(max_iter=(100000//n))
    nca.fit(self.iris_points, self.iris_labels)
    csep = class_separation(nca.transform(self.iris_points), self.iris_labels)
    self.assertLess(csep, 0.15)

    # With dimension reduction
    nca = NCA(max_iter=(100000//n), num_dims=2)
    nca.fit(self.iris_points, self.iris_labels)
    csep = class_separation(nca.transform(self.iris_points), self.iris_labels)
    self.assertLess(csep, 0.20)

  def test_finite_differences(self):
    """Test gradient of loss function

    Assert that the gradient is almost equal to its finite differences
    approximation.
    """
    # Initialize the transformation `M`, as well as `X` and `y` and `NCA`
    X, y = make_classification()
    M = np.random.randn(np.random.randint(1, X.shape[1] + 1), X.shape[1])
    mask = y[:, np.newaxis] == y[np.newaxis, :]
    nca = NCA()
    nca.n_iter_ = 0

    def fun(M):
      return nca._loss_grad_lbfgs(M, X, mask)[0]

    def grad(M):
      return nca._loss_grad_lbfgs(M, X, mask)[1].ravel()

    # compute relative error
    rel_diff = check_grad(fun, grad, M.ravel()) / np.linalg.norm(grad(M))
    np.testing.assert_almost_equal(rel_diff, 0., decimal=6)

  def test_simple_example(self):
    """Test on a simple example.

    Puts four points in the input space where the opposite labels points are
    next to each other. After transform the same labels points should be next
    to each other.

    """
    X = np.array([[0, 0], [0, 1], [2, 0], [2, 1]])
    y = np.array([1, 0, 1, 0])
    nca = NCA(num_dims=2,)
    nca.fit(X, y)
    Xansformed = nca.transform(X)
    np.testing.assert_equal(pairwise_distances(Xansformed).argsort()[:, 1],
                            np.array([2, 3, 0, 1]))

  def test_singleton_class(self):
      X = self.iris_points
      y = self.iris_labels

      # one singleton class: test fitting works
      singleton_class = 1
      ind_singleton, = np.where(y == singleton_class)
      y[ind_singleton] = 2
      y[ind_singleton[0]] = singleton_class

      nca = NCA(max_iter=30)
      nca.fit(X, y)

      # One non-singleton class: test fitting works
      ind_1, = np.where(y == 1)
      ind_2, = np.where(y == 2)
      y[ind_1] = 0
      y[ind_1[0]] = 1
      y[ind_2] = 0
      y[ind_2[0]] = 2

      nca = NCA(max_iter=30)
      nca.fit(X, y)

      # Only singleton classes: test fitting does nothing (the gradient
      # must be null in this case, so the final matrix must stay like
      # the initialization)
      ind_0, = np.where(y == 0)
      ind_1, = np.where(y == 1)
      ind_2, = np.where(y == 2)
      X = X[[ind_0[0], ind_1[0], ind_2[0]]]
      y = y[[ind_0[0], ind_1[0], ind_2[0]]]

      EPS = np.finfo(float).eps
      A = np.zeros((X.shape[1], X.shape[1]))
      np.fill_diagonal(A,
                       1. / (np.maximum(X.max(axis=0) - X.min(axis=0), EPS)))
      nca = NCA(max_iter=30, num_dims=X.shape[1])
      nca.fit(X, y)
      assert_array_equal(nca.transformer_, A)

  def test_one_class(self):
      # if there is only one class the gradient is null, so the final matrix
      #  must stay like the initialization
      X = self.iris_points[self.iris_labels == 0]
      y = self.iris_labels[self.iris_labels == 0]
      EPS = np.finfo(float).eps
      A = np.zeros((X.shape[1], X.shape[1]))
      np.fill_diagonal(A,
                       1. / (np.maximum(X.max(axis=0) - X.min(axis=0), EPS)))
      nca = NCA(max_iter=30, num_dims=X.shape[1])
      nca.fit(X, y)
      assert_array_equal(nca.transformer_, A)


class TestLFDA(MetricTestCase):
  def test_iris(self):
    lfda = LFDA(k=2, num_dims=2)
    lfda.fit(self.iris_points, self.iris_labels)
    csep = class_separation(lfda.transform(self.iris_points), self.iris_labels)
    self.assertLess(csep, 0.15)

    # Sanity checks for learned matrices.
    self.assertEqual(lfda.get_mahalanobis_matrix().shape, (4, 4))
    self.assertEqual(lfda.transformer_.shape, (2, 4))


class TestRCA(MetricTestCase):
  def test_iris(self):
    rca = RCA_Supervised(num_dims=2, num_chunks=30, chunk_size=2)
    rca.fit(self.iris_points, self.iris_labels)
    csep = class_separation(rca.transform(self.iris_points), self.iris_labels)
    self.assertLess(csep, 0.25)

  def test_feature_null_variance(self):
    X = np.hstack((self.iris_points, np.eye(len(self.iris_points), M=1)))

    # Apply PCA with the number of components
    rca = RCA_Supervised(num_dims=2, pca_comps=3, num_chunks=30, chunk_size=2)
    rca.fit(X, self.iris_labels)
    csep = class_separation(rca.transform(X), self.iris_labels)
    self.assertLess(csep, 0.30)

    # Apply PCA with the minimum variance ratio
    rca = RCA_Supervised(num_dims=2, pca_comps=0.95, num_chunks=30,
                         chunk_size=2)
    rca.fit(X, self.iris_labels)
    csep = class_separation(rca.transform(X), self.iris_labels)
    self.assertLess(csep, 0.30)


class TestMLKR(MetricTestCase):
  def test_iris(self):
    mlkr = MLKR()
    mlkr.fit(self.iris_points, self.iris_labels)
    csep = class_separation(mlkr.transform(self.iris_points), self.iris_labels)
    self.assertLess(csep, 0.25)

  def test_finite_differences(self):
    """Test gradient of loss function

    Assert that the gradient is almost equal to its finite differences
    approximation.
    """
    # Initialize the transformation `M`, as well as `X`, and `y` and `MLKR`
    X, y = make_regression(n_features=4, random_state=1, n_samples=20)
    X, y = check_X_y(X, y)
    M = np.random.randn(2, X.shape[1])
    mlkr = MLKR()
    mlkr.n_iter_ = 0

    def fun(M):
      return mlkr._loss(M, X, y)[0]

    def grad_fn(M):
      return mlkr._loss(M, X, y)[1].ravel()

    # compute relative error
    rel_diff = check_grad(fun, grad_fn, M.ravel()) / np.linalg.norm(grad_fn(M))
    np.testing.assert_almost_equal(rel_diff, 0.)


class TestMMC(MetricTestCase):
  def test_iris(self):

    # Generate full set of constraints for comparison with reference implementation
    n = self.iris_points.shape[0]
    mask = (self.iris_labels[None] == self.iris_labels[:,None])
    a, b = np.nonzero(np.triu(mask, k=1))
    c, d = np.nonzero(np.triu(~mask, k=1))

    # Full metric
    mmc = MMC(convergence_threshold=0.01)
    mmc.fit(*wrap_pairs(self.iris_points, [a,b,c,d]))
    expected = [[+0.000514, +0.000868, -0.001195, -0.001703],
                [+0.000868, +0.001468, -0.002021, -0.002879],
                [-0.001195, -0.002021, +0.002782, +0.003964],
                [-0.001703, -0.002879, +0.003964, +0.005648]]
    assert_array_almost_equal(expected, mmc.get_mahalanobis_matrix(),
                              decimal=6)

    # Diagonal metric
    mmc = MMC(diagonal=True)
    mmc.fit(*wrap_pairs(self.iris_points, [a,b,c,d]))
    expected = [0, 0, 1.210220, 1.228596]
    assert_array_almost_equal(np.diag(expected), mmc.get_mahalanobis_matrix(),
                              decimal=6)

    # Supervised Full
    mmc = MMC_Supervised()
    mmc.fit(self.iris_points, self.iris_labels)
    csep = class_separation(mmc.transform(self.iris_points), self.iris_labels)
    self.assertLess(csep, 0.15)
    
    # Supervised Diagonal
    mmc = MMC_Supervised(diagonal=True)
    mmc.fit(self.iris_points, self.iris_labels)
    csep = class_separation(mmc.transform(self.iris_points), self.iris_labels)
    self.assertLess(csep, 0.2)

  def test_deprecation_num_labeled(self):
    # test that a deprecation message is thrown if num_labeled is set at
    # initialization
    # TODO: remove in v.0.6
    X = np.array([[0, 0], [0, 1], [2, 0], [2, 1]])
    y = np.array([1, 0, 1, 0])
    mmc_supervised = MMC_Supervised(num_labeled=np.inf)
    msg = ('"num_labeled" parameter is not used.'
           ' It has been deprecated in version 0.5.0 and will be'
           'removed in 0.6.0')
    assert_warns_message(DeprecationWarning, msg, mmc_supervised.fit, X, y)


@pytest.mark.parametrize(('algo_class', 'dataset'),
                         [(NCA, make_classification()),
                          (MLKR, make_regression())])
def test_verbose(algo_class, dataset, capsys):
  # assert there is proper output when verbose = True
  X, y = dataset
  model = algo_class(verbose=True)
  model.fit(X, y)
  out, _ = capsys.readouterr()

  # check output
  lines = re.split('\n+', out)
  header = '{:>10} {:>20} {:>10}'.format('Iteration', 'Objective Value',
                                         'Time(s)')
  assert lines[0] == '[{}]'.format(algo_class.__name__)
  assert lines[1] == '[{}] {}'.format(algo_class.__name__, header)
  assert lines[2] == '[{}] {}'.format(algo_class.__name__, '-' * len(header))
  for line in lines[3:-2]:
    # The following regex will match for instance:
    # '[NCA]          0         6.988936e+01       0.01'
    assert re.match("\[" + algo_class.__name__ + "\]\ *\d+\ *\d\.\d{6}e[+|-]"
                    "\d+\ *\d+\.\d{2}", line)
  assert re.match("\[" + algo_class.__name__ + "\] Training took\ *"
                  "\d+\.\d{2}s\.", lines[-2])
  assert lines[-1] == ''


@pytest.mark.parametrize(('algo_class', 'dataset'),
                         [(NCA, make_classification()),
                          (MLKR, make_regression(n_features=10))])
def test_no_verbose(dataset, algo_class, capsys):
  # assert by default there is no output (verbose=False)
  X, y = dataset
  model = algo_class()
  model.fit(X, y)
  out, _ = capsys.readouterr()
  # check output
  assert (out == '')


@pytest.mark.parametrize(('algo_class', 'dataset'),
                         [(NCA, make_classification()),
                          (MLKR, make_regression(n_features=10))])
def test_convergence_warning(dataset, algo_class):
    X, y = dataset
    model = algo_class(max_iter=2, verbose=True)
    cls_name = model.__class__.__name__
    assert_warns_message(ConvergenceWarning,
                         '[{}] {} did not converge'.format(cls_name, cls_name),
                         model.fit, X, y)


if __name__ == '__main__':
  unittest.main()
