import unittest
import re
import pytest
import numpy as np
from scipy.optimize import check_grad, approx_fprime
from six.moves import xrange
from sklearn.metrics import pairwise_distances
from sklearn.datasets import load_iris, make_classification, make_regression
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose)
from sklearn.utils.testing import assert_warns_message
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_X_y
try:
  from inverse_covariance import quic
except ImportError:
  HAS_SKGGM = False
else:
  HAS_SKGGM = True
from metric_learn import (LMNN, NCA, LFDA, Covariance, MLKR, MMC,
                          LSML_Supervised, ITML_Supervised, SDML_Supervised,
                          RCA_Supervised, MMC_Supervised, SDML, ITML)
# Import this specially for testing.
from metric_learn.constraints import wrap_pairs
from metric_learn.lmnn import python_LMNN, _sum_outer_products


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

  def test_singular_returns_pseudo_inverse(self):
    """Checks that if the input covariance matrix is singular, we return
    the pseudo inverse"""
    X, y = load_iris(return_X_y=True)
    # We add a virtual column that is a linear combination of the other
    # columns so that the covariance matrix will be singular
    X = np.concatenate([X, X[:, :2].dot([[2], [3]])], axis=1)
    cov_matrix = np.cov(X, rowvar=False)
    covariance = Covariance()
    covariance.fit(X)
    pseudo_inverse = covariance.get_mahalanobis_matrix()
    # here is the definition of a pseudo inverse according to wikipedia:
    assert_allclose(cov_matrix.dot(pseudo_inverse).dot(cov_matrix),
                    cov_matrix)
    assert_allclose(pseudo_inverse.dot(cov_matrix).dot(pseudo_inverse),
                    pseudo_inverse)


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


@pytest.mark.parametrize('bounds', [None, (20., 100.), [20., 100.],
                                    np.array([20., 100.]),
                                    np.array([[20., 100.]]),
                                    np.array([[20], [100]])])
def test_bounds_parameters_valid(bounds):
  """Asserts that we can provide any array-like of two elements as bounds,
  and that the attribute bound_ is a numpy array"""

  pairs = np.array([[[-10., 0.], [10., 0.]], [[0., 50.], [0., -60]]])
  y_pairs = [1, -1]
  itml = ITML()
  itml.fit(pairs, y_pairs, bounds=bounds)

  X = np.array([[0, 0], [0, 1], [2, 0], [2, 1]])
  y = np.array([1, 0, 1, 0])
  itml_supervised = ITML_Supervised()
  itml_supervised.fit(X, y, bounds=bounds)


@pytest.mark.parametrize('bounds', ['weird', ['weird1', 'weird2'],
                                    np.array([1, 2, 3])])
def test_bounds_parameters_invalid(bounds):
  """Assert that if a non array-like is put for bounds, or an array-like
  of length different than 2, an error is returned"""
  pairs = np.array([[[-10., 0.], [10., 0.]], [[0., 50.], [0., -60]]])
  y_pairs = [1, -1]
  itml = ITML()
  with pytest.raises(Exception):
    itml.fit(pairs, y_pairs, bounds=bounds)

  X = np.array([[0, 0], [0, 1], [2, 0], [2, 1]])
  y = np.array([1, 0, 1, 0])
  itml_supervised = ITML_Supervised()
  with pytest.raises(Exception):
    itml_supervised.fit(X, y, bounds=bounds)


class TestLMNN(MetricTestCase):
  def test_iris(self):
    # Test both impls, if available.
    for LMNN_cls in set((LMNN, python_LMNN)):
      lmnn = LMNN_cls(k=5, learn_rate=1e-6, verbose=False)
      lmnn.fit(self.iris_points, self.iris_labels)

      csep = class_separation(lmnn.transform(self.iris_points),
                              self.iris_labels)
      self.assertLess(csep, 0.25)

  def test_loss_grad_lbfgs(self):
    """Test gradient of loss function
    Assert that the gradient is almost equal to its finite differences
    approximation.
    """
    rng = np.random.RandomState(42)
    X, y = make_classification(random_state=rng)
    L = rng.randn(rng.randint(1, X.shape[1] + 1), X.shape[1])
    lmnn = LMNN()

    k = lmnn.k
    reg = lmnn.regularization

    X, y = lmnn._prepare_inputs(X, y, dtype=float,
                                ensure_min_samples=2)
    num_pts, num_dims = X.shape
    unique_labels, label_inds = np.unique(y, return_inverse=True)
    lmnn.labels_ = np.arange(len(unique_labels))
    lmnn.transformer_ = np.eye(num_dims)

    target_neighbors = lmnn._select_targets(X, label_inds)
    impostors = lmnn._find_impostors(target_neighbors[:, -1], X, label_inds)

    # sum outer products
    dfG = _sum_outer_products(X, target_neighbors.flatten(),
                              np.repeat(np.arange(X.shape[0]), k))
    df = np.zeros_like(dfG)

    # storage
    a1 = [None]*k
    a2 = [None]*k
    for nn_idx in xrange(k):
      a1[nn_idx] = np.array([])
      a2[nn_idx] = np.array([])

    # initialize L
    def loss_grad(flat_L):
      return lmnn._loss_grad(X, flat_L.reshape(-1, X.shape[1]), dfG, impostors,
                             1, k, reg, target_neighbors, df.copy(),
                             list(a1), list(a2))

    def fun(x):
      return loss_grad(x)[1]

    def grad(x):
      return loss_grad(x)[0].ravel()

    # compute relative error
    epsilon = np.sqrt(np.finfo(float).eps)
    rel_diff = (check_grad(fun, grad, L.ravel()) /
                np.linalg.norm(approx_fprime(L.ravel(), fun, epsilon)))
    np.testing.assert_almost_equal(rel_diff, 0., decimal=5)


@pytest.mark.parametrize('X, y, loss', [(np.array([[0], [1], [2], [3]]),
                                         [1, 1, 0, 0], 3.0),
                                        (np.array([[0], [1], [2], [3]]),
                                         [1, 0, 0, 1], 26.)])
def test_toy_ex_lmnn(X, y, loss):
  """Test that the loss give the right result on a toy example"""
  L = np.array([[1]])
  lmnn = LMNN(k=1, regularization=0.5)

  k = lmnn.k
  reg = lmnn.regularization

  X, y = lmnn._prepare_inputs(X, y, dtype=float,
                              ensure_min_samples=2)
  num_pts, num_dims = X.shape
  unique_labels, label_inds = np.unique(y, return_inverse=True)
  lmnn.labels_ = np.arange(len(unique_labels))
  lmnn.transformer_ = np.eye(num_dims)

  target_neighbors = lmnn._select_targets(X, label_inds)
  impostors = lmnn._find_impostors(target_neighbors[:, -1], X, label_inds)

  # sum outer products
  dfG = _sum_outer_products(X, target_neighbors.flatten(),
                            np.repeat(np.arange(X.shape[0]), k))
  df = np.zeros_like(dfG)

  # storage
  a1 = [None]*k
  a2 = [None]*k
  for nn_idx in xrange(k):
    a1[nn_idx] = np.array([])
    a2[nn_idx] = np.array([])

  #  assert that the loss equals the one computed by hand
  assert lmnn._loss_grad(X, L.reshape(-1, X.shape[1]), dfG, impostors, 1, k,
                         reg, target_neighbors, df, a1, a2)[1] == loss


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


class TestSDML(MetricTestCase):

  @pytest.mark.skipif(HAS_SKGGM,
                      reason="The warning can be thrown only if skggm is "
                             "not installed.")
  def test_sdml_supervised_raises_warning_msg_not_installed_skggm(self):
    """Tests that the right warning message is raised if someone tries to
    use SDML_Supervised but has not installed skggm, and that the algorithm
    fails to converge"""
    # TODO: remove if we don't need skggm anymore
    # load_iris: dataset where we know scikit-learn's graphical lasso fails
    # with a Floating Point error
    X, y = load_iris(return_X_y=True)
    sdml_supervised = SDML_Supervised(balance_param=0.5, use_cov=True,
                                      sparsity_param=0.01)
    msg = ("There was a problem in SDML when using scikit-learn's graphical "
           "lasso solver. skggm's graphical lasso can sometimes converge on "
           "non SPD cases where scikit-learn's graphical lasso fails to "
           "converge. Try to install skggm and rerun the algorithm (see "
           "the README.md for the right version of skggm). The following "
           "error message was thrown:")
    with pytest.raises(RuntimeError) as raised_error:
      sdml_supervised.fit(X, y)
    assert str(raised_error.value).startswith(msg)

  @pytest.mark.skipif(HAS_SKGGM,
                      reason="The warning can be thrown only if skggm is "
                             "not installed.")
  def test_sdml_raises_warning_msg_not_installed_skggm(self):
    """Tests that the right warning message is raised if someone tries to
    use SDML but has not installed skggm, and that the algorithm fails to
    converge"""
    # TODO: remove if we don't need skggm anymore
    # case on which we know that scikit-learn's graphical lasso fails
    # because it will return a non SPD matrix
    pairs = np.array([[[-10., 0.], [10., 0.]], [[0., 50.], [0., -60]]])
    y_pairs = [1, -1]
    sdml = SDML(use_cov=False, balance_param=100, verbose=True)

    msg = ("There was a problem in SDML when using scikit-learn's graphical "
           "lasso solver. skggm's graphical lasso can sometimes converge on "
           "non SPD cases where scikit-learn's graphical lasso fails to "
           "converge. Try to install skggm and rerun the algorithm (see "
           "the README.md for the right version of skggm).")
    with pytest.raises(RuntimeError) as raised_error:
      sdml.fit(pairs, y_pairs)
    assert msg == str(raised_error.value)

  @pytest.mark.skipif(not HAS_SKGGM,
                      reason="The warning can be thrown only if skggm is "
                             "installed.")
  def test_sdml_raises_warning_msg_installed_skggm(self):
    """Tests that the right warning message is raised if someone tries to
    use SDML but has not installed skggm, and that the algorithm fails to
    converge"""
    # TODO: remove if we don't need skggm anymore
    # case on which we know that skggm's graphical lasso fails
    # because it will return non finite values
    pairs = np.array([[[-10., 0.], [10., 0.]], [[0., 50.], [0., -60]]])
    y_pairs = [1, -1]
    sdml = SDML(use_cov=False, balance_param=100, verbose=True)

    msg = ("There was a problem in SDML when using skggm's graphical "
           "lasso solver.")
    with pytest.raises(RuntimeError) as raised_error:
      sdml.fit(pairs, y_pairs)
    assert msg == str(raised_error.value)

  @pytest.mark.skipif(not HAS_SKGGM,
                      reason="The warning can be thrown only if skggm is "
                             "installed.")
  def test_sdml_supervised_raises_warning_msg_installed_skggm(self):
    """Tests that the right warning message is raised if someone tries to
    use SDML_Supervised but has not installed skggm, and that the algorithm
    fails to converge"""
    # TODO: remove if we don't need skggm anymore
    # case on which we know that skggm's graphical lasso fails
    # because it will return non finite values
    rng = np.random.RandomState(42)
    # This example will create a diagonal em_cov with a negative coeff (
    # pathological case)
    X = np.array([[-10., 0.], [10., 0.], [5., 0.], [3., 0.]])
    y = [0, 0, 1, 1]
    sdml_supervised = SDML_Supervised(balance_param=0.5, use_cov=False,
                                      sparsity_param=0.01)
    msg = ("There was a problem in SDML when using skggm's graphical "
           "lasso solver.")
    with pytest.raises(RuntimeError) as raised_error:
      sdml_supervised.fit(X, y, random_state=rng)
    assert msg == str(raised_error.value)

  @pytest.mark.skipif(not HAS_SKGGM,
                      reason="It's only in the case where skggm is installed"
                             "that no warning should be thrown.")
  def test_raises_no_warning_installed_skggm(self):
    # otherwise we should be able to instantiate and fit SDML and it
    # should raise no warning
    pairs = np.array([[[-10., 0.], [10., 0.]], [[0., -55.], [0., -60]]])
    y_pairs = [1, -1]
    X, y = make_classification(random_state=42)
    with pytest.warns(None) as record:
      sdml = SDML()
      sdml.fit(pairs, y_pairs)
    assert len(record) == 0
    with pytest.warns(None) as record:
      sdml = SDML_Supervised(use_cov=False, balance_param=1e-5)
      sdml.fit(X, y)
    assert len(record) == 0

  def test_iris(self):
    # Note: this is a flaky test, which fails for certain seeds.
    # TODO: un-flake it!
    rs = np.random.RandomState(5555)

    sdml = SDML_Supervised(num_constraints=1500, use_cov=False,
                           balance_param=5e-5)
    sdml.fit(self.iris_points, self.iris_labels, random_state=rs)
    csep = class_separation(sdml.transform(self.iris_points),
                            self.iris_labels)
    self.assertLess(csep, 0.22)

  def test_deprecation_num_labeled(self):
    # test that a deprecation message is thrown if num_labeled is set at
    # initialization
    # TODO: remove in v.0.6
    X, y = make_classification(random_state=42)
    sdml_supervised = SDML_Supervised(num_labeled=np.inf, use_cov=False,
                                      balance_param=5e-5)
    msg = ('"num_labeled" parameter is not used.'
           ' It has been deprecated in version 0.5.0 and will be'
           'removed in 0.6.0')
    assert_warns_message(DeprecationWarning, msg, sdml_supervised.fit, X, y)

  def test_sdml_raises_warning_non_psd(self):
    """Tests that SDML raises a warning on a toy example where we know the
    pseudo-covariance matrix is not PSD"""
    pairs = np.array([[[-10., 0.], [10., 0.]], [[0., 50.], [0., -60]]])
    y = [1, -1]
    sdml = SDML(use_cov=True, sparsity_param=0.01, balance_param=0.5)
    msg = ("Warning, the input matrix of graphical lasso is not "
           "positive semi-definite (PSD). The algorithm may diverge, "
           "and lead to degenerate solutions. "
           "To prevent that, try to decrease the balance parameter "
           "`balance_param` and/or to set use_cov=False.")
    with pytest.warns(ConvergenceWarning) as raised_warning:
      try:
        sdml.fit(pairs, y)
      except Exception:
        pass
    # we assert that this warning is in one of the warning raised by the
    # estimator
    assert msg in list(map(lambda w: str(w.message), raised_warning))

  def test_sdml_converges_if_psd(self):
    """Tests that sdml converges on a simple problem where we know the
    pseudo-covariance matrix is PSD"""
    pairs = np.array([[[-10., 0.], [10., 0.]], [[0., -55.], [0., -60]]])
    y = [1, -1]
    sdml = SDML(use_cov=True, sparsity_param=0.01, balance_param=0.5)
    sdml.fit(pairs, y)
    assert np.isfinite(sdml.get_mahalanobis_matrix()).all()

  @pytest.mark.skipif(not HAS_SKGGM,
                      reason="sklearn's graphical_lasso can sometimes not "
                             "work on some non SPD problems. We test that "
                             "is works only if skggm is installed.")
  def test_sdml_works_on_non_spd_pb_with_skggm(self):
    """Test that SDML works on a certain non SPD problem on which we know
    it should work, but scikit-learn's graphical_lasso does not work"""
    X, y = load_iris(return_X_y=True)
    sdml = SDML_Supervised(balance_param=0.5, sparsity_param=0.01,
                           use_cov=True)
    sdml.fit(X, y)


@pytest.mark.skipif(not HAS_SKGGM,
                    reason='The message should be printed only if skggm is '
                           'installed.')
def test_verbose_has_installed_skggm_sdml(capsys):
  # Test that if users have installed skggm, a message is printed telling them
  # skggm's solver is used (when they use SDML)
  # TODO: remove if we don't need skggm anymore
  pairs = np.array([[[-10., 0.], [10., 0.]], [[0., -55.], [0., -60]]])
  y_pairs = [1, -1]
  sdml = SDML(verbose=True)
  sdml.fit(pairs, y_pairs)
  out, _ = capsys.readouterr()
  assert "SDML will use skggm's graphical lasso solver." in out


@pytest.mark.skipif(not HAS_SKGGM,
                    reason='The message should be printed only if skggm is '
                           'installed.')
def test_verbose_has_installed_skggm_sdml_supervised(capsys):
  # Test that if users have installed skggm, a message is printed telling them
  # skggm's solver is used (when they use SDML_Supervised)
  # TODO: remove if we don't need skggm anymore
  X, y = make_classification(random_state=42)
  sdml = SDML_Supervised(verbose=True)
  sdml.fit(X, y)
  out, _ = capsys.readouterr()
  assert "SDML will use skggm's graphical lasso solver." in out


@pytest.mark.skipif(HAS_SKGGM,
                    reason='The message should be printed only if skggm is '
                           'not installed.')
def test_verbose_has_not_installed_skggm_sdml(capsys):
  # Test that if users have installed skggm, a message is printed telling them
  # skggm's solver is used (when they use SDML)
  # TODO: remove if we don't need skggm anymore
  pairs = np.array([[[-10., 0.], [10., 0.]], [[0., -55.], [0., -60]]])
  y_pairs = [1, -1]
  sdml = SDML(verbose=True)
  sdml.fit(pairs, y_pairs)
  out, _ = capsys.readouterr()
  assert "SDML will use scikit-learn's graphical lasso solver." in out


@pytest.mark.skipif(HAS_SKGGM,
                    reason='The message should be printed only if skggm is '
                           'not installed.')
def test_verbose_has_not_installed_skggm_sdml_supervised(capsys):
  # Test that if users have installed skggm, a message is printed telling them
  # skggm's solver is used (when they use SDML_Supervised)
  # TODO: remove if we don't need skggm anymore
  X, y = make_classification(random_state=42)
  sdml = SDML_Supervised(verbose=True, balance_param=1e-5, use_cov=False)
  sdml.fit(X, y)
  out, _ = capsys.readouterr()
  assert "SDML will use scikit-learn's graphical lasso solver." in out


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
    epsilon = np.sqrt(np.finfo(float).eps)
    rel_diff = (check_grad(fun, grad, M.ravel()) /
                np.linalg.norm(approx_fprime(M.ravel(), fun, epsilon)))
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
