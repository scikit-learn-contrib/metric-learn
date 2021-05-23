import unittest
import re
import pytest
import numpy as np
from scipy.optimize import check_grad, approx_fprime
from sklearn.metrics import pairwise_distances
from sklearn.datasets import (load_iris, make_classification, make_regression,
                              make_spd_matrix)
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose)
from metric_learn.sklearn_shims import assert_warns_message
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import StandardScaler
try:
  from inverse_covariance import quic
  assert(quic)
except ImportError:
  HAS_SKGGM = False
else:
  HAS_SKGGM = True
from metric_learn import (LMNN, NCA, LFDA, Covariance, MLKR, MMC,
                          SCML_Supervised, LSML_Supervised,
                          ITML_Supervised, SDML_Supervised, RCA_Supervised,
                          MMC_Supervised, SDML, RCA, ITML, SCML)
# Import this specially for testing.
from metric_learn.constraints import wrap_pairs, Constraints
from metric_learn.lmnn import (
    _sum_weighted_outer_products,
    _make_knn_graph,
    _push_loss_grad
)


def class_separation(X, labels):
  unique_labels, label_inds = np.unique(labels, return_inverse=True)
  ratio = 0
  for li in range(len(unique_labels)):
    Xc = X[label_inds == li]
    Xnc = X[label_inds != li]
    ratio += pairwise_distances(Xc).mean() / pairwise_distances(Xc, Xnc).mean()
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


class TestSCML(object):
  @pytest.mark.parametrize('basis', ('lda', 'triplet_diffs'))
  def test_iris(self, basis):
    X, y = load_iris(return_X_y=True)
    scml = SCML_Supervised(basis=basis, n_basis=85, k_genuine=7, k_impostor=5,
                           random_state=42)
    scml.fit(X, y)
    csep = class_separation(scml.transform(X), y)
    assert csep < 0.24

  def test_big_n_features(self):
    X, y = make_classification(n_samples=100, n_classes=3, n_features=60,
                               n_informative=60, n_redundant=0, n_repeated=0,
                               random_state=42)
    X = StandardScaler().fit_transform(X)
    scml = SCML_Supervised(random_state=42)
    scml.fit(X, y)
    csep = class_separation(scml.transform(X), y)
    assert csep < 0.7

  @pytest.mark.parametrize(('estimator', 'data'),
                           [(SCML, (np.ones((3, 3, 3)),)),
                            (SCML_Supervised, (np.array([[0, 0], [0, 1],
                                                         [2, 0], [2, 1]]),
                                               np.array([1, 0, 1, 0])))])
  def test_bad_basis(self, estimator, data):
    model = estimator(basis='bad_basis')
    msg = ("`basis` must be one of the options '{}' or an array of shape "
           "(n_basis, n_features)."
           .format("', '".join(model._authorized_basis)))
    with pytest.raises(ValueError) as raised_error:
      model.fit(*data)
    assert msg == raised_error.value.args[0]

  def test_dimension_reduction_msg(self):
    scml = SCML(n_basis=2)
    triplets = np.array([[[0, 1], [2, 1], [0, 0]],
                         [[2, 1], [0, 1], [2, 0]],
                         [[0, 0], [2, 0], [0, 1]],
                         [[2, 0], [0, 0], [2, 1]]])
    msg = ("The number of bases with nonzero weight is less than the "
           "number of features of the input, in consequence the "
           "learned transformation reduces the dimension to 1.")
    with pytest.warns(UserWarning) as raised_warning:
      scml.fit(triplets)
    assert msg == raised_warning[0].message.args[0]

  @pytest.mark.parametrize(('estimator', 'data'),
                           [(SCML, (np.array([[[0, 1], [2, 1], [0, 0]],
                                              [[2, 1], [0, 1], [2, 0]],
                                              [[0, 0], [2, 0], [0, 1]],
                                              [[2, 0], [0, 0], [2, 1]]]),)),
                           (SCML_Supervised, (np.array([[0, 0], [1, 1],
                                                       [3, 3]]),
                                              np.array([1, 2, 3])))])
  def test_n_basis_wrong_type(self, estimator, data):
    n_basis = 4.0
    model = estimator(n_basis=n_basis)
    msg = ("n_basis should be an integer, instead it is of type %s"
           % type(n_basis))
    with pytest.raises(ValueError) as raised_error:
      model.fit(*data)
    assert msg == raised_error.value.args[0]

  def test_small_n_basis_lda(self):
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1, 1])

    n_class = 2
    scml = SCML_Supervised(n_basis=n_class-1)
    msg = ("The number of basis is less than the number of classes, which may"
           " lead to poor discriminative performance.")
    with pytest.warns(UserWarning) as raised_warning:
      scml.fit(X, y)
    assert msg == raised_warning[0].message.args[0]

  def test_big_n_basis_lda(self):
    X = np.array([[0, 0], [1, 1], [3, 3]])
    y = np.array([1, 2, 3])

    n_class = 3
    num_eig = min(n_class - 1, X.shape[1])
    n_basis = X.shape[0] * 2 * num_eig

    scml = SCML_Supervised(n_basis=n_basis)
    msg = ("Not enough samples to generate %d LDA bases, n_basis"
           "should be smaller than %d" %
           (n_basis, n_basis))
    with pytest.raises(ValueError) as raised_error:
      scml.fit(X, y)
    assert msg == raised_error.value.args[0]

  @pytest.mark.parametrize(('estimator', 'data'),
                           [(SCML, (np.random.rand(3, 3, 2),)),
                           (SCML_Supervised, (np.array([[0, 0], [0, 1],
                                                        [2, 0], [2, 1]]),
                                              np.array([1, 0, 1, 0])))])
  def test_array_basis(self, estimator, data):
    """ Test that the proper error is raised when the shape of the input basis
    array is not consistent with the input
    """
    basis = np.eye(3)
    scml = estimator(n_basis=3, basis=basis)

    msg = ('The dimensionality ({}) of the provided bases must match the '
           'dimensionality of the data ({}).'
           .format(basis.shape[1], data[0].shape[-1]))
    with pytest.raises(ValueError) as raised_error:
      scml.fit(*data)
    assert msg == raised_error.value.args[0]

  @pytest.mark.parametrize(('estimator', 'data'),
                           [(SCML, (np.array([[0, 1, 2], [0, 1, 3], [1, 0, 2],
                                              [1, 0, 3], [2, 3, 1], [2, 3, 0],
                                              [3, 2, 1], [3, 2, 0]]),)),
                           (SCML_Supervised, (np.array([0, 1, 2, 3]),
                                              np.array([0, 0, 1, 1])))])
  def test_verbose(self, estimator, data, capsys):
    # assert there is proper output when verbose = True
    model = estimator(preprocessor=np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
                      max_iter=1, output_iter=1, batch_size=1,
                      basis='triplet_diffs', random_state=42, verbose=True)
    model.fit(*data)
    out, _ = capsys.readouterr()
    expected_out = ('[%s] iter 1\t obj 0.569946\t num_imp 2\n'
                    'max iteration reached.\n' % estimator.__name__)
    assert out == expected_out

  def test_triplet_diffs_toy(self):
    expected_n_basis = 10
    model = SCML_Supervised(n_basis=expected_n_basis)
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    triplets = np.array([[0, 1, 2], [0, 1, 3], [1, 0, 2], [1, 0, 3],
                         [2, 3, 1], [2, 3, 0], [3, 2, 1], [3, 2, 0]])
    basis, n_basis = model._generate_bases_dist_diff(triplets, X)
    # All points are along the same line, so the only possible basis will be
    # the vector along that line normalized.
    expected_basis = np.ones((expected_n_basis, 2))/np.sqrt(2)
    assert n_basis == expected_n_basis
    np.testing.assert_allclose(basis, expected_basis)

  def test_lda_toy(self):
    expected_n_basis = 7
    model = SCML_Supervised(n_basis=expected_n_basis)
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1, 1])
    basis, n_basis = model._generate_bases_LDA(X, y)
    # All points are along the same line, so the only possible basis will be
    # the vector along that line normalized. In this case it is possible to
    # obtain it with positive or negative orientations.
    expected_basis = np.ones((expected_n_basis, 2))/np.sqrt(2)
    assert n_basis == expected_n_basis
    np.testing.assert_allclose(np.abs(basis), expected_basis)

  @pytest.mark.parametrize('n_samples', [100, 500])
  @pytest.mark.parametrize('n_features', [10, 50, 100])
  @pytest.mark.parametrize('n_classes', [5, 10, 15])
  def test_triplet_diffs(self, n_samples, n_features, n_classes):
    X, y = make_classification(n_samples=n_samples, n_classes=n_classes,
                               n_features=n_features, n_informative=n_features,
                               n_redundant=0, n_repeated=0)
    X = StandardScaler().fit_transform(X)

    model = SCML_Supervised()
    constraints = Constraints(y)
    triplets = constraints.generate_knntriplets(X, model.k_genuine,
                                                model.k_impostor)
    basis, n_basis = model._generate_bases_dist_diff(triplets, X)

    expected_n_basis = n_features * 80
    assert n_basis == expected_n_basis
    assert basis.shape == (expected_n_basis, n_features)

  @pytest.mark.parametrize('n_samples', [100, 500])
  @pytest.mark.parametrize('n_features', [10, 50, 100])
  @pytest.mark.parametrize('n_classes', [5, 10, 15])
  def test_lda(self, n_samples, n_features, n_classes):
    X, y = make_classification(n_samples=n_samples, n_classes=n_classes,
                               n_features=n_features, n_informative=n_features,
                               n_redundant=0, n_repeated=0)
    X = StandardScaler().fit_transform(X)

    model = SCML_Supervised()
    basis, n_basis = model._generate_bases_LDA(X, y)

    num_eig = min(n_classes - 1, n_features)
    expected_n_basis = min(20 * n_features, n_samples * 2 * num_eig - 1)
    assert n_basis == expected_n_basis
    assert basis.shape == (expected_n_basis, n_features)

  @pytest.mark.parametrize('name', ['max_iter', 'output_iter', 'batch_size',
                                    'n_basis'])
  def test_int_inputs(self, name):
    value = 1.0
    d = {name: value}
    scml = SCML(**d)
    triplets = np.array([[[0, 1], [2, 1], [0, 0]]])

    msg = ("%s should be an integer, instead it is of type"
           " %s" % (name, type(value)))
    with pytest.raises(ValueError) as raised_error:
      scml.fit(triplets)
    assert msg == raised_error.value.args[0]

  @pytest.mark.parametrize('name', ['max_iter', 'output_iter', 'batch_size',
                                    'k_genuine', 'k_impostor', 'n_basis'])
  def test_int_inputs_supervised(self, name):
    value = 1.0
    d = {name: value}
    scml = SCML_Supervised(**d)
    X = np.array([[0, 0], [1, 1], [3, 3], [4, 4]])
    y = np.array([1, 1, 0, 0])
    msg = ("%s should be an integer, instead it is of type"
           " %s" % (name, type(value)))
    with pytest.raises(ValueError) as raised_error:
      scml.fit(X, y)
    assert msg == raised_error.value.args[0]

  def test_large_output_iter(self):
    scml = SCML(max_iter=1, output_iter=2)
    triplets = np.array([[[0, 1], [2, 1], [0, 0]]])
    msg = ("The value of output_iter must be equal or smaller than"
           " max_iter.")

    with pytest.raises(ValueError) as raised_error:
      scml.fit(triplets)
    assert msg == raised_error.value.args[0]


class TestLSML(MetricTestCase):
  def test_iris(self):
    lsml = LSML_Supervised(num_constraints=200)
    lsml.fit(self.iris_points, self.iris_labels)

    csep = class_separation(lsml.transform(self.iris_points), self.iris_labels)
    self.assertLess(csep, 0.8)  # it's pretty terrible


class TestITML(MetricTestCase):
  def test_iris(self):
    itml = ITML_Supervised(num_constraints=200)
    itml.fit(self.iris_points, self.iris_labels)

    csep = class_separation(itml.transform(self.iris_points), self.iris_labels)
    self.assertLess(csep, 0.2)


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
    lmnn = LMNN(n_neighbors=5, verbose=False)
    lmnn.fit(self.iris_points, self.iris_labels)

    csep = class_separation(lmnn.transform(self.iris_points), self.iris_labels)
    self.assertLess(csep, 0.26)

  def test_loss_grad_lbfgs(self):
    """Test gradient of loss function
    Assert that the gradient is almost equal to its finite differences
    approximation.
    """
    rng = np.random.RandomState(42)
    X, y = make_classification(random_state=rng)
    L = rng.randn(rng.randint(1, X.shape[1] + 1), X.shape[1])
    lmnn = LMNN()
    lmnn.n_iter_ = 0
    lmnn.n_neighbors_ = lmnn.n_neighbors

    X, y = lmnn._prepare_inputs(X, y, dtype=float, ensure_min_samples=2)
    num_pts, n_components = X.shape
    unique_labels, y_inverse = np.unique(y, return_inverse=True)
    classes = np.arange(len(unique_labels))
    lmnn.components_ = np.eye(n_components)

    target_neighbors = lmnn._select_target_neighbors(X, y_inverse, classes)

    # sum outer products
    tn_graph = _make_knn_graph(target_neighbors)
    pull_loss_grad_m = _sum_weighted_outer_products(X, tn_graph)

    kwargs = {
        'classes': classes,
        'target_neighbors': target_neighbors,
        'pull_loss_grad_m': pull_loss_grad_m,
    }

    def fun(L):
        return lmnn._loss_grad_lbfgs(L, X, y, **kwargs)[0]

    def grad(L):
        return lmnn._loss_grad_lbfgs(L, X, y, **kwargs)[1]

    # compute gradient with and without finite differences
    epsilon = np.sqrt(np.finfo(float).eps)
    grad_fin_diff = approx_fprime(L.ravel(), fun, epsilon)
    grad_lmnn = grad(L)

    # compute absolute error
    grad_error = np.sqrt((grad_lmnn - grad_fin_diff)**2)

    # compute relative error
    # rel_diff1 = grad_error / np.linalg.norm(grad_fin_diff)
    # rel_diff2 = grad_error / np.linalg.norm(grad_lmnn)

    rel_diff = grad_error / np.linalg.norm(grad_lmnn)
    np.testing.assert_almost_equal(rel_diff, 0., decimal=5)


def test_compute_push_loss():
    """Test if the push loss is computed correctly

    This test continues on the example from test_find_impostors. The push
    loss is easy to compute, as we have only 4 violations and all of them
    amount to 1 (squared distance to target neighbor + 1 - squared distance
    to impostor = 4 + 1 - 4).
    """

    class_distance = 4.
    X_a = np.array([[-1., 1], [-1., -1.], [1., 1.], [1., -1.]])
    X_b = X_a + np.array([class_distance, 0])
    X = np.concatenate((X_a, X_b))
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    lmnn = LMNN(n_neighbors=1)
    lmnn.n_neighbors_ = 1
    classes = np.unique(y)
    target_neighbors = lmnn._select_target_neighbors(X, y, classes)
    diffs = X - X[target_neighbors[:, 0]]
    dist_tn = np.einsum('ij,ij->i', diffs, diffs)
    dist_tn = dist_tn[:, None]
    dist_tn += 1
    margin_radii = dist_tn[:, -1]
    impostors_graph = lmnn._find_impostors(X, y, classes, margin_radii)
    loss, grad, _ = _push_loss_grad(X, target_neighbors, dist_tn,
                                    impostors_graph)

    # The loss should be 4. (1. for each of the 4 violation)
    assert loss == 4.


@pytest.mark.parametrize('X, y, loss', [(np.array([[0], [1], [2], [3]]),
                                         [1, 1, 0, 0], 6.0),
                                        (np.array([[0], [1], [2], [3]]),
                                         [1, 0, 0, 1], 256.)])
def test_toy_ex_lmnn(X, y, loss):
  """Test that the loss give the right result on a toy example"""
  L = np.array([[1]])
  lmnn = LMNN(n_neighbors=1, push_loss_weight=0.5)

  lmnn.n_neighbors_ = lmnn.n_neighbors

  X, y = lmnn._prepare_inputs(X, y, dtype=float, ensure_min_samples=2)
  num_pts, n_components = X.shape
  unique_labels, label_inds = np.unique(y, return_inverse=True)
  classes = np.arange(len(unique_labels))
  lmnn.components_ = np.eye(n_components)

  target_neighbors = lmnn._select_target_neighbors(X, label_inds, classes)

  # sum outer products
  tn_graph = _make_knn_graph(target_neighbors)
  const_grad = _sum_weighted_outer_products(X, tn_graph)

  #  assert that the loss equals the one computed by hand
  lmnn.n_iter_ = 0
  predicted_loss = lmnn._loss_grad_lbfgs(L, X, y, classes, target_neighbors,
                                         const_grad)[0]
  assert predicted_loss == loss


def test_convergence_simple_example(capsys):
  # LMNN should converge on this simple example, which it did not with
  # this issue: https://github.com/scikit-learn-contrib/metric-learn/issues/88
  X, y = make_classification(random_state=0)
  lmnn = LMNN(verbose=True)
  lmnn.fit(X, y)
  out, _ = capsys.readouterr()
  assert "LMNN did not converge" not in out


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
    sdml_supervised = SDML_Supervised(balance_param=0.5, sparsity_param=0.01)
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
    sdml = SDML(prior='identity', balance_param=100, verbose=True)

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
    use SDML and has installed skggm, and that the algorithm fails to
    converge"""
    # TODO: remove if we don't need skggm anymore
    # case on which we know that skggm's graphical lasso fails
    # because it will return non finite values
    pairs = np.array([[[-10., 0.], [10., 0.]], [[0., 50.], [0., -60]]])
    y_pairs = [1, -1]
    sdml = SDML(prior='identity', balance_param=100, verbose=True)

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
    sdml_supervised = SDML_Supervised(balance_param=0.5, prior='identity',
                                      sparsity_param=0.01, random_state=rng)
    msg = ("There was a problem in SDML when using skggm's graphical "
           "lasso solver.")
    with pytest.raises(RuntimeError) as raised_error:
      sdml_supervised.fit(X, y)
    assert msg == str(raised_error.value)

  @pytest.mark.skipif(not HAS_SKGGM,
                      reason="It's only in the case where skggm is installed"
                             "that no warning should be thrown.")
  def test_raises_no_warning_installed_skggm(self):
    # otherwise we should be able to instantiate and fit SDML and it
    # should raise no error and no ConvergenceWarning
    pairs = np.array([[[-10., 0.], [10., 0.]], [[0., -55.], [0., -60]]])
    y_pairs = [1, -1]
    X, y = make_classification(random_state=42)
    with pytest.warns(None) as records:
      sdml = SDML(prior='covariance')
      sdml.fit(pairs, y_pairs)
    for record in records:
      assert record.category is not ConvergenceWarning
    with pytest.warns(None) as records:
      sdml_supervised = SDML_Supervised(prior='identity', balance_param=1e-5)
      sdml_supervised.fit(X, y)
    for record in records:
      assert record.category is not ConvergenceWarning

  def test_iris(self):
    # Note: this is a flaky test, which fails for certain seeds.
    # TODO: un-flake it!
    rs = np.random.RandomState(5555)

    sdml = SDML_Supervised(num_constraints=1500, prior='identity',
                           balance_param=5e-5, random_state=rs)
    sdml.fit(self.iris_points, self.iris_labels)
    csep = class_separation(sdml.transform(self.iris_points),
                            self.iris_labels)
    self.assertLess(csep, 0.22)

  def test_sdml_raises_warning_non_psd(self):
    """Tests that SDML raises a warning on a toy example where we know the
    pseudo-covariance matrix is not PSD"""
    pairs = np.array([[[-10., 0.], [10., 0.]], [[0., 50.], [0., -60]]])
    y = [1, -1]
    sdml = SDML(prior='covariance', sparsity_param=0.01, balance_param=0.5)
    msg = ("Warning, the input matrix of graphical lasso is not "
           "positive semi-definite (PSD). The algorithm may diverge, "
           "and lead to degenerate solutions. "
           "To prevent that, try to decrease the balance parameter "
           "`balance_param` and/or to set prior='identity'.")
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
    sdml = SDML(prior='covariance', sparsity_param=0.01, balance_param=0.5)
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
                           prior='covariance',
                           random_state=np.random.RandomState(42))
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
  sdml = SDML(verbose=True, prior='covariance')
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
  X, y = load_iris(return_X_y=True)
  sdml = SDML_Supervised(verbose=True, prior='identity', balance_param=1e-5)
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
  sdml = SDML(verbose=True, prior='covariance')
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
  sdml = SDML_Supervised(verbose=True, balance_param=1e-5, prior='identity')
  sdml.fit(X, y)
  out, _ = capsys.readouterr()
  assert "SDML will use scikit-learn's graphical lasso solver." in out


class TestNCA(MetricTestCase):
  def test_iris(self):
    n = self.iris_points.shape[0]

    # Without dimension reduction
    nca = NCA(max_iter=(100000 // n))
    nca.fit(self.iris_points, self.iris_labels)
    csep = class_separation(nca.transform(self.iris_points), self.iris_labels)
    self.assertLess(csep, 0.15)

    # With dimension reduction
    nca = NCA(max_iter=(100000 // n), n_components=2)
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
    nca = NCA(n_components=2,)
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

      A = make_spd_matrix(X.shape[1], X.shape[1])
      nca = NCA(init=A, max_iter=30, n_components=X.shape[1])
      nca.fit(X, y)
      assert_array_equal(nca.components_, A)

  def test_one_class(self):
      # if there is only one class the gradient is null, so the final matrix
      #  must stay like the initialization
      X = self.iris_points[self.iris_labels == 0]
      y = self.iris_labels[self.iris_labels == 0]

      A = make_spd_matrix(X.shape[1], X.shape[1])
      nca = NCA(init=A, max_iter=30, n_components=X.shape[1])
      nca.fit(X, y)
      assert_array_equal(nca.components_, A)


class TestLFDA(MetricTestCase):
  def test_iris(self):
    lfda = LFDA(k=2, n_components=2)
    lfda.fit(self.iris_points, self.iris_labels)
    csep = class_separation(lfda.transform(self.iris_points), self.iris_labels)
    self.assertLess(csep, 0.15)

    # Sanity checks for learned matrices.
    self.assertEqual(lfda.get_mahalanobis_matrix().shape, (4, 4))
    self.assertEqual(lfda.components_.shape, (2, 4))


class TestRCA(MetricTestCase):
  def test_iris(self):
    rca = RCA_Supervised(n_components=2, num_chunks=30, chunk_size=2)
    rca.fit(self.iris_points, self.iris_labels)
    csep = class_separation(rca.transform(self.iris_points), self.iris_labels)
    self.assertLess(csep, 0.29)

  def test_rank_deficient_returns_warning(self):
    """Checks that if the covariance matrix is not invertible, we raise a
    warning message advising to use PCA"""
    X, y = load_iris(return_X_y=True)
    # we make the fourth column a linear combination of the two first,
    # so that the covariance matrix will not be invertible:
    X[:, 3] = X[:, 0] + 3 * X[:, 1]
    rca = RCA()
    msg = ('The inner covariance matrix is not invertible, '
           'so the transformation matrix may contain Nan values. '
           'You should remove any linearly dependent features and/or '
           'reduce the dimensionality of your input, '
           'for instance using `sklearn.decomposition.PCA` as a '
           'preprocessing step.')

    with pytest.warns(None) as raised_warnings:
      rca.fit(X, y)
    assert any(str(w.message) == msg for w in raised_warnings)

  def test_unknown_labels(self):
    n = 200
    num_chunks = 50
    X, y = make_classification(random_state=42, n_samples=2 * n,
                               n_features=6, n_informative=6, n_redundant=0)
    y2 = np.concatenate((y[:n], -np.ones(n)))

    rca = RCA_Supervised(num_chunks=num_chunks, random_state=42)
    rca.fit(X[:n], y[:n])

    rca2 = RCA_Supervised(num_chunks=num_chunks, random_state=42)
    rca2.fit(X, y2)

    assert not np.any(np.isnan(rca.components_))
    assert not np.any(np.isnan(rca2.components_))

    np.testing.assert_array_equal(rca.components_, rca2.components_)

  def test_bad_parameters(self):
    n = 200
    num_chunks = 3
    X, y = make_classification(random_state=42, n_samples=n,
                               n_features=6, n_informative=6, n_redundant=0)

    rca = RCA_Supervised(num_chunks=num_chunks, random_state=42)
    msg = ('Due to the parameters of RCA_Supervised, '
           'the inner covariance matrix is not invertible, '
           'so the transformation matrix will contain Nan values. '
           'Increase the number or size of the chunks to correct '
           'this problem.'
           )
    with pytest.warns(None) as raised_warning:
      rca.fit(X, y)
    assert any(str(w.message) == msg for w in raised_warning)


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

    # Generate full set of constraints for comparison with reference
    # implementation
    mask = self.iris_labels[None] == self.iris_labels[:, None]
    a, b = np.nonzero(np.triu(mask, k=1))
    c, d = np.nonzero(np.triu(~mask, k=1))

    # Full metric
    n_features = self.iris_points.shape[1]
    mmc = MMC(convergence_threshold=0.01, init=np.eye(n_features) / 10)
    mmc.fit(*wrap_pairs(self.iris_points, [a, b, c, d]))
    expected = [[+0.000514, +0.000868, -0.001195, -0.001703],
                [+0.000868, +0.001468, -0.002021, -0.002879],
                [-0.001195, -0.002021, +0.002782, +0.003964],
                [-0.001703, -0.002879, +0.003964, +0.005648]]
    assert_array_almost_equal(expected, mmc.get_mahalanobis_matrix(),
                              decimal=6)

    # Diagonal metric
    mmc = MMC(diagonal=True)
    mmc.fit(*wrap_pairs(self.iris_points, [a, b, c, d]))
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
    assert re.match(r"\[" + algo_class.__name__ + r"\]\ *\d+\ *\d\.\d{6}e[+|-]"
                    r"\d+\ *\d+\.\d{2}", line)
  assert re.match(r"\[" + algo_class.__name__ + r"\] Training took\ *"
                  r"\d+\.\d{2}s\.", lines[-2])
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
