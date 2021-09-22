from metric_learn.base_metric import BilinearMixin
import numpy as np
from numpy.testing import assert_array_almost_equal

class IdentityBilinearMixin(BilinearMixin):
    """A simple Identity bilinear mixin that returns an identity matrix
    M as learned. Can change M for a random matrix calling random_M.
    Class for testing purposes.
    """
    def __init__(self, preprocessor=None):
        super().__init__(preprocessor=preprocessor)

    def fit(self, X, y):
        X, y = self._prepare_inputs(X, y, ensure_min_samples=2)
        self.d = np.shape(X[0])[-1]
        self.components_ = np.identity(self.d)
        return self
    
    def random_M(self):
        self.components_ = np.random.rand(self.d, self.d)

def test_same_similarity_with_two_methods():
    d = 100
    u = np.random.rand(d)
    v = np.random.rand(d)
    mixin = IdentityBilinearMixin()
    mixin.fit([u, v], [0, 0])  # Dummy fit
    mixin.random_M()

    # The distances must match, whether calc with get_metric() or score_pairs()
    dist1 = mixin.score_pairs([[u, v], [v, u]])
    dist2 = [mixin.get_metric()(u, v), mixin.get_metric()(v, u)]

    assert_array_almost_equal(dist1, dist2)

def test_check_correctness_similarity():
    d = 100
    u = np.random.rand(d)
    v = np.random.rand(d)
    mixin = IdentityBilinearMixin()
    mixin.fit([u, v], [0, 0])  # Dummy fit
    dist1 = mixin.score_pairs([[u, v], [v, u]])
    u_v = np.dot(np.dot(u.T, np.identity(d)), v)
    v_u = np.dot(np.dot(v.T, np.identity(d)), u)
    desired = [u_v, v_u]
    assert_array_almost_equal(dist1, desired)

def test_check_handmade_example():
    u = np.array([0, 1, 2])
    v = np.array([3, 4, 5])
    mixin = IdentityBilinearMixin()
    mixin.fit([u, v], [0, 0])
    c = np.array([[2, 4, 6], [6, 4, 2], [1, 2, 3]])
    mixin.components_ = c # Force a components_
    dists = mixin.score_pairs([[u, v], [v, u]])
    assert_array_almost_equal(dists, [96, 120])

def test_check_handmade_symmetric_example():
    u = np.array([0, 1, 2])
    v = np.array([3, 4, 5])
    mixin = IdentityBilinearMixin()
    mixin.fit([u, v], [0, 0])
    dists = mixin.score_pairs([[u, v], [v, u]])
    assert_array_almost_equal(dists, [14, 14])