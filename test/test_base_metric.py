import unittest
import metric_learn


class TestStringRepr(unittest.TestCase):

  def test_covariance(self):
    self.assertEqual(str(metric_learn.Covariance()), "Covariance()")

  def test_lmnn(self):
    self.assertRegexpMatches(
        str(metric_learn.LMNN()),
        r"(python_)?LMNN\(convergence_tol=0.001, k=3, learn_rate=1e-07, "
        r"max_iter=1000,\n      min_iter=50, regularization=0.5, "
        r"use_pca=True, verbose=False\)")

  def test_nca(self):
    self.assertEqual(str(metric_learn.NCA()),
                     ("NCA(learning_rate='deprecated', max_iter=100, "
                      "num_dims=None, tol=None)"))

  def test_lfda(self):
    self.assertEqual(str(metric_learn.LFDA()),
                     "LFDA(embedding_type='weighted', k=None, num_dims=None)")

  def test_itml(self):
    self.assertEqual(str(metric_learn.ITML()), """
ITML(A0=None, convergence_threshold=0.001, gamma=1.0, max_iter=1000,
   verbose=False)
""".strip('\n'))
    self.assertEqual(str(metric_learn.ITML_Supervised()), """
ITML_Supervised(A0=None, bounds=None, convergence_threshold=0.001, gamma=1.0,
        max_iter=1000, num_constraints=None, num_labeled=inf,
        verbose=False)
""".strip('\n'))

  def test_lsml(self):
    self.assertEqual(
        str(metric_learn.LSML()),
        "LSML(max_iter=1000, prior=None, tol=0.001, verbose=False)")
    self.assertEqual(str(metric_learn.LSML_Supervised()), """
LSML_Supervised(max_iter=1000, num_constraints=None, num_labeled=inf,
        prior=None, tol=0.001, verbose=False, weights=None)
""".strip('\n'))

  def test_sdml(self):
    self.assertEqual(str(metric_learn.SDML()),
                     "SDML(balance_param=0.5, sparsity_param=0.01, "
                     "use_cov=True, verbose=False)")
    self.assertEqual(str(metric_learn.SDML_Supervised()), """
SDML_Supervised(balance_param=0.5, num_constraints=None, num_labeled=inf,
        sparsity_param=0.01, use_cov=True, verbose=False)
""".strip('\n'))

  def test_rca(self):
    self.assertEqual(str(metric_learn.RCA()),
                     "RCA(num_dims=None, pca_comps=None)")
    self.assertEqual(str(metric_learn.RCA_Supervised()),
                     "RCA_Supervised(chunk_size=2, num_chunks=100, "
                     "num_dims=None, pca_comps=None)")

  def test_mlkr(self):
    self.assertEqual(str(metric_learn.MLKR()),
                     "MLKR(A0=None, alpha=0.0001, epsilon=0.01, "
                     "max_iter=1000, num_dims=None)")

  def test_mmc(self):
    self.assertEqual(str(metric_learn.MMC()), """
MMC(A0=None, convergence_threshold=0.001, diagonal=False, diagonal_c=1.0,
  max_iter=100, max_proj=10000, verbose=False)
""".strip('\n'))
    self.assertEqual(str(metric_learn.MMC_Supervised()), """
MMC_Supervised(A0=None, convergence_threshold=1e-06, diagonal=False,
        diagonal_c=1.0, max_iter=100, max_proj=10000, num_constraints=None,
        num_labeled=inf, verbose=False)
""".strip('\n'))

if __name__ == '__main__':
  unittest.main()
