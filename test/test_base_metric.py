import unittest
import metric_learn


class TestBaseMetric(unittest.TestCase):

  def test_string_repr(self):
    # we don't test LMNN here because it could be python_LMNN

    self.assertEqual(str(metric_learn.Covariance()), "Covariance()")

    self.assertEqual(str(metric_learn.NCA()),
                     "NCA(learning_rate=0.01, max_iter=100, num_dims=None)")

    self.assertEqual(str(metric_learn.LFDA()),
                     "LFDA(dim=None, k=7, metric='weighted')")

    self.assertEqual(str(metric_learn.ITML()), """
ITML(convergence_threshold=0.001, gamma=1.0, max_iters=1000, verbose=False)
""".strip('\n'))
    self.assertEqual(str(metric_learn.ITML_Supervised()), """
ITML_Supervised(A0=None, bounds=None, convergence_threshold=0.001, gamma=1.0,
                max_iters=1000, num_constraints=None, num_labeled=inf,
                verbose=False)
""".strip('\n'))

    self.assertEqual(str(metric_learn.LSML()),
                     "LSML(max_iter=1000, tol=0.001, verbose=False)")
    self.assertEqual(str(metric_learn.LSML_Supervised()), """
LSML_Supervised(max_iter=1000, num_constraints=None, num_labeled=inf,
                prior=None, tol=0.001, verbose=False, weights=None)
""".strip('\n'))

    self.assertEqual(str(metric_learn.SDML()), """
SDML(balance_param=0.5, sparsity_param=0.01, use_cov=True, verbose=False)
""".strip('\n'))
    self.assertEqual(str(metric_learn.SDML_Supervised()), """
SDML_Supervised(balance_param=0.5, num_constraints=None, num_labeled=inf,
                sparsity_param=0.01, use_cov=True, verbose=False)
""".strip('\n'))

    self.assertEqual(str(metric_learn.RCA()), "RCA(dim=None, pca_comps=None)")
    self.assertEqual(str(metric_learn.RCA_Supervised()),
                     "RCA_Supervised(chunk_size=2, dim=None, num_chunks=100, pca_comps=None)")

    self.assertEqual(str(metric_learn.MLKR()), """
MLKR(A0=None, alpha=0.0001, epsilon=0.01, max_iter=1000, num_dims=None)
""".strip('\n'))

if __name__ == '__main__':
  unittest.main()
