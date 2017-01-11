import unittest
import numpy as np

from metric_learn import (
    LMNN, NCA, LFDA, Covariance, MLKR,
    LSML_Supervised, ITML_Supervised, SDML_Supervised, RCA_Supervised)


class TestBaseMetric(unittest.TestCase):

  def test_string_repr(self):
    '''
    Test string representation of some of the learning methods.
    '''
    self.assertEqual(str(Covariance()), "Covariance()")
    self.assertEqual(str(NCA()), "NCA(learning_rate=0.01, max_iter=100)")
    self.assertEqual(str(LFDA()), "LFDA(dim=None, k=7, metric='weighted')")

if __name__ == '__main__':
  unittest.main()
