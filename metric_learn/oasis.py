from .base_metric import BilinearMixin
import numpy as np

class OASIS(BilinearMixin):

    def __init__(self, preprocessor=None):
        super().__init__(preprocessor=preprocessor)

    def fit(self, X, y):
        """
        Fit OASIS model

        Parameters
        ----------
        X : (n x d) array of samples
        y : (n) data labels
        """
        X = self._prepare_inputs(X, y, ensure_min_samples=2)
        
        # Handmade dummy fit
        #self.components_ = np.identity(np.shape(X[0])[-1]) # Identity matrix
        #self.components_ = np.array([[2,4,6], [6,4,2], [1, 2, 3]])

        # Dummy fit
        self.components_ = np.random.rand(np.shape(X[0])[-1], np.shape(X[0])[-1])
        return self