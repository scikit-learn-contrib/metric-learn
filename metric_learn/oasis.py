from .base_metric import BilinearMixin, _TripletsClassifierMixin
import numpy as np


class OASIS(BilinearMixin, _TripletsClassifierMixin):
    """
    Key params:

    max_iter: Max number of iterations. Can differ from n_samples

    c: Passive-agressive param. Controls trade-off bewteen remaining
         close to previous W_i-1 OR minimizing loss of current triplet

    seed: For random sampling

    shuffle: If True will shuffle the triplets given to fit.
    """

    def __init__(
            self,
            preprocessor=None,
            max_iter=10,
            c=1e-6,
            random_seed=33,
            shuffle=False):
        super().__init__(preprocessor=preprocessor)
        self.components_ = None  # W matrix
        self.d = 0  # n_features
        self.max_iter = max_iter  # Max iterations
        self.c = c  # Trade-off param
        self.random_state = random_seed  # RNG

    def fit(self, X, y):
        """
        Fit OASIS model

        Parameters
        ----------
        X : (n x d) array of samples
        y : (n) data labels
        """
        X = self._prepare_inputs(X, y, ensure_min_samples=2)

        # Dummy fit
        self.components_ = np.random.rand(
            np.shape(X[0])[-1], np.shape(X[0])[-1])
        return self
