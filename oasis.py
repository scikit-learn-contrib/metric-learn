from metric_learn.oasis import OASIS
import numpy as np
from numpy.random.mtrand import random_integers, seed
from sklearn.utils import check_random_state

class _BaseOASIS():
    """
    Key params:

    n_iter: Can differ from n_samples
    c: passive-agressive param. Controls trade-off bewteen remaining close to previous W_i-1 OR minimizing loss of current triplet
    seed: For random sampling
    """

    def __init__(self, n_iter=10, c=1e-6, seed=33) -> None:
        self.components_ = None
        self.d = 0
        self.n_iter = n_iter
        self.c = c
        self.random_state = seed
        

    def _fit(self, triplets):
        """
        Triplets : [[pi, pi+, pi-], [pi, pi+, pi-] , ... , ]

        Matrix W is already defined as I at __init___
        """
        self.d = np.shape(triplets)[2]       # Number of features
        self.components_ = np.identity(self.d)  # W_0 = I, Here and once, to reuse self.components_ for partial_fit

        n_triplets = np.shape(triplets)[0]
        rng = check_random_state(self.random_state)

        # Gen n_iter random indices
        random_indices = rng.randint(low=0, high=n_triplets, size=(self.n_iter))

        i = 0
        while i < self.n_iter:
            current_triplet = np.array(triplets[random_indices[i]])
            #print(f'i={i} {current_triplet}')
            loss = self._loss(current_triplet)
            vi = self._vi_matrix(current_triplet)
            fs = self._frobenius_squared(vi)
            tau_i = np.minimum(self.c, loss / fs)

            # Update components
            self.components_ = np.add(self.components_, tau_i * vi)
            print(self.components_)

            i = i + 1
    
    def _partial_fit(self, new_triplets):
        """
        self.components_ already defined, we reuse previous fit
        """
        self._fit(new_triplets)


    def _frobenius_squared(self, v):
        """
        Returns Frobenius norm of a point, squared
        """
        return np.trace(np.dot(v, v.T))

    def _score_pairs(self, pairs):
        """
        Computes bilinear similarity between a list of pairs.
        Parameters
        ----------
        pairs : array-like, shape=(n_pairs, 2, n_features) or (n_pairs, 2)
        3D Array of pairs to score, with each row corresponding to two points,
        for 2D array of indices of pairs if the metric learner uses a
        preprocessor.

        It uses self.components_ as current matrix W
        """
        pairs = np.array(pairs)
        return np.diagonal(np.dot(np.dot(pairs[:, 0, :], self.components_), pairs[:, 1, :].T))


    def _loss(self, triplet):
        """
        Loss function in a triplet
        """
        #print(np.shape(triplet[0]))
        return np.maximum(0, 1 - self._score_pairs([ [triplet[0], triplet[1]]])[0] + self._score_pairs([ [triplet[0], triplet[2]], ])[0] )
    

    def _vi_matrix(self, triplet):
        """
        Computes V_i, the gradient matrix in a triplet
        """
        diff = np.subtract(triplet[1], triplet[2]) # (, d)
        result = []

        for v in triplet[0]:
            result.append( v * diff)

        return np.array(result) # (d, d)


def test_OASIS():
    triplets = np.array([[[0, 1], [2, 1], [0, 0]],
                         [[2, 1], [0, 1], [2, 0]],
                         [[0, 0], [2, 0], [0, 1]],
                         [[2, 0], [0, 0], [2, 1]]])
    
    oasis = _BaseOASIS(n_iter=10, c=1e-5, seed=33)
    oasis._fit(triplets)

test_OASIS()