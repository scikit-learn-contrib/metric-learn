import numpy as np

from .matrix import MatrixTransformer


class TriangularMatrixTransformer(MatrixTransformer):
    def __init__(self):
        self.params = {}

    def individual_size(self, input_dim):
        return input_dim * (input_dim + 1) // 2

    def fit(self, X, y, flat_weights):
        input_dim = X.shape[1]
        if self.individual_size(input_dim) != len(flat_weights):
            raise Exception(
                '`input_dim` and `flat_weights` sizes do not match: {} vs {}'
                .format(self.individual_size(input_dim), len(flat_weights)))

        self.L = np.zeros((input_dim, input_dim))
        self.L[np.tril_indices(input_dim, 0)] = flat_weights
        return self
