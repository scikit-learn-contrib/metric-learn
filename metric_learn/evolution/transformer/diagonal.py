import numpy as np

from .matrix import MatrixTransformer


class DiagonalMatrixTransformer(MatrixTransformer):
    def __init__(self):
        self.params = {}

    def individual_size(self, input_dim):
        return input_dim

    def fit(self, X, y, flat_weights):
        self.input_dim = X.shape[1]
        if self.input_dim != len(flat_weights):
            raise Exception(
                '`input_dim` and `flat_weights` sizes do not match: {} vs {}'
                .format(self.input_dim, len(flat_weights)))

        self.L = np.diag(flat_weights)
        return self
