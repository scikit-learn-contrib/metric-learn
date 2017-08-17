import numpy as np

import scipy

from .matrix import MatrixTransformer


class NeuralNetworkTransformer(MatrixTransformer):
    def __init__(self, layers=None, activation='relu', use_biases=False):
        self.layers = layers
        self.activation = activation
        self.use_biases = use_biases

    def _build_activation(self):
        activation = self.activation

        if activation is None:
            return lambda X: X  # identity
        elif activation == 'relu':
            return lambda X: np.maximum(X, 0)  # ReLU
        elif activation == 'tanh':
            return np.tanh
        elif activation == 'sigm':
            return scipy.special.expit
        else:
            raise ValueError('Invalid activation paramater value')

    def individual_size(self, input_dim):
        last_layer = input_dim

        size = 0
        for layer in self.layers or (input_dim,):
            size += last_layer * layer

            if self.use_biases:
                size += layer

            last_layer = layer

        return size

    def fit(self, X, y, flat_weights):
        input_dim = X.shape[1]

        flat_weights = np.array(flat_weights)
        flat_weights_len = len(flat_weights)
        if flat_weights_len != self.individual_size(input_dim):
            raise Exception('Invalid size of the flat_weights')

        weights = []

        last_layer = input_dim
        offset = 0
        for layer in self.layers or (input_dim,):
            W = flat_weights[offset:offset + last_layer * layer].reshape(
                (last_layer, layer))
            offset += last_layer * layer

            if self.use_biases:
                b = flat_weights[offset:offset + layer]
                offset += layer
            else:
                b = np.zeros((layer))

            assert(offset <= flat_weights_len)
            weights.append((W, b))
            last_layer = layer

        self._parsed_weights = weights
        self._activation = self._build_activation()
        return self

    def transform(self, X):
        for i, (W, b) in enumerate(self._parsed_weights):
            X = np.add(np.matmul(X, W), b)

            if i + 1 < len(self._parsed_weights):
                X = self._activation(X)

        return X

    def weights(self):
        return self._parsed_weights

    def transformer(self):
        raise Exception(
            '`NeuralNetworkTransformer` does not use '
            'Mahalanobis matrix transformer.')

    def metric(self):
        raise Exception(
            '`NeuralNetworkTransformer` does not use '
            'Mahalanobis matrix metric.')
