import numpy as np
from .base_metric import BaseMetricLearner

class EvoMetric(BaseMetricLearner):
    def __init__(self, L, N):
        self.params = {}
        self.N = N
        if N == len(L):
            self.L = np.diag(L)
        elif N**2 == len(L):
            self.L = np.reshape(L, (N, N))
        else:
            raise Error('Invalid size of N')

    def transform(self, X):
        return X.dot(self.transformer().T)
        
    def transformer(self):
        return self.L
