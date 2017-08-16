import numpy as np

from .base import BaseFitness


class RandomFitness(BaseFitness):
    @staticmethod
    def available(method):
        return method in ['random']

    def __call__(self, X_train, X_test, y_train, y_test):
        return np.random.random()
