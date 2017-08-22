import numpy as np

from .base_fitness import BaseFitness


class RandomFitness(BaseFitness):
    def __init__(self, **kwargs):
        super(RandomFitness, self).__init__(**kwargs)

    @staticmethod
    def available(method):
        return method in ['random']

    def __call__(self, X_train, X_test, y_train, y_test):
        return np.random.random()
