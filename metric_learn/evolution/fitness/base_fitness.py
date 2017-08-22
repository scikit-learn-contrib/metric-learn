import numpy as np


class BaseFitness(object):
    def __init__(self, random_state=None):
        self.random_state = random_state
        np.random.seed(random_state)

    def inject_params(self, random_state):
        self.random_state = random_state
        np.random.seed(random_state)

    @staticmethod
    def available(method):
        return False

    def __call__(self, X_train, X_test, y_train, y_test):
        raise NotImplementedError('__call__ has not been implemented')
