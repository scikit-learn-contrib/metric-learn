class BaseFitness(object):
    def __init__(self):
        pass

    @staticmethod
    def available(method):
        return False

    def __call__(self, X_train, X_test, y_train, y_test):
        raise NotImplementedError('__call__ has not been implemented')
